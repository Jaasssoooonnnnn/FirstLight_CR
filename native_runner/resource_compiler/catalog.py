"""Stable global card ordering for the universal policy."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
from torch import Tensor

from native_runner.resource_compiler.card_features import (
    CARD_FEATURE_SCHEMA_VERSION,
    CARD_MECHANIC_FEATURE_NAMES,
    CARD_STATIC_FEATURE_NAMES,
    CardFeatureRow,
    build_card_feature_rows,
    pack_mechanic_features,
    pack_static_features,
)
from native_runner.resource_compiler.card_logic import StaticCardLogicCatalogV1, build_static_card_logic_catalog
from native_runner.resource_compiler.card_specs import CardSpecCatalog
from native_runner.contracts import AbilitySpecV1, CardSpecV1, TargetKind, content_hash
from native_runner.training.v4.config import ABILITY_FEATURE_NAMES, ModelConfigV4, UNIVERSAL_CARD_CATALOG_VERSION

if TYPE_CHECKING:
    from native_runner.resource_compiler.card_logic import StaticCardLogicCatalogV1
    from native_runner.resource_compiler.projectile_catalog import NativeProjectileCatalogV1


PAD_CARD_VOCAB_ID = 0
UNKNOWN_CARD_VOCAB_ID = 1
FIRST_REAL_CARD_VOCAB_ID = 2
PAD_ABILITY_VOCAB_ID = 0
PAD_ENTITY_ARCHETYPE_VOCAB_ID = 0
UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID = 1
FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID = 2
NORMAL_MODE_SOURCE_CARD_COUNT = 152
NORMAL_MODE_POLICY_CARD_COUNT = 122
ENTITY_ARCHETYPE_STATIC_DAMAGE_LEVEL = 11


@dataclass(frozen=True, slots=True)
class EntityArchetypeMetadataV1:
    """Minimal exact static semantics for one EntityArchetype row.

    ``static_damage_basis`` is an unnormalized tournament-level-11 damage
    value, projected with the exact carrier rarity curve in ``static_logic``.
    A false ``*_known`` flag means that the exact static record does not
    establish that value; no parent-card fallback is used.
    """

    archetype_key: str
    child_kind: str = "unknown"
    child_kind_known: bool = False
    is_airborne: bool = False
    is_airborne_known: bool = False
    static_damage_basis: float = 0.0
    static_damage_basis_known: bool = False
    projectile_radius_tiles: float = 0.0
    projectile_radius_known: bool = False
    lifetime_ms: float = 0.0
    lifetime_known: bool = False


def _unknown_archetype_metadata(key: str) -> EntityArchetypeMetadataV1:
    return EntityArchetypeMetadataV1(archetype_key=str(key))


_ENTITY_ROOT_NAMESPACES = frozenset({"CHARACTER", "BUILDING", "PROJECTILE", "AEO"})

# Crown-tower and Tower Troop projectiles are public normal-mode runtime
# objects, but no playable card owns their static roots.  Keep that system
# boundary explicit so every materialized projectile has an exact identity.
_NORMAL_MODE_SYSTEM_ENTITY_NODES = frozenset(
    {
        "EXT.ChefTower_pancake_projectile",
        "EXT.ChefTower_spatula_projectile",
        "PROJECTILE.CannoneerProjectile",
        "PROJECTILE.KingProjectile",
        "PROJECTILE.TowerKnifeThrowerProjectile",
        "PROJECTILE.TowerPrincessProjectile",
    }
)

# Native exposes two CharacterData identities for the left/right Ghostly
# Guardian spawn positions.  Their complete effective gameplay records are
# identical; the side is already represented by each live entity's position.
# Keep one semantic row while retaining both runtime IDs and source-node names
# as aliases.  The builder re-validates exact record equality before merging.
_ENTITY_ARCHETYPE_EQUIVALENCE_GROUPS = (
    (
        "form:Ghost_EV1_Summon_Guardian",
        "EXT.Ghost_EV1_Summon_Left",
        ("EXT.Ghost_EV1_Summon_Left", "EXT.Ghost_EV1_Summon_Right"),
    ),
)

_DIRECT_ATTACK_PROJECTILE_FIELDS = frozenset(
    {
        "CustomFirstProjectile",
        "OverrideProjectile",
        "Projectile",
        "Projectile2",
        "Projectile3",
        "ProjectileSpecial",
        "SpecialProjectile",
    }
)


def _effective_root_namespace(node_id: str, static_logic: "StaticCardLogicCatalogV1") -> str:
    node = static_logic.nodes.get(str(node_id))
    if not isinstance(node, Mapping):
        return ""
    root = str(node.get("inheritance_root_node_id") or node_id)
    return root.split(".", 1)[0]


def _effective_record(node_id: str, static_logic: "StaticCardLogicCatalogV1") -> Mapping[str, Any] | None:
    node = static_logic.nodes.get(str(node_id))
    if not isinstance(node, Mapping):
        return None
    record = node.get("effective_record")
    return record if isinstance(record, Mapping) else None


def _entity_archetype_key(node_id: str, global_id: int, root_namespace: str) -> str:
    name = str(node_id).split(".", 1)[1]
    if root_namespace in {"CHARACTER", "BUILDING"}:
        return f"form:{name}"
    if root_namespace == "PROJECTILE":
        return f"projectile:{int(global_id)}"
    if root_namespace == "AEO":
        return f"area:{int(global_id)}"
    raise ValueError(f"unsupported EntityArchetype root {root_namespace!r}")


def _level_11_static_value(
    record: Mapping[str, Any], field_name: str, static_logic: "StaticCardLogicCatalogV1"
) -> tuple[float, bool]:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0, False
    rarity = record.get("Rarity")
    curve = static_logic.rarity_curves.get(str(rarity))
    if not isinstance(curve, Mapping):
        return 0.0, False
    levels = tuple(curve.get("card_levels", ()))
    multipliers = tuple(curve.get("active_power_level_multipliers_percent", ()))
    if ENTITY_ARCHETYPE_STATIC_DAMAGE_LEVEL not in levels or len(levels) != len(multipliers):
        return 0.0, False
    multiplier = multipliers[levels.index(ENTITY_ARCHETYPE_STATIC_DAMAGE_LEVEL)]
    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        return 0.0, False
    return float(math.trunc(float(value) * multiplier / 100.0)), True


def _level_11_static_damage(record: Mapping[str, Any], static_logic: "StaticCardLogicCatalogV1") -> tuple[float, bool]:
    return _level_11_static_value(record, "Damage", static_logic)


def _direct_projectile_damage(node_id: str, static_logic: "StaticCardLogicCatalogV1") -> tuple[float, bool]:
    values: set[float] = set()
    for reference in static_logic.references:
        if str(reference.get("source_node_id")) != node_id or str(reference.get("status")) != "resolved":
            continue
        leaf = str(reference.get("field_path") or "").rsplit(".", 1)[-1]
        leaf = leaf.split("[", 1)[0]
        if leaf not in _DIRECT_ATTACK_PROJECTILE_FIELDS:
            continue
        for target in reference.get("target_node_ids", ()):
            target_id = str(target)
            if _effective_root_namespace(target_id, static_logic) != "PROJECTILE":
                continue
            target_record = _effective_record(target_id, static_logic)
            if target_record is None:
                continue
            damage, known = _level_11_static_damage(target_record, static_logic)
            if known and damage > 0:
                values.add(damage)
    return (next(iter(values)), True) if len(values) == 1 else (0.0, False)


def _entity_archetype_metadata(
    key: str,
    node_id: str,
    *,
    static_logic: "StaticCardLogicCatalogV1",
    projectile_catalog: "NativeProjectileCatalogV1 | None",
) -> EntityArchetypeMetadataV1:
    del projectile_catalog  # Exact effective records cover direct and EXT rows.
    namespace = _effective_root_namespace(node_id, static_logic)
    record = _effective_record(node_id, static_logic)
    if record is None or namespace not in _ENTITY_ROOT_NAMESPACES:
        return _unknown_archetype_metadata(key)

    if namespace in {"CHARACTER", "BUILDING"}:
        flying_height = record.get("FlyingHeight", 0)
        airborne_known = isinstance(flying_height, (int, float)) and not isinstance(flying_height, bool)
        damage, damage_known = _level_11_static_damage(record, static_logic)
        damage_known = damage_known and damage > 0
        if not damage_known:
            damage, damage_known = _direct_projectile_damage(node_id, static_logic)
        return EntityArchetypeMetadataV1(
            archetype_key=key,
            child_kind="character" if namespace == "CHARACTER" else "building",
            child_kind_known=True,
            is_airborne=bool(airborne_known and flying_height > 0),
            is_airborne_known=airborne_known,
            static_damage_basis=damage,
            static_damage_basis_known=damage_known,
        )

    if namespace == "PROJECTILE":
        damage, damage_known = _level_11_static_damage(record, static_logic)
        damage_known = damage_known and damage > 0
        radius_field = "ProjectileRadius"
        if not isinstance(record.get(radius_field), (int, float)):
            # Projectile Radius is the authoritative damage/splash radius in
            # the native projectile rows that omit a separate collision
            # ProjectileRadius (Fire Spirit, Wall Breakers, Phoenix, ...).
            radius_field = "Radius"
        raw_radius = record.get(radius_field)
        radius_known = isinstance(raw_radius, (int, float)) and not isinstance(raw_radius, bool)
        return EntityArchetypeMetadataV1(
            archetype_key=key,
            child_kind="projectile",
            child_kind_known=True,
            static_damage_basis=damage,
            static_damage_basis_known=damage_known,
            projectile_radius_tiles=(float(raw_radius) / 1_000.0 if radius_known else 0.0),
            projectile_radius_known=radius_known,
        )

    if namespace == "AEO":
        damage, damage_known = _level_11_static_damage(record, static_logic)
        damage_known = damage_known and damage > 0
        raw_radius = record.get("Radius")
        radius_known = isinstance(raw_radius, (int, float)) and not isinstance(raw_radius, bool)
        if not radius_known:
            defaults = static_logic.global_scaling_inputs.get("native_loader_defaults", {})
            radius_default = defaults.get("LogicAreaEffectObjectData.Radius") if isinstance(defaults, Mapping) else None
            if isinstance(radius_default, Mapping) and radius_default.get("value") == 0:
                raw_radius = 0
                radius_known = True
        raw_lifetime = record.get("LifeDuration")
        lifetime_known = isinstance(raw_lifetime, (int, float)) and not isinstance(raw_lifetime, bool)
        return EntityArchetypeMetadataV1(
            archetype_key=key,
            child_kind="area",
            child_kind_known=True,
            static_damage_basis=damage,
            static_damage_basis_known=damage_known,
            projectile_radius_tiles=(float(raw_radius) / 1_000.0 if radius_known else 0.0),
            projectile_radius_known=radius_known,
            lifetime_ms=float(raw_lifetime) if lifetime_known else 0.0,
            lifetime_known=lifetime_known,
        )
    return _unknown_archetype_metadata(key)


def _ability_duration_ms(spec: AbilitySpecV1) -> int | float | None:
    values: list[float] = []
    definition = spec.attributes.get("definition")
    records = [definition] if isinstance(definition, Mapping) else []
    records.extend(operation.parameters for operation in spec.effect_graph)
    duration_keys = {
        "abilityduration",
        "abilitystateduration",
        "bufftime",
        "duration",
        "durationms",
        "durationraw",
        "validduration",
    }
    for record in records:
        for key, value in record.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized in duration_keys and isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return max(values) if values else None


def _pack_ability_features(spec: AbilitySpecV1) -> tuple[float, ...]:
    """Pack only exact, varying semantics in the current competitive pool."""

    if spec.elixir_cost is None or spec.cast_time_ms is None or spec.charges is None or not spec.effect_graph:
        raise ValueError(f"Ability {spec.ability_id!r} has an incomplete static contract")
    if set(spec.target_schema.allowed) != {TargetKind.NONE}:
        raise ValueError(f"Ability {spec.ability_id!r} is outside the V4 target contract")
    duration_ms = _ability_duration_ms(spec)
    spawns_form = any(operation.effect.lower() == "spawn" for operation in spec.effect_graph)
    creates_area = any(operation.effect.lower() == "createarea" for operation in spec.effect_graph)
    applies_buff = any(
        operation.effect.lower() == "customop" and operation.custom_op == "ApplyBuff" for operation in spec.effect_graph
    )
    invokes_native_group = any(
        operation.effect.lower() == "customop" and operation.custom_op != "ApplyBuff" for operation in spec.effect_graph
    )
    return (
        float(spec.elixir_cost) / 10.0,
        float(spec.cooldown_ms or 0) / 30_000.0,
        float(spec.cast_time_ms) / 5_000.0,
        float(spec.charges) / 4.0,
        float(duration_ms or 0) / 10_000.0,
        float(spawns_form),
        float(creates_area),
        float(applies_buff),
        float(invokes_native_group),
        len(spec.effect_graph) / 4.0,
    )


def _nested_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(item for key, nested in value.items() for item in (str(key), *_nested_strings(nested)))
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(item for nested in value for item in _nested_strings(nested))
    return ()


_RUNTIME_UNIT_RESOURCE_FIELDS = frozenset(
    {
        "ActivationSpawnCharacter",
        "AttachedCharacter",
        "ChampionCharacterData",
        "ClonedVersion",
        "DeathSpawn",
        "DeathSpawnCharacter",
        "DeathSpawnCharacter2",
        "DeathSpawnCharacter3",
        "LeftSummonType",
        "LinkedChampionCharacter",
        "MorphCharacter",
        "MorphTarget",
        "NewCharacterData",
        "SpawnCharacter",
        "SpawnCharacter2",
        "SpawnCharacter3",
        "SpawnCharacterWithDeploy",
        "SpawnObject",
        "SpawnPathfindMorph",
        "SummonCharacter",
        "SummonCharacterSecond",
        "SummonCharactersList",
        "RightSummonType",
    }
)

_RUNTIME_PROJECTILE_RESOURCE_FIELDS = frozenset(
    {
        "BombProjectile",
        "CustomFirstProjectile",
        "DeathSpawnProjectile",
        "NewProjectileData",
        "OverrideProjectile",
        "Projectile",
        "Projectile2",
        "Projectile3",
        "ProjectileSpecial",
        "ProjectileType",
        "Projectiles",
        "ReRollProjectile",
        "SpecialProjectile",
        "SpawnProjectile",
    }
)

_RUNTIME_AEO_RESOURCE_FIELDS = frozenset(
    {
        "Aeo",
        "AppearAreaObject",
        "AeoList",
        "AreaEffectObject",
        "AreaEffectOnDash",
        "AreaEffectOnHit",
        "AreaEffectOnMorph",
        "BombAreaEffectObjects",
        "ContainerAeoList",
        "DamageAEO",
        "DeathAreaEffect",
        "DeathAreaEffectData",
        "LeftSummonAreaType",
        "RightSummonAreaType",
        "SpawnAreaEffectObject",
        "SpawnAreaObject",
        "SpawnsAEO",
        "TargetAoE",
    }
)

# Kept for the synthetic no-static-logic fallback below.
_RUNTIME_ENTITY_REFERENCE_FIELDS = _RUNTIME_UNIT_RESOURCE_FIELDS


def _runtime_entity_resource_field(leaf: str, root_namespace: str) -> bool:
    if leaf == "SpawnData":
        return True
    if root_namespace in {"CHARACTER", "BUILDING"}:
        return leaf in _RUNTIME_UNIT_RESOURCE_FIELDS
    if root_namespace == "PROJECTILE":
        return leaf in _RUNTIME_PROJECTILE_RESOURCE_FIELDS
    if root_namespace == "AEO":
        return leaf in _RUNTIME_AEO_RESOURCE_FIELDS
    return False


def _competitive_static_nodes(
    card_specs: Mapping[int, CardSpecV1], static_logic: "StaticCardLogicCatalogV1"
) -> frozenset[str]:
    """Return the exact native graph reachable by competitive card variants."""

    scoped = {int(card_id): spec for card_id, spec in card_specs.items()}
    cards = {int(card["card_id"]): card for card in static_logic.cards}
    missing = sorted(set(scoped).difference(cards))
    if missing:
        raise ValueError(f"static logic lacks scoped competitive cards: {missing}")

    # Start from the runtime-selectable roots, not the legacy card closure.
    # The latter includes Base/template nodes as if they executed alongside the
    # derived row, which is exactly how normal and evolved spawn definitions
    # used to be double counted.
    nodes = {str(cards[card_id]["root_node_id"]) for card_id in scoped}
    exact_roots: set[str] = set(nodes)
    for spec in scoped.values():
        exact_roots.update(f"ABILITY.{ability_id}" for ability_id in spec.ability_ids)
        if spec.evolution is not None and spec.evolution.evolution_form_id:
            exact_roots.add(f"SPELL_EVOLVED.{spec.evolution.evolution_form_id}")
        for attribute_name in ("resolved_hero_form", "resolved_direct_hero"):
            resolved_hero = spec.attributes.get(attribute_name)
            if isinstance(resolved_hero, Mapping):
                form_id = resolved_hero.get("form_id")
                if isinstance(form_id, str) and form_id:
                    exact_roots.add(f"SPELL_HERO.{form_id}")
    missing_roots = sorted(exact_roots.difference(static_logic.nodes))
    if missing_roots:
        raise ValueError(f"static logic lacks authoritative competitive variant roots: {missing_roots}")
    nodes.update(exact_roots)

    references: dict[str, list[Mapping[str, Any]]] = {}
    for reference in static_logic.references:
        references.setdefault(str(reference["source_node_id"]), []).append(reference)
    queue = list(nodes)
    for source in queue:
        for reference in references.get(source, ()):
            if str(reference.get("status")) != "resolved":
                continue
            # Effective records already contain inherited fields.  Only
            # executable control edges and typed runtime resources extend the
            # competitive graph; Base, filters, static selectors and VFX do
            # not execute recursively.
            if str(reference.get("edge_role")) not in {"control", "resource"}:
                continue
            if bool(reference.get("target_visual")):
                continue
            for raw_target in reference.get("target_node_ids", ()):
                target = str(raw_target)
                if target in static_logic.nodes and target not in nodes:
                    nodes.add(target)
                    queue.append(target)
    return frozenset(nodes)


def _runtime_reference_form_names(static_logic: "StaticCardLogicCatalogV1", reachable: frozenset[str]) -> set[str]:
    result: set[str] = set()
    for reference in static_logic.references:
        if str(reference.get("source_node_id")) not in reachable or str(reference.get("status")) != "resolved":
            continue
        path = str(reference.get("field_path") or "").rsplit(".", 1)[-1]
        path = path.split("[", 1)[0]
        if path == "SpawnData":
            expected = {str(value) for value in reference.get("expected_namespaces", ())}
            if not expected.intersection({"CHARACTER", "BUILDING"}):
                continue
        elif path not in _RUNTIME_ENTITY_REFERENCE_FIELDS:
            continue
        result.update(
            target.split(".", 1)[1]
            for raw_target in reference.get("target_node_ids", ())
            for target in (str(raw_target),)
            if target.startswith(("CHARACTER.", "BUILDING.", "EXT."))
        )

    # ActivationSpawnCharacter is an exact Ability field but is not emitted by
    # the current generic static-reference schema.
    for node_id in reachable:
        value = static_logic.nodes[node_id]["merged_record"].get("ActivationSpawnCharacter")
        if not isinstance(value, str) or not value:
            continue
        if any(f"{namespace}.{value}" in static_logic.nodes for namespace in ("CHARACTER", "BUILDING", "EXT")):
            result.add(value)
    return result


def _spawned_form_names(spec: CardSpecV1) -> set[str]:
    result = set(spec.summoned_forms)
    for operation in spec.mechanics:
        if operation.effect.lower() != "spawn":
            continue
        form = operation.parameters.get("form")
        if isinstance(form, str) and form:
            result.add(form)
        forms = operation.parameters.get("forms")
        if isinstance(forms, Sequence) and not isinstance(forms, (str, bytes, bytearray)):
            for entry in forms:
                if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                    result.add(str(entry["name"]))
    if spec.evolution is not None:
        result.update(
            value for value in (spec.evolution.base_form_id, spec.evolution.evolution_form_id) if value is not None
        )
        for operation in spec.evolution.effects:
            summoned_form = operation.parameters.get("summoned_form")
            if isinstance(summoned_form, str) and summoned_form:
                result.add(summoned_form)
            overrides = operation.parameters.get("explicit_unit_overrides")
            if not isinstance(overrides, Mapping):
                continue
            for field_name, override in overrides.items():
                if str(field_name) not in _RUNTIME_ENTITY_REFERENCE_FIELDS or not isinstance(override, Mapping):
                    continue
                evolved = override.get("evolved")
                if isinstance(evolved, str) and evolved:
                    result.add(evolved)
    for attribute_name in ("resolved_hero_form", "resolved_direct_hero"):
        resolved = spec.attributes.get(attribute_name)
        if not isinstance(resolved, Mapping):
            continue
        carrier = resolved.get("ability_carrier")
        if isinstance(carrier, str) and carrier:
            result.add(carrier)
    return result


def normal_mode_card_specs(
    card_specs: Mapping[int, CardSpecV1], *, require_frozen_baseline: bool = False
) -> dict[int, CardSpecV1]:
    """Return only cards admitted to ordinary competitive matches.

    The native catalog intentionally retains hidden/event definitions.  V4 is
    a frozen competitive policy baseline, so its production builder also
    checks the reviewed 152 -> 122 partition instead of silently changing the
    vocabulary when the native source drifts.
    """

    scoped = {
        int(card_id): spec
        for card_id, spec in card_specs.items()
        if spec.attributes.get("NotVisible") is not True and spec.attributes.get("NotInUse") is not True
    }
    if require_frozen_baseline:
        from native_runner.semantic_subset import NORMAL_MODE_POLICY_CARD_IDS, NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS

        source_ids = frozenset(int(card_id) for card_id in card_specs)
        if (
            len(card_specs) != NORMAL_MODE_SOURCE_CARD_COUNT
            or len(scoped) != NORMAL_MODE_POLICY_CARD_COUNT
            or frozenset(scoped) != NORMAL_MODE_POLICY_CARD_IDS
            or source_ids != NORMAL_MODE_POLICY_CARD_IDS | NORMAL_MODE_POLICY_EXCLUDED_CARD_IDS
        ):
            raise ValueError(
                "V4 normal-mode card partition drifted: "
                f"source={len(card_specs)}, eligible={len(scoped)}, "
                f"expected={NORMAL_MODE_SOURCE_CARD_COUNT}/"
                f"{NORMAL_MODE_POLICY_CARD_COUNT}"
            )
    return scoped


@dataclass(frozen=True, slots=True)
class AbilityCatalogV1:
    """Full competitive Ability ordering plus its exact static feature rows."""

    card_scope: tuple[int, ...]
    ability_ids: tuple[str, ...]
    static_features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        card_scope = tuple(int(value) for value in self.card_scope)
        if tuple(sorted(set(card_scope))) != card_scope:
            raise ValueError("AbilityCatalog card scope must be unique and sorted")
        normalized = tuple(str(value) for value in self.ability_ids)
        if any(not value for value in normalized):
            raise ValueError("ability IDs must not be empty")
        if tuple(sorted(set(normalized))) != normalized:
            raise ValueError("ability IDs must be unique and sorted")
        if len(self.static_features) != len(normalized):
            raise ValueError("AbilityCatalog feature rows do not match its ordering")
        if any(len(row) != len(ABILITY_FEATURE_NAMES) for row in self.static_features):
            raise ValueError("AbilityCatalog static rows do not match their named schema")
        if any(not math.isfinite(value) for row in self.static_features for value in row):
            raise ValueError("AbilityCatalog static rows must be finite")
        object.__setattr__(self, "card_scope", card_scope)
        object.__setattr__(self, "ability_ids", normalized)
        object.__setattr__(
            self, "static_features", tuple(tuple(float(value) for value in row) for row in self.static_features)
        )

    @classmethod
    def from_specs(
        cls, card_specs: Mapping[int, CardSpecV1], ability_specs: Sequence[AbilitySpecV1]
    ) -> "AbilityCatalogV1":
        scoped = {int(card_id): spec for card_id, spec in card_specs.items()}
        ability_ids = tuple(sorted({str(ability_id) for spec in scoped.values() for ability_id in spec.ability_ids}))
        by_id = {spec.ability_id: spec for spec in ability_specs}
        missing = sorted(set(ability_ids).difference(by_id))
        if missing:
            raise ValueError(f"competitive abilities lack exact specs: {missing}")
        return cls(
            card_scope=tuple(sorted(scoped)),
            ability_ids=ability_ids,
            static_features=tuple(_pack_ability_features(by_id[ability_id]) for ability_id in ability_ids),
        )

    @classmethod
    def from_card_spec_catalog(cls, card_catalog: CardSpecCatalog) -> "AbilityCatalogV1":
        return cls.from_specs(
            normal_mode_card_specs(card_catalog.by_id, require_frozen_baseline=True), card_catalog.abilities
        )

    @classmethod
    def empty(cls, card_scope: Sequence[int]) -> "AbilityCatalogV1":
        """Build an explicit no-Ability catalog for synthetic card fixtures."""

        return cls(tuple(sorted(int(value) for value in card_scope)), (), ())

    @property
    def vocab_size(self) -> int:
        return 1 + len(self.ability_ids)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "feature_names": ABILITY_FEATURE_NAMES,
                "card_scope": self.card_scope,
                "ability_ids": self.ability_ids,
                "static_features": self.static_features,
            }
        )

    def vocab_id(self, ability_id: str) -> int:
        try:
            return 1 + self.ability_ids.index(str(ability_id))
        except ValueError as error:
            raise ValueError(f"unknown ability ID {ability_id!r}") from error

    def ability_id(self, vocab_id: int) -> str | None:
        if int(vocab_id) == PAD_ABILITY_VOCAB_ID:
            return None
        index = int(vocab_id) - 1
        if not 0 <= index < len(self.ability_ids):
            raise ValueError("ability vocab ID is outside this catalog")
        return self.ability_ids[index]

    def feature_tensor(
        self, config: ModelConfigV4, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> Tensor:
        if self.vocab_size > config.ability_vocab_size:
            raise ValueError("AbilityCatalog exceeds ModelConfigV4 capacity")
        rows = (
            ((0.0,) * config.ability_feature_dim,)
            + self.static_features
            + ((0.0,) * config.ability_feature_dim,) * (config.ability_vocab_size - self.vocab_size)
        )
        return torch.tensor(rows, dtype=dtype, device=device)


@dataclass(frozen=True, slots=True)
class EntityArchetypeCatalogV1:
    """Canonical semantics plus exact live LogicData runtime aliases."""

    card_scope: tuple[int, ...]
    archetype_keys: tuple[str, ...]
    character_global_ids: tuple[tuple[int, str], ...] = ()
    archetype_metadata: tuple[EntityArchetypeMetadataV1, ...] = ()
    archetype_node_ids: tuple[str, ...] = ()
    archetype_key_aliases: tuple[tuple[str, str], ...] = ()
    archetype_node_aliases: tuple[tuple[str, str], ...] = ()
    _vocab_by_key: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _vocab_by_node_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _character_vocab_by_global_id: Mapping[int, int] = field(init=False, repr=False, compare=False)
    _runtime_vocab_by_global_id: Mapping[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        card_scope = tuple(int(value) for value in self.card_scope)
        keys = tuple(str(value) for value in self.archetype_keys)
        character_global_ids = tuple((int(global_id), str(form_id)) for global_id, form_id in self.character_global_ids)
        metadata = tuple(self.archetype_metadata) or tuple(_unknown_archetype_metadata(key) for key in keys)
        node_ids = tuple(str(value) for value in self.archetype_node_ids) or tuple("" for _ in keys)
        key_aliases = tuple((str(alias), str(canonical)) for alias, canonical in self.archetype_key_aliases)
        node_aliases = tuple((str(alias), str(canonical)) for alias, canonical in self.archetype_node_aliases)
        if tuple(sorted(set(card_scope))) != card_scope:
            raise ValueError("archetype card scope must be unique and sorted")
        if tuple(sorted(set(keys))) != keys or any(not value for value in keys):
            raise ValueError("archetype keys must be non-empty, unique, and sorted")
        if tuple(sorted(character_global_ids)) != character_global_ids:
            raise ValueError("Character global IDs must be sorted")
        if len({value[0] for value in character_global_ids}) != len(character_global_ids) or len(
            {value[1] for value in character_global_ids}
        ) != len(character_global_ids):
            raise ValueError("Character global IDs and form identities must be unique")
        addressable_keys = set(keys) | {alias for alias, _canonical in key_aliases}
        if any(
            global_id <= 0 or global_id > 0xFFFFFFFF or not form_id or f"form:{form_id}" not in addressable_keys
            for global_id, form_id in character_global_ids
        ):
            raise ValueError("Character global-ID mapping is outside the catalog")
        if tuple(row.archetype_key for row in metadata) != keys:
            raise ValueError("archetype metadata must align with archetype keys")
        if len(node_ids) != len(keys):
            raise ValueError("archetype node identities must align with keys")
        materialized_nodes = tuple(value for value in node_ids if value)
        if len(set(materialized_nodes)) != len(materialized_nodes):
            raise ValueError("materialized archetype node identities must be unique")
        for label, aliases, occupied in (
            ("archetype key", key_aliases, set(keys)),
            ("archetype node", node_aliases, set(materialized_nodes)),
        ):
            if tuple(sorted(aliases)) != aliases:
                raise ValueError(f"{label} aliases must be sorted")
            alias_names = tuple(alias for alias, _canonical in aliases)
            if (
                any(not alias or not canonical for alias, canonical in aliases)
                or len(set(alias_names)) != len(alias_names)
                or set(alias_names).intersection(occupied)
                or any(canonical not in keys for _alias, canonical in aliases)
            ):
                raise ValueError(f"{label} aliases must be unique and reference canonical keys")
        if any(
            not math.isfinite(value)
            for row in metadata
            for value in (row.static_damage_basis, row.projectile_radius_tiles, row.lifetime_ms)
        ):
            raise ValueError("archetype metadata numeric values must be finite")
        object.__setattr__(self, "card_scope", card_scope)
        object.__setattr__(self, "archetype_keys", keys)
        object.__setattr__(self, "character_global_ids", character_global_ids)
        object.__setattr__(self, "archetype_metadata", metadata)
        object.__setattr__(self, "archetype_node_ids", node_ids)
        object.__setattr__(self, "archetype_key_aliases", key_aliases)
        object.__setattr__(self, "archetype_node_aliases", node_aliases)
        vocab_by_key = {key: FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID + index for index, key in enumerate(keys)}
        vocab_by_key.update((alias, vocab_by_key[canonical]) for alias, canonical in key_aliases)
        vocab_by_node_id = {
            node_id: FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID + index for index, node_id in enumerate(node_ids) if node_id
        }
        vocab_by_node_id.update((alias, vocab_by_key[canonical]) for alias, canonical in node_aliases)
        character_vocab_by_global_id = {
            global_id: vocab_by_key[f"form:{form_id}"] for global_id, form_id in character_global_ids
        }
        runtime_vocab_by_global_id = dict(character_vocab_by_global_id)
        for key, vocab_id in vocab_by_key.items():
            namespace, raw_identity = key.split(":", 1)
            if namespace not in {"projectile", "area"}:
                continue
            global_id = int(raw_identity)
            existing = runtime_vocab_by_global_id.get(global_id)
            if existing is not None and existing != vocab_id:
                raise ValueError("one runtime LogicData global ID maps to multiple EntityArchetypes")
            runtime_vocab_by_global_id[global_id] = vocab_id
        object.__setattr__(self, "_vocab_by_key", MappingProxyType(vocab_by_key))
        object.__setattr__(self, "_vocab_by_node_id", MappingProxyType(vocab_by_node_id))
        object.__setattr__(self, "_character_vocab_by_global_id", MappingProxyType(character_vocab_by_global_id))
        object.__setattr__(self, "_runtime_vocab_by_global_id", MappingProxyType(runtime_vocab_by_global_id))

    @classmethod
    def from_card_specs(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        *,
        static_logic: "StaticCardLogicCatalogV1 | None" = None,
        projectile_catalog: "NativeProjectileCatalogV1 | None" = None,
        character_global_ids: Mapping[str, int] | None = None,
    ) -> "EntityArchetypeCatalogV1":
        scoped = {int(card_id): spec for card_id, spec in card_specs.items()}
        if static_logic is None:
            forms = {form for spec in scoped.values() for form in _spawned_form_names(spec)}
            return cls(tuple(sorted(scoped)), tuple(sorted(f"form:{form}" for form in forms)))

        from native_runner.resource_compiler.projectile_catalog import load_logic_data_global_ids, logic_data_global_id

        reachable = _competitive_static_nodes(scoped, static_logic)
        missing_system_nodes = sorted(_NORMAL_MODE_SYSTEM_ENTITY_NODES.difference(static_logic.nodes))
        if missing_system_nodes:
            raise ValueError(f"static logic lacks normal-mode system EntityArchetypes: {missing_system_nodes}")
        selected_nodes: set[str] = set(_NORMAL_MODE_SYSTEM_ENTITY_NODES)
        for reference in static_logic.references:
            if str(reference.get("source_node_id")) not in reachable or str(reference.get("status")) != "resolved":
                continue
            leaf = str(reference.get("field_path") or "").rsplit(".", 1)[-1]
            leaf = leaf.split("[", 1)[0]
            edge_role = str(reference.get("edge_role"))
            if edge_role != "resource" and not (edge_role == "control" and leaf == "SpawnData"):
                continue
            for raw_target in reference.get("target_node_ids", ()):
                target = str(raw_target)
                root_namespace = _effective_root_namespace(target, static_logic)
                if root_namespace not in _ENTITY_ROOT_NAMESPACES:
                    continue
                if not _runtime_entity_resource_field(leaf, root_namespace):
                    continue
                selected_nodes.add(target)

        registered_by_type = {
            data_type: load_logic_data_global_ids(static_logic, data_type=data_type)
            for data_type in ("Character", "Projectile", "AreaEffect")
        }
        if character_global_ids is not None:
            registered_by_type["Character"] = {str(name): int(value) for name, value in character_global_ids.items()}
        type_for_root = {
            "CHARACTER": "Character",
            "BUILDING": "Character",
            "PROJECTILE": "Projectile",
            "AEO": "AreaEffect",
        }
        raw_rows: dict[str, tuple[str, str, int, str]] = {}
        for node_id in sorted(selected_nodes):
            root_namespace = _effective_root_namespace(node_id, static_logic)
            data_type = type_for_root[root_namespace]
            name = node_id.split(".", 1)[1]
            global_id = int(registered_by_type[data_type].get(name, logic_data_global_id(data_type, name)))
            raw_rows[node_id] = (
                _entity_archetype_key(node_id, global_id, root_namespace),
                node_id,
                global_id,
                root_namespace,
            )

        canonical_by_node: dict[str, str] = {}
        representative_by_key: dict[str, str] = {}
        key_aliases: list[tuple[str, str]] = []
        node_aliases: list[tuple[str, str]] = []
        for canonical_key, representative, members in _ENTITY_ARCHETYPE_EQUIVALENCE_GROUPS:
            missing_members = sorted(set(members).difference(raw_rows))
            if missing_members or representative not in members:
                raise ValueError(
                    "semantic EntityArchetype equivalence group is outside the "
                    f"runtime selection: key={canonical_key!r}, missing={missing_members}"
                )
            records = tuple(_effective_record(member, static_logic) for member in members)
            namespaces = {_effective_root_namespace(member, static_logic) for member in members}
            if (
                any(record is None for record in records)
                or any(record != records[0] for record in records[1:])
                or len(namespaces) != 1
            ):
                raise ValueError(f"semantic EntityArchetype aliases lack identical exact records: {members}")
            representative_by_key[canonical_key] = representative
            for member in members:
                canonical_by_node[member] = canonical_key
                raw_key = raw_rows[member][0]
                if raw_key != canonical_key:
                    key_aliases.append((raw_key, canonical_key))
                if member != representative:
                    node_aliases.append((member, canonical_key))

        rows_by_key: dict[str, tuple[str, str, int, str]] = {}
        for node_id, raw_row in raw_rows.items():
            raw_key, _raw_node_id, global_id, root_namespace = raw_row
            key = canonical_by_node.get(node_id, raw_key)
            representative = representative_by_key.get(key, node_id)
            if node_id != representative:
                continue
            if key in rows_by_key:
                raise ValueError(f"runtime EntityArchetype key is duplicated: {key!r}")
            rows_by_key[key] = (key, node_id, global_id, root_namespace)

        rows = sorted(rows_by_key.values(), key=lambda value: value[0])
        keys = tuple(value[0] for value in rows)
        if len(set(keys)) != len(keys):
            raise ValueError("runtime EntityArchetype keys are not unique")
        character_rows = tuple(
            sorted(
                (global_id, node_id.split(".", 1)[1])
                for _key, node_id, global_id, root_namespace in raw_rows.values()
                if root_namespace in {"CHARACTER", "BUILDING"}
            )
        )
        metadata = tuple(
            _entity_archetype_metadata(key, node_id, static_logic=static_logic, projectile_catalog=projectile_catalog)
            for key, node_id, _global_id, _root_namespace in rows
        )
        return cls(
            card_scope=tuple(sorted(scoped)),
            archetype_keys=keys,
            character_global_ids=character_rows,
            archetype_metadata=metadata,
            archetype_node_ids=tuple(value[1] for value in rows),
            archetype_key_aliases=tuple(sorted(key_aliases)),
            archetype_node_aliases=tuple(sorted(node_aliases)),
        )

    @classmethod
    def from_card_spec_catalog(
        cls,
        card_catalog: CardSpecCatalog,
        *,
        static_logic: "StaticCardLogicCatalogV1",
        projectile_catalog: "NativeProjectileCatalogV1",
    ) -> "EntityArchetypeCatalogV1":
        return cls.from_card_specs(
            normal_mode_card_specs(card_catalog.by_id, require_frozen_baseline=True),
            static_logic=static_logic,
            projectile_catalog=projectile_catalog,
        )

    @property
    def vocab_size(self) -> int:
        return FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID + len(self.archetype_keys)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "version": "entity-archetype-catalog.v4",
                "card_scope": self.card_scope,
                "archetype_keys": self.archetype_keys,
                "character_global_ids": self.character_global_ids,
                "archetype_node_ids": self.archetype_node_ids,
                "archetype_key_aliases": self.archetype_key_aliases,
                "archetype_node_aliases": self.archetype_node_aliases,
                "metadata": tuple(
                    (
                        row.archetype_key,
                        row.child_kind,
                        row.child_kind_known,
                        row.is_airborne,
                        row.is_airborne_known,
                        row.static_damage_basis,
                        row.static_damage_basis_known,
                        row.projectile_radius_tiles,
                        row.projectile_radius_known,
                        row.lifetime_ms,
                        row.lifetime_known,
                    )
                    for row in self.archetype_metadata
                ),
            }
        )

    def vocab_id(self, key: str) -> int:
        return self._vocab_by_key.get(str(key), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def form_vocab_id(self, form_id: str) -> int:
        return self.vocab_id(f"form:{form_id}")

    def projectile_vocab_id(self, projectile_global_id: int) -> int:
        return self.vocab_id(f"projectile:{int(projectile_global_id)}")

    def area_vocab_id(self, area_global_id: int) -> int:
        return self.vocab_id(f"area:{int(area_global_id)}")

    def node_vocab_id(self, node_id: str) -> int:
        return self._vocab_by_node_id.get(str(node_id), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def character_vocab_id(self, character_global_id: int) -> int:
        return self._character_vocab_by_global_id.get(int(character_global_id), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def runtime_global_vocab_id(self, data_global_id: int) -> int:
        """Resolve an exact live LogicData identity across runtime namespaces."""

        return self._runtime_vocab_by_global_id.get(int(data_global_id), UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID)

    def metadata_for_vocab_id(self, vocab_id: int) -> EntityArchetypeMetadataV1:
        if int(vocab_id) in {PAD_ENTITY_ARCHETYPE_VOCAB_ID, UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID}:
            return _unknown_archetype_metadata("unknown")
        index = int(vocab_id) - FIRST_REAL_ENTITY_ARCHETYPE_VOCAB_ID
        if not 0 <= index < len(self.archetype_metadata):
            raise ValueError("entity archetype vocab ID is outside this catalog")
        return self.archetype_metadata[index]


@dataclass(frozen=True, slots=True)
class CardCatalogV1:
    """Canonical raw-card ordering plus deterministic semantic features.

    Real cards are sorted by their positive raw native card ID.  Therefore the
    same card set always produces the same vocabulary rows without depending
    on deck order or checkpoint metadata.  PAD and UNKNOWN occupy rows 0 and 1.
    """

    raw_card_ids: tuple[int, ...]
    static_features: tuple[tuple[float, ...], ...]
    mechanic_features: tuple[tuple[float, ...], ...]
    version: str = UNIVERSAL_CARD_CATALOG_VERSION

    def __post_init__(self) -> None:
        if self.version != UNIVERSAL_CARD_CATALOG_VERSION:
            raise ValueError("unsupported universal card catalog version")
        if any(
            isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0 for card_id in self.raw_card_ids
        ):
            raise ValueError("raw card IDs must be positive integers")
        if tuple(sorted(set(self.raw_card_ids))) != self.raw_card_ids:
            raise ValueError("raw card IDs must be unique and sorted")
        expected = len(self.raw_card_ids)
        if not len(self.static_features) == len(self.mechanic_features) == expected:
            raise ValueError("catalog feature rows do not match card ordering")
        for label, rows in (("static", self.static_features), ("mechanic", self.mechanic_features)):
            widths = {len(row) for row in rows}
            if len(widths) > 1:
                raise ValueError(f"catalog {label} rows have inconsistent widths")
            if any(not math.isfinite(value) for row in rows for value in row):
                raise ValueError(f"catalog {label} rows must be finite")
        if self.static_features and self.static_dim != len(CARD_STATIC_FEATURE_NAMES):
            raise ValueError("catalog static rows do not match their named schema")
        if self.mechanic_features and self.mechanic_dim != len(CARD_MECHANIC_FEATURE_NAMES):
            raise ValueError("catalog mechanic rows do not match their named schema")

    @classmethod
    def from_feature_rows(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        feature_rows: Mapping[int, CardFeatureRow],
        *,
        config: ModelConfigV4 | None = None,
    ) -> "CardCatalogV1":
        """Build the canonical catalog from the existing exact card features."""

        actual = config or ModelConfigV4()
        raw_card_ids = tuple(sorted(int(card_id) for card_id in card_specs))
        if set(raw_card_ids) != set(int(card_id) for card_id in feature_rows):
            missing_rows = sorted(set(raw_card_ids).difference(feature_rows))
            extra_rows = sorted(set(feature_rows).difference(raw_card_ids))
            raise ValueError(f"card specs and feature rows differ: missing={missing_rows}, extra={extra_rows}")
        static: list[tuple[float, ...]] = []
        mechanics: list[tuple[float, ...]] = []
        for card_id in raw_card_ids:
            spec = card_specs[card_id]
            row = feature_rows[card_id]
            static.append(tuple(pack_static_features(spec, row, width=actual.card_static_dim)))
            mechanic = tuple(pack_mechanic_features(spec, row, width=actual.card_mechanic_dim))
            mechanics.append(mechanic)
        return cls(raw_card_ids=raw_card_ids, static_features=tuple(static), mechanic_features=tuple(mechanics))

    @classmethod
    def from_card_spec_catalog(
        cls,
        card_catalog: CardSpecCatalog,
        *,
        static_logic: StaticCardLogicCatalogV1 | None = None,
        config: ModelConfigV4 | None = None,
    ) -> "CardCatalogV1":
        """Build every global row from exact static and Ability evidence."""

        exact_logic = static_logic or build_static_card_logic_catalog(card_catalog=card_catalog)
        scoped_specs = normal_mode_card_specs(card_catalog.by_id, require_frozen_baseline=True)
        feature_rows = build_card_feature_rows(
            scoped_specs, static_logic=exact_logic, ability_specs=card_catalog.abilities
        )
        return cls.from_feature_rows(scoped_specs, feature_rows, config=config)

    @classmethod
    def from_raw_ids(
        cls,
        raw_card_ids: Sequence[int],
        *,
        static_dim: int = len(CARD_STATIC_FEATURE_NAMES),
        mechanic_dim: int = len(CARD_MECHANIC_FEATURE_NAMES),
    ) -> "CardCatalogV1":
        """Create a zero-feature catalog for tests and explicit fixtures."""

        ordered = tuple(sorted({int(card_id) for card_id in raw_card_ids}))
        return cls(
            raw_card_ids=ordered,
            static_features=tuple((0.0,) * static_dim for _ in ordered),
            mechanic_features=tuple((0.0,) * mechanic_dim for _ in ordered),
        )

    @property
    def vocab_size(self) -> int:
        return FIRST_REAL_CARD_VOCAB_ID + len(self.raw_card_ids)

    @property
    def catalog_id(self) -> str:
        """Content identity for audit logs; checkpoints need not own ordering."""

        return content_hash(
            {
                "version": self.version,
                "feature_schema_version": CARD_FEATURE_SCHEMA_VERSION,
                "static_feature_names": CARD_STATIC_FEATURE_NAMES,
                "mechanic_feature_names": CARD_MECHANIC_FEATURE_NAMES,
                "raw_card_ids": self.raw_card_ids,
                "static_features": self.static_features,
                "mechanic_features": self.mechanic_features,
            }
        )

    @property
    def static_dim(self) -> int:
        return len(self.static_features[0]) if self.static_features else 0

    @property
    def mechanic_dim(self) -> int:
        return len(self.mechanic_features[0]) if self.mechanic_features else 0

    def vocab_id(self, raw_card_id: int | None) -> int:
        if raw_card_id is None:
            return PAD_CARD_VOCAB_ID
        try:
            return FIRST_REAL_CARD_VOCAB_ID + self.raw_card_ids.index(int(raw_card_id))
        except ValueError:
            return UNKNOWN_CARD_VOCAB_ID

    def require_vocab_id(self, raw_card_id: int, *, context: str = "card") -> int:
        """Map one concrete public card ID and reject off-scope content."""

        vocab_id = self.vocab_id(raw_card_id)
        if vocab_id == UNKNOWN_CARD_VOCAB_ID:
            raise ValueError(
                f"{context} {int(raw_card_id)} is outside this V4 catalog "
                "(the production baseline is the frozen 122-card normal-mode set)"
            )
        return vocab_id

    def raw_card_id(self, vocab_id: int) -> int | None:
        if vocab_id in {PAD_CARD_VOCAB_ID, UNKNOWN_CARD_VOCAB_ID}:
            return None
        index = int(vocab_id) - FIRST_REAL_CARD_VOCAB_ID
        if not 0 <= index < len(self.raw_card_ids):
            raise ValueError("card vocab ID is outside this catalog")
        return self.raw_card_ids[index]

    def feature_tensors(
        self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return PAD/UNKNOWN-prefixed catalog tensors in canonical order."""

        static_dim = self.static_dim
        mechanic_dim = self.mechanic_dim
        static_rows = ((0.0,) * static_dim,) * 2 + self.static_features
        mechanic_rows = ((0.0,) * mechanic_dim,) * 2 + self.mechanic_features
        return (
            torch.tensor(static_rows, dtype=dtype, device=device),
            torch.tensor(mechanic_rows, dtype=dtype, device=device),
        )
