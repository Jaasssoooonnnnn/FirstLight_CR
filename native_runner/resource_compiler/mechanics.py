"""Exact static mechanic and live-effect catalogs for policy V4.

The native logic graph remains the source of truth.  This module only projects
explicit graph structure, typed native actions, Buff definitions, and exact
runtime Buff global IDs; it does not infer mechanics from card names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import torch
from torch import Tensor

from native_runner.resource_compiler.card_logic import StaticCardLogicCatalogV1, classify_native_action
from native_runner.contracts import CardSpecV1, content_hash
from native_runner.resource_compiler.effect_catalog import NativeEffectCatalogV1, NativeEffectDefinitionV1
from native_runner.resource_compiler.catalog import (
    AbilityCatalogV1,
    CardCatalogV1,
    EntityArchetypeCatalogV1,
    PAD_ENTITY_ARCHETYPE_VOCAB_ID,
    UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID,
    _competitive_static_nodes,
    _level_11_static_value,
)

if TYPE_CHECKING:
    from native_runner.resource_compiler.projectile_catalog import NativeProjectileCatalogV1


PAD_EFFECT_VOCAB_ID = 0
UNKNOWN_EFFECT_VOCAB_ID = 1
FIRST_REAL_EFFECT_VOCAB_ID = 2
PAD_MECHANIC_PROFILE_ID = 0

# Tower-troop projectiles are public normal-mode objects but are outside the
# 122 playable-card graph.  Keep their static roots as one explicit system
# boundary instead of special-casing the Buff names they happen to apply.
_NORMAL_MODE_SYSTEM_EFFECT_SOURCES = frozenset({"EXT.ChefTower_pancake_projectile"})


EFFECT_TAG_NAMES = (
    "slow",
    "haste",
    "rage",
    "stun",
    "freeze",
    "movement_lock",
    "attack_lock",
    "spawn_lock",
    "invisibility",
    "minimum_hp",
    "periodic_damage",
    "periodic_heal",
    "damage_multiplier",
    "damage_reduction",
    "damage_immunity",
    "pushback_immunity",
    "attraction",
    "target_reset",
    "target_lock_control",
    "charge",
    "curse",
    "periodic_spawn",
    "clone",
    "projectile_override",
    "remove_on_attack",
    "on_start_action",
    "on_remove_action",
    "stackable",
    "opaque_native_buff",
)

EFFECT_NUMERIC_FEATURE_NAMES = (
    "move_speed_delta",
    "hit_speed_delta",
    "spawn_speed_delta",
    "damage_multiplier_delta",
    "damage_reduction_ratio",
    "damage_per_second_over_1000",
    "damage_per_second_known",
    "heal_per_second_over_1000",
    "attraction_over_500",
    "push_speed_over_500",
    "crown_tower_damage_delta",
    "override_charge_range_over_10000",
    "allowed_overheal_ratio",
    "building_damage_ratio",
    "character_crown_tower_damage_ratio",
)

EFFECT_SEMANTIC_FEATURE_NAMES = EFFECT_TAG_NAMES + EFFECT_NUMERIC_FEATURE_NAMES

ACTIVE_EFFECT_REMAINING_CAP_MS = 300_000
ACTIVE_EFFECT_RUNTIME_FEATURE_NAMES = (
    "remaining_over_10000ms_capped_at_300000ms",
    "remaining_known",
    "non_expiring",
    "stacks_over_10",
    "stacks_known",
    "magnitude",
    "magnitude_known",
)

MECHANIC_TAG_NAMES = (
    "grounding",
    "pull",
    "knock_up",
    "knockback",
    "taunt",
    "target_reset",
    "retarget",
    "reflect",
    "parry",
    "damage_multiplier",
    "charge",
    "dash",
    "jump",
    "warp",
    "recoil",
    "revive",
    "spawn",
    "death_spawn",
    "periodic_spawn",
    "clone",
    "splash",
    "multi_projectile",
    "pierce",
    "chain",
    "bounce_return",
    "persistent_area",
    "periodic_damage",
    "periodic_heal",
    "heal",
    "transform",
    "projectile_override",
    "active_ability",
    "ammo",
    "target_lock",
    "damage_ramp",
    "attack_sequence",
    "variable_damage_stage",
    "periodic_support",
    "projectile",
    "area_effect",
    "attachment",
    "scheduled_spawn",
    "self_destruct",
    "non_deflectable",
    "damage_immunity",
    "pushback_immunity",
    "target_seeking",
    "non_expiring",
    "capture",
    "visibility_transition",
    "movement_modifier",
    "phase_change",
    "ability_progress",
    "effect_immunity",
)

MECHANIC_NUMERIC_FEATURE_NAMES = (
    "count_or_resource_over_10",
    "duration_over_10000ms",
    "interval_over_5000ms",
    "extent_over_10000_units",
    "range_min_over_10000_units",
    "range_max_over_10000_units",
    "amount_over_1000",
    "speed_over_1000",
    "acceleration_over_1000",
    "angle_over_360deg",
    "ratio_over_100",
    "interaction_scalar_over_1000",
    "offset_x_over_10000_units",
    "offset_y_over_10000_units",
    "ordinal_over_10",
)

MECHANIC_GUARD_LIMIT = 4


_EFFECT_TAG_ALIASES: Mapping[str, str] = {"target_reset_allowed": "target_reset"}

_MECHANIC_TAG_ALIASES: Mapping[str, str] = {
    "damage_modifier": "damage_multiplier",
    "damage_over_time": "periodic_damage",
    "multi_projectile": "multi_projectile",
    "variable_damage_stage": "variable_damage_stage",
    "health_phase": "phase_change",
}

_EXACT_CLASS_TAGS: Mapping[str, tuple[str, ...]] = {
    "ActionAirToGround": ("grounding",),
    "ActionChainProjectileAttack": ("chain",),
    "ActionClone": ("clone", "spawn"),
    "ActionCounter": ("parry",),
    "ActionDamagingPushBack": ("knockback",),
    "ActionDoPushbackFromInstigator": ("knockback",),
    "ActionHeal": ("heal",),
    "ActionResetTarget": ("target_reset", "retarget"),
    "ActionSetAttackSequenceIndex": ("attack_sequence",),
    "ActionTaunt": ("taunt", "retarget"),
    "ActionWarpCharacter": ("warp",),
    "ActionMegaKnightUppercut": ("knockback", "knock_up"),
}

_EXACT_FIELD_TAGS: Mapping[str, tuple[str, ...]] = {
    "AreaDamageRadius": ("splash",),
    "AttackPushBack": ("recoil",),
    "ChainedHitCount": ("chain",),
    "ChainTargets": ("chain",),
    "ChargeRange": ("charge",),
    "ChargeSpeedMultiplier": ("charge",),
    "DashDamage": ("dash",),
    "DashRange": ("dash",),
    "DashMaxRange": ("dash",),
    "DeathSpawnCharacter": ("death_spawn", "spawn"),
    "DeathSpawnCharacter2": ("death_spawn", "spawn"),
    "DeathSpawnProjectile": ("death_spawn", "spawn"),
    "DamagePerSecond": ("periodic_damage",),
    "DeflectProjectilesEnabled": ("reflect",),
    "DragBackAsAttractor": ("pull",),
    "FirstStrongHitPushback": ("knockback",),
    "HealPerSecond": ("periodic_heal",),
    "JumpEnabled": ("jump",),
    "JumpHeight": ("jump",),
    "JumpSpeed": ("jump",),
    "MultipleProjectiles": ("multi_projectile",),
    "ProjectileCount": ("multi_projectile",),
    "OnPickNewTargetAction": ("retarget",),
    "AttractPercentage": ("pull",),
    "PingpongVisualTime": ("bounce_return",),
    "PushBackStrength": ("knockback",),
    "Pushback": ("knockback",),
    "ResetTarget": ("target_reset", "retarget"),
    "ResurrectBaseCount": ("revive", "spawn"),
    "SpawnClones": ("clone", "spawn"),
    "HideHpThresholds": ("phase_change",),
    "TempResurrect": ("revive",),
    "TetherDamage": ("periodic_damage",),
    "WarpAction": ("warp",),
    "WarpMode": ("warp",),
}

_SPAWN_REFERENCE_TRIGGERS: Mapping[str, str] = {
    "SummonCharacter": "on_deploy",
    "SummonCharacterSecond": "on_deploy",
    "SummonCharactersList": "on_deploy",
    "DeathSpawnCharacter": "on_death",
    "DeathSpawnCharacter2": "on_death",
    "SpawnCharacter": "spawn",
    "SpawnCharacterWithDeploy": "spawn",
    "ActivationSpawnCharacter": "ability_activate",
}

_TRANSFORM_REFERENCE_FIELDS = frozenset({"SpawnPathfindMorph"})

# These references expose an exact alternate runtime identity, but following
# them as an executable child branch invents behaviour.  ``ClonedVersion`` is
# the important normal-mode example: the clone form belongs in the archetype
# catalog, while its own spawn/death graph does not run at the point where the
# identity is declared.
_NON_EXECUTING_RESOURCE_FIELDS = frozenset({"ClonedVersion"})


def _active_native_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, Mapping)):
        return bool(value)
    return True


def _multiplier_delta(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number / 100.0 - 1.0 if number >= 100.0 else number / 100.0


def _number(record: Mapping[str, Any], names: Sequence[str], scale: float) -> float:
    values = [
        abs(float(record[name]))
        for name in names
        if name in record
        and isinstance(record[name], (int, float))
        and not isinstance(record[name], bool)
        and math.isfinite(float(record[name]))
    ]
    return max(values, default=0.0) / scale


def _signed_number(record: Mapping[str, Any], name: str, scale: float) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0
    return float(value) / scale


def _effect_tags(effect: NativeEffectDefinitionV1, record: Mapping[str, Any] | None = None) -> frozenset[str]:
    tags = {
        _EFFECT_TAG_ALIASES.get(tag, tag)
        for tag in effect.mechanic_tags
        if _EFFECT_TAG_ALIASES.get(tag, tag) in EFFECT_TAG_NAMES
    }
    resolved = effect.resolved_record if record is None else record
    raw_game_tags = resolved.get("GameTagsToSet")
    if isinstance(raw_game_tags, str):
        game_tags = {value.strip() for value in raw_game_tags.split(",") if value.strip()}
    elif isinstance(raw_game_tags, Sequence) and not isinstance(raw_game_tags, (bytes, bytearray)):
        game_tags = {str(value) for value in raw_game_tags}
    else:
        game_tags = set()
    if "UNKILLABLE" in game_tags:
        tags.add("minimum_hp")
    if "NO_DAMAGE" in game_tags:
        tags.add("damage_immunity")
    if "NO_PUSHED_BY_ENEMY" in game_tags:
        tags.add("pushback_immunity")
    if _active_native_value(record.get("SpawnInterval")) and any(
        _active_native_value(record.get(field_name))
        for field_name in (
            "ActivationSpawnCharacter",
            "SpawnCharacter",
            "SpawnCharacter2",
            "SpawnCharacter3",
            "SpawnData",
            "SpawnObject",
        )
    ):
        tags.update(("periodic_spawn", "spawn"))
    if "IGNORE_RANGE_EXTENSION_TO_KEEP_TARGET" in game_tags:
        tags.add("target_lock_control")
    if resolved.get("EnableStacking") is True:
        tags.add("stackable")
    return frozenset(tags)


def _scaled_effect_amount(
    record: Mapping[str, Any], field_name: str, static_logic: StaticCardLogicCatalogV1
) -> tuple[float, float]:
    value, known = _level_11_static_value(record, field_name, static_logic)
    return (value / 1_000.0, 1.0) if known else (0.0, 0.0)


def _pack_effect_features(
    effect: NativeEffectDefinitionV1, *, static_logic: StaticCardLogicCatalogV1, record: Mapping[str, Any] | None = None
) -> tuple[float, ...]:
    resolved = effect.resolved_record if record is None else record
    tags = _effect_tags(effect, resolved)
    modifiers = resolved
    damage, damage_known = _scaled_effect_amount(resolved, "DamagePerSecond", static_logic)
    heal, _heal_known = _scaled_effect_amount(resolved, "HealPerSecond", static_logic)
    numeric = (
        _multiplier_delta(modifiers.get("SpeedMultiplier")),
        _multiplier_delta(modifiers.get("HitSpeedMultiplier")),
        _multiplier_delta(modifiers.get("SpawnSpeedMultiplier")),
        _multiplier_delta(modifiers.get("DamageMultiplier")),
        _signed_number(modifiers, "DamageReduction", 100.0),
        damage,
        damage_known,
        heal,
        _number(modifiers, ("AttractPercentage",), 500.0),
        _number(modifiers, ("PushSpeedFactor",), 500.0),
        _signed_number(modifiers, "CrownTowerDamagePercent", 100.0),
        _signed_number(modifiers, "OverrideChargeRange", 10_000.0),
        _signed_number(modifiers, "AllowedOverHealPerc", 100.0),
        _signed_number(modifiers, "BuildingDamagePercent", 100.0),
        _signed_number(modifiers, "CharacterCrownTowerDamagePercent", 100.0),
    )
    return tuple(float(name in tags) for name in EFFECT_TAG_NAMES) + numeric


@dataclass(frozen=True, slots=True)
class EffectSemanticCatalogV1:
    """Source-release Buff identities and shared exact semantic rows."""

    card_scope: tuple[int, ...]
    buff_global_ids: tuple[int, ...]
    effect_names: tuple[str, ...]
    semantic_features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        scope = tuple(int(value) for value in self.card_scope)
        ids = tuple(int(value) for value in self.buff_global_ids)
        names = tuple(str(value) for value in self.effect_names)
        rows = tuple(tuple(float(value) for value in row) for row in self.semantic_features)
        if tuple(sorted(set(scope))) != scope:
            raise ValueError("Effect catalog card scope must be unique and sorted")
        if len(ids) != len(names) or len(ids) != len(rows):
            raise ValueError("Effect catalog identities and feature rows differ")
        if tuple(sorted(set(ids))) != ids or any(value <= 0 for value in ids):
            raise ValueError("Effect Buff global IDs must be positive, unique, and sorted")
        if len(set(names)) != len(names) or any(not value for value in names):
            raise ValueError("Effect names must be non-empty and unique")
        if any(len(row) != len(EFFECT_SEMANTIC_FEATURE_NAMES) for row in rows):
            raise ValueError("Effect semantic rows do not match their named schema")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("Effect semantic rows must be finite")
        object.__setattr__(self, "card_scope", scope)
        object.__setattr__(self, "buff_global_ids", ids)
        object.__setattr__(self, "effect_names", names)
        object.__setattr__(self, "semantic_features", rows)

    @classmethod
    def from_native_catalog(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        *,
        static_logic: StaticCardLogicCatalogV1,
        native_effect_catalog: NativeEffectCatalogV1,
    ) -> "EffectSemanticCatalogV1":
        scope = tuple(sorted(int(value) for value in card_specs))
        reachable = _competitive_static_nodes(card_specs, static_logic)
        application_sources = reachable.union(_NORMAL_MODE_SYSTEM_EFFECT_SOURCES)
        by_name = native_effect_catalog.by_name
        applied_nodes: set[str] = set()
        for effect in native_effect_catalog.effects:
            for application in effect.application_sources:
                if str(application["source_node_id"]) not in application_sources:
                    continue
                target = str(application.get("target_node_id") or effect.node_id)
                if target in static_logic.nodes:
                    applied_nodes.add(target)

        registry = native_effect_catalog.global_id_registry
        explicit = registry["buff_entries"]
        prefix = str(registry["fallback_prefix"])
        offset = int(registry["fallback_offset_basis"])
        prime = int(registry["fallback_prime"])

        def global_id(name: str) -> int:
            registered = explicit.get(name)
            if isinstance(registered, int) and not isinstance(registered, bool):
                return registered
            value = offset
            for byte in f"{prefix}{name}".encode("utf-8"):
                value = ((value ^ byte) * prime) & 0xFFFFFFFF
            return value

        rows: list[tuple[int, str, tuple[float, ...]]] = []
        for node_id in sorted(applied_nodes):
            node = static_logic.nodes[node_id]
            root = str(node.get("inheritance_root_node_id") or node_id)
            if not root.startswith("BUFF."):
                continue
            base_name = root.split(".", 1)[1]
            base = by_name.get(base_name)
            if base is None:
                raise ValueError(f"runtime Buff root {root!r} lacks native semantics")
            record = _node_record(static_logic, node_id)
            name = node_id.split(".", 1)[1]
            rows.append((global_id(name), name, _pack_effect_features(base, static_logic=static_logic, record=record)))
        represented_names = {row[1] for row in rows}
        for effect in native_effect_catalog.effects:
            if effect.effect_name in represented_names:
                continue
            rows.append(
                (effect.buff_global_id, effect.effect_name, _pack_effect_features(effect, static_logic=static_logic))
            )
        rows.sort(key=lambda item: item[0])
        return cls(scope, tuple(row[0] for row in rows), tuple(row[1] for row in rows), tuple(row[2] for row in rows))

    @classmethod
    def empty(cls, card_scope: Sequence[int]) -> "EffectSemanticCatalogV1":
        return cls(tuple(sorted(int(value) for value in card_scope)), (), (), ())

    @property
    def vocab_size(self) -> int:
        return FIRST_REAL_EFFECT_VOCAB_ID + len(self.buff_global_ids)

    @property
    def semantic_dim(self) -> int:
        return len(EFFECT_SEMANTIC_FEATURE_NAMES)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "version": "effect-semantic-catalog.v1",
                "feature_names": EFFECT_SEMANTIC_FEATURE_NAMES,
                "card_scope": self.card_scope,
                "buff_global_ids": self.buff_global_ids,
                "effect_names": self.effect_names,
                "semantic_features": self.semantic_features,
            }
        )

    def vocab_id(self, buff_global_id: int) -> int:
        try:
            return FIRST_REAL_EFFECT_VOCAB_ID + self.buff_global_ids.index(int(buff_global_id))
        except ValueError:
            return UNKNOWN_EFFECT_VOCAB_ID

    def runtime_vocab_id(self, effect: object) -> int:
        attributes = getattr(effect, "attributes", {})
        global_id = attributes.get("native_buff_global_id") if isinstance(attributes, Mapping) else None
        if global_id is None:
            raw_id = str(getattr(effect, "effect_id", ""))
            if raw_id.startswith("buff:"):
                try:
                    global_id = int(raw_id.split(":", 1)[1])
                except ValueError:
                    return UNKNOWN_EFFECT_VOCAB_ID
        if isinstance(global_id, bool) or not isinstance(global_id, int):
            return UNKNOWN_EFFECT_VOCAB_ID
        return self.vocab_id(global_id)

    def feature_tensor(self, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> Tensor:
        rows = ((0.0,) * self.semantic_dim,) + ((0.0,) * self.semantic_dim,) + self.semantic_features
        return torch.tensor(rows, dtype=dtype, device=device)


@dataclass(frozen=True, slots=True)
class MechanicOperationV1:
    trigger: str
    operation: str
    target: str
    mechanic_tags: tuple[str, ...] = ()
    effect_vocab_id: int = PAD_EFFECT_VOCAB_ID
    produced_archetype_id: int = PAD_ENTITY_ARCHETYPE_VOCAB_ID
    numeric_features: tuple[float, ...] = (0.0,) * len(MECHANIC_NUMERIC_FEATURE_NAMES)
    source_archetype: str = "unknown"
    control_path: str = "root"
    condition: str = "always"
    amount_basis: str = "none"
    guard_archetypes: tuple[str, ...] = ()
    guard_polarities: tuple[int, ...] = ()
    guard_numeric: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.trigger or not self.operation or not self.target:
            raise ValueError("mechanic operation categorical fields cannot be empty")
        if self.effect_vocab_id < 0 or self.produced_archetype_id < 0:
            raise ValueError("mechanic operation vocabulary IDs cannot be negative")
        if not all(
            isinstance(value, str) and value
            for value in (self.source_archetype, self.control_path, self.condition, self.amount_basis)
        ):
            raise ValueError("mechanic operation context fields cannot be empty")
        tags = tuple(sorted(set(str(value) for value in self.mechanic_tags)))
        if not set(tags).issubset(MECHANIC_TAG_NAMES):
            raise ValueError("mechanic operation contains an unknown semantic tag")
        numeric = tuple(float(value) for value in self.numeric_features)
        if len(numeric) != len(MECHANIC_NUMERIC_FEATURE_NAMES) or any(not math.isfinite(value) for value in numeric):
            raise ValueError("mechanic operation numeric row is invalid")
        guard_archetypes = tuple(str(value) for value in self.guard_archetypes)
        guard_polarities = tuple(int(value) for value in self.guard_polarities)
        guard_numeric = tuple((float(value[0]), float(value[1])) for value in self.guard_numeric)
        if not (len(guard_archetypes) == len(guard_polarities) == len(guard_numeric) <= MECHANIC_GUARD_LIMIT):
            raise ValueError("mechanic guard rows do not align")
        if any(not value for value in guard_archetypes) or any(value not in {-1, 1} for value in guard_polarities):
            raise ValueError("mechanic guard categoricals are invalid")
        if any(not math.isfinite(number) for pair in guard_numeric for number in pair):
            raise ValueError("mechanic guard numeric values must be finite")
        object.__setattr__(self, "mechanic_tags", tags)
        object.__setattr__(self, "numeric_features", numeric)
        object.__setattr__(self, "guard_archetypes", guard_archetypes)
        object.__setattr__(self, "guard_polarities", guard_polarities)
        object.__setattr__(self, "guard_numeric", guard_numeric)


def _reference_index(static_logic: StaticCardLogicCatalogV1) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for reference in static_logic.references:
        grouped.setdefault(str(reference["source_node_id"]), []).append(reference)
    return {key: tuple(value) for key, value in grouped.items()}


_PATH_SEGMENT = re.compile(r"(?P<name>[^.\[]+)(?:\[(?P<index>\d+)\])?")
_CONDITION_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_CONDITION_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _node_record(static_logic: StaticCardLogicCatalogV1, node_id: str) -> Mapping[str, Any]:
    node = static_logic.nodes[node_id]
    record = node.get("effective_record", node.get("merged_record", {}))
    if not isinstance(record, Mapping):
        raise ValueError(f"static node {node_id!r} lacks an effective record")
    return record


def _source_archetype(static_logic: StaticCardLogicCatalogV1, node_id: str, record: Mapping[str, Any]) -> str:
    node = static_logic.nodes[node_id]
    root = str(node.get("inheritance_root_node_id") or node_id)
    namespace = root.split(".", 1)[0].lower()
    if namespace == "action":
        class_type = record.get("ClassType")
        return str(class_type) if isinstance(class_type, str) and class_type else "action"
    return namespace


def _path_tokens(field_path: str) -> tuple[tuple[str, int | None], ...]:
    result: list[tuple[str, int | None]] = []
    for raw in str(field_path).split("."):
        match = _PATH_SEGMENT.fullmatch(raw)
        if match is None:
            continue
        index = match.group("index")
        result.append((match.group("name"), None if index is None else int(index)))
    return tuple(result)


def _path_signature(record: Mapping[str, Any], field_path: str) -> str:
    owner: Any = record
    segments: list[str] = []
    for name, index in _path_tokens(field_path):
        value = owner.get(name) if isinstance(owner, Mapping) else None
        if index is None:
            segments.append(name)
        else:
            length = len(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else 0
            segments.append(f"{name}[{index}/{length}]")
            value = value[index] if 0 <= index < length else None
        owner = value
    return ">".join(segments) or "field"


def _canonical_guard(expression: object) -> tuple[str, tuple[float, float]] | None:
    if not isinstance(expression, str) or not expression.strip():
        return None
    numbers = tuple(float(value) for value in _CONDITION_NUMBER.findall(expression))
    if len(numbers) > 2:
        # Native conditions with more than two literals are represented by
        # their exact expression shape; current competitive paths never need
        # more than two operands per guard occurrence.
        numbers = numbers[:2]
    canonical = _CONDITION_NUMBER.sub("#", expression)
    canonical = " ".join(canonical.split())
    return f"expr:{canonical}", (numbers[0] if numbers else 0.0, numbers[1] if len(numbers) > 1 else 0.0)


def _append_guard(
    guards: tuple[tuple[str, int, tuple[float, float]], ...], expression: object, polarity: int
) -> tuple[tuple[str, int, tuple[float, float]], ...]:
    # The fixed guard side-channel is for numeric thresholds.  Pure boolean
    # branch identity is already retained by the control-path categorical and
    # would otherwise let long target-filter chains consume the four slots
    # needed by real state thresholds (Little Prince: 0/3/6/7).
    if not isinstance(expression, str) or _CONDITION_NUMBER.search(expression) is None:
        return guards
    parsed = _canonical_guard(expression)
    if parsed is None:
        return guards
    row = (parsed[0], int(polarity), parsed[1])
    result = guards if row in guards else (*guards, row)
    if len(result) > MECHANIC_GUARD_LIMIT:
        raise ValueError("competitive mechanic path exceeds guard capacity")
    return result


def _local_guards(
    record: Mapping[str, Any], guards: tuple[tuple[str, int, tuple[float, float]], ...]
) -> tuple[tuple[str, int, tuple[float, float]], ...]:
    result = _append_guard(guards, record.get("ExecuteIfTrue"), 1)
    return _append_guard(result, record.get("ForceStopIfTrue"), -1)


def _guards_for_reference(
    record: Mapping[str, Any], field_path: str, guards: tuple[tuple[str, int, tuple[float, float]], ...]
) -> tuple[tuple[str, int, tuple[float, float]], ...]:
    owner: Any = record
    result = guards
    for name, index in _path_tokens(field_path):
        if not isinstance(owner, Mapping):
            break
        result = _local_guards(owner, result)
        if name == "OnTrueAction":
            result = _append_guard(result, owner.get("Condition"), 1)
        elif name == "OnFalseAction":
            result = _append_guard(result, owner.get("Condition"), -1)
        elif name == "OnActivateAction":
            result = _append_guard(result, owner.get("Condition"), 1)
        value = owner.get(name)
        if index is not None:
            if (
                name == "SubActions"
                and isinstance(owner.get("PerActionConditions"), Sequence)
                and not isinstance(owner.get("PerActionConditions"), (str, bytes, bytearray))
            ):
                conditions = tuple(owner["PerActionConditions"])
                if index < len(conditions):
                    result = _append_guard(result, conditions[index], 1)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                break
            if not 0 <= index < len(value):
                break
            value = value[index]
        owner = value
    return result


def _predicate_path_suffix(record: Mapping[str, Any], field_path: str | None = None) -> str:
    """Encode predicate identity/polarity that has no numeric side channel."""

    owner: Any = record
    segments: list[str] = []

    def append(name: str, expression: object, polarity: int) -> None:
        parsed = _canonical_guard(expression)
        if parsed is not None:
            segments.append(f"{name}:{'T' if polarity > 0 else 'F'}:{parsed[0]}")

    tokens = _path_tokens(field_path or "")
    for name, index in tokens:
        if not isinstance(owner, Mapping):
            break
        append("ExecuteIfTrue", owner.get("ExecuteIfTrue"), 1)
        append("ForceStopIfTrue", owner.get("ForceStopIfTrue"), -1)
        if name == "OnTrueAction":
            append("Condition", owner.get("Condition"), 1)
        elif name == "OnFalseAction":
            append("Condition", owner.get("Condition"), -1)
        elif name == "OnActivateAction":
            append("Condition", owner.get("Condition"), 1)
        value = owner.get(name)
        if index is not None:
            if (
                name == "SubActions"
                and isinstance(owner.get("PerActionConditions"), Sequence)
                and not isinstance(owner.get("PerActionConditions"), (str, bytes, bytearray))
            ):
                conditions = tuple(owner["PerActionConditions"])
                if index < len(conditions):
                    append("PerActionCondition", conditions[index], 1)
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes, bytearray))
                or not 0 <= index < len(value)
            ):
                break
            value = value[index]
        owner = value
    if not tokens:
        append("ExecuteIfTrue", record.get("ExecuteIfTrue"), 1)
        append("ForceStopIfTrue", record.get("ForceStopIfTrue"), -1)
    return "" if not segments else "|" + "|".join(segments)


def _inline_sequence_context(owner: Mapping[str, Any], field_name: str, index: int) -> dict[str, Any]:
    """Materialize exact list position and companion values for inline actions."""

    context: dict[str, Any] = {"OperationOrdinal": index}
    if field_name == "SubActions":
        delays = owner.get("SubActionsDelay")
        if (
            isinstance(delays, Sequence)
            and not isinstance(delays, (str, bytes, bytearray))
            and 0 <= index < len(delays)
        ):
            context["ActionDelay"] = delays[index]
    if field_name == "OnDetectedUnitActionList":
        thresholds = owner.get("MaxUnitPerActionList")
        if isinstance(thresholds, Sequence) and not isinstance(thresholds, (str, bytes, bytearray)):
            if index > 0 and index - 1 < len(thresholds):
                context["TargetCountLowerExclusive"] = thresholds[index - 1]
            if index < len(thresholds):
                context["TargetCountUpperInclusive"] = thresholds[index]
    return context


def _inline_action_records(
    value: object,
    *,
    path: str = "",
    context: Mapping[str, Any] | None = None,
    parent: Mapping[str, Any] | None = None,
    parent_field: str | None = None,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return typed inline actions and inline native resources losslessly.

    Most native resources are named graph nodes.  A small, legitimate subset
    is embedded directly inside an ``ActionSpawn`` record.  Treat those
    records exactly like their named counterparts while retaining the list
    occurrence and its companion thresholds; do not manufacture global IDs.
    """

    inherited = dict(context or {})
    result: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        materialized = dict(value)
        for key, item in inherited.items():
            materialized.setdefault(key, item)
        class_type = materialized.get("ClassType")
        if isinstance(class_type, str):
            result.append((path or "inline", materialized))
        elif (
            parent_field == "SpawnData"
            and isinstance(parent, Mapping)
            and parent.get("ClassType") == "ActionSpawn"
            and isinstance(materialized.get("Name"), str)
        ):
            spawn_type = str(parent.get("SpawnType") or "UnknownType")
            materialized["ClassType"] = f"Inline{spawn_type}Data"
            result.append((path or "inline", materialized))
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                for index, item in enumerate(nested):
                    item_path = f"{nested_path}[{index}]"
                    item_context = {**inherited, **_inline_sequence_context(value, str(key), index)}
                    result.extend(
                        _inline_action_records(
                            item, path=item_path, context=item_context, parent=value, parent_field=str(key)
                        )
                    )
            else:
                result.extend(
                    _inline_action_records(
                        nested, path=nested_path, context=inherited, parent=value, parent_field=str(key)
                    )
                )
    return tuple(result)


_NATIVE_CONTROLLER_OWNED_AMOUNT_FIELDS: Mapping[str, frozenset[str]] = {
    # ActionLaserBall selects one of its embedded damage resources.  The AEO
    # container's Damage field is a native controller placeholder, not an
    # additional hit to add beside the selected tier.
    "target_count_damage_controller": frozenset({"Damage"})
}


def _nested_controller_amount_fields(record: Mapping[str, Any]) -> frozenset[str]:
    result: set[str] = set()
    for field_name, nested in record.items():
        for _path, inline_record in _inline_action_records(nested, path=str(field_name)):
            classification = classify_native_action(inline_record)
            if classification is None:
                continue
            result.update(_NATIVE_CONTROLLER_OWNED_AMOUNT_FIELDS.get(str(classification["operation"]), ()))
    return frozenset(result)


def _contextualize(
    operation: MechanicOperationV1,
    *,
    source_archetype: str,
    control_path: str,
    guards: tuple[tuple[str, int, tuple[float, float]], ...],
) -> MechanicOperationV1:
    return replace(
        operation,
        source_archetype=source_archetype,
        control_path=control_path,
        guard_archetypes=tuple(value[0] for value in guards),
        guard_polarities=tuple(value[1] for value in guards),
        guard_numeric=tuple(value[2] for value in guards),
    )


def _root_closure(
    root: str,
    *,
    static_logic: StaticCardLogicCatalogV1,
    references: Mapping[str, Sequence[Mapping[str, Any]]],
    include_ability: bool,
) -> frozenset[str]:
    if root not in static_logic.nodes:
        return frozenset()
    seen = {root}
    queue = [root]
    for source in queue:
        for reference in references.get(source, ()):
            if str(reference.get("status")) != "resolved":
                continue
            if str(reference.get("edge_role")) not in {"control", "resource"}:
                continue
            path = str(reference.get("field_path") or "")
            lowered = path.lower()
            if lowered.startswith("ignorebuff"):
                continue
            if not include_ability and (lowered in {"ability", "heroability"} or lowered.startswith("evolvedspells")):
                continue
            for raw_target in reference.get("target_node_ids", ()):
                target = str(raw_target)
                if target in static_logic.nodes and target not in seen:
                    seen.add(target)
                    queue.append(target)
    return frozenset(seen)


def _mechanic_numeric(record: Mapping[str, Any]) -> tuple[float, ...]:
    """Project one record without combining unlike native quantities.

    This is the conservative fallback for typed rows.  Field-specific emitters
    below override it for spawn schedules, damage stages and control actions.
    The first exact field in each semantic family wins; values are never
    absolute-valued or max-pooled across unrelated meanings.
    """

    def first(names: Sequence[str]) -> float | None:
        for name in names:
            value = record.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                return float(value)
        return None

    def bounded_time(names: Sequence[str]) -> float | None:
        value = first(names)
        if value is None or value < 0 or value >= 99_999:
            return None
        return value

    count = first(("Count", "SpawnCount", "SpawnNumber", "MaxCharges", "MultipleProjectiles"))
    duration = bounded_time(("BuffTime", "Duration", "AbilityDuration", "LifeDuration", "SpawnTime"))
    interval = bounded_time(("HitFrequency", "SpawnInterval", "Cooldown"))
    extent = first(("Radius", "SpawnRadius", "CaptureRadius", "AreaDamageRadius"))
    range_min = first(("MinRange", "DashMinRange", "SpecialMinRange"))
    range_max = first(("MaxRange", "Range", "DashMaxRange", "DashRange", "ChargeRange", "SnipeMaxRange"))
    amount = first(("Damage", "DeathDamage", "TetherDamage", "AddedDamage", "Heal"))
    speed = first(("Speed", "JumpSpeed", "DragBackSpeed"))
    acceleration = first(("Acceleration", "Gravity"))
    angle = first(("Angle", "SpawnAngleShift", "SingleDeployOffsetAngle"))
    ratio = first(("DamageScalar", "DefenseScalar", "DamageReduction", "CrownTowerDamagePercent"))
    interaction = first(("Mass", "PushBackStrength", "Pushback", "AttractPercentage"))
    offset_x = first(("OffsetX", "HorizontalOffset", "WarpX"))
    offset_y = first(("OffsetY", "VerticalOffset", "WarpY"))
    values = (
        (count, 10.0),
        (duration, 10_000.0),
        (interval, 5_000.0),
        (extent, 10_000.0),
        (range_min, 10_000.0),
        (range_max, 10_000.0),
        (amount, 1_000.0),
        (speed, 1_000.0),
        (acceleration, 1_000.0),
        (angle, 360.0),
        (ratio, 100.0),
        (interaction, 1_000.0),
        (offset_x, 10_000.0),
        (offset_y, 10_000.0),
        (None, 10.0),
    )
    return tuple(0.0 if value is None else value / scale for value, scale in values)


def _attack_sequence_damage(record: Mapping[str, Any]) -> tuple[float, ...]:
    sequence = record.get("AttackSequence")
    sequence_list = record.get("AttackSequenceList")
    if (
        not isinstance(sequence, Sequence)
        or isinstance(sequence, (str, bytes, bytearray))
        or not sequence
        or not isinstance(sequence_list, Sequence)
        or isinstance(sequence_list, (str, bytes, bytearray))
    ):
        return ()
    result: list[float] = []
    for stage in sequence:
        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < len(sequence_list):
            return ()
        entry = sequence_list[stage]
        if not isinstance(entry, Mapping):
            return ()
        damage = entry.get("Damage")
        if isinstance(damage, bool) or not isinstance(damage, (int, float)):
            return ()
        result.append(float(damage))
    return tuple(result)


def _damage_stage_operations(
    record: Mapping[str, Any], *, static_logic: StaticCardLogicCatalogV1, inherited_rarity: str | None = None
) -> tuple[MechanicOperationV1, ...]:
    sequence_damage = _attack_sequence_damage(record)
    if sequence_damage:
        projected_sequence: list[float] = []
        rarity = record.get("Rarity") or inherited_rarity
        for damage in sequence_damage:
            projected, known = _level_11_static_value({"Damage": damage, "Rarity": rarity}, "Damage", static_logic)
            if not known:
                return ()
            projected_sequence.append(projected)
        operations = []
        for index, damage in enumerate(projected_sequence):
            numeric = [0.0] * len(MECHANIC_NUMERIC_FEATURE_NAMES)
            numeric[6] = damage / 1_000.0
            operations.append(
                MechanicOperationV1(
                    trigger="attack_sequence",
                    operation=f"attack_sequence_step_{index + 1}",
                    target="attack_target",
                    mechanic_tags=("attack_sequence",),
                    numeric_features=tuple(numeric),
                    amount_basis="level11",
                )
            )
        return tuple(operations)

    damage_fields = ("Damage", "VariableDamage2", "VariableDamage3")
    if not any(record.get(name) is not None for name in damage_fields[1:]):
        return ()
    projected: list[float] = []
    for field_name in damage_fields:
        damage, known = _level_11_static_value(record, field_name, static_logic)
        if not known:
            return ()
        projected.append(damage)

    first_time = record.get("VariableDamageTime1")
    second_time = record.get("VariableDamageTime2")
    timed_ramp = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value > 0
        for value in (first_time, second_time)
    )
    if timed_ramp:
        stage_rows = tuple(
            (index, projected[index], duration)
            for index, duration in enumerate((float(first_time), float(second_time), 0.0))
        )
        trigger = "locked_target_elapsed"
        operation_prefix = "damage_ramp_stage"
    else:
        sequence = record.get("AttackSequence")
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
            return ()
        stage_rows_list: list[tuple[int, float, float]] = []
        for step, stage in enumerate(sequence):
            if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < len(projected):
                return ()
            stage_rows_list.append((step, projected[stage], 0.0))
        stage_rows = tuple(stage_rows_list)
        trigger = "attack_sequence"
        operation_prefix = "attack_sequence_step"

    operations = []
    for index, damage, duration_ms in stage_rows:
        numeric = [0.0] * len(MECHANIC_NUMERIC_FEATURE_NAMES)
        numeric[1] = duration_ms / 10_000.0
        numeric[6] = damage / 1_000.0
        operations.append(
            MechanicOperationV1(
                trigger=trigger,
                operation=f"{operation_prefix}_{index + 1}",
                target="attack_target",
                numeric_features=tuple(numeric),
                amount_basis="level11",
            )
        )
    return tuple(operations)


def _timed_damage_prefix(record: Mapping[str, Any]) -> tuple[float, ...]:
    values = tuple(record.get(name) for name in ("Damage", "VariableDamage2", "VariableDamage3"))
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return ()
    if not all(
        isinstance(record.get(name), (int, float))
        and not isinstance(record.get(name), bool)
        and float(record[name]) > 0
        for name in ("VariableDamageTime1", "VariableDamageTime2")
    ):
        return ()
    return tuple(float(value) for value in values)


def _record_rarity(
    node_id: str, *, static_logic: StaticCardLogicCatalogV1, references: Mapping[str, Sequence[Mapping[str, Any]]]
) -> str | None:
    seen: set[str] = set()
    current = node_id
    while current not in seen:
        seen.add(current)
        record = static_logic.nodes[current]["merged_record"]
        rarity = record.get("Rarity")
        if isinstance(rarity, str) and rarity:
            return rarity
        bases = tuple(
            str(target)
            for reference in references.get(current, ())
            if str(reference.get("status")) == "resolved"
            and str(reference.get("reference_class")) == "inheritance"
            and str(reference.get("field_path") or "").rsplit(".", 1)[-1] == "Base"
            for target in reference.get("target_node_ids", ())
            if str(target) in static_logic.nodes
        )
        if len(bases) != 1:
            return None
        current = bases[0]
    return None


def _spawn_numeric(record: Mapping[str, Any], field_path: str) -> tuple[float, ...]:
    result = list(_mechanic_numeric(record))
    count_field = {
        "SummonCharacter": "SummonNumber",
        "SummonCharacterSecond": "SummonCharacterSecondCount",
        "SummonCharactersList": None,
        "DeathSpawnCharacter": "DeathSpawnCount",
        "DeathSpawnCharacter2": "DeathSpawnCount2",
        "SpawnCharacter": "SpawnCharacterCount",
        "SpawnCharacterWithDeploy": "SpawnCharacterCount",
        "ActivationSpawnCharacter": "SpawnCharacterCount",
    }.get(field_path, "")
    if count_field == "":
        return tuple(result)
    if count_field is None:
        result[0] = 0.1
    else:
        value = record.get(count_field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[0] = float(value) / 10.0
    return tuple(result)


def _node_tags(node_id: str, record: Mapping[str, Any], *, root_namespace: str | None = None) -> set[str]:
    namespace = root_namespace or node_id.split(".", 1)[0]
    tags = {
        tag
        for field_name, semantic_tags in _EXACT_FIELD_TAGS.items()
        if field_name in record and _active_native_value(record[field_name])
        for tag in semantic_tags
    }
    class_type = str(record.get("ClassType") or "")
    tags.update(_EXACT_CLASS_TAGS.get(class_type, ()))
    if class_type == "ActionKnockback":
        tags.add("knockback")
        height = record.get("Height")
        if isinstance(height, (int, float)) and not isinstance(height, bool) and float(height) > 0:
            tags.add("knock_up")
    if record.get("Kamikaze") is True:
        tags.add("self_destruct")
    if record.get("DeflectBehaviour") == "NoDeflect":
        tags.add("non_deflectable")
    if record.get("IgnorePushback") is True or record.get("IgnorePushBack") is True:
        tags.add("pushback_immunity")
    if namespace == "PROJECTILE":
        projectile_range = record.get("ProjectileRange")
        long_projectile = (
            isinstance(projectile_range, (int, float))
            and not isinstance(projectile_range, bool)
            and float(projectile_range) > 1_000.0
        )
        if _active_native_value(record.get("PingpongVisualTime")):
            tags.update(("bounce_return", "pierce"))
        if long_projectile and (
            _active_native_value(record.get("ProjectileRadiusY"))
            or _active_native_value(record.get("Pushback"))
            or _active_native_value(record.get("HomingTime"))
        ):
            tags.add("pierce")
        radius = record.get("Radius")
        if (
            isinstance(radius, (int, float))
            and not isinstance(radius, bool)
            and float(radius) > 0.0
            and any(record.get(field_name) is True for field_name in ("AoeToAir", "AoeToGround"))
        ):
            tags.add("splash")
        spawn_count = record.get("SpawnCount")
        if isinstance(spawn_count, (int, float)) and not isinstance(spawn_count, bool) and float(spawn_count) > 1.0:
            tags.add("multi_projectile")
    raw_game_tags = record.get("GameTagsToSet")
    if isinstance(raw_game_tags, str):
        game_tags = {value.strip() for value in raw_game_tags.split(",")}
    elif isinstance(raw_game_tags, Sequence) and not isinstance(raw_game_tags, (str, bytes, bytearray)):
        game_tags = {str(value) for value in raw_game_tags}
    else:
        game_tags = set()
    if "NO_DAMAGE" in game_tags:
        tags.add("damage_immunity")
    if "NO_PUSHED_BY_ENEMY" in game_tags:
        tags.add("pushback_immunity")
    if _active_native_value(record.get("VariableDamage2")) or _active_native_value(record.get("VariableDamage3")):
        if _active_native_value(record.get("AttackSequence")):
            tags.add("attack_sequence")
        elif _active_native_value(record.get("VariableDamageTime1")) or (
            _active_native_value(record.get("VariableDamageTime2"))
        ):
            tags.add("damage_ramp")
        else:
            tags.add("variable_damage_stage")
    life_duration = record.get("LifeDuration")
    if (
        namespace == "AEO"
        and isinstance(life_duration, (int, float))
        and not isinstance(life_duration, bool)
        and life_duration > 250
        and any(
            field in record for field in ("Buff", "Damage", "HitSpeed", "OnStartingAction", "SpawnAreaEffectObject")
        )
    ):
        tags.add("persistent_area")
    return tags


def _target_for_application(application_kind: str) -> str:
    if application_kind == "projectile_target_buff":
        return "hit_target"
    if application_kind in {"area_or_ability_buff", "crown_tower_buff"}:
        return "area_or_ability_targets"
    if application_kind in {"idle_lifecycle_buff", "pending_ability_buff"}:
        return "self"
    return "native_selected_targets"


def _reference_owner(record: Mapping[str, Any], field_path: str) -> tuple[Mapping[str, Any], str, int | None] | None:
    owner: Any = record
    tokens = _path_tokens(field_path)
    if not tokens:
        return None
    for name, index in tokens[:-1]:
        if not isinstance(owner, Mapping):
            return None
        value = owner.get(name)
        if index is not None:
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes, bytearray))
                or not 0 <= index < len(value)
            ):
                return None
            value = value[index]
        owner = value
    return (owner, *tokens[-1]) if isinstance(owner, Mapping) else None


def _reference_numeric(record: Mapping[str, Any], field_path: str, leaf: str) -> tuple[float, ...]:
    row = [0.0] * len(MECHANIC_NUMERIC_FEATURE_NAMES)
    tokens = _path_tokens(field_path)
    if not tokens:
        return tuple(row)
    owner: Any = record
    final_owner: Mapping[str, Any] | None = None
    companions: Mapping[str, tuple[tuple[str, int, float], ...]] = {
        "SubActions": (("SubActionsDelay", 1, 10_000.0),),
        "Actions": (("Delays", 1, 10_000.0), ("HealthPercentages", 10, 100.0)),
        "AeoList": (("StackAmountChecks", 0, 10.0),),
        "ContainerAeoList": (("OffsetXList", 12, 10_000.0), ("OffsetYList", 13, 10_000.0)),
        "BuffAfterHits": (("BuffAfterHitsCount", 0, 10.0), ("BuffAfterHitsTime", 1, 10_000.0)),
        "SummonCharactersList": (
            ("SummonCharactersOffsetsX", 12, 10_000.0),
            ("SummonCharactersOffsetsY", 13, 10_000.0),
        ),
        # Native barrage offsets are exported in half-tile cells, whereas the
        # model's coordinate slots are normalized native units (10,000/tile).
        "BombAreaEffectObjects": (("BombHorizontalOffsets", 12, 20.0), ("BombVerticalOffsets", 13, 20.0)),
    }
    for token_index, (name, item_index) in enumerate(tokens):
        if not isinstance(owner, Mapping):
            return tuple(row)
        final_owner = owner
        value = owner.get(name)
        if item_index is not None:
            row[14] = float(item_index) / 10.0
            for companion, slot, scale in companions.get(name, ()):
                values = owner.get(companion)
                if (
                    isinstance(values, Sequence)
                    and not isinstance(values, (str, bytes, bytearray))
                    and 0 <= item_index < len(values)
                ):
                    companion_value = values[item_index]
                    if isinstance(companion_value, (int, float)) and not isinstance(companion_value, bool):
                        row[slot] = float(companion_value) / scale
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes, bytearray))
                or not 0 <= item_index < len(value)
            ):
                return tuple(row)
            value = value[item_index]
        owner = value
        if token_index == len(tokens) - 1:
            break
    if final_owner is None:
        return tuple(row)
    count_field = {
        "SummonCharacter": "SummonNumber",
        "SummonCharacterSecond": "SummonCharacterSecondCount",
        "DeathSpawnCharacter": "DeathSpawnCount",
        "DeathSpawnCharacter2": "DeathSpawnCount2",
        "DeathSpawnCharacter3": "DeathSpawnCount3",
        "SpawnCharacter": "SpawnCharacterCount",
        "SpawnCharacter2": "SpawnCharacterCount2",
        "SpawnCharacter3": "SpawnCharacterCount3",
        "SpawnCharacterWithDeploy": "SpawnCharacterCount",
        "ActivationSpawnCharacter": "SpawnCharacterCount",
        "ProjectileType": "ProjectileCount",
    }.get(leaf)
    raw_count = final_owner.get(count_field) if count_field else None
    if isinstance(raw_count, (int, float)) and not isinstance(raw_count, bool):
        row[0] = float(raw_count) / 10.0
    elif leaf in {
        "AeoList",
        "AppearAreaObject",
        "AreaEffectObject",
        "AreaEffectOnDash",
        "AreaEffectOnHit",
        "AreaEffectOnMorph",
        "AttachedCharacter",
        "ContainerAeoList",
        "CustomFirstProjectile",
        "DamageAEO",
        "DeathAreaEffect",
        "DeathAreaEffectData",
        "SummonCharacter",
        "SummonCharacterSecond",
        "SummonCharactersList",
        "DeathSpawnCharacter",
        "DeathSpawnCharacter2",
        "DeathSpawnCharacter3",
        "SpawnCharacter",
        "SpawnCharacter2",
        "SpawnCharacter3",
        "SpawnCharacterWithDeploy",
        "ActivationSpawnCharacter",
        "SpawnData",
        "SpawnObject",
        "DeathSpawn",
        "DeathSpawnProjectile",
        "LeftSummonAreaType",
        "Projectile",
        "Projectile2",
        "Projectile3",
        "ProjectileSpecial",
        "ProjectileType",
        "Projectiles",
        "RightSummonAreaType",
        "SpecialProjectile",
        "SpawnAreaEffectObject",
        "SpawnAreaObject",
        "SpawnProjectile",
        "SpawnsAEO",
        "BombAreaEffectObjects",
    }:
        row[0] = 0.1
    if leaf == "SpellData":
        available = final_owner.get("AvailableManaTrigger")
        if isinstance(available, (int, float)) and not isinstance(available, bool):
            row[0] = float(available) / 10_000.0
        pending = final_owner.get("PrecastPendingTime")
        if isinstance(pending, (int, float)) and not isinstance(pending, bool):
            row[1] = float(pending) / 10_000.0
    return tuple(row)


def _semantic_trigger(field_path: str, leaf: str) -> str:
    fields = {name for name, _index in _path_tokens(field_path)}
    if leaf.startswith("Death") or fields.intersection({"OnKilledAction", "OnKillAction"}):
        return "on_death"
    if fields.intersection({"OnAttackAction", "OnStartingAttackAction"}):
        return "on_attack"
    if fields.intersection({"OnHitAction", "TetherHitAction"}):
        return "on_hit"
    if leaf.startswith("Summon"):
        return "on_deploy"
    if leaf == "ActivationSpawnCharacter":
        return "ability_activate"
    if leaf in {"MorphCharacter", "MorphTarget", "NewCharacterData", "SpawnPathfindMorph"}:
        return "state_transition"
    if leaf == "SpawnData":
        return "native_action_spawn"
    return "native_action"


def _resource_operation(
    reference: Mapping[str, Any],
    target_node: str,
    *,
    record: Mapping[str, Any],
    static_logic: StaticCardLogicCatalogV1,
    entity_catalog: EntityArchetypeCatalogV1,
) -> MechanicOperationV1 | None:
    field_path = str(reference.get("field_path") or "")
    leaf = field_path.rsplit(".", 1)[-1].split("[", 1)[0]
    root_namespace = str(static_logic.nodes[target_node].get("inheritance_root_node_id") or target_node).split(".", 1)[
        0
    ]
    numeric = _reference_numeric(record, field_path, leaf)
    if leaf == "SpellData" and root_namespace.startswith("SPELL_"):
        return MechanicOperationV1(
            trigger="elixir_variant",
            operation="select_card_variant",
            target="effective_deployment",
            numeric_features=numeric,
            condition=f"variant:{target_node}",
        )
    produced = entity_catalog.node_vocab_id(target_node)
    if produced in {PAD_ENTITY_ARCHETYPE_VOCAB_ID, UNKNOWN_ENTITY_ARCHETYPE_VOCAB_ID}:
        return None
    trigger = _semantic_trigger(field_path, leaf)
    if root_namespace in {"CHARACTER", "BUILDING"}:
        if leaf in {"NewCharacterData", "MorphCharacter", "MorphTarget", "SpawnPathfindMorph"}:
            operation = "transform_character"
            target = "self"
            tags = ("transform",)
        elif leaf == "ClonedVersion":
            operation = "alternate_clone_form"
            target = "spawned_clone"
            tags = ("clone",)
        elif leaf == "AttachedCharacter":
            operation = "attach_character"
            target = "self"
            tags = ("attachment", "spawn")
        else:
            operation = "spawn_character"
            target = "spawn_position"
            tags = ("spawn", *(("death_spawn",) if trigger == "on_death" else ()))
    elif root_namespace == "PROJECTILE":
        operation = {
            "NewProjectileData": "change_projectile_data",
            "OverrideProjectile": "override_projectile",
            "ReRollProjectile": "reroll_projectile",
        }.get(leaf, "spawn_projectile")
        target = "projectile_path"
        tags = (
            "projectile",
            *(("projectile_override",) if leaf == "OverrideProjectile" else ()),
            *(("death_spawn",) if trigger == "on_death" else ()),
        )
    elif root_namespace == "AEO":
        operation = "create_area_effect"
        target = "area_center"
        tags = ("area_effect", *(("death_spawn",) if trigger == "on_death" else ()))
    else:
        return None
    return MechanicOperationV1(
        trigger=trigger,
        operation=operation,
        target=target,
        mechanic_tags=tags,
        produced_archetype_id=produced,
        numeric_features=numeric,
        condition=f"resource:{leaf}",
    )


def _effect_vocab_for_node(target_node: str, effect_catalog: EffectSemanticCatalogV1) -> int:
    if not target_node.startswith(("BUFF.", "EXT.")):
        return UNKNOWN_EFFECT_VOCAB_ID
    effect_name = target_node.split(".", 1)[1]
    try:
        return FIRST_REAL_EFFECT_VOCAB_ID + effect_catalog.effect_names.index(effect_name)
    except ValueError:
        return UNKNOWN_EFFECT_VOCAB_ID


def _data_dependency_operation(
    reference: Mapping[str, Any], target_node: str, *, effect_catalog: EffectSemanticCatalogV1
) -> MechanicOperationV1 | None:
    field_path = str(reference.get("field_path") or "")
    leaf = field_path.rsplit(".", 1)[-1].split("[", 1)[0]
    if leaf == "IgnoreBuff":
        return MechanicOperationV1(
            trigger="static_contract",
            operation="ignore_effect",
            target="self",
            mechanic_tags=("effect_immunity",),
            effect_vocab_id=_effect_vocab_for_node(target_node, effect_catalog),
            condition=f"effect_immunity:{target_node}",
        )
    if leaf == "IgnoreTargetsWithBuff":
        return MechanicOperationV1(
            trigger="target_selection",
            operation="avoid_effect_target",
            target="native_selected_targets",
            mechanic_tags=("retarget", "target_lock"),
            effect_vocab_id=_effect_vocab_for_node(target_node, effect_catalog),
            condition=f"target_exclusion:{target_node}",
        )
    if leaf in {
        "Filter",
        "GameObjectFilter",
        "ObjectFilter",
        "ResurrectChargeFilter",
        "Resolver",
        "SnipeTargetFilter",
        "TargetFilter",
        "TargetResolver",
        "TetherDamageTargets",
        "TroopFilter",
    }:
        return MechanicOperationV1(
            trigger="target_selection",
            operation="target_selector",
            target="native_selected_targets",
            mechanic_tags=("target_lock",),
            condition=f"selector:{target_node}",
        )
    return None


_PHYSICS_FIELDS: Mapping[str, tuple[int, float, str]] = {
    # cardinality/resources
    "AttackAmount": (0, 10.0, "raw_fixed"),
    "AttackIndex": (14, 10.0, "raw_fixed"),
    "ChainedHitCount": (0, 10.0, "raw_fixed"),
    "ChainCount": (0, 10.0, "raw_fixed"),
    "DashCount": (0, 10.0, "raw_fixed"),
    "MaxCharges": (0, 10.0, "raw_fixed"),
    "MaxFriendlyTroops": (0, 10.0, "raw_fixed"),
    "MaxTargets": (0, 10.0, "raw_fixed"),
    "MultipleTargets": (0, 10.0, "raw_fixed"),
    "MultipleProjectiles": (0, 10.0, "raw_fixed"),
    "NumberOfUnitsToCapture": (0, 10.0, "raw_fixed"),
    "AmmoCount": (0, 10.0, "raw_fixed"),
    "GroupMaxSize": (0, 10.0, "raw_fixed"),
    "MaxChainLength": (0, 10.0, "raw_fixed"),
    "ProjectileCount": (0, 10.0, "raw_fixed"),
    "ProjectileWaves": (0, 10.0, "raw_fixed"),
    "ResurrectBaseCount": (0, 10.0, "raw_fixed"),
    "SpawnLimit": (0, 10.0, "raw_fixed"),
    "SpawnCount": (0, 10.0, "raw_fixed"),
    "SpawnNumber": (0, 10.0, "raw_fixed"),
    "TargetCountLowerExclusive": (0, 10.0, "raw_fixed"),
    "TargetCountUpperInclusive": (0, 10.0, "raw_fixed"),
    "TotalBalloons": (0, 10.0, "raw_fixed"),
    # timing
    "ActionDelay": (1, 10_000.0, "raw_fixed"),
    "ActionDuration": (1, 10_000.0, "raw_fixed"),
    "AbilityDuration": (1, 10_000.0, "raw_fixed"),
    "AbilityStateDuration": (1, 10_000.0, "raw_fixed"),
    "ActivationTime": (1, 10_000.0, "raw_fixed"),
    "BuffOnDamageTime": (1, 10_000.0, "raw_fixed"),
    "BuffTime": (1, 10_000.0, "raw_fixed"),
    "CaptureDragTime": (1, 10_000.0, "raw_fixed"),
    "CaptureCooldown": (1, 10_000.0, "raw_fixed"),
    "CastTime": (1, 10_000.0, "raw_fixed"),
    "ConstantFlightDuration": (1, 10_000.0, "raw_fixed"),
    "Cooldown": (1, 10_000.0, "raw_fixed"),
    "DamageAEOSpawnDelay": (1, 10_000.0, "raw_fixed"),
    "DeathSpawnDeployTime": (1, 10_000.0, "raw_fixed"),
    "DashCooldown": (1, 10_000.0, "raw_fixed"),
    "DeployTime": (1, 10_000.0, "raw_fixed"),
    "Duration": (1, 10_000.0, "raw_fixed"),
    "FirstHitDelay": (1, 10_000.0, "raw_fixed"),
    "HideTimeMs": (1, 10_000.0, "raw_fixed"),
    "HideTime": (1, 10_000.0, "raw_fixed"),
    "InitialCooldown": (1, 10_000.0, "raw_fixed"),
    "LifeDuration": (1, 10_000.0, "raw_fixed"),
    "LifeTime": (1, 10_000.0, "raw_fixed"),
    "LoadTime": (1, 10_000.0, "raw_fixed"),
    "LockDelay": (1, 10_000.0, "raw_fixed"),
    "PushbackDelay": (1, 10_000.0, "raw_fixed"),
    "RandomDelay": (1, 10_000.0, "raw_fixed"),
    "ReflectedAttackBuffDuration": (1, 10_000.0, "raw_fixed"),
    "ReleaseLockDelay": (1, 10_000.0, "raw_fixed"),
    "SpawnCharacterDeployTime": (1, 10_000.0, "raw_fixed"),
    "SpawnDelay": (1, 10_000.0, "raw_fixed"),
    "SpawnDeployDelay": (1, 10_000.0, "raw_fixed"),
    "SpawnStartTime": (1, 10_000.0, "raw_fixed"),
    "SpawnTime": (1, 10_000.0, "raw_fixed"),
    "SpecialLoadTime": (1, 10_000.0, "raw_fixed"),
    "SummonDeployDelay": (1, 10_000.0, "raw_fixed"),
    "SummonDeployDelaySecond": (1, 10_000.0, "raw_fixed"),
    "TransitionTime": (1, 10_000.0, "raw_fixed"),
    "TransitionDuration": (1, 10_000.0, "raw_fixed"),
    "TotalDuration": (1, 10_000.0, "raw_fixed"),
    "TrapCastTime": (1, 10_000.0, "raw_fixed"),
    "TriggerDelay": (1, 10_000.0, "raw_fixed"),
    "TimeThreshold": (1, 10_000.0, "raw_fixed"),
    "TetherDuration": (1, 10_000.0, "raw_fixed"),
    "ValidDuration": (1, 10_000.0, "raw_fixed"),
    "WarpDelay": (1, 10_000.0, "raw_fixed"),
    "FinishIfInstigatorDies": (1, 10_000.0, "raw_fixed"),
    "HitFrequency": (2, 5_000.0, "raw_fixed"),
    "AttackCooldown": (2, 5_000.0, "raw_fixed"),
    "AttackDelay": (2, 5_000.0, "raw_fixed"),
    "HitSpeed": (2, 5_000.0, "raw_fixed"),
    "Interval": (2, 5_000.0, "raw_fixed"),
    "ProjectileWaveInterval": (2, 5_000.0, "raw_fixed"),
    "SpawnInterval": (2, 5_000.0, "raw_fixed"),
    "SpawnPauseTime": (2, 5_000.0, "raw_fixed"),
    "TetherHitInterval": (2, 5_000.0, "raw_fixed"),
    "TetherHitActionInterval": (2, 5_000.0, "raw_fixed"),
    # geometry and motion
    "AreaDamageRadius": (3, 10_000.0, "raw_fixed"),
    "CaptureRadius": (3, 10_000.0, "raw_fixed"),
    "ChainedHitRadius": (3, 10_000.0, "raw_fixed"),
    "DashRadius": (3, 10_000.0, "raw_fixed"),
    "DeathDamageRadius": (3, 10_000.0, "raw_fixed"),
    "DetectionRadius": (3, 10_000.0, "raw_fixed"),
    "DeathSpawnRadius": (3, 10_000.0, "raw_fixed"),
    "DeflectRadius": (3, 10_000.0, "raw_fixed"),
    "Height": (3, 10_000.0, "raw_fixed"),
    "JumpHeight": (3, 10_000.0, "raw_fixed"),
    "Radius": (3, 10_000.0, "raw_fixed"),
    "PushBackRadius": (3, 10_000.0, "raw_fixed"),
    "ReflectedAttackRadius": (3, 10_000.0, "raw_fixed"),
    "SpawnCharaterRadius": (3, 10_000.0, "raw_fixed"),
    "SummonRadius": (3, 10_000.0, "raw_fixed"),
    "SpawnRadius": (3, 10_000.0, "raw_fixed"),
    "SummonWidth": (3, 10_000.0, "raw_fixed"),
    "TetherWidth": (3, 10_000.0, "raw_fixed"),
    "ChargeRange": (4, 10_000.0, "raw_fixed"),
    "DashMinRange": (4, 10_000.0, "raw_fixed"),
    "DashFollowUpMinRange": (4, 10_000.0, "raw_fixed"),
    "MinimumRange": (4, 10_000.0, "raw_fixed"),
    "MinRange": (4, 10_000.0, "raw_fixed"),
    "SpecialMinRange": (4, 10_000.0, "raw_fixed"),
    "DragMargin": (4, 10_000.0, "raw_fixed"),
    "DeathSpawnMinRadius": (4, 10_000.0, "raw_fixed"),
    "SpawnMinRadius": (4, 10_000.0, "raw_fixed"),
    "SnipeMinRange": (4, 10_000.0, "raw_fixed"),
    "DashMaxRange": (5, 10_000.0, "raw_fixed"),
    "DashFollowUpMaxRange": (5, 10_000.0, "raw_fixed"),
    "DashRange": (5, 10_000.0, "raw_fixed"),
    "MaxRange": (5, 10_000.0, "raw_fixed"),
    "Range": (5, 10_000.0, "raw_fixed"),
    "SnipeMaxRange": (5, 10_000.0, "raw_fixed"),
    "SpecialRange": (5, 10_000.0, "raw_fixed"),
    "StrongDamageRange": (5, 10_000.0, "raw_fixed"),
    "SpawnMaxRadius": (5, 10_000.0, "raw_fixed"),
    "ProjectileDistance": (3, 10_000.0, "raw_fixed"),
    "TargetRadius": (3, 10_000.0, "raw_fixed"),
    "DragBackSpeed": (7, 1_000.0, "raw_fixed"),
    "DragSelfSpeed": (7, 1_000.0, "raw_fixed"),
    "JumpSpeed": (7, 1_000.0, "raw_fixed"),
    "Speed": (7, 1_000.0, "raw_fixed"),
    "SpeedOverride": (7, 1_000.0, "raw_fixed"),
    "Acceleration": (8, 1_000.0, "raw_fixed"),
    "Gravity": (8, 1_000.0, "raw_fixed"),
    "SingleDeployOffsetAngle": (9, 360.0, "raw_fixed"),
    "SpawnAngleShift": (9, 360.0, "raw_fixed"),
    "DamageReduction": (10, 100.0, "raw_fixed"),
    "DamageScalar": (10, 100.0, "raw_fixed"),
    "DefenseScalar": (10, 100.0, "raw_fixed"),
    "ChargeSpeedMultiplier": (10, 100.0, "raw_fixed"),
    "CrownTowerDamagePercent": (10, 100.0, "raw_fixed"),
    "AllowedOverHealPerc": (10, 100.0, "raw_fixed"),
    "MaxOverHealPercent": (10, 100.0, "raw_fixed"),
    "ShieldPercent": (10, 100.0, "raw_fixed"),
    "ManaOnDeathForOpponent": (0, 10.0, "raw_fixed"),
    "AttractPercentage": (11, 1_000.0, "raw_fixed"),
    "Mass": (11, 1_000.0, "raw_fixed"),
    "PushBackStrength": (11, 1_000.0, "raw_fixed"),
    "Pushback": (11, 1_000.0, "raw_fixed"),
    "DashingPushback": (11, 1_000.0, "raw_fixed"),
    "DashPushBack": (11, 1_000.0, "raw_fixed"),
    "FirstStrongHitPushback": (11, 1_000.0, "raw_fixed"),
    "DeathPushBack": (11, 1_000.0, "raw_fixed"),
    "MeleePushback3": (11, 1_000.0, "raw_fixed"),
    "WarpX": (12, 10_000.0, "raw_fixed"),
    "WarpY": (13, 10_000.0, "raw_fixed"),
    "OperationOrdinal": (14, 10.0, "raw_fixed"),
    "InstigatorDepth": (14, 10.0, "raw_fixed"),
}

_AMOUNT_FIELDS = frozenset(
    {
        "AddedCrownTowerDamage",
        "AddedDamage",
        "BaseDamageAmount",
        "CrownTowerDamagePerHit",
        "Damage",
        "DamagePerHit",
        "DamagePerSecond",
        "DamageSpecial",
        "DashDamage",
        "DeathDamage",
        "HealPerSecond",
        "PushBackDamage",
        "ReflectAttackCrownTowerDamage",
        "ReflectedAttackDamage",
        "StrongDamage",
        "TetherCrownTowerDamage",
        "TetherDamage",
        "VariableDamage2",
        "VariableDamage3",
    }
)

_NON_EXPIRING_TIME_FIELDS = frozenset(
    {"AbilityDuration", "BuffTime", "Duration", "LifeDuration", "LifeTime", "ValidDuration"}
)

_EXACT_SENTINELS: Mapping[str, tuple[str, float]] = {
    "DashMaxRange": ("unbounded", 999_999.0),
    "MaxChainLength": ("unbounded_negative", 0.0),
    "NumberOfUnitsToCapture": ("unbounded", 1_000.0),
}

_GAMEPLAY_CATEGORICAL_FIELDS = frozenset(
    {
        "AffectInvisible",
        "AffectsHidden",
        "AllTargetsHit",
        "AllowBuildingRetargeting",
        "AllowAreaDmgWhenInvisible",
        "AllowWarpWhenAttackSpeedZero",
        "AllowWarpWhenMovementSpeedZero",
        "AoeToAir",
        "AoeToGround",
        "DeprioritizeRepeatTargets",
        "DeprioritizeTargetsWithBuff",
        "DeflectBehaviour",
        "ForceKeepTargetAfterWarp",
        "FullPushBackCollisionCheck",
        "HitsAir",
        "HitsGround",
        "IgnoreBuildings",
        "IgnoreResurrect",
        "IgnorePushbackChecks",
        "LoadFirstHit",
        "MatchOnlyOwnSpawnedTroops",
        "NotCloned",
        "OnlyEnemies",
        "OnlyOwnTroops",
        "OncePerTarget",
        "ParentAsInstigatorForSelfActions",
        "PauseIfAttackSpeedZero",
        "PauseTag",
        "PerformAttackOnReach",
        "RepeatTargets",
        "ResurrectChargeFilter",
        "ResetPathInAir",
        "ResetPathWhenBackToGround",
        "ResetPendingDamageAtWarp",
        "ResetTargetAfterReach",
        "Scatter",
        "SpawnerAliveRequired",
        "SpellAsDeploy",
        "SpawnAttach",
        "StopMovementWhenAtTarget",
        "TargetSelectionMode",
        "UseSpellsTowerDamageMul",
        "UseDeployForSummons",
        "UseDistanceBasedPositioning",
        "UseAttackRange",
        "WaitForTarget",
    }
)


def _level_11_candidate(
    static_logic: StaticCardLogicCatalogV1, node_id: str, field_path: str
) -> tuple[float, str] | None:
    candidates = [
        candidate
        for card in static_logic.cards
        for candidate in card.get("level_scaling_candidates", ())
        if str(candidate.get("node_id")) == node_id and str(candidate.get("field_path")) == field_path
    ]
    known = {
        (float(candidate["level_11_value"]), str(candidate["scaling_family"]))
        for candidate in candidates
        if isinstance(candidate.get("level_11_value"), (int, float))
        and not isinstance(candidate.get("level_11_value"), bool)
    }
    return next(iter(known)) if len(known) == 1 else None


def _physics_operations(
    node_id: str,
    record: Mapping[str, Any],
    *,
    static_logic: StaticCardLogicCatalogV1,
    inherited_rarity: str | None = None,
    suppress_amount_fields: frozenset[str] = frozenset(),
) -> tuple[MechanicOperationV1, ...]:
    operations: list[MechanicOperationV1] = []
    amount_fields = {
        key
        for key in _AMOUNT_FIELDS.intersection(record).difference(suppress_amount_fields)
        for value in (record[key],)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) != 0.0
    }
    for field_name, (slot, scale, basis) in _PHYSICS_FIELDS.items():
        value = record.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        sentinel = _EXACT_SENTINELS.get(field_name)
        if sentinel is not None:
            sentinel_kind, threshold = sentinel
            is_sentinel = value < threshold if sentinel_kind == "unbounded_negative" else value >= threshold
            if is_sentinel:
                operations.append(
                    MechanicOperationV1(
                        trigger="static_contract",
                        operation=f"physics:{field_name}",
                        target="native_owner",
                        condition=(f"sentinel:{sentinel_kind}:{field_name}:{value:g}"),
                    )
                )
                continue
        if slot in {1, 2} and (value < 0 or value >= 99_999):
            non_expiring = field_name in _NON_EXPIRING_TIME_FIELDS and value >= 99_999
            operations.append(
                MechanicOperationV1(
                    trigger="static_contract",
                    operation=f"physics:{field_name}",
                    target="native_owner",
                    mechanic_tags=("non_expiring",) if non_expiring else (),
                    condition=f"sentinel:{field_name}:{value:g}",
                )
            )
            continue
        numeric = [0.0] * len(MECHANIC_NUMERIC_FEATURE_NAMES)
        numeric[slot] = float(value) / scale
        operations.append(
            MechanicOperationV1(
                trigger="static_contract",
                operation=f"physics:{field_name}",
                target="native_owner",
                numeric_features=tuple(numeric),
                condition=f"field:{field_name}",
                amount_basis=basis,
            )
        )
    for field_name in sorted(amount_fields):
        value = float(record[field_name])
        candidate = _level_11_candidate(static_logic, node_id, field_name)
        if candidate is not None:
            value = candidate[0]
            basis = "level11"
        else:
            rarity = record.get("Rarity") or inherited_rarity
            projected, known = _level_11_static_value({**record, "Rarity": rarity}, field_name, static_logic)
            if known:
                value = projected
                basis = "level11"
            else:
                basis = "raw_unbound"
        numeric = [0.0] * len(MECHANIC_NUMERIC_FEATURE_NAMES)
        numeric[6] = value / 1_000.0
        operations.append(
            MechanicOperationV1(
                trigger="static_contract",
                operation=f"amount:{field_name}",
                target="native_selected_targets",
                numeric_features=tuple(numeric),
                condition=f"field:{field_name}",
                amount_basis=basis,
            )
        )
    return tuple(operations)


def _categorical_operations(record: Mapping[str, Any]) -> tuple[MechanicOperationV1, ...]:
    operations: list[MechanicOperationV1] = []
    for field_name in sorted(_GAMEPLAY_CATEGORICAL_FIELDS.intersection(record)):
        value = record[field_name]
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif isinstance(value, str) and value.strip():
            encoded = value.strip()
        else:
            continue
        operations.append(
            MechanicOperationV1(
                trigger="static_contract",
                operation="categorical_contract",
                target="native_owner",
                condition=f"field:{field_name}={encoded}",
            )
        )
    relocation_thresholds = record.get("HideHpThresholds")
    if isinstance(relocation_thresholds, Sequence) and not isinstance(relocation_thresholds, (str, bytes, bytearray)):
        for index, threshold in enumerate(relocation_thresholds):
            if type(threshold) is not int or not 0 <= threshold <= 100:
                continue
            numeric = [0.0] * len(MECHANIC_NUMERIC_FEATURE_NAMES)
            numeric[10] = float(threshold) / 100.0
            numeric[14] = float(index) / 10.0
            operations.append(
                MechanicOperationV1(
                    trigger="health_threshold",
                    operation="health_threshold_relocate_stage",
                    target="native_owner",
                    mechanic_tags=("phase_change", "visibility_transition"),
                    numeric_features=tuple(numeric),
                    condition=f"field:HideHpThresholds[{index}]",
                )
            )
    return tuple(operations)


def _ops_for_root(
    root: str,
    *,
    static_logic: StaticCardLogicCatalogV1,
    references: Mapping[str, Sequence[Mapping[str, Any]]],
    native_effects: Mapping[int, NativeEffectDefinitionV1],
    effect_catalog: EffectSemanticCatalogV1,
    entity_catalog: EntityArchetypeCatalogV1,
    include_ability: bool,
    carrier_rarity: str | None = None,
) -> tuple[MechanicOperationV1, ...]:
    operations: list[MechanicOperationV1] = []
    applications_by_source: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for global_id, effect in native_effects.items():
        effect_vocab_id = effect_catalog.vocab_id(global_id)
        if effect_vocab_id == UNKNOWN_EFFECT_VOCAB_ID:
            continue
        for application in effect.application_sources:
            applications_by_source.setdefault(str(application["source_node_id"]), []).append(
                (effect_vocab_id, application)
            )

    def contextual(
        operation: MechanicOperationV1,
        *,
        source_archetype: str,
        path: str,
        guards: tuple[tuple[str, int, tuple[float, float]], ...],
    ) -> None:
        operations.append(
            _contextualize(operation, source_archetype=source_archetype, control_path=path, guards=guards)
        )

    def emit_record(
        node_id: str,
        record: Mapping[str, Any],
        *,
        path: str,
        guards: tuple[tuple[str, int, tuple[float, float]], ...],
        physical_rarity: str | None,
        inline: bool = False,
    ) -> None:
        root_namespace = str(static_logic.nodes[node_id].get("inheritance_root_node_id") or node_id).split(".", 1)[0]
        source = (
            str(record.get("ClassType"))
            if inline and isinstance(record.get("ClassType"), str)
            else _source_archetype(static_logic, node_id, record)
        )
        local = _local_guards(record, guards)
        semantic_path = f"{path}{_predicate_path_suffix(record)}"
        tags = _node_tags(node_id, record, root_namespace=root_namespace)
        classification = classify_native_action(record)
        if classification is not None:
            tags.update(
                _MECHANIC_TAG_ALIASES.get(tag, tag)
                for tag in classification["mechanic_tags"]
                if _MECHANIC_TAG_ALIASES.get(tag, tag) in MECHANIC_TAG_NAMES
            )
            contextual(
                MechanicOperationV1(
                    trigger="native_action",
                    operation=f"typed:{classification['operation']}",
                    target="native_selected_targets",
                    mechanic_tags=tuple(tags),
                ),
                source_archetype=source,
                path=semantic_path,
                guards=local,
            )
        elif tags:
            contextual(
                MechanicOperationV1(
                    trigger="static_state",
                    operation=(
                        f"class:{record['ClassType']}"
                        if record.get("ClassType") in _EXACT_CLASS_TAGS
                        else "field_semantics"
                    ),
                    target="native_selected_targets",
                    mechanic_tags=tuple(tags),
                ),
                source_archetype=source,
                path=semantic_path,
                guards=local,
            )
        stage_operations = _damage_stage_operations(record, static_logic=static_logic, inherited_rarity=physical_rarity)
        staged_amount_fields = _nested_controller_amount_fields(record)
        if stage_operations:
            staged_amount_fields = staged_amount_fields.union({"Damage", "VariableDamage2", "VariableDamage3"})
        for operation in _physics_operations(
            node_id,
            record,
            static_logic=static_logic,
            inherited_rarity=physical_rarity,
            suppress_amount_fields=staged_amount_fields,
        ):
            contextual(operation, source_archetype=source, path=semantic_path, guards=local)
        for operation in _categorical_operations(record):
            contextual(operation, source_archetype=source, path=semantic_path, guards=local)
        for operation in stage_operations:
            contextual(operation, source_archetype=source, path=semantic_path, guards=local)

    def visit(
        node_id: str,
        *,
        path: str,
        guards: tuple[tuple[str, int, tuple[float, float]], ...],
        active_nodes: tuple[str, ...],
        active_roles: tuple[str, ...],
        physical_rarity: str | None,
    ) -> None:
        if len(active_nodes) > 64:
            raise ValueError("mechanic control/resource path exceeds 64 nodes")
        record = _node_record(static_logic, node_id)
        root_namespace = str(static_logic.nodes[node_id].get("inheritance_root_node_id") or node_id).split(".", 1)[0]
        node_physical_rarity = physical_rarity
        if root_namespace in {"CHARACTER", "BUILDING", "PROJECTILE", "AEO"}:
            node_physical_rarity = (
                _record_rarity(node_id, static_logic=static_logic, references=references) or physical_rarity
            )
        local = _local_guards(record, guards)
        source = _source_archetype(static_logic, node_id, record)
        emit_record(node_id, record, path=path, guards=guards, physical_rarity=node_physical_rarity)
        for field_name, nested in record.items():
            for inline_path, inline_record in _inline_action_records(nested, path=str(field_name)):
                inline_guards = _guards_for_reference(record, inline_path, local)
                emit_record(
                    node_id,
                    inline_record,
                    path=(
                        f"{path}>{_path_signature(record, inline_path)}{_predicate_path_suffix(record, inline_path)}"
                    ),
                    guards=inline_guards,
                    physical_rarity=node_physical_rarity,
                    inline=True,
                )

        for effect_vocab_id, application in sorted(
            applications_by_source.get(node_id, ()), key=lambda value: (str(value[1].get("field_path")), value[0])
        ):
            field_path = str(application.get("field_path") or "")
            application_path = (
                f"{path}>{_path_signature(record, field_path)}{_predicate_path_suffix(record, field_path)}"
            )
            application_guards = _guards_for_reference(record, field_path, local)
            parameters = application.get("parameters", {})
            contextual(
                MechanicOperationV1(
                    trigger=str(application["application_kind"]),
                    operation="apply_effect",
                    target=_target_for_application(str(application["application_kind"])),
                    effect_vocab_id=effect_vocab_id,
                    numeric_features=_mechanic_numeric(parameters if isinstance(parameters, Mapping) else {}),
                    condition=f"effect:{application['application_kind']}",
                ),
                source_archetype=source,
                path=application_path,
                guards=application_guards,
            )

        outgoing = sorted(
            references.get(node_id, ()),
            key=lambda value: (str(value.get("field_path")), str(value.get("reference_id"))),
        )
        for reference in outgoing:
            if str(reference.get("status")) != "resolved":
                continue
            role = str(reference.get("edge_role"))
            if role not in {"control", "data_dependency", "resource"}:
                continue
            field_path = str(reference.get("field_path") or "")
            lowered = field_path.lower()
            if not include_ability and (lowered in {"ability", "heroability"} or lowered.startswith("evolvedspells")):
                continue
            reference_path = f"{path}>{_path_signature(record, field_path)}{_predicate_path_suffix(record, field_path)}"
            reference_guards = _guards_for_reference(record, field_path, local)
            targets = tuple(
                str(value) for value in reference.get("target_node_ids", ()) if str(value) in static_logic.nodes
            )
            for target_node in targets:
                leaf = field_path.rsplit(".", 1)[-1].split("[", 1)[0]
                if role == "data_dependency":
                    operation = _data_dependency_operation(reference, target_node, effect_catalog=effect_catalog)
                    if operation is not None:
                        contextual(operation, source_archetype=source, path=reference_path, guards=reference_guards)
                    continue
                if role == "resource":
                    operation = _resource_operation(
                        reference, target_node, record=record, static_logic=static_logic, entity_catalog=entity_catalog
                    )
                    if operation is not None:
                        contextual(operation, source_archetype=source, path=reference_path, guards=reference_guards)
                if role == "control":
                    control_numeric = _reference_numeric(record, field_path, leaf)
                    if any(control_numeric[:-1]):
                        contextual(
                            MechanicOperationV1(
                                trigger="control_transition",
                                operation="timed_control_edge",
                                target="next_action",
                                numeric_features=control_numeric,
                                condition=f"edge:{leaf}",
                            ),
                            source_archetype=source,
                            path=reference_path,
                            guards=reference_guards,
                        )
                if role == "resource" and leaf in _NON_EXECUTING_RESOURCE_FIELDS:
                    continue
                if role == "control" and bool(reference.get("target_visual")):
                    contextual(
                        MechanicOperationV1(
                            trigger="control_terminal",
                            operation="control_branch_terminal",
                            target="visual_terminal",
                            condition=f"terminal:{target_node}",
                        ),
                        source_archetype=source,
                        path=reference_path,
                        guards=reference_guards,
                    )
                    continue
                if target_node in active_nodes:
                    depth = active_nodes.index(target_node)
                    cycle_roles = (*active_roles[depth:], role)
                    pure_control = all(value == "control" for value in cycle_roles)
                    contextual(
                        MechanicOperationV1(
                            trigger=("control_cycle" if pure_control else "lifecycle_cycle"),
                            operation=("control_cycle_back" if pure_control else "resource_reentry"),
                            target=f"back_to_depth:{depth}",
                            condition="cycle_back:" + ">".join(cycle_roles),
                        ),
                        source_archetype=source,
                        path=reference_path,
                        guards=reference_guards,
                    )
                    continue
                visit(
                    target_node,
                    path=reference_path,
                    guards=reference_guards,
                    active_nodes=(*active_nodes, target_node),
                    active_roles=(*active_roles, role),
                    physical_rarity=node_physical_rarity,
                )

    visit(root, path="root", guards=(), active_nodes=(root,), active_roles=(), physical_rarity=carrier_rarity)
    return tuple(operations)


@dataclass(frozen=True, slots=True)
class MechanicProfileCatalogV1:
    """Structured operation lists addressed by form, Ability, and archetype."""

    card_scope: tuple[int, ...]
    profile_keys: tuple[str, ...]
    profile_operations: tuple[tuple[MechanicOperationV1, ...], ...]
    card_form_profile_ids: tuple[tuple[int, int, int, int], ...]
    ability_profile_ids: tuple[int, ...]
    archetype_profile_ids: tuple[int, ...]
    tower_troop_profile_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        scope = tuple(int(value) for value in self.card_scope)
        keys = tuple(str(value) for value in self.profile_keys)
        operations = tuple(tuple(row) for row in self.profile_operations)
        if tuple(sorted(set(scope))) != scope:
            raise ValueError("MechanicProfile card scope must be unique and sorted")
        if len(keys) != len(operations) or any(not value for value in keys):
            raise ValueError("Mechanic profile keys and operation rows differ")
        if len(set(keys)) != len(keys):
            raise ValueError("Mechanic profile keys must be unique")
        profile_count = 1 + len(keys)
        mappings = (value for row in self.card_form_profile_ids for value in row)
        mappings = (*mappings, *self.ability_profile_ids, *self.archetype_profile_ids, *self.tower_troop_profile_ids)
        if any(not 0 <= int(value) < profile_count for value in mappings):
            raise ValueError("Mechanic profile mapping is outside the catalog")
        if any(len(row) != 4 for row in self.card_form_profile_ids):
            raise ValueError("Card mechanic profiles require four runtime forms")
        object.__setattr__(self, "card_scope", scope)
        object.__setattr__(self, "profile_keys", keys)
        object.__setattr__(self, "profile_operations", operations)

    @classmethod
    def from_sources(
        cls,
        card_specs: Mapping[int, CardSpecV1],
        *,
        card_catalog: CardCatalogV1,
        ability_catalog: AbilityCatalogV1,
        entity_archetype_catalog: EntityArchetypeCatalogV1,
        effect_catalog: EffectSemanticCatalogV1,
        static_logic: StaticCardLogicCatalogV1,
        native_effect_catalog: NativeEffectCatalogV1,
        projectile_catalog: "NativeProjectileCatalogV1",
    ) -> "MechanicProfileCatalogV1":
        scope = tuple(sorted(int(value) for value in card_specs))
        if not (
            card_catalog.raw_card_ids
            == ability_catalog.card_scope
            == entity_archetype_catalog.card_scope
            == effect_catalog.card_scope
            == scope
        ):
            raise ValueError("V4 mechanic catalogs must use one competitive scope")
        card_logic = {int(value["card_id"]): value for value in static_logic.cards}
        references = _reference_index(static_logic)
        native_effects = {effect.buff_global_id: effect for effect in native_effect_catalog.effects}
        root_by_card: dict[int, tuple[str, str, str, str]] = {}
        roots: set[tuple[str, bool]] = set()
        for card_id in scope:
            logic_row = card_logic[card_id]
            normal_root = str(logic_row["root_node_id"])
            evolution_root = normal_root
            evolution = card_specs[card_id].evolution
            if evolution is not None and evolution.evolution_form_id:
                candidate = f"SPELL_EVOLVED.{evolution.evolution_form_id}"
                if candidate not in static_logic.nodes:
                    raise ValueError(f"static logic lacks Evolution root {candidate!r}")
                evolution_root = candidate
            hero_roots = sorted(
                str(value)
                for reference in references.get(normal_root, ())
                if str(reference.get("field_path") or "").startswith("EvolvedSpells")
                for value in reference.get("target_node_ids", ())
                if str(value).startswith("SPELL_HERO.")
            )
            if len(hero_roots) > 1:
                raise ValueError(f"card {card_id} has ambiguous Hero roots: {hero_roots}")
            resolved_hero = card_specs[card_id].attributes.get("resolved_hero_form")
            if isinstance(resolved_hero, Mapping):
                form_id = resolved_hero.get("form_id")
                if not isinstance(form_id, str) or not form_id:
                    raise ValueError(f"card {card_id} has an invalid resolved Hero form")
                hero_root = f"SPELL_HERO.{form_id}"
                if hero_root not in static_logic.nodes:
                    raise ValueError(f"static logic lacks Hero root {hero_root!r}")
                if hero_roots and hero_roots != [hero_root]:
                    raise ValueError(f"card {card_id} Hero roots disagree with its exact binding")
            else:
                hero_root = hero_roots[0] if hero_roots else normal_root
            root_by_card[card_id] = (normal_root, evolution_root, hero_root, hero_root)
            roots.update((root, False) for root in root_by_card[card_id])

        ability_roots: dict[str, str] = {}
        for ability_id in ability_catalog.ability_ids:
            root = f"ABILITY.{ability_id}"
            if root not in static_logic.nodes:
                raise ValueError(f"static logic lacks Ability root {root!r}")
            ability_roots[ability_id] = root
            roots.add((root, True))

        archetype_roots: dict[str, str] = {}
        for key, root in zip(
            entity_archetype_catalog.archetype_keys, entity_archetype_catalog.archetype_node_ids, strict=True
        ):
            if root:
                archetype_roots[key] = root
                roots.add((root, False))

        profile_entries: dict[str, tuple[MechanicOperationV1, ...]] = {}
        for root, include_ability in sorted(roots):
            key = f"root:{root}:ability={int(include_ability)}"
            carrier_rarity = _record_rarity(root, static_logic=static_logic, references=references)
            if carrier_rarity is None and include_ability:
                ability_id = root.split(".", 1)[1]
                source_cards = tuple(card_id for card_id, spec in card_specs.items() if ability_id in spec.ability_ids)
                if len(source_cards) != 1:
                    raise ValueError(f"Ability {ability_id!r} lacks one exact source card")
                carrier_rarity = _record_rarity(
                    root_by_card[source_cards[0]][0], static_logic=static_logic, references=references
                )
            profile_entries[key] = _ops_for_root(
                root,
                static_logic=static_logic,
                references=references,
                native_effects=native_effects,
                effect_catalog=effect_catalog,
                entity_catalog=entity_archetype_catalog,
                include_ability=include_ability,
                carrier_rarity=carrier_rarity,
            )
        profile_entries["tower:dagger_duchess"] = (
            MechanicOperationV1(
                trigger="attack_resource", operation="ammo_stockpile", target="attack_target", mechanic_tags=("ammo",)
            ),
        )
        profile_entries["tower:royal_chef"] = (
            MechanicOperationV1(
                trigger="periodic_support",
                operation="cook_and_buff_ally",
                target="friendly_unit",
                mechanic_tags=("periodic_support",),
            ),
        )
        keys = tuple(sorted(profile_entries))
        profile_id = {key: index + 1 for index, key in enumerate(keys)}

        def root_profile(root: str, include_ability: bool = False) -> int:
            return profile_id[f"root:{root}:ability={int(include_ability)}"]

        card_mapping = [(0, 0, 0, 0)] * card_catalog.vocab_size
        for card_id, card_roots in root_by_card.items():
            card_mapping[card_catalog.vocab_id(card_id)] = tuple(root_profile(root) for root in card_roots)
        ability_mapping = [0] * ability_catalog.vocab_size
        for ability_id, root in ability_roots.items():
            ability_mapping[ability_catalog.vocab_id(ability_id)] = root_profile(root, True)
        archetype_mapping = [0] * entity_archetype_catalog.vocab_size
        for key, root in archetype_roots.items():
            archetype_mapping[entity_archetype_catalog.vocab_id(key)] = root_profile(root)
        tower_mapping = (0, 0, 0, profile_id["tower:dagger_duchess"], profile_id["tower:royal_chef"])
        return cls(
            scope,
            keys,
            tuple(profile_entries[key] for key in keys),
            tuple(card_mapping),
            tuple(ability_mapping),
            tuple(archetype_mapping),
            tower_mapping,
        )

    @classmethod
    def empty(
        cls,
        *,
        card_catalog: CardCatalogV1,
        ability_catalog: AbilityCatalogV1,
        entity_archetype_catalog: EntityArchetypeCatalogV1,
    ) -> "MechanicProfileCatalogV1":
        return cls(
            card_catalog.raw_card_ids,
            (),
            (),
            tuple((0, 0, 0, 0) for _ in range(card_catalog.vocab_size)),
            tuple(0 for _ in range(ability_catalog.vocab_size)),
            tuple(0 for _ in range(entity_archetype_catalog.vocab_size)),
            (0, 0, 0, 0, 0),
        )

    @property
    def profile_count(self) -> int:
        return 1 + len(self.profile_keys)

    @property
    def catalog_id(self) -> str:
        return content_hash(
            {
                "version": "mechanic-profile-catalog.v1",
                "effect_tags": EFFECT_TAG_NAMES,
                "mechanic_tags": MECHANIC_TAG_NAMES,
                "numeric_features": MECHANIC_NUMERIC_FEATURE_NAMES,
                "guard_limit": MECHANIC_GUARD_LIMIT,
                "card_scope": self.card_scope,
                "profile_keys": self.profile_keys,
                "profile_operations": self.profile_operations,
                "card_form_profile_ids": self.card_form_profile_ids,
                "ability_profile_ids": self.ability_profile_ids,
                "archetype_profile_ids": self.archetype_profile_ids,
                "tower_troop_profile_ids": self.tower_troop_profile_ids,
            }
        )

    def tensor_tables(self) -> dict[str, Tensor | tuple[str, ...]]:
        flat = [
            (profile_index, operation)
            for profile_index, operations in enumerate(self.profile_operations, start=1)
            for operation in operations
        ]
        trigger_keys = tuple(sorted({operation.trigger for _, operation in flat}))
        operation_keys = tuple(sorted({operation.operation for _, operation in flat}))
        target_keys = tuple(sorted({operation.target for _, operation in flat}))
        source_keys = tuple(sorted({operation.source_archetype for _, operation in flat}))
        control_path_keys = tuple(sorted({operation.control_path for _, operation in flat}))
        condition_keys = tuple(sorted({operation.condition for _, operation in flat}))
        amount_basis_keys = tuple(sorted({operation.amount_basis for _, operation in flat}))
        guard_keys = ("<pad>",) + tuple(
            sorted({guard for _, operation in flat for guard in operation.guard_archetypes})
        )
        trigger_id = {key: index for index, key in enumerate(trigger_keys)}
        operation_id = {key: index for index, key in enumerate(operation_keys)}
        target_id = {key: index for index, key in enumerate(target_keys)}
        source_id = {key: index for index, key in enumerate(source_keys)}
        control_path_id = {key: index for index, key in enumerate(control_path_keys)}
        condition_id = {key: index for index, key in enumerate(condition_keys)}
        amount_basis_id = {key: index for index, key in enumerate(amount_basis_keys)}
        guard_id = {key: index for index, key in enumerate(guard_keys)}

        def padded_guards(operation: MechanicOperationV1) -> tuple[int, ...]:
            values = tuple(guard_id[value] for value in operation.guard_archetypes)
            return values + (0,) * (MECHANIC_GUARD_LIMIT - len(values))

        def padded_polarities(operation: MechanicOperationV1) -> tuple[float, ...]:
            values = tuple(float(value) for value in operation.guard_polarities)
            return values + (0.0,) * (MECHANIC_GUARD_LIMIT - len(values))

        def scaled_guard_number(value: float) -> float:
            return math.copysign(math.log1p(abs(float(value))) / math.log1p(10_000.0), float(value))

        def padded_guard_numeric(operation: MechanicOperationV1) -> tuple[tuple[float, float], ...]:
            values = tuple(
                (scaled_guard_number(left), scaled_guard_number(right)) for left, right in operation.guard_numeric
            )
            return values + ((0.0, 0.0),) * (MECHANIC_GUARD_LIMIT - len(values))

        return {
            "trigger_keys": trigger_keys,
            "operation_keys": operation_keys,
            "target_keys": target_keys,
            "source_keys": source_keys,
            "control_path_keys": control_path_keys,
            "condition_keys": condition_keys,
            "amount_basis_keys": amount_basis_keys,
            "guard_keys": guard_keys,
            "profile_id": torch.tensor([profile for profile, _ in flat], dtype=torch.long),
            "trigger_id": torch.tensor([trigger_id[operation.trigger] for _, operation in flat], dtype=torch.long),
            "operation_id": torch.tensor(
                [operation_id[operation.operation] for _, operation in flat], dtype=torch.long
            ),
            "target_id": torch.tensor([target_id[operation.target] for _, operation in flat], dtype=torch.long),
            "source_id": torch.tensor(
                [source_id[operation.source_archetype] for _, operation in flat], dtype=torch.long
            ),
            "control_path_id": torch.tensor(
                [control_path_id[operation.control_path] for _, operation in flat], dtype=torch.long
            ),
            "condition_id": torch.tensor(
                [condition_id[operation.condition] for _, operation in flat], dtype=torch.long
            ),
            "amount_basis_id": torch.tensor(
                [amount_basis_id[operation.amount_basis] for _, operation in flat], dtype=torch.long
            ),
            "guard_archetype_id": torch.tensor(
                [padded_guards(operation) for _, operation in flat], dtype=torch.long
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT),
            "guard_polarity": torch.tensor(
                [padded_polarities(operation) for _, operation in flat], dtype=torch.float32
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT),
            "guard_numeric": torch.tensor(
                [padded_guard_numeric(operation) for _, operation in flat], dtype=torch.float32
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT, 2),
            "guard_mask": torch.tensor(
                [
                    [index < len(operation.guard_archetypes) for index in range(MECHANIC_GUARD_LIMIT)]
                    for _, operation in flat
                ],
                dtype=torch.bool,
            ).reshape(len(flat), MECHANIC_GUARD_LIMIT),
            "tags": torch.tensor(
                [[float(name in operation.mechanic_tags) for name in MECHANIC_TAG_NAMES] for _, operation in flat],
                dtype=torch.float32,
            ).reshape(len(flat), len(MECHANIC_TAG_NAMES)),
            "effect_vocab_id": torch.tensor([operation.effect_vocab_id for _, operation in flat], dtype=torch.long),
            "produced_archetype_id": torch.tensor(
                [operation.produced_archetype_id for _, operation in flat], dtype=torch.long
            ),
            "numeric": torch.tensor([operation.numeric_features for _, operation in flat], dtype=torch.float32).reshape(
                len(flat), len(MECHANIC_NUMERIC_FEATURE_NAMES)
            ),
        }
