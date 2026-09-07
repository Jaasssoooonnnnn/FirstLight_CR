"""Stable level-11 card and mechanic features for semantic policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from native_runner.arena import native_building_footprint
from native_runner.contracts import AbilitySpecV1, CardKind, CardSpecV1
from native_runner.normal_form_evidence import NORMAL_MODE_HERO_FORM_TO_BASE_CARD

if TYPE_CHECKING:
    from native_runner.resource_compiler.card_logic import StaticCardLogicCatalogV1


CARD_FEATURE_SCHEMA_VERSION = "semantic-card-feature-vector.v4"

CARD_STATIC_FEATURE_NAMES = (
    "elixir_cost_over_10",
    "hitpoints_level11_over_5000",
    "primary_damage_level11_over_1000",
    "damage_per_second_level11_over_1000",
    "hit_speed_over_5000ms",
    "range_over_10_tiles",
    "move_speed_over_1000",
    "deploy_time_over_5000ms",
    "deploy_count_over_10",
    "circular_radius_over_10_tiles",
    "projectile_speed_over_10000",
    "knockback_over_10000",
    "effect_duration_over_60000ms",
    "shield_level11_over_5000",
    "primary_crown_tower_damage_ratio",
    "stun_duration_over_5000ms",
    "chain_targets_over_10",
    "chain_radius_over_10000",
    "secondary_damage_level11_over_1000",
    "taunt_radius_over_10000",
    "ability_shield_level11_over_5000",
    "ability_duration_over_10000ms",
    "ability_cooldown_over_30000ms",
    "ability_elixir_cost_over_10",
    "ability_cast_time_over_5000ms",
    "linear_sweep_range_over_10000",
    "linear_sweep_width_over_10000",
    "linear_sweep_depth_over_10000",
    "building_width_over_10_tiles",
    "building_damage_ratio",
    "attack_sequence_hit_count_over_10",
)

CARD_MECHANIC_FEATURE_NAMES = (
    "kind_troop",
    "kind_spell",
    "kind_building",
    "kind_hero",
    "attacks_ground",
    "attacks_air",
    "targets_only_buildings",
    "has_evolution",
    "has_ability",
    "has_timed_effect",
    "has_circular_area",
    "has_projectile",
    "has_knockback",
    "has_shield",
)


LEVEL_11_MULTIPLIER_PERCENT: Mapping[str, int] = {
    "Common": 256,
    "Rare": 212,
    "Epic": 160,
    "Legendary": 121,
    "Champion": 100,
    "Experimental": 121,
}

# Exact-build native Hero forms, bound to the decoded class-203 source table
# and the shared policy-form evidence. Evolution entities are mapped through
# EntityStateV1.evolution_state.card_id instead.
HERO_FORM_TO_BASE_CARD: Mapping[int, int] = NORMAL_MODE_HERO_FORM_TO_BASE_CARD
ENTITY_FORM_TO_BASE_CARD: Mapping[int, int] = MappingProxyType(
    {
        **NORMAL_MODE_HERO_FORM_TO_BASE_CARD,
        # Spirit Empress is the visible policy card. Native combat entities
        # use one of two hidden spell-character records for her grounded and
        # mounted forms; both remain compositional children of the same card.
        26_000_104: 28_000_025,
        26_000_105: 28_000_025,
    }
)
_FORWARD_LINEAR_SWEEP_PROJECTILES = frozenset({"LogProjectileRolling", "BarbLogProjectileRolling"})


def level_11_value(spec: CardSpecV1, value: float | int | None, *, carrier_rarity: str | None = None) -> float | None:
    """Project one scalable base stat to tournament level 11.

    Native v15 uses integer truncation after applying the rarity multiplier.
    Non-scalable values such as durations, speeds, ranges and costs must not use
    this helper.  Entity/projectile/area-effect records can declare a scaling
    rarity distinct from the card root, so callers with carrier evidence pass
    that rarity explicitly.
    """

    if value is None:
        return None
    rarity = carrier_rarity or str(spec.categorical_features.get("rarity") or "")
    multiplier = LEVEL_11_MULTIPLIER_PERCENT.get(rarity)
    if multiplier is None:
        raise ValueError(f"card {spec.card_id} has no level-11 rarity curve for {rarity!r}")
    return float(math.floor(float(value) * multiplier / 100.0))


@dataclass(frozen=True, slots=True)
class _NumericSource:
    node_id: str
    record: Mapping[str, Any]
    scaling_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class _CarrierNumber:
    value: float
    rarity: str | None
    node_id: str
    field_name: str


@dataclass(frozen=True, slots=True)
class _PeriodicDamage:
    per_hit: _CarrierNumber
    per_second: _CarrierNumber
    interval_ms: float


@dataclass(frozen=True, slots=True)
class _LogicProjection:
    sources: tuple[_NumericSource, ...]
    records: tuple[Mapping[str, Any], ...]


def _numeric_sources(card_id: int, static_logic: "StaticCardLogicCatalogV1 | None") -> tuple[_NumericSource, ...]:
    if static_logic is None:
        return ()
    card = next((item for item in static_logic.cards if int(item["card_id"]) == card_id), None)
    if card is None:
        return ()
    scaling_fields_by_node: dict[str, set[str]] = {}
    for candidate in card.get("level_scaling_candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        node_id = str(candidate.get("node_id") or "")
        field_path = str(candidate.get("field_path") or "").lower()
        if node_id and field_path:
            scaling_fields_by_node.setdefault(node_id, set()).add(field_path)
    sources: list[_NumericSource] = []
    for node_id in card.get("closure_node_ids", ()):
        node = static_logic.nodes.get(str(node_id))
        if not isinstance(node, Mapping):
            continue
        record = node.get("effective_record", node.get("merged_record"))
        if isinstance(record, Mapping):
            sources.append(
                _NumericSource(str(node_id), record, frozenset(scaling_fields_by_node.get(str(node_id), ())))
            )
    return tuple(sources)


def _numeric_records(sources: Sequence[_NumericSource]) -> tuple[Mapping[str, Any], ...]:
    return tuple(source.record for source in sources)


def _card_logic_record(card_id: int, static_logic: "StaticCardLogicCatalogV1 | None") -> Mapping[str, Any] | None:
    if static_logic is None:
        return None
    return next((card for card in static_logic.cards if int(card["card_id"]) == int(card_id)), None)


def _reference_index(static_logic: "StaticCardLogicCatalogV1 | None") -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for reference in getattr(static_logic, "references", ()):
        if not isinstance(reference, Mapping):
            continue
        grouped.setdefault(str(reference.get("source_node_id") or ""), []).append(reference)
    return {
        source: tuple(
            sorted(
                references,
                key=lambda item: (
                    str(item.get("field_path") or ""),
                    tuple(str(value) for value in item.get("target_node_ids", ())),
                ),
            )
        )
        for source, references in grouped.items()
        if source
    }


def _role_projection(
    card_id: int,
    static_logic: "StaticCardLogicCatalogV1 | None",
    all_sources: Sequence[_NumericSource],
    reference_index: Mapping[str, tuple[Mapping[str, Any], ...]],
    *,
    roots: Sequence[str] | None = None,
    include_card_variants: bool = False,
) -> _LogicProjection:
    """Return only nodes reachable through one concrete gameplay role.

    ``closure_node_ids`` is an inventory, not an effect ordering.  In
    particular it may contain inherited templates and evolution/Hero branches
    which are not part of the base card effect.  Static feature selection must
    therefore walk the typed reference graph from the card root (or an exact
    Ability root) instead of taking the first matching number in that closure.
    """

    card = _card_logic_record(card_id, static_logic)
    if card is None or not reference_index:
        source_tuple = tuple(all_sources)
        return _LogicProjection(source_tuple, _numeric_records(source_tuple))

    allowed = {str(value) for value in card.get("closure_node_ids", ())}
    selected_roots = tuple(
        str(value)
        for value in (roots if roots is not None else (str(card.get("root_node_id") or ""),))
        if value and str(value) in allowed
    )
    if not selected_roots:
        return _LogicProjection((), ())

    distance = {root: 0 for root in selected_roots}
    queue = list(selected_roots)
    cursor = 0
    while cursor < len(queue):
        source_id = queue[cursor]
        cursor += 1
        for reference in reference_index.get(source_id, ()):
            if str(reference.get("status") or "") != "resolved":
                continue
            reference_class = str(reference.get("reference_class") or "")
            if reference_class == "inheritance":
                continue
            if reference.get("gameplay_reference") is not True:
                continue
            field_path = str(reference.get("field_path") or "")
            lowered_path = field_path.lower()
            leaf = field_path.rsplit(".", 1)[-1].split("[", 1)[0]
            if source_id not in selected_roots and leaf in {
                "ActivationSpawnCharacter",
                "AttachedCharacter",
                "ClonedVersion",
                "DeathSpawn",
                "DeathSpawnCharacter",
                "DeathSpawnCharacter2",
                "DeathSpawnCharacter3",
                "DeathSpawnProjectile",
                "SpawnCharacter",
                "SpawnCharacter2",
                "SpawnCharacter3",
                "SpawnCharacterWithDeploy",
                "SpawnObject",
                "SummonCharacter",
                "SummonCharacterSecond",
                "SummonCharactersList",
            }:
                continue
            if lowered_path.startswith("ignorebuff") or (not include_card_variants and lowered_path == "ability"):
                continue
            for raw_target in reference.get("target_node_ids", ()):
                target_id = str(raw_target)
                if target_id not in allowed:
                    continue
                if not include_card_variants and (
                    lowered_path.startswith("evolvedspells") or target_id.startswith(("SPELL_EVOLVED.", "SPELL_HERO."))
                ):
                    continue
                if target_id in distance:
                    continue
                distance[target_id] = distance[source_id] + 1
                queue.append(target_id)

    namespace_priority = {"AEO": 0, "PROJECTILE": 1, "CHARACTER": 2, "BUILDING": 3}
    by_id = {source.node_id: source for source in all_sources}
    sources = tuple(
        by_id[node_id]
        for node_id in sorted(
            set(distance).intersection(by_id),
            key=lambda node_id: (distance[node_id], namespace_priority.get(node_id.split(".", 1)[0], 10), node_id),
        )
    )
    return _LogicProjection(sources, _numeric_records(sources))


def _source_rarity(
    source: _NumericSource, source_index: Mapping[str, _NumericSource], *, visited: frozenset[str] = frozenset()
) -> str | None:
    rarity = source.record.get("Rarity")
    if rarity is not None:
        candidate = str(rarity)
        if candidate in LEVEL_11_MULTIPLIER_PERCENT:
            return candidate

    base = source.record.get("Base")
    if not isinstance(base, str) or base in visited:
        return None
    base_source = source_index.get(base)
    if base_source is None and "." in base:
        base_name = base.split(".", 1)[1]
        name_matches = tuple(
            candidate
            for node_id, candidate in source_index.items()
            if "." in node_id and node_id.split(".", 1)[1] == base_name
        )
        if len(name_matches) == 1:
            base_source = name_matches[0]
    if base_source is None:
        return None
    return _source_rarity(base_source, source_index, visited=visited | {source.node_id})


def _carrier_number(
    source: _NumericSource, source_index: Mapping[str, _NumericSource], field_name: str, value: float | int
) -> _CarrierNumber:
    return _CarrierNumber(
        value=float(value), rarity=_source_rarity(source, source_index), node_id=source.node_id, field_name=field_name
    )


def _first_numeric_carrier(sources: Sequence[_NumericSource], *keys: str) -> _CarrierNumber | None:
    lowered = {key.lower() for key in keys}
    source_index = {source.node_id: source for source in sources}
    for source in sources:
        for key, value in source.record.items():
            if str(key).lower() not in lowered:
                continue
            if str(key).lower() not in source.scaling_fields:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            return _carrier_number(source, source_index, str(key), value)
    return None


def _numeric_carriers(sources: Sequence[_NumericSource], *keys: str) -> tuple[_CarrierNumber, ...]:
    lowered = {key.lower() for key in keys}
    source_index = {source.node_id: source for source in sources}
    result: list[_CarrierNumber] = []
    for source in sources:
        for key, value in source.record.items():
            if str(key).lower() not in lowered:
                continue
            if str(key).lower() not in source.scaling_fields:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            result.append(_carrier_number(source, source_index, str(key), value))
    return tuple(result)


def _attack_sequence_carriers(sources: Sequence[_NumericSource]) -> tuple[_CarrierNumber, ...]:
    source_index = {source.node_id: source for source in sources}
    for source in sources:
        sequence = source.record.get("AttackSequenceList")
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes, bytearray)):
            continue
        result: list[_CarrierNumber] = []
        for index, entry in enumerate(sequence):
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("Damage")
            field_path = f"attacksequencelist[{index}].damage"
            if (
                field_path not in source.scaling_fields
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            result.append(_carrier_number(source, source_index, field_path, value))
        if result:
            return tuple(result)
    return ()


def _periodic_damage(sources: Sequence[_NumericSource]) -> _PeriodicDamage | None:
    for per_second in _numeric_carriers(sources, "DamagePerSecond"):
        source = next(item for item in sources if item.node_id == per_second.node_id)
        frequency = source.record.get("HitFrequency")
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or float(frequency) <= 0.0:
            continue
        per_hit = per_second.value * float(frequency) / 1000.0
        return _PeriodicDamage(
            per_hit=_CarrierNumber(
                value=per_hit,
                rarity=per_second.rarity,
                node_id=per_second.node_id,
                field_name="DamagePerSecond*HitFrequency",
            ),
            per_second=per_second,
            interval_ms=float(frequency),
        )
    return None


def _carrier_record(carrier: _CarrierNumber | None, sources: Sequence[_NumericSource]) -> Mapping[str, Any]:
    if carrier is None:
        return {}
    return next((source.record for source in sources if source.node_id == carrier.node_id), {})


def _crown_tower_damage_ratio(
    spec: CardSpecV1,
    carrier: _CarrierNumber | None,
    sources: Sequence[_NumericSource],
    *,
    projected_damage: float | None,
) -> float:
    """Return the tower ratio declared by the selected damage carrier itself."""

    if carrier is None:
        return 0.0
    record = _carrier_record(carrier, sources)
    percent = record.get("CrownTowerDamagePercent")
    if isinstance(percent, (int, float)) and not isinstance(percent, bool):
        return max(0.0, 1.0 + float(percent) / 100.0)

    absolute = record.get("CrownTowerDamagePerHit")
    if (
        isinstance(absolute, (int, float))
        and not isinstance(absolute, bool)
        and projected_damage is not None
        and projected_damage > 0.0
    ):
        projected_tower_damage = _project_carrier_value(
            spec,
            _CarrierNumber(
                value=float(absolute),
                rarity=carrier.rarity,
                node_id=carrier.node_id,
                field_name="CrownTowerDamagePerHit",
            ),
        )
        if projected_tower_damage is not None:
            return max(0.0, projected_tower_damage / projected_damage)
    return 1.0


def _building_damage_ratio(carrier: _CarrierNumber | None, sources: Sequence[_NumericSource]) -> float:
    """Return the building multiplier declared by the primary carrier."""

    if carrier is None:
        return 0.0
    value = _carrier_record(carrier, sources).get("BuildingDamagePercent")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value) / 100.0)
    return 1.0


def _effect_duration_ms(spec: CardSpecV1, sources: Sequence[_NumericSource]) -> float:
    if spec.kind != CardKind.SPELL:
        return 0.0
    durations = tuple(
        float(value)
        for source in sources
        if source.node_id.startswith("AEO.")
        for value in (source.record.get("LifeDuration"),)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 50.0
    )
    return max(durations, default=0.0)


def _matching_numeric_carrier(
    card_id: int, sources: Sequence[_NumericSource], value: float | int | None, *keys: str
) -> _CarrierNumber | None:
    if value is None:
        return None
    lowered = {key.lower() for key in keys}
    source_index = {source.node_id: source for source in sources}
    matches: list[_CarrierNumber] = []
    for source in sources:
        for key, candidate in source.record.items():
            if str(key).lower() not in lowered:
                continue
            if str(key).lower() not in source.scaling_fields:
                continue
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                continue
            if not math.isclose(float(candidate), float(value)):
                continue
            matches.append(_carrier_number(source, source_index, str(key), candidate))

    rarities = {match.rarity for match in matches if match.rarity is not None}
    if len(rarities) > 1:
        evidence = ", ".join(f"{match.node_id}.{match.field_name}={match.rarity!r}" for match in matches)
        raise ValueError(f"card {card_id} value {value} has conflicting carrier rarities: {evidence}")
    if rarities:
        rarity = next(iter(rarities))
        match = next(item for item in matches if item.rarity == rarity)
        return _CarrierNumber(value=float(value), rarity=rarity, node_id=match.node_id, field_name=match.field_name)
    if matches:
        match = matches[0]
        return _CarrierNumber(value=float(value), rarity=None, node_id=match.node_id, field_name=match.field_name)
    return _CarrierNumber(
        value=float(value), rarity=None, node_id="<card-root>", field_name=keys[0] if keys else "<unknown>"
    )


def _project_carrier_value(spec: CardSpecV1, carrier: _CarrierNumber | None) -> float | None:
    if carrier is None:
        return None
    return level_11_value(spec, carrier.value, carrier_rarity=carrier.rarity)


def _first_numeric(records: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    lowered = {key.lower() for key in keys}
    for record in records:
        for key, value in record.items():
            if str(key).lower() not in lowered:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            return float(value)
    return None


@dataclass(frozen=True, slots=True)
class CardFeatureRow:
    """Named values before packing into the model's stable feature schema."""

    level_hitpoints: float | None
    level_damage: float | None
    level_damage_per_second: float | None
    building_damage_ratio: float
    attack_sequence_hit_count: int
    level_shield: float | None
    crown_tower_damage_ratio: float
    stun_duration_ms: float
    chained_targets: float
    chained_radius_units: float
    special_damage: float | None
    effect_duration_ms: float
    taunt_radius_units: float
    ability_shield: float | None
    ability_duration_ms: float
    ability_cooldown_ms: float
    ability_cost: float
    ability_cast_time_ms: float
    forward_linear_sweep: bool
    linear_sweep_range_units: float
    linear_sweep_width_units: float
    linear_sweep_depth_units: float
    linear_sweep_direction_x: float
    linear_sweep_direction_y: float


def _ability_index(
    ability_specs: (Mapping[str, AbilitySpecV1] | Sequence[AbilitySpecV1] | None),
) -> Mapping[str, tuple[AbilitySpecV1, ...]]:
    if ability_specs is None:
        return {}
    if isinstance(ability_specs, Mapping):
        for ability_id, ability in ability_specs.items():
            if str(ability_id) != str(ability.ability_id):
                raise ValueError(f"ability index key {ability_id!r} does not match AbilitySpec {ability.ability_id!r}")
        source = tuple(ability_specs.values())
    else:
        source = tuple(ability_specs)
    grouped: dict[str, list[AbilitySpecV1]] = {}
    for ability in source:
        grouped.setdefault(str(ability.ability_id), []).append(ability)
    return {ability_id: tuple(matches) for ability_id, matches in grouped.items()}


def _exact_abilities(
    spec: CardSpecV1, ability_index: Mapping[str, tuple[AbilitySpecV1, ...]], *, required: bool
) -> tuple[AbilitySpecV1, ...]:
    resolved: list[AbilitySpecV1] = []
    for ability_id in spec.ability_ids:
        matches = ability_index.get(str(ability_id), ())
        if len(matches) != 1:
            if required:
                raise ValueError(
                    f"card {spec.card_id} ability {ability_id!r} resolved to {len(matches)} exact AbilitySpec rows"
                )
            continue
        ability = matches[0]
        if ability.source_card_id is not None and int(ability.source_card_id) != int(spec.card_id):
            raise ValueError(
                f"card {spec.card_id} ability {ability_id!r} declares source card {ability.source_card_id}"
            )
        resolved.append(ability)
    return tuple(resolved)


def _unique_exact_number(card_id: int, abilities: Sequence[AbilitySpecV1], attribute: str) -> float | None:
    values = tuple(
        float(value) for ability in abilities for value in (getattr(ability, attribute),) if value is not None
    )
    if not values:
        return None
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"card {card_id} has invalid exact ability {attribute}: {values}")
    first = values[0]
    if any(not math.isclose(value, first) for value in values[1:]):
        raise ValueError(f"card {card_id} has ambiguous exact ability {attribute}: {values}")
    return first


def _unique_exact_cooldown(card_id: int, abilities: Sequence[AbilitySpecV1]) -> float | None:
    """Return the policy cooldown, normalizing exact single-use abilities.

    The runtime content omits ``Cooldown`` for a one-charge ability because it
    can never become ready again on that entity.  That is an exact zero-valued
    policy feature, not missing evidence.  Any other omitted cooldown remains
    incomplete and is rejected by the caller.
    """

    if not abilities:
        return None
    values: list[float] = []
    for ability in abilities:
        cooldown = ability.cooldown_ms
        if cooldown is None:
            if ability.charges != 1:
                return None
            cooldown = 0
        value = float(cooldown)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"card {card_id} has invalid exact ability cooldown_ms: {value}")
        values.append(value)
    first = values[0]
    if any(not math.isclose(value, first) for value in values[1:]):
        raise ValueError(f"card {card_id} has ambiguous exact ability cooldown_ms: {tuple(values)}")
    return first


def _exact_effect_duration(card_id: int, abilities: Sequence[AbilitySpecV1]) -> float | None:
    duration_keys = frozenset(
        {
            "abilityduration",
            "abilitystateduration",
            "bufftime",
            "duration",
            "durationms",
            "durationraw",
            "validduration",
        }
    )
    durations: list[float] = []
    for ability in abilities:
        definition = ability.attributes.get("definition")
        records: list[Mapping[str, Any]] = []
        if isinstance(definition, Mapping):
            records.append(definition)
        records.extend(effect.parameters for effect in ability.effect_graph if isinstance(effect.parameters, Mapping))
        for record in records:
            for key, value in record.items():
                normalized = "".join(character for character in str(key).lower() if character.isalnum())
                if normalized not in duration_keys:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"card {card_id} has invalid exact ability duration {key}={value!r}")
                durations.append(float(value))
    return max(durations) if durations else None


def _reconcile_exact_number(card_id: int, label: str, inferred: float | None, exact: float | None) -> float | None:
    if inferred is not None and exact is not None and not math.isclose(inferred, exact):
        raise ValueError(f"card {card_id} inferred {label} {inferred} conflicts with exact AbilitySpec value {exact}")
    return exact if exact is not None else inferred


def _ability_projection(
    card_id: int,
    exact_abilities: Sequence[AbilitySpecV1],
    static_logic: "StaticCardLogicCatalogV1 | None",
    all_sources: Sequence[_NumericSource],
    reference_index: Mapping[str, tuple[Mapping[str, Any], ...]],
) -> _LogicProjection:
    if not exact_abilities:
        return _LogicProjection((), ())
    ability_roots = {f"ABILITY.{ability.ability_id}" for ability in exact_abilities}
    roots = set(ability_roots)
    # The concrete Hero form owns shield/HP state while the Ability node owns
    # activation timing.  Seed only forms explicitly bound to this Ability;
    # do not admit every Hero/evolution node in the card closure.
    for source_id, references in reference_index.items():
        if any(
            str(reference.get("field_path") or "").lower() == "ability"
            and ability_roots.intersection(str(value) for value in reference.get("target_node_ids", ()))
            for reference in references
        ):
            roots.add(source_id)
    return _role_projection(
        card_id, static_logic, all_sources, reference_index, roots=tuple(sorted(roots)), include_card_variants=True
    )


def _typed_secondary_damage_carrier(
    spec: CardSpecV1,
    sources: Sequence[_NumericSource],
    primary: _CarrierNumber | None,
    sequence: Sequence[_CarrierNumber],
) -> _CarrierNumber | None:
    """Resolve one explicit secondary damage channel, or return unknown.

    A closure can contain dozens of Damage fields.  A stable secondary channel
    must instead be named by the normalized card mechanics (deploy/death/etc.)
    or by a concrete attack sequence.  Ambiguous multi-effect cases stay zero.
    """

    raw_values = {
        float(value)
        for operation in spec.mechanics
        if operation.effect.lower() == "damage"
        for value in (operation.parameters.get("raw"),)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (primary is None or not math.isclose(float(value), primary.value))
    }
    if len(raw_values) == 1:
        value = next(iter(raw_values))
        return _matching_numeric_carrier(spec.card_id, sources, value, "Damage", "DeathDamage", "DamageSpecial")
    if len(raw_values) > 1:
        return None

    distinct_sequence = tuple(
        carrier for carrier in sequence if primary is None or not math.isclose(carrier.value, primary.value)
    )
    distinct_values = {carrier.value for carrier in distinct_sequence}
    if len(distinct_values) == 1:
        return distinct_sequence[0]
    return None


def build_card_feature_rows(
    card_specs: Mapping[int, CardSpecV1],
    *,
    static_logic: "StaticCardLogicCatalogV1 | None" = None,
    ability_specs: (Mapping[str, AbilitySpecV1] | Sequence[AbilitySpecV1] | None) = None,
) -> Mapping[int, CardFeatureRow]:
    """Build fixed-semantics rows; no feature position depends on card type.

    Passing the exact catalog ``ability_specs`` enables the policy-ready V3
    contract.  Every referenced ability must resolve uniquely; the decoded
    AbilitySpec is authoritative while any value independently inferred from
    the static closure must agree with it.  Omitting the argument preserves
    the static-only fallback API used by non-policy callers.
    """

    ability_index = _ability_index(ability_specs)
    references = _reference_index(static_logic)
    result: dict[int, CardFeatureRow] = {}
    for card_id, spec in card_specs.items():
        exact_abilities = _exact_abilities(spec, ability_index, required=ability_specs is not None)
        exact_cost = _unique_exact_number(card_id, exact_abilities, "elixir_cost")
        exact_cooldown = _unique_exact_cooldown(card_id, exact_abilities)
        exact_cast_time = _unique_exact_number(card_id, exact_abilities, "cast_time_ms")
        if exact_abilities and (
            exact_cost is None
            or exact_cooldown is None
            or exact_cast_time is None
            or any(not ability.effect_graph for ability in exact_abilities)
        ):
            raise ValueError(f"card {card_id} has an incomplete exact AbilitySpec cost/cooldown/cast/effect contract")
        exact_effect_duration = _exact_effect_duration(card_id, exact_abilities)
        all_sources = _numeric_sources(card_id, static_logic)
        base_projection = _role_projection(card_id, static_logic, all_sources, references)
        sources = base_projection.sources
        records = base_projection.records
        ability_projection = _ability_projection(card_id, exact_abilities, static_logic, all_sources, references)
        has_stun = any(
            any(marker in str(value).lower() for marker in ("stun", "freeze", "zap"))
            for record in records
            for key, value in record.items()
            if str(key).lower() in {"targetbuff", "buff", "debuff"}
        )
        stun_duration = _first_numeric(records, "StunDuration", "FreezeTime", "BuffTime") if has_stun else None
        chained_targets = _first_numeric(records, "ChainedHitCount")
        chained_radius = _first_numeric(records, "ChainedHitRadius")
        sweep_records = tuple(
            record for record in records if str(record.get("Name", "")) in _FORWARD_LINEAR_SWEEP_PROJECTILES
        )
        forward_linear_sweep = bool(sweep_records)
        sweep_range = 0.0
        sweep_width = 0.0
        sweep_depth = 0.0
        if sweep_records:
            sweep_record = sweep_records[0]
            required = {
                key: sweep_record.get(key) for key in ("ProjectileRange", "ProjectileRadius", "ProjectileRadiusY")
            }
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0
                for value in required.values()
            ):
                raise ValueError(f"card {card_id} has incomplete native linear-sweep geometry")
            sweep_range = float(required["ProjectileRange"])
            # Native rolling projectiles use ProjectileRadius across the lane
            # and ProjectileRadiusY along their direction of travel.
            sweep_width = 2.0 * float(required["ProjectileRadius"])
            sweep_depth = 2.0 * float(required["ProjectileRadiusY"])

        taunt_radius = 0.0
        ability_shield: float | None = None
        ability_shield_keys: set[str] = set()
        inferred_ability_duration: float | None = None
        ability_definition_records = tuple(
            source.record
            for source in ability_projection.sources
            if source.node_id in {f"ABILITY.{ability.ability_id}" for ability in exact_abilities}
        )
        inferred_ability_cooldown = _first_numeric(ability_definition_records, "Cooldown")
        inferred_ability_cost = _first_numeric(ability_definition_records, "ManaCost")
        inferred_ability_cast_time = _first_numeric(ability_definition_records, "CastTime")
        for source in ability_projection.sources:
            record = source.record
            for key, value in record.items():
                lowered = str(key).lower()
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                stats_tags = record.get("StatsTags")
                tagged_taunt_radius = (
                    isinstance(stats_tags, Mapping)
                    and str(stats_tags.get("Radius", "")).lower() == "taunt_radius"
                    and lowered == "radius"
                )
                if ("taunt" in lowered and "radius" in lowered) or tagged_taunt_radius:
                    taunt_radius = max(taunt_radius, number)
                elif "shieldhitpoint" in lowered or lowered.endswith("maxshield"):
                    ability_shield_keys.add(str(key))
                    if ability_shield is None or number > ability_shield:
                        ability_shield = number
                elif lowered in {"validduration", "abilityduration"}:
                    inferred_ability_duration = max(inferred_ability_duration or 0.0, number)

        ability_shield_carrier = _matching_numeric_carrier(
            card_id, ability_projection.sources, ability_shield, *sorted(ability_shield_keys)
        )

        ability_duration = _reconcile_exact_number(
            card_id, "ability effect duration", inferred_ability_duration, exact_effect_duration
        )
        ability_cooldown = _reconcile_exact_number(
            card_id, "ability cooldown", inferred_ability_cooldown, exact_cooldown
        )
        ability_cost = _reconcile_exact_number(card_id, "ability cost", inferred_ability_cost, exact_cost)
        ability_cast_time = _reconcile_exact_number(
            card_id, "ability cast time", inferred_ability_cast_time, exact_cast_time
        )

        compound_fields = set(str(value) for value in spec.categorical_features.get("compound_declared", ()))
        not_applicable = set(str(value) for value in spec.categorical_features.get("not_applicable", ()))
        hitpoints_carrier = _matching_numeric_carrier(
            card_id, sources, None if "hitpoints" in compound_fields else spec.hitpoints, "Hitpoints"
        )
        sequence_damage = _attack_sequence_carriers(sources)
        periodic_damage = _periodic_damage(sources)
        damage_carrier = (
            _matching_numeric_carrier(card_id, sources, spec.damage, "Damage")
            if spec.damage is not None and "damage" not in compound_fields
            else None
        )
        if damage_carrier is None and sequence_damage and "damage" not in compound_fields:
            damage_carrier = sequence_damage[0]
        if damage_carrier is None and "damage" not in compound_fields and "damage" not in not_applicable:
            damage_carrier = next(
                (carrier for carrier in _numeric_carriers(sources, "Damage") if carrier.value > 0.0), None
            )
        if damage_carrier is None and periodic_damage is not None and "damage" not in compound_fields:
            damage_carrier = periodic_damage.per_hit
        level_damage = _project_carrier_value(spec, damage_carrier)
        level_damage_per_second = (
            _project_carrier_value(spec, periodic_damage.per_second)
            if periodic_damage is not None
            and damage_carrier is not None
            and damage_carrier.node_id == periodic_damage.per_hit.node_id
            and damage_carrier.field_name == periodic_damage.per_hit.field_name
            else None
        )
        special_damage_carrier = _typed_secondary_damage_carrier(spec, sources, damage_carrier, sequence_damage)
        shield_carrier = _matching_numeric_carrier(
            card_id,
            sources,
            None if "shield_hitpoints" in compound_fields else spec.shield_hitpoints,
            "ShieldHitpoints",
        )
        crown_ratio = _crown_tower_damage_ratio(spec, damage_carrier, sources, projected_damage=level_damage)
        result[card_id] = CardFeatureRow(
            level_hitpoints=_project_carrier_value(spec, hitpoints_carrier),
            level_damage=level_damage,
            level_damage_per_second=level_damage_per_second,
            building_damage_ratio=_building_damage_ratio(damage_carrier, sources),
            attack_sequence_hit_count=len(sequence_damage),
            level_shield=_project_carrier_value(spec, shield_carrier),
            crown_tower_damage_ratio=crown_ratio,
            stun_duration_ms=float(stun_duration or 0.0),
            chained_targets=float(chained_targets or 0.0),
            chained_radius_units=float(chained_radius or 0.0),
            special_damage=(_project_carrier_value(spec, special_damage_carrier) or 0.0),
            effect_duration_ms=_effect_duration_ms(spec, sources),
            taunt_radius_units=taunt_radius,
            ability_shield=_project_carrier_value(spec, ability_shield_carrier),
            ability_duration_ms=float(ability_duration or 0.0),
            ability_cooldown_ms=float(ability_cooldown or 0.0),
            ability_cost=float(ability_cost or 0.0),
            ability_cast_time_ms=float(ability_cast_time or 0.0),
            forward_linear_sweep=forward_linear_sweep,
            linear_sweep_range_units=sweep_range,
            linear_sweep_width_units=sweep_width,
            linear_sweep_depth_units=sweep_depth,
            # PerspectiveTransformV1 always puts own towers at low model y,
            # so forward toward the opponent is +y for both native owners.
            linear_sweep_direction_x=0.0,
            linear_sweep_direction_y=(1.0 if forward_linear_sweep else 0.0),
        )
    return result


def pack_static_features(spec: CardSpecV1, row: CardFeatureRow, *, width: int) -> list[float]:
    if width != len(CARD_STATIC_FEATURE_NAMES):
        raise ValueError("static card feature width does not match the V3 schema")
    compound = {str(value) for value in spec.categorical_features.get("compound_declared", ())}
    damage = 0.0 if "damage" in compound else row.level_damage or 0.0
    building_footprint = native_building_footprint(spec)
    circular_radius_tiles = (
        0.0 if row.forward_linear_sweep or "radius_tiles" in compound else float(spec.radius_tiles or 0.0)
    )
    hit_speed_seconds = (
        float(spec.hit_speed_ms) / 1000.0
        if "hit_speed_ms" not in compound and spec.hit_speed_ms is not None and spec.hit_speed_ms > 0
        else 0.0
    )
    damage_per_second = (
        float(row.level_damage_per_second)
        if row.level_damage_per_second is not None
        else (damage / max(hit_speed_seconds, 1e-3) if hit_speed_seconds else 0.0)
    )
    effect_duration_ms = 0.0 if "duration_ms" in compound else float(spec.duration_ms or row.effect_duration_ms or 0.0)
    result = [
        float(spec.elixir_cost or 0.0) / 10.0,
        (0.0 if "hitpoints" in compound else float(row.level_hitpoints or 0.0) / 5000.0),
        damage / 1000.0,
        damage_per_second / 1000.0,
        (0.0 if "hit_speed_ms" in compound else float(spec.hit_speed_ms or 0.0) / 5000.0),
        (0.0 if "range_tiles" in compound else float(spec.range_tiles or 0.0) / 10.0),
        (0.0 if "move_speed" in compound else float(spec.move_speed or 0.0) / 1000.0),
        (0.0 if "deploy_time_ms" in compound else float(spec.deploy_time_ms or 0.0) / 5000.0),
        float(spec.count or 0.0) / 10.0,
        circular_radius_tiles / 10.0,
        (0.0 if "projectile_speed" in compound else float(spec.projectile_speed or 0.0) / 10_000.0),
        (0.0 if "knockback" in compound else float(spec.knockback or 0.0) / 10_000.0),
        effect_duration_ms / 60_000.0,
        (0.0 if "shield_hitpoints" in compound else float(row.level_shield or 0.0) / 5000.0),
        row.crown_tower_damage_ratio,
        row.stun_duration_ms / 5000.0,
        row.chained_targets / 10.0,
        row.chained_radius_units / 10_000.0,
        (0.0 if "damage" in compound else float(row.special_damage or 0.0) / 1000.0),
        row.taunt_radius_units / 10_000.0,
        float(row.ability_shield or 0.0) / 5000.0,
        row.ability_duration_ms / 10_000.0,
        row.ability_cooldown_ms / 30_000.0,
        row.ability_cost / 10.0,
        row.ability_cast_time_ms / 5000.0,
        row.linear_sweep_range_units / 10_000.0,
        row.linear_sweep_width_units / 10_000.0,
        row.linear_sweep_depth_units / 10_000.0,
        (float(building_footprint.width_tiles) / 10.0 if building_footprint is not None else 0.0),
        row.building_damage_ratio,
        float(row.attack_sequence_hit_count) / 10.0,
    ]
    return result


def pack_mechanic_features(spec: CardSpecV1, row: CardFeatureRow, *, width: int) -> list[float]:
    if width != len(CARD_MECHANIC_FEATURE_NAMES):
        raise ValueError("mechanic card feature width does not match the V3 schema")
    kind_order = (CardKind.TROOP, CardKind.SPELL, CardKind.BUILDING, CardKind.HERO)
    categorical = spec.categorical_features
    compound = {str(value) for value in categorical.get("compound_declared", ())}
    circular_radius_tiles = (
        0.0 if row.forward_linear_sweep or "radius_tiles" in compound else float(spec.radius_tiles or 0.0)
    )
    effect_duration_ms = 0.0 if "duration_ms" in compound else float(spec.duration_ms or row.effect_duration_ms or 0.0)
    return [
        *(float(spec.kind == kind) for kind in kind_order),
        float(categorical.get("attacks_ground") is True),
        float(categorical.get("attacks_air") is True),
        float(categorical.get("target_only_buildings") is True),
        float(spec.evolution is not None),
        float(bool(spec.ability_ids)),
        float(effect_duration_ms > 0.0),
        float(circular_radius_tiles > 0.0),
        float("projectile_speed" not in compound and spec.projectile_speed is not None),
        float("knockback" not in compound and spec.knockback is not None),
        float("shield_hitpoints" not in compound and (row.level_shield is not None or row.ability_shield is not None)),
    ]
