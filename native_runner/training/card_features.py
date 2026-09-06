"""Frozen competitive feature facts and runtime feature packing.

The reviewed facts in data/competitive/model_catalogs.json replace general
source-graph traversal. Card and Ability hashes reject a changed release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..arena import native_building_footprint
from ..contracts import AbilitySpecV1, CardKind, CardSpecV1
from ..normal_form_evidence import NORMAL_MODE_HERO_FORM_TO_BASE_CARD

if TYPE_CHECKING:
    from ..card_logic import StaticCardLogicCatalogV1


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


def _compiled_model_data() -> Mapping[str, Any]:
    from ..competitive_data import load_competitive_data

    return load_competitive_data("model_catalogs")


def _compiled_fields(name: str) -> dict[str, Any]:
    """Restore immutable sequence fields from reviewed JSON facts."""

    def tuples(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(tuples(item) for item in value)
        if isinstance(value, Mapping):
            return {key: tuples(item) for key, item in value.items()}
        return value

    return tuples(_compiled_model_data()["catalogs"][name])


def _require_compiled_specs(card_specs: Mapping[int, CardSpecV1], *, complete: bool = False) -> None:
    from ..contracts import content_hash

    expected = _compiled_model_data()["card_spec_hashes"]
    if complete and {str(key) for key in card_specs} != set(expected):
        raise ValueError("compiled catalogs require the frozen 122-card competitive scope")
    for card_id, spec in card_specs.items():
        if expected.get(str(card_id)) != content_hash(spec.to_dict()):
            raise ValueError(f"card {card_id} differs from the compiled competitive release")


def _require_compiled_source(name: str, source: object | None) -> None:
    if source is None:
        return
    expected = _compiled_model_data()["input_catalog_ids"][name]
    if getattr(source, "catalog_id", None) != expected:
        raise ValueError(f"{name} differs from the compiled competitive release")


def _require_compiled_abilities(
    card_specs: Mapping[int, CardSpecV1], ability_specs: Mapping[str, AbilitySpecV1] | Sequence[AbilitySpecV1] | None
) -> None:
    from ..contracts import content_hash

    if ability_specs is None:
        return
    specs = tuple(ability_specs.values() if isinstance(ability_specs, Mapping) else ability_specs)
    expected = _compiled_model_data()["ability_spec_hashes"]
    for ability_id in {name for card in card_specs.values() for name in card.ability_ids}:
        matches = tuple(spec for spec in specs if spec.ability_id == ability_id)
        if len(matches) != 1:
            raise ValueError(f"Ability {ability_id!r} resolved to {len(matches)} exact AbilitySpec rows")
        if content_hash(matches[0].to_dict()) != expected.get(ability_id):
            raise ValueError(f"Ability {ability_id!r} differs from the compiled competitive release")


def build_card_feature_rows(
    card_specs: Mapping[int, CardSpecV1],
    *,
    static_logic: "StaticCardLogicCatalogV1 | None" = None,
    ability_specs: Mapping[str, AbilitySpecV1] | Sequence[AbilitySpecV1] | None = None,
) -> Mapping[int, CardFeatureRow]:
    """Load exact tournament-level feature facts for current competitive cards.

    Arbitrary source graphs and event cards are deliberately unsupported. The
    supplied evidence must match the release from which these rows were built.
    """
    _require_compiled_specs(card_specs)
    _require_compiled_source("card_logic", static_logic)
    _require_compiled_abilities(card_specs, ability_specs)
    rows = _compiled_model_data()["card_feature_rows"]
    return MappingProxyType({int(card_id): CardFeatureRow(**rows[str(card_id)]) for card_id in card_specs})
