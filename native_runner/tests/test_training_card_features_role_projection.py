from __future__ import annotations

from functools import lru_cache

import pytest

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_card_catalog
from native_runner.training.card_features import (
    CARD_STATIC_FEATURE_NAMES,
    CardFeatureRow,
    build_card_feature_rows,
    pack_static_features,
)
from native_runner.training.v4.catalog import normal_mode_card_specs


@lru_cache(maxsize=1)
def _rows() -> tuple[dict[str, object], dict[str, CardFeatureRow]]:
    native = build_card_catalog()
    logic = build_static_card_logic_catalog(card_catalog=native)
    scoped = normal_mode_card_specs(
        native.by_id,
        require_frozen_baseline=True,
    )
    specs = {spec.name: spec for spec in scoped.values()}
    rows_by_id = build_card_feature_rows(
        scoped,
        static_logic=logic,
        ability_specs=native.abilities,
    )
    return specs, {
        name: rows_by_id[spec.card_id] for name, spec in specs.items()
    }


def test_base_damage_and_tower_ratio_share_one_concrete_carrier() -> None:
    _, rows = _rows()

    assert rows["Fireball"].level_damage == 688.0
    assert rows["Fireball"].crown_tower_damage_ratio == pytest.approx(0.25)

    # These closures also contain Hero/evolution side effects with tower
    # penalties.  They must not redefine the base attack's tower ratio.
    assert rows["Balloon"].level_damage == 640.0
    assert rows["Balloon"].crown_tower_damage_ratio == pytest.approx(1.0)
    assert rows["Valkyrie"].crown_tower_damage_ratio == pytest.approx(1.0)
    assert rows["RoyalGiant"].crown_tower_damage_ratio == pytest.approx(1.0)
    assert rows["RoyalGiant"].stun_duration_ms == 0.0


@pytest.mark.parametrize(
    (
        "name",
        "damage",
        "damage_per_second",
        "duration_ms",
        "tower_ratio",
    ),
    (
        ("Poison", 92.0, 92.0, 8_000.0, 0.23),
        ("Tornado", 84.0, 153.0, 1_050.0, 0.30),
        ("Earthquake", 81.0, 81.0, 3_000.0, 0.60),
        ("Vines", 153.0, 153.0, 2_000.0, 35.0 / 153.0),
    ),
)
def test_periodic_damage_uses_linked_buff_and_area_lifetime(
    name: str,
    damage: float,
    damage_per_second: float,
    duration_ms: float,
    tower_ratio: float,
) -> None:
    specs, rows = _rows()
    spec = specs[name]
    row = rows[name]

    assert row.level_damage == pytest.approx(damage)
    assert row.level_damage_per_second == pytest.approx(damage_per_second)
    assert row.effect_duration_ms == pytest.approx(duration_ms)
    assert row.crown_tower_damage_ratio == pytest.approx(tower_ratio)

    packed = pack_static_features(
        spec,
        row,
        width=len(CARD_STATIC_FEATURE_NAMES),
    )
    assert packed[2] == pytest.approx(damage / 1_000.0)
    assert packed[3] == pytest.approx(damage_per_second / 1_000.0)
    assert packed[12] == pytest.approx(duration_ms / 60_000.0)
    assert packed[14] == pytest.approx(tower_ratio)
    assert packed[29] == pytest.approx(row.building_damage_ratio)


def test_attack_sequence_supplies_berserker_primary_damage() -> None:
    specs, rows = _rows()
    row = rows["Berserker"]

    assert row.level_damage == 102.0
    assert row.special_damage == 0.0
    assert row.effect_duration_ms == 0.0
    packed = pack_static_features(
        specs["Berserker"],
        row,
        width=len(CARD_STATIC_FEATURE_NAMES),
    )
    assert packed[3] == pytest.approx(0.170)
    assert row.attack_sequence_hit_count == 3
    assert packed[30] == pytest.approx(0.3)


def test_earthquake_keeps_its_exact_building_multiplier() -> None:
    _, rows = _rows()

    assert rows["Earthquake"].building_damage_ratio == pytest.approx(3.5)
    assert rows["Poison"].building_damage_ratio == pytest.approx(1.0)


def test_secondary_damage_requires_an_explicit_distinct_channel() -> None:
    _, rows = _rows()

    assert rows["Golem"].special_damage == 225.0
    assert rows["IceWizard"].special_damage == 0.0
    assert rows["ElectroWizard"].special_damage == 0.0

    # Repeating the sole primary projectile is not a secondary semantic role.
    assert rows["Xbow"].special_damage == 0.0
    assert rows["Fireball"].special_damage == 0.0
    assert rows["Firecracker"].special_damage == 0.0


def test_ability_numbers_only_come_from_exact_ability_role() -> None:
    _, rows = _rows()

    assert rows["Knight"].ability_shield == 512.0
    assert rows["Knight"].ability_duration_ms == 4_000.0

    # Evolution shields and unrelated closure timers are not Ability values.
    assert rows["Wizard"].ability_shield is None
    assert rows["GoblinDemolisher"].ability_duration_ms == 0.0
    assert rows["Hunter"].ability_cooldown_ms == 0.0
    assert rows["GiantBuffer"].ability_cast_time_ms == 0.0
    assert rows["Berserker"].ability_cast_time_ms == 1_450.0
