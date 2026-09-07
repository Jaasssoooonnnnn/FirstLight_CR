from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from native_runner.card_logic import build_static_card_logic_catalog
from native_runner.card_specs import build_card_catalog
from native_runner.training.card_features import (
    CARD_FEATURE_SCHEMA_VERSION,
    CARD_MECHANIC_FEATURE_NAMES,
    CARD_STATIC_FEATURE_NAMES,
    build_card_feature_rows,
    level_11_value,
    pack_mechanic_features,
    pack_static_features,
)
from native_runner.training.v4.catalog import CardCatalogV1, NORMAL_MODE_POLICY_CARD_COUNT, NORMAL_MODE_SOURCE_CARD_COUNT, UNKNOWN_CARD_VOCAB_ID, normal_mode_card_specs


def test_all_exact_ability_cards_pack_catalog_cost_cooldown_and_cast() -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    rows = build_card_feature_rows(
        normal_mode_card_specs(catalog.by_id, require_frozen_baseline=True),
        static_logic=static_logic,
        ability_specs=catalog.abilities,
    )
    ability_cards = tuple(
        spec for spec in catalog.specs if spec.ability_ids
    )
    ability_index = {
        ability.ability_id: ability for ability in catalog.abilities
    }

    assert CARD_FEATURE_SCHEMA_VERSION == "semantic-card-feature-vector.v4"
    assert len(ability_cards) == 24
    assert len(ability_index) == len(catalog.abilities) == 24
    for spec in ability_cards:
        assert len(spec.ability_ids) == 1
        ability = ability_index[spec.ability_ids[0]]
        row = rows[spec.card_id]
        expected_cooldown = float(ability.cooldown_ms or 0)
        if ability.cooldown_ms is None:
            assert ability.charges == 1
        assert row.ability_cost == pytest.approx(ability.elixir_cost)
        assert row.ability_cooldown_ms == pytest.approx(
            expected_cooldown
        )
        assert row.ability_cast_time_ms == pytest.approx(
            ability.cast_time_ms
        )

        static = pack_static_features(
            spec,
            row,
            width=len(CARD_STATIC_FEATURE_NAMES),
        )
        assert static[22] == pytest.approx(
            expected_cooldown / 30_000.0
        )
        assert static[23] == pytest.approx(
            float(ability.elixir_cost) / 10.0
        )
        assert static[24] == pytest.approx(
            float(ability.cast_time_ms) / 5_000.0
        )

    assert rows[26_000_002].ability_cost == 1.0
    assert rows[26_000_014].ability_cooldown_ms == 0.0
    assert rows[26_000_039].ability_cooldown_ms == 1_000.0
    assert rows[26_000_103].ability_cooldown_ms == 3_000.0
    assert rows[26_000_017].ability_cast_time_ms == 950.0
    assert rows[26_000_072].ability_duration_ms == 3_500.0
    assert rows[26_000_077].ability_duration_ms == 4_000.0
    for card_id, duration in ((26_000_072, 3_500.0), (26_000_077, 4_000.0)):
        spec = catalog.by_id[card_id]
        row = rows[card_id]
        assert pack_static_features(
            spec,
            row,
            width=len(CARD_STATIC_FEATURE_NAMES),
        )[21] == pytest.approx(
            duration / 10_000.0
        )


@pytest.mark.parametrize(
    ("card_id", "carrier_node_id", "raw_damage", "level_11_damage"),
    (
        (27_000_008, "PROJECTILE.xbow_projectile", 23.0, 58.0),
        (26_000_058, "PROJECTILE.WallbreakerProjectile", 110.0, 281.0),
        (26_000_062, "PROJECTILE.EliteArcherArrow", 53.0, 135.0),
        (28_000_000, "PROJECTILE.FireballSpell", 269.0, 688.0),
        (28_000_005, "AEO.Freeze", 58.0, 148.0),
    ),
)
def test_level_11_damage_uses_the_numeric_carrier_rarity(
    card_id: int,
    carrier_node_id: str,
    raw_damage: float,
    level_11_damage: float,
) -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    spec = catalog.by_id[card_id]
    row = build_card_feature_rows(
        {card_id: spec},
        static_logic=static_logic,
    )[card_id]

    carrier = static_logic.nodes[carrier_node_id]["merged_record"]
    assert spec.categorical_features["rarity"] != carrier["Rarity"]
    assert carrier["Rarity"] == "Common"
    assert carrier["Damage"] == raw_damage
    assert spec.damage == raw_damage
    assert row.level_damage == level_11_damage

    assert len(
        pack_static_features(spec, row, width=len(CARD_STATIC_FEATURE_NAMES))
    ) == len(CARD_STATIC_FEATURE_NAMES)
    assert len(
        pack_mechanic_features(spec, row, width=len(CARD_MECHANIC_FEATURE_NAMES))
    ) == len(CARD_MECHANIC_FEATURE_NAMES)


def test_xbow_carrier_curve_projects_old_and_current_raw_stats() -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    spec = catalog.by_id[27_000_008]
    row = build_card_feature_rows(
        {spec.card_id: spec},
        static_logic=static_logic,
    )[spec.card_id]

    assert spec.categorical_features["rarity"] == "Epic"
    assert level_11_value(spec, 17, carrier_rarity="Common") == 43.0
    assert row.level_damage == 58.0
    # ``special_damage`` is a distinct secondary channel; it must not repeat
    # the X-Bow's primary attack merely because both use the same carrier.
    assert row.special_damage == 0.0
    assert row.level_hitpoints == 1_600.0


def test_level_11_features_reject_an_unreviewed_source_graph() -> None:
    spec = build_card_catalog().by_id[27_000_008]
    static_logic = SimpleNamespace(
        cards=(
            {
                "card_id": spec.card_id,
                "closure_node_ids": (
                    "PROJECTILE.common_damage",
                    "AEO.epic_damage",
                ),
                "level_scaling_candidates": (
                    {
                        "node_id": "PROJECTILE.common_damage",
                        "field_path": "Damage",
                        "base_value": 23,
                    },
                    {
                        "node_id": "AEO.epic_damage",
                        "field_path": "Damage",
                        "base_value": 23,
                    },
                ),
            },
        ),
        nodes={
            "PROJECTILE.common_damage": {
                "merged_record": {"Damage": 23, "Rarity": "Common"},
            },
            "AEO.epic_damage": {
                "merged_record": {"Damage": 23, "Rarity": "Epic"},
            },
        },
    )

    with pytest.raises(ValueError, match="card_logic differs from the compiled competitive release"):
        build_card_feature_rows(
            {spec.card_id: spec},
            static_logic=static_logic,
        )


def test_level_11_carrier_selection_ignores_percentage_damage_overrides() -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    spec = catalog.by_id[26_000_058]

    percent_override = static_logic.nodes[
        "PROJECTILE.WallbreakerProjectile_EV1"
    ]["merged_record"]["Damage"]
    assert percent_override == ("%", 100)
    row = build_card_feature_rows(
        {spec.card_id: spec},
        static_logic=static_logic,
    )[spec.card_id]
    assert row.level_damage == 281.0


def test_v4_global_catalog_matches_the_exact_feature_contract() -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    scoped_specs = normal_mode_card_specs(
        catalog.by_id,
        require_frozen_baseline=True,
    )
    expected = build_card_feature_rows(
        scoped_specs,
        static_logic=static_logic,
        ability_specs=catalog.abilities,
    )
    actual = CardCatalogV1.from_card_spec_catalog(
        catalog,
        static_logic=static_logic,
    )
    repeated = CardCatalogV1.from_card_spec_catalog(
        catalog,
        static_logic=static_logic,
    )

    assert len(catalog.specs) == NORMAL_MODE_SOURCE_CARD_COUNT
    assert len(scoped_specs) == NORMAL_MODE_POLICY_CARD_COUNT
    assert actual == CardCatalogV1.from_feature_rows(scoped_specs, expected)
    assert actual.raw_card_ids == tuple(sorted(scoped_specs))
    assert actual.catalog_id == repeated.catalog_id
    assert actual.vocab_size == NORMAL_MODE_POLICY_CARD_COUNT + 2
    excluded = sorted(set(catalog.by_id).difference(scoped_specs))
    assert len(excluded) == 30
    assert all(actual.vocab_id(card_id) == UNKNOWN_CARD_VOCAB_ID for card_id in excluded)
    with pytest.raises(ValueError, match="production baseline is the frozen 122-card"):
        actual.require_vocab_id(excluded[0], context="observed card")


def test_exact_ability_resolution_fails_closed_on_missing_or_duplicate() -> None:
    catalog = build_card_catalog()
    spec = catalog.by_id[26_000_002]
    ability = next(
        item
        for item in catalog.abilities
        if item.ability_id == spec.ability_ids[0]
    )

    with pytest.raises(ValueError, match="resolved to 0 exact AbilitySpec"):
        build_card_feature_rows(
            {spec.card_id: spec},
            ability_specs=(),
        )
    with pytest.raises(ValueError, match="resolved to 2 exact AbilitySpec"):
        build_card_feature_rows(
            {spec.card_id: spec},
            ability_specs=(ability, ability),
        )


def test_exact_ability_conflict_with_compiled_release_fails_closed() -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    spec = catalog.by_id[26_000_103]
    ability = next(
        item
        for item in catalog.abilities
        if item.ability_id == spec.ability_ids[0]
    )
    conflicting = replace(
        ability,
        elixir_cost=float(ability.elixir_cost) + 1.0,
    )

    with pytest.raises(ValueError, match="Ability .* differs from the compiled competitive release"):
        build_card_feature_rows(
            {spec.card_id: spec},
            static_logic=static_logic,
            ability_specs=(conflicting,),
        )


def test_missing_exact_cooldown_is_only_valid_for_one_charge() -> None:
    catalog = build_card_catalog()
    static_logic = build_static_card_logic_catalog(card_catalog=catalog)
    spec = catalog.by_id[26_000_000]
    ability = next(
        item
        for item in catalog.abilities
        if item.ability_id == spec.ability_ids[0]
    )
    assert ability.cooldown_ms is None
    assert ability.charges == 1

    invalid = replace(ability, charges=2)
    with pytest.raises(ValueError, match="Ability .* differs from the compiled competitive release"):
        build_card_feature_rows(
            {spec.card_id: spec},
            static_logic=static_logic,
            ability_specs=(invalid,),
        )


def test_compiled_features_reject_event_cards_and_changed_competitive_facts() -> None:
    native = build_card_catalog()
    scoped = normal_mode_card_specs(native.by_id, require_frozen_baseline=True)
    event_id = next(card_id for card_id in native.by_id if card_id not in scoped)
    with pytest.raises(ValueError, match="compiled competitive release"):
        build_card_feature_rows({event_id: native.by_id[event_id]})

    knight = scoped[26_000_000]
    changed = replace(knight, elixir_cost=float(knight.elixir_cost) + 1)
    with pytest.raises(ValueError, match="compiled competitive release"):
        build_card_feature_rows({changed.card_id: changed})
