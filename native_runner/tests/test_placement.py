from __future__ import annotations

from typing import Any

import pytest

from native_runner.arena import (
    GRID_HEIGHT,
    GRID_WIDTH,
    OccupiedFootprintV1,
    PlacementAccuracy,
    PlacementMaskV1,
    PlacementRule,
    building_footprint_mask,
    card_placement_legal_world,
    card_placement_mask,
    native_building_footprint,
    raw_collision_radius,
    raw_collision_radius_units,
)
from native_runner.card_specs import build_card_catalog
from native_runner.paths import WORKSPACE_ROOT


RULESET_ID = "a" * 64


@pytest.fixture(scope="module")
def cards() -> dict[int, Any]:
    catalog = build_card_catalog(WORKSPACE_ROOT)
    return {spec.card_id: spec for spec in catalog.specs}


def test_building_masks_use_native_tile_footprints_not_collision_radius(
    cards: dict[int, Any],
) -> None:
    cannon = cards[27_000_000]
    goblin_hut = cards[27_000_001]
    tesla = cards[27_000_006]
    assert cannon.radius_tiles is None
    assert raw_collision_radius(cannon) == 600
    assert raw_collision_radius_units(cannon) == 600
    assert raw_collision_radius(goblin_hut) == 1000

    cannon_mask = building_footprint_mask(cannon, 0, ruleset_id=RULESET_ID)
    hut_mask = building_footprint_mask(goblin_hut, 0, ruleset_id=RULESET_ID)
    tesla_mask = building_footprint_mask(tesla, 0, ruleset_id=RULESET_ID)
    assert cannon_mask.accuracy == PlacementAccuracy.EXACT_COARSE_CENTERS
    assert not cannon_mask.conservative and not cannon_mask.blocked
    assert "native_probed_tile_footprint:3x3" in cannon_mask.reasons
    assert cannon_mask.collision_radius_units == 600
    assert hut_mask.collision_radius_units == 1000
    assert cannon_mask.footprint_width_tiles == 3
    assert cannon_mask.footprint_height_tiles == 3
    assert cannon_mask.model_subcell_offset == (0.0, 0.0)
    assert hut_mask.rows == cannon_mask.rows
    assert hut_mask.legal_cell_count == cannon_mask.legal_cell_count == 116
    assert hut_mask.mask_id != cannon_mask.mask_id
    assert tesla_mask.footprint_width_tiles == 2
    assert tesla_mask.footprint_height_tiles == 2
    assert tesla_mask.model_subcell_offset == (0.5, 0.5)
    assert tesla_mask.legal_cell_count == 167
    assert tesla_mask.rows != cannon_mask.rows


def test_unverified_special_building_footprint_fails_closed(
) -> None:
    unknown_radius = {
        "card_id": 27_000_015,
        "kind": "building",
        "target_schema": {"placement_mask_key": "full_arena"},
        "attributes": {"CollisionRadius": 9_999},
    }
    result = building_footprint_mask(unknown_radius, 0, ruleset_id=RULESET_ID)
    assert result.accuracy == PlacementAccuracy.BLOCKED
    assert result.blocked and not result.conservative
    assert result.legal_cell_count == 0
    assert any(
        reason.startswith("unverified_native_building_footprint:")
        for reason in result.reasons
    )


def test_special_spell_as_deploy_building_is_not_approximated(
    cards: dict[int, Any],
) -> None:
    drill = cards[27_000_013]
    result = building_footprint_mask(drill, 0, ruleset_id=RULESET_ID)
    assert result.rule == PlacementRule.ANYWHERE
    # The landed GoblinDrill has an exact combat CollisionRadius, but its
    # underground full-arena deployment still has no stationary tile footprint.
    assert result.collision_radius_units == 500
    assert result.accuracy == PlacementAccuracy.COARSE_CONSERVATIVE
    assert result.conservative and not result.blocked
    assert result.legal_cell_count == 454
    assert result.footprint_width_tiles is None
    assert result.footprint_height_tiles is None
    assert result.model_subcell_offset is None
    assert "evidence_backed_special_building_deploy" in result.reasons
    assert "no_stationary_collision_footprint" in result.reasons
    assert any(
        reason.startswith("building_placement_evidence:case-sha256:")
        for reason in result.reasons
    )


def test_card_mask_is_content_addressed_and_owner_symmetric(cards: dict[int, Any]) -> None:
    cannon = cards[27_000_000]
    bottom = card_placement_mask(cannon, 0, ruleset_id=RULESET_ID)
    repeat = card_placement_mask(cannon, 0, ruleset_id=RULESET_ID)
    top = card_placement_mask(cannon, 1, ruleset_id=RULESET_ID)
    pocket = card_placement_mask(
        cannon,
        0,
        ruleset_id=RULESET_ID,
        destroyed_enemy_princess_lanes=("left",),
    )
    assert bottom.mask_id == repeat.mask_id
    assert bottom.mask_id != top.mask_id != pocket.mask_id
    assert all(
        bottom.rows[y][x] == top.rows[GRID_HEIGHT - 1 - y][x]
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
    )
    assert pocket.accuracy == PlacementAccuracy.COARSE_CONSERVATIVE

    restored = PlacementMaskV1.from_mapping(bottom.to_dict())
    assert restored == bottom
    assert restored.content_hash() == bottom.mask_id
    tampered = bottom.to_dict()
    tampered["rows"][4][4] = not tampered["rows"][4][4]
    with pytest.raises(ValueError, match="integrity failure"):
        PlacementMaskV1.from_mapping(tampered)


def test_full_arena_non_spell_uses_terrain_guard_but_spell_does_not() -> None:
    miner = {
        "card_id": 26_000_032,
        "kind": "troop",
        "target_schema": {"placement_mask_key": "full_arena"},
        "attributes": {},
    }
    spell = {
        "card_id": 28_000_000,
        "kind": "spell",
        "target_schema": {"placement_mask_key": "full_arena"},
        "attributes": {},
    }
    miner_mask = card_placement_mask(miner, 0, ruleset_id=RULESET_ID)
    spell_mask = card_placement_mask(spell, 0, ruleset_id=RULESET_ID)
    assert miner_mask.legal_cell_count < GRID_WIDTH * GRID_HEIGHT
    assert miner_mask.accuracy == PlacementAccuracy.COARSE_CONSERVATIVE
    assert "full_arena_non_spell_uses_terrain_guard" in miner_mask.reasons
    assert spell_mask.legal_cell_count == GRID_WIDTH * GRID_HEIGHT
    assert spell_mask.accuracy == PlacementAccuracy.EXACT_COARSE_CENTERS


def test_spirit_empress_variants_avoid_live_tower_bodies(
    cards: dict[int, Any],
) -> None:
    spirit_empress = cards[28_000_025]
    assert spirit_empress.kind.value == "spell"
    assert spirit_empress.formation == "Troop"
    assert {
        option["AvailableManaTrigger"]
        for option in spirit_empress.attributes["Options"]
    } == {3_000, 6_000}

    bottom = card_placement_mask(
        spirit_empress,
        0,
        ruleset_id=RULESET_ID,
    )
    top = card_placement_mask(
        spirit_empress,
        1,
        ruleset_id=RULESET_ID,
    )

    assert "active_tower_rectangles_enforced" in bottom.reasons
    assert "active_tower_rectangles_enforced" in top.reasons
    # The rejected PPO action targeted top-owner cell (9, 27), inside the
    # live king-tower rectangle.  Exact edge/outside controls remain legal.
    assert not top.rows[27][9]
    assert top.rows[26][9]
    assert top.rows[27][11]
    assert not bottom.rows[4][9]
    assert bottom.rows[5][9]
    assert bottom.rows[4][11]
    assert all(
        bottom.rows[y][x] == top.rows[GRID_HEIGHT - 1 - y][x]
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
    )


def test_unknown_grid_rule_fails_closed() -> None:
    unknown = {
        "card_id": 26_999_999,
        "kind": "troop",
        "target_schema": {"placement_mask_key": "future_pointer_target"},
        "attributes": {},
    }
    result = card_placement_mask(unknown, 0, ruleset_id=RULESET_ID)
    assert result.blocked
    assert result.legal_cell_count == 0
    assert "unsupported_grid_placement_rule:unknown" in result.reasons


def test_evolution_uses_the_root_cards_native_deployment_lattice(
    cards: dict[int, Any],
) -> None:
    tesla = cards[27_000_006]
    assert raw_collision_radius(tesla, form="base") == 500
    assert raw_collision_radius(tesla, form="evolution") == 500
    base = card_placement_mask(tesla, 0, ruleset_id=RULESET_ID, form="base")
    evolution = card_placement_mask(
        tesla,
        0,
        ruleset_id=RULESET_ID,
        form="evolution",
    )
    assert native_building_footprint(tesla, form="base") == (
        native_building_footprint(tesla, form="evolution")
    )
    assert evolution.rows == base.rows
    assert evolution.model_subcell_offset == base.model_subcell_offset == (
        0.5,
        0.5,
    )
    assert evolution.mask_id != base.mask_id


def test_building_footprint_checks_global_terrain_not_enemy_territory(
    cards: dict[int, Any],
) -> None:
    xbow = cards[27_000_008]
    mask = card_placement_mask(xbow, 0, ruleset_id=RULESET_ID)
    # A 3x3 center on row 13 occupies terrain rows 12..14 and is legal.
    assert mask.rows[13][2]
    # Row 14 would occupy river row 15, so the complete footprint is illegal
    # even though its center remains inside owner 0's deployment territory.
    assert not mask.rows[14][2]


def test_destroyed_tower_expands_only_its_lane_and_keeps_buildings_off_river(
    cards: dict[int, Any],
) -> None:
    knight = cards[26_000_000]
    cannon = cards[27_000_000]
    opening_troop = card_placement_mask(knight, 0, ruleset_id=RULESET_ID)
    left_pocket_troop = card_placement_mask(
        knight,
        0,
        ruleset_id=RULESET_ID,
        destroyed_enemy_princess_lanes=("left",),
    )
    opening_building = card_placement_mask(cannon, 0, ruleset_id=RULESET_ID)
    left_pocket_building = card_placement_mask(
        cannon,
        0,
        ruleset_id=RULESET_ID,
        destroyed_enemy_princess_lanes=("left",),
    )

    assert not opening_troop.rows[17][2]
    assert left_pocket_troop.rows[17][2]
    assert left_pocket_troop.rows[17][8]
    assert not left_pocket_troop.rows[17][9]
    assert not left_pocket_troop.rows[17][13]

    assert not opening_building.rows[18][2]
    # A 3x3 Cannon centered on pocket row 17 would still occupy river row 16.
    assert not left_pocket_building.rows[17][2]
    assert left_pocket_building.rows[18][2]
    assert left_pocket_building.rows[18][8]
    assert not left_pocket_building.rows[18][9]
    assert not left_pocket_building.rows[18][13]


def test_even_building_requires_integer_tile_center_and_live_rectangles(
    cards: dict[int, Any],
) -> None:
    tesla = cards[27_000_006]
    mask = card_placement_mask(tesla, 0, ruleset_id=RULESET_ID)
    assert mask.rows[10][8]
    # Model cell (8, 10) plus (+0.5, +0.5) lands on native (9000, 11000).
    assert card_placement_legal_world(tesla, 0, 9000, 11000)
    assert not card_placement_legal_world(tesla, 0, 8500, 10500)

    occupied = (
        OccupiedFootprintV1(9500, 10500, 3, 3, "live-xbow"),
    )
    blocked = card_placement_mask(
        tesla,
        0,
        ruleset_id=RULESET_ID,
        occupied_building_footprints=occupied,
    )
    # Tesla center (9000, 11000) overlaps the 3x3 X-Bow rectangle.
    assert not blocked.rows[10][8]
    # Center (12000, 11000) is exactly 2.5 tiles away horizontally; native
    # edge contact is legal.
    assert blocked.rows[10][11]


def test_placement_mapping_rejects_missing_wire_version(cards: dict[int, Any]) -> None:
    placement = card_placement_mask(cards[27_000_000], 0, ruleset_id=RULESET_ID)
    payload = placement.to_dict()
    payload.pop("version")
    with pytest.raises(ValueError, match="unsupported"):
        PlacementMaskV1.from_mapping(payload)
