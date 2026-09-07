from __future__ import annotations

import pytest

from native_runner.arena import (
    NATIVE_BUILDING_FOOTPRINTS_V15,
    PlacementAccuracy,
    PlacementRule,
    building_footprint_mask,
    card_placement_legal_world,
    native_building_footprint,
)
from native_runner.building_placement_profiles import (
    BLOCKED_BUILDING_IDS,
    BUILDING_PLACEMENT_PROFILES,
    EXACT_RECT_BUILDING_IDS,
    POLICY_READY_BUILDING_IDS,
)
from native_runner.building_semantics import BuildingPlacementKind


RULESET_ID = "b" * 64
EXPECTED_3X3 = frozenset(
    {
        27_000_000,
        27_000_001,
        27_000_002,
        27_000_003,
        27_000_004,
        27_000_005,
        27_000_007,
        27_000_008,
        27_000_009,
        27_000_012,
    }
)
EXCLUDED_ACTIVITY_BUILDINGS = frozenset(
    {
        27_000_011,
        27_000_014,
        27_000_015,
        27_000_016,
        27_000_017,
    }
)


def _card(
    card_id: int,
    *,
    collision_radius: int = 1_000,
) -> dict[str, object]:
    profile = BUILDING_PLACEMENT_PROFILES.get(card_id)
    placement = profile.placement_kind if profile is not None else None
    mask_key = (
        "full_arena"
        if placement
        in {
            BuildingPlacementKind.FULL_ARENA_PATHFIND_MORPH,
        }
        or card_id == 27_000_015
        else "own_deployment_zone"
    )
    return {
        "card_id": card_id,
        "kind": "building",
        "target_schema": {"placement_mask_key": mask_key},
        "attributes": {"CollisionRadius": collision_radius},
    }


def test_placement_profiles_cover_only_competitive_buildings() -> None:
    assert set(BUILDING_PLACEMENT_PROFILES) == {
        27_000_000 + index for index in range(18)
    } - EXCLUDED_ACTIVITY_BUILDINGS
    assert EXACT_RECT_BUILDING_IDS == EXPECTED_3X3 | {27_000_006}
    assert BLOCKED_BUILDING_IDS == set()
    assert POLICY_READY_BUILDING_IDS == set(BUILDING_PLACEMENT_PROFILES)


def test_native_3x3_and_2x2_profiles_remain_exact() -> None:
    assert set(NATIVE_BUILDING_FOOTPRINTS_V15) == EXACT_RECT_BUILDING_IDS
    for card_id in EXPECTED_3X3:
        profile = BUILDING_PLACEMENT_PROFILES[card_id]
        footprint = native_building_footprint(_card(card_id))
        assert profile.placement_kind is BuildingPlacementKind.NATIVE_RECT_3X3
        assert footprint is not None
        assert (footprint.width_tiles, footprint.height_tiles) == (3, 3)

        mask = building_footprint_mask(
            _card(card_id),
            0,
            ruleset_id=RULESET_ID,
        )
        assert mask.accuracy is PlacementAccuracy.EXACT_COARSE_CENTERS
        assert not mask.blocked
        assert (mask.footprint_width_tiles, mask.footprint_height_tiles) == (
            3,
            3,
        )
        assert "building_placement_profile:native_rect_3x3" in mask.reasons

    tesla = BUILDING_PLACEMENT_PROFILES[27_000_006]
    assert tesla.placement_kind is BuildingPlacementKind.NATIVE_RECT_2X2
    footprint = native_building_footprint(_card(27_000_006))
    assert footprint is not None
    assert (footprint.width_tiles, footprint.height_tiles) == (2, 2)
    mask = building_footprint_mask(
        _card(27_000_006),
        0,
        ruleset_id=RULESET_ID,
    )
    assert mask.accuracy is PlacementAccuracy.EXACT_COARSE_CENTERS
    assert mask.model_subcell_offset == (0.5, 0.5)
    assert "building_placement_profile:native_rect_2x2" in mask.reasons


@pytest.mark.parametrize("card_id", [*sorted(EXCLUDED_ACTIVITY_BUILDINGS), 27_999_999])
def test_activity_and_unknown_buildings_fail_closed_without_radius_guess(card_id: int) -> None:
    assert card_id not in BUILDING_PLACEMENT_PROFILES
    assert native_building_footprint(_card(card_id)) is None

    mask = building_footprint_mask(
        _card(card_id, collision_radius=9_999),
        0,
        ruleset_id=RULESET_ID,
    )
    assert mask.accuracy is PlacementAccuracy.BLOCKED
    assert mask.legal_cell_count == 0
    assert mask.collision_radius_units == 9_999
    assert "building_placement_profile:missing" in mask.reasons
    assert "building_placement_blocker:missing_explicit_building_placement_profile" in mask.reasons
    assert not card_placement_legal_world(
        _card(card_id, collision_radius=9_999),
        0,
        9_000,
        10_000,
    )


@pytest.mark.parametrize(
    ("card_id", "kind", "rule"),
    (
        (
            27_000_010,
            BuildingPlacementKind.MOVING_CHARACTER_DEPLOY,
            PlacementRule.OWN_TERRITORY,
        ),
        (
            27_000_013,
            BuildingPlacementKind.FULL_ARENA_PATHFIND_MORPH,
            PlacementRule.ANYWHERE,
        ),
    ),
)
def test_evidence_backed_special_deploys_are_conservative_without_footprints(
    card_id: int,
    kind: BuildingPlacementKind,
    rule: PlacementRule,
) -> None:
    profile = BUILDING_PLACEMENT_PROFILES[card_id]
    assert profile.placement_kind is kind
    assert profile.deploy_target_ready
    assert not profile.stationary_collision_rectangle
    assert profile.evidence
    assert native_building_footprint(_card(card_id)) is None

    mask = building_footprint_mask(
        _card(card_id),
        0,
        ruleset_id=RULESET_ID,
    )
    assert mask.rule is rule
    assert mask.accuracy is PlacementAccuracy.COARSE_CONSERVATIVE
    assert mask.legal_cell_count > 0
    assert mask.footprint_width_tiles is None
    assert mask.model_subcell_offset is None
    assert "evidence_backed_special_building_deploy" in mask.reasons


def test_drill_is_ready_but_entertainment_bottle_stays_blocked() -> None:
    drill = building_footprint_mask(
        _card(27_000_013),
        0,
        ruleset_id=RULESET_ID,
    )
    bottle = building_footprint_mask(
        _card(27_000_015),
        0,
        ruleset_id=RULESET_ID,
    )
    assert drill.rule is bottle.rule is PlacementRule.ANYWHERE
    assert not drill.blocked and drill.legal_cell_count > 0
    assert bottle.blocked and bottle.legal_cell_count == 0
