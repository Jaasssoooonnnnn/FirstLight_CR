"""Native building placement kinds; serialized class path remains stable."""

from enum import Enum


class BuildingPlacementKind(str, Enum):
    """Native deployment/occupancy model required by a building card."""

    NATIVE_RECT_3X3 = "native_rect_3x3"
    NATIVE_RECT_2X2 = "native_rect_2x2"
    UNVERIFIED_STATIONARY = "unverified_stationary"
    MOVING_CHARACTER_DEPLOY = "moving_character_deploy"
    FULL_ARENA_PATHFIND_MORPH = "full_arena_pathfind_morph"
    FULL_ARENA_SPELL_AS_DEPLOY = "full_arena_spell_as_deploy"
    FULL_LANE_STAGED_DEPLOY = "full_lane_staged_deploy"
