"""Evidence-backed native placement profiles for competitive buildings.

Combat ``CollisionRadius`` is not deployment geometry.  Only profiles backed
by native evidence expose a rectangular footprint.  Evidence-backed moving
and pathfinding deploys expose a target mask without pretending that their
transient root is a stationary collision rectangle.  Unknown or entertainment
mode-only deploys have no profile and remain blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .building_semantics import BuildingPlacementKind


BUILDING_PLACEMENT_PROFILE_VERSION = "building-placement-profile.v2"
NATIVE_PLACEMENT_BUILD = "15.535.13"

_EXACT_RECT_DIMS: Mapping[BuildingPlacementKind, tuple[int, int]] = MappingProxyType(
    {BuildingPlacementKind.NATIVE_RECT_3X3: (3, 3), BuildingPlacementKind.NATIVE_RECT_2X2: (2, 2)}
)
_EXHAUSTIVE_GRID_CARDS = frozenset({27_000_006, 27_000_008})
_BUILDINGS = {
    27_000_000: "Cannon",
    27_000_001: "GoblinHut",
    27_000_002: "Mortar",
    27_000_003: "InfernoTower",
    27_000_004: "BombTower",
    27_000_005: "BarbarianHut",
    27_000_006: "Tesla",
    27_000_007: "Elixir Collector",
    27_000_008: "Xbow",
    27_000_009: "Tombstone",
    27_000_010: "FirespiritHut",
    27_000_012: "GoblinCage",
    27_000_013: "GoblinDrill",
}
_SPECIAL_PLACEMENTS = {
    27_000_006: BuildingPlacementKind.NATIVE_RECT_2X2,
    27_000_010: BuildingPlacementKind.MOVING_CHARACTER_DEPLOY,
    27_000_013: BuildingPlacementKind.FULL_ARENA_PATHFIND_MORPH,
}
_EVIDENCE_OVERRIDES: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {
        27_000_007: (
            "mechanics-readiness:building_elixir_collector_runtime",
            "case-sha256:d475db2eddc2ab48fa15647bb29165497d9945bf36950ec6189f080669532654",
            "static-root:ElixirCollector:CollisionRadius=1000",
            "native-accepted-anchor:14500,10500",
        ),
        27_000_010: (
            "mechanics-readiness:building_furnace_runtime",
            "case-sha256:20ca6b09ce212a3b190fabafcc5a1f0caaae0e8fa1a65fdccf1376926a26d486",
            "static-root:Furnace_rework:Speed=60:CollisionRadius=600",
            "native-accepted-moving-character-deploy",
        ),
        27_000_013: (
            "mechanics-readiness:building_goblin_drill_runtime",
            "case-sha256:8e42ef2d7a4b4479f08455522a4c2a7ac9bf435ec1d0582536bff7efe68d8ae0",
            "static-root:GoblinDrillDig:SpawnPathfindMorph=GoblinDrill",
            "native-accepted-full-arena-pathfind-target",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class BuildingPlacementProfileV1:
    card_id: int
    name: str
    placement_kind: BuildingPlacementKind
    footprint_width_tiles: int | None
    footprint_height_tiles: int | None
    evidence: tuple[str, ...] = ()
    blocker: str | None = None
    version: str = field(default=BUILDING_PLACEMENT_PROFILE_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.card_id <= 0 or not self.name:
            raise ValueError("building placement profile identity is invalid")
        object.__setattr__(self, "placement_kind", BuildingPlacementKind(self.placement_kind))
        dimensions = (self.footprint_width_tiles, self.footprint_height_tiles)
        exact_dimensions = _EXACT_RECT_DIMS.get(self.placement_kind)
        if exact_dimensions is None:
            if dimensions != (None, None):
                raise ValueError("special building placement cannot claim a rectangle")
            if bool(self.evidence) == bool(self.blocker):
                raise ValueError("special building placement requires exactly one of evidence or a blocker")
        else:
            if dimensions != exact_dimensions:
                raise ValueError("native rectangle dimensions conflict with placement kind")
            if self.blocker or not self.evidence:
                raise ValueError("exact building placement requires evidence and no blocker")
        object.__setattr__(self, "evidence", tuple(sorted({str(item) for item in self.evidence if str(item)})))

    @property
    def exact_rectangle(self) -> bool:
        return self.placement_kind in _EXACT_RECT_DIMS

    @property
    def blocked(self) -> bool:
        return self.blocker is not None

    @property
    def deploy_target_ready(self) -> bool:
        """Whether the policy may expose placement targets for this card."""

        return not self.blocked

    @property
    def stationary_collision_rectangle(self) -> bool:
        """Whether live placement must reserve an occupied rectangle."""

        return self.exact_rectangle


def _profile(card_id: int) -> BuildingPlacementProfileV1:
    placement_kind = _SPECIAL_PLACEMENTS.get(card_id, BuildingPlacementKind.NATIVE_RECT_3X3)
    dimensions = _EXACT_RECT_DIMS.get(placement_kind, (None, None))
    evidence_scope = "owner_grid_exhaustive" if card_id in _EXHAUSTIVE_GRID_CARDS else "ordinary_boundary_collision"
    evidence = _EVIDENCE_OVERRIDES.get(
        card_id,
        (
            f"v{NATIVE_PLACEMENT_BUILD}_native_probe:{evidence_scope}",
            "native_anchor_lattice",
            "native_rectangle_collision",
        ),
    )
    return BuildingPlacementProfileV1(
        card_id=card_id,
        name=_BUILDINGS[card_id],
        placement_kind=placement_kind,
        footprint_width_tiles=dimensions[0],
        footprint_height_tiles=dimensions[1],
        evidence=evidence,
    )


BUILDING_PLACEMENT_PROFILES: Mapping[int, BuildingPlacementProfileV1] = MappingProxyType(
    {card_id: _profile(card_id) for card_id in _BUILDINGS}
)
EXACT_RECT_BUILDING_IDS = frozenset(
    card_id for card_id, profile in BUILDING_PLACEMENT_PROFILES.items() if profile.exact_rectangle
)
BLOCKED_BUILDING_IDS = frozenset(card_id for card_id, profile in BUILDING_PLACEMENT_PROFILES.items() if profile.blocked)
POLICY_READY_BUILDING_IDS = frozenset(
    card_id for card_id, profile in BUILDING_PLACEMENT_PROFILES.items() if profile.deploy_target_ready
)


def building_placement_profile(card_id: int) -> BuildingPlacementProfileV1 | None:
    """Return an explicit v15 profile; unknown future buildings fail closed."""

    return BUILDING_PLACEMENT_PROFILES.get(int(card_id))
