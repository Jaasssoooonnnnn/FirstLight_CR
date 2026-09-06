"""Versioned arena coordinates and deployment masks for the v15 1v1 map.

The native replay executor trusts already-authorized commands.  It rejects
solid terrain, but it does *not* enforce the player's current deployment
territory.  Training code must therefore apply this mask before commands are
submitted; otherwise an agent can learn to deploy troops on the enemy side.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
import math
import re
from typing import Iterable, Mapping, Sequence

from .building_placement_profiles import BUILDING_PLACEMENT_PROFILES, building_placement_profile
from .contracts import CardKind, CardSpecV1, ContractMixin, content_hash


GRID_WIDTH = 18
GRID_HEIGHT = 32
CELL_UNITS = 1000
WORLD_WIDTH = GRID_WIDTH * CELL_UNITS
WORLD_HEIGHT = GRID_HEIGHT * CELL_UNITS
PLACEMENT_MASK_VERSION = "placement-mask.v4"
FOOTPRINT_ALGORITHM_VERSION = "native-tile-footprint.v4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FURNACE_CARD_ID = 27_000_010
_HEAL_SPIRIT_CARD_ID = 28_000_016


class PlacementRule(StrEnum):
    """Deployment-territory rule supplied by a card's public CardSpec."""

    OWN_TERRITORY = "own_territory"
    ANYWHERE = "anywhere"
    OWN_TERRITORY_AREA = "own_territory_area"
    NO_TARGET = "no_target"
    ENTITY_TARGET = "entity_target"
    UNKNOWN = "unknown"


class PlacementAccuracy(StrEnum):
    """How strongly a generated mask is supported by native evidence."""

    EXACT_COARSE_CENTERS = "exact_coarse_centers"
    COARSE_CONSERVATIVE = "coarse_conservative"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class NativeBuildingFootprintV1:
    """Native deployment lattice and occupied tile rectangle for one building.

    Clash Royale building placement is not derived from combat
    ``CollisionRadius``.  The v15 engine snaps conventional buildings to a
    tile-aligned square: odd-sized footprints use half-tile centers while
    even-sized footprints use integer-tile centers.
    """

    width_tiles: int
    height_tiles: int
    evidence: str = "v15.535.13-native-probe"

    def __post_init__(self) -> None:
        if self.width_tiles <= 0 or self.height_tiles <= 0:
            raise ValueError("building footprint dimensions must be positive")

    @property
    def model_subcell_offset(self) -> tuple[float, float]:
        """Offset from a policy cell center in the owner-normalized frame."""

        return (0.5 if self.width_tiles % 2 == 0 else 0.0, 0.5 if self.height_tiles % 2 == 0 else 0.0)


@dataclass(frozen=True, slots=True)
class OccupiedFootprintV1:
    """One live axis-aligned native obstacle, expressed in world units."""

    center_x_units: float
    center_y_units: float
    width_tiles: int
    height_tiles: int
    source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.center_x_units) or not math.isfinite(self.center_y_units):
            raise ValueError("occupied footprint center must be finite")
        if self.width_tiles <= 0 or self.height_tiles <= 0:
            raise ValueError("occupied footprint dimensions must be positive")
        if not self.source:
            raise ValueError("occupied footprint source must be non-empty")


# Placement geometry comes from the explicit 18-card placement partition.
# Only stationary rectangles are admitted here. Evidence-backed moving and
# pathfinding deploys intentionally remain absent because their live roots do
# not reserve a stationary building rectangle.
NATIVE_BUILDING_FOOTPRINTS_V15: Mapping[int, NativeBuildingFootprintV1] = {
    card_id: NativeBuildingFootprintV1(
        int(profile.footprint_width_tiles), int(profile.footprint_height_tiles), evidence=profile.evidence[0]
    )
    for card_id, profile in BUILDING_PLACEMENT_PROFILES.items()
    if profile.exact_rectangle
}


_STANDARD_TOWER_FOOTPRINTS = tuple(
    (owner, kind, OccupiedFootprintV1(x, y if owner == 0 else WORLD_HEIGHT - y, size, size, f"tower:{owner}:{kind}"))
    for owner in (0, 1)
    for kind, x, y, size in (
        ("king", 9000, 3000, 4),
        ("princess_left", 3500, 6500, 3),
        ("princess_right", 14500, 6500, 3),
    )
)


def native_building_footprint(
    card_spec: CardSpecV1 | Mapping[str, object], *, form: str = "base"
) -> NativeBuildingFootprintV1 | None:
    """Return only a native-probed deployment footprint.

    Base and evolution forms share the root spell's deployment lattice.  A
    missing result means the building has special or unverified placement and
    must not be approximated from combat radius.
    """

    normalized_form = str(form).lower()
    if normalized_form not in {"base", "evolution"}:
        raise ValueError("placement form must be base or evolution")
    card_id = int(_card_field(card_spec, "card_id") or 0)
    return NATIVE_BUILDING_FOOTPRINTS_V15.get(card_id)


def standard_active_tower_footprints(
    owner: int, destroyed_enemy_princess_lanes: Iterable[str] = ()
) -> tuple[OccupiedFootprintV1, ...]:
    """Return standard tower obstacles after known enemy tower destruction."""

    if owner not in (0, 1):
        raise ValueError("owner must be 0 or 1")
    lanes = frozenset(str(lane).lower() for lane in destroyed_enemy_princess_lanes)
    if any(lane not in {"left", "right"} for lane in lanes):
        raise ValueError("destroyed princess lanes must be left or right")
    enemy = 1 - owner
    return tuple(
        footprint
        for tower_owner, kind, footprint in _STANDARD_TOWER_FOOTPRINTS
        if not (
            tower_owner == enemy
            and ((kind == "princess_left" and "left" in lanes) or (kind == "princess_right" and "right" in lanes))
        )
    )


def cell_to_world(cell: Sequence[int]) -> tuple[int, int]:
    """Convert an ``(x, y)`` coarse cell to its native world-coordinate center."""

    if len(cell) != 2:
        raise ValueError("grid cell must contain x and y")
    x, y = int(cell[0]), int(cell[1])
    if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
        raise ValueError(f"grid cell {(x, y)} is outside the 18x32 arena")
    return x * CELL_UNITS + CELL_UNITS // 2, y * CELL_UNITS + CELL_UNITS // 2


def world_to_cell(x: int, y: int, *, clamp: bool = False) -> tuple[int, int]:
    """Convert native world coordinates to an ``(x, y)`` coarse cell."""

    gx, gy = int(x) // CELL_UNITS, int(y) // CELL_UNITS
    if clamp:
        return min(max(gx, 0), GRID_WIDTH - 1), min(max(gy, 0), GRID_HEIGHT - 1)
    if not (0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT):
        raise ValueError(f"world position {(x, y)} is outside the arena")
    return gx, gy


def _bottom_walkable(x: int, y: int) -> bool:
    """Native-probed coarse-cell terrain for rows 0..16 of tilemap.csv."""

    if y == 0:
        return 5 <= x <= 11
    if 1 <= y <= 3:
        return x <= 6 or x >= 10
    if 4 <= y <= 13:
        return True
    if y == 14:
        return 1 <= x <= 16
    if 15 <= y <= 16:
        return x in (2, 3, 13, 14)
    return False


def terrain_walkable(x: int, y: int) -> bool:
    """Whether a troop-sized placement can occupy a coarse cell.

    This mask was exhaustively checked against all 576 coarse cell centers on
    the exact v15.535.13 native engine.  It includes tower footprints, arena
    bounds, river, and the two bridges.  Larger building footprints remain a
    card-specific refinement and are marked as such in mask metadata.
    """

    if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
        return False
    return _bottom_walkable(x, y) if y <= 16 else _bottom_walkable(x, 31 - y)


def building_terrain_walkable(x: int, y: int) -> bool:
    """Native tile terrain accepted by conventional building footprints."""

    return y not in {0, 15, 16, 31} and terrain_walkable(x, y)


def _rectangles_overlap(left: OccupiedFootprintV1, right: OccupiedFootprintV1) -> bool:
    """Native placement allows edge contact but rejects positive overlap."""

    return (
        abs(left.center_x_units - right.center_x_units) * 2.0 < (left.width_tiles + right.width_tiles) * CELL_UNITS
        and abs(left.center_y_units - right.center_y_units) * 2.0
        < (left.height_tiles + right.height_tiles) * CELL_UNITS
    )


def building_footprints_overlap(left: OccupiedFootprintV1, right: OccupiedFootprintV1) -> bool:
    """Public spelling for exact native rectangular overlap."""

    return _rectangles_overlap(left, right)


def _center_matches_native_lattice(center_units: float, size_tiles: int) -> bool:
    expected_remainder = 0.0 if size_tiles % 2 == 0 else CELL_UNITS / 2.0
    remainder = center_units % CELL_UNITS
    return math.isclose(remainder, expected_remainder, abs_tol=1e-6)


def _building_footprint_cells(footprint: OccupiedFootprintV1) -> tuple[range, range] | None:
    half_width = footprint.width_tiles * CELL_UNITS / 2.0
    half_height = footprint.height_tiles * CELL_UNITS / 2.0
    left = footprint.center_x_units - half_width
    right = footprint.center_x_units + half_width
    bottom = footprint.center_y_units - half_height
    top = footprint.center_y_units + half_height
    if left < 0 or bottom < 0 or right > WORLD_WIDTH or top > WORLD_HEIGHT:
        return None
    minimum_x = int(round(left / CELL_UNITS))
    maximum_x = int(round(right / CELL_UNITS))
    minimum_y = int(round(bottom / CELL_UNITS))
    maximum_y = int(round(top / CELL_UNITS))
    if not all(
        math.isclose(value, round(value), abs_tol=1e-6)
        for value in (left / CELL_UNITS, right / CELL_UNITS, bottom / CELL_UNITS, top / CELL_UNITS)
    ):
        return None
    return range(minimum_x, maximum_x), range(minimum_y, maximum_y)


def _exact_building_legal_world(
    *,
    owner: int,
    rule: PlacementRule,
    world_x: float,
    world_y: float,
    footprint: NativeBuildingFootprintV1,
    destroyed_enemy_princess_lanes: Iterable[str],
    active_tower_footprints: Sequence[OccupiedFootprintV1] | None,
    occupied_building_footprints: Sequence[OccupiedFootprintV1],
) -> bool:
    if not _center_matches_native_lattice(world_x, footprint.width_tiles) or not _center_matches_native_lattice(
        world_y, footprint.height_tiles
    ):
        return False
    cell_x, cell_y = int(world_x // CELL_UNITS), int(world_y // CELL_UNITS)
    if not _center_satisfies_rule(owner, rule, cell_x, cell_y, destroyed_enemy_princess_lanes):
        return False
    candidate = OccupiedFootprintV1(world_x, world_y, footprint.width_tiles, footprint.height_tiles, "candidate")
    cells = _building_footprint_cells(candidate)
    if cells is None:
        return False
    x_cells, y_cells = cells
    if any(not building_terrain_walkable(x, y) for y in y_cells for x in x_cells):
        return False
    towers = (
        standard_active_tower_footprints(owner, destroyed_enemy_princess_lanes)
        if active_tower_footprints is None
        else tuple(active_tower_footprints)
    )
    return not any(
        _rectangles_overlap(candidate, obstacle) for obstacle in (*towers, *tuple(occupied_building_footprints))
    )


def _base_territory(owner: int, y: int) -> bool:
    if owner == 0:
        return y <= 16
    if owner == 1:
        return y >= 15
    raise ValueError("owner must be 0 or 1")


def _pocket_territory(owner: int, x: int, y: int, destroyed_enemy_princess_lanes: Iterable[str]) -> bool:
    """Return the post-princess-tower pocket extension.

    Real RoyaleAPI command traces contain accepted deployments through the
    center seam of the destroyed tower's half-arena: columns 0..8 for the left
    pocket and 9..17 for the right pocket.  Keeping the older two-column inset
    (0..6 / 11..17) made valid expert targets impossible for the policy even
    when the native engine executed them and terminal fidelity was exact.
    """

    lanes = frozenset(str(lane).lower() for lane in destroyed_enemy_princess_lanes)
    if owner == 0:
        in_depth = 17 <= y <= 22
    elif owner == 1:
        in_depth = 9 <= y <= 14
    else:
        raise ValueError("owner must be 0 or 1")
    if not in_depth:
        return False
    return ("left" in lanes and x <= 8) or ("right" in lanes and x >= 9)


def deployment_mask(
    owner: int, rule: PlacementRule | str, *, destroyed_enemy_princess_lanes: Iterable[str] = ()
) -> tuple[tuple[bool, ...], ...]:
    """Build a row-major ``[32][18]`` target mask."""

    placement = PlacementRule(rule)
    return tuple(
        tuple(
            _center_satisfies_rule(owner, placement, x, y, destroyed_enemy_princess_lanes)
            and (placement != PlacementRule.OWN_TERRITORY or terrain_walkable(x, y))
            for x in range(GRID_WIDTH)
        )
        for y in range(GRID_HEIGHT)
    )


def _validate_rows(rows: Sequence[Sequence[bool]]) -> tuple[tuple[bool, ...], ...]:
    materialized = tuple(tuple(row) for row in rows)
    if len(materialized) != GRID_HEIGHT or any(len(row) != GRID_WIDTH for row in materialized):
        raise ValueError("placement mask must be row-major [32][18]")
    if any(not isinstance(cell, bool) for row in materialized for cell in row):
        raise ValueError("placement mask cells must be booleans")
    return materialized


def raw_collision_radius(card_spec: CardSpecV1 | Mapping[str, object], *, form: str = "base") -> float | int | None:
    """Return the public combat ``CollisionRadius`` for audit compatibility.

    The generic ``CardSpecV1.radius_tiles`` describes spell/attack radii and is
    not a building footprint. This value is retained in placement artifacts
    for diagnostics only; native deployment masks use the probed tile
    footprint table. Conflicting raw values are rejected instead of guessed.
    """

    attributes: Mapping[str, object]
    wrapped_spec = isinstance(card_spec, CardSpecV1) or "attributes" in card_spec
    if isinstance(card_spec, CardSpecV1):
        attributes = card_spec.attributes
    else:
        raw_attributes = card_spec.get("attributes", card_spec)
        attributes = raw_attributes if isinstance(raw_attributes, Mapping) else {}
    candidates: list[float | int] = []
    normalized_form = str(form).lower()
    if normalized_form not in {"base", "evolution"}:
        raise ValueError("placement form must be base or evolution")
    records: list[Mapping[str, object]] = [] if wrapped_spec else [attributes]
    keys = (
        ("resolved_summoned_form",)
        if normalized_form == "base"
        else ("resolved_evolution_form", "resolved_evolution_unit")
    )
    if normalized_form == "base":
        records.append(attributes)
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, Mapping):
            records.append(value)
    for record in records:
        value = record.get("CollisionRadius")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("raw CollisionRadius must be finite and non-negative")
            candidates.append(int(number) if number.is_integer() else number)
    if not candidates:
        return None
    if any(float(value) != float(candidates[0]) for value in candidates[1:]):
        raise ValueError(f"conflicting raw CollisionRadius values: {candidates}")
    return candidates[0]


# Public name makes the unit boundary explicit while retaining the shorter
# compatibility spelling used by the initial placement-contract prototype.
raw_collision_radius_units = raw_collision_radius


def _card_field(card_spec: CardSpecV1 | Mapping[str, object], name: str) -> object:
    if isinstance(card_spec, CardSpecV1):
        return getattr(card_spec, name)
    return card_spec.get(name)


def _card_kind(card_spec: CardSpecV1 | Mapping[str, object]) -> CardKind:
    kind = _card_field(card_spec, "kind")
    try:
        return kind if isinstance(kind, CardKind) else CardKind(str(kind))
    except ValueError:
        return CardKind.UNKNOWN


def uses_unit_deployment_center(card_spec: CardSpecV1 | Mapping[str, object]) -> bool:
    """Whether placement creates a moving unit at the requested center."""

    kind = _card_kind(card_spec)
    formation = _card_field(card_spec, "formation")
    card_id = int(_card_field(card_spec, "card_id") or 0)
    return (
        kind in {CardKind.TROOP, CardKind.HERO}
        # Spirit Empress is a troop in gameplay, but its 3/6-elixir selector
        # is stored as a spell-variant root.  The canonical formation keeps
        # that deployment semantic explicit without card-ID special casing.
        or (kind == CardKind.SPELL and formation == "Troop")
        or card_id in {_FURNACE_CARD_ID, _HEAL_SPIRIT_CARD_ID}
    )


def uses_entity_deployment_center(card_spec: CardSpecV1 | Mapping[str, object]) -> bool:
    """Whether a grid action materializes a collidable entity at its center.

    Historical card-table categories are not sufficient here: Furnace is a
    moving troop in the current ruleset, Heal Spirit is encoded as a deploy
    spell, Spirit Empress is encoded as a spell-variant troop selector, and
    Goblin Drill is a pathfinding building deploy.  All of them, plus ordinary
    troops, heroes, and buildings, must avoid live tower bodies.
    """

    return _card_kind(card_spec) == CardKind.BUILDING or uses_unit_deployment_center(card_spec)


def _placement_rule_for_card(card_spec: CardSpecV1 | Mapping[str, object]) -> PlacementRule:
    if isinstance(card_spec, CardSpecV1):
        key = card_spec.target_schema.placement_mask_key
    else:
        target_schema = card_spec.get("target_schema", {})
        key = target_schema.get("placement_mask_key", "unknown") if isinstance(target_schema, Mapping) else "unknown"
    return {
        "full_arena": PlacementRule.ANYWHERE,
        "own_deployment_zone": PlacementRule.OWN_TERRITORY,
        "own_territory": PlacementRule.OWN_TERRITORY,
        "own_territory_area": PlacementRule.OWN_TERRITORY_AREA,
        "no_target": PlacementRule.NO_TARGET,
        "entity_target": PlacementRule.ENTITY_TARGET,
    }.get(str(key), PlacementRule.UNKNOWN)


@dataclass(frozen=True, slots=True)
class PlacementMaskV1(ContractMixin):
    """Content-addressed card-conditioned native placement rule."""

    ruleset_id: str
    card_id: int
    owner: int
    rule: PlacementRule
    rows: tuple[tuple[bool, ...], ...]
    card_spec_hash: str
    collision_radius_units: float | int | None
    form: str
    accuracy: PlacementAccuracy
    footprint_width_tiles: int | None = None
    footprint_height_tiles: int | None = None
    model_subcell_offset: tuple[float, float] | None = None
    reasons: tuple[str, ...] = ()
    destroyed_enemy_princess_lanes: tuple[str, ...] = ()
    algorithm: str = FOOTPRINT_ALGORITHM_VERSION
    mask_id: str = ""
    version: str = field(default=PLACEMENT_MASK_VERSION, init=False)
    VERSION = PLACEMENT_MASK_VERSION

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.ruleset_id):
            raise ValueError("ruleset_id must be a lowercase SHA-256 digest")
        if not _SHA256.fullmatch(self.card_spec_hash):
            raise ValueError("card_spec_hash must be a lowercase SHA-256 digest")
        if self.card_id <= 0 or self.owner not in (0, 1):
            raise ValueError("invalid card placement identity")
        object.__setattr__(self, "rule", PlacementRule(self.rule))
        object.__setattr__(self, "accuracy", PlacementAccuracy(self.accuracy))
        if self.algorithm != FOOTPRINT_ALGORITHM_VERSION:
            raise ValueError(f"unsupported placement algorithm: {self.algorithm}")
        if self.collision_radius_units is not None:
            radius = float(self.collision_radius_units)
            if not math.isfinite(radius) or radius < 0:
                raise ValueError("collision_radius_units must be finite/non-negative")
        dimensions = (self.footprint_width_tiles, self.footprint_height_tiles)
        if (dimensions[0] is None) != (dimensions[1] is None):
            raise ValueError("building footprint dimensions must appear together")
        if any(value is not None and value <= 0 for value in dimensions):
            raise ValueError("building footprint dimensions must be positive")
        if self.model_subcell_offset is not None:
            offset = tuple(float(value) for value in self.model_subcell_offset)
            if len(offset) != 2 or any(not math.isfinite(value) or value not in {0.0, 0.5} for value in offset):
                raise ValueError("model_subcell_offset must contain native lattice offsets")
            object.__setattr__(self, "model_subcell_offset", offset)
        if dimensions[0] is None and self.model_subcell_offset is not None:
            raise ValueError("non-building masks cannot carry a subcell offset")
        normalized_form = str(self.form).lower()
        if normalized_form not in {"base", "evolution"}:
            raise ValueError("placement form must be base or evolution")
        object.__setattr__(self, "form", normalized_form)
        object.__setattr__(self, "rows", _validate_rows(self.rows))
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        lanes = tuple(sorted(set(str(item).lower() for item in self.destroyed_enemy_princess_lanes)))
        if any(item not in {"left", "right"} for item in lanes):
            raise ValueError("destroyed princess lanes must be left or right")
        object.__setattr__(self, "destroyed_enemy_princess_lanes", lanes)
        if self.accuracy == PlacementAccuracy.BLOCKED and any(cell for row in self.rows for cell in row):
            raise ValueError("blocked placement masks cannot expose legal cells")
        expected = self.calculate_mask_id()
        if self.mask_id and self.mask_id != expected:
            raise ValueError(f"placement mask integrity failure: expected {expected}, got {self.mask_id}")
        object.__setattr__(self, "mask_id", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "ruleset_id": self.ruleset_id,
            "card_id": self.card_id,
            "owner": self.owner,
            "rule": self.rule.value,
            "rows": self.rows,
            "card_spec_hash": self.card_spec_hash,
            "collision_radius_units": self.collision_radius_units,
            "form": self.form,
            "accuracy": self.accuracy.value,
            "footprint_width_tiles": self.footprint_width_tiles,
            "footprint_height_tiles": self.footprint_height_tiles,
            "model_subcell_offset": self.model_subcell_offset,
            "reasons": self.reasons,
            "destroyed_enemy_princess_lanes": self.destroyed_enemy_princess_lanes,
            "algorithm": self.algorithm,
        }

    def calculate_mask_id(self) -> str:
        return content_hash(self.identity_dict())

    def content_hash(self) -> str:
        """The artifact's public content address, excluding its stored ID."""

        return self.mask_id

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PlacementMaskV1":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown placement mask fields: {unknown}")
        if value.get("version") != PLACEMENT_MASK_VERSION:
            raise ValueError(f"unsupported placement mask: {value.get('version')}")
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError("placement mask rows must be a sequence")
        radius = value.get("collision_radius_units")
        if radius is not None and (not isinstance(radius, (int, float)) or isinstance(radius, bool)):
            raise ValueError("collision_radius_units must be numeric or null")
        return cls(
            ruleset_id=str(value["ruleset_id"]),
            card_id=int(value["card_id"]),
            owner=int(value["owner"]),
            rule=PlacementRule(str(value["rule"])),
            rows=_validate_rows(raw_rows),
            card_spec_hash=str(value["card_spec_hash"]),
            collision_radius_units=radius,
            form=str(value["form"]),
            accuracy=PlacementAccuracy(str(value["accuracy"])),
            footprint_width_tiles=(
                int(value["footprint_width_tiles"]) if value.get("footprint_width_tiles") is not None else None
            ),
            footprint_height_tiles=(
                int(value["footprint_height_tiles"]) if value.get("footprint_height_tiles") is not None else None
            ),
            model_subcell_offset=(
                tuple(float(item) for item in value["model_subcell_offset"])
                if value.get("model_subcell_offset") is not None
                else None
            ),
            reasons=tuple(str(item) for item in value.get("reasons", ())),
            destroyed_enemy_princess_lanes=tuple(str(item) for item in value.get("destroyed_enemy_princess_lanes", ())),
            algorithm=str(value.get("algorithm", FOOTPRINT_ALGORITHM_VERSION)),
            mask_id=str(value.get("mask_id", "")),
        )

    @property
    def legal_cell_count(self) -> int:
        return sum(sum(row) for row in self.rows)

    @property
    def conservative(self) -> bool:
        return self.accuracy == PlacementAccuracy.COARSE_CONSERVATIVE

    @property
    def blocked(self) -> bool:
        return self.accuracy == PlacementAccuracy.BLOCKED


def _center_satisfies_rule(owner: int, rule: PlacementRule, cell_x: int, cell_y: int, lanes: Iterable[str]) -> bool:
    if rule == PlacementRule.ANYWHERE:
        return True
    if rule in {PlacementRule.OWN_TERRITORY, PlacementRule.OWN_TERRITORY_AREA}:
        return _base_territory(owner, cell_y) or _pocket_territory(owner, cell_x, cell_y, lanes)
    return False


def card_placement_legal_world(
    card_spec: CardSpecV1 | Mapping[str, object],
    owner: int,
    position_or_x: Sequence[float | int] | float | int,
    y: float | int | None = None,
    *,
    form: str = "base",
    destroyed_enemy_princess_lanes: Iterable[str] = (),
    active_tower_footprints: Sequence[OccupiedFootprintV1] | None = None,
    occupied_building_footprints: Sequence[OccupiedFootprintV1] = (),
) -> bool:
    """Validate a final native world coordinate for a card.

    Conventional buildings use their native tile rectangle and anchor lattice;
    combat ``CollisionRadius`` is deliberately not used as deployment size.
    Evidence-backed moving/pathfinding building cards use their public target
    rule and terrain guard without inventing a stationary footprint. Unknown
    special building producers fail closed.

    ``position_or_x`` accepts either ``(x, y)`` or an x value followed by the
    separate ``y`` argument so environment adapters can use it directly.
    """

    if owner not in (0, 1):
        raise ValueError("owner must be 0 or 1")
    if y is None:
        if (
            not isinstance(position_or_x, Sequence)
            or isinstance(position_or_x, (str, bytes))
            or len(position_or_x) != 2
        ):
            raise ValueError("world position must contain x and y")
        world_x, world_y = float(position_or_x[0]), float(position_or_x[1])
    else:
        if isinstance(position_or_x, Sequence):
            raise ValueError("pass a coordinate pair or separate x/y, not both")
        world_x, world_y = float(position_or_x), float(y)
    if not math.isfinite(world_x) or not math.isfinite(world_y):
        raise ValueError("world position must be finite")
    if not (0 <= world_x < WORLD_WIDTH and 0 <= world_y < WORLD_HEIGHT):
        return False

    kind = _card_kind(card_spec)
    rule = _placement_rule_for_card(card_spec)
    cell_x, cell_y = int(world_x // CELL_UNITS), int(world_y // CELL_UNITS)
    lanes = tuple(sorted(set(str(item).lower() for item in destroyed_enemy_princess_lanes)))
    if any(item not in {"left", "right"} for item in lanes):
        raise ValueError("destroyed princess lanes must be left or right")
    if kind == CardKind.BUILDING:
        placement_profile = building_placement_profile(int(_card_field(card_spec, "card_id") or 0))
        if placement_profile is None or placement_profile.blocked:
            return False
        if placement_profile.stationary_collision_rectangle:
            footprint = native_building_footprint(card_spec, form=form)
            if footprint is None:
                return False
            return _exact_building_legal_world(
                owner=owner,
                rule=rule,
                world_x=world_x,
                world_y=world_y,
                footprint=footprint,
                destroyed_enemy_princess_lanes=lanes,
                active_tower_footprints=active_tower_footprints,
                occupied_building_footprints=occupied_building_footprints,
            )
    if uses_entity_deployment_center(card_spec):
        towers = (
            standard_active_tower_footprints(owner, lanes)
            if active_tower_footprints is None
            else tuple(active_tower_footprints)
        )
        if any(
            abs(world_x - tower.center_x_units) * 2.0 < tower.width_tiles * CELL_UNITS
            and abs(world_y - tower.center_y_units) * 2.0 < tower.height_tiles * CELL_UNITS
            for tower in towers
        ):
            return False
    if not _center_satisfies_rule(owner, rule, cell_x, cell_y, lanes):
        return False
    if rule == PlacementRule.ANYWHERE and kind == CardKind.SPELL:
        return True
    if rule == PlacementRule.OWN_TERRITORY_AREA:
        return True
    # Full-arena non-spells (for example Miner) and ordinary troop placements
    # still require the coarse native terrain guard.
    return terrain_walkable(cell_x, cell_y)


def card_placement_mask(
    card_spec: CardSpecV1 | Mapping[str, object],
    owner: int,
    *,
    ruleset_id: str,
    form: str = "base",
    destroyed_enemy_princess_lanes: Iterable[str] = (),
    active_tower_footprints: Sequence[OccupiedFootprintV1] | None = None,
    occupied_building_footprints: Sequence[OccupiedFootprintV1] = (),
) -> PlacementMaskV1:
    """Build a content-addressed, card-specific 18x32 placement mask.

    Native-probed conventional buildings use their exact tile rectangle,
    anchor lattice, live tower rectangles and live building rectangles.
    Evidence-backed moving/pathfinding deploys use a conservative target mask
    without a stationary footprint. Unknown special producers remain blocked.
    """

    if owner not in (0, 1):
        raise ValueError("owner must be 0 or 1")
    card_id = int(_card_field(card_spec, "card_id") or 0)
    raw_kind = _card_field(card_spec, "kind")
    kind = raw_kind if isinstance(raw_kind, CardKind) else CardKind(str(raw_kind))
    rule = _placement_rule_for_card(card_spec)
    lanes = tuple(sorted(set(str(item).lower() for item in destroyed_enemy_princess_lanes)))
    if isinstance(card_spec, CardSpecV1):
        card_spec_hash = card_spec.content_hash()
    else:
        card_spec_hash = content_hash(card_spec)

    reasons: list[str] = []
    accuracy = PlacementAccuracy.EXACT_COARSE_CENTERS
    collision_radius: float | int | None = None
    footprint = native_building_footprint(card_spec, form=form) if kind == CardKind.BUILDING else None
    building_profile = building_placement_profile(card_id) if kind == CardKind.BUILDING else None
    model_subcell_offset = footprint.model_subcell_offset if footprint is not None else None
    native_offset_sign = 1.0 if owner == 0 else -1.0

    def candidate_world(x: int, y: int) -> tuple[float, float]:
        world_x, world_y = cell_to_world((x, y))
        if model_subcell_offset is not None:
            world_x += int(round(model_subcell_offset[0] * native_offset_sign * CELL_UNITS))
            world_y += int(round(model_subcell_offset[1] * native_offset_sign * CELL_UNITS))
        return world_x, world_y

    rows = tuple(
        tuple(
            card_placement_legal_world(
                card_spec,
                owner,
                candidate_world(x, y),
                form=form,
                destroyed_enemy_princess_lanes=lanes,
                active_tower_footprints=active_tower_footprints,
                occupied_building_footprints=occupied_building_footprints,
            )
            for x in range(GRID_WIDTH)
        )
        for y in range(GRID_HEIGHT)
    )
    if rule in {PlacementRule.UNKNOWN, PlacementRule.NO_TARGET, PlacementRule.ENTITY_TARGET}:
        rows = tuple(tuple(False for _ in range(GRID_WIDTH)) for _ in range(GRID_HEIGHT))
        accuracy = PlacementAccuracy.BLOCKED
        reasons.append(f"unsupported_grid_placement_rule:{rule.value}")
    if lanes:
        if accuracy != PlacementAccuracy.BLOCKED:
            accuracy = PlacementAccuracy.COARSE_CONSERVATIVE
        reasons.append("post_tower_pocket_bounds_are_conservative")
    if uses_entity_deployment_center(card_spec):
        reasons.append("active_tower_rectangles_enforced")
    if kind == CardKind.BUILDING:
        collision_radius = raw_collision_radius(card_spec, form=form)
        if building_profile is None or building_profile.blocked:
            rows = tuple(tuple(False for _ in range(GRID_WIDTH)) for _ in range(GRID_HEIGHT))
            accuracy = PlacementAccuracy.BLOCKED
            reasons.append(f"unverified_native_building_footprint:{str(form).lower()}")
            if building_profile is None:
                reasons.append("building_placement_profile:missing")
                reasons.append("building_placement_blocker:missing_explicit_building_placement_profile")
            else:
                reasons.append(f"building_placement_profile:{building_profile.placement_kind.value}")
                reasons.append(f"building_placement_blocker:{building_profile.blocker}")
        elif footprint is not None:
            reasons.extend(
                (
                    (f"native_probed_tile_footprint:{footprint.width_tiles}x{footprint.height_tiles}"),
                    "native_anchor_lattice_enforced",
                    "active_tower_rectangles_enforced",
                    "occupied_building_rectangles_enforced",
                )
            )
            if building_profile is not None:
                reasons.append(f"building_placement_profile:{building_profile.placement_kind.value}")
                reasons.extend(f"building_placement_evidence:{item}" for item in building_profile.evidence)
        else:
            accuracy = PlacementAccuracy.COARSE_CONSERVATIVE
            reasons.extend(
                (
                    "evidence_backed_special_building_deploy",
                    "no_stationary_collision_footprint",
                    (f"building_placement_profile:{building_profile.placement_kind.value}"),
                )
            )
            reasons.extend(f"building_placement_evidence:{item}" for item in building_profile.evidence)
    elif rule == PlacementRule.ANYWHERE and kind != CardKind.SPELL:
        accuracy = PlacementAccuracy.COARSE_CONSERVATIVE
        reasons.extend(
            ("full_arena_non_spell_uses_terrain_guard", "continuous_native_placement_not_exhaustively_probed")
        )
    return PlacementMaskV1(
        ruleset_id=ruleset_id,
        card_id=card_id,
        owner=owner,
        rule=rule,
        rows=rows,
        card_spec_hash=card_spec_hash,
        collision_radius_units=collision_radius,
        form=form,
        accuracy=accuracy,
        footprint_width_tiles=(footprint.width_tiles if footprint is not None else None),
        footprint_height_tiles=(footprint.height_tiles if footprint is not None else None),
        model_subcell_offset=model_subcell_offset,
        reasons=tuple(reasons),
        destroyed_enemy_princess_lanes=lanes,
    )


def building_footprint_mask(
    card_spec: CardSpecV1 | Mapping[str, object],
    owner: int,
    *,
    ruleset_id: str,
    form: str = "base",
    destroyed_enemy_princess_lanes: Iterable[str] = (),
    active_tower_footprints: Sequence[OccupiedFootprintV1] | None = None,
    occupied_building_footprints: Sequence[OccupiedFootprintV1] = (),
) -> PlacementMaskV1:
    raw_kind = _card_field(card_spec, "kind")
    kind = raw_kind if isinstance(raw_kind, CardKind) else CardKind(str(raw_kind))
    if kind != CardKind.BUILDING:
        raise ValueError("building_footprint_mask requires a building CardSpec")
    return card_placement_mask(
        card_spec,
        owner,
        ruleset_id=ruleset_id,
        form=form,
        destroyed_enemy_princess_lanes=destroyed_enemy_princess_lanes,
        active_tower_footprints=active_tower_footprints,
        occupied_building_footprints=occupied_building_footprints,
    )
