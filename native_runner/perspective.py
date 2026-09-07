"""Reversible owner-relative views and semantics-preserving augmentation.

The native engine uses one absolute 18x32 arena.  Policies should instead see
their own side at the bottom, independent of the native owner seat.  This
module transforms private ordering, action masks and actions together; the
V4 tensorizer applies the coordinate signs while writing scene tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
import math
from typing import Any

from .arena import GRID_HEIGHT, GRID_WIDTH
from .contracts import ActionMaskV1, ActionV1, ObservationTier, ObservationV1, content_hash, frozen_mapping


PERSPECTIVE_TRANSFORM_VERSION = "owner-perspective-transform.v1"


class PerspectiveError(ValueError):
    """Raised when a requested view is ambiguous or non-invertible."""


def _trusted_feature_replace(value: Any, /, **changes: Any) -> Any:
    """Copy a validated frozen contract without rerunning its full validator."""

    result = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(result, item.name, changes.get(item.name, getattr(value, item.name)))
    return result


def _permutation(value: Sequence[int], length: int, label: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if len(result) != length or set(result) != set(range(length)):
        raise PerspectiveError(f"{label} must be a permutation of 0..{length - 1}")
    return result


def _inverse_permutation(value: Sequence[int]) -> tuple[int, ...]:
    result = [0] * len(value)
    for transformed_index, source_index in enumerate(value):
        result[source_index] = transformed_index
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PerspectiveTransformV1:
    """One reversible native/model coordinate and token transform.

    ``hand_slot_permutation[model_slot]`` gives the corresponding native slot.
    ``deck_permutation`` only reorders the public/static deck-token input; it
    never changes the engine's actual card cycle.
    """

    actor_owner: int
    horizontal_mirror: bool = False
    hand_slot_permutation: tuple[int, ...] = (0, 1, 2, 3)
    deck_permutation: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    version: str = field(default=PERSPECTIVE_TRANSFORM_VERSION, init=False)
    _placement_cache: dict[tuple[int, int], tuple[Mapping[str, Any], Mapping[str, Any]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.actor_owner not in (0, 1):
            raise PerspectiveError("actor_owner must be 0 or 1")
        object.__setattr__(
            self, "hand_slot_permutation", _permutation(self.hand_slot_permutation, 4, "hand_slot_permutation")
        )
        object.__setattr__(self, "deck_permutation", _permutation(self.deck_permutation, 8, "deck_permutation"))

    @property
    def flip_x(self) -> bool:
        return bool(self.actor_owner == 1) ^ bool(self.horizontal_mirror)

    @property
    def flip_y(self) -> bool:
        return self.actor_owner == 1

    @property
    def transform_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "actor_owner": self.actor_owner,
            "horizontal_mirror": self.horizontal_mirror,
            "hand_slot_permutation": list(self.hand_slot_permutation),
            "deck_permutation": list(self.deck_permutation),
            "native_to_model": {"flip_x": self.flip_x, "flip_y": self.flip_y, "own_side": "bottom"},
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "PerspectiveTransformV1") -> "PerspectiveTransformV1":
        if isinstance(value, cls):
            return value
        data = dict(value)
        version = data.pop("version", None)
        if version != PERSPECTIVE_TRANSFORM_VERSION:
            raise PerspectiveError(f"expected {PERSPECTIVE_TRANSFORM_VERSION}, got {version!r}")
        data.pop("native_to_model", None)
        return cls(**data)

    def _grid(self, point: Sequence[int | float]) -> list[int]:
        if len(point) != 2:
            raise PerspectiveError("grid coordinate must contain x and y")
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            raise PerspectiveError("grid coordinate is outside the 18x32 arena")
        return [GRID_WIDTH - 1 - x if self.flip_x else x, GRID_HEIGHT - 1 - y if self.flip_y else y]

    def _vector(self, point: Sequence[int | float]) -> list[float]:
        if len(point) != 2:
            raise PerspectiveError("vector must contain x and y")
        x, y = float(point[0]), float(point[1])
        return [-x if self.flip_x else x, -y if self.flip_y else y]

    def _subcell(self, point: Sequence[int | float]) -> list[float]:
        return self._vector(point)

    def _placement_subcell(self, point: Sequence[int | float]) -> list[float]:
        """Mirror an owner-normalized building anchor for model augmentation."""

        if len(point) != 2:
            raise PerspectiveError("building subcell offset must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise PerspectiveError("building subcell offset must be finite")
        return [-x if self.horizontal_mirror else x, y]

    def _rows(self, rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
        materialized = [list(row) for row in rows]
        if len(materialized) != GRID_HEIGHT or any(len(row) != GRID_WIDTH for row in materialized):
            raise PerspectiveError("spatial matrices must be row-major [32][18]")
        if self.flip_y:
            materialized.reverse()
        if self.flip_x:
            materialized = [list(reversed(row)) for row in materialized]
        return materialized

    def _transform_action_mapping(self, value: Mapping[str, Any], *, model_to_native: bool) -> dict[str, Any]:
        data = dict(value)
        if int(data.get("owner", self.actor_owner)) != self.actor_owner:
            raise PerspectiveError("action owner does not match perspective owner")
        target = data.get("target_grid")
        if target is not None:
            data["target_grid"] = self._grid(target)
        subcell = data.get("subcell_offset")
        if subcell is not None:
            data["subcell_offset"] = self._subcell(subcell)
        slot = data.get("hand_slot")
        if slot is not None:
            slot = int(slot)
            if not 0 <= slot < 4:
                raise PerspectiveError("action hand slot is outside 0..3")
            permutation = (
                self.hand_slot_permutation if model_to_native else _inverse_permutation(self.hand_slot_permutation)
            )
            data["hand_slot"] = permutation[slot]
        metadata = data.get("metadata")
        audit = dict(metadata) if isinstance(metadata, Mapping) else {}
        audit["perspective_transform_hash"] = self.transform_hash
        audit["perspective_direction"] = "model-to-native" if model_to_native else "native-to-model"
        data["metadata"] = audit
        return data

    def action_to_native(self, action: ActionV1 | Mapping[str, Any]) -> ActionV1:
        data = action.to_dict() if isinstance(action, ActionV1) else dict(action)
        return ActionV1.from_mapping(self._transform_action_mapping(data, model_to_native=True))

    def action_to_model(self, action: ActionV1 | Mapping[str, Any]) -> ActionV1:
        data = action.to_dict() if isinstance(action, ActionV1) else dict(action)
        return ActionV1.from_mapping(self._transform_action_mapping(data, model_to_native=False))

    def _transform_mask_contract(self, value: ActionMaskV1) -> ActionMaskV1:
        """Transform an already validated mask without JSON round-tripping."""

        native_slots = value.hand_slots
        placements: dict[str, Any] = {}
        for model_slot, native_slot in enumerate(self.hand_slot_permutation):
            source = value.placement_masks.get(str(native_slot))
            if not isinstance(source, Mapping):
                continue
            cache_key = (id(source), native_slot)
            cached = self._placement_cache.get(cache_key)
            if cached is not None and cached[0] is source:
                transformed = cached[1]
            else:
                entry = dict(source)
                rows = entry.get("row_major")
                if isinstance(rows, Sequence):
                    entry["row_major"] = self._rows(rows)
                subcell = entry.get("model_subcell_offset")
                if subcell is not None:
                    entry["model_subcell_offset"] = self._placement_subcell(subcell)
                entry["source_native_hand_slot"] = native_slot
                transformed = frozen_mapping(entry)
                if len(self._placement_cache) >= 4096:
                    self._placement_cache.clear()
                self._placement_cache[cache_key] = (source, transformed)
            placements[str(model_slot)] = transformed

        reasons = dict(value.reasons)
        raw_slots = reasons.get("slots")
        if isinstance(raw_slots, Mapping):
            reasons["slots"] = {
                str(model_slot): raw_slots.get(str(native_slot), raw_slots.get(native_slot))
                for model_slot, native_slot in enumerate(self.hand_slot_permutation)
            }
        return _trusted_feature_replace(
            value,
            hand_slots=tuple(native_slots[source] for source in self.hand_slot_permutation),
            placement_masks=frozen_mapping(placements),
            reasons=frozen_mapping(reasons),
        )

    def _card_order_to_model(
        self, hand: tuple[int, ...], deck: tuple[int, ...], metadata: Mapping[str, Any]
    ) -> tuple[tuple[int, ...], tuple[int, ...], Mapping[str, Any]]:
        """Share the exact private slot/deck permutation across all views."""

        if hand:
            raw_slots = metadata.get("hand_slot_by_card")
            if isinstance(raw_slots, Mapping):
                inverse = _inverse_permutation(self.hand_slot_permutation)
                model_slots = {str(card_id): inverse[int(slot)] for card_id, slot in raw_slots.items()}
                if set(map(int, model_slots)) != set(map(int, hand)):
                    raise PerspectiveError("actor hand-slot metadata does not match hand")
                hand = tuple(sorted(hand, key=lambda card: model_slots[str(card)]))
                metadata = dict(metadata, hand_slot_by_card=model_slots)
            else:
                if len(hand) != 4:
                    raise PerspectiveError("partial actor hand requires slot metadata")
                hand = tuple(hand[source] for source in self.hand_slot_permutation)
        if deck:
            if len(deck) != 8:
                raise PerspectiveError("actor deck must have eight cards to permute")
            deck = tuple(deck[source] for source in self.deck_permutation)
        return hand, deck, metadata

    def observation_policy_metadata_to_model(self, observation: ObservationV1) -> ObservationV1:
        """Transform private card ordering and legality without copying scene rows.

        The V4 tensorizer can apply the coordinate signs directly while it
        writes tensors.  Keeping the already validated tower/entity/event
        contracts shared avoids rebuilding every public scene object solely
        to rotate its coordinates.
        """

        if observation.tier == ObservationTier.ORACLE:
            raise PerspectiveError("owner-relative transforms require an actor observation")
        if observation.owner != self.actor_owner:
            raise PerspectiveError("observation owner does not match perspective owner")
        identity = (
            not self.flip_x
            and not self.flip_y
            and self.hand_slot_permutation == (0, 1, 2, 3)
            and self.deck_permutation == (0, 1, 2, 3, 4, 5, 6, 7)
        )
        if identity:
            return observation

        players = []
        for player in observation.players:
            if player.owner != self.actor_owner:
                players.append(player)
                continue
            hand, deck, metadata = self._card_order_to_model(player.hand, player.deck, player.metadata)
            if hand == player.hand and deck == player.deck and metadata is player.metadata:
                players.append(player)
            else:
                players.append(
                    _trusted_feature_replace(player, hand=hand, deck=deck, metadata=frozen_mapping(metadata))
                )
        players.sort(key=lambda item: item.owner != self.actor_owner)
        return _trusted_feature_replace(
            observation,
            players=tuple(players),
            action_mask=self._transform_mask_contract(observation.action_mask),
            raster=None,
        )
