"""Torch-free native action contracts for the V4 policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ...contracts import ActionV1, ObservationV1, TargetKind
from ..card_features import HERO_FORM_TO_BASE_CARD


NATIVE_TICK_MS = 50
MIRROR_CARD_ID = 28_000_006


@dataclass(frozen=True, slots=True)
class DecodedActionSequenceV4:
    """One owner's fixed-turn WAIT or ordered native commands."""

    owner: int
    actions: tuple[ActionV1, ...] = ()

    def __post_init__(self) -> None:
        if self.owner not in (0, 1):
            raise ValueError("decoded action owner must be 0 or 1")
        if len(self.actions) > 2:
            raise ValueError("V4 supports at most two native commands")
        if any(action.owner != self.owner for action in self.actions):
            raise ValueError("decoded action contains a command for another owner")


@dataclass(frozen=True, slots=True)
class AbilityRuntimeContractV4:
    ability_id: str
    source_card_id: int
    cost: float
    cooldown_group_id: int
    charges: int | None
    target_mode: TargetKind

    def __post_init__(self) -> None:
        if not self.ability_id:
            raise ValueError("ability runtime identity must not be empty")
        if self.source_card_id <= 0:
            raise ValueError("ability source card identity must be positive")
        if not math.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("ability runtime cost must be finite and non-negative")
        if self.cooldown_group_id < 0:
            raise ValueError("ability cooldown group must be non-negative")
        if self.charges is not None and self.charges < 0:
            raise ValueError("ability charges must be non-negative")
        if self.target_mode != TargetKind.NONE:
            raise ValueError("current V4 native abilities must be target-free")


def ability_source_card_id(observation: ObservationV1, *, source_entity: int, deck: Sequence[int]) -> int | None:
    """Resolve one public Ability source to exactly one deck card identity."""

    deck_ids = {int(card_id) for card_id in deck}
    entity = next(
        (
            item
            for item in observation.entities
            if int(item.entity_id) == source_entity and item.owner == observation.owner
        ),
        None,
    )
    player = next((item for item in observation.players if item.owner == observation.owner), None)
    if entity is None or player is None:
        return None

    runtime_states = tuple(entity.ability_states) + tuple(
        state for state in player.ability_runtime_states if state.source_entity == source_entity
    )
    candidates: set[int] = set()
    if entity.evolution_state is not None:
        candidates.add(int(entity.evolution_state.card_id))
    for state in runtime_states:
        raw = state.attributes.get("source_card_id")
        if raw is not None:
            candidates.add(int(raw))
    if entity.card_id is not None:
        candidates.add(int(HERO_FORM_TO_BASE_CARD.get(int(entity.card_id), entity.card_id)))
    if entity.projectile_state is not None and entity.projectile_state.source_card_id is not None:
        candidates.add(int(entity.projectile_state.source_card_id))

    in_deck = candidates & deck_ids
    if len(in_deck) != 1:
        return None
    return next(iter(in_deck))


def ability_runtime_contract(
    observation: ObservationV1, *, source_entity: int, deck: Sequence[int]
) -> AbilityRuntimeContractV4:
    """Normalize one legal public Ability source for builder and decoder."""

    player = next((item for item in observation.players if item.owner == observation.owner), None)
    if player is None:
        raise ValueError("FAIR observation has no actor player state")
    states = tuple(state for state in player.ability_runtime_states if state.source_entity == source_entity)
    if len(states) != 1:
        raise ValueError("ability source lacks one exact player runtime state")
    state = states[0]
    source_card_id = ability_source_card_id(observation, source_entity=source_entity, deck=deck)
    if source_card_id is None:
        raise ValueError("ability source cannot be bound to one actor deck card")
    controller_slot = state.attributes.get("controller_slot")
    if isinstance(controller_slot, bool) or not isinstance(controller_slot, int):
        raise ValueError("ability runtime state lacks an exact cooldown controller")
    if state.elixir_cost is None:
        raise ValueError("ability runtime state lacks an exact public cost")
    return AbilityRuntimeContractV4(
        ability_id=state.ability_id,
        source_card_id=source_card_id,
        cost=float(state.elixir_cost),
        cooldown_group_id=int(controller_slot),
        charges=state.charges,
        target_mode=TargetKind.NONE,
    )


def mirror_play_runtime_contract(
    *,
    player: object,
    hand_slot: int,
    deck: Sequence[int],
    card_costs: Mapping[int, float],
    placement_entry: Mapping[str, Any],
) -> tuple[int, float, int]:
    """Join actor-private hand and legality projections for Mirror."""

    metadata = getattr(player, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Mirror play has no actor-private player metadata")
    runtime_by_slot = metadata.get("hand_runtime_by_slot")
    if not isinstance(runtime_by_slot, Mapping):
        raise ValueError("Mirror play has no exact hand runtime contract")
    hand_runtime = runtime_by_slot.get(str(hand_slot), runtime_by_slot.get(hand_slot))
    if not isinstance(hand_runtime, Mapping):
        raise ValueError("Mirror play has no exact runtime entry for its hand slot")

    required = {"hand_slot", "visible_card_id", "effective_card_id", "effective_cost", "form_code"}
    if not required.issubset(hand_runtime) or not required.difference({"hand_slot"}).issubset(placement_entry):
        raise ValueError("Mirror play runtime contract is incomplete")

    def exact_int(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Mirror {label} must be an exact integer")
        return int(value)

    def exact_cost(value: object, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Mirror {label} must be a finite non-negative cost")
        return float(value)

    hand_values = (
        exact_int(hand_runtime["visible_card_id"], "visible card identity"),
        exact_int(hand_runtime["effective_card_id"], "effective card identity"),
        exact_cost(hand_runtime["effective_cost"], "effective cost"),
        exact_int(hand_runtime["form_code"], "form code"),
    )
    placement_values = (
        exact_int(placement_entry["visible_card_id"], "placement visible identity"),
        exact_int(placement_entry["effective_card_id"], "placement effective identity"),
        exact_cost(placement_entry["effective_cost"], "placement effective cost"),
        exact_int(placement_entry["form_code"], "placement form code"),
    )
    if exact_int(hand_runtime["hand_slot"], "hand slot") != hand_slot:
        raise ValueError("Mirror runtime contract changed its hand slot")
    if hand_values != placement_values:
        raise ValueError("Mirror hand and placement runtime contracts disagree")

    visible_card_id, effective_card_id, effective_cost, form_code = hand_values
    if visible_card_id != MIRROR_CARD_ID:
        raise ValueError("Mirror runtime contract changed the visible action identity")
    if effective_card_id == MIRROR_CARD_ID or effective_card_id not in {int(card_id) for card_id in deck}:
        raise ValueError("Mirror effective card is not one non-Mirror deck card")
    if form_code not in {0, 1, 2}:
        raise ValueError("Mirror runtime contract has an unsupported form")
    try:
        expected_cost = float(card_costs[effective_card_id]) + 1.0
    except KeyError as error:
        raise ValueError("Mirror effective card has no exact static cost") from error
    if not math.isclose(effective_cost, expected_cost, abs_tol=1e-6):
        raise ValueError("Mirror effective cost does not equal source cost plus one")
    placement_card_id = placement_entry.get("card_id", effective_card_id)
    if exact_int(placement_card_id, "placement card identity") != effective_card_id:
        raise ValueError("Mirror placement mask belongs to a different card")
    return effective_card_id, effective_cost, form_code
