"""Deterministic public card-cycle and elixir tracking for self-play."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Mapping, Sequence

from ..battle_env import ELIXIR_SEGMENTS, TICK_MS
from ..contracts import EventV1, ObservationV1


MIRROR_CARD_ID = 28_000_006


class AvailabilityIndex(IntEnum):
    UNKNOWN_INITIAL = 0
    AVAILABLE = 1
    CYCLING = 2


@dataclass(frozen=True, slots=True)
class PublicCardState:
    availability: AvailabilityIndex
    cycle_distance: int
    initial_state_known: bool
    evolution_cycle_required: int = 0
    evolution_cycle_remaining: int = 0
    evolution_ready: bool = False


@dataclass(frozen=True, slots=True)
class PublicAbilityState:
    """Deterministic public Champion cooldown state."""

    activated_tick: int | None
    ready_tick: int
    cooldown_remaining_ticks: int
    ready: bool
    max_charges: int | None = None
    remaining_charges: int | None = None
    exhausted: bool = False

    @property
    def cooldown_remaining_ms(self) -> int:
        return int(self.cooldown_remaining_ticks) * TICK_MS


def _generated_elixir(start_tick: int, end_tick: int) -> float:
    cursor = max(0, int(start_tick))
    stop = max(cursor, int(end_tick))
    generated = 0.0
    while cursor < stop:
        boundary = stop
        full_bar_ms = ELIXIR_SEGMENTS[-1][2]
        for start, end, segment_ms in ELIXIR_SEGMENTS:
            if cursor >= start and (end is None or cursor < end):
                boundary = min(stop, end if end is not None else stop)
                full_bar_ms = segment_ms
                break
        generated += (boundary - cursor) * (10.0 * TICK_MS / full_bar_ms)
        cursor = boundary
    return generated


def _event_identity(event: EventV1) -> tuple[object, ...]:
    return (
        event.tick,
        event.event_type,
        event.owner,
        event.entity_id,
        event.card_id,
        event.data.get("action_id"),
        event.combat.projectile_id if event.combat is not None else None,
    )


class DeterministicPublicTracker:
    """Public, probability-free state for one policy perspective.

    The tracker must be reset with the exact ruleset initial elixir for both
    owners.  That value is public deterministic match state, not privileged
    opponent information.  Public events may be fed to two independent
    instances, but private hand/cycle reconciliation must only touch the
    instance belonging to that observing owner.
    """

    CARD_EXECUTION_EVENTS = frozenset({"action_executed"})
    ABILITY_EXECUTION_EVENTS = frozenset({"ability_activation", "runtime_ability_activation"})

    def __init__(
        self,
        *,
        decks: Mapping[int, Sequence[int]],
        card_costs: Mapping[int, float],
        ability_cost_by_owner: Mapping[int, float],
        ability_cooldown_ms_by_owner: Mapping[int, int],
        evolution_cycle_required: Mapping[int, int],
        evolution_cycle_required_by_owner: (Mapping[int, Mapping[int, int]] | None) = None,
        ability_cost_by_owner_card: (Mapping[int, Mapping[int, float]] | None) = None,
        ability_cooldown_ms_by_owner_card: (Mapping[int, Mapping[int, int]] | None) = None,
        ability_max_charges_by_owner_card: (Mapping[int, Mapping[int, int]] | None) = None,
        ability_form_to_base_card: Mapping[int, int] | None = None,
    ) -> None:
        self.decks = {int(owner): tuple(int(card) for card in deck) for owner, deck in decks.items()}
        if set(self.decks) != {0, 1} or any(len(deck) != 8 or len(set(deck)) != 8 for deck in self.decks.values()):
            raise ValueError("tracker requires two distinct eight-card decks")
        self.card_costs = {int(card): float(cost) for card, cost in card_costs.items()}
        self.ability_cost_by_owner = {int(owner): float(cost) for owner, cost in ability_cost_by_owner.items()}
        self.ability_cooldown_ticks_by_owner = {
            owner: int(math.ceil(int(milliseconds) / TICK_MS))
            for owner, milliseconds in ability_cooldown_ms_by_owner.items()
        }
        if set(self.ability_cost_by_owner) != {0, 1}:
            raise ValueError("ability costs must cover both owners")
        if set(self.ability_cooldown_ticks_by_owner) != {0, 1} or any(
            value <= 0 for value in self.ability_cooldown_ticks_by_owner.values()
        ):
            raise ValueError("ability cooldowns must be positive for both owners")
        if (ability_cost_by_owner_card is None) != (ability_cooldown_ms_by_owner_card is None):
            raise ValueError("card-aware ability cost and cooldown contracts must be supplied together")
        if ability_cost_by_owner_card is None:
            self.ability_cost_by_owner_card = {0: {}, 1: {}}
            self.ability_cooldown_ticks_by_owner_card = {0: {}, 1: {}}
        else:
            assert ability_cooldown_ms_by_owner_card is not None
            if set(ability_cost_by_owner_card) != {0, 1} or set(ability_cooldown_ms_by_owner_card) != {0, 1}:
                raise ValueError("card-aware ability contracts must cover both owners")
            self.ability_cost_by_owner_card = {
                owner: {int(card): float(cost) for card, cost in ability_cost_by_owner_card[owner].items()}
                for owner in (0, 1)
            }
            self.ability_cooldown_ticks_by_owner_card = {
                owner: {
                    int(card): int(math.ceil(int(milliseconds) / TICK_MS))
                    for card, milliseconds in (ability_cooldown_ms_by_owner_card[owner].items())
                }
                for owner in (0, 1)
            }
            for owner in (0, 1):
                costs = self.ability_cost_by_owner_card[owner]
                cooldowns = self.ability_cooldown_ticks_by_owner_card[owner]
                if set(costs) != set(cooldowns):
                    raise ValueError("card-aware ability cost/cooldown cards must match")
                if any(cost < 0.0 for cost in costs.values()):
                    raise ValueError("ability costs must be non-negative")
                if any(ticks <= 0 for ticks in cooldowns.values()):
                    raise ValueError("ability cooldowns must be positive")
                outside_deck = set(costs).difference(self.decks[owner])
                if outside_deck:
                    raise ValueError(
                        f"card-aware ability contracts contain cards outside owner {owner} deck: {sorted(outside_deck)}"
                    )
        if ability_max_charges_by_owner_card is None:
            self.ability_max_charges_by_owner_card = {0: {}, 1: {}}
        else:
            if set(ability_max_charges_by_owner_card) != {0, 1}:
                raise ValueError("card-aware ability charge contracts must cover both owners")
            self.ability_max_charges_by_owner_card = {
                owner: {int(card): int(maximum) for card, maximum in (ability_max_charges_by_owner_card[owner].items())}
                for owner in (0, 1)
            }
            for owner, charges in self.ability_max_charges_by_owner_card.items():
                if any(maximum <= 0 for maximum in charges.values()):
                    raise ValueError("ability max charges must be positive")
                outside_deck = set(charges).difference(self.decks[owner])
                if outside_deck:
                    raise ValueError(
                        "card-aware ability charge contracts contain cards "
                        f"outside owner {owner} deck: {sorted(outside_deck)}"
                    )
        raw_ability_form_to_base_card = {
            int(form): int(card) for form, card in (ability_form_to_base_card or {}).items()
        }
        known_ability_cards = set().union(
            *(set(cards) for cards in self.ability_cost_by_owner_card.values()),
            *(set(cards) for cards in self.ability_max_charges_by_owner_card.values()),
        )
        # Callers may provide the catalog-wide form alias table.  Bind only
        # aliases whose base card has an ability contract in this match; a
        # suffix-Hero definition that is not enabled by the episode form mask
        # must not silently turn into a policy ability contract.
        self.ability_form_to_base_card = {
            form: card for form, card in raw_ability_form_to_base_card.items() if card in known_ability_cards
        }
        self.evolution_cycle_required = {
            int(card): int(required) for card, required in evolution_cycle_required.items()
        }
        if any(required <= 0 for required in self.evolution_cycle_required.values()):
            raise ValueError("evolution cycle requirements must be positive")
        if evolution_cycle_required_by_owner is None:
            self.evolution_cycle_required_by_owner = {
                owner: {
                    card: self.evolution_cycle_required[card]
                    for card in self.decks[owner]
                    if card in self.evolution_cycle_required
                }
                for owner in (0, 1)
            }
        else:
            if set(evolution_cycle_required_by_owner) != {0, 1}:
                raise ValueError("owner-specific evolution cycles must cover both owners")
            self.evolution_cycle_required_by_owner = {
                owner: {int(card): int(required) for card, required in evolution_cycle_required_by_owner[owner].items()}
                for owner in (0, 1)
            }
            for owner, requirements in self.evolution_cycle_required_by_owner.items():
                if any(required <= 0 for required in requirements.values()):
                    raise ValueError("evolution cycle requirements must be positive")
                outside_deck = set(requirements).difference(self.decks[owner])
                if outside_deck:
                    raise ValueError(
                        "owner-specific evolution cycles contain cards outside "
                        f"owner {owner} deck: {sorted(outside_deck)}"
                    )
        self.tick = 0
        self.elixir = {0: 0.0, 1: 0.0}
        # ``elixir`` is the deterministic upper bound once an observing FAIR
        # owner is bound.  Exact own state is reconciled every decision;
        # opponent ability casts currently have no public, uniquely joined
        # event, so the lower bound remains conservative instead of consuming
        # oracle/private runtime telemetry.
        self.elixir_bounds: dict[int, tuple[float, float]] = {0: (0.0, 0.0), 1: (0.0, 0.0)}
        self._observing_owner: int | None = None
        self._card_states: dict[int, dict[int, PublicCardState]] = {}
        self._seen_events: set[tuple[object, ...]] = set()
        self._revealed_card_order: dict[int, list[int]] = {0: [], 1: []}
        self._ability_activated_tick: dict[int, int | None] = {0: None, 1: None}
        self._ability_activated_tick_by_source: dict[tuple[int, int], int] = {}
        self._ability_card_by_source: dict[tuple[int, int], int] = {}
        self._ability_charge_state_by_source: dict[tuple[int, int], tuple[int, int]] = {}
        self._ability_activation_identities: set[tuple[int, int, int, int]] = set()
        self._ability_exact_spend_identities: set[tuple[int, int, int, int]] = set()
        self._ready = False

    def reset(self, *, tick: int, initial_elixir: Mapping[int, float]) -> None:
        if set(initial_elixir) != {0, 1}:
            raise ValueError("initial elixir must cover both owners")
        values = {owner: float(initial_elixir[owner]) for owner in (0, 1)}
        if any(value < 0.0 or value > 10.0001 for value in values.values()):
            raise ValueError("initial elixir must be in [0, 10]")
        self.tick = int(tick)
        self.elixir = values
        self.elixir_bounds = {owner: (value, value) for owner, value in values.items()}
        self._observing_owner = None
        self._card_states = {
            owner: {
                card: PublicCardState(
                    availability=AvailabilityIndex.UNKNOWN_INITIAL,
                    cycle_distance=0,
                    initial_state_known=False,
                    evolution_cycle_required=(self.evolution_cycle_required_by_owner[owner].get(card, 0)),
                    evolution_cycle_remaining=(self.evolution_cycle_required_by_owner[owner].get(card, 0)),
                    evolution_ready=False,
                )
                for card in deck
            }
            for owner, deck in self.decks.items()
        }
        self._seen_events.clear()
        self._revealed_card_order = {0: [], 1: []}
        self._ability_activated_tick = {0: None, 1: None}
        self._ability_activated_tick_by_source.clear()
        self._ability_card_by_source.clear()
        self._ability_charge_state_by_source.clear()
        self._ability_activation_identities.clear()
        self._ability_exact_spend_identities.clear()
        self._ready = True

    def end_episode(self) -> None:
        """Invalidate this tracker until the next episode reset."""

        self._ready = False
        self._observing_owner = None

    def _advance_to(self, tick: int) -> None:
        tick = int(tick)
        if tick < self.tick:
            raise ValueError("public tracker cannot move backwards")
        generated = _generated_elixir(self.tick, tick)
        for owner in (0, 1):
            low, high = self.elixir_bounds[owner]
            low = min(10.0, low + generated)
            high = min(10.0, high + generated)
            if self._observing_owner is not None and owner != self._observing_owner:
                low = 0.0
            self.elixir_bounds[owner] = (low, high)
            self.elixir[owner] = high
        self.tick = tick

    def _mirror_play_cost(self, event: EventV1) -> float:
        """Validate one public Mirror execution and return its exact spend."""

        owner = int(event.owner)  # update() already restricts owners.
        required = {"visible_card_id", "effective_card_id", "effective_cost", "form_code"}
        missing = sorted(required.difference(event.data))
        if missing:
            raise ValueError(f"public Mirror execution is missing exact fields: {missing}")

        def exact_integer(key: str) -> int:
            value = event.data[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"public Mirror {key} must be an exact integer")
            return int(value)

        visible_card_id = exact_integer("visible_card_id")
        effective_card_id = exact_integer("effective_card_id")
        form_code = exact_integer("form_code")
        raw_cost = event.data["effective_cost"]
        if (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, (int, float))
            or not math.isfinite(float(raw_cost))
            or float(raw_cost) < 0.0
        ):
            raise ValueError("public Mirror effective_cost must be finite and non-negative")
        if event.card_id != MIRROR_CARD_ID or visible_card_id != MIRROR_CARD_ID:
            raise ValueError("public Mirror execution does not preserve its visible root identity")
        if effective_card_id == MIRROR_CARD_ID or effective_card_id not in self.decks[owner]:
            raise ValueError("public Mirror effective card must be one non-Mirror owner deck card")
        if form_code not in {0, 1, 2}:
            raise ValueError("public Mirror execution has an unsupported normal-1v1 form code")
        try:
            source_cost = self.card_costs[effective_card_id]
        except KeyError as error:
            raise ValueError("public Mirror effective card has no deterministic cost") from error
        cost = float(raw_cost)
        if not math.isclose(cost, source_cost + 1.0, abs_tol=1e-6):
            raise ValueError("public Mirror effective cost must equal source cost plus one")
        return cost

    def _record_card_play(self, owner: int, card_id: int, *, exact_cost: float | None = None) -> None:
        if card_id not in self._card_states[owner]:
            raise ValueError(f"public play used card {card_id} outside owner {owner} deck")
        if card_id not in self._revealed_card_order[owner]:
            self._revealed_card_order[owner].append(card_id)
        current = self._card_states[owner]
        for card, state in tuple(current.items()):
            if state.availability == AvailabilityIndex.CYCLING:
                distance = max(0, state.cycle_distance - 1)
                current[card] = PublicCardState(
                    availability=(AvailabilityIndex.AVAILABLE if distance == 0 else AvailabilityIndex.CYCLING),
                    cycle_distance=distance,
                    initial_state_known=True,
                    evolution_cycle_required=state.evolution_cycle_required,
                    evolution_cycle_remaining=state.evolution_cycle_remaining,
                    evolution_ready=state.evolution_ready,
                )
        previous = current[card_id]
        evolution_remaining = previous.evolution_cycle_remaining
        evolution_ready = previous.evolution_ready
        if previous.evolution_cycle_required:
            if previous.evolution_ready:
                evolution_remaining = previous.evolution_cycle_required
                evolution_ready = False
            else:
                evolution_remaining = max(0, evolution_remaining - 1)
                evolution_ready = evolution_remaining == 0
        current[card_id] = PublicCardState(
            availability=AvailabilityIndex.CYCLING,
            cycle_distance=4,
            initial_state_known=True,
            evolution_cycle_required=previous.evolution_cycle_required,
            evolution_cycle_remaining=evolution_remaining,
            evolution_ready=evolution_ready,
        )
        if exact_cost is None:
            try:
                cost = self.card_costs[card_id]
            except KeyError as error:
                raise ValueError(f"card {card_id} has no deterministic cost") from error
        else:
            cost = float(exact_cost)
        low, high = self.elixir_bounds[owner]
        low = max(0.0, low - cost)
        high = max(0.0, high - cost)
        if self._observing_owner is not None and owner != self._observing_owner:
            low = 0.0
        self.elixir_bounds[owner] = (low, high)
        self.elixir[owner] = high

    def _normalize_ability_card(self, card_id: object) -> int | None:
        if card_id is None:
            return None
        try:
            normalized = int(card_id)
        except (TypeError, ValueError):
            return None
        return self.ability_form_to_base_card.get(normalized, normalized)

    def _ability_max_charges(self, owner: int, card_id: int | None) -> int | None:
        if card_id is None:
            return None
        return self.ability_max_charges_by_owner_card[int(owner)].get(int(card_id))

    def _bind_ability_source(
        self, owner: int, source_entity: int, card_id: int, *, activation_tick: int | None = None
    ) -> None:
        key = (int(owner), int(source_entity))
        card_id = int(card_id)
        previous_card = self._ability_card_by_source.get(key)
        if previous_card is not None and previous_card != card_id:
            raise ValueError(
                "public ability source changed card identity: "
                f"owner={owner}, source={source_entity}, "
                f"before={previous_card}, after={card_id}"
            )
        self._ability_card_by_source[key] = card_id
        maximum = self._ability_max_charges(owner, card_id)
        if maximum is None:
            return
        previous_charge_state = self._ability_charge_state_by_source.get(key)
        fallback_key = (int(owner), -card_id)
        fallback_tick = self._ability_activated_tick_by_source.get(fallback_key)
        fallback_state = self._ability_charge_state_by_source.get(fallback_key)
        if activation_tick is not None and fallback_tick == int(activation_tick) and fallback_state is not None:
            if previous_charge_state is not None and previous_charge_state[0] != maximum:
                raise ValueError("public ability source changed max-charge contract")
            self._ability_charge_state_by_source[key] = fallback_state
            self._ability_activated_tick_by_source[key] = int(activation_tick)
        elif previous_charge_state is None:
            self._ability_charge_state_by_source[key] = (maximum, maximum)
        elif previous_charge_state[0] != maximum:
            raise ValueError(
                "public ability source changed max-charge contract: "
                f"owner={owner}, source={source_entity}, "
                f"before={previous_charge_state[0]}, after={maximum}"
            )

    def _consume_ability_charge(self, owner: int, source_entity: int | None, card_id: int | None) -> None:
        maximum = self._ability_max_charges(owner, card_id)
        if maximum is None:
            return
        source_key = int(source_entity) if source_entity is not None else -int(card_id)
        key = (int(owner), source_key)
        if source_entity is not None and card_id is not None:
            self._bind_ability_source(owner, source_entity, card_id)
        bound_maximum, remaining = self._ability_charge_state_by_source.get(key, (maximum, maximum))
        if bound_maximum != maximum:
            raise ValueError("public ability activation conflicts with max-charge contract")
        self._ability_charge_state_by_source[key] = (maximum, max(0, remaining - 1))

    def _learn_public_ability_sources(self, observation: ObservationV1) -> None:
        for player in observation.players:
            owner = int(player.owner)
            contracts = (
                self.ability_cost_by_owner_card[owner].keys() | self.ability_max_charges_by_owner_card[owner].keys()
            )
            for state in player.ability_runtime_states:
                if state.source_entity is None:
                    continue
                source_card = self._normalize_ability_card(state.attributes.get("source_card_id"))
                if source_card in contracts:
                    self._bind_ability_source(owner, int(state.source_entity), int(source_card))
        for entity in observation.entities:
            if entity.owner not in (0, 1):
                continue
            owner = int(entity.owner)
            contracts = (
                self.ability_cost_by_owner_card[owner].keys() | self.ability_max_charges_by_owner_card[owner].keys()
            )
            candidates = {
                card_id
                for card_id in (
                    self._normalize_ability_card(entity.card_id),
                    *(
                        self._normalize_ability_card(state.attributes.get("source_card_id"))
                        for state in entity.ability_states
                    ),
                )
                if card_id in contracts
            }
            if len(candidates) == 1:
                self._bind_ability_source(owner, int(entity.entity_id), candidates.pop())

    def _ability_card_for_event(self, event: EventV1) -> int | None:
        owner = int(event.owner)  # update() already restricts owners.
        contracts = self.ability_cost_by_owner_card[owner].keys() | self.ability_max_charges_by_owner_card[owner].keys()
        if not contracts:
            return None
        candidates: set[int] = set()
        event_card = self._normalize_ability_card(event.card_id)
        if event_card in contracts:
            candidates.add(int(event_card))
        data_card = self._normalize_ability_card(event.data.get("source_card_id"))
        if data_card in contracts:
            candidates.add(int(data_card))
        if event.entity_id is not None:
            learned = self._ability_card_by_source.get((owner, int(event.entity_id)))
            if learned is not None:
                candidates.add(learned)
        if len(candidates) == 1:
            return candidates.pop()
        if not candidates and len(contracts) == 1:
            return next(iter(contracts))
        raise ValueError(
            "public ability activation cannot be joined to one owner card: "
            f"owner={owner}, tick={event.tick}, entity={event.entity_id}, "
            f"card={event.card_id}, candidates={sorted(candidates)}"
        )

    @staticmethod
    def _event_visible_to_observer(event: EventV1, observer: int | None) -> bool:
        if observer not in (0, 1):
            return event.data.get("oracle_only") is not True
        if event.combat is not None and event.combat.visible_by_owner:
            return event.combat.visible_by_owner.get(str(observer)) is True
        if event.data.get("oracle_only") is True:
            return False
        private_to = event.data.get("private_to")
        return private_to is None or int(private_to) == int(observer)

    def update(self, observation: ObservationV1) -> None:
        if not self._ready:
            raise RuntimeError("tracker.reset must be called before update")
        if observation.tick < self.tick:
            raise ValueError("observation tick precedes tracker state")
        if observation.owner in (0, 1):
            owner = int(observation.owner)
            if self._observing_owner is None:
                self._observing_owner = owner
            elif self._observing_owner != owner:
                raise ValueError("public tracker cannot switch observing owner")
            enemy = 1 - owner
            _low, high = self.elixir_bounds[enemy]
            self.elixir_bounds[enemy] = (0.0, high)
            self.elixir[enemy] = high
        self._learn_public_ability_sources(observation)
        candidates = [
            event
            for event in observation.events
            if _event_identity(event) not in self._seen_events
            and event.tick >= self.tick
            and event.tick <= observation.tick
            and event.owner in (0, 1)
            and self._event_visible_to_observer(event, observation.owner)
        ]
        candidates.sort(
            key=lambda event: (event.tick, event.event_type, event.owner if event.owner is not None else -1)
        )
        for event in candidates:
            self._advance_to(event.tick)
            owner = int(event.owner)
            if event.event_type in self.CARD_EXECUTION_EVENTS and event.data.get("kind") != "activate_ability":
                if event.card_id is None:
                    raise ValueError("public card execution has no card_id")
                card_id = int(event.card_id)
                self._record_card_play(
                    owner, card_id, exact_cost=(self._mirror_play_cost(event) if card_id == MIRROR_CARD_ID else None)
                )
            elif event.event_type in self.ABILITY_EXECUTION_EVENTS:
                exact_activation = event.data.get("fair_ability_activation_exact") is True
                public_activation = event.event_type == "ability_activation"
                if not public_activation and not exact_activation:
                    # Exact-build runtime ability transitions are currently
                    # owner-private/oracle-only and do not uniquely establish
                    # a FAIR opponent cast.  The opponent interval already
                    # accounts for such unobserved spend; never convert these
                    # events into a fabricated exact value.
                    self._seen_events.add(_event_identity(event))
                    continue
                ability_card = self._ability_card_for_event(event)
                source_entity = None if event.entity_id is None else int(event.entity_id)
                activation_identity = (
                    owner,
                    int(event.tick),
                    ability_card if ability_card is not None else -1,
                    source_entity if source_entity is not None else -1,
                )
                source_agnostic_identity = (owner, int(event.tick), -1, -1)
                card_source_agnostic_identity = (
                    owner,
                    int(event.tick),
                    ability_card if ability_card is not None else -1,
                    -1,
                )
                activation_already_recorded = (
                    activation_identity in self._ability_activation_identities
                    or source_agnostic_identity in self._ability_activation_identities
                    or card_source_agnostic_identity in self._ability_activation_identities
                )
                exact_spend_already_recorded = (
                    activation_identity in self._ability_exact_spend_identities
                    or source_agnostic_identity in self._ability_exact_spend_identities
                    or card_source_agnostic_identity in self._ability_exact_spend_identities
                )
                if ability_card is not None and source_entity is not None:
                    self._bind_ability_source(owner, source_entity, ability_card, activation_tick=int(event.tick))
                activation_already_recorded = activation_already_recorded or any(
                    identity[:3] == (owner, int(event.tick), ability_card)
                    and (identity[3] == -1 or source_entity is None)
                    for identity in self._ability_activation_identities
                )
                exact_spend_already_recorded = exact_spend_already_recorded or any(
                    identity[:3] == (owner, int(event.tick), ability_card)
                    and (identity[3] == -1 or source_entity is None)
                    for identity in self._ability_exact_spend_identities
                )
                if exact_activation and not exact_spend_already_recorded:
                    cost = (
                        self.ability_cost_by_owner[owner]
                        if ability_card is None
                        else self.ability_cost_by_owner_card[owner].get(ability_card, self.ability_cost_by_owner[owner])
                    )
                    low, high = self.elixir_bounds[owner]
                    low = max(0.0, low - cost)
                    high = max(0.0, high - cost)
                    if self._observing_owner is not None and owner != self._observing_owner:
                        low = 0.0
                    self.elixir_bounds[owner] = (low, high)
                    self.elixir[owner] = high
                    self._ability_exact_spend_identities.add(activation_identity)
                if not activation_already_recorded:
                    self._ability_activated_tick[owner] = int(event.tick)
                    source_key = (
                        source_entity
                        if source_entity is not None
                        else -(int(ability_card) if ability_card is not None else 1)
                    )
                    self._ability_activated_tick_by_source[(owner, source_key)] = int(event.tick)
                    if ability_card is not None:
                        self._ability_activated_tick_by_source[(owner, -ability_card)] = int(event.tick)
                        self._ability_card_by_source[(owner, source_key)] = ability_card
                        if source_entity is not None:
                            self._bind_ability_source(
                                owner, source_entity, ability_card, activation_tick=int(event.tick)
                            )
                    self._consume_ability_charge(owner, source_entity, ability_card)
                    self._ability_activation_identities.add(activation_identity)
            self._seen_events.add(_event_identity(event))
        self._advance_to(observation.tick)

    def reconcile_owner_private_state(self, observation: ObservationV1) -> None:
        """Synchronize the observing owner's exact elixir, hand and cycle."""

        if observation.owner not in (0, 1):
            raise ValueError("owner reconciliation requires a fair actor observation")
        player = next(item for item in observation.players if item.owner == observation.owner)
        if player.elixir_exact is None:
            raise ValueError("actor observation has no exact own elixir")
        owner = int(observation.owner)
        self.elixir[owner] = float(player.elixir_exact)
        self.elixir_bounds[owner] = (float(player.elixir_exact), float(player.elixir_exact))
        private_cards = (*player.hand, *player.cycle)
        if (
            len(player.hand) > 4
            or len(private_cards) != 8
            or len(set(private_cards)) != len(private_cards)
            or set(private_cards) != set(self.decks[owner])
        ):
            raise ValueError(
                "private owner state requires at most four hand cards plus "
                "the complementary native cycle to partition the deck: "
                f"owner={owner}, tick={observation.tick}, "
                f"hand={player.hand!r}, cycle={player.cycle!r}, "
                f"next_card={player.next_card!r}"
            )
        for card_id in player.hand:
            previous = self._card_states[owner][card_id]
            self._card_states[owner][card_id] = PublicCardState(
                availability=AvailabilityIndex.AVAILABLE,
                cycle_distance=0,
                initial_state_known=True,
                evolution_cycle_required=previous.evolution_cycle_required,
                evolution_cycle_remaining=previous.evolution_cycle_remaining,
                evolution_ready=previous.evolution_ready,
            )
        pending_refills = 4 - len(player.hand)
        for native_distance, card_id in enumerate(player.cycle, start=1):
            previous = self._card_states[owner][card_id]
            self._card_states[owner][card_id] = PublicCardState(
                availability=AvailabilityIndex.CYCLING,
                # Native keeps cards awaiting the 20-tick refill cadence at
                # the front of ``cycle``. They need time, not another card
                # play, before becoming available. The remaining four cards
                # retain the public 1..4 play-distance convention.
                cycle_distance=max(0, native_distance - pending_refills),
                initial_state_known=True,
                evolution_cycle_required=previous.evolution_cycle_required,
                evolution_cycle_remaining=previous.evolution_cycle_remaining,
                evolution_ready=previous.evolution_ready,
            )

    def card_state(self, owner: int, card_id: int) -> PublicCardState:
        if not self._ready:
            raise RuntimeError("tracker is not initialized")
        return self._card_states[int(owner)][int(card_id)]

    def owner_card_states(self, owner: int) -> Mapping[int, PublicCardState]:
        if not self._ready:
            raise RuntimeError("tracker is not initialized")
        return dict(self._card_states[int(owner)])

    def revealed_cards(self, owner: int) -> tuple[int, ...]:
        """Return public card identities in deterministic first-play order."""

        if not self._ready:
            raise RuntimeError("tracker is not initialized")
        return tuple(self._revealed_card_order[int(owner)])

    def elixir_interval(self, owner: int) -> tuple[float, float]:
        """Return the FAIR lower/upper elixir bounds for ``owner``."""

        if not self._ready:
            raise RuntimeError("tracker is not initialized")
        return self.elixir_bounds[int(owner)]

    def ability_state(
        self, owner: int, *, tick: int | None = None, source_entity: int | None = None, card_id: int | None = None
    ) -> PublicAbilityState:
        if not self._ready:
            raise RuntimeError("tracker is not initialized")
        owner = int(owner)
        current_tick = self.tick if tick is None else int(tick)
        normalized_card = self._normalize_ability_card(card_id)
        source_key = None if source_entity is None else int(source_entity)
        if normalized_card is None and source_key is not None:
            normalized_card = self._ability_card_by_source.get((owner, source_key))
        if source_key is None and normalized_card is not None:
            source_key = -normalized_card
        charge_key = None if source_key is None else (owner, int(source_key))
        charge_state = None if charge_key is None else self._ability_charge_state_by_source.get(charge_key)
        contract_maximum = self._ability_max_charges(owner, normalized_card)
        if charge_state is None:
            max_charges = contract_maximum
            remaining_charges = contract_maximum
        else:
            max_charges, remaining_charges = charge_state
        exhausted = remaining_charges == 0 if remaining_charges is not None else False
        activated = (
            self._ability_activated_tick[owner]
            if source_key is None
            else self._ability_activated_tick_by_source.get((owner, source_key))
        )
        if activated is None:
            return PublicAbilityState(
                activated_tick=None,
                ready_tick=0,
                cooldown_remaining_ticks=0,
                ready=not exhausted,
                max_charges=max_charges,
                remaining_charges=remaining_charges,
                exhausted=exhausted,
            )
        if exhausted:
            return PublicAbilityState(
                activated_tick=activated,
                ready_tick=0,
                cooldown_remaining_ticks=0,
                ready=False,
                max_charges=max_charges,
                remaining_charges=remaining_charges,
                exhausted=True,
            )
        cooldown_ticks = (
            self.ability_cooldown_ticks_by_owner_card[owner].get(
                normalized_card, self.ability_cooldown_ticks_by_owner[owner]
            )
            if normalized_card is not None
            else self.ability_cooldown_ticks_by_owner[owner]
        )
        ready_tick = activated + cooldown_ticks
        remaining = max(0, ready_tick - current_tick)
        return PublicAbilityState(
            activated_tick=activated,
            ready_tick=ready_tick,
            cooldown_remaining_ticks=remaining,
            ready=remaining == 0,
            max_charges=max_charges,
            remaining_charges=remaining_charges,
            exhausted=False,
        )
