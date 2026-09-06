from __future__ import annotations

from native_runner.contracts import (
    ActionKind,
    ActionMaskV1,
    EventV1,
    ObservationTier,
    ObservationV1,
    PlayerStateV1,
    TimeStateV1,
)
from native_runner.training.tracking import (
    AvailabilityIndex,
    DeterministicPublicTracker,
)


DECK = tuple(range(101, 109))
MIRROR_CARD_ID = 28_000_006


def _tracker() -> DeterministicPublicTracker:
    tracker = DeterministicPublicTracker(
        decks={0: DECK, 1: DECK},
        card_costs={card_id: 3.0 for card_id in DECK},
        ability_cost_by_owner={0: 2.0, 1: 2.0},
        ability_cooldown_ms_by_owner={0: 25_000, 1: 25_000},
        evolution_cycle_required={103: 2},
    )
    tracker.reset(tick=0, initial_elixir={0: 6.0, 1: 6.0})
    return tracker


def _ability_tracker(
    *,
    max_charges: int,
) -> DeterministicPublicTracker:
    tracker = DeterministicPublicTracker(
        decks={0: DECK, 1: DECK},
        card_costs={card_id: 3.0 for card_id in DECK},
        ability_cost_by_owner={0: 2.0, 1: 2.0},
        ability_cooldown_ms_by_owner={0: 25_000, 1: 25_000},
        evolution_cycle_required={},
        ability_cost_by_owner_card={0: {101: 1.0}, 1: {101: 1.0}},
        ability_cooldown_ms_by_owner_card={
            0: {101: 1_000},
            1: {101: 1_000},
        },
        ability_max_charges_by_owner_card={
            0: {101: max_charges},
            1: {101: max_charges},
        },
    )
    tracker.reset(tick=0, initial_elixir={0: 6.0, 1: 6.0})
    return tracker


def _observation(
    *,
    tick: int,
    events: tuple[EventV1, ...] = (),
) -> ObservationV1:
    return ObservationV1(
        tier=ObservationTier.FAIR,
        owner=0,
        tick=tick,
        time=TimeStateV1(
            elapsed_ms=tick * 50,
            remaining_ms=max(0, 300_000 - tick * 50),
        ),
        phase="regulation",
        players=(
            PlayerStateV1(
                owner=0,
                elixir_exact=6.0,
                elixir_visible=6.0,
                hand=DECK[:4],
                deck=DECK,
                private_state_visible=True,
            ),
            PlayerStateV1(owner=1),
        ),
        events=events,
        action_mask=ActionMaskV1(
            kinds={
                ActionKind.WAIT.value: True,
                ActionKind.PLAY_CARD.value: False,
                ActionKind.ACTIVATE_ABILITY.value: False,
            }
        ),
    )


def test_public_tracker_cycles_only_from_executed_public_events() -> None:
    tracker = _tracker()
    tracker.update(
        _observation(
            tick=10,
            events=(
                EventV1(
                    tick=5,
                    event_type="action_executed",
                    owner=1,
                    card_id=103,
                ),
            ),
        )
    )

    state = tracker.card_state(1, 103)
    assert state.availability == AvailabilityIndex.CYCLING
    assert state.cycle_distance == 4
    assert state.evolution_cycle_remaining == 1
    assert tracker.revealed_cards(1) == (103,)


def test_public_tracker_defers_events_ahead_of_the_atomic_tick() -> None:
    tracker = _tracker()
    event = EventV1(
        tick=12,
        event_type="action_executed",
        owner=1,
        card_id=103,
    )

    tracker.update(_observation(tick=10, events=(event,)))
    assert (
        tracker.card_state(1, 103).availability
        == AvailabilityIndex.UNKNOWN_INITIAL
    )

    tracker.update(_observation(tick=12, events=(event,)))
    assert tracker.card_state(1, 103).availability == AvailabilityIndex.CYCLING


def test_public_tracker_preserves_first_reveal_order() -> None:
    tracker = _tracker()
    observation = _observation(
        tick=10,
        events=(
            EventV1(
                tick=5,
                event_type="action_executed",
                owner=1,
                card_id=104,
            ),
            EventV1(
                tick=6,
                event_type="action_executed",
                owner=1,
                card_id=103,
            ),
        ),
    )
    tracker.update(observation)
    tracker.update(observation)

    assert tracker.revealed_cards(1) == (104, 103)


def test_single_use_ability_exhausts_only_the_public_source() -> None:
    tracker = _ability_tracker(max_charges=1)
    tracker.update(
        _observation(
            tick=5,
            events=(
                EventV1(
                    tick=5,
                    event_type="ability_activation",
                    owner=1,
                    entity_id=501,
                    card_id=101,
                    data={"source_card_id": 101},
                ),
            ),
        )
    )

    used = tracker.ability_state(1, source_entity=501, card_id=101)
    unused = tracker.ability_state(1, source_entity=777, card_id=101)
    assert used.max_charges == 1
    assert used.remaining_charges == 0
    assert used.exhausted and not used.ready
    assert unused.remaining_charges == 1
    assert unused.ready and not unused.exhausted


def test_two_charge_ability_cools_down_then_exhausts() -> None:
    tracker = _ability_tracker(max_charges=2)
    first = EventV1(
        tick=5,
        event_type="ability_activation",
        owner=1,
        entity_id=501,
        card_id=101,
        data={"source_card_id": 101},
    )
    tracker.update(_observation(tick=5, events=(first,)))

    cooling = tracker.ability_state(1, source_entity=501, card_id=101)
    assert cooling.remaining_charges == 1
    assert cooling.cooldown_remaining_ticks == 20
    assert not cooling.ready and not cooling.exhausted

    tracker.update(_observation(tick=25, events=(first,)))
    assert tracker.ability_state(
        1,
        source_entity=501,
        card_id=101,
    ).ready

    second = EventV1(
        tick=30,
        event_type="ability_activation",
        owner=1,
        entity_id=501,
        card_id=101,
        data={"source_card_id": 101},
    )
    tracker.update(_observation(tick=30, events=(first, second)))
    exhausted = tracker.ability_state(1, source_entity=501, card_id=101)
    assert exhausted.remaining_charges == 0
    assert exhausted.exhausted and not exhausted.ready



def test_unattested_runtime_ability_event_never_consumes_public_charge() -> None:
    tracker = _ability_tracker(max_charges=1)
    tracker.update(
        _observation(
            tick=5,
            events=(
                EventV1(
                    tick=5,
                    event_type="runtime_ability_activation",
                    owner=1,
                    entity_id=501,
                    card_id=101,
                    data={"source_card_id": 101},
                ),
            ),
        )
    )

    state = tracker.ability_state(1, source_entity=501, card_id=101)
    assert state.remaining_charges == 1
    assert state.ready and not state.exhausted


def test_mirror_cycles_visible_card_and_spends_exact_effective_cost() -> None:
    deck = (MIRROR_CARD_ID, *DECK[:7])
    tracker = DeterministicPublicTracker(
        decks={0: deck, 1: deck},
        card_costs={MIRROR_CARD_ID: 0.0, **{card_id: 3.0 for card_id in DECK[:7]}},
        ability_cost_by_owner={0: 2.0, 1: 2.0},
        ability_cooldown_ms_by_owner={0: 25_000, 1: 25_000},
        evolution_cycle_required={},
    )
    tracker.reset(tick=0, initial_elixir={0: 10.0, 1: 10.0})
    tracker.update(
        _observation(
            tick=0,
            events=(
                EventV1(
                    tick=0,
                    event_type="action_executed",
                    owner=1,
                    card_id=MIRROR_CARD_ID,
                    data={
                        "visible_card_id": MIRROR_CARD_ID,
                        "effective_card_id": DECK[0],
                        "effective_cost": 4.0,
                        "form_code": 0,
                    },
                ),
            ),
        )
    )

    assert tracker.revealed_cards(1) == (MIRROR_CARD_ID,)
    assert (
        tracker.card_state(1, MIRROR_CARD_ID).availability
        == AvailabilityIndex.CYCLING
    )
    assert tracker.elixir_interval(1) == (0.0, 6.0)
