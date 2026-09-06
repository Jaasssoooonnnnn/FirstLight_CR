from __future__ import annotations

from copy import deepcopy

import pytest

from native_runner.contracts import (
    ActionKind,
    ActionV1,
    ContractError,
    EpisodeConfigV1,
    LatencyConfigV1,
    ObservationV1,
    OpponentBeliefV1,
    PlayerStateV1,
    TargetKind,
    TerminalV1,
    canonical_json,
)


DECK = (
    26000000, 26000001, 26000005, 28000001,
    28000000, 26000003, 26000014, 26000018,
)


def test_action_round_trip_and_stable_hash() -> None:
    action = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=2,
        card_id=26000000,
        target_kind=TargetKind.GRID,
        target_grid=(8, 24),
        subcell_offset=(0.25, -0.25),
        execute_offset_ticks=3,
        next_decision_ticks=4,
        metadata={"policy": "test", "nested": {"b": 2, "a": 1}},
    )
    restored = ActionV1.from_mapping(action.to_dict())
    assert restored == action
    assert restored.content_hash() == action.content_hash()
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_frozen_contracts_are_safe_for_in_process_snapshot_deepcopy() -> None:
    episode = EpisodeConfigV1(
        ruleset_id="a" * 64,
        deck0=DECK,
        deck1=DECK,
        seed=7,
        policy_hashes={"owner0": "b" * 64},
        tags={"nested": {"source": "snapshot"}},
    )

    restored = deepcopy(episode)

    assert restored == episode
    assert restored.policy_hashes == episode.policy_hashes
    assert restored.tags == episode.tags


@pytest.mark.parametrize(
    "kwargs",
    [
        {"owner": 2, "kind": "wait"},
        {"owner": 0, "kind": "play_card"},
        {"owner": 0, "kind": "activate_ability"},
        {"owner": 0, "kind": "wait", "next_decision_ticks": 0},
        {"owner": 0, "kind": "play_card", "hand_slot": 0, "target_kind": "grid"},
        {"owner": 0, "kind": "play_card", "hand_slot": 4},
    ],
)
def test_action_validation_rejects_illegal_branches(kwargs: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        ActionV1(**kwargs)  # type: ignore[arg-type]


def test_fair_observation_strips_native_opponent_private_state() -> None:
    native = {
        "version": ObservationV1.VERSION,
        "tick": 200,
        "players": [
            {
                "owner": 0,
                "elixirRaw": 65000,
                "elixir": 6.5,
                "hand": [{"cardId": card} for card in DECK[:4]],
                "deck": [{"cardId": card} for card in DECK],
                "nextCard": {"cardId": DECK[4]},
            },
            {
                "owner": 1,
                "elixirRaw": 90000,
                "elixir": 9.0,
                "hand": [{"cardId": card} for card in DECK[:4]],
                "deck": [{"cardId": card} for card in DECK],
                "nextCard": {"cardId": DECK[4]},
            },
        ],
        "objects": [],
        "ended": False,
    }
    observation = ObservationV1(
        tick=native["tick"], tier="fair", owner=0,
        players=tuple(
            PlayerStateV1.from_mapping(player, private_default=player["owner"] == 0)
            for player in native["players"]
        ),
    )
    own, opponent = observation.players
    assert own.private_state_visible
    assert own.elixir_exact == 6.5
    assert own.hand == DECK[:4]
    assert not opponent.private_state_visible
    assert opponent.elixir_exact is None
    assert opponent.elixir_visible is None
    assert opponent.hand == ()
    assert opponent.deck == ()
    assert observation.native_digest is None


def test_canonical_wire_contracts_require_explicit_versions() -> None:
    with pytest.raises(ContractError, match="got None"):
        ActionV1.from_mapping({"owner": 0, "kind": "wait"})
    with pytest.raises(ContractError, match="got None"):
        EpisodeConfigV1.from_mapping({
            "ruleset_id": "r" * 64,
            "deck0": DECK,
            "deck1": tuple(reversed(DECK)),
            "seed": 1,
        })
    assert LatencyConfigV1.from_mapping(None) == LatencyConfigV1()


def test_fair_observation_rejects_explicit_privileged_leaks() -> None:
    opponent = PlayerStateV1(
        owner=1,
        elixir_exact=5.0,
        hand=DECK[:4],
        deck=DECK,
        private_state_visible=True,
    )
    with pytest.raises(ContractError, match="opponent private state"):
        ObservationV1(tier="fair", tick=1, owner=0, players=(opponent,))
    with pytest.raises(ContractError, match="native_digest"):
        ObservationV1(tier="human", tick=1, owner=0, native_digest="secret")


def test_zero_sum_terminal_and_latency_episode_contract() -> None:
    terminal = TerminalV1(ended=True, winner=1, result_by_owner=(-1, 1), terminal_tick=100)
    assert terminal.result_by_owner[1] == 1
    with pytest.raises(ContractError, match="zero-sum"):
        TerminalV1(ended=True, result_by_owner=(1, 1), terminal_tick=100)

    latency = LatencyConfigV1(
        observation_delay_ms=(120, 300),
        network_jitter_ms=(0, 40),
        execution_delay_ms=(50, 100),
        frame_drop_probability=0.05,
        min_reaction_ms=120,
    )
    episode = EpisodeConfigV1(
        ruleset_id="r" * 64,
        deck0=DECK,
        deck1=tuple(reversed(DECK)),
        seed=7,
        latency=latency,
        policy_hashes={"0": "p0", "1": "p1"},
    )
    assert EpisodeConfigV1.from_mapping(episode.to_dict()) == episode
    with pytest.raises(ContractError, match="duplicate"):
        EpisodeConfigV1(
            ruleset_id="r" * 64,
            deck0=(DECK[0],) * 8,
            deck1=DECK,
            seed=8,
        )
    with pytest.raises(ContractError):
        LatencyConfigV1(frame_drop_probability=1.1)


def test_opponent_belief_round_trip_covers_evolution_and_ability_hypotheses() -> None:
    belief = OpponentBeliefV1(
        opponent_owner=1,
        deck_probabilities={str(DECK[0]): 1.0},
        elixir_probabilities=(0.0,) * 6 + (1.0,) + (0.0,) * 4,
        evolution_hypotheses={str(DECK[1]): {"ready_probability": 0.25}},
        ability_hypotheses={"ArcherQueenRapid": {"available_probability": 0.4}},
        updated_tick=50,
    )

    assert OpponentBeliefV1.from_mapping(belief.to_dict()) == belief
