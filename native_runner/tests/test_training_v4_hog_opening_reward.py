from types import SimpleNamespace

import pytest

from native_runner.contracts import ActionKind
from native_runner.training.v4 import async_cluster_self_play as collector
from native_runner.training.v4 import remote_ppo_worker_server as server


def event(tick, identity="hog", owner=0, **changes):
    fields = dict(tick=tick, event_type="action_executed", owner=owner,
                  card_id=26000021,
                  data={"action_id": identity, "kind": ActionKind.PLAY_CARD.value})
    fields.update(changes)
    return SimpleNamespace(**fields)


def credit(events, *, owner=0, seen=None, accrued=0.0, opening=0.03):
    return collector._credit_hog_deployments_v4(
        events, owner=owner, seen=set() if seen is None else seen,
        reward_per_deployment=0.0075, accrued_reward=accrued,
        episode_cap=0.15, opening_reward_per_deployment=opening,
    )


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("tick,expected", [
    (0, .0075), (199, .0075), (200, .03), (599, .03), (600, .0075),
    (5800, .0075),
])
def test_exact_window_both_seats_no_segment_or_overtime_reset(owner, tick, expected):
    count, bonus, first = credit([event(tick, owner=owner)], owner=owner)
    assert count == 1 and first == tick
    assert bonus == pytest.approx(expected)


def test_window_is_replacement_for_every_play_not_extra_or_first_only():
    count, bonus, first = credit([event(200, "a"), event(599, "b"), event(600, "c")])
    assert count == 3 and first == 200
    assert bonus == pytest.approx(.03 + .03 + .0075)


def test_executed_event_tick_and_cross_segment_deduplication():
    seen = set()
    # Delayed receipt of the event cannot change its executed timestamp.
    assert credit([event(599)], seen=seen) == (1, .03, 599)
    count, bonus, first = credit([event(599), event(800, "next")], seen=seen, accrued=.03)
    assert count == 1 and first == 800 and bonus == pytest.approx(.0075)
    assert credit([event(200)], seen=set()) == (1, .03, 200)


def test_shared_episode_cap_does_not_reset_after_window():
    seen = set()
    count, bonus, first = credit([event(210+i, str(i)) for i in range(6)], seen=seen)
    assert count == 6 and first == 210 and bonus == pytest.approx(.15)
    assert credit([event(900, "late")], seen=seen, accrued=bonus) == (1, 0., 900)
    assert credit([event(300)], accrued=.14)[1] == pytest.approx(.01)


def test_rejected_requested_other_card_opponent_and_ability_never_credited():
    events = [event(300, "opp", owner=1), event(300, "spell", card_id=28000000),
              event(300, "reject", event_type="action_rejected"),
              event(300, "request", event_type="action_requested"),
              event(300, "cancel", event_type="action_canceled_terminal"),
              event(300, "ability", data={"action_id":"ability", "kind":ActionKind.ACTIVATE_ABILITY.value}),
              event(300, data={"kind":ActionKind.PLAY_CARD.value})]
    assert credit(events) == (0, 0., None)


def test_disabled_window_and_zero_override_are_distinct():
    assert credit([event(300)], opening=None) == (1, .0075, 300)
    assert credit([event(300)], opening=0.) == (1, 0., 300)


@pytest.mark.parametrize("value", [-.01, float("nan"), float("inf")])
def test_invalid_opening_reward_rejected(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        credit([], opening=value)


def test_remote_worker_forwards_window_to_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args: calls.append(args))
    request = dict(spec=object.__new__(collector.AsyncResidentEngineSpecV4), assignments=(),
                   gamma_per_decision=.9997, shaping_beta=.15,
                   mutual_elixir_overflow_penalty=.0075, mutual_elixir_overflow_grace=2.,
                   unilateral_elixir_overflow_penalty=.01875, elixir_overflow_step_penalty_cap=.3,
                   policy_state_id="test", validate_tensors=False, max_decision_steps=1204,
                   hog_deploy_reward=.0075, hog_deploy_reward_episode_cap=.15,
                   hog_deploy_opening_reward=.03,
                   first_hog_timing_reward_max=0.,
                   first_hog_timing_start_seconds=10.,
                   first_hog_timing_deadline_seconds=30.,
                   first_hog_missed_deadline_penalty=0.,
                   first_hog_deferral_penalty=0.,
                   first_hog_deferral_penalty_episode_cap=0.)
    server._run_worker(None, request, -1)
    assert calls[-1][-9:] == (.0075, .15, .03, 0., 10., 30., 0., 0., 0.)
    del request["hog_deploy_opening_reward"]
    server._run_worker(None, request, -1)
    assert calls[-1][-9:] == (.0075, .15, None, 0., 10., 30., 0., 0., 0.)


@pytest.mark.parametrize("tick,expected", [
    (199, 0.0),
    (200, 0.05),
    (300, 0.0375),
    (400, 0.025),
    (500, 0.0125),
    (599, 0.000125),
    (600, 0.0),
])
def test_first_hog_timing_bonus_decays_continuously_once(tick, expected):
    assert collector._first_hog_timing_bonus_v4(
        first_hog_tick=tick,
        first_hog_was_already_deployed=False,
        maximum_bonus=.05,
        start_seconds=10.,
        deadline_seconds=30.,
    ) == pytest.approx(expected)
    assert collector._first_hog_timing_bonus_v4(
        first_hog_tick=tick,
        first_hog_was_already_deployed=True,
        maximum_bonus=.05,
        start_seconds=10.,
        deadline_seconds=30.,
    ) == 0.0


def test_first_hog_deadline_penalty_charges_exact_crossing_once():
    common = dict(
        before_tick=595,
        after_tick=600,
        deadline_seconds=30.,
        penalty=.03,
    )
    assert collector._first_hog_deadline_penalty_v4(
        **common, first_hog_tick=None
    ) == pytest.approx(.03)
    assert collector._first_hog_deadline_penalty_v4(
        **common, first_hog_tick=600
    ) == pytest.approx(.03)
    assert collector._first_hog_deadline_penalty_v4(
        **common, first_hog_tick=599
    ) == 0.0
    assert collector._first_hog_deadline_penalty_v4(
        before_tick=600,
        after_tick=605,
        first_hog_tick=None,
        deadline_seconds=30.,
        penalty=.03,
    ) == 0.0


def deferral(**changes):
    values = dict(
        tick=200,
        first_hog_tick=None,
        hog_playable=True,
        submitted_hog=False,
        own_elixir=10.,
        gate_wait=True,
        penalty_per_decision=.0005,
        accrued_penalty=0.,
        episode_cap=.02,
        start_seconds=10.,
        deadline_seconds=30.,
    )
    values.update(changes)
    return collector._first_hog_deferral_penalty_v4(**values)


def test_first_hog_deferral_targets_playable_hog_or_full_elixir_wait_only():
    assert deferral() == pytest.approx(.0005)
    assert deferral(submitted_hog=True) == 0.0
    assert deferral(tick=199) == 0.0
    assert deferral(tick=600) == 0.0
    assert deferral(first_hog_tick=200) == 0.0
    assert deferral(hog_playable=False, own_elixir=10., gate_wait=True) == pytest.approx(.0005)
    assert deferral(hog_playable=False, own_elixir=9.99, gate_wait=True) == 0.0
    assert deferral(hog_playable=False, own_elixir=10., gate_wait=False) == 0.0
    assert deferral(accrued_penalty=.0198) == pytest.approx(.0002)
    assert deferral(accrued_penalty=.02) == 0.0
