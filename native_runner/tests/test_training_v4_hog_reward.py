from __future__ import annotations

from types import SimpleNamespace

import pytest

from native_runner.contracts import ActionKind
from native_runner.training.v4.async_cluster_self_play import (
    _credit_hog_deployments_v4,
)


def event(action_id="hog-1", *, owner=0, card_id=26000021,
          event_type="action_executed", kind=None, tick=400):
    return SimpleNamespace(
        event_type=event_type, owner=owner, card_id=card_id, tick=tick,
        data={"action_id": action_id,
              "kind": ActionKind.PLAY_CARD.value if kind is None else kind},
    )


def credit(events, *, owner=0, seen=None, accrued=0.0, coefficient=0.01, cap=0.1):
    return _credit_hog_deployments_v4(
        events, owner=owner, seen=set() if seen is None else seen,
        reward_per_deployment=coefficient, accrued_reward=accrued, episode_cap=cap,
    )


@pytest.mark.parametrize("owner", [0, 1])
def test_only_successful_own_hog_play_is_credited(owner):
    rows = [event(owner=owner), event("opponent", owner=1-owner),
            event("fireball", owner=owner, card_id=28000000),
            event("rejected", owner=owner, event_type="action_rejected"),
            event("requested", owner=owner, event_type="action_requested"),
            event("canceled", owner=owner, event_type="action_canceled_terminal"),
            event("ability", owner=owner, kind=ActionKind.ACTIVATE_ABILITY.value),
            event(None, owner=owner)]
    assert credit(rows, owner=owner) == (1, 0.01, 400)


def test_repeated_observation_and_segment_boundary_do_not_recredit():
    seen = set()
    assert credit([event(), event()], seen=seen) == (1, 0.01, 400)
    assert credit([event()], seen=seen, accrued=0.01) == (0, 0.0, None)
    assert credit([event(), event("hog-2", tick=800)], seen=seen, accrued=0.01) == (1, 0.01, 800)


def test_cap_counts_later_hogs_but_cannot_farm_extra_reward():
    seen = set()
    count, bonus, _ = credit([event(str(i)) for i in range(12)], seen=seen)
    assert count == 12
    assert bonus == pytest.approx(0.1)
    assert credit([event("later")], seen=seen, accrued=bonus) == (1, 0.0, 400)
    assert credit([event("partial")], accrued=0.095)[1] == pytest.approx(0.005)


def test_zero_coefficient_preserves_metrics_without_reward():
    assert credit([event()], coefficient=0.0) == (1, 0.0, 400)


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_invalid_reward_is_rejected(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        credit([], coefficient=value)


def test_remote_worker_forwards_reward_configuration(monkeypatch):
    from native_runner.training.v4 import remote_ppo_worker_server as server
    from native_runner.training.v4.async_cluster_self_play import AsyncResidentEngineSpecV4

    captured = []
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args: captured.append(args))
    request = dict(
        spec=object.__new__(AsyncResidentEngineSpecV4), assignments=(),
        gamma_per_decision=0.9997, shaping_beta=0.05,
        mutual_elixir_overflow_penalty=0.0025, mutual_elixir_overflow_grace=2.0,
        unilateral_elixir_overflow_penalty=0.00625, elixir_overflow_step_penalty_cap=0.1,
        policy_state_id="test", validate_tensors=False, max_decision_steps=1204,
        hog_deploy_reward=0.01, hog_deploy_reward_episode_cap=0.1,
    )
    server._run_worker(None, request, -1)
    assert captured[-1][-9:] == (
        0.01,
        0.1,
        None,
        0.0,
        10.0,
        30.0,
        0.0,
        0.0,
        0.0,
    )
    del request["hog_deploy_reward"]
    del request["hog_deploy_reward_episode_cap"]
    server._run_worker(None, request, -1)
    assert captured[-1][-9:] == (
        0.0,
        0.1,
        None,
        0.0,
        10.0,
        30.0,
        0.0,
        0.0,
        0.0,
    )
