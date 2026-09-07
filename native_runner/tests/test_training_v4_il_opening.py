from __future__ import annotations

from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest

from native_runner.training.v4.async_cluster_self_play import (
    AsyncResidentEngineSpecV4,
    _elixir_overflow_penalty_transition_v4,
    _frozen_opponent_opening_force_owner_v4,
)
from native_runner.training.v4.matchmaking import POLICY_IL, POLICY_HISTORY


def opening(*, learner=0, tick=300, counts=(0, 0), policy=POLICY_IL,
            il_timeout=15, general_timeout=None):
    return _frozen_opponent_opening_force_owner_v4(
        current_owners=(learner,), active_actions=counts, tick=tick,
        after_seconds=general_timeout, policy_matchup=policy,
        il_after_seconds=il_timeout,
    )


@pytest.mark.parametrize("learner", [0, 1])
def test_il_timeout_boundary_and_owner_symmetry(learner):
    assert opening(learner=learner, tick=299) is None
    assert opening(learner=learner, tick=300) == 1 - learner
    assert opening(learner=learner, tick=301) == 1 - learner


@pytest.mark.parametrize("learner", [0, 1])
def test_il_still_forced_when_learner_already_opened(learner):
    counts = [0, 0]
    counts[learner] = 3
    assert opening(learner=learner, counts=tuple(counts)) == 1 - learner


@pytest.mark.parametrize("learner", [0, 1])
def test_il_first_action_permanently_disables_force(learner):
    counts = [0, 0]
    counts[1 - learner] = 1
    for tick in (300, 600, 3600):
        assert opening(learner=learner, tick=tick, counts=tuple(counts)) is None


def test_il_only_timeout_does_not_force_fixed_u460_or_self_play():
    assert opening(policy=POLICY_HISTORY, tick=3600) is None
    assert _frozen_opponent_opening_force_owner_v4(
        current_owners=(0, 1), active_actions=(0, 0), tick=3600,
        after_seconds=None, policy_matchup="current-current", il_after_seconds=15,
    ) is None


def test_disabled_default_and_legacy_eval_timeout_are_preserved():
    assert opening(il_timeout=None, tick=3600) is None
    assert opening(policy=POLICY_HISTORY, il_timeout=None, general_timeout=15) == 1


@pytest.mark.parametrize("timeout", [0, -1])
def test_invalid_il_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="IL opening timeout"):
        opening(il_timeout=timeout)


def test_il_timeout_survives_segment_spec_and_remote_worker_transport(monkeypatch):
    from native_runner.training.v4 import remote_ppo_worker_server as server

    spec = AsyncResidentEngineSpecV4(
        engine_index=0, cpu=0, force_il_opponent_opening_after_seconds=15,
    )
    spec = pickle.loads(pickle.dumps(replace(spec, segment_decision_steps=160)))
    assert spec.force_frozen_opponent_opening_after_seconds is None
    assert spec.force_il_opponent_opening_after_seconds == 15
    captured = []
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args: captured.append(args))
    request = dict(
        spec=spec, assignments=(), gamma_per_decision=0.9997, shaping_beta=0.15,
        mutual_elixir_overflow_penalty=0.0075, mutual_elixir_overflow_grace=2.0,
        unilateral_elixir_overflow_penalty=0.01875, elixir_overflow_step_penalty_cap=0.3,
        policy_state_id="test", validate_tensors=False, max_decision_steps=1204,
    )
    server._run_worker(SimpleNamespace(), request, -1)
    assert captured[0][1].force_il_opponent_opening_after_seconds == 15


def test_new_overflow_step_cap_is_shared_by_both_components():
    result = _elixir_overflow_penalty_transition_v4(
        personal_wasted_elixir=100.0, personal_accrued_cost=0.0,
        unilateral_wasted_elixir=100.0, unilateral_accrued_cost=0.0,
        before_elixir=10.0, after_elixir=10.0, generated_elixir=0.5,
        both_full=False, personal_coefficient=0.0075, personal_grace_elixir=2.0,
        unilateral_coefficient=0.01875, step_penalty_cap=0.3,
    )
    assert sum(result[4:]) == pytest.approx(0.3)
