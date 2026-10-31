from types import SimpleNamespace as NS

import pytest

from native_runner.training.v4.async_cluster_self_play import _scaled_terminal_rewards_v4
from native_runner.training.v4.train_ppo_self_play_cluster import build_argument_parser


@pytest.mark.parametrize("result", [(1., -1.), (-1., 1.), (0., 0.)])
def test_terminal_scale_changes_outcome_without_scaling_shaping(result):
    shaping = (.37, -.37)
    incoming = {i: result[i] + shaping[i] for i in (0, 1)}
    actual = _scaled_terminal_rewards_v4(incoming, NS(ended=True, result_by_owner=result), 2.)
    assert actual == pytest.approx({i: 2. * result[i] + shaping[i] for i in (0, 1)})


def test_nonterminal_shaping_is_unchanged():
    assert _scaled_terminal_rewards_v4({0: .4, 1: -.4}, NS(ended=False), 2.) == {0: .4, 1: -.4}


def test_terminal_scale_is_forwarded_through_remote_worker(monkeypatch):
    from native_runner.training.v4 import remote_ppo_worker_server as server
    from native_runner.training.v4.async_cluster_self_play import AsyncResidentEngineSpecV4
    got = {}
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args, **kwargs: got.update(kwargs))
    server._run_worker(None, dict(spec=object.__new__(AsyncResidentEngineSpecV4), assignments=(),
        gamma_per_decision=.9997, shaping_beta=.8, mutual_elixir_overflow_penalty=.015,
        mutual_elixir_overflow_grace=4., unilateral_elixir_overflow_penalty=.075,
        elixir_overflow_step_penalty_cap=.1, policy_state_id="test", validate_tensors=False,
        max_decision_steps=1204, terminal_reward_scale=2.), -1)
    assert got == {"terminal_reward_scale": 2.}


def test_terminal_scale_cli_is_explicit_and_legacy_default_preserved():
    parser = build_argument_parser()
    common = ["--deck-pool", "pool.json", "--checkpoint", "il.pt", "--engine-ready", "ready.json",
              "--output-dir", "out"]
    assert parser.parse_args(common).terminal_reward_scale == 1.
    assert parser.parse_args([*common, "--terminal-reward-scale", "2"]).terminal_reward_scale == 2.
