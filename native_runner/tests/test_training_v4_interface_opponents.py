"""Deployment inference parity and training opening interventions."""

from dataclasses import fields

import pytest
import torch

from native_runner.tests.test_training_v4_head_to_head import _done_report, run_fake_worker
from native_runner.tests.test_training_v4_model import _batch
from native_runner.tests.test_training_v4_train_ppo_strong_v1 import _fixture_model, _il_checkpoint, _parse
from native_runner.training.v4 import policy_session, train_ppo_self_play_cluster as trainer
from native_runner.training.v4.checkpoint import save_actor_critic_checkpoint
from native_runner.training.v4.league import LeagueEntryV4
from native_runner.training.v4.matchmaking import POLICY_CURRENT, POLICY_HISTORY, POLICY_IL
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT


@pytest.fixture
def interface_builder(monkeypatch):
    def build(*, device=None, config=None):
        model = _fixture_model(config)
        return model if device is None else model.to(device)

    monkeypatch.setattr(policy_session, "build_production_model_v4", build)
    monkeypatch.setattr(trainer, "build_production_model_v4", build)


def test_opponent_matches_interface_actions_and_keeps_raw_gate(tmp_path, interface_builder):
    path, source = _il_checkpoint(tmp_path)
    with torch.no_grad():
        for parameter in source.gate_head.parameters():
            parameter.zero_()
        source.gate_head[-1].bias[GATE_ACT] = 0.25
    source.set_ppo_gate_temperature(0.6)
    source.set_ppo_action_temperature(1.3)
    source.set_ppo_continue_temperature(2.2)
    save_actor_critic_checkpoint(
        path, source, training_stage="imitation", gamma_per_decision=0.99, extra={"gate_act_weight": 8.0}
    )
    opponent = trainer._load_interface_opponent_model(path, device=torch.device("cpu")).model
    ui = policy_session.load_policy_v4(path, device="cpu", delay_policy="earliest").model
    reference, _ = trainer._load_frozen_model(
        path, device=torch.device("cpu"), delay_policy="earliest", gate_logit_offset=-1.25,
        gate_temperature=1.0, continue_temperature=1.0,
    )
    assert opponent is not reference
    assert not opponent.ppo_rollout_sampling
    assert not opponent.training and not any(p.requires_grad for p in opponent.parameters())
    assert (opponent.ppo_gate_temperature, opponent.ppo_action_temperature, opponent.ppo_continue_temperature) == (
        0.6, 1.3, 2.2
    )
    assert torch.equal(opponent.gate_head[-1].bias, source.gate_head[-1].bias)
    assert reference.gate_head[-1].bias[GATE_ACT].item() == pytest.approx(-1.0)
    batch = _batch(1)
    with torch.inference_mode():
        actual = opponent.act(batch, opponent.initial_state(1))
        expected = ui.act(batch, ui.initial_state(1))
        calibrated = reference.act(batch, reference.initial_state(1))
    assert actual.actions.gate.tolist() == [GATE_ACT]
    assert calibrated.actions.gate.tolist() == [GATE_WAIT]
    for field in fields(actual.actions):
        assert torch.equal(getattr(actual.actions, field.name), getattr(expected.actions, field.name)), field.name


def test_history_and_fixed_opponents_keep_baked_ppo_weights(tmp_path, interface_builder):
    from types import SimpleNamespace

    path, source = _il_checkpoint(tmp_path)
    with torch.no_grad():
        source.gate_head[-1].bias[GATE_ACT] -= 1.25
    checkpoint_id = save_actor_critic_checkpoint(
        path, source, training_stage="ppo-league-snapshot", update_step=10, gamma_per_decision=0.99,
        extra={"gate_logit_offset_applied": -1.25},
    )
    # Fixed opponents and PFSP history share this assignment/loading path.
    entry = LeagueEntryV4(checkpoint_id=checkpoint_id, checkpoint_path=str(path), update_step=10)
    cache = {}
    models = trainer._load_history_models((SimpleNamespace(opponent=entry),), cache, device=torch.device("cpu"))
    opponent = models[entry.checkpoint_id]
    assert torch.equal(opponent.gate_head[-1].bias, source.gate_head[-1].bias)
    assert not opponent.ppo_rollout_sampling and opponent.delay_policy == "earliest"
    assert trainer._load_history_models(
        (SimpleNamespace(opponent=entry),), cache, device=torch.device("cpu")
    )[entry.checkpoint_id] is opponent


@pytest.mark.parametrize("policy", [POLICY_IL, POLICY_HISTORY, POLICY_CURRENT])
@pytest.mark.parametrize("opened_owner", [0, 1])
def test_training_forces_idle_owner_even_after_other_owner_opens(monkeypatch, policy, opened_owner):
    options = trainer._training_opening_force_options(_parse())
    connection, _ = run_fake_worker(
        monkeypatch, policy_matchup=policy, spec_overrides=options,
        acts=lambda tick, owner: owner == opened_owner and tick == 130,
        deck_average_elixir=(2.9, 3.8),
    )
    idle_owner = 1 - opened_owner
    assert [(tick, rows) for tick, rows in connection.frames if rows] == [(300, (idle_owner,))]
    report = _done_report(connection)
    assert report["opening_forced_owner"] == idle_owner and report["opening_forced_tick"] == 300
    expected_ticks = [None, None]
    expected_ticks[idle_owner] = 300
    assert report["opening_forced_ticks_by_owner"] == tuple(expected_ticks)


@pytest.mark.parametrize("policy", [POLICY_IL, POLICY_HISTORY, POLICY_CURRENT])
def test_training_forces_both_idle_owners_once_in_every_matchup(monkeypatch, policy):
    connection, _ = run_fake_worker(
        monkeypatch, policy_matchup=policy,
        spec_overrides=trainer._training_opening_force_options(_parse()),
    )
    assert [(tick, rows) for tick, rows in connection.frames if rows] == [(300, (0, 1))]
    assert _done_report(connection)["opening_forced_ticks_by_owner"] == (300, 300)


@pytest.mark.parametrize("policy", [POLICY_IL, POLICY_HISTORY, POLICY_CURRENT])
def test_training_does_not_force_owners_who_already_opened(monkeypatch, policy):
    connection, _ = run_fake_worker(
        monkeypatch, policy_matchup=policy,
        spec_overrides=trainer._training_opening_force_options(_parse()),
        acts=lambda tick, owner: tick == 130,
    )
    assert not any(rows for _, rows in connection.frames)


@pytest.mark.parametrize("policy", [POLICY_IL, POLICY_HISTORY, POLICY_CURRENT])
def test_both_owner_opening_deadlines_survive_segment_boundaries(monkeypatch, policy):
    options = {**trainer._training_opening_force_options(_parse()), "segment_decision_steps": 20}
    connection, _ = run_fake_worker(monkeypatch, policy_matchup=policy, spec_overrides=options)
    assert any(packet.get("segment_boundary") for packet in connection.sent)
    assert [(tick, rows) for tick, rows in connection.frames if rows] == [(300, (0, 1))]
    assert _done_report(connection)["opening_forced_ticks_by_owner"] == (300, 300)


@pytest.mark.parametrize("timeout", [0, -1])
def test_training_rejects_disabled_or_negative_opening_timeout(timeout):
    with pytest.raises(ValueError, match="training opening timeout"):
        trainer._training_opening_force_options(_parse("--opening-force-after-seconds", str(timeout)))


def test_training_opening_options_survive_remote_worker_transport(monkeypatch):
    import pickle
    from dataclasses import replace
    from types import SimpleNamespace

    from native_runner.training.v4 import remote_ppo_worker_server as server
    from native_runner.training.v4.async_cluster_self_play import AsyncResidentEngineSpecV4

    spec = AsyncResidentEngineSpecV4(engine_index=0, cpu=0, **trainer._training_opening_force_options(_parse()))
    spec = pickle.loads(pickle.dumps(replace(spec, segment_decision_steps=160)))
    captured = []
    monkeypatch.setattr(server, "_prepare_worker_process", lambda _: None)
    monkeypatch.setattr(server, "_engine_worker", lambda *args: captured.append(args))
    server._run_worker(SimpleNamespace(), {
        "spec": spec, "assignments": (), "gamma_per_decision": 0.9997, "shaping_beta": 0.05,
        "mutual_elixir_overflow_penalty": 0.0, "mutual_elixir_overflow_grace": 4.0,
        "unilateral_elixir_overflow_penalty": 0.0, "elixir_overflow_step_penalty_cap": 0.1,
        "policy_state_id": "test", "validate_tensors": False, "max_decision_steps": 1204,
    }, -1)
    forwarded = captured[0][1]
    assert forwarded.force_all_owners_opening_after_seconds == 15
