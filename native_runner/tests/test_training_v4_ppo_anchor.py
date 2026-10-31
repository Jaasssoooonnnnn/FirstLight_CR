"""IL anchor KL, critic warmup and the gate ACT correction."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4
from native_runner.training.v4.ppo_anchor import (
    GATE_LOGIT_OFFSET_EXTRA_KEY,
    AdaptiveKLControllerV4,
    AnchorKLConfigV4,
    anchor_kl_terms,
    apply_gate_logit_offset_once,
    binary_kl_from_selected_log_prob,
    resolve_gate_logit_offset,
    sampled_kl_k3,
    value_only_parameter_names,
)
from native_runner.training.v4.tensors import (
    GATE_ACT,
    GATE_WAIT,
    concatenate_padded_tensor_records,
    concatenate_tensor_records,
)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def test_binary_kl_matches_torch_and_is_zero_for_identical_heads() -> None:
    torch.manual_seed(0)
    p = torch.rand(64) * 0.98 + 0.01
    q = torch.rand(64) * 0.98 + 0.01
    selected = torch.randint(0, 2, (64,))
    learner = torch.where(selected == 1, p, 1 - p).log()
    reference = torch.where(selected == 1, q, 1 - q).log()
    expected = torch.distributions.kl_divergence(
        torch.distributions.Bernoulli(probs=p), torch.distributions.Bernoulli(probs=q)
    )
    torch.testing.assert_close(binary_kl_from_selected_log_prob(learner, reference), expected, rtol=1e-4, atol=1e-6)
    assert torch.count_nonzero(binary_kl_from_selected_log_prob(learner, learner)) == 0
    # The other class is masked: p(a) = 1 under both policies, KL exactly 0.
    zero = torch.zeros(3)
    assert torch.equal(binary_kl_from_selected_log_prob(zero, zero), torch.zeros(3))


def test_binary_kl_gradient_flows_only_through_the_learner() -> None:
    learner = torch.tensor([-0.3, -1.2], requires_grad=True)
    reference = torch.tensor([-0.7, -0.4], requires_grad=True)
    binary_kl_from_selected_log_prob(learner, reference).sum().backward()
    assert learner.grad is not None and bool(torch.all(learner.grad != 0))
    assert reference.grad is None
    same = torch.tensor([-0.5], requires_grad=True)
    binary_kl_from_selected_log_prob(same, same.detach()).sum().backward()
    assert same.grad is not None and abs(float(same.grad)) < 1e-6


def test_k3_is_nonnegative_and_zero_when_equal() -> None:
    torch.manual_seed(1)
    learner = -torch.rand(100) * 5
    reference = -torch.rand(100) * 5
    values = sampled_kl_k3(learner, reference)
    assert bool(torch.all(values >= 0))
    assert torch.count_nonzero(sampled_kl_k3(learner, learner)) == 0


def _components(gate: float, candidate: float, target: float, delay: float, cont: float) -> torch.Tensor:
    return torch.tensor([[[gate, candidate, target, delay, cont]]])


def test_anchor_terms_only_continue_moves_when_only_continue_differs() -> None:
    gate = torch.tensor([[GATE_ACT]])
    micro = torch.tensor([[1]])
    learner = _components(-0.2, -1.0, -3.0, -0.5, -0.1)
    same = anchor_kl_terms(learner, learner.clone(), gate=gate, micro_action_count=micro)
    for name in ("gate", "candidate", "target", "continue"):
        assert float(same[name]) == 0.0
    continued = anchor_kl_terms(
        learner, _components(-0.2, -1.0, -3.0, -0.5, -0.9), gate=gate, micro_action_count=micro
    )
    assert float(continued["continue"]) > 0.0
    for name in ("gate", "candidate", "target"):
        assert float(continued[name]) == 0.0
    # The delay column never enters the KL.
    delayed = anchor_kl_terms(learner, _components(-0.2, -1.0, -3.0, -4.0, -0.1), gate=gate, micro_action_count=micro)
    for name in ("gate", "candidate", "target", "continue"):
        assert float(delayed[name]) == 0.0
    # P(SECOND) recovered from the selected STOP action.
    assert float(continued["learner_second_prob"]) == pytest.approx(1 - math.exp(-0.1), rel=1e-5)
    assert float(continued["reference_second_prob"]) == pytest.approx(1 - math.exp(-0.9), rel=1e-5)


# ---------------------------------------------------------------------------
# Adaptive coefficient
# ---------------------------------------------------------------------------


def test_adaptive_controller_moves_beta_toward_the_target() -> None:
    config = AnchorKLConfigV4(initial_coefficient=0.05, target_kl=0.02, initial_continue_coefficient=0.25)
    controller = AdaptiveKLControllerV4(config)
    assert controller.continue_value == pytest.approx(0.25)
    controller.update(0.2)  # far above target: +20% / horizon
    assert controller.value == pytest.approx(0.05 * (1 + 0.2 / 4.0))
    assert controller.continue_value == pytest.approx(5 * controller.value)  # tied
    up = controller.value
    controller.update(0.0)  # below target: decrease
    assert controller.value < up
    controller.update(0.02)  # on target: unchanged
    assert controller.value == pytest.approx(up * (1 - 0.2 / 4.0))
    for _ in range(1000):
        controller.update(0.0)
    assert controller.value == pytest.approx(config.minimum_coefficient)
    fixed = AdaptiveKLControllerV4(
        AnchorKLConfigV4(initial_coefficient=0.05, initial_continue_coefficient=0.25, continue_mode="fixed")
    )
    fixed.update(1.0)
    assert fixed.value > 0.05 and fixed.continue_value == pytest.approx(0.25)


def test_adaptive_controller_never_lowers_beta_during_the_no_decrease_window() -> None:
    controller = AdaptiveKLControllerV4(AnchorKLConfigV4(no_decrease_updates=3))
    for _ in range(3):
        controller.update(0.0)  # KL far below target: would lower beta, held instead
    assert controller.value == pytest.approx(0.05)
    controller.update(0.0)  # fourth update: the window is over
    assert controller.value == pytest.approx(0.05 * (1 - 0.2 / 4.0))
    rising = AdaptiveKLControllerV4(AnchorKLConfigV4(no_decrease_updates=3))
    rising.update(1.0)  # raising is allowed inside the window
    assert rising.value == pytest.approx(0.05 * (1 + 0.2 / 4.0))
    with pytest.raises(ValueError, match="no-decrease"):
        AnchorKLConfigV4(no_decrease_updates=-1)


def test_adaptive_controller_state_round_trip_requires_the_same_config() -> None:
    config = AnchorKLConfigV4()
    controller = AdaptiveKLControllerV4(config)
    controller.update(1.0)
    restored = AdaptiveKLControllerV4(config)
    assert restored.load_state_dict(controller.state_dict())
    assert restored.value == controller.value and restored.updates == 1
    changed = AdaptiveKLControllerV4(AnchorKLConfigV4(initial_coefficient=0.1, initial_continue_coefficient=0.5))
    assert not changed.load_state_dict(controller.state_dict())
    assert changed.value == 0.1


# ---------------------------------------------------------------------------
# Gate ACT correction
# ---------------------------------------------------------------------------


def test_gate_offset_is_applied_once_and_never_to_ppo_checkpoints() -> None:
    il_payload = {"extra": {"gate_act_weight": 8.0}}
    assert resolve_gate_logit_offset("auto", il_payload) == pytest.approx(-math.log(8.0))
    assert resolve_gate_logit_offset("auto", {"extra": {}}) == 0.0
    model = _model()
    before = model.gate_head[-1].bias.detach().clone()
    applied = apply_gate_logit_offset_once(model, il_payload, "auto")
    after = model.gate_head[-1].bias.detach()
    assert applied == pytest.approx(-math.log(8.0))
    assert float(after[GATE_ACT] - before[GATE_ACT]) == pytest.approx(-math.log(8.0), rel=1e-6)
    assert float(after[GATE_WAIT]) == float(before[GATE_WAIT])
    with pytest.raises(RuntimeError, match="already applied"):
        apply_gate_logit_offset_once(model, il_payload, "auto")
    snapshot = _model()
    snapshot_before = snapshot.gate_head[-1].bias.detach().clone()
    assert apply_gate_logit_offset_once(
        snapshot, {"extra": {GATE_LOGIT_OFFSET_EXTRA_KEY: -math.log(8.0), "gate_act_weight": 8.0}}, "auto"
    ) == pytest.approx(-math.log(8.0))
    assert torch.equal(snapshot.gate_head[-1].bias.detach(), snapshot_before)


# ---------------------------------------------------------------------------
# distributed_ppo_update_v4 with the reference model
# ---------------------------------------------------------------------------


def _rollout(model, *, steps: int = 4, lanes: int = 4, first_start: bool = True) -> StoredRolloutV4:
    """Sample a real trajectory from ``model`` (T=1) and score it as the behavior policy."""

    batch = _batch(batch_size=lanes)
    starts = torch.zeros(steps, lanes, dtype=torch.bool)
    starts[0] = first_start
    initial_state = model.initial_state(lanes)
    generator_state = torch.get_rng_state()
    torch.manual_seed(1234)
    actions = []
    state = initial_state
    with torch.no_grad():
        for step in range(steps):
            output = model.sample_for_ppo_rollout(batch, state, episode_start=starts[step])
            actions.append(output.actions)
            state = output.next_state
    torch.set_rng_state(generator_state)
    actions = tuple(actions)
    assert any(bool((action.gate == GATE_ACT).any()) for action in actions)
    observations = tuple(batch for _ in range(steps))
    with torch.no_grad():
        behavior = evaluate_recurrent_sequence(model, observations, actions, starts, initial_state=initial_state)
    return StoredRolloutV4(
        observations=observations,
        actions=actions,
        episode_start=starts,
        valid_mask=torch.ones(steps, lanes, dtype=torch.bool),
        behavior_log_prob=behavior.log_prob,
        behavior_value=behavior.value,
        next_value=torch.zeros(steps, lanes),
        rewards=torch.ones(steps, lanes),
        terminated=torch.zeros(steps, lanes, dtype=torch.bool),
        truncated=torch.zeros(steps, lanes, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
        policy_state_id="anchor-fixture",
    )


def _clone(model):
    clone = _model().eval()
    clone.load_state_dict(model.state_dict())
    for name in ("ppo_gate_temperature", "ppo_action_temperature", "ppo_continue_temperature"):
        getattr(clone, f"set_{name}")(getattr(model, name))
    clone.set_delay_policy(model.delay_policy)
    return clone


def _learner_and_reference():
    torch.manual_seed(7)
    model = _model().eval()
    model.set_ppo_gate_temperature(1.0)
    model.set_ppo_action_temperature(1.0)
    model.set_ppo_continue_temperature(1.0)
    with torch.no_grad():  # ACT and SECOND both likely, so every head is exercised.
        model.gate_head[-1].bias.copy_(torch.tensor([0.0, 1.0]))
        model.continue_head[-1].bias.copy_(torch.tensor([0.0, 0.0]))
    reference = _clone(model)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return model, reference


def _update(model, reference, rollout, *, chunk: int = 3, lr: float = 0.0, **kwargs):
    return distributed_ppo_update_v4(
        model,
        torch.optim.AdamW(model.parameters(), lr=lr),
        rollout,
        epochs=1,
        lane_minibatch_size=rollout.batch_size,
        sequence_chunk_steps=chunk,
        config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None),
        autocast_dtype=None,
        reference_model=reference,
        anchor_kl_coefficient=0.05,
        anchor_continue_kl_coefficient=0.25,
        **kwargs,
    )


def test_identical_reference_gives_zero_kl_and_no_extra_gradient() -> None:
    model, reference = _learner_and_reference()
    rollout = _rollout(model)
    row = _update(model, reference, rollout)[0]
    for key in ("anchor_kl_gate", "anchor_kl_candidate", "anchor_kl_target", "anchor_kl_continue", "anchor_kl"):
        assert row[key] == pytest.approx(0.0, abs=1e-6)
    assert row["anchor_kl_coefficient"] == 0.05 and row["anchor_continue_kl_coefficient"] == 0.25
    # Same gradients with and without the anchor (the KL gradient vanishes at pi == pi_ref).
    gradients = []
    for with_anchor in (True, False):
        learner = _clone(model)
        optimizer = torch.optim.SGD(learner.parameters(), lr=0.0)
        distributed_ppo_update_v4(
            learner,
            optimizer,
            rollout,
            epochs=1,
            lane_minibatch_size=4,
            sequence_chunk_steps=3,
            config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None, max_grad_norm=None),
            autocast_dtype=None,
            reference_model=reference if with_anchor else None,
            anchor_kl_coefficient=0.05 if with_anchor else 0.0,
            anchor_continue_kl_coefficient=0.25 if with_anchor else 0.0,
        )
        gradients.append(learner.gate_head[-1].bias.grad.detach().clone())
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-5, atol=1e-7)


def test_changing_only_the_reference_continue_head_moves_only_continue_kl() -> None:
    model, reference = _learner_and_reference()
    with torch.no_grad():
        reference.continue_head[-1].bias.add_(torch.tensor([0.0, 2.0]))
    row = _update(model, reference, _rollout(model))[0]
    assert row["anchor_kl_continue"] > 1e-4
    for key in ("anchor_kl_gate", "anchor_kl_candidate", "anchor_kl_target"):
        assert row[key] == pytest.approx(0.0, abs=1e-6)
    assert row["anchor_reference_second_prob"] > row["anchor_learner_second_prob"]


def test_reference_delay_head_never_enters_the_kl() -> None:
    model, reference = _learner_and_reference()
    with torch.no_grad():
        reference.delay_offset_head[-1].bias.add_(torch.linspace(-3.0, 3.0, reference.delay_offset_head[-1].bias.numel()))
    row = _update(model, reference, _rollout(model))[0]
    for key in ("anchor_kl_gate", "anchor_kl_candidate", "anchor_kl_target", "anchor_kl_continue"):
        assert row[key] == pytest.approx(0.0, abs=1e-6)


def test_reference_runs_from_a_zero_state_and_carries_it_across_chunks() -> None:
    model, reference = _learner_and_reference()
    with torch.no_grad():
        reference.gate_head[-1].bias.add_(torch.tensor([0.0, 1.5]))
        reference.lstm_core.weight_hh.mul_(1.7)  # makes the result depend on the carried state
    rollout = _rollout(model)
    whole = _update(model, reference, rollout, chunk=3)[0]
    chunked = _update(model, reference, rollout, chunk=1)[0]
    assert whole["anchor_kl_gate"] > 1e-4
    for key in ("anchor_kl_gate", "anchor_kl_candidate", "anchor_kl_target", "anchor_kl_continue"):
        assert chunked[key] == pytest.approx(whole[key], rel=1e-4, abs=1e-7)
    # Direct computation from zero LSTM states over the same trajectory.
    flat_observations = concatenate_padded_tensor_records(rollout.observations).to_model_input("cpu")
    flat_actions = concatenate_tensor_records(rollout.actions)
    terms = []
    with torch.no_grad():
        for policy in (model, reference):
            evaluated = policy.evaluate_encoded_action_sequence(
                flat_observations,
                policy.encode_observation(flat_observations),
                policy.initial_state(rollout.batch_size),
                flat_actions,
                rollout.episode_start,
                time_steps=rollout.time_steps,
                gate_temperature=1.0,
                action_temperature=1.0,
                continue_temperature=1.0,
                return_log_prob_components=True,
            )
            terms.append(evaluated[4])
    expected = anchor_kl_terms(
        terms[0],
        terms[1],
        gate=flat_actions.gate.reshape(rollout.time_steps, rollout.batch_size),
        micro_action_count=flat_actions.micro_action_count.reshape(rollout.time_steps, rollout.batch_size),
    )
    assert whole["anchor_kl_gate"] == pytest.approx(float(expected["gate"].mean()), rel=1e-4)


def test_anchor_requires_reference_state_for_continuations_and_accepts_temperatures() -> None:
    model, reference = _learner_and_reference()
    with pytest.raises(ValueError, match="reference initial state"):
        _update(model, reference, _rollout(model, first_start=False))
    hot = _rollout(model)
    hot = replace(hot, continue_temperature=2.0)
    model.set_ppo_continue_temperature(2.0)
    assert _update(model, reference, hot)[0]["anchor_kl"] == pytest.approx(0.0, abs=1e-6)
    model.set_ppo_continue_temperature(1.0)
    reference.set_delay_policy("earliest")
    with pytest.raises(ValueError, match="delay policies differ"):
        _update(model, reference, _rollout(model))


def test_kl_loss_pulls_the_learner_toward_the_reference() -> None:
    model, reference = _learner_and_reference()
    with torch.no_grad():
        reference.gate_head[-1].bias.add_(torch.tensor([0.0, 2.0]))
    rollout = _rollout(model)
    before = _update(model, reference, rollout)[0]["anchor_kl_gate"]
    for _ in range(5):
        distributed_ppo_update_v4(
            model,
            torch.optim.SGD(model.parameters(), lr=0.5),
            rollout,
            epochs=1,
            lane_minibatch_size=4,
            sequence_chunk_steps=3,
            config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None, value_coefficient=0.0, clip_range=1e-9),
            autocast_dtype=None,
            reference_model=reference,
            anchor_kl_coefficient=1.0,
            anchor_continue_kl_coefficient=0.0,
        )
    after = _update(model, reference, rollout)[0]["anchor_kl_gate"]
    assert after < before


# ---------------------------------------------------------------------------
# Critic warmup
# ---------------------------------------------------------------------------


def test_value_only_update_freezes_the_actor_and_restores_requires_grad() -> None:
    model, reference = _learner_and_reference()
    rollout = _rollout(model)
    value_names = set(value_only_parameter_names(model))
    assert any(name.startswith("value_head") for name in value_names)
    assert not any(name.startswith(("gate_head", "lstm_core", "continue_head")) for name in value_names)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    rows = _update(model, reference, rollout, lr=1e-2, value_only=True)
    assert rows[0]["value_only"] == 1.0 and rows[0]["anchor_loss"] == 0.0
    changed = {name for name, parameter in model.named_parameters() if not torch.equal(parameter.detach(), before[name])}
    assert changed and changed <= value_names
    assert all(parameter.requires_grad for parameter in model.parameters())
    # The actor did not move: the rollout's own log-probs are reproduced (approx_kl ~ 0).
    assert _update(model, reference, rollout)[0]["approx_kl"] == pytest.approx(0.0, abs=1e-6)


def test_activation_checkpointing_keeps_the_ppo_anchor_update_exact() -> None:
    model, reference = _learner_and_reference()
    with torch.no_grad():
        reference.gate_head[-1].bias.add_(torch.tensor([0.0, 1.0]))
    rollout = _rollout(model)
    results = []
    for enabled in (False, True):
        learner = _clone(model)
        learner.set_activation_checkpointing(enabled)
        row = _update(learner, reference, rollout, lr=0.0)[0]
        results.append((row, learner.gate_head[-1].bias.grad.detach().clone()))
    for key in ("policy", "value", "anchor_kl_gate", "anchor_kl_continue"):
        assert results[1][0][key] == pytest.approx(results[0][0][key], rel=1e-5, abs=1e-7)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Training mode for the cuDNN LSTM backward
# ---------------------------------------------------------------------------


def test_ppo_update_runs_the_lstm_backward_in_training_mode_and_restores_eval(monkeypatch) -> None:
    model, reference = _learner_and_reference()
    rollout = _rollout(model)
    model.eval()  # the trainer hands over the rollout model in eval() mode
    seen = []
    original = model.evaluate_encoded_action_sequence

    def recording(*args, **kwargs):
        seen.append(model.training)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "evaluate_encoded_action_sequence", recording)
    before = model.lstm_core.weight_hh.detach().clone()
    _update(model, reference, rollout, lr=1e-3)
    assert seen and all(seen)  # every learner forward/backward ran with training=True
    assert not reference.training  # the no-grad reference stays in eval()
    assert not model.training  # restored for the next rollout
    assert not torch.equal(model.lstm_core.weight_hh.detach(), before)  # the LSTM backward really ran


def test_train_mode_cannot_change_log_probs() -> None:
    model = _model()
    assert not any(isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for module in model.modules())
    # The fused PPO path calls torch._VF.lstm with dropout 0.0 (no train-time noise).
    assert float(getattr(model.lstm_core, "dropout", 0.0)) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuDNN RNN backward needs CUDA")
def test_cuda_ppo_update_from_an_eval_model_does_not_hit_the_cudnn_training_check() -> None:
    model, reference = _learner_and_reference()
    rollout = _rollout(model)
    device = torch.device("cuda")
    model.to(device).eval()
    reference.to(device)
    rows = distributed_ppo_update_v4(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        rollout,
        epochs=1,
        lane_minibatch_size=rollout.batch_size,
        sequence_chunk_steps=2,
        config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None),
        reference_model=reference,
        anchor_kl_coefficient=0.05,
        anchor_continue_kl_coefficient=0.25,
    )
    assert rows and not model.training


def _gloo_shared_gpu_rank(rank: int, world: int, store: str, queue) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file:///{store}", rank=rank, world_size=world)
    try:
        torch.cuda.set_device(0)
        model, reference = _learner_and_reference()
        rollout = _rollout(model)
        model.to("cuda").eval()
        reference.to("cuda")
        rows = distributed_ppo_update_v4(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            rollout,
            epochs=1,
            lane_minibatch_size=2,
            sequence_chunk_steps=2,
            config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None),
            reference_model=reference,
            anchor_kl_coefficient=0.05,
            anchor_continue_kl_coefficient=0.25,
        )
        queue.put((rank, rows[0]["valid_count"], float(model.gate_head[-1].bias.sum())))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs one CUDA GPU")
def test_two_gloo_ranks_share_one_gpu_for_the_ppo_update(tmp_path, monkeypatch) -> None:
    import torch.multiprocessing as mp

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", __import__("os").environ.get("GLOO_SOCKET_IFNAME", "lo"))
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    store = str(tmp_path / "store").replace("\\", "/")
    mp.spawn(_gloo_shared_gpu_rank, args=(2, store, queue), nprocs=2)
    results = sorted(queue.get() for _ in range(2))
    assert results[0][1] == results[1][1] == 2 * 4 * 4 // 2  # global valid frames of the minibatch
    assert results[0][2] == results[1][2]  # identical parameters after the gloo all-reduce
