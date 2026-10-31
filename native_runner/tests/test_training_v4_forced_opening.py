from __future__ import annotations

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4
from native_runner.training.v4.train_ppo_self_play_cluster import (
    _rollout_behavior_counts,
)


@pytest.mark.parametrize("batch_size", (1, 2))
def test_ppo_forced_act_uses_conditional_behavior_probability(batch_size) -> None:
    from native_runner.training.v4.tensors import concatenate_tensor_records

    torch.manual_seed(6)
    model = _model().eval()
    batch = concatenate_tensor_records(tuple(_batch(batch_size=1) for _ in range(batch_size)))
    starts = torch.ones(1, batch_size, dtype=torch.bool)
    initial_state = model.initial_state(batch_size)
    with torch.no_grad():
        context = model.forward(
            batch,
            initial_state,
            episode_start=starts[0],
        )
        behavior = model.sample_after_preselected_act(batch, context)
    assert behavior.action_components is not None
    conditional_log_prob = (
        behavior.log_prob - behavior.action_components.gate_log_prob
    )[None]
    rollout = StoredRolloutV4(
        observations=(batch,),
        actions=(behavior.actions,),
        episode_start=starts,
        valid_mask=torch.ones(1, batch_size, dtype=torch.bool),
        behavior_log_prob=conditional_log_prob,
        behavior_value=behavior.value[None],
        next_value=torch.zeros(1, batch_size),
        rewards=torch.ones(1, batch_size),
        terminated=torch.ones(1, batch_size, dtype=torch.bool),
        truncated=torch.zeros(1, batch_size, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
        policy_state_id="forced-act-policy",
        forced_gate_mask=torch.ones(1, batch_size, dtype=torch.bool),
    )

    counts = _rollout_behavior_counts(rollout, model)
    assert counts[1].item() == batch_size
    assert counts[2].item() == batch_size
    assert counts[3].item() == batch_size
    assert counts[4].item() == batch_size
    assert counts[5].item() + counts[6].item() == batch_size
    assert counts[8].item() + counts[6].item() == counts[7].item()
    assert counts[10:].sum().item() == behavior.actions.micro_action_count.sum().item()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    metrics = distributed_ppo_update_v4(
        model,
        optimizer,
        rollout,
        epochs=1,
        lane_minibatch_size=1,
        lane_microbatch_size=1,
        sequence_chunk_steps=1,
        config=PPOConfigV4(entropy_coefficient=0.0, target_kl=None),
        autocast_dtype=None,
        expected_policy_state_id="forced-act-policy",
        validate=False,
    )

    # The behavior and current policies are the same network; the KL is pure fp32
    # round-off from re-evaluating the log-probability in a separate forward pass.
    assert metrics[0]["approx_kl"] == pytest.approx(0.0, abs=1e-5)


def test_behavior_counts_use_the_match_elixir_rate() -> None:
    from dataclasses import replace

    from native_runner.tests.test_training_v4_delay_aware_legality import EXPENSIVE, _with_elixir
    from native_runner.tests.test_training_v4_model import _with_candidate_mask
    from native_runner.training.v4.tensors import GATE_WAIT

    torch.manual_seed(7)
    model = _model().eval()
    with torch.no_grad():
        model.gate_head[-1].bias.copy_(torch.tensor([20.0, -20.0]))  # always WAIT
    eligible: dict[float, int] = {}
    for multiplier in (1.0, 2.0):
        # 0.1 elixir short of the only candidate: reachable in the window only at 2x.
        batch = _with_elixir(_batch(batch_size=1), 9.0 - 0.1, multiplier=multiplier, costs={EXPENSIVE: 9.0})
        mask = torch.zeros_like(batch.candidates.mask)
        mask[0, EXPENSIVE] = True
        batch = replace(batch, candidates=_with_candidate_mask(batch.candidates, mask))
        initial_state = model.initial_state(1)
        with torch.no_grad():
            behavior = model.act(batch, initial_state)
        assert behavior.actions.gate.tolist() == [GATE_WAIT]
        rollout = StoredRolloutV4(
            observations=(batch.to_storage("cpu", float_dtype=torch.float16),),
            actions=(behavior.actions,),
            episode_start=torch.tensor([[True]]),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
            behavior_log_prob=behavior.log_prob[None],
            behavior_value=behavior.value[None],
            next_value=torch.zeros(1, 1),
            rewards=torch.zeros(1, 1),
            terminated=torch.ones(1, 1, dtype=torch.bool),
            truncated=torch.zeros(1, 1, dtype=torch.bool),
            initial_state=initial_state,
            gate_temperature=1.0,
            action_temperature=1.0,
            continue_temperature=1.0,
            policy_state_id="elixir-rate-policy",
        )
        counts = _rollout_behavior_counts(rollout, model)
        assert counts[0].item() == 1
        assert counts[4].item() == 0
        eligible[multiplier] = int(counts[1].item())
        assert counts[2].item() == eligible[multiplier]  # eligible but waited
    assert eligible == {1.0: 0, 2.0: 1}
