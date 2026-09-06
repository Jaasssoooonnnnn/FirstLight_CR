from __future__ import annotations

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4
from native_runner.training.v4.train_ppo_self_play_cluster import (
    _rollout_behavior_counts,
)


def test_ppo_forced_act_uses_conditional_behavior_probability() -> None:
    torch.manual_seed(6)
    model = _model().eval()
    batch = _batch(batch_size=1)
    starts = torch.tensor([[True]])
    initial_state = model.initial_state(1)
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
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        behavior_log_prob=conditional_log_prob,
        behavior_value=behavior.value[None],
        next_value=torch.zeros(1, 1),
        rewards=torch.ones(1, 1),
        terminated=torch.ones(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        initial_state=initial_state,
        gate_temperature=1.0,
        action_temperature=1.0,
        continue_temperature=1.0,
        policy_state_id="forced-act-policy",
        forced_gate_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    counts = _rollout_behavior_counts(rollout, model)
    assert counts[1].item() == 1
    assert counts[2].item() == 1
    assert counts[3].item() == 1
    assert counts[4].item() == 1
    assert counts[5].item() + counts[6].item() == 1
    assert counts[8].item() + counts[6].item() == counts[7].item()
    assert counts[10:].sum().item() == behavior.actions.micro_action_count.item()

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

    assert metrics[0]["approx_kl"] == pytest.approx(0.0, abs=1e-6)
