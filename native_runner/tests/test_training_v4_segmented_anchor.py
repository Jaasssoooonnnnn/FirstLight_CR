"""Frozen-reference KL matches unbroken replay at non-unit sampling temperatures."""
from dataclasses import replace

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch
from native_runner.tests.test_training_v4_ppo_anchor import _learner_and_reference
from native_runner.training.v4.distributed_ppo import _reference_rollout_components_v4, distributed_ppo_update_v4
from native_runner.training.v4.ppo import PPOConfigV4, StoredRolloutV4
from native_runner.training.v4.ppo_reference_state import ReferenceStateCacheV4


def trajectory(model, *, steps=6, lanes=3):
    model.set_ppo_gate_temperature(0.3)
    model.set_ppo_action_temperature(1.0)
    model.set_ppo_continue_temperature(1.0)
    model.set_ppo_rollout_sampling(True)
    batch = _batch(lanes)
    starts = torch.zeros(steps, lanes, dtype=torch.bool)
    starts[0] = True
    states = [model.initial_state(lanes)]
    actions, logs, values = [], [], []
    torch.manual_seed(101)
    with torch.no_grad():
        for t in range(steps):
            output = model.sample_for_ppo_rollout(batch, states[-1], episode_start=starts[t])
            actions.append(output.actions)
            logs.append(output.log_prob)
            values.append(output.value)
            states.append(output.next_state)
    rollout = StoredRolloutV4(
        observations=(batch,) * steps, actions=tuple(actions), episode_start=starts,
        valid_mask=torch.ones(steps, lanes, dtype=torch.bool), behavior_log_prob=torch.stack(logs),
        behavior_value=torch.stack(values), next_value=torch.zeros(steps, lanes), rewards=torch.ones(steps, lanes),
        terminated=torch.zeros(steps, lanes, dtype=torch.bool), truncated=torch.zeros(steps, lanes, dtype=torch.bool),
        initial_state=states[0], gate_temperature=0.3, action_temperature=1.0, continue_temperature=1.0,
        policy_state_id="gate03", lane_keys=tuple((7, i) for i in range(lanes)),
    )
    return rollout, states


def segment(rollout, states, start, end):
    return replace(rollout,
        observations=rollout.observations[start:end], actions=rollout.actions[start:end],
        episode_start=rollout.episode_start[start:end], valid_mask=rollout.valid_mask[start:end],
        behavior_log_prob=rollout.behavior_log_prob[start:end], behavior_value=rollout.behavior_value[start:end],
        next_value=rollout.next_value[start:end], rewards=rollout.rewards[start:end],
        terminated=rollout.terminated[start:end], truncated=rollout.truncated[start:end], initial_state=states[start],
        forced_gate_mask=None if rollout.forced_gate_mask is None else rollout.forced_gate_mask[start:end],
    )


def replay(reference, rollout, initial, *, chunk=2):
    return _reference_rollout_components_v4(reference, rollout, initial,
        microbatch_size=2, chunk_steps=chunk, autocast_dtype=None)


def test_gate03_reference_segments_equal_full_replay_with_lane_reordering_and_removal():
    model, reference = _learner_and_reference()
    with torch.no_grad():
        reference.lstm_core.weight_hh.add_(0.01)
    rollout, states = trajectory(model)
    whole, whole_final = replay(reference, rollout, reference.initial_state(3))
    first = segment(rollout, states, 0, 3)
    second = segment(rollout, states, 3, 6).select_lanes(torch.tensor([2, 0]))
    cache = ReferenceStateCacheV4()
    left, left_final = replay(reference, first, cache.prepare(reference, first))
    cache.commit(first, left_final)
    initial = cache.prepare(reference, second)
    # Reference recurrence comes from reference weights, not the learner's carried state.
    assert not torch.allclose(initial.hidden, second.initial_state.hidden)
    right, right_final = replay(reference, second, initial)
    torch.testing.assert_close(left, whole[:3], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(right, whole[3:, [2, 0]], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(right_final.hidden, whole_final.hidden[[2, 0]], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(right_final.cell, whole_final.cell[[2, 0]], rtol=1e-5, atol=1e-6)


def test_reference_cache_requires_known_continuations_and_resets_new_games():
    model, reference = _learner_and_reference()
    rollout, states = trajectory(model)
    cache = ReferenceStateCacheV4()
    second = segment(rollout, states, 3, 6)
    with pytest.raises(ValueError, match="no prior reference state"):
        cache.prepare(reference, second)
    first = segment(rollout, states, 0, 3)
    _, final = replay(reference, first, cache.prepare(reference, first))
    first.terminated[-1, 0] = True
    cache.commit(first, final)
    assert (7, 0) not in cache.states and (7, 1) in cache.states
    fresh = replace(second, episode_start=second.episode_start.clone())
    fresh.episode_start[0] = True
    initial = cache.prepare(reference, fresh)
    assert torch.count_nonzero(initial.hidden) == torch.count_nonzero(initial.cell) == 0
    cache.clear()
    assert not cache.states


def test_reference_cache_keeps_memory_at_bootstrapped_segment_truncation():
    model, reference = _learner_and_reference()
    rollout, states = trajectory(model)
    first = segment(rollout, states, 0, 3)
    second = segment(rollout, states, 3, 6)
    first.truncated[-1] = True
    cache = ReferenceStateCacheV4()
    _, final = replay(reference, first, cache.prepare(reference, first))
    cache.commit(first, final)
    initial = cache.prepare(reference, second)
    torch.testing.assert_close(initial.hidden, final.hidden)
    torch.testing.assert_close(initial.cell, final.cell)


@pytest.mark.parametrize("device,autocast", [("cpu", None), ("cuda", None), ("cuda", torch.bfloat16)])
def test_gate03_kl_is_zero_for_identical_policies_across_updates(device, autocast):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model, reference = _learner_and_reference()
    rollout, states = trajectory(model)
    model.to(device)
    reference.to(device)
    cache = ReferenceStateCacheV4()
    for start, end in [(0, 3), (3, 6)]:
        part = segment(rollout, states, start, end)
        final = reference.initial_state(part.batch_size)
        rows = distributed_ppo_update_v4(model, torch.optim.AdamW(model.parameters(), lr=0), part,
            epochs=2, lane_minibatch_size=2, lane_microbatch_size=1, sequence_chunk_steps=2,
            config=PPOConfigV4(entropy_coefficient=0, target_kl=None), autocast_dtype=autocast,
            reference_model=reference, reference_initial_state=cache.prepare(reference, part),
            reference_final_state_out=final, anchor_kl_coefficient=0.05, anchor_continue_kl_coefficient=0.25)
        cache.commit(part, final)
        for row in rows:
            assert row["anchor_kl"] == pytest.approx(0, abs=2e-5)
            assert row["anchor_kl_continue"] == pytest.approx(0, abs=2e-5)
        assert all(p.grad is None for p in reference.parameters())


def test_reference_final_state_covers_all_lanes_even_when_ppo_early_stops():
    model, reference = _learner_and_reference()
    rollout, _ = trajectory(model)
    rollout = replace(rollout, behavior_log_prob=rollout.behavior_log_prob + 1)
    initial = reference.initial_state(rollout.batch_size)
    _, expected = replay(reference, rollout, initial)
    actual = reference.initial_state(rollout.batch_size)
    rows = distributed_ppo_update_v4(model, torch.optim.SGD(model.parameters(), lr=0), rollout,
        epochs=2, lane_minibatch_size=1, sequence_chunk_steps=2,
        config=PPOConfigV4(entropy_coefficient=0, target_kl=1e-8), autocast_dtype=None,
        reference_model=reference, reference_initial_state=initial, reference_final_state_out=actual,
        anchor_kl_coefficient=0.05)
    assert len(rows) == 1
    torch.testing.assert_close(actual.hidden, expected.hidden, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.cell, expected.cell, rtol=1e-5, atol=1e-6)
