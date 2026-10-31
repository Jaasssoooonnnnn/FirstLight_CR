from dataclasses import fields, replace

import pytest
import torch

from native_runner.training.v4.async_cluster_self_play import _RowPackedTensorRecordV4
from native_runner.training.v4.ppo_device_rollout import PackedRolloutDeviceCacheV4
from native_runner.training.v4.tensors import concatenate_padded_tensor_records, concatenate_tensor_records
from native_runner.tests.test_training_v4_async_cuda_graph import _tensor_rows
from native_runner.tests.test_training_v4_ppo_anchor import _learner_and_reference, _rollout, _clone, _update


def _packed_rollout():
    model, reference = _learner_and_reference()
    model.set_delay_policy('earliest')
    reference.set_delay_policy('earliest')
    rollout = _rollout(model, steps=5, lanes=4)
    packed = _RowPackedTensorRecordV4(concatenate_padded_tensor_records(rollout.observations).to_storage('cpu'))
    rollout = replace(rollout, packed_observation_rows=packed,
        observations=tuple(packed.narrow_rows(t * 4, 4).record for t in range(5)))
    return model, reference, rollout


@pytest.mark.parametrize('device_name,budget', [('cpu', 0), ('cuda', 0), ('cuda', 2**30)])
def test_packed_chunks_preserve_time_lane_order_and_all_tensor_fields(device_name, budget):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    _model, _reference, rollout = _packed_rollout()
    device = torch.device(device_name)
    cache = PackedRolloutDeviceCacheV4(rollout, device, budget_bytes=budget)
    assert cache.resident == (device_name == 'cuda' and budget > 0)
    for start, end, lane_list in ((0, 3, [3, 0, 2]), (3, 5, [1]), (0, 5, [0, 1, 2, 3])):
        lanes = torch.tensor(lane_list)
        actual, actions = cache.chunk(start, end, lanes)
        expected = concatenate_padded_tensor_records(tuple(
            value.index_select(lanes) for value in rollout.observations[start:end])).to_model_input(device)
        expected_actions = concatenate_tensor_records(tuple(
            value.index_select(lanes) for value in rollout.actions[start:end])).to(device)
        for (path, got), (reference_path, value) in zip(_tensor_rows(actual), _tensor_rows(expected), strict=True):
            assert path == reference_path and got.dtype == value.dtype
            assert torch.equal(got, value), path
            assert got.is_contiguous(), path
        for item in fields(actions):
            assert torch.equal(getattr(actions, item.name), getattr(expected_actions, item.name))
    assert rollout.select_lanes(torch.tensor([0, 2])).packed_observation_rows is not None


@pytest.mark.parametrize('indices', [[3, 0, 2], [1], [2, 2, 0]])
def test_removing_failed_lanes_preserves_packed_fields_and_time_order(indices):
    _model, _reference, rollout = _packed_rollout()
    selected = torch.tensor(indices)
    actual = rollout.select_lanes(selected)
    expected = replace(rollout, packed_observation_rows=None).select_lanes(selected)
    assert actual.packed_observation_rows.batch_size == rollout.time_steps * len(indices)
    for (path, got), (other, value) in zip(_tensor_rows(actual), _tensor_rows(expected), strict=True):
        # _tensor_rows traverses dataclasses; compare the public rollout fields
        # separately below because the packed sidecar has its own tensors.
        assert path == other
        assert torch.equal(got, value), path
    actual.validate()
    for t, observation in enumerate(actual.observations):
        for (path, got), (other, value) in zip(_tensor_rows(observation), _tensor_rows(expected.observations[t]), strict=True):
            assert path == other
            assert torch.equal(got, value), path
    with pytest.raises((IndexError, RuntimeError)):
        rollout.select_lanes(torch.tensor([rollout.batch_size]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('microbatch', [None, 2])
@pytest.mark.parametrize('remove_failed', [False, True])
def test_resident_rollout_preserves_ppo_kl_loss_and_optimizer_result(monkeypatch, microbatch, remove_failed):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model, reference, rollout = _packed_rollout()
        # Uneven episode lengths exercise compacted active lanes in tail chunks.
        rollout.valid_mask[3:, 2:] = False
        rollout.terminated[2, 2:] = True
        rollout.rewards[3:, 2:] = 0
        if remove_failed:
            rollout = rollout.select_lanes(torch.tensor([3, 0, 2]))
        legacy = _clone(model).to('cuda')
        model.to('cuda')
        reference.to('cuda')
        actual = _update(model, reference, rollout, chunk=2, lr=1e-4, device_rollout_budget_gib=1,
                         lane_microbatch_size=microbatch)
        expected = _update(legacy, reference, rollout, chunk=2, lr=1e-4, lane_microbatch_size=microbatch)
        assert actual[0]['rollout_device_cache_gib'] > 0
        for got, expected_row in zip(actual, expected, strict=True):
            for key in ('policy', 'value', 'total', 'anchor_kl', 'approx_kl', 'clip_fraction', 'valid_count'):
                assert got[key] == pytest.approx(expected_row[key], rel=1e-6, abs=1e-7), key
        for (name, parameter), (other_name, value) in zip(model.named_parameters(), legacy.named_parameters(), strict=True):
            assert name == other_name
            torch.testing.assert_close(parameter, value, rtol=1e-6, atol=1e-7, msg=name)
    finally:
        torch.use_deterministic_algorithms(previous)
