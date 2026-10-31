from dataclasses import fields

import pytest
import torch

from native_runner.training.v4.async_cluster_self_play import (
    _ModelPackedArenaV4, _RowPackedTensorRecordV4, _concatenate_row_packed_records_v4,
    _pack_action_batch_to_cpu, _pack_model_actions_to_cpu_v4, _StochasticCudaGraphV4,
)
from native_runner.training.v4.tensors import ActionSequenceV4
from native_runner.tests.test_training_v4_model import _batch
from native_runner.tests.test_training_v4_model import _model


@pytest.mark.parametrize('pin_memory', [False, True])
def test_partition_matches_legacy_gather_and_padding_and_owns_storage(pin_memory):
    if pin_memory and not torch.cuda.is_available():
        pytest.skip('pinned allocator requires CUDA')
    first = _RowPackedTensorRecordV4(_batch(batch_size=2))
    second = _RowPackedTensorRecordV4(_batch(batch_size=2))
    for buffer in second.buffers.values():
        if buffer.dtype != torch.bool:
            buffer.add_(1)
    values = [(first, 2), (second, 2)]
    rows = {'current': [0, 2, 3], 'anchor': [1], 'unused': []}
    arena = _ModelPackedArenaV4(pin_memory=pin_memory)
    actual = arena.assemble(values, rows)
    legacy = _concatenate_row_packed_records_v4([first, second])
    for key, indices in rows.items():
        if not indices:
            assert key not in actual
            continue
        indices = indices + [indices[0]] * (actual[key].batch_size - len(indices))
        expected = legacy.index_select(torch.tensor(indices))
        for dtype in expected.buffers:
            assert torch.equal(actual[key].buffers[dtype], expected.buffers[dtype])
            assert actual[key].buffers[dtype].is_pinned() == pin_memory
    stored = _RowPackedTensorRecordV4._from_buffers(actual['current'], {
        dtype: buffer[:3].clone() for dtype, buffer in actual['current'].buffers.items()
    })
    preserved = {dtype: buffer.clone() for dtype, buffer in stored.buffers.items()}
    pointers = {dtype: buffer.data_ptr() for dtype, buffer in actual['current'].buffers.items()}
    for buffer in first.buffers.values():
        buffer.zero_()
    again = arena.assemble(values, rows)
    for dtype in stored.buffers:
        assert torch.equal(stored.buffers[dtype], preserved[dtype])
        assert again['current'].buffers[dtype].data_ptr() == pointers[dtype]


def test_partition_respects_partial_packet_and_schema_fallback():
    first = _RowPackedTensorRecordV4(_batch(batch_size=2))
    second = _RowPackedTensorRecordV4(_batch(batch_size=2))
    arena = _ModelPackedArenaV4(pin_memory=False)
    result = arena.assemble([(first, 1), (second, 2)], {'current': [0, 2]})
    for dtype, buffer in result['current'].buffers.items():
        assert torch.equal(buffer, torch.stack((first.buffers[dtype][0], second.buffers[dtype][1])))
    second.schema = ('value', 'incompatible')
    assert arena.assemble([(first, 1), (second, 2)], {'current': [0, 2]}) is None
    with pytest.raises(ValueError, match='outside'):
        arena.assemble([(first, 1)], {'current': [1]})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fused_action_transfer_matches_each_legacy_transfer(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    actions = ActionSequenceV4(
        gate=torch.arange(4, device=device), micro_action_count=torch.arange(4, device=device),
        **{name: torch.arange(8, device=device).reshape(4, 2) + index * 100
           for index, name in enumerate(('candidate_index', 'candidate_uid', 'target_cell', 'delay_offset_bin'))},
    )
    values = {'current': (actions, 3), 'anchor': (actions, 1)}
    combined = _pack_model_actions_to_cpu_v4(values)
    for key, (action, length) in values.items():
        expected_action, expected_packed = _pack_action_batch_to_cpu(action, length)
        actual_action, actual_packed = combined[key]
        assert torch.equal(actual_packed, expected_packed)
        for field in fields(ActionSequenceV4):
            assert torch.equal(getattr(actual_action, field.name), getattr(expected_action, field.name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_pinned_partition_graph_matches_legacy_actions_log_prob_value_and_state():
    device = torch.device('cuda', 0)
    model = _model().to(device).eval()
    model.set_ppo_rollout_sampling(False)
    model.set_delay_policy('earliest')
    source = _RowPackedTensorRecordV4(_batch(batch_size=2).to_storage('cpu', float_dtype=torch.float16))
    arena = _ModelPackedArenaV4()
    partition = arena.assemble([(source, 2)], {'current': [0, 1]})['current']
    graph = _StochasticCudaGraphV4(device,
        effect_capacity=max(1, source.record.active_effects.mask.shape[1]),
        relation_capacity=max(1, source.record.relation_edges.mask.shape[1]))
    state = model.initial_state(2, device=device)
    starts = torch.ones(2, device=device, dtype=torch.bool)
    with torch.no_grad():
        legacy = graph.run(model, source.record, state, starts, source)
        expected = [getattr(legacy.actions, field.name).clone() for field in fields(ActionSequenceV4)]
        expected += [legacy.log_prob.clone(), legacy.value.clone(),
                     legacy.next_state.hidden.clone(), legacy.next_state.cell.clone()]
        actual = graph.run(model, partition.record, state, starts, partition)
        torch.cuda.synchronize()
    observed = [getattr(actual.actions, field.name) for field in fields(ActionSequenceV4)]
    observed += [actual.log_prob, actual.value, actual.next_state.hidden, actual.next_state.cell]
    for got, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(got, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_retained_graph_reads_weights_after_real_ppo_update():
    from native_runner.tests.test_training_v4_ppo_anchor import _learner_and_reference, _rollout, _update
    model, reference = _learner_and_reference()
    model.set_delay_policy('earliest')
    reference.set_delay_policy('earliest')
    rollout = _rollout(model)
    device = torch.device('cuda', 0)
    model.to(device).eval()
    reference.to(device)
    model.set_ppo_rollout_sampling(False)
    source = _RowPackedTensorRecordV4(_batch(batch_size=2).to_storage('cpu', float_dtype=torch.float16))
    kwargs = dict(effect_capacity=max(1, source.record.active_effects.mask.shape[1]),
                  relation_capacity=max(1, source.record.relation_edges.mask.shape[1]))
    retained = _StochasticCudaGraphV4(device, **kwargs)
    state = model.initial_state(2, device=device)
    starts = torch.ones(2, device=device, dtype=torch.bool)
    with torch.no_grad():
        before = retained.run(model, source.record, state, starts, source).value.clone()
    _update(model, reference, rollout, chunk=2, lr=1e-3)
    assert not model.training
    with torch.no_grad():
        updated = retained.run(model, source.record, state, starts, source)
        values = [getattr(updated.actions, field.name).clone() for field in fields(ActionSequenceV4)]
        values += [updated.log_prob.clone(), updated.value.clone(),
                   updated.next_state.hidden.clone(), updated.next_state.cell.clone()]
        fresh = _StochasticCudaGraphV4(device, **kwargs).run(model, source.record, state, starts, source)
        expected = [getattr(fresh.actions, field.name) for field in fields(ActionSequenceV4)]
        expected += [fresh.log_prob, fresh.value, fresh.next_state.hidden, fresh.next_state.cell]
        assert not torch.equal(before, fresh.value)
        for actual, reference_value in zip(values, expected, strict=True):
            torch.testing.assert_close(actual, reference_value, rtol=0, atol=0)
