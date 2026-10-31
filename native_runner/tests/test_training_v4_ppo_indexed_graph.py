from dataclasses import fields

import pytest
import torch

from native_runner.training.v4.async_cluster_self_play import (
    _IndexedStateCudaGraphV4, _StochasticCudaGraphV4, _RowPackedTensorRecordV4,
    _ppo_cuda_graph_row_capacity_v4, _rollout_state_bank_v4,
)
from native_runner.training.v4.tensors import ActionSequenceV4
from native_runner.tests.test_training_v4_model import _batch, _model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_indexed_graph_matches_legacy_recurrence_with_reordering_padding_and_resets():
    device = torch.device('cuda', 0)
    model = _model().to(device).eval()
    model.set_ppo_rollout_sampling(False)
    model.set_delay_policy('earliest')
    actors = 5
    bank = _rollout_state_bank_v4(model, 'current', actors, device)
    reference = model.initial_state(actors, device=device)
    graphs = {}
    seen = set()
    sequences = ([2, 0, 4], [4], [0, 1], [3, 4, 1, 0], [2, 3, 4], [2, 0, 4])
    with torch.no_grad():
        for iteration, rows in enumerate(sequences):
            capacity = _ppo_cuda_graph_row_capacity_v4(len(rows))
            source = _batch(batch_size=len(rows)).to_storage('cpu', float_dtype=torch.float16)
            source.match_scalars[:, 0] = torch.tensor(rows) / 10
            source = _RowPackedTensorRecordV4(source)
            pad = torch.tensor(list(range(len(rows))) + [0] * (capacity - len(rows)))
            packed = source.index_select(pad)
            starts = [row not in seen or iteration == 5 for row in rows]
            indices = torch.tensor(rows, device=device)
            selected = reference.index_select(indices).index_select(pad.to(device))
            episode_start = torch.tensor(starts + [False] * (capacity - len(rows)), device=device)
            if capacity not in graphs:
                options = dict(effect_capacity=max(1, source.record.active_effects.mask.shape[1]),
                               relation_capacity=max(1, source.record.relation_edges.mask.shape[1]))
                graphs[capacity] = (
                    _StochasticCudaGraphV4(device, **options),
                    _IndexedStateCudaGraphV4(device, state_bank=bank, actor_capacity=actors,
                                            capacity=capacity, **options),
                )
            legacy, indexed = graphs[capacity]
            output = legacy.run(model, packed.record, selected, episode_start, packed)
            expected = [getattr(output.actions, item.name)[:len(rows)].clone() for item in fields(ActionSequenceV4)]
            expected += [output.log_prob[:len(rows)].clone(), output.value[:len(rows)].clone()]
            reference.hidden.index_copy_(0, indices, output.next_state.hidden[:len(rows)])
            reference.cell.index_copy_(0, indices, output.next_state.cell[:len(rows)])
            indexed.prepare(rows, starts)
            actual = indexed.run_indexed(model, packed.record, packed)
            actual_rows = [getattr(actual.actions, item.name)[:len(rows)] for item in fields(ActionSequenceV4)]
            actual_rows += [actual.log_prob[:len(rows)], actual.value[:len(rows)]]
            for got, value in zip(actual_rows, expected, strict=True):
                torch.testing.assert_close(got, value, rtol=0, atol=0)
            torch.testing.assert_close(bank.hidden[:actors], reference.hidden, rtol=0, atol=0)
            torch.testing.assert_close(bank.cell[:actors], reference.cell, rtol=0, atol=0)
            seen.update(rows)
        pointer = bank.hidden.data_ptr()
        reset = _rollout_state_bank_v4(model, 'current', actors, device)
        assert reset.hidden.data_ptr() == pointer
        assert not reset.hidden.any() and not reset.cell.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_sampled_indexed_graph_log_prob_uses_the_pre_step_state():
    device = torch.device('cuda', 0)
    model = _model().to(device).eval()
    model.set_delay_policy('earliest')
    model.set_ppo_gate_temperature(1.0)
    model.set_ppo_action_temperature(1.0)
    model.set_ppo_continue_temperature(1.0)
    model.set_ppo_rollout_sampling(True)
    bank = _rollout_state_bank_v4(model, 'sampled', 4, device)
    source = _RowPackedTensorRecordV4(_batch(batch_size=4).to_storage('cpu', float_dtype=torch.float16))
    graph = _IndexedStateCudaGraphV4(device, state_bank=bank, actor_capacity=4, capacity=4,
        effect_capacity=max(1, source.record.active_effects.mask.shape[1]),
        relation_capacity=max(1, source.record.relation_edges.mask.shape[1]))
    indices = torch.tensor([3, 0, 2, 1], device=device)
    with torch.no_grad():
        for step in range(3):
            before = bank.index_select(indices)
            starts = torch.full((4,), step == 0, device=device, dtype=torch.bool)
            graph.prepare(indices.cpu().tolist(), [step == 0] * 4)
            output = graph.run_indexed(model, source.record, source)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rescored = model.evaluate_actions(source.record.to_model_input(device), before, output.actions,
                    episode_start=starts, validate=False)
            torch.testing.assert_close(output.log_prob, rescored.log_prob, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(bank.hidden.index_select(0, indices), rescored.next_state.hidden, rtol=0, atol=0)
