"""PPO capacity expansion must preserve the V4 masked-field sentinels."""

from dataclasses import fields

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch
from native_runner.tests.test_training_v4_learning import _clone_tensor_tree
from native_runner.tests.test_training_v4_collector_protocol import collect_recorded_segments
from native_runner.training.v4.async_cluster_self_play import _pad_active_effects, _pad_batch_rows
from native_runner.training.v4.cluster_self_play import _padded_records, _scatter_storage_rows
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.distributed_ppo import distributed_ppo_update_v4
from native_runner.training.v4.ppo import PPOConfigV4
from native_runner.training.v4.tensors import tensor_padding_value


def test_worker_effect_capacity_and_actor_rows_keep_canonical_padding():
    source = _batch(2)
    padded = _pad_active_effects(source, 64)
    effects = padded.active_effects
    assert torch.all(effects.parent_index[~effects.mask] == -1)
    effects.validate(ModelConfigV4(), batch_size=2, effect_vocab_size=10, child_count=6, tower_count=6)
    expanded = _pad_batch_rows(padded, 4)
    for name in ("active_effects", "candidates", "towers", "groups", "events", "previous_action"):
        before, after = getattr(padded, name), getattr(expanded, name)
        for item in fields(before):
            original = getattr(before, item.name)
            result = getattr(after, item.name)
            torch.testing.assert_close(result[:2], original, rtol=0, atol=0)
            assert torch.all(result[2:] == tensor_padding_value(before, item.name))


def test_collector_concat_and_scatter_only_pad_new_effect_addresses():
    small = _pad_active_effects(_batch(1), 4).active_effects
    wide = _pad_active_effects(_batch(1), 8).active_effects
    combined = _padded_records((small, wide))
    assert combined.mask.shape == (2, 8)
    assert torch.all(combined.parent_index == -1)
    scattered = _scatter_storage_rows(_padded_records((small, small)), wide, torch.tensor([1]), total_rows=2)
    for item in fields(combined):
        torch.testing.assert_close(getattr(scattered, item.name), getattr(combined, item.name), rtol=0, atol=0)
    # Existing malformed rows must still fail validation instead of being repaired silently.
    small.parent_index[0, 0] = 0
    corrupt = _padded_records((small, wide))
    assert corrupt.parent_index[0, 0] == 0 and torch.all(corrupt.parent_index[0, 4:] == -1)
    with pytest.raises(ValueError, match="parent_index.*padding"):
        corrupt.validate(ModelConfigV4(), batch_size=2, effect_vocab_size=10, child_count=6, tower_count=6)


def test_dynamic_effect_frames_bootstrap_resume_and_optimize_with_validation(monkeypatch):
    model, segments = collect_recorded_segments(monkeypatch, validate=True, dynamic_effects=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for rollout in segments:
        # Production transfers CPU storage to CUDA before autograd; the CPU-only
        # test materializes ordinary tensors explicitly instead of a device hop.
        rollout = _clone_tensor_tree(rollout)
        metrics = distributed_ppo_update_v4(
            model,
            optimizer,
            rollout,
            epochs=1,
            lane_minibatch_size=4,
            lane_microbatch_size=2,
            sequence_chunk_steps=1,
            config=PPOConfigV4(target_kl=None),
            autocast_dtype=None,
            expected_policy_state_id=rollout.policy_state_id,
            validate=True,
        )
        assert metrics and all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in metrics)
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
