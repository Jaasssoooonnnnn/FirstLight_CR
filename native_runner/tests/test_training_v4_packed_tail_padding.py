from dataclasses import fields, replace

import pytest
import torch

from native_runner.tests.test_training_v4_model import _batch
from native_runner.training.v4.async_cluster_self_play import (
    _RowPackedTensorRecordV4,
    _pad_packed_rollout_tail_v4,
)
from native_runner.training.v4.cluster_self_play import _scatter_storage_rows


def _relations(batch, size):
    edges = batch.relation_edges
    if size <= edges.mask.shape[1]:
        values = {f.name: getattr(edges, f.name)[:, :size] for f in fields(edges)}
    else:
        values = {f.name: torch.cat((getattr(edges, f.name), getattr(edges, f.name)[:, :1]), 1) for f in fields(edges)}
    return replace(batch, relation_edges=type(edges)(**values))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("pattern", ["mixed", "valid", "invalid"])
@pytest.mark.parametrize("smaller_last", [False, True])
def test_packed_tail_matches_existing_scatter_and_preserves_aliases(dtype, pattern, smaller_last):
    torch.manual_seed(713)
    steps, lanes = 7, 5
    last = _batch(batch_size=lanes).to_storage("cpu", float_dtype=dtype)
    base = _RowPackedTensorRecordV4(last)
    if smaller_last:
        last = _relations(last, 1)
    original = base.empty_rows(steps * lanes)
    for buffer in original.buffers.values():
        buffer.copy_(torch.randint(0, 7, buffer.shape).to(buffer.dtype))
    valid = torch.arange(steps)[:, None] < torch.tensor([0, 1, 3, 6, 7])[None]
    if pattern != "mixed":
        valid.fill_(pattern == "valid")
    expected = base._from_buffers(base, {k: v.clone() for k, v in original.buffers.items()})
    actual = base._from_buffers(base, {k: v.clone() for k, v in original.buffers.items()})
    # These are the observation views finalized before the padding pass.
    aliases = [actual.narrow_rows(t * lanes, lanes).record for t in range(steps)]
    for t, mask in enumerate(valid):
        missing = (~mask).nonzero(as_tuple=False).flatten()
        if missing.numel():
            _scatter_storage_rows(
                expected.narrow_rows(t * lanes, lanes).record, last.index_select(missing), missing,
                total_rows=lanes, clone_previous=False,
            )
    # Tiny budget exercises repeated chunks, including one-row chunks.
    assert _pad_packed_rollout_tail_v4(actual, last, valid, scratch_bytes=256)
    for key in original.buffers:
        assert torch.equal(actual.buffers[key], expected.buffers[key])
        assert torch.equal(actual.buffers[key][valid.flatten()], original.buffers[key][valid.flatten()])
    for t, alias in enumerate(aliases):
        packed = _RowPackedTensorRecordV4(alias)
        for key, buffer in packed.buffers.items():
            assert torch.equal(buffer, expected.buffers[key][t * lanes:(t + 1) * lanes])


def test_expanded_last_schema_falls_back_without_modifying_storage():
    last = _batch(batch_size=2).to_storage("cpu", float_dtype=torch.float16)
    storage = _RowPackedTensorRecordV4(last).empty_rows(6)
    for buffer in storage.buffers.values():
        buffer.fill_(1)
    before = {key: buffer.clone() for key, buffer in storage.buffers.items()}
    assert not _pad_packed_rollout_tail_v4(storage, _relations(last, 4), torch.zeros(3, 2, dtype=torch.bool))
    assert all(torch.equal(buffer, before[key]) for key, buffer in storage.buffers.items())
