"""Semantics-preserving IL training speed-ups.

* chunk padding trim (``--trim-padding``): padded group/child/effect/event/edge
  rows are cut per chunk; outputs and gradients equal the padded run (fp32);
* tail compression (``--compact-tail-frames``): repeated frames past a lane's
  end are encoded once and only valid rows are decoded;
* relation attention through ``scaled_dot_product_attention`` with an additive
  bias, and the pair-bias MLP evaluated on spatial x spatial pairs only;
* fixed-shape relation bias / spatial scatter (no data-dependent host syncs)
  and the trainer's single deferred host sync per step.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import math
from pathlib import Path
import sys

import pytest
import torch

from native_runner.tests.test_training_v4_il_prefetch import _fixture as _prefetch_fixture
from native_runner.tests.test_training_v4_il_prefetch import _references
from native_runner.tests.test_training_v4_model import _model
from native_runner.tests.test_training_v4_trainer_options import _write_cache, single_process_group  # noqa: F401
from native_runner.training.v4 import evaluate_imitation_cache, train_imitation_cache
from native_runner.training.v4.cache import (
    _NEGATIVE_PADDING_FIELDS,
    ILChunkSequenceV4,
    _record_tensor_items,
    collate_il_cache_chunk,
    load_il_cache_shards,
)
from native_runner.training.v4.components import (
    GlobalTokenInputs,
    SparseRelationAttentionBlock,
    SpatialScatterEncoder,
    spatial_pair_features,
    spatial_pair_inputs,
)
from native_runner.training.v4.config import ModelConfigV4
from native_runner.training.v4.imitation import ILLossWeightsV4, ILSequenceV4
from native_runner.training.v4.learning import _repeated_tail_rows
from native_runner.training.v4.tensors import RelationEdgesV4, UniversalSemanticBatchV4
from native_runner.training.v4.train_imitation_cache import (
    PendingChunkMetricsV1,
    _DistributedImitationModule,
    _masked_chunk,
    build_training_plan,
    chunk_gradients,
    global_chunk_frames,
    slice_sequence_lanes,
)

TIME_STEPS = 3
_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _chunks(tmp_path: Path, **options: object) -> list[tuple[ILSequenceV4, int]]:
    """Every TBPTT chunk of the two-shard fixture (lanes of length 5, 3, 4, 4, 2, 2)."""

    index_path, tags_path = _write_cache(tmp_path, contract={})
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    max_length = max(length for plan in plans for length in plan.sequence_lengths)
    return [
        _masked_chunk(
            shards,
            plans,
            start=index * TIME_STEPS,
            time_steps=TIME_STEPS,
            device=torch.device("cpu"),
            dummy=False,
            **options,  # type: ignore[arg-type]
        )
        for index in range(-(-max_length // TIME_STEPS))
    ]


# Canonical padding values that differ from 0 / the cache's negative fields.
_PADDING = {
    ("groups", "owner_type"): 2,
    ("events", "owner_type"): 2,
    ("events", "target_token_index"): -1,
}


def _pad_collection(record: object, collection: str, rows: int, child_rows: int) -> object:
    values = {}
    for item in fields(record):  # type: ignore[arg-type]
        tensor = getattr(record, item.name)
        if tensor.ndim < 2:
            values[item.name] = tensor
            continue
        extra = child_rows if collection == "groups" and item.name.startswith("child_") else rows
        fill = _PADDING.get((collection, item.name), -1 if item.name in _NEGATIVE_PADDING_FIELDS else 0)
        padding = torch.full((tensor.shape[0], extra, *tensor.shape[2:]), fill, dtype=tensor.dtype)
        values[item.name] = torch.cat((tensor, padding), dim=1)
    return type(record)(**values)


def _pad_observation(observation: UniversalSemanticBatchV4) -> UniversalSemanticBatchV4:
    """Append canonical padding rows to every trimmable collection (as a full-capacity cache stores them)."""

    return replace(
        observation,
        groups=_pad_collection(observation.groups, "groups", 5, 7),
        events=_pad_collection(observation.events, "events", 4, 0),
        relation_edges=_pad_collection(observation.relation_edges, "relation_edges", 6, 0),
        active_effects=_pad_collection(observation.active_effects, "active_effects", 3, 0),
    )


def _run(
    chunks: list[tuple[ILSequenceV4, int]],
    *,
    micro_batches: int = 1,
    fused: bool = True,
    compact_tail: bool = False,
    defer: bool = False,
    autocast: torch.dtype | None = None,
):
    torch.manual_seed(3)
    policy = _model()
    module = _DistributedImitationModule(policy)
    state = policy.initial_state(chunks[0][0].batch_size)
    rows = []
    for sequence, _frames in chunks:
        policy.zero_grad(set_to_none=True)
        state, metrics = chunk_gradients(
            module,
            sequence,
            state,
            weights=ILLossWeightsV4(),
            world_size=1,
            loss_scale=1.3,
            micro_batches=micro_batches,
            autocast_dtype=autocast,
            validate=True,
            preencode_observations=True,
            fused_sequence=fused,
            compact_tail=compact_tail,
            defer_host_sync=defer,
        )
        if defer:
            assert isinstance(metrics, PendingChunkMetricsV1)
            metrics, _extra = metrics.resolve()
        gradients = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
        rows.append({"metrics": metrics, "gradients": gradients, "state": state})
    return rows


def _assert_runs_close(expected, actual, *, tolerance: float = 1e-6) -> None:
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual, strict=True):
        assert left["metrics"].keys() == right["metrics"].keys()
        for name, value in left["metrics"].items():
            if name.endswith("_count"):
                assert right["metrics"][name] == value, name
            else:
                assert right["metrics"][name] == pytest.approx(value, rel=tolerance, abs=tolerance), name
        assert left["gradients"].keys() == right["gradients"].keys()
        scale = max(float(g.abs().max()) for g in left["gradients"].values())
        for name, gradient in left["gradients"].items():
            assert float((gradient - right["gradients"][name]).abs().max()) <= tolerance * max(scale, 1.0), name
        for field_name in ("hidden", "cell"):
            assert torch.allclose(getattr(left["state"], field_name), getattr(right["state"], field_name), atol=tolerance)


# ---------------------------------------------------------------------------
# Padding trim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pin_memory", (False, True) if _CUDA else (False,))
def test_collate_trim_cuts_only_rows_no_frame_uses(tmp_path: Path, pin_memory: bool) -> None:
    plans, shards = _prefetch_fixture(tmp_path)
    for start in (0, 2, 4):
        references = _references(plans, start=start, time_steps=2)
        full = collate_il_cache_chunk(shards, references, time_steps=2)
        trimmed = collate_il_cache_chunk(shards, references, time_steps=2, trim_padding=True, pin_memory=pin_memory)
        if pin_memory:
            assert all(tensor.is_pinned() for tensor in trimmed.tensors() if tensor.numel())
        assert trimmed.lane_steps == full.lane_steps == tuple(ref.steps for refs in references for ref in refs)
        assert trimmed.mid_chunk_reset == full.mid_chunk_reset
        full_items = dict(_record_tensor_items(full.observations))
        trimmed_items = dict(_record_tensor_items(trimmed.observations))
        assert full_items.keys() == trimmed_items.keys()
        cut_any = False
        for name, tensor in full_items.items():
            kept = trimmed_items[name]
            collection, _, field = name.partition(".")
            if collection not in ("groups", "active_effects", "events", "relation_edges") or tensor.ndim < 2:
                # Candidates keep their width: the shadow features average over it.
                assert torch.equal(kept, tensor), name
                continue
            width = int(kept.shape[1])
            assert 1 <= width <= int(tensor.shape[1])
            assert torch.equal(kept, tensor[:, :width]), name
            mask_name = "child_mask" if collection == "groups" and field.startswith("child_") else "mask"
            assert not full_items[f"{collection}.{mask_name}"][:, width:].any(), name
            cut_any |= width < int(tensor.shape[1])
        assert cut_any  # the widened shard's effects/edges are cut
        for name in ("actions", "time_fields"):
            left, right = getattr(full, name), getattr(trimmed, name)
            left_items = dict(left.items()) if name == "time_fields" else dict(_record_tensor_items(left))
            right_items = dict(right.items()) if name == "time_fields" else dict(_record_tensor_items(right))
            for key, value in left_items.items():
                assert (value is None and right_items[key] is None) or torch.equal(value, right_items[key]), key


def test_trimmed_chunk_matches_padded_chunk_fp32(tmp_path: Path, single_process_group: None) -> None:  # noqa: F811
    trimmed = _chunks(tmp_path, trim_padding=True)
    padded = [
        (replace(sequence, observations=tuple(_pad_observation(item) for item in sequence.observations)), frames)
        for sequence, frames in trimmed
    ]
    assert padded[0][0].observations[0].groups.child_mask.shape[1] > trimmed[0][0].observations[0].groups.child_mask.shape[1]
    for fused in (True, False):
        # validate=True inside _run also proves the appended padding is canonical.
        _assert_runs_close(_run(padded, fused=fused), _run(trimmed, fused=fused))
    _assert_runs_close(_run(padded, micro_batches=2), _run(trimmed, micro_batches=2))


def test_trim_padding_defaults_off_in_library_calls(tmp_path: Path) -> None:
    chunk, _frames = _chunks(tmp_path)[0]
    assert type(chunk) is ILSequenceV4
    (tmp_path / "planned").mkdir()
    planned, _frames = _chunks(tmp_path / "planned", plan=True)[0]
    assert isinstance(planned, ILChunkSequenceV4)


# ---------------------------------------------------------------------------
# Tail compression
# ---------------------------------------------------------------------------


def test_repeated_tail_rows_maps_every_frame_to_its_distinct_encoding() -> None:
    assert _repeated_tail_rows([3, 3], time_steps=3, batch_size=2) is None
    distinct, source = _repeated_tail_rows([3, 1, 2], time_steps=3, batch_size=3)
    assert distinct.tolist() == [0, 1, 2, 3, 5, 6]
    assert source.tolist() == [0, 1, 2, 3, 1, 4, 5, 1, 4]
    with pytest.raises(ValueError, match="lane_steps"):
        _repeated_tail_rows([0, 3], time_steps=3, batch_size=2)


def test_chunk_plan_records_lane_steps_and_reset_flag(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, trim_padding=True, plan=True)
    lengths = (5, 3, 4, 4, 2, 2)
    for index, (sequence, frames) in enumerate(chunks):
        assert isinstance(sequence, ILChunkSequenceV4)
        start = index * TIME_STEPS
        assert sequence.lane_steps == tuple(max(1, min(TIME_STEPS, length - start)) for length in lengths)
        assert sequence.mid_chunk_reset is bool(sequence.episode_start[1:].any())
        assert frames == sum(max(0, min(TIME_STEPS, length - start)) for length in lengths)
    sliced = slice_sequence_lanes(chunks[1][0], 2, 5)
    assert isinstance(sliced, ILChunkSequenceV4)
    assert sliced.lane_steps == chunks[1][0].lane_steps[2:5]
    assert sliced.mid_chunk_reset is False


@pytest.mark.parametrize(("fused", "micro_batches"), ((True, 1), (False, 1), (True, 2)))
def test_tail_compression_matches_full_encode_fp32(
    tmp_path: Path, single_process_group: None, fused: bool, micro_batches: int  # noqa: F811
) -> None:
    chunks = _chunks(tmp_path, trim_padding=True, plan=True)
    assert any(min(sequence.lane_steps) < TIME_STEPS for sequence, _frames in chunks)
    full = _run(chunks, fused=fused, micro_batches=micro_batches)
    compact = _run(chunks, fused=fused, micro_batches=micro_batches, compact_tail=True)
    _assert_runs_close(full, compact)


def test_global_chunk_frames_equals_rank_sum(tmp_path: Path) -> None:
    index_path, tags_path = _write_cache(tmp_path, contract={})
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    for start in range(0, 6, 2):
        per_rank = [
            _masked_chunk(shards[i : i + 1], plans[i : i + 1], start=start, time_steps=2, device=torch.device("cpu"), dummy=False)[1]
            for i in range(len(plans))
        ]
        assert global_chunk_frames(plans, start=start, time_steps=2) == sum(per_rank)


# ---------------------------------------------------------------------------
# Relation attention: SDPA + spatial-pair-only bias MLP
# ---------------------------------------------------------------------------


def _attention_case(device: str, *, seed: int = 0):
    torch.manual_seed(seed)
    config = ModelConfigV4()
    block = SparseRelationAttentionBlock(config).to(device)
    towers, own, opponent, groups = 6, 8, 9, 5
    count = towers + own + opponent + groups
    batch = 3
    spatial = torch.zeros(batch, count, dtype=torch.bool, device=device)
    spatial[:, :towers] = torch.rand(batch, towers, device=device) > 0.2
    spatial[:, towers + own + opponent :] = torch.rand(batch, groups, device=device) > 0.4
    mask = spatial.clone()
    mask[:, towers : towers + own + opponent] = torch.rand(batch, own + opponent, device=device) > 0.3
    position = torch.rand(batch, count, 2, device=device) * torch.tensor([18.0, 32.0], device=device)
    extent = torch.cat((position - 0.5, position + 0.5), dim=-1)
    position = position * spatial.unsqueeze(-1)
    extent = extent * spatial.unsqueeze(-1)
    inputs = GlobalTokenInputs(
        tokens=torch.randn(batch, count, config.token_dim, device=device),
        mask=mask,
        position=position,
        extent=extent,
        spatial_mask=spatial,
        own_start=towers,
        own_count=own,
        group_start=towers + own + opponent,
        group_count=groups,
    )
    edge_count = 7
    edges = RelationEdgesV4(
        source=torch.randint(0, count, (batch, edge_count), device=device),
        target=torch.randint(0, count, (batch, edge_count), device=device),
        relation_type=torch.randint(0, config.relation_type_count, (batch, edge_count), device=device),
        mask=torch.rand(batch, edge_count, device=device) > 0.3,
    )
    edges = RelationEdgesV4(
        source=edges.source * edges.mask, target=edges.target * edges.mask,
        relation_type=edges.relation_type * edges.mask, mask=edges.mask,
    )
    tokens = torch.cat((torch.randn(batch, 1, config.token_dim, device=device), inputs.tokens), dim=1).requires_grad_(True)
    token_mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=device), mask), dim=1)
    return config, block, inputs, edges, tokens, token_mask


def _reference_attention(block, config, inputs, edges, tokens, token_mask):
    """The pre-SDPA block: full [B,N,N,P] pair MLP, boolean-indexed relation bias, explicit masked softmax."""

    batch, count, width = tokens.shape
    zeros = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)
    pair = spatial_pair_features(
        torch.cat((inputs.position.new_zeros(batch, 1, 2), inputs.position), dim=1),
        torch.cat((inputs.extent.new_zeros(batch, 1, 4), inputs.extent), dim=1),
        torch.cat((zeros, inputs.spatial_mask), dim=1),
        config,
    )
    query, key, value = block.qkv(block.norm1(tokens)).chunk(3, dim=-1)
    split = lambda item: item.view(batch, count, block.heads, block.head_dim).transpose(1, 2)  # noqa: E731
    query, key, value = split(query), split(key), split(value)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.head_dim)
    values = block.relation_bias(edges.relation_type).to(scores.dtype)
    flat_index = (
        torch.arange(batch, device=tokens.device)[:, None] * count * count + (edges.source + 1) * count + edges.target + 1
    )
    relation = torch.zeros(batch * count * count, block.heads, device=tokens.device, dtype=scores.dtype)
    relation.index_add_(0, flat_index[edges.mask], values[edges.mask])
    scores = scores + relation.view(batch, count, count, block.heads).permute(0, 3, 1, 2)
    scores = scores + block.spatial_bias(pair).permute(0, 3, 1, 2)
    scores = scores.masked_fill(~token_mask[:, None, None, :], torch.finfo(scores.dtype).min)
    attended = torch.matmul(torch.softmax(scores, dim=-1), value).transpose(1, 2).contiguous().view(batch, count, width)
    out = tokens + block.output(attended) * token_mask.unsqueeze(-1)
    out = out + block.ffn(block.norm2(out)) * token_mask.unsqueeze(-1)
    return out * token_mask.unsqueeze(-1)


def test_spatial_pair_block_reproduces_full_pair_bias_mlp() -> None:
    config, block, inputs, _edges, _tokens, _mask = _attention_case("cpu")
    batch, count = inputs.tokens.shape[:2]
    zeros = torch.zeros(batch, 1, dtype=torch.bool)
    full_pair = spatial_pair_features(
        torch.cat((inputs.position.new_zeros(batch, 1, 2), inputs.position), dim=1),
        torch.cat((inputs.extent.new_zeros(batch, 1, 4), inputs.extent), dim=1),
        torch.cat((zeros, inputs.spatial_mask), dim=1),
        config,
    )
    pair = spatial_pair_inputs(inputs, config, leading_rows=1)
    assert pair.features.shape[1] == 6 + inputs.group_count  # S x S, not N x N
    expected = block.spatial_bias(full_pair).permute(0, 3, 1, 2)
    actual = block._spatial_pair_bias(pair.features, pair.index)
    assert actual.shape == (batch, config.attention_heads, count + 1, count + 1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("checkpoint_scope", ((), ("pair-bias",), ("attention",)))
def test_sdpa_block_matches_math_reference_fp32(checkpoint_scope: tuple[str, ...]) -> None:
    config, block, inputs, edges, tokens, token_mask = _attention_case("cpu")
    block.checkpointing.set_scopes(checkpoint_scope)
    pair = spatial_pair_inputs(inputs, config, leading_rows=1)
    actual = block(tokens, token_mask, edges, pair)
    actual_grads = torch.autograd.grad(actual.square().sum(), [tokens, *block.parameters()])
    expected = _reference_attention(block, config, inputs, edges, tokens, token_mask)
    expected_grads = torch.autograd.grad(expected.square().sum(), [tokens, *block.parameters()])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for (name, _parameter), left, right in zip(
        [("tokens", tokens), *block.named_parameters()], actual_grads, expected_grads, strict=True
    ):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5, msg=name)
    # Both additive biases receive gradients through the fused attention.
    named = dict(zip([name for name, _p in block.named_parameters()], actual_grads[1:], strict=True))
    assert float(named["relation_bias.weight"].abs().sum()) > 0.0
    assert float(named["spatial_bias.0.weight"].abs().sum()) > 0.0


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_sdpa_block_bf16_matches_math_reference_at_rounding_level() -> None:
    config, block, inputs, edges, tokens, token_mask = _attention_case("cuda")
    pair = spatial_pair_inputs(inputs, config, leading_rows=1)
    with torch.no_grad():
        exact = _reference_attention(block, config, inputs, edges, tokens, token_mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = block(tokens, token_mask, edges, pair)
        reference = _reference_attention(block, config, inputs, edges, tokens, token_mask)
    # The fused kernel keeps scores in fp32 (the math path rounds them to bf16),
    # so it is at least as close to the fp32 result as the old bf16 path.
    new_error = float((actual.float() - exact).abs().max())
    old_error = float((reference.float() - exact).abs().max())
    assert new_error <= 1.5 * old_error + 1e-3
    assert new_error <= 0.05 * float(exact.abs().max())


# ---------------------------------------------------------------------------
# Fixed-shape (host-sync-free) relation bias and spatial scatter
# ---------------------------------------------------------------------------


def test_fixed_shape_relation_bias_equals_boolean_indexing() -> None:
    _config, block, _inputs, edges, tokens, _mask = _attention_case("cpu", seed=4)
    batch, count = tokens.shape[:2]
    actual = block._sparse_bias(edges, token_count=count, dtype=torch.float32)
    values = block.relation_bias(edges.relation_type)
    flat_index = torch.arange(batch)[:, None] * count * count + (edges.source + 1) * count + edges.target + 1
    expected = torch.zeros(batch * count * count, block.heads)
    expected.index_add_(0, flat_index[edges.mask], values[edges.mask])
    assert torch.equal(actual, expected.view(batch, count, count, block.heads).permute(0, 3, 1, 2))


def _brute_force_scatter(encoder: SpatialScatterEncoder, child, position, extent, radius, mask) -> torch.Tensor:
    """Per-entity reference of ``SpatialScatterEncoder.scatter`` (shapes cover cell centres; else bilinear point)."""

    config = encoder.config
    values = encoder.child_value(child)
    output = values.new_zeros(values.shape[0], config.learned_scatter_channels, config.board_height, config.board_width)
    for b in range(values.shape[0]):
        for c in range(values.shape[1]):
            if not mask[b, c]:
                continue
            x0, y0, x1, y1 = (float(v) for v in extent[b, c])
            r = float(radius[b, c])
            covered = []
            if r > 0.0 or x1 - x0 > 0.0 or y1 - y0 > 0.0:
                for gy in range(config.board_height):
                    for gx in range(config.board_width):
                        cx, cy = gx + 0.5, gy + 0.5
                        inside = (
                            (cx - float(position[b, c, 0])) ** 2 + (cy - float(position[b, c, 1])) ** 2 <= r * r + 1e-6
                            if r > 0.0
                            else (x0 <= cx < x1 and y0 <= cy < y1)
                        )
                        if inside:
                            covered.append((gy, gx))
            if covered:
                for gy, gx in covered:
                    output[b, :, gy, gx] += values[b, c]
                continue
            px = min(max(float(position[b, c, 0]) - 0.5, 0.0), config.board_width - 1.0)
            py = min(max(float(position[b, c, 1]) - 0.5, 0.0), config.board_height - 1.0)
            ix, iy = math.floor(px), math.floor(py)
            wx, wy = px - ix, py - iy
            jx, jy = min(ix + 1, config.board_width - 1), min(iy + 1, config.board_height - 1)
            for cx, cy, weight in ((ix, iy, (1 - wx) * (1 - wy)), (jx, iy, wx * (1 - wy)), (ix, jy, (1 - wx) * wy), (jx, jy, wx * wy)):
                output[b, :, cy, cx] += values[b, c] * weight
    return output


def _scatter_case(device: str):
    torch.manual_seed(5)
    encoder = SpatialScatterEncoder(ModelConfigV4()).to(device)
    child = torch.randn(2, 5, encoder.config.child_dim, device=device)
    position = torch.tensor(
        [[[4.5, 8.5], [4.0, 8.0], [9.0, 16.0], [0.3, 31.9], [12.2, 3.3]],
         [[3.75, 7.75], [17.9, 0.1], [6.0, 6.0], [10.0, 20.0], [1.0, 1.0]]],
        device=device,
    )
    extent = torch.cat((position, position), dim=-1)
    extent[1, 0] = torch.tensor([3.6, 7.6, 3.9, 7.9], device=device)  # sub-cell rectangle: point fallback
    extent[1, 2] = torch.tensor([4.0, 4.0, 8.0, 7.0], device=device)  # building footprint
    radius = torch.tensor([[0.0, 0.5, 3.5, 0.0, 2.0], [0.0, 0.0, 0.0, 1.2, 0.0]], device=device)
    mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]], device=device)
    return encoder, child, position, extent, radius, mask


def test_fixed_shape_scatter_matches_brute_force_with_gradients() -> None:
    encoder, child, position, extent, radius, mask = _scatter_case("cpu")
    child.requires_grad_(True)
    actual = encoder.scatter(child, position, extent, radius, mask)
    expected = _brute_force_scatter(encoder, child, position, extent, radius, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    weights = torch.randn_like(actual)
    left = torch.autograd.grad((actual * weights).sum(), [child, encoder.child_value.weight])
    right = torch.autograd.grad((expected * weights).sum(), [child, encoder.child_value.weight])
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_attention_and_scatter_run_without_host_syncs_on_cuda() -> None:
    config, block, inputs, edges, tokens, token_mask = _attention_case("cuda")
    encoder, child, position, extent, radius, mask = _scatter_case("cuda")
    child.requires_grad_(True)
    block.checkpointing.set_scopes(("pair-bias",))
    pair = spatial_pair_inputs(inputs, config, leading_rows=1)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = block(tokens, token_mask, edges, spatial_pair_inputs(inputs, config, leading_rows=1))
            scattered = encoder.scatter(child, position, extent, radius, mask)
        (out.float().sum() + scattered.float().sum()).backward()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert pair.features.shape[1] == 6 + inputs.group_count
    assert tokens.grad is not None and child.grad is not None


# ---------------------------------------------------------------------------
# One deferred host sync per step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("micro_batches", (1, 2))
def test_deferred_sync_returns_identical_metrics_and_gradients(
    tmp_path: Path, single_process_group: None, micro_batches: int  # noqa: F811
) -> None:
    chunks = _chunks(tmp_path, trim_padding=True, plan=True)
    plain = _run(chunks, micro_batches=micro_batches, compact_tail=True)
    deferred = _run(chunks, micro_batches=micro_batches, compact_tail=True, defer=True)
    for left, right in zip(plain, deferred, strict=True):
        assert left["metrics"] == right["metrics"]
        for name, gradient in left["gradients"].items():
            assert torch.equal(gradient, right["gradients"][name]), name


@pytest.mark.parametrize("micro_batches", (1, 2))
def test_deferred_sync_still_rejects_non_finite_loss(
    tmp_path: Path, single_process_group: None, micro_batches: int  # noqa: F811
) -> None:
    sequence, _frames = _chunks(tmp_path, trim_padding=True, plan=True)[0]
    torch.manual_seed(3)
    policy = _model()
    with torch.no_grad():
        policy.value_head[-1].bias.fill_(float("nan"))
    _state, pending = chunk_gradients(
        _DistributedImitationModule(policy),
        sequence,
        policy.initial_state(sequence.batch_size),
        weights=ILLossWeightsV4(),
        world_size=1,
        micro_batches=micro_batches,
        context="chunk=7",
        compact_tail=True,
        defer_host_sync=True,
    )
    assert isinstance(pending, PendingChunkMetricsV1)
    with pytest.raises(FloatingPointError, match="non-finite distributed IL loss at chunk=7"):
        pending.resolve(torch.tensor(1.0))


def test_pending_metrics_resolve_carries_extra_scalars() -> None:
    head_count = len(train_imitation_cache._LOSS_HEADS)
    stats = torch.cat((torch.full((head_count,), 0.5), torch.full((head_count,), 3.0), torch.full((head_count,), 2.0), torch.tensor([1.0, 1.0])))
    pending = PendingChunkMetricsV1(stats=stats, finite=torch.tensor(1, dtype=torch.int32), consistent=torch.tensor(True), loss_scale=2.0, context="")
    metrics, (norm,) = pending.resolve(torch.tensor(0.25))
    assert norm == 0.25
    assert metrics["gate"] == 0.5 and metrics["gate_count"] == 3.0 and metrics["chunk_loss_scale"] == 2.0
    assert metrics["continue_second_weighted_fraction"] == 0.5
    broken = replace(pending, consistent=torch.tensor(False))
    with pytest.raises(RuntimeError, match="micro-batch loss populations disagree"):
        broken.resolve()


# ---------------------------------------------------------------------------
# Command-line flags
# ---------------------------------------------------------------------------


class _Parsed(Exception):
    def __init__(self, namespace: argparse.Namespace) -> None:
        super().__init__()
        self.namespace = namespace


def _parse_main(monkeypatch: pytest.MonkeyPatch, main, argv: list[str]) -> argparse.Namespace:
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        raise _Parsed(original(self, *args, **kwargs))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(_Parsed) as parsed:
        main()
    return parsed.value.namespace


def test_trainer_speedup_flags_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    required = ["--index", "i.json", "--train-tags", "t.txt", "--output-dir", "out"]
    defaults = _parse_main(monkeypatch, train_imitation_cache.main, required)
    assert defaults.trim_padding is True and defaults.compact_tail_frames is True
    off = _parse_main(monkeypatch, train_imitation_cache.main, [*required, "--no-trim-padding", "--no-compact-tail-frames"])
    assert off.trim_padding is False and off.compact_tail_frames is False


def test_evaluator_speedup_flags_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    required = ["--index", "i.json", "--tags", "t.txt", "--checkpoint", "c.pt", "--output", "o.jsonl"]
    defaults = _parse_main(monkeypatch, evaluate_imitation_cache.main, required)
    assert defaults.trim_padding is True and defaults.compact_tail_frames is True
    off = _parse_main(monkeypatch, evaluate_imitation_cache.main, [*required, "--no-trim-padding", "--no-compact-tail-frames"])
    assert off.trim_padding is False and off.compact_tail_frames is False
