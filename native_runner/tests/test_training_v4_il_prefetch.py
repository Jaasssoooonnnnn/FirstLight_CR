from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_evaluate_imitation_cache import _fixture_lanes
from native_runner.training.v4.cache import (
    ILCacheChunkRefV1,
    _pad_dynamic_observation,
    assert_il_sequences_exact,
    collate_il_cache_chunk,
    collate_loaded_il_batches,
    load_il_cache_shards,
    write_il_cache_index,
    write_il_cache_shard,
)
from native_runner.training.v4.imitation import ILSequenceV4
from native_runner.training.v4.train_imitation_cache import (
    ReplayWeightsV1,
    ShardTrainingPlanV1,
    _ChunkPrefetcher,
    _group_chunks,
    _GroupShardLoader,
    _masked_chunk,
    _masked_host_chunk,
    build_training_plan,
)

_DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


def _fixture(tmp_path: Path):
    """Shards (a) and (b, c); shard (a) stores wider dynamic collections than (b, c)."""

    lanes = _fixture_lanes()
    widened = tuple(
        replace(
            lane,
            observations=tuple(
                _pad_dynamic_observation(
                    observation,
                    active_effect_count=int(observation.active_effects.mask.shape[1]) + 2,
                    relation_edge_count=int(observation.relation_edges.mask.shape[1]) + 3,
                    candidate_count=int(observation.candidates.mask.shape[1]) + 2,
                )
                for observation in lane.observations
            ),
        )
        for lane in lanes["a"]
    )
    directory = tmp_path / "cache"
    directory.mkdir()
    summaries = []
    for shard_index, results in enumerate(
        ((_result("a", *widened),), (_result("b", *lanes["b"]), _result("c", *lanes["c"])))
    ):
        data_path = directory / f"shard-{shard_index:05d}.cril.zst"
        summary = write_il_cache_shard(data_path, results, shard_id=f"shard-{shard_index:05d}", contract={})
        summary["summary_file"] = data_path.with_suffix(".json").name
        summaries.append(summary)
    index_path = directory / "index.json"
    write_il_cache_index(index_path, summaries, contract={})
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("a\nb\nc\n", encoding="utf-8")
    plans, _metadata = build_training_plan([index_path], tags_path, manifest_workers=1)
    shards = load_il_cache_shards([plan.summary_path for plan in plans])
    return plans, shards


def _legacy_masked_chunk(
    shards,
    plans,
    *,
    start: int,
    time_steps: int,
    device: torch.device,
    dummy: bool,
    replay_weights: ReplayWeightsV1 | None = None,
) -> tuple[ILSequenceV4, int]:
    """The trainer's chunk path before the single-gather collation (verbatim)."""

    batches = []
    active_steps: list[int] = []
    lane_weights: list[float] = []
    for shard, plan in zip(shards, plans, strict=True):
        if replay_weights is not None:
            lane_weights.extend(replay_weights.plan_lane_weights(plan))
        references = []
        for sequence_index, length in zip(plan.sequence_indices, plan.sequence_lengths, strict=True):
            steps = max(0, min(time_steps, length - start))
            references.append(
                ILCacheChunkRefV1(sequence_index=sequence_index, start=min(start, length - 1), steps=max(1, steps))
            )
            active_steps.append(0 if dummy else steps)
        batches.append(shard.load_batch(references, time_steps=time_steps))
    sequence = collate_loaded_il_batches(batches, device=device).sequence
    active = torch.arange(time_steps, device=device)[:, None] < torch.tensor(active_steps, device=device)[None, :]
    sequence = replace(
        sequence,
        episode_start=sequence.episode_start & active,
        valid_mask=sequence.valid_mask & active,
        returns=torch.where(active, sequence.returns, torch.zeros_like(sequence.returns)),
        gate_loss_mask=(None if sequence.gate_loss_mask is None else sequence.gate_loss_mask & active),
        value_loss_mask=(None if sequence.value_loss_mask is None else sequence.value_loss_mask & active),
    )
    if replay_weights is not None:
        lanes = torch.tensor(lane_weights, dtype=torch.float32, device=device)[None, :].expand(time_steps, -1)
        sequence = replace(sequence, frame_weight=torch.where(active, lanes, torch.zeros_like(lanes)).contiguous())
    return sequence, sum(active_steps)


def _assert_exact(expected: ILSequenceV4, actual: ILSequenceV4) -> None:
    assert_il_sequences_exact(expected.to("cpu"), actual.to("cpu"))
    for left, right in ((expected.frame_weight, actual.frame_weight),):
        assert (left is None) == (right is None)
        if left is not None and right is not None:
            assert left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())


def _references(plans, *, start: int, time_steps: int) -> list[list[ILCacheChunkRefV1]]:
    return [
        [
            ILCacheChunkRefV1(
                sequence_index=index, start=min(start, length - 1), steps=max(1, min(time_steps, length - start))
            )
            for index, length in zip(plan.sequence_indices, plan.sequence_lengths, strict=True)
        ]
        for plan in plans
    ]


@pytest.mark.parametrize("pin_memory", (False, True) if torch.cuda.is_available() else (False,))
def test_single_gather_collation_matches_load_batch_collation(tmp_path: Path, pin_memory: bool) -> None:
    plans, shards = _fixture(tmp_path)
    # The two shards pad their dynamic collections differently, so the lane-wise
    # join must re-pad the narrower shard exactly as the per-step path does.
    names = ("active_effects", "relation_edges", "candidates")
    capacities = [tuple(int(getattr(shard.observations, name).mask.shape[1]) for name in names) for shard in shards]
    assert capacities[0] != capacities[1]
    for time_steps in (1, 2, 3, 5, 7):
        for start in range(0, 6, time_steps):
            references = _references(plans, start=start, time_steps=time_steps)
            for subset in ((0,), (1,), (0, 1), (1, 0)):
                selected = [shards[index] for index in subset]
                selected_references = [references[index] for index in subset]
                expected = collate_loaded_il_batches(
                    [
                        shard.load_batch(shard_references, time_steps=time_steps)
                        for shard, shard_references in zip(selected, selected_references, strict=True)
                    ]
                )
                chunk = collate_il_cache_chunk(
                    selected, selected_references, time_steps=time_steps, pin_memory=pin_memory
                )
                assert chunk.owner_frames == expected.owner_frames
                assert_il_sequences_exact(expected.sequence, chunk.sequence())
                if pin_memory:
                    # Empty tensors own no storage to pin.
                    assert all(tensor.is_pinned() for tensor in chunk.tensors() if tensor.numel())
                for device in _DEVICES:
                    assert_il_sequences_exact(
                        expected.sequence.to(device).to("cpu"),
                        chunk.to_model_input(device, non_blocking=pin_memory).sequence().to("cpu"),
                    )


@pytest.mark.parametrize("device", _DEVICES)
def test_masked_chunk_matches_previous_trainer_path(tmp_path: Path, device: str) -> None:
    plans, shards = _fixture(tmp_path)
    weights = ReplayWeightsV1(weights={("a", 0): 2.0, ("b", None): 0.5, ("c", 1): 3.0})
    for time_steps in (2, 3, 5):
        for start in range(0, 6, time_steps):
            for dummy in (False, True):
                for replay_weights in (None, weights):
                    options = dict(
                        start=start,
                        time_steps=time_steps,
                        device=torch.device(device),
                        dummy=dummy,
                        replay_weights=replay_weights,
                    )
                    expected, expected_frames = _legacy_masked_chunk(shards, plans, **options)
                    actual, actual_frames = _masked_chunk(shards, plans, **options)
                    assert actual_frames == expected_frames
                    assert actual.returns.device.type == device
                    _assert_exact(expected, actual)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("prefetch_chunks", (1, 2, 4))
def test_prefetching_group_chunks_match_synchronous_order(tmp_path: Path, device: str, prefetch_chunks: int) -> None:
    plans, shards = _fixture(tmp_path)
    weights = ReplayWeightsV1(weights={("a", 1): 0.25})
    options = dict(chunk_count=3, time_steps=2, device=torch.device(device), dummy=False, replay_weights=weights)
    synchronous = _group_chunks(shards, plans, prefetch_chunks=0, **options)
    expected = list(synchronous)
    prefetched = _group_chunks(shards, plans, prefetch_chunks=prefetch_chunks, **options)
    actual = list(prefetched)
    prefetched.close()
    assert [frames for _sequence, frames in actual] == [frames for _sequence, frames in expected] == [12, 7, 1]
    for (left, _left_frames), (right, _right_frames) in zip(expected, actual, strict=True):
        assert right.returns.device.type == device
        _assert_exact(left, right)


def test_prefetcher_reraises_worker_errors_and_stops_early(tmp_path: Path) -> None:
    plans, shards = _fixture(tmp_path)
    produced: list[int] = []

    def produce(index: int):
        produced.append(index)
        if index == 1:
            raise ValueError("broken chunk 1")
        return _masked_host_chunk(shards, plans, start=index * 2, time_steps=2, dummy=False)

    failing = _ChunkPrefetcher(produce, chunk_count=4, depth=2, device=torch.device("cpu"))
    _sequence, frames = next(failing)
    assert frames == 12
    with pytest.raises(ValueError, match="broken chunk 1"):
        next(failing)
    with pytest.raises(StopIteration):
        next(failing)
    failing.close()
    assert produced == [0, 1]

    abandoned = _ChunkPrefetcher(
        lambda index: _masked_host_chunk(shards, plans, start=index, time_steps=1, dummy=False),
        chunk_count=1_000,
        depth=1,
        device=torch.device("cpu"),
    )
    next(abandoned)
    abandoned.close()
    assert not abandoned._thread.is_alive()


def test_group_loader_preloads_next_group_within_memory_budget(tmp_path: Path) -> None:
    plans, _shards = _fixture(tmp_path)
    path_groups = [[plan.summary_path] for plan in plans] + [[plans[0].summary_path, plans[1].summary_path]]
    reference = [load_il_cache_shards(paths) for paths in path_groups]

    def assert_same(expected, actual) -> None:
        assert [shard.path for shard in expected] == [shard.path for shard in actual]
        for left, right in zip(expected, actual, strict=True):
            for index in range(len(left.descriptors)):
                assert_il_sequences_exact(left.sequence(index), right.sequence(index))

    roomy = _GroupShardLoader(path_groups, workers=2, preload_next=True, available_memory=lambda: 1 << 50)
    try:
        for index, expected in enumerate(reference):
            shards, preloaded = roomy.get(index)
            assert preloaded == (index > 0)
            assert_same(expected, shards)
            status = roomy.last_preload["next_group_preload"]
            assert status == ("started" if index + 1 < len(path_groups) else "none")
    finally:
        roomy.close()

    tight = _GroupShardLoader(path_groups, workers=2, preload_next=True, local_world_size=8, available_memory=lambda: 1)
    try:
        shards, preloaded = tight.get(0)
        assert not preloaded and tight.last_preload["next_group_preload"] == "skipped-memory"
        assert int(tight.last_preload["preload_required_bytes"]) > 0
        shards, preloaded = tight.get(1)
        assert not preloaded
        assert_same(reference[1], shards)
    finally:
        tight.close()

    disabled = _GroupShardLoader(path_groups, workers=1, preload_next=False)
    shards, preloaded = disabled.get(2)
    assert not preloaded and disabled.last_preload["next_group_preload"] == "disabled"
    assert_same(reference[2], shards)
    disabled.close()


def test_dummy_rank_chunks_are_fully_masked(tmp_path: Path) -> None:
    plans, shards = _fixture(tmp_path)
    source = plans[0]
    dummy_plans = (ShardTrainingPlanV1(summary_path=source.summary_path, units=(source.units[0],)),)
    sequence, frames = _masked_chunk(
        shards[:1], dummy_plans, start=0, time_steps=3, device=torch.device("cpu"), dummy=True
    )
    assert frames == 0
    assert not bool(sequence.valid_mask.any()) and not bool(sequence.episode_start.any())
