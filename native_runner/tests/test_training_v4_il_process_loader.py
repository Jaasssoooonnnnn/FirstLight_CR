"""Process/thread shard loaders: byte identity with the legacy reader, errors, fallback, end-to-end trainer."""

from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import functools
import json
from pathlib import Path
import sys

import pytest
import torch

from native_runner.tests.test_training_v4_cache import _result
from native_runner.tests.test_training_v4_evaluate_imitation_cache import (
    _evaluate,
    _fixture_lanes,
    _save_checkpoint,
    _without_timing,
)
from native_runner.tests.test_training_v4_model import _model
from native_runner.training.v4 import train_imitation_cache as trainer
from native_runner.training.v4.cache import (
    ILCacheProcessLoader,
    ILCacheShardV1,
    _CONTAINER_HEADER,
    _decode_il_cache_shard,
    _ILCacheTensorSpec,
    _tensor_from_spec,
    assert_il_sequences_exact,
    load_il_cache_shard,
    load_il_cache_shards,
    write_il_cache_index,
    write_il_cache_shard,
)
from native_runner.training.v4.train_imitation_cache import _GroupShardLoader


def _write_shards(tmp_path: Path, groups=(("a",), ("b", "c")), *, contract: dict[str, object] | None = None):
    lanes = _fixture_lanes()
    directory = tmp_path / "cache"
    directory.mkdir()
    summaries = []
    for shard_index, tags in enumerate(groups):
        data_path = directory / f"shard-{shard_index:05d}.cril.zst"
        summary = write_il_cache_shard(
            data_path,
            tuple(_result(tag, *lanes[tag]) for tag in tags),
            shard_id=f"shard-{shard_index:05d}",
            contract=contract or {},
        )
        summary["summary_file"] = data_path.with_suffix(".json").name
        summaries.append(summary)
    index_path = directory / "index.json"
    write_il_cache_index(index_path, summaries, contract=contract or {})
    return index_path, [directory / f"shard-{index:05d}.cril.json" for index in range(len(groups))]


def _legacy_tensors(summary_path: Path) -> dict[str, torch.Tensor]:
    """The pre-change reader: inflate to bytes, slice, ``safetensors.torch.load``."""

    import zstandard
    from safetensors.torch import load

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compressed = (summary_path.parent / summary["data_file"]).read_bytes()
    container = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=int(summary["container_bytes"]))
    _magic, manifest_length = _CONTAINER_HEADER.unpack_from(container)
    return load(container[_CONTAINER_HEADER.size + manifest_length :])


def _raw_tensors(decoded) -> dict[str, torch.Tensor]:
    return {spec.name: _tensor_from_spec(decoded.buffer, spec) for spec in decoded.specs}


def _assert_bytes_equal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    assert sorted(expected) == sorted(actual)
    for name, left in expected.items():
        right = actual[name]
        assert (left.dtype, tuple(left.shape)) == (right.dtype, tuple(right.shape)), name
        assert right.is_contiguous(), name
        assert left.reshape(-1).view(torch.uint8).equal(right.reshape(-1).view(torch.uint8)), name


def _assert_shards_exact(expected: ILCacheShardV1, actual: ILCacheShardV1) -> None:
    assert expected.path == actual.path and expected.manifest == actual.manifest
    for index in range(len(expected.descriptors)):
        assert_il_sequences_exact(expected.sequence(index), actual.sequence(index))
    for name in ("episode_start", "valid_mask", "returns", "sequence_offsets", "candidate_counts"):
        assert getattr(expected, name).equal(getattr(actual, name))


@pytest.fixture(scope="module")
def process_loader():
    with ILCacheProcessLoader(2) as loader:
        yield loader


def test_thread_and_process_loaders_are_byte_identical_to_legacy_reader(tmp_path: Path, process_loader) -> None:
    _index, summaries = _write_shards(tmp_path)
    for summary_path in summaries:
        reference = _legacy_tensors(summary_path)
        threaded = _decode_il_cache_shard(summary_path, verify=True)
        processed = process_loader.submit(summary_path).result()
        assert processed.buffer.is_shared()
        _assert_bytes_equal(reference, _raw_tensors(threaded))
        _assert_bytes_equal(reference, _raw_tensors(processed))
        # Every tensor is an aligned view into the shard's single buffer.
        for spec in threaded.specs:
            assert spec.offset % max(1, spec.dtype.itemsize) == 0
        legacy = ILCacheShardV1(
            path=Path(threaded.data_path), manifest=threaded.manifest, tensors=reference, summary=threaded.summary
        )
        _assert_shards_exact(legacy, threaded.build())
        _assert_shards_exact(legacy, processed.build())

    by_thread = load_il_cache_shards(summaries, workers=2)
    by_process = load_il_cache_shards(summaries, workers=2, process_loader=process_loader)
    for left, right in zip(by_thread, by_process, strict=True):
        _assert_shards_exact(left, right)


def test_unaligned_tensor_offsets_are_copied_into_aligned_storage() -> None:
    values = torch.arange(6, dtype=torch.int32)
    buffer = torch.zeros(3 + values.numel() * 4, dtype=torch.uint8)
    buffer[3:] = values.view(torch.uint8)
    spec = _ILCacheTensorSpec("x", torch.int32, (2, 3), 3, values.numel() * 4)
    tensor = _tensor_from_spec(buffer, spec)
    assert tensor.equal(values.view(2, 3))
    empty = _tensor_from_spec(buffer, _ILCacheTensorSpec("e", torch.float32, (0, 4), 0, 0))
    assert tuple(empty.shape) == (0, 4) and empty.dtype == torch.float32


def _corrupt_compressed(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data_path = summary_path.parent / summary["data_file"]
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 0xFF
    data_path.write_bytes(payload)


def _corrupt_container_digest(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["container_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize(
    ("corrupt", "message"),
    ((_corrupt_compressed, "compressed SHA-256"), (_corrupt_container_digest, "container SHA-256 mismatch")),
)
def test_sha256_mismatch_raises_the_same_error_in_both_loaders(
    tmp_path: Path, process_loader, corrupt, message: str
) -> None:
    _index, summaries = _write_shards(tmp_path)
    corrupt(summaries[1])
    with pytest.raises(ValueError, match=message):
        load_il_cache_shard(summaries[1])
    with pytest.raises(ValueError, match=message):
        load_il_cache_shards(summaries, workers=2)
    with pytest.raises(ValueError, match=message):
        load_il_cache_shards(summaries, process_loader=process_loader)
    if corrupt is _corrupt_container_digest:
        # verify=False skips the digests, exactly like the legacy reader.
        (unverified,) = load_il_cache_shards([summaries[1]], verify=False, process_loader=process_loader)
        assert unverified.owner_frames > 0


class _BrokenLoader:
    sharing_strategy = "file_system"
    start_method = "spawn"

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.closed = False

    def load(self, paths, *, verify: bool = True, config=None):
        raise self.error

    def close(self) -> None:
        self.closed = True


def _groups(summaries):
    return [[summaries[0]], [summaries[1]], [summaries[0], summaries[1]]]


def _assert_groups(loader: _GroupShardLoader, summaries) -> list[str]:
    modes = []
    for index, paths in enumerate(_groups(summaries)):
        shards, preloaded = loader.get(index)
        assert preloaded == (index > 0)
        modes.append(str(loader.last_preload["cache_loader_mode"]))
        for left, right in zip(load_il_cache_shards(paths), shards, strict=True):
            _assert_shards_exact(left, right)
        loader.release(shards)
        del shards
    return modes


def test_group_loader_process_mode_preloads_through_processes(tmp_path: Path) -> None:
    _index, summaries = _write_shards(tmp_path)
    loader = _GroupShardLoader(
        _groups(summaries), workers=2, preload_next=True, mode="process", available_memory=lambda: 1 << 50
    )
    try:
        assert loader.describe()["preload_workers_mode_effective"] == "process"
        assert _assert_groups(loader, summaries) == ["sync-thread", "process", "process"]
        assert loader.process_error is None
    finally:
        loader.close()


def test_group_loader_falls_back_to_threads(tmp_path: Path) -> None:
    _index, summaries = _write_shards(tmp_path)

    def cannot_start(_workers: int):
        raise OSError("no processes here")

    unstartable = _GroupShardLoader(
        _groups(summaries),
        workers=2,
        preload_next=True,
        mode="process",
        available_memory=lambda: 1 << 50,
        process_loader_factory=cannot_start,
    )
    try:
        described = unstartable.describe()
        assert described["preload_workers_mode_effective"] == "thread"
        assert "no processes here" in str(described["preload_process_error"])
        assert _assert_groups(unstartable, summaries) == ["sync-thread", "thread", "thread"]
    finally:
        unstartable.close()

    broken = _BrokenLoader(BrokenProcessPool("worker died"))
    breaking = _GroupShardLoader(
        _groups(summaries),
        workers=2,
        preload_next=True,
        mode="process",
        available_memory=lambda: 1 << 50,
        process_loader_factory=lambda _w: broken,
    )
    try:
        assert _assert_groups(breaking, summaries) == ["sync-thread", "thread-fallback", "thread"]
        assert broken.closed and breaking.mode == "thread"
        assert "worker died" in str(breaking.last_preload["preload_process_error"])
    finally:
        breaking.close()

    tight_shm = _GroupShardLoader(
        _groups(summaries),
        workers=2,
        preload_next=True,
        mode="process",
        available_memory=lambda: 1 << 50,
        shared_memory=lambda: 1,
    )
    try:
        assert _assert_groups(tight_shm, summaries) == ["sync-thread", "thread-shm", "thread-shm"]
        assert tight_shm.mode == "process"
    finally:
        tight_shm.close()


def test_group_loader_does_not_mask_cache_errors_as_process_failures(tmp_path: Path) -> None:
    _index, summaries = _write_shards(tmp_path)
    corrupt = _BrokenLoader(ValueError("cache container SHA-256 mismatch"))
    loader = _GroupShardLoader(
        _groups(summaries),
        workers=2,
        preload_next=True,
        mode="process",
        available_memory=lambda: 1 << 50,
        process_loader_factory=lambda _w: corrupt,
    )
    try:
        loader.get(0)
        with pytest.raises(ValueError, match="container SHA-256"):
            loader.get(1)
        assert loader.mode == "process"
    finally:
        loader.close()


def test_release_counts_toward_the_preload_memory_gate(tmp_path: Path) -> None:
    _index, summaries = _write_shards(tmp_path)
    groups = _groups(summaries)
    required = trainer._group_load_bytes(groups[2], workers=1)
    budget = {"available": 0}
    loader = _GroupShardLoader(
        groups, workers=1, preload_next=True, mode="thread", available_memory=lambda: budget["available"]
    )
    try:
        shards, _ = loader.get(0)
        assert loader.last_preload["next_group_preload"] == "skipped-memory"
        shards, _ = loader.get(1)
        # Pretend the release is still in flight: its bytes count as available.
        with loader._release_lock:
            loader._releasing_bytes = 2 * required
        loader._start_preload(2)
        assert loader.last_preload["next_group_preload"] == "started"
        assert loader.last_preload["host_releasing_bytes"] == 2 * required
        with loader._release_lock:
            loader._releasing_bytes = 0
        shards, preloaded = loader.get(2)
        assert preloaded
    finally:
        loader.close()


def test_evaluator_rows_do_not_depend_on_the_loader_mode(tmp_path: Path) -> None:
    index_path, _summaries = _write_shards(tmp_path)
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("a\nb\nc\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint, 5, update_step=3)
    rows = {
        name: [_without_timing(row) for row in _evaluate(index_path, tags_path, [checkpoint], **options)]
        for name, options in {
            "sync": {"preload_next_group": False, "prefetch_chunks": 0},
            "thread": {"preload_workers_mode": "thread", "prefetch_chunks": 2},
            "process": {"preload_workers_mode": "process", "prefetch_chunks": 1, "load_workers": 2},
        }.items()
    }
    assert rows["sync"] == rows["thread"] == rows["process"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer requires CUDA")
@pytest.mark.parametrize("mode", ("process", "thread"))
def test_trainer_runs_end_to_end_with_preload_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    index_path, _summaries = _write_shards(
        tmp_path, groups=(("a",), ("b",), ("c",)), contract={"gamma_per_decision": 0.99}
    )
    tags_path = tmp_path / "tags.txt"
    tags_path.write_text("a\nb\nc\n", encoding="utf-8")
    group_file = (tmp_path / "process-group").resolve().as_uri()
    original_init = dist.init_process_group

    def init_process_group(*_args, **_kwargs) -> None:
        original_init("gloo", init_method=group_file, rank=0, world_size=1)

    def small_model(*, device, config=None):
        torch.manual_seed(0)
        return _model().to(device)

    monkeypatch.setattr(trainer.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(trainer, "build_production_model_v4", small_model)
    # The fixture-catalog model leaves some parameters out of the loss (also
    # with the pre-change trainer); production DDP settings are otherwise kept.
    monkeypatch.setattr(
        trainer,
        "DistributedDataParallel",
        functools.partial(torch.nn.parallel.DistributedDataParallel, find_unused_parameters=True),
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_imitation_cache",
            "--index",
            str(index_path),
            "--train-tags",
            str(tags_path),
            "--output-dir",
            str(output),
            "--expected-world-size",
            "1",
            "--shards-per-rank",
            "1",
            "--time-steps",
            "2",
            "--load-workers",
            "2",
            "--autocast",
            "none",
            "--epochs",
            "2",
            "--save-every-groups",
            "2",
            "--log-every",
            "1",
            "--preload-workers-mode",
            mode,
        ],
    )
    trainer.main()
    rows = [json.loads(line) for line in (output / "train-metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    startup = next(row for row in rows if row["event"] == "startup")
    assert startup["preload_workers_mode"] == mode
    assert startup["preload_workers_mode_effective"] == mode
    groups = [row for row in rows if row["event"] == "group_complete"]
    assert len(groups) == 6
    assert groups[0]["cache_loader_mode"] == "sync-thread"
    assert {row["cache_loader_mode"] for row in groups[1:]} == {mode}
    assert all(row["cache_preloaded_rank0"] for row in groups[1:])
    assert sum(row["owner_frames"] for row in groups) == 2 * sum(
        len(lane.observations) for pair in _fixture_lanes().values() for lane in pair
    )
    assert any(row["event"] == "checkpoint" for row in rows)
    assert all(row["next_group_open_s_rank0"] >= 0 for row in groups)
