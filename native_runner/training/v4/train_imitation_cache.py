"""Train UniversalCardPolicyV4 from sharded replay caches with stateful DDP."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import threading
import time
from typing import Callable, Iterator, Mapping, Sequence, cast

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .cache import (
    IL_CACHE_SUMMARY_SCHEMA,
    SHARD_LOAD_MODES,
    CollatedILChunkV1,
    ILChunkSequenceV4,
    ILCacheChunkRefV1,
    ILCacheProcessLoader,
    ILCacheSequenceDescriptorV1,
    ILCacheShardV1,
    collate_il_cache_chunk,
    load_il_cache_index,
    load_il_cache_manifest,
    load_il_cache_shards,
    shared_memory_free_bytes,
)
from .checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from .components import ACTIVATION_CHECKPOINT_SCOPES, DEFAULT_ACTIVATION_CHECKPOINT_SCOPES
from .factory import build_production_model_v4, model_config_from_overrides
from .imitation import (
    DEFAULT_IL_DELAY_NEIGHBOR_WEIGHT,
    DEFAULT_IL_GATE_ACT_WEIGHT,
    IL_ANNOTATION_FIELDS,
    IL_TIME_FIELDS,
    ILLossV4,
    ILLossWeightsV4,
    ILSequenceV4,
    continue_label_class_weight,
    imitation_loss,
)
from .learning import RecurrentEvaluationV4, evaluate_recurrent_sequence
from .model import UniversalCardPolicyV4
from .tensors import GATE_ACT, RecurrentPolicyStateV4


@dataclass(frozen=True, slots=True)
class ReplayUnitV1:
    replay_tag: str
    sequence_indices: tuple[int, int]
    sequence_lengths: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ShardTrainingPlanV1:
    summary_path: Path
    units: tuple[ReplayUnitV1, ...]

    @property
    def sequence_indices(self) -> tuple[int, ...]:
        return tuple(index for unit in self.units for index in unit.sequence_indices)

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return tuple(length for unit in self.units for length in unit.sequence_lengths)

    @property
    def replay_count(self) -> int:
        return len(self.units)

    @property
    def owner_frames(self) -> int:
        return sum(self.sequence_lengths)


REPLAY_WEIGHTS_HEADER = ("replay_tag", "owner", "weight")
CHUNK_NORMALIZATIONS = ("per-chunk", "group-expected")


@dataclass(frozen=True, slots=True)
class ReplayWeightsV1:
    """Per-(replay, owner) loss weights; ``owner=None`` is the ``*`` wildcard."""

    weights: Mapping[tuple[str, int | None], float]
    strict: bool = False

    def lane_weight(self, replay_tag: str, owner: int) -> float:
        if owner not in (0, 1):
            raise ValueError(f"replay owner must be 0 or 1, got {owner}")
        for key in ((replay_tag, owner), (replay_tag, None)):
            if key in self.weights:
                return self.weights[key]
        if self.strict:
            raise ValueError(f"replay weights have no row for replay {replay_tag} owner {owner}")
        return 1.0

    def plan_lane_weights(self, plan: ShardTrainingPlanV1) -> tuple[float, ...]:
        """Weights in ``plan.sequence_indices`` order (each unit: owner 0, then owner 1)."""

        return tuple(self.lane_weight(unit.replay_tag, owner) for unit in plan.units for owner in (0, 1))


def load_replay_weights(path: Path, *, strict: bool) -> ReplayWeightsV1:
    """Parse ``replay_tag<TAB>owner<TAB>weight`` (owner 0, 1 or ``*``)."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != REPLAY_WEIGHTS_HEADER:
        header = "\t".join(REPLAY_WEIGHTS_HEADER)
        raise ValueError(f"replay weight file must start with the header {header!r}: {path}")
    weights: dict[tuple[str, int | None], float] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        columns = line.split("\t")
        if len(columns) != 3 or not columns[0] or columns[0] != columns[0].strip():
            raise ValueError(f"replay weight file line {line_number} is malformed: {path}")
        replay_tag, raw_owner, raw_weight = columns
        if raw_owner not in ("0", "1", "*"):
            raise ValueError(f"replay weight file line {line_number} owner must be 0, 1 or *: {path}")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"replay weight file line {line_number} weight is not a number: {path}") from error
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"replay weight file line {line_number} weight must be finite and >= 0: {path}")
        key = (replay_tag, None if raw_owner == "*" else int(raw_owner))
        if key in weights:
            raise ValueError(f"replay weight file line {line_number} duplicates {replay_tag} owner {raw_owner}: {path}")
        weights[key] = weight
    for replay_tag, owner in weights:
        if owner is not None and (replay_tag, None) in weights:
            raise ValueError(f"replay weight file gives {replay_tag} both a * row and an owner {owner} row: {path}")
    return ReplayWeightsV1(weights=weights, strict=strict)


def replay_weight_summary(replay_weights: ReplayWeightsV1, plans: Sequence[ShardTrainingPlanV1]) -> dict[str, object]:
    """Check every planned lane against the weight table (raises in strict mode)."""

    lane_count = 0
    explicit_lanes = 0
    weighted_frames = 0.0
    owner_frames = 0
    planned_tags: set[str] = set()
    for plan in plans:
        for unit in plan.units:
            planned_tags.add(unit.replay_tag)
            for owner, length in zip((0, 1), unit.sequence_lengths, strict=True):
                weight = replay_weights.lane_weight(unit.replay_tag, owner)
                lane_count += 1
                explicit_lanes += int(
                    (unit.replay_tag, owner) in replay_weights.weights
                    or (unit.replay_tag, None) in replay_weights.weights
                )
                weighted_frames += weight * length
                owner_frames += length
    return {
        "replay_weight_rows": len(replay_weights.weights),
        "replay_weight_rows_outside_split": sum(tag not in planned_tags for tag, _owner in replay_weights.weights),
        "replay_weight_explicit_lanes": explicit_lanes,
        "replay_weight_default_lanes": lane_count - explicit_lanes,
        "replay_weight_frame_mean": weighted_frames / max(owner_frames, 1),
        "replay_weights_strict": replay_weights.strict,
    }


def load_model_config_overrides(value: str) -> dict[str, object]:
    """Parse ``--model-config-json``: inline JSON, or ``@path`` to a JSON file."""

    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    try:
        overrides = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"--model-config-json is not valid JSON: {error}") from error
    if not isinstance(overrides, dict):
        raise ValueError("--model-config-json must be a JSON object")
    return overrides


def require_cache_gamma(dataset_metadata: Mapping[str, object], gamma: float) -> float:
    """The value targets were discounted at cache-build time; --gamma must agree."""

    gammas = dataset_metadata.get("cache_gamma_per_decision")
    if not isinstance(gammas, Mapping) or not gammas:
        raise ValueError("training plan metadata has no cache gamma_per_decision")
    for index_path, cache_gamma in gammas.items():
        if cache_gamma is None:
            raise ValueError(f"cache index contract has no gamma_per_decision: {index_path}")
        if float(cache_gamma) != float(gamma):
            raise ValueError(f"--gamma {gamma} differs from cache gamma_per_decision {cache_gamma}: {index_path}")
    return float(gamma)


def chunk_loss_scale(mode: str, *, chunk_frames: int, group_owner_frames: int, chunk_count: int) -> float:
    """Scale one TBPTT chunk's per-head means.

    ``per-chunk`` keeps every chunk at unit scale (a tail chunk with a few
    long games gets a full optimizer step).  ``group-expected`` multiplies by
    ``chunk_frames / (group_owner_frames / chunk_count)`` so chunk scales
    average to one over a group and each chunk counts in proportion to the
    owner frames it actually contains.
    """

    if mode == "per-chunk":
        return 1.0
    if mode != "group-expected":
        raise ValueError(f"chunk normalization must be one of {CHUNK_NORMALIZATIONS}, got {mode!r}")
    if group_owner_frames <= 0 or chunk_count <= 0 or chunk_frames < 0:
        raise ValueError("group-expected chunk normalization needs positive group frames and chunk count")
    return float(chunk_frames) * float(chunk_count) / float(group_owner_frames)


def global_chunk_frames(global_group: Sequence[ShardTrainingPlanV1], *, start: int, time_steps: int) -> int:
    """Active owner frames of the chunk at ``start`` summed over every lane of the group (all ranks).

    Equals the all-rank sum of ``_masked_host_chunk``'s frame counts: real
    lanes count ``min(T, length - start)`` clipped at zero, dummy ranks none.
    """

    return sum(
        max(0, min(time_steps, length - start)) for plan in global_group for length in plan.sequence_lengths
    )


class _DistributedImitationModule(nn.Module):
    """Put the complete recurrent forward under DDP reducer bookkeeping."""

    def __init__(self, policy: UniversalCardPolicyV4) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        sequence: ILSequenceV4,
        state: RecurrentPolicyStateV4,
        *,
        validate: bool,
        preencode_observations: bool,
        fused_sequence: bool = False,
        compact_tail: bool = False,
    ) -> RecurrentEvaluationV4:
        return evaluate_recurrent_sequence(
            self.policy,
            sequence.observations,
            sequence.actions,
            sequence.episode_start,
            initial_state=state,
            gate_temperature=1.0,
            action_temperature=1.0,
            continue_temperature=1.0,
            validate=validate,
            preencode_observations=preencode_observations,
            fused=fused_sequence,
            **sequence_plan_options(sequence, compact_tail=compact_tail),
        )


def sequence_plan_options(sequence: ILSequenceV4, *, compact_tail: bool) -> dict[str, object]:
    """``evaluate_recurrent_sequence`` options from an ``ILChunkSequenceV4``'s host plan.

    The host-computed reset flag always replaces the device check.  With
    ``compact_tail`` repeated tail frames are encoded once (``lane_steps``)
    and only ``valid_mask`` rows are decoded: every IL loss and held-out
    statistic mask is a subset of ``valid_mask``.
    """

    if not isinstance(sequence, ILChunkSequenceV4):
        return {}
    options: dict[str, object] = {"mid_chunk_reset": sequence.mid_chunk_reset}
    if compact_tail:
        options["lane_steps"] = sequence.lane_steps
        options["decode_mask"] = sequence.valid_mask
    return options


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_replay_tags(path: Path) -> set[str]:
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not row for row in rows):
        raise ValueError(f"replay tag file is empty or malformed: {path}")
    result = set(rows)
    if len(result) != len(rows):
        raise ValueError(f"replay tag file contains duplicates: {path}")
    return result


def _index_gamma(index: Mapping[str, object], index_path: Path) -> float | None:
    """Return the cache contract's gamma_per_decision (None when the contract lacks it)."""

    contract = index.get("contract")
    if not isinstance(contract, Mapping) or "gamma_per_decision" not in contract:
        return None
    gamma = contract["gamma_per_decision"]
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(float(gamma)):
        raise ValueError(f"cache index contract gamma_per_decision is not a finite number: {index_path}")
    return float(gamma)


def _summary_paths(
    index_paths: Sequence[Path], gammas: dict[str, float | None] | None = None
) -> tuple[Path, ...]:
    """Resolve shard summaries; optionally record each index's contract gamma into ``gammas``."""

    result: list[Path] = []
    seen: set[Path] = set()
    for index_path in index_paths:
        index = load_il_cache_index(index_path)
        if gammas is not None:
            gammas[str(index_path)] = _index_gamma(index, index_path)
        raw_shards = index.get("shards")
        if not isinstance(raw_shards, list):
            raise ValueError(f"cache index has no shards: {index_path}")
        for raw in raw_shards:
            if not isinstance(raw, Mapping):
                raise ValueError(f"cache index has an invalid shard: {index_path}")
            name = raw.get("summary_file")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"cache shard has no local summary file: {index_path}")
            path = (index_path.parent / name).resolve()
            if path in seen:
                raise ValueError(f"duplicate cache summary path: {path}")
            seen.add(path)
            result.append(path)
    if not result:
        raise ValueError("no cache shard summaries were selected")
    return tuple(result)


def _units_from_manifest(manifest: Mapping[str, object], selected_tags: set[str]) -> tuple[ReplayUnitV1, ...]:
    raw_descriptors = manifest.get("sequences")
    if not isinstance(raw_descriptors, list):
        raise ValueError("cache manifest has no sequence descriptors")
    by_tag: dict[str, list[tuple[int, ILCacheSequenceDescriptorV1]]] = {}
    for index, raw in enumerate(raw_descriptors):
        if not isinstance(raw, dict):
            raise ValueError("cache sequence descriptor is not an object")
        descriptor = ILCacheSequenceDescriptorV1(**raw)
        if descriptor.replay_tag in selected_tags:
            by_tag.setdefault(descriptor.replay_tag, []).append((index, descriptor))
    result: list[ReplayUnitV1] = []
    for replay_tag, rows in by_tag.items():
        ordered = sorted(rows, key=lambda row: row[1].owner)
        if len(ordered) != 2 or [row[1].owner for row in ordered] != [0, 1]:
            raise ValueError(f"replay {replay_tag} does not have exactly two owner lanes")
        result.append(
            ReplayUnitV1(
                replay_tag=replay_tag,
                sequence_indices=cast(tuple[int, int], tuple(row[0] for row in ordered)),
                sequence_lengths=cast(tuple[int, int], tuple(row[1].length for row in ordered)),
            )
        )
    return tuple(result)


def build_training_plan(
    index_paths: Sequence[Path], train_tags_path: Path, *, manifest_workers: int
) -> tuple[tuple[ShardTrainingPlanV1, ...], dict[str, object]]:
    """Build an exact split plan, with later indexes overriding earlier copies."""

    if manifest_workers <= 0:
        raise ValueError("manifest_workers must be positive")
    selected_tags = _read_replay_tags(train_tags_path)
    gammas: dict[str, float | None] = {}
    paths = _summary_paths(index_paths, gammas)
    with ThreadPoolExecutor(max_workers=min(manifest_workers, len(paths))) as pool:
        manifests = tuple(pool.map(load_il_cache_manifest, paths))

    latest: dict[str, tuple[int, ReplayUnitV1]] = {}
    duplicate_tags: set[str] = set()
    for path_index, manifest in enumerate(manifests):
        for unit in _units_from_manifest(manifest, selected_tags):
            if unit.replay_tag in latest:
                duplicate_tags.add(unit.replay_tag)
            latest[unit.replay_tag] = (path_index, unit)
    missing = selected_tags.difference(latest)
    if missing:
        examples = ", ".join(sorted(missing)[:8])
        raise ValueError(f"training split has {len(missing)} replay tags absent from caches: {examples}")

    by_path: dict[int, list[ReplayUnitV1]] = {}
    for path_index, unit in latest.values():
        by_path.setdefault(path_index, []).append(unit)
    plans = tuple(
        ShardTrainingPlanV1(
            summary_path=paths[path_index], units=tuple(sorted(units, key=lambda unit: unit.sequence_indices))
        )
        for path_index, units in sorted(by_path.items())
    )
    replay_count = sum(plan.replay_count for plan in plans)
    if replay_count != len(selected_tags):
        raise RuntimeError("training plan replay count disagrees with the split")
    return plans, {
        "cache_indexes": [str(path) for path in index_paths],
        "cache_index_sha256": {str(path): _sha256(path) for path in index_paths},
        "cache_gamma_per_decision": gammas,
        "train_tags_path": str(train_tags_path),
        "train_tags_sha256": _sha256(train_tags_path),
        "selected_replays": replay_count,
        "selected_owner_sequences": replay_count * 2,
        "selected_owner_frames": sum(plan.owner_frames for plan in plans),
        "selected_shards": len(plans),
        "overridden_replay_copies": len(duplicate_tags),
    }


def _autocast_dtype(name: str) -> torch.dtype | None:
    return {"none": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _masked_host_chunk(
    shards: Sequence[ILCacheShardV1],
    plans: Sequence[ShardTrainingPlanV1],
    *,
    start: int,
    time_steps: int,
    dummy: bool,
    replay_weights: ReplayWeightsV1 | None = None,
    pin_memory: bool = False,
    trim_padding: bool = False,
) -> tuple[CollatedILChunkV1, int]:
    """One TBPTT chunk on the host with lanes past their sequence end masked out.

    ``trim_padding`` slices padded collection rows to the chunk's widest real
    use (``collate_il_cache_chunk``; same model outputs up to float rounding).
    """

    references: list[list[ILCacheChunkRefV1]] = []
    active_steps: list[int] = []
    lane_weights: list[float] = []
    for plan in plans:
        if replay_weights is not None:
            lane_weights.extend(replay_weights.plan_lane_weights(plan))
        shard_references: list[ILCacheChunkRefV1] = []
        for sequence_index, length in zip(plan.sequence_indices, plan.sequence_lengths, strict=True):
            steps = max(0, min(time_steps, length - start))
            reference_start = min(start, length - 1)
            shard_references.append(
                ILCacheChunkRefV1(sequence_index=sequence_index, start=reference_start, steps=max(1, steps))
            )
            active_steps.append(0 if dummy else steps)
        references.append(shard_references)
    chunk = collate_il_cache_chunk(
        shards, references, time_steps=time_steps, pin_memory=pin_memory, trim_padding=trim_padding
    )
    fields = chunk.time_fields
    returns = cast(Tensor, fields["returns"])
    active = torch.arange(time_steps)[:, None] < torch.tensor(active_steps)[None, :]
    masked: dict[str, Tensor | None] = {
        **fields,
        "episode_start": cast(Tensor, fields["episode_start"]) & active,
        "valid_mask": cast(Tensor, fields["valid_mask"]) & active,
        "returns": torch.where(active, returns, torch.zeros_like(returns)),
        "gate_loss_mask": None if (gate := fields["gate_loss_mask"]) is None else gate & active,
        "value_loss_mask": None if (value := fields["value_loss_mask"]) is None else value & active,
    }
    if replay_weights is not None:
        lanes = torch.tensor(lane_weights, dtype=torch.float32)[None, :].expand(time_steps, -1)
        masked["frame_weight"] = torch.where(active, lanes, torch.zeros_like(lanes)).contiguous()
    mid_chunk_reset = bool(cast(Tensor, masked["episode_start"])[1:].any())
    if pin_memory:
        masked = {name: None if tensor is None else tensor.pin_memory() for name, tensor in masked.items()}
    return replace(chunk, time_fields=masked, mid_chunk_reset=mid_chunk_reset), sum(active_steps)


def _masked_chunk(
    shards: Sequence[ILCacheShardV1],
    plans: Sequence[ShardTrainingPlanV1],
    *,
    start: int,
    time_steps: int,
    device: torch.device,
    dummy: bool,
    replay_weights: ReplayWeightsV1 | None = None,
    trim_padding: bool = False,
    plan: bool = False,
) -> tuple[ILSequenceV4, int]:
    """``plan=True`` returns an ``ILChunkSequenceV4`` carrying the host chunk plan."""

    chunk, frames = _masked_host_chunk(
        shards,
        plans,
        start=start,
        time_steps=time_steps,
        dummy=dummy,
        replay_weights=replay_weights,
        trim_padding=trim_padding,
    )
    return chunk.to_model_input(device).sequence(plan=plan), frames


@dataclass(frozen=True, slots=True)
class _PrefetchFailure:
    error: BaseException


class _ChunkPrefetcher:
    """Prepare a group's TBPTT chunks up to ``depth`` ahead on a background thread.

    The thread gathers each chunk into pinned host memory, issues its device
    copy on a side CUDA stream and records an event; ``__next__`` makes the
    consumer's current stream wait on that event.  Chunks arrive in order with
    the contents of the synchronous ``_masked_chunk`` path, while slicing,
    padding and the host-to-device copy overlap the previous chunk's compute.
    Up to ``depth`` prepared chunks are resident on the device at once.
    """

    def __init__(
        self,
        produce: Callable[[int], tuple[CollatedILChunkV1, int]],
        *,
        chunk_count: int,
        depth: int,
        device: torch.device,
        plan: bool = False,
    ) -> None:
        if depth <= 0:
            raise ValueError("chunk prefetch depth must be positive")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._plan = plan
        self._produce = produce
        self._chunk_count = chunk_count
        self._device = device
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._queue: queue.Queue[object] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._consumed = 0
        self._thread = threading.Thread(target=self._run, name="il-chunk-prefetch", daemon=True)
        self._thread.start()

    def _put(self, item: object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            if self._stream is not None:
                torch.cuda.set_device(self._device)
            for index in range(self._chunk_count):
                if self._stop.is_set():
                    return
                host, frames = self._produce(index)
                event = None
                if self._stream is None:
                    chunk = host.to_model_input(self._device)
                else:
                    with torch.cuda.stream(self._stream):
                        chunk = host.to_model_input(self._device, non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(self._stream)
                del host
                if not self._put((chunk, frames, event)):
                    return
        except BaseException as error:  # re-raised on the consumer thread
            self._put(_PrefetchFailure(error))

    def __iter__(self) -> _ChunkPrefetcher:
        return self

    def __next__(self) -> tuple[ILSequenceV4, int]:
        if self._consumed >= self._chunk_count:
            raise StopIteration
        item = self._queue.get()
        if isinstance(item, _PrefetchFailure):
            self._consumed = self._chunk_count
            raise item.error
        chunk, frames, event = cast(tuple[CollatedILChunkV1, int, "torch.cuda.Event | None"], item)
        if event is not None:
            stream = torch.cuda.current_stream(self._device)
            stream.wait_event(event)
            # Allocated on the side stream: keep the allocator from reusing
            # these blocks before the consumer stream is done with them.
            for tensor in chunk.tensors():
                tensor.record_stream(stream)
        self._consumed += 1
        return chunk.sequence(plan=self._plan), frames

    def close(self) -> None:
        self._stop.set()
        while self._thread.is_alive():
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(timeout=0.1)


def _group_chunks(
    shards: Sequence[ILCacheShardV1],
    plans: Sequence[ShardTrainingPlanV1],
    *,
    chunk_count: int,
    time_steps: int,
    device: torch.device,
    dummy: bool,
    replay_weights: ReplayWeightsV1 | None = None,
    prefetch_chunks: int = 0,
    trim_padding: bool = False,
    plan: bool = False,
) -> Iterator[tuple[ILSequenceV4, int]]:
    """Yield a group's ``(chunk, local owner frames)`` in order; call ``close()`` when done.

    ``prefetch_chunks=0`` prepares every chunk on the calling thread just
    before it is used; a positive depth starts a ``_ChunkPrefetcher`` now.
    ``trim_padding`` trims each chunk's padded collection rows (identical
    model outputs); ``plan`` yields ``ILChunkSequenceV4`` chunks.
    """

    if prefetch_chunks < 0:
        raise ValueError("prefetch_chunks must be non-negative")
    if prefetch_chunks == 0:
        return (
            _masked_chunk(
                shards,
                plans,
                start=chunk_index * time_steps,
                time_steps=time_steps,
                device=device,
                dummy=dummy,
                replay_weights=replay_weights,
                trim_padding=trim_padding,
                plan=plan,
            )
            for chunk_index in range(chunk_count)
        )
    pin_memory = device.type == "cuda"
    return _ChunkPrefetcher(
        lambda chunk_index: _masked_host_chunk(
            shards,
            plans,
            start=chunk_index * time_steps,
            time_steps=time_steps,
            dummy=dummy,
            replay_weights=replay_weights,
            pin_memory=pin_memory,
            trim_padding=trim_padding,
        ),
        chunk_count=chunk_count,
        depth=prefetch_chunks,
        device=device,
        plan=plan,
    )


def _available_host_memory_bytes() -> int | None:
    """``MemAvailable`` on Linux (psutil elsewhere); ``None`` when unknown."""

    try:
        with open("/proc/meminfo", encoding="ascii") as source:
            for line in source:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def _group_load_bytes(summary_paths: Sequence[Path], *, workers: int) -> int:
    """Resident bytes of a loaded group plus the per-worker load transient.

    The loader inflates straight into each shard's final buffer, so a worker
    only additionally holds its shard's compressed bytes while it works.
    """

    sizes = []
    compressed = []
    for path in summary_paths:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("schema") != IL_CACHE_SUMMARY_SCHEMA:
            raise ValueError(f"unsupported V4 IL cache summary: {path}")
        sizes.append(int(summary["container_bytes"]))
        compressed.append(int(summary.get("compressed_bytes", 0)))
    return sum(sizes) + min(workers, len(sizes)) * max(compressed)


def _shard_bytes(shards: Sequence[ILCacheShardV1]) -> int:
    return sum(int(shard.summary.get("container_bytes", 0)) for shard in shards)


class _GroupShardLoader:
    """Load each scheduled group's shards in order, optionally one group ahead.

    With ``preload_next`` the next group is read (zstd, SHA-256, tensor views)
    on a background thread while the current group trains, so two groups are
    resident at once.  ``mode="process"`` decodes those preloads in ``workers``
    spawned processes into shared memory (no shard work touches this
    interpreter's GIL); ``"thread"`` uses ``workers`` threads, whose decoding
    also runs with the GIL released.  If the process pool cannot start or
    breaks, or ``/dev/shm`` lacks room for a group, that group (or, for a
    failure, every later group) is loaded with threads instead.

    Each preload is gated on host memory: the next group's bytes times
    ``local_world_size`` (every rank on the node preloads at the same time)
    must fit in 90% of ``MemAvailable`` plus the bytes of groups handed to
    ``release`` whose release is still in flight; otherwise that group is
    loaded synchronously when it is needed.  ``release`` frees a finished
    group on the preload thread (ahead of the next preload) instead of the
    training thread.
    """

    def __init__(
        self,
        path_groups: Sequence[Sequence[Path]],
        *,
        workers: int,
        preload_next: bool,
        mode: str = "thread",
        verify: bool = True,
        local_world_size: int = 1,
        available_memory: Callable[[], int | None] = _available_host_memory_bytes,
        shared_memory: Callable[[], int | None] = shared_memory_free_bytes,
        process_loader_factory: Callable[[int], ILCacheProcessLoader] = ILCacheProcessLoader,
    ) -> None:
        if mode not in SHARD_LOAD_MODES:
            raise ValueError(f"unknown shard load mode {mode!r}")
        self._paths = tuple(tuple(paths) for paths in path_groups)
        self._workers = workers
        self._verify = verify
        self._local_world_size = max(1, local_world_size)
        self._available_memory = available_memory
        self._shared_memory = shared_memory
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="il-group-preload") if preload_next else None
        )
        self._pending: dict[int, Future[tuple[tuple[ILCacheShardV1, ...], str]]] = {}
        self._release_lock = threading.Lock()
        self._releasing_bytes = 0
        self._process: ILCacheProcessLoader | None = None
        self.process_error: str | None = None
        self.requested_mode = mode
        if self._executor is not None and mode == "process" and len(self._paths) > 1:
            # Workers spawn lazily on the first preload (in the background).
            pool_size = min(workers, max(len(paths) for paths in self._paths))
            try:
                self._process = process_loader_factory(pool_size)
            except Exception as error:  # noqa: BLE001 - fall back to threads
                self.process_error = f"{type(error).__name__}: {error}"
        self.last_preload: dict[str, object] = {"next_group_preload": "disabled" if not preload_next else "none"}

    @property
    def mode(self) -> str:
        """The mode preloads currently use (``process`` falls back to ``thread``)."""

        return "process" if self._process is not None else "thread"

    def describe(self) -> dict[str, object]:
        row: dict[str, object] = {
            "preload_workers_mode": self.requested_mode,
            "preload_workers_mode_effective": self.mode if self._executor is not None else "disabled",
            "load_workers": self._workers,
        }
        if self._process is not None:
            row["preload_sharing_strategy"] = self._process.sharing_strategy
            row["preload_start_method"] = self._process.start_method
        if self.process_error is not None:
            row["preload_process_error"] = self.process_error
        return row

    def _disable_processes(self, error: BaseException) -> None:
        self.process_error = f"{type(error).__name__}: {error}"
        process, self._process = self._process, None
        if process is not None:
            try:
                process.close()
            except Exception:  # noqa: BLE001 - already broken
                pass

    def _load_threads(self, index: int) -> tuple[ILCacheShardV1, ...]:
        return load_il_cache_shards(self._paths[index], verify=self._verify, workers=self._workers)

    def _preload(self, index: int) -> tuple[tuple[ILCacheShardV1, ...], str]:
        process = self._process
        if process is None:
            return self._load_threads(index), "thread"
        free = self._shared_memory()
        required = _group_load_bytes(self._paths[index], workers=0)
        if free is not None and required * self._local_world_size > 0.9 * free:
            return self._load_threads(index), "thread-shm"
        try:
            return process.load(self._paths[index], verify=self._verify), "process"
        except ValueError:
            raise  # cache content errors (SHA-256, layout) are not loader failures
        except Exception as error:  # noqa: BLE001 - broken pool, shm allocation, spawn
            self._disable_processes(error)
            return self._load_threads(index), "thread-fallback"

    def get(self, index: int) -> tuple[tuple[ILCacheShardV1, ...], bool]:
        """Return group ``index``'s shards and whether they were preloaded."""

        future = self._pending.pop(index, None)
        for stale in self._pending.values():
            stale.cancel()
        self._pending.clear()
        if future is None:
            shards, loader_mode = self._load_threads(index), "sync-thread"
        else:
            shards, loader_mode = future.result()
        if self._executor is not None:
            self._start_preload(index + 1)
        self.last_preload["cache_loader_mode"] = loader_mode
        if self.process_error is not None:
            self.last_preload["preload_process_error"] = self.process_error
        return shards, future is not None

    def release(self, shards: tuple[ILCacheShardV1, ...]) -> None:
        """Free a finished group's shards on the preload thread, ahead of the next preload.

        The caller must drop every other reference (e.g. its chunk prefetcher).
        """

        if self._executor is None:
            return
        size = _shard_bytes(shards)
        holder = [shards]
        del shards
        with self._release_lock:
            self._releasing_bytes += size

        def drop() -> None:
            holder.clear()
            with self._release_lock:
                self._releasing_bytes -= size

        self._executor.submit(drop)

    def _start_preload(self, index: int) -> None:
        if index >= len(self._paths):
            self.last_preload = {"next_group_preload": "none"}
            return
        required = _group_load_bytes(self._paths[index], workers=self._workers)
        available = self._available_memory()
        with self._release_lock:
            releasing = self._releasing_bytes
        row: dict[str, object] = {"preload_required_bytes": required, "host_available_bytes": available}
        if releasing:
            row["host_releasing_bytes"] = releasing
        if available is not None and required * self._local_world_size > 0.9 * (available + releasing):
            self.last_preload = {"next_group_preload": "skipped-memory", **row}
            return
        assert self._executor is not None
        self._pending[index] = self._executor.submit(self._preload, index)
        self.last_preload = {"next_group_preload": "started", "preload_loader_mode": self.mode, **row}

    def close(self) -> None:
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._process is not None:
            self._process.close()
            self._process = None


# (loss, unweighted count, weighted count = mean denominator)
_LOSS_HEADS = (
    ("gate", "gate_count", "gate_weighted_count"),
    ("candidate", "candidate_count", "candidate_weighted_count"),
    ("target", "target_count", "target_weighted_count"),
    ("delay", "delay_count", "delay_weighted_count"),
    ("continue_action", "continue_count", "continue_weighted_count"),
    ("value", "value_count", "value_weighted_count"),
)
_CONTINUE_HEAD = next(index for index, (name, _count, _weighted) in enumerate(_LOSS_HEADS) if name == "continue_action")
# Logged-only SECOND-label statistics, reduced alongside the head sums.
_CONTINUE_SECOND_STATS = ("continue_second_count", "continue_second_weighted_count")


def _local_second_stats(losses: ILLossV4) -> Tensor:
    return torch.stack([getattr(losses, name).detach().to(torch.float32) for name in _CONTINUE_SECOND_STATS])


def _population_scale(local_weight: Tensor, global_weight: Tensor, *, world_size: int) -> Tensor:
    """``local_weight * world_size / global_weight`` for a populated head, else exactly 0 (no host branch).

    For ``global_weight > 0`` this is the same fp32 expression the per-head
    host branch evaluated; an empty head (``local_weight`` is then 0 too)
    contributes ``loss * 0``, which leaves the total and the gradients unchanged.
    """

    populated = global_weight > 0.0
    safe = torch.where(populated, global_weight, torch.ones_like(global_weight))
    return torch.where(populated, local_weight * float(world_size) / safe, torch.zeros_like(local_weight))


def _distributed_loss_parts(
    losses: ILLossV4, weights: ILLossWeightsV4, *, world_size: int, loss_scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """``_distributed_loss`` without host syncs: ``(total, fp32 device stats)``.

    ``stats`` is ``[ratios, counts, weighted, second]`` of the global heads
    (``ratio = global sum / global weighted population``, fp32), turned into
    the metrics dict on the host by ``_loss_metrics``.
    """

    if not math.isfinite(loss_scale) or loss_scale < 0.0:
        raise ValueError("chunk loss scale must be finite and non-negative")
    total = losses.total * 0.0
    local_counts = torch.stack(
        [getattr(losses, count_name).detach().to(torch.float32) for _loss_name, count_name, _weighted in _LOSS_HEADS]
    )
    local_weighted = torch.stack(
        [getattr(losses, weighted).detach().to(torch.float32) for _loss_name, _count_name, weighted in _LOSS_HEADS]
    )
    local_sums = torch.stack(
        [
            getattr(losses, loss_name).detach().to(torch.float32) * local_weighted[index]
            for index, (loss_name, _count_name, _weighted) in enumerate(_LOSS_HEADS)
        ]
    )
    head_count = len(_LOSS_HEADS)
    global_stats = torch.cat((local_counts, local_weighted, local_sums, _local_second_stats(losses)))
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
    global_counts, global_weighted, global_sums = global_stats[: 3 * head_count].chunk(3)
    global_second = global_stats[3 * head_count :]
    for index, (loss_name, _count_name, _weighted_name) in enumerate(_LOSS_HEADS):
        scale = _population_scale(local_weighted[index], global_weighted[index], world_size=world_size)
        total = total + getattr(weights, loss_name) * getattr(losses, loss_name) * scale
    if loss_scale != 1.0:
        total = total * loss_scale
    return total, _head_stats(global_sums, global_counts, global_weighted, global_second)


def _head_stats(global_sums: Tensor, global_counts: Tensor, global_weighted: Tensor, global_second: Tensor) -> Tensor:
    safe = torch.where(global_weighted > 0.0, global_weighted, torch.ones_like(global_weighted))
    return torch.cat((global_sums / safe, global_counts, global_weighted, global_second))


def _loss_metrics(stats: Sequence[float], loss_scale: float) -> dict[str, float]:
    """Host metrics from ``_head_stats`` values (same numbers and key order as the per-head ``.item()`` path)."""

    head_count = len(_LOSS_HEADS)
    ratios = stats[:head_count]
    counts = stats[head_count : 2 * head_count]
    weighted = stats[2 * head_count : 3 * head_count]
    second_count, second_weighted = stats[3 * head_count : 3 * head_count + 2]
    metrics: dict[str, float] = {}
    for index, (loss_name, count_name, _weighted_name) in enumerate(_LOSS_HEADS):
        metrics[loss_name] = float(ratios[index]) if weighted[index] > 0.0 else 0.0
        metrics[count_name] = float(counts[index])
    # ``continue_second_count`` (unweighted) and the SECOND share of the
    # continue head's weighted population.
    continue_weighted = float(weighted[_CONTINUE_HEAD])
    metrics["continue_second_count"] = float(second_count)
    metrics["continue_second_weighted_fraction"] = (
        float(second_weighted) / continue_weighted if continue_weighted > 0.0 else 0.0
    )
    metrics["chunk_loss_scale"] = float(loss_scale)
    return metrics


def _distributed_loss(
    losses: ILLossV4, weights: ILLossWeightsV4, *, world_size: int, loss_scale: float = 1.0
) -> tuple[Tensor, dict[str, float]]:
    """Globally normalized loss; ``loss_scale`` is the chunk normalization factor.

    Each head's local mean is re-weighted by its share of the global weighted
    population, so the DDP gradient average equals the global weighted mean.
    """

    total, stats = _distributed_loss_parts(losses, weights, world_size=world_size, loss_scale=loss_scale)
    return total, _loss_metrics(stats.tolist(), loss_scale)


@dataclass
class PendingChunkMetricsV1:
    """A chunk's device-side results, read back with one host sync by ``resolve``."""

    stats: Tensor
    finite: Tensor
    consistent: Tensor | None
    loss_scale: float
    context: str

    def resolve(self, *extra: Tensor) -> tuple[dict[str, float], list[float]]:
        """Check finiteness (and micro-batch population agreement); return ``(metrics, extra values)``.

        ``extra`` scalars (e.g. the gradient norm) ride on the same transfer.
        """

        parts = [self.finite.reshape(1).to(torch.float64), self.stats.to(torch.float64)]
        if self.consistent is not None:
            parts.append(self.consistent.reshape(1).to(torch.float64))
        parts.extend(item.detach().reshape(1).to(device=self.stats.device, dtype=torch.float64) for item in extra)
        values = torch.cat(parts).tolist()
        if not int(values[0]):
            raise FloatingPointError(f"non-finite distributed IL loss at {self.context}")
        stats_end = 1 + int(self.stats.numel())
        if self.consistent is not None:
            if not int(values[stats_end]):
                raise RuntimeError("micro-batch loss populations disagree with imitation_loss populations")
            stats_end += 1
        return _loss_metrics(values[1 : 1 + int(self.stats.numel())], self.loss_scale), values[stats_end:]


def lane_slices(batch_size: int, micro_batches: int) -> tuple[tuple[int, int], ...]:
    """Contiguous ``(start, stop)`` lane ranges, sizes differing by at most one.

    A chunk with fewer lanes than ``micro_batches`` (a dummy tail rank) gets
    one lane per slice.
    """

    if batch_size <= 0 or micro_batches <= 0:
        raise ValueError("lane slicing needs a positive batch size and micro-batch count")
    count = min(micro_batches, batch_size)
    base, extra = divmod(batch_size, count)
    result = []
    start = 0
    for index in range(count):
        stop = start + base + int(index < extra)
        result.append((start, stop))
        start = stop
    return tuple(result)


def slice_sequence_lanes(sequence: ILSequenceV4, start: int, stop: int) -> ILSequenceV4:
    """Lanes ``[start, stop)`` of a ``[T,B]`` chunk (observation/action records are zero-copy views)."""

    batch_size = sequence.batch_size
    if not 0 <= start < stop <= batch_size:
        raise ValueError("lane slice must be a non-empty in-range interval")
    length = stop - start
    plan: dict[str, object] = {}
    if isinstance(sequence, ILChunkSequenceV4):
        # A chunk without resets has none in any lane slice; otherwise the
        # slice's own episode_start decides on the device.
        plan["lane_steps"] = None if sequence.lane_steps is None else tuple(sequence.lane_steps[start:stop])
        plan["mid_chunk_reset"] = False if sequence.mid_chunk_reset is False else None
    return replace(
        sequence,
        observations=tuple(item.narrow_batch(start, length) for item in sequence.observations),
        actions=tuple(item.narrow_batch(start, length) for item in sequence.actions),
        **{
            name: None if (value := getattr(sequence, name)) is None else value[:, start:stop].contiguous()
            for name in IL_TIME_FIELDS + IL_ANNOTATION_FIELDS
        },
        **plan,
    )


def loss_populations(sequence: ILSequenceV4, *, continue_second_weight: float = 1.0) -> tuple[Tensor, Tensor]:
    """Per-head ``(counts, weighted counts)`` in ``_LOSS_HEADS`` order, float32.

    The populations ``imitation_loss`` normalizes by, computed from the chunk
    alone (they never depend on model outputs), so a micro-batched step can
    reduce the global denominators before any slice runs backward.
    ``continue_second_weight`` scales SECOND labels in the continue head's
    weighted population exactly as ``imitation_loss`` does.
    """

    max_micro_actions = sequence.actions[0].candidate_index.shape[1]
    micro_counts = torch.stack([item.micro_action_count for item in sequence.actions])
    micro = (
        torch.arange(max_micro_actions, device=micro_counts.device)[None, None, :] < micro_counts[:, :, None]
    ) & sequence.valid_mask[:, :, None]
    target_cells = torch.stack([item.target_cell for item in sequence.actions])
    gates = torch.stack([item.gate for item in sequence.actions])
    masks = (
        sequence.valid_mask if sequence.gate_loss_mask is None else sequence.gate_loss_mask,  # gate
        micro,  # candidate
        micro & (target_cells >= 0),  # target
        micro,  # delay
        sequence.valid_mask & (gates == GATE_ACT),  # continue
        sequence.valid_mask if sequence.value_loss_mask is None else sequence.value_loss_mask,  # value
    )
    frame_weight = sequence.frame_weight
    continue_class_weight = continue_label_class_weight(micro_counts, continue_second_weight)
    counts = []
    weighted = []
    for index, mask in enumerate(masks):
        valid = mask.to(torch.float32)
        counts.append(valid.sum())
        class_weight = continue_class_weight if index == _CONTINUE_HEAD else None
        if frame_weight is None and class_weight is None:
            weighted.append(counts[-1])
            continue
        weight = valid
        if frame_weight is not None:
            weight = weight * frame_weight.to(torch.float32).reshape(*frame_weight.shape, *((1,) * (mask.ndim - 2)))
        if class_weight is not None:
            weight = weight * class_weight
        weighted.append(weight.sum())
    return torch.stack(counts), torch.stack(weighted)


def chunk_gradients(
    module: nn.Module,
    sequence: ILSequenceV4,
    state: RecurrentPolicyStateV4,
    *,
    weights: ILLossWeightsV4,
    world_size: int,
    loss_scale: float = 1.0,
    micro_batches: int = 1,
    autocast_dtype: torch.dtype | None = None,
    validate: bool = True,
    preencode_observations: bool = True,
    fused_sequence: bool = True,
    gate_act_weight: float = DEFAULT_IL_GATE_ACT_WEIGHT,
    delay_neighbor_weight: float = DEFAULT_IL_DELAY_NEIGHBOR_WEIGHT,
    context: str = "",
    compact_tail: bool = False,
    defer_host_sync: bool = False,
) -> tuple[RecurrentPolicyStateV4, dict[str, float] | PendingChunkMetricsV1]:
    """Accumulate one TBPTT chunk's gradients; return (detached final state, chunk metrics).

    ``compact_tail`` (for ``ILChunkSequenceV4`` chunks) encodes repeated tail
    frames once and decodes only valid rows (``sequence_plan_options``).
    ``defer_host_sync`` returns a ``PendingChunkMetricsV1`` instead of the
    metrics dict: no host sync happens here, and the caller's ``resolve()``
    (after backward) raises the non-finite-loss error and yields the same
    metrics, so a training step needs a single host round trip.

    ``module`` is the (DDP-wrapped) ``_DistributedImitationModule``.  With
    ``micro_batches > 1`` the chunk's lanes run as contiguous slices, each a
    forward plus backward, so only one slice's activations are alive at a
    time.  The global per-head weighted populations are reduced first from
    the chunk's masks; every slice then contributes
    ``weight * slice_mean * slice_population * world_size / global_population``
    per head, whose sum over slices is exactly the single-pass
    ``_distributed_loss`` total.  All but the last slice run under
    ``no_sync`` so DDP all-reduces the accumulated gradients once.  Logged
    metrics come from the accumulated per-head sums and counts, i.e. the
    single-pass numbers.
    """

    if micro_batches <= 0:
        raise ValueError("micro_batches must be positive")
    device_type = sequence.valid_mask.device.type
    autocast = torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None)
    forward_options: dict[str, object] = {
        "validate": validate,
        "preencode_observations": preencode_observations,
        "fused_sequence": fused_sequence,
    }
    if compact_tail:
        forward_options["compact_tail"] = True
    loss_options = {"weights": weights, "gate_act_weight": gate_act_weight, "delay_neighbor_weight": delay_neighbor_weight}
    if micro_batches == 1:
        with autocast:
            evaluation = module(sequence, state, **forward_options)
            losses = imitation_loss(sequence, evaluation, **loss_options)
            loss, stats = _distributed_loss_parts(losses, weights, world_size=world_size, loss_scale=loss_scale)
        finite = torch.isfinite(loss).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        pending = PendingChunkMetricsV1(stats=stats, finite=finite, consistent=None, loss_scale=loss_scale, context=context)
        if defer_host_sync:
            loss.backward()
            return evaluation.final_state.detach(), pending
        metrics, _extra = pending.resolve()
        loss.backward()
        return evaluation.final_state.detach(), metrics

    if not math.isfinite(loss_scale) or loss_scale < 0.0:
        raise ValueError("chunk loss scale must be finite and non-negative")
    local_counts, local_weighted = loss_populations(sequence, continue_second_weight=weights.continue_second_weight)
    global_weighted = local_weighted.clone()
    dist.all_reduce(global_weighted, op=dist.ReduceOp.SUM)
    head_count = len(_LOSS_HEADS)
    slice_counts = torch.zeros(head_count, dtype=torch.float32, device=local_counts.device)
    slice_weighted = torch.zeros_like(slice_counts)
    slice_sums = torch.zeros_like(slice_counts)
    slice_second = torch.zeros(len(_CONTINUE_SECOND_STATS), dtype=torch.float32, device=local_counts.device)
    finite = torch.ones((), dtype=torch.int32, device=local_counts.device)
    final_states: list[RecurrentPolicyStateV4] = []
    no_sync = getattr(module, "no_sync", None)
    slices = lane_slices(sequence.batch_size, micro_batches)
    for slice_index, (start, stop) in enumerate(slices):
        lanes = slice_sequence_lanes(sequence, start, stop)
        lane_state = state.narrow_batch(start, stop - start)
        sync = nullcontext() if no_sync is None or slice_index + 1 == len(slices) else no_sync()
        with sync:
            with autocast:
                evaluation = module(lanes, lane_state, **forward_options)
                losses = imitation_loss(lanes, evaluation, **loss_options)
                slice_second += _local_second_stats(losses)
                total = losses.total * 0.0
                for index, (loss_name, count_name, weighted_name) in enumerate(_LOSS_HEADS):
                    head_loss = getattr(losses, loss_name)
                    head_weight = getattr(losses, weighted_name).detach().to(torch.float32)
                    slice_counts[index] += getattr(losses, count_name).detach().to(torch.float32)
                    slice_weighted[index] += head_weight
                    slice_sums[index] += head_loss.detach().to(torch.float32) * head_weight
                    scale = _population_scale(head_weight, global_weighted[index], world_size=world_size)
                    total = total + getattr(weights, loss_name) * head_loss * scale
                if loss_scale != 1.0:
                    total = total * loss_scale
            finite = finite & torch.isfinite(total).all().to(torch.int32)
            total.backward()
        final_states.append(evaluation.final_state.detach())
        del lanes, lane_state, evaluation, losses, total

    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    # torch.allclose / torch.equal, evaluated on the device (read back by resolve()).
    consistent = torch.isclose(slice_weighted, local_weighted, rtol=1e-5, atol=1e-3).all() & torch.eq(
        slice_counts, local_counts
    ).all()
    global_stats = torch.cat((slice_counts, slice_sums, slice_second))
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
    global_counts, global_sums = global_stats[: 2 * head_count].chunk(2)
    global_second = global_stats[2 * head_count :]
    pending = PendingChunkMetricsV1(
        stats=_head_stats(global_sums, global_counts, global_weighted, global_second),
        finite=finite,
        consistent=consistent,
        loss_scale=loss_scale,
        context=context,
    )
    final_state = RecurrentPolicyStateV4(
        hidden=torch.cat([item.hidden for item in final_states]),
        cell=torch.cat([item.cell for item in final_states]),
    )
    if defer_host_sync:
        return final_state, pending
    metrics, _extra = pending.resolve()
    return final_state, metrics


def _broadcast_training_plan(
    rank: int, index_paths: Sequence[Path], train_tags_path: Path, manifest_workers: int
) -> tuple[tuple[ShardTrainingPlanV1, ...], dict[str, object]]:
    payload: list[object | None] = [None]
    if rank == 0:
        try:
            plans, metadata = build_training_plan(index_paths, train_tags_path, manifest_workers=manifest_workers)
            payload[0] = {"plans": plans, "metadata": metadata}
        except Exception as error:  # broadcast the failure so peers do not hang
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("rank 0 did not broadcast a training plan")
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return (cast(tuple[ShardTrainingPlanV1, ...], result["plans"]), cast(dict[str, object], result["metadata"]))


def _epoch_groups(
    plans: Sequence[ShardTrainingPlanV1], *, seed: int, epoch: int, group_size: int
) -> tuple[tuple[ShardTrainingPlanV1, ...], ...]:
    # TBPTT still computes padded lanes, so mixing a 300-frame shard with a
    # 1,200-frame shard wastes most of the tail.  Shuffle ties first, then
    # create length-homogeneous groups and randomize group order.  Every replay
    # remains present exactly once; only the epoch traversal order changes.
    rng = random.Random(seed + epoch)
    ordered = list(plans)
    rng.shuffle(ordered)
    ordered.sort(key=lambda plan: max(plan.sequence_lengths))
    groups = [tuple(ordered[start : start + group_size]) for start in range(0, len(ordered), group_size)]
    rng.shuffle(groups)
    return tuple(groups)


def _local_group_plans(
    global_group: Sequence[ShardTrainingPlanV1],
    plans: Sequence[ShardTrainingPlanV1],
    *,
    rank: int,
    shards_per_rank: int,
) -> tuple[tuple[ShardTrainingPlanV1, ...], bool]:
    """This rank's slice of a group; a rank past the tail trains one masked dummy replay."""

    local_start = rank * shards_per_rank
    local_plans = tuple(global_group[local_start : local_start + shards_per_rank])
    if local_plans:
        return local_plans, False
    source = plans[0]
    return (ShardTrainingPlanV1(summary_path=source.summary_path, units=(source.units[0],)),), True


def _write_json_line(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True) + "\n")


def _nonnegative_finite_float(value: str) -> float:
    """argparse ``type`` for loss weights: a finite float >= 0."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"must be finite and non-negative, got {value!r}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--train-tags", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--shards-per-rank", type=int, default=6)
    parser.add_argument("--manifest-workers", type=int, default=32)
    parser.add_argument("--load-workers", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gate-act-weight", type=float, default=DEFAULT_IL_GATE_ACT_WEIGHT)
    parser.add_argument("--delay-neighbor-weight", type=float, default=DEFAULT_IL_DELAY_NEIGHBOR_WEIGHT)
    parser.add_argument(
        "--delay-weight",
        type=float,
        default=1.0,
        help="loss weight of the delay head (0 disables it; the delay label is the touch phase within the decision window)",
    )
    parser.add_argument(
        "--continue-second-weight",
        type=_nonnegative_finite_float,
        default=1.0,
        help="class weight of SECOND labels inside the continue head's weighted mean (numerator and denominator; "
        "STOP keeps 1.0; multiplies replay frame weights; 1.0 = unweighted, bit-identical; continue_count unchanged)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="must equal the cache contract gamma_per_decision (value targets)"
    )
    parser.add_argument(
        "--model-config-json",
        help='ModelConfigV4 overrides as inline JSON (\'{"lstm_hidden_dim": 1024}\') or @path/to/overrides.json',
    )
    parser.add_argument(
        "--replay-weights", type=Path, help="TSV with header replay_tag<TAB>owner<TAB>weight; owner is 0, 1 or *"
    )
    parser.add_argument(
        "--replay-weights-strict",
        action="store_true",
        help="fail when a training lane has no replay weight row (default: weight 1.0)",
    )
    parser.add_argument("--chunk-normalization", choices=CHUNK_NORMALIZATIONS, default="group-expected")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-groups", type=int, default=50)
    parser.add_argument("--max-groups", type=int, help="limit each epoch for an explicit end-to-end smoke run")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--autocast", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preencode-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fused-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate each TBPTT chunk with one fused LSTM call and one ACT-row decode (needs "
        "--preencode-observations; chunks with a mid-chunk reset fall back to the stepped decoder)",
    )
    parser.add_argument(
        "--micro-batches",
        type=int,
        default=1,
        help="split each chunk's lanes into this many contiguous slices, each a forward+backward with gradient "
        "accumulation (one optimizer step per chunk; same loss, gradients and metrics, lower activation peak)",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute the --activation-checkpointing-scope activations in backward instead of keeping them "
        "(exact math; training only)",
    )
    parser.add_argument(
        "--activation-checkpointing-scope",
        default=",".join(DEFAULT_ACTIVATION_CHECKPOINT_SCOPES),
        help=f"comma-separated subset of {','.join(ACTIVATION_CHECKPOINT_SCOPES)}",
    )
    parser.add_argument(
        "--prefetch-chunks",
        type=int,
        default=2,
        help="TBPTT chunks a background thread prepares ahead (pinned gather + async copy); 0 = on the training thread",
    )
    parser.add_argument(
        "--trim-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cut each chunk's padded group/child/effect/event/edge rows to the chunk's widest real use "
        "before the copy and the forward (same losses and gradients up to float rounding; far less padding compute)",
    )
    parser.add_argument(
        "--compact-tail-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode the repeated frames past a lane's sequence end once per lane and decode only valid rows "
        "(the LSTM still runs over every frame; same losses, gradients and carried state up to float rounding)",
    )
    parser.add_argument(
        "--preload-next-group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read the next group's shards while the current group trains (two groups resident: about twice the "
        "host RAM; skipped for a group whose bytes x LOCAL_WORLD_SIZE exceed 90%% of MemAvailable)",
    )
    parser.add_argument(
        "--preload-workers-mode",
        choices=SHARD_LOAD_MODES,
        default="thread",
        help="where the next-group preload decodes shards (both decode with the GIL released and give identical "
        "tensors): 'thread' = --load-workers threads in this process; 'process' = --load-workers spawned "
        "processes handing each shard back as one shared-memory segment (/dev/shm; falls back to threads for a "
        "group when /dev/shm is too small, and for the rest of the run if the pool fails)",
    )
    args = parser.parse_args()
    positive = (
        args.epochs,
        args.time_steps,
        args.shards_per_rank,
        args.manifest_workers,
        args.load_workers,
        args.torch_threads,
        args.learning_rate,
        args.max_grad_norm,
        args.gate_act_weight,
        args.gamma,
        args.log_every,
        args.save_every_groups,
        args.expected_world_size,
        args.micro_batches,
    )
    if any(value <= 0 for value in positive) or args.weight_decay < 0:
        raise ValueError("training counts and rates must be positive")
    if not 0.0 <= args.delay_neighbor_weight <= 1.0 / 3.0:
        raise ValueError("delay neighbor weight must be in [0, 1/3]")
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("max_groups must be positive")
    if args.prefetch_chunks < 0:
        raise ValueError("prefetch_chunks must be non-negative")
    if args.replay_weights_strict and args.replay_weights is None:
        raise ValueError("--replay-weights-strict requires --replay-weights")
    checkpoint_scopes = (
        tuple(scope for scope in args.activation_checkpointing_scope.split(",") if scope)
        if args.activation_checkpointing
        else ()
    )
    if args.activation_checkpointing and not checkpoint_scopes:
        raise ValueError("--activation-checkpointing needs at least one --activation-checkpointing-scope")
    unknown_scopes = set(checkpoint_scopes).difference(ACTIVATION_CHECKPOINT_SCOPES)
    if unknown_scopes:
        raise ValueError(f"unknown activation checkpoint scopes: {sorted(unknown_scopes)}")
    model_config_overrides = (
        {} if args.model_config_json is None else load_model_config_overrides(args.model_config_json)
    )
    model_config = model_config_from_overrides(model_config_overrides)
    replay_weights_path = None if args.replay_weights is None else args.replay_weights.resolve()
    replay_weights = (
        None
        if replay_weights_path is None
        else load_replay_weights(replay_weights_path, strict=args.replay_weights_strict)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("stateful cache training requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(f"expected {args.expected_world_size} DDP ranks, received {world_size}")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    index_paths = tuple(path.resolve() for path in args.index)
    train_tags_path = args.train_tags.resolve()
    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    preflight_started = time.perf_counter()
    plans, dataset_metadata = _broadcast_training_plan(rank, index_paths, train_tags_path, args.manifest_workers)
    preflight_s = time.perf_counter() - preflight_started
    if not plans:
        raise ValueError("training split selected no cache shards")
    require_cache_gamma(dataset_metadata, args.gamma)
    training_options: dict[str, object] = {
        "model_config_overrides": model_config_overrides,
        "chunk_normalization": args.chunk_normalization,
        "replay_weights_path": None if replay_weights_path is None else str(replay_weights_path),
        "replay_weights_sha256": None if replay_weights_path is None else _sha256(replay_weights_path),
    }
    if replay_weights is not None:
        training_options.update(replay_weight_summary(replay_weights, plans))

    # A resumed checkpoint must match this resolved config (the loader compares contracts).
    policy = build_production_model_v4(device=device, config=model_config)
    # Runtime-only (not in the checkpoint contract); evaluation leaves it off.
    policy.set_activation_checkpointing(checkpoint_scopes)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True
    )
    start_epoch = 0
    start_group = 0
    update_step = 0
    if args.resume is not None:
        payload = load_actor_critic_checkpoint(args.resume.resolve(), policy, optimizer=optimizer, map_location=device)
        update_step = int(payload["update_step"])
        extra = payload.get("extra")
        if not isinstance(extra, Mapping):
            raise ValueError("resume checkpoint has no cache training cursor")
        start_epoch = int(extra.get("next_epoch", 0))
        start_group = int(extra.get("next_group", 0))

    training_module = _DistributedImitationModule(policy)
    ddp = DistributedDataParallel(
        training_module, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
    )
    if not math.isfinite(args.delay_weight) or args.delay_weight < 0.0:
        raise ValueError("--delay-weight must be finite and non-negative")
    weights = ILLossWeightsV4(delay=args.delay_weight, continue_second_weight=args.continue_second_weight)
    autocast_dtype = _autocast_dtype(args.autocast)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    total_owner_frames = int(dataset_metadata["selected_owner_frames"])
    log_path = output_dir / "train-metrics.jsonl"
    run_started = time.perf_counter()
    processed_frames = 0
    dropped_legacy_expert_frames = 0

    group_width = world_size * args.shards_per_rank
    # (epoch, group index, groups in epoch, global group) in training order.
    schedule: list[tuple[int, int, int, tuple[ShardTrainingPlanV1, ...]]] = []
    for epoch in range(start_epoch, args.epochs):
        groups = _epoch_groups(plans, seed=args.seed, epoch=epoch, group_size=group_width)
        if args.max_groups is not None:
            groups = groups[: args.max_groups]
        epoch_first_group = start_group if epoch == start_epoch else 0
        schedule.extend((epoch, index, len(groups), groups[index]) for index in range(epoch_first_group, len(groups)))
    local_schedule = [
        _local_group_plans(global_group, plans, rank=rank, shards_per_rank=args.shards_per_rank)
        for _epoch, _index, _count, global_group in schedule
    ]
    shard_loader = _GroupShardLoader(
        [[plan.summary_path for plan in local_plans] for local_plans, _dummy in local_schedule],
        workers=args.load_workers,
        preload_next=args.preload_next_group,
        mode=args.preload_workers_mode,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
    )

    if rank == 0:
        startup = {
            "event": "startup",
            "world_size": world_size,
            "parameter_count": parameter_count,
            "time_steps": args.time_steps,
            "shards_per_rank": args.shards_per_rank,
            "autocast": args.autocast,
            "validate": args.validate,
            "preencode_observations": args.preencode_observations,
            "fused_sequence": args.fused_sequence,
            "micro_batches": args.micro_batches,
            "activation_checkpointing": list(policy.activation_checkpointing),
            "prefetch_chunks": args.prefetch_chunks,
            "trim_padding": args.trim_padding,
            "compact_tail_frames": args.compact_tail_frames,
            "preload_next_group": args.preload_next_group,
            **shard_loader.describe(),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gate_act_weight": args.gate_act_weight,
            "delay_neighbor_weight": args.delay_neighbor_weight,
            "delay_weight": args.delay_weight,
            "continue_second_weight": args.continue_second_weight,
            "max_groups": args.max_groups,
            "gamma_per_decision": args.gamma,
            "model_config": asdict(model_config),
            "resume": None if args.resume is None else str(args.resume.resolve()),
            "preflight_s": round(preflight_s, 3),
            **training_options,
            **dataset_metadata,
        }
        print(json.dumps(startup, sort_keys=True), flush=True)
        _write_json_line(log_path, startup)

    def save_checkpoint(next_epoch: int, next_group: int) -> None:
        nonlocal update_step
        dist.barrier()
        if rank == 0:
            path = output_dir / f"checkpoint-step-{update_step:08d}.pt"
            checkpoint_id = save_actor_critic_checkpoint(
                path,
                policy,
                optimizer=optimizer,
                update_step=update_step,
                training_stage="imitation",
                gamma_per_decision=args.gamma,
                extra={
                    **dataset_metadata,
                    "next_epoch": next_epoch,
                    "next_group": next_group,
                    "epochs": args.epochs,
                    "time_steps": args.time_steps,
                    "shards_per_rank": args.shards_per_rank,
                    "world_size": world_size,
                    "seed": args.seed,
                    "gate_act_weight": args.gate_act_weight,
                    "delay_neighbor_weight": args.delay_neighbor_weight,
                    "continue_second_weight": args.continue_second_weight,
                    **training_options,
                },
            )
            row = {
                "event": "checkpoint",
                "path": str(path),
                "checkpoint_id": checkpoint_id,
                "update_step": update_step,
                "next_epoch": next_epoch,
                "next_group": next_group,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            _write_json_line(log_path, row)
        dist.barrier()

    @dataclass
    class _OpenGroup:
        shards: tuple[ILCacheShardV1, ...]
        chunks: Iterator[tuple[ILSequenceV4, int]]
        chunk_count: int
        local_dropped: int
        cache_load_s: float
        cache_preloaded: bool
        preload_row: dict[str, object]

    def open_group(position: int) -> _OpenGroup:
        """Fetch a group's shards and start preparing its chunks (no collectives)."""

        local_plans, dummy = local_schedule[position]
        global_group = schedule[position][3]
        cache_started = time.perf_counter()
        shards, cache_preloaded = shard_loader.get(position)
        cache_load_s = time.perf_counter() - cache_started
        local_dropped = sum(
            int(
                shard.legacy_dropped_expert_mask[
                    shard.descriptors[sequence_index].offset : shard.descriptors[sequence_index].offset
                    + shard.descriptors[sequence_index].length
                ].sum()
            )
            for shard, plan in zip(shards, local_plans, strict=True)
            for sequence_index in plan.sequence_indices
        )
        max_length = max(length for plan in global_group for length in plan.sequence_lengths)
        chunk_count = math.ceil(max_length / args.time_steps)
        chunks = _group_chunks(
            shards,
            local_plans,
            chunk_count=chunk_count,
            time_steps=args.time_steps,
            device=device,
            dummy=dummy,
            replay_weights=replay_weights,
            prefetch_chunks=args.prefetch_chunks,
            trim_padding=args.trim_padding,
            plan=True,
        )
        return _OpenGroup(
            shards=shards,
            chunks=chunks,
            chunk_count=chunk_count,
            local_dropped=local_dropped,
            cache_load_s=cache_load_s,
            cache_preloaded=cache_preloaded,
            preload_row=dict(shard_loader.last_preload),
        )

    def close_group(group: _OpenGroup) -> None:
        close = getattr(group.chunks, "close", None)
        if close is not None:
            close()
        shards = group.shards
        # Drop every reference before handing the shards off to be freed
        # on the preload thread rather than this one.
        group.chunks = iter(())
        group.shards = ()
        shard_loader.release(shards)

    next_open: _OpenGroup | None = None
    try:
        for position, (epoch, group_index, group_count, global_group) in enumerate(schedule):
            local_plans, dummy = local_schedule[position]
            current = next_open if next_open is not None else open_group(position)
            next_open = None
            chunks = current.chunks
            chunk_count = current.chunk_count
            dropped_tensor = torch.tensor(current.local_dropped, device=device, dtype=torch.long)
            dist.all_reduce(dropped_tensor, op=dist.ReduceOp.SUM)
            group_dropped = int(dropped_tensor.item())
            dropped_legacy_expert_frames += group_dropped
            batch_size = sum(len(plan.sequence_indices) for plan in local_plans)
            state = policy.initial_state(batch_size, device=device)
            group_owner_frames = sum(plan.owner_frames for plan in global_group)
            torch.cuda.reset_peak_memory_stats(device)
            group_started = time.perf_counter()
            group_frames = 0
            local_group_frames = 0
            data_wait_s = 0.0
            next_open_s = 0.0
            last_metrics: dict[str, float] = {}
            for chunk_index in range(chunk_count):
                wait_started = time.perf_counter()
                sequence, local_frames = next(chunks)
                data_wait_s += time.perf_counter() - wait_started
                if chunk_index == chunk_count - 1:
                    # Every host chunk of this group is prepared: free its
                    # shards off-thread and start the next group's chunks so
                    # their gather overlaps this last chunk's compute.
                    del chunks
                    close_group(current)
                    if position + 1 < len(schedule):
                        open_started = time.perf_counter()
                        next_open = open_group(position + 1)
                        next_open_s = time.perf_counter() - open_started
                # The all-rank sum of the chunk's active frames, from the group
                # plan every rank holds (dummy ranks add none); cross-checked
                # against an all-reduce once per group.
                global_frames = global_chunk_frames(
                    global_group, start=chunk_index * args.time_steps, time_steps=args.time_steps
                )
                local_group_frames += local_frames
                loss_scale = chunk_loss_scale(
                    args.chunk_normalization,
                    chunk_frames=global_frames,
                    group_owner_frames=group_owner_frames,
                    chunk_count=chunk_count,
                )
                optimizer.zero_grad(set_to_none=True)
                state, pending = chunk_gradients(
                    ddp,
                    sequence,
                    state,
                    weights=weights,
                    world_size=world_size,
                    loss_scale=loss_scale,
                    micro_batches=args.micro_batches,
                    autocast_dtype=autocast_dtype,
                    validate=args.validate,
                    preencode_observations=args.preencode_observations,
                    fused_sequence=args.fused_sequence,
                    gate_act_weight=args.gate_act_weight,
                    delay_neighbor_weight=args.delay_neighbor_weight,
                    context=f"epoch={epoch} group={group_index} chunk={chunk_index}",
                    compact_tail=args.compact_tail_frames,
                    defer_host_sync=True,
                )
                grad_norm = nn.utils.clip_grad_norm_(ddp.parameters(), args.max_grad_norm)
                # The step's one host round trip: loss finiteness, metrics and
                # the gradient norm, all checked before the optimizer step.
                last_metrics, (grad_norm_value,) = cast(PendingChunkMetricsV1, pending).resolve(grad_norm)
                if not math.isfinite(grad_norm_value):
                    raise FloatingPointError(
                        f"non-finite gradient at epoch={epoch} group={group_index} chunk={chunk_index}"
                    )
                optimizer.step()
                update_step += 1
                group_frames += global_frames
                processed_frames += global_frames
                if update_step % args.log_every == 0:
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - run_started
                    rate = processed_frames / max(elapsed, 1e-9)
                    progress = processed_frames / max(total_owner_frames * args.epochs, 1)
                    row = {
                        "event": "train",
                        "epoch": epoch,
                        "group": group_index,
                        "chunk": chunk_index,
                        "update_step": update_step,
                        "processed_owner_frames": processed_frames,
                        "progress": round(progress, 8),
                        "owner_frames_s": round(rate, 3),
                        "eta_s": round(
                            max(0.0, total_owner_frames * args.epochs - processed_frames) / max(rate, 1e-9), 3
                        ),
                        "grad_norm": grad_norm_value,
                        **last_metrics,
                    }
                    if rank == 0:
                        print(json.dumps(row, sort_keys=True), flush=True)
                        _write_json_line(log_path, row)
            del sequence

            torch.cuda.synchronize(device)
            group_elapsed_s = time.perf_counter() - group_started
            reduced_frames = torch.tensor(local_group_frames, device=device, dtype=torch.long)
            dist.all_reduce(reduced_frames, op=dist.ReduceOp.SUM)
            if int(reduced_frames.item()) != group_frames:
                raise RuntimeError(
                    f"group {group_index}: ranks trained {int(reduced_frames.item())} frames, plan says {group_frames}"
                )
            peak = torch.tensor(
                [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                row = {
                    "event": "group_complete",
                    "epoch": epoch,
                    "group": group_index,
                    "groups": group_count,
                    "update_step": update_step,
                    "owner_frames": group_frames,
                    "owner_frames_s": round(group_frames / max(group_elapsed_s, 1e-9), 3),
                    "cache_load_s_rank0": round(current.cache_load_s, 3),
                    "cache_preloaded_rank0": current.cache_preloaded,
                    "chunk_data_wait_s_rank0": round(data_wait_s, 3),
                    "next_group_open_s_rank0": round(next_open_s, 3),
                    **current.preload_row,
                    "group_dropped_legacy_expert_frames": group_dropped,
                    "dropped_legacy_expert_frames": (dropped_legacy_expert_frames),
                    "cuda_peak_allocated_bytes": int(peak[0]),
                    "cuda_peak_reserved_bytes": int(peak[1]),
                    **last_metrics,
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                _write_json_line(log_path, row)
            del current, state
            if (group_index + 1) % args.save_every_groups == 0 or group_index + 1 == group_count:
                next_epoch = epoch
                next_group = group_index + 1
                if next_group == group_count:
                    next_epoch += 1
                    next_group = 0
                save_checkpoint(next_epoch, next_group)
    finally:
        if next_open is not None:
            close = getattr(next_open.chunks, "close", None)
            if close is not None:
                close()
        shard_loader.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
