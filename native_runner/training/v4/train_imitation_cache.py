"""Train UniversalCardPolicyV4 from sharded replay caches with stateful DDP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping, Sequence, cast

import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .cache import (
    ILCacheChunkRefV1,
    ILCacheSequenceDescriptorV1,
    ILCacheShardV1,
    LoadedILBatchV1,
    collate_loaded_il_batches,
    load_il_cache_index,
    load_il_cache_manifest,
    load_il_cache_shards,
)
from .checkpoint import load_actor_critic_checkpoint, save_actor_critic_checkpoint
from .factory import build_production_model_v4
from .imitation import (
    DEFAULT_IL_DELAY_NEIGHBOR_WEIGHT,
    DEFAULT_IL_GATE_ACT_WEIGHT,
    ILLossV4,
    ILLossWeightsV4,
    ILSequenceV4,
    imitation_loss,
)
from .learning import RecurrentEvaluationV4, evaluate_recurrent_sequence
from .model import UniversalCardPolicyV4
from .tensors import RecurrentPolicyStateV4


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


class _DistributedImitationModule(nn.Module):
    """Put the complete recurrent forward under DDP reducer bookkeeping."""

    def __init__(self, policy: UniversalCardPolicyV4) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self, sequence: ILSequenceV4, state: RecurrentPolicyStateV4, *, validate: bool, preencode_observations: bool
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
        )


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


def _summary_paths(index_paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for index_path in index_paths:
        index = load_il_cache_index(index_path)
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
    paths = _summary_paths(index_paths)
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


def _masked_chunk(
    shards: Sequence[ILCacheShardV1],
    plans: Sequence[ShardTrainingPlanV1],
    *,
    start: int,
    time_steps: int,
    device: torch.device,
    dummy: bool,
) -> tuple[ILSequenceV4, int]:
    batches: list[LoadedILBatchV1] = []
    active_steps: list[int] = []
    for shard, plan in zip(shards, plans, strict=True):
        references: list[ILCacheChunkRefV1] = []
        for sequence_index, length in zip(plan.sequence_indices, plan.sequence_lengths, strict=True):
            steps = max(0, min(time_steps, length - start))
            reference_start = min(start, length - 1)
            references.append(
                ILCacheChunkRefV1(sequence_index=sequence_index, start=reference_start, steps=max(1, steps))
            )
            active_steps.append(0 if dummy else steps)
        batches.append(shard.load_batch(references, time_steps=time_steps))
    loaded = collate_loaded_il_batches(batches, device=device)
    sequence = loaded.sequence
    active_lengths = torch.tensor(active_steps, device=device)[None, :]
    time = torch.arange(time_steps, device=device)[:, None]
    active = time < active_lengths
    sequence = replace(
        sequence,
        episode_start=sequence.episode_start & active,
        valid_mask=sequence.valid_mask & active,
        returns=torch.where(active, sequence.returns, torch.zeros_like(sequence.returns)),
        gate_loss_mask=(None if sequence.gate_loss_mask is None else sequence.gate_loss_mask & active),
        value_loss_mask=(None if sequence.value_loss_mask is None else sequence.value_loss_mask & active),
    )
    return sequence, sum(active_steps)


_LOSS_HEADS = (
    ("gate", "gate_count"),
    ("candidate", "candidate_count"),
    ("target", "target_count"),
    ("delay", "delay_count"),
    ("continue_action", "continue_count"),
    ("value", "value_count"),
)


def _distributed_loss(
    losses: ILLossV4, weights: ILLossWeightsV4, *, world_size: int
) -> tuple[Tensor, dict[str, float]]:
    total = losses.total * 0.0
    metrics: dict[str, float] = {}
    local_counts = torch.stack(
        [getattr(losses, count_name).detach().to(torch.float32) for _loss_name, count_name in _LOSS_HEADS]
    )
    local_sums = torch.stack(
        [
            getattr(losses, loss_name).detach().to(torch.float32) * local_counts[index]
            for index, (loss_name, _count_name) in enumerate(_LOSS_HEADS)
        ]
    )
    global_stats = torch.cat((local_counts, local_sums))
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
    global_counts, global_sums = global_stats.chunk(2)
    for index, (loss_name, count_name) in enumerate(_LOSS_HEADS):
        local_loss = getattr(losses, loss_name)
        local_count = local_counts[index]
        global_count = global_counts[index]
        count_value = float(global_count.item())
        metrics[loss_name] = float((global_sums[index] / global_count).item()) if count_value else 0.0
        metrics[count_name] = count_value
        if count_value:
            scale = local_count * float(world_size) / global_count
            total = total + getattr(weights, loss_name) * local_loss * scale
    return total, metrics


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


def _write_json_line(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True) + "\n")


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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-groups", type=int, default=50)
    parser.add_argument("--max-groups", type=int, help="limit each epoch for an explicit end-to-end smoke run")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--autocast", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preencode-observations", action=argparse.BooleanOptionalAction, default=True)
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
    )
    if any(value <= 0 for value in positive) or args.weight_decay < 0:
        raise ValueError("training counts and rates must be positive")
    if not 0.0 <= args.delay_neighbor_weight <= 1.0 / 3.0:
        raise ValueError("delay neighbor weight must be in [0, 1/3]")
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("max_groups must be positive")
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

    policy = build_production_model_v4(device=device)
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
    weights = ILLossWeightsV4()
    autocast_dtype = _autocast_dtype(args.autocast)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    total_owner_frames = int(dataset_metadata["selected_owner_frames"])
    log_path = output_dir / "train-metrics.jsonl"
    run_started = time.perf_counter()
    processed_frames = 0
    dropped_legacy_expert_frames = 0

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
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gate_act_weight": args.gate_act_weight,
            "delay_neighbor_weight": args.delay_neighbor_weight,
            "max_groups": args.max_groups,
            "preflight_s": round(preflight_s, 3),
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

    group_width = world_size * args.shards_per_rank
    try:
        for epoch in range(start_epoch, args.epochs):
            groups = _epoch_groups(plans, seed=args.seed, epoch=epoch, group_size=group_width)
            if args.max_groups is not None:
                groups = groups[: args.max_groups]
            epoch_first_group = start_group if epoch == start_epoch else 0
            for group_index in range(epoch_first_group, len(groups)):
                global_group = groups[group_index]
                local_start = rank * args.shards_per_rank
                local_plans = global_group[local_start : local_start + args.shards_per_rank]
                dummy = not local_plans
                if dummy:
                    source = plans[0]
                    unit = source.units[0]
                    local_plans = (ShardTrainingPlanV1(summary_path=source.summary_path, units=(unit,)),)
                cache_started = time.perf_counter()
                shards = load_il_cache_shards([plan.summary_path for plan in local_plans], workers=args.load_workers)
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
                dropped_tensor = torch.tensor(local_dropped, device=device, dtype=torch.long)
                dist.all_reduce(dropped_tensor, op=dist.ReduceOp.SUM)
                group_dropped = int(dropped_tensor.item())
                dropped_legacy_expert_frames += group_dropped
                batch_size = sum(len(plan.sequence_indices) for plan in local_plans)
                state = policy.initial_state(batch_size, device=device)
                max_length = max(length for plan in global_group for length in plan.sequence_lengths)
                chunk_count = math.ceil(max_length / args.time_steps)
                torch.cuda.reset_peak_memory_stats(device)
                group_started = time.perf_counter()
                group_frames = 0
                last_metrics: dict[str, float] = {}
                for chunk_index in range(chunk_count):
                    sequence, local_frames = _masked_chunk(
                        shards,
                        local_plans,
                        start=chunk_index * args.time_steps,
                        time_steps=args.time_steps,
                        device=device,
                        dummy=dummy,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                        evaluation = ddp(
                            sequence, state, validate=args.validate, preencode_observations=args.preencode_observations
                        )
                        losses = imitation_loss(
                            sequence,
                            evaluation,
                            weights=weights,
                            gate_act_weight=args.gate_act_weight,
                            delay_neighbor_weight=args.delay_neighbor_weight,
                        )
                        loss, last_metrics = _distributed_loss(losses, weights, world_size=world_size)
                    finite = torch.tensor(int(bool(torch.isfinite(loss).all())), device=device, dtype=torch.int32)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not int(finite.item()):
                        raise FloatingPointError(
                            f"non-finite distributed IL loss at epoch={epoch} group={group_index} chunk={chunk_index}"
                        )
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(ddp.parameters(), args.max_grad_norm)
                    if not bool(torch.isfinite(grad_norm)):
                        raise FloatingPointError(
                            f"non-finite gradient at epoch={epoch} group={group_index} chunk={chunk_index}"
                        )
                    optimizer.step()
                    state = evaluation.final_state.detach()
                    update_step += 1
                    frame_tensor = torch.tensor(local_frames, device=device, dtype=torch.long)
                    dist.all_reduce(frame_tensor, op=dist.ReduceOp.SUM)
                    global_frames = int(frame_tensor.item())
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
                            "grad_norm": float(grad_norm.detach().cpu()),
                            **last_metrics,
                        }
                        if rank == 0:
                            print(json.dumps(row, sort_keys=True), flush=True)
                            _write_json_line(log_path, row)

                torch.cuda.synchronize(device)
                group_elapsed_s = time.perf_counter() - group_started
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
                        "groups": len(groups),
                        "update_step": update_step,
                        "owner_frames": group_frames,
                        "owner_frames_s": round(group_frames / max(group_elapsed_s, 1e-9), 3),
                        "cache_load_s_rank0": round(cache_load_s, 3),
                        "group_dropped_legacy_expert_frames": group_dropped,
                        "dropped_legacy_expert_frames": (dropped_legacy_expert_frames),
                        "cuda_peak_allocated_bytes": int(peak[0]),
                        "cuda_peak_reserved_bytes": int(peak[1]),
                        **last_metrics,
                    }
                    print(json.dumps(row, sort_keys=True), flush=True)
                    _write_json_line(log_path, row)
                del shards, state
                if (group_index + 1) % args.save_every_groups == 0 or group_index + 1 == len(groups):
                    next_epoch = epoch
                    next_group = group_index + 1
                    if next_group == len(groups):
                        next_epoch += 1
                        next_group = 0
                    save_checkpoint(next_epoch, next_group)
            start_group = 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
