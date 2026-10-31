"""Lossless sharded cache for model-ready V4 imitation sequences."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import ctypes
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from functools import reduce
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, TypeVar, get_type_hints

import torch
from torch import Tensor

from .config import PENDING_TICKS_REMAINING_FEATURE_INDEX, ModelConfigV4
from .imitation import IL_ANNOTATION_FIELDS, IL_TIME_FIELDS, ILSequenceV4, collate_il_sequences
from .tensors import ActionSequenceV4, TARGET_GRID, TensorRecordV4, UniversalSemanticBatchV4, concatenate_tensor_records

if TYPE_CHECKING:
    from .producer import ReplayProductionResultV4


IL_CACHE_SHARD_SCHEMA = "v4-il-model-input-cache-shard.v1"
IL_CACHE_SUMMARY_SCHEMA = "v4-il-model-input-cache-summary.v1"
IL_CACHE_INDEX_SCHEMA = "v4-il-model-input-cache-index.v1"
_CONTAINER_MAGIC = b"CRILC001"
_CONTAINER_HEADER = struct.Struct("<8sQ")
_LEGACY_CONSERVATIVE_POCKET_BATTLE_ENV_SHA256 = "abd2f41522420f27e3810fcaba28cc666e3d52ebc8b4219d924d8d9d6b660de5"


def _uses_legacy_conservative_pocket(manifest: Mapping[str, object]) -> bool:
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        return False
    source_sha256 = contract.get("source_sha256")
    return (
        isinstance(source_sha256, Mapping)
        and source_sha256.get("battle_env.py") == _LEGACY_CONSERVATIVE_POCKET_BATTLE_ENV_SHA256
    )


def _upgrade_legacy_pocket_placement(observations: UniversalSemanticBatchV4) -> None:
    """Upgrade cached 0..6/11..17 pocket masks to the proven half-arena.

    Cache tensors predate the replay-backed pocket-boundary correction.  The
    model coordinate system always places the enemy pocket in rows 17..22.
    Existing legal cells identify which tower lane was opened; only the two
    missing center-seam columns are added.  New caches carry a different source
    hash and never enter this migration.
    """

    candidates = observations.candidates
    placement = candidates.placement
    if placement.ndim != 4 or tuple(placement.shape[-2:]) != (32, 18):
        raise ValueError("legacy pocket migration requires placement [B,C,32,18]")
    upgraded = placement.clone()
    pocket = placement[:, :, 17:23]
    left_open = pocket[:, :, :, :7].any(dim=-1)
    right_open = pocket[:, :, :, 11:].any(dim=-1)
    upgraded[:, :, 17:23, 7] |= left_open
    upgraded[:, :, 17:23, 8] |= left_open
    upgraded[:, :, 17:23, 9] |= right_open
    upgraded[:, :, 17:23, 10] |= right_open
    upgraded &= candidates.mask[:, :, None, None]
    candidates.placement = upgraded
    runtime = candidates.runtime_features.clone()
    runtime[:, :, 1] = upgraded.to(runtime.dtype).mean(dim=(-2, -1))
    candidates.runtime_features = runtime


def _upgrade_legacy_bridge_placement(observations: UniversalSemanticBatchV4) -> None:
    """Expose the native-proven own-side bridge cells to troop candidates."""

    candidates = observations.candidates
    placement = candidates.placement.clone()
    troop = candidates.mask & ~candidates.is_building & (candidates.target_mode == TARGET_GRID)
    for y in (15, 16):
        for x in (2, 3, 13, 14):
            placement[:, :, y, x] |= troop
    candidates.placement = placement
    runtime = candidates.runtime_features.clone()
    runtime[:, :, 1] = placement.to(runtime.dtype).mean(dim=(-2, -1))
    candidates.runtime_features = runtime


def _neutralize_pending_ticks_remaining(observations: UniversalSemanticBatchV4) -> bool:
    """Zero the own-card ``pending_ticks_remaining_over_20`` column in place.

    Caches built before the feature was neutralised carry the decaying
    in-flight countdown that live play can never observe.  Returns whether any
    value changed; an already-zero column is left untouched (no copy).
    """

    own_cards = observations.own_cards
    runtime = own_cards.runtime_features
    if runtime.ndim != 3 or int(runtime.shape[-1]) <= PENDING_TICKS_REMAINING_FEATURE_INDEX:
        raise ValueError("pending-ticks neutralisation requires own-card runtime features [B,N,F]")
    if not bool(torch.any(runtime[..., PENDING_TICKS_REMAINING_FEATURE_INDEX] != 0)):
        return False
    runtime = runtime.clone()
    runtime[..., PENDING_TICKS_REMAINING_FEATURE_INDEX] = 0
    own_cards.runtime_features = runtime
    return True


def _sanitize_legacy_illegal_expert_targets(
    observations: UniversalSemanticBatchV4, actions: ActionSequenceV4, gate_loss_mask: Tensor
) -> Tensor:
    """Drop old action labels that the stored decision mask cannot represent.

    The current producer already converts an unrepresentable expert window to
    WAIT and clears its gate-loss bit.  Legacy shards predate that boundary.
    Apply the same rule in memory instead of adding the target cell to the
    observation mask, which would leak the supervised answer into the input.
    """

    candidates = observations.candidates
    placement = candidates.placement
    frame_count = int(actions.gate.shape[0])
    frames = torch.arange(frame_count, device=placement.device)
    invalid_frames = torch.zeros(frame_count, dtype=torch.bool, device=placement.device)
    for micro in range(int(actions.candidate_index.shape[1])):
        active = actions.micro_action_count > micro
        target = actions.target_cell[:, micro]
        grid = active & (target >= 0)
        candidate = actions.candidate_index[:, micro].clamp_min(0)
        safe_target = target.clamp_min(0)
        x = safe_target.remainder(18)
        y = torch.div(safe_target, 18, rounding_mode="floor")
        legal = placement[frames, candidate, y, x]
        invalid_frames |= grid & ~legal
    if not torch.any(invalid_frames):
        return invalid_frames
    actions.gate = actions.gate.clone()
    actions.micro_action_count = actions.micro_action_count.clone()
    actions.gate[invalid_frames] = 0
    actions.micro_action_count[invalid_frames] = 0
    for name in ("candidate_index", "candidate_uid", "target_cell", "delay_offset_bin"):
        value = getattr(actions, name).clone()
        value[invalid_frames] = -1
        setattr(actions, name, value)
    gate_loss_mask[invalid_frames] = False
    return invalid_frames


@dataclass(frozen=True, slots=True)
class ILCacheSequenceDescriptorV1:
    """Address one owner sequence in a shard's concatenated frame tensors."""

    sequence_id: str
    replay_tag: str
    owner: int
    offset: int
    length: int
    natural_time: bool
    gate_loss_mask_present: bool
    value_loss_mask_present: bool


@dataclass(frozen=True, slots=True)
class ILCacheChunkRefV1:
    """Address a natural-time chunk; a short final chunk is padded on load."""

    sequence_index: int
    start: int
    steps: int


@dataclass(frozen=True, slots=True)
class LoadedILBatchV1:
    """A training-ready recurrent batch and its non-padding owner-frame count."""

    sequence: ILSequenceV4
    owner_frames: int


@dataclass(slots=True)
class ILChunkSequenceV4(ILSequenceV4):
    """An IL chunk plus its host-side plan (``evaluate_recurrent_sequence`` tail compression).

    ``lane_steps[b]``: lane ``b``'s frames at ``t >= lane_steps[b]`` repeat
    frame ``lane_steps[b] - 1`` exactly (the chunk ran past the lane's end).
    ``mid_chunk_reset``: ``episode_start[1:].any()`` computed on the host
    (``None`` = unknown, check on the device).
    """

    lane_steps: tuple[int, ...] | None = None
    mid_chunk_reset: bool | None = None


_RecordT = TypeVar("_RecordT", bound=TensorRecordV4)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _flatten_record(value: TensorRecordV4, *, prefix: str, destination: dict[str, Tensor]) -> None:
    for key, tensor in _record_tensor_items(value, prefix=prefix):
        destination[key] = tensor.detach().cpu().contiguous()


def _restore_record(record_type: type[_RecordT], *, prefix: str, tensors: Mapping[str, Tensor]) -> _RecordT:
    hints = get_type_hints(record_type)
    values: dict[str, object] = {}
    for item in fields(record_type):
        key = f"{prefix}.{item.name}"
        annotation = hints[item.name]
        if isinstance(annotation, type) and issubclass(annotation, TensorRecordV4):
            values[item.name] = _restore_record(annotation, prefix=key, tensors=tensors)
        else:
            try:
                values[item.name] = tensors[key]
            except KeyError as error:
                raise ValueError(f"cache shard is missing tensor {key!r}") from error
    return record_type(**values)


def _pad_dim_one(value: Tensor, target: int, padding_value: object, *, pin_memory: bool = False) -> Tensor:
    """Pad a batch-size-one tensor's variable row dimension canonically."""

    if value.ndim < 2:
        raise ValueError("variable V4 cache tensors must have at least two dimensions")
    current = int(value.shape[1])
    if current > target:
        raise ValueError("V4 cache padding target is smaller than the source")
    if current == target:
        return value
    shape = list(value.shape)
    shape[1] = target
    padded = torch.full(shape, padding_value, dtype=value.dtype, device=value.device, pin_memory=pin_memory)
    padded[:, :current] = value
    return padded


_NEGATIVE_PADDING_FIELDS = frozenset(
    {
        "parent_index",
        "own_card_row",
        "source_group_index",
        "source_child_index",
        "exclusion_group_id",
        "native_hand_slot",
        "native_source_entity",
        "native_visible_card_id",
    }
)
_DYNAMIC_COUNTS = {
    "active_effects": "active_effect_count",
    "relation_edges": "relation_edge_count",
    "candidates": "candidate_count",
}


def _pad_collection(value: _RecordT, target: int) -> _RecordT:
    return type(value)(
        **{
            item.name: (
                tensor
                if item.name == "elixir"
                else _pad_dim_one(tensor, target, -1 if item.name in _NEGATIVE_PADDING_FIELDS else 0)
            )
            for item in fields(value)
            for tensor in (getattr(value, item.name),)
        }
    )


def _pad_dynamic_observation(
    value: UniversalSemanticBatchV4, *, active_effect_count: int, relation_edge_count: int, candidate_count: int
) -> UniversalSemanticBatchV4:
    counts = (active_effect_count, relation_edge_count, candidate_count)
    return replace(
        value,
        **{
            name: _pad_collection(getattr(value, name), count)
            for name, count in zip(_DYNAMIC_COUNTS, counts, strict=True)
        },
    )


def _pad_dim_one_concatenated(values: tuple[Tensor, ...], target: int, padding_value: object) -> Tensor:
    """``torch.cat`` of ``_pad_dim_one`` over ``values``, padding once per shard.

    Frames of equal row width are concatenated together and written into one
    canonically padded tensor instead of materializing a padded copy per frame.
    Anything ``torch.cat`` would promote or reject takes the per-frame path.
    """

    first = values[0]
    if (
        not isinstance(first, Tensor)
        or first.ndim < 2
        or not all(
            isinstance(value, Tensor)
            and value.dtype == first.dtype
            and value.device == first.device
            and value.ndim == first.ndim
            and value.shape[2:] == first.shape[2:]
            and int(value.shape[1]) <= target
            for value in values
        )
    ):
        return torch.cat([_pad_dim_one(value, target, padding_value) for value in values], dim=0)
    rows_by_width: dict[int, tuple[list[Tensor], list[int]]] = {}
    row = 0
    for value in values:
        count = int(value.shape[0])
        width = int(value.shape[1])
        if width:
            tensors, rows = rows_by_width.setdefault(width, ([], []))
            tensors.append(value)
            rows.extend(range(row, row + count))
        row += count
    output = torch.full((row, target, *first.shape[2:]), padding_value, dtype=first.dtype, device=first.device)
    for width, (tensors, rows) in rows_by_width.items():
        output[torch.tensor(rows, dtype=torch.long, device=first.device), :width] = torch.cat(tensors, dim=0)
    return output


def _concatenate_padded_observations(
    observations: tuple[UniversalSemanticBatchV4, ...],
    *,
    active_effect_count: int,
    relation_edge_count: int,
    candidate_count: int,
) -> UniversalSemanticBatchV4:
    """``concatenate_tensor_records`` of each frame's ``_pad_dynamic_observation``."""

    counts = dict(zip(_DYNAMIC_COUNTS, (active_effect_count, relation_edge_count, candidate_count), strict=True))
    first = observations[0]
    if not is_dataclass(first) or not all(type(observation) is type(first) for observation in observations):
        return concatenate_tensor_records(
            tuple(
                _pad_dynamic_observation(
                    observation,
                    active_effect_count=active_effect_count,
                    relation_edge_count=relation_edge_count,
                    candidate_count=candidate_count,
                )
                for observation in observations
            )
        )
    values: dict[str, object] = {}
    for item in fields(first):
        children = tuple(getattr(observation, item.name) for observation in observations)
        target = counts.get(item.name)
        if target is None:
            values[item.name] = concatenate_tensor_records(children)
            continue
        collection = children[0]
        if not all(type(child) is type(collection) for child in children):
            raise TypeError("cannot concatenate different V4 record types")
        values[item.name] = type(collection)(
            **{
                field.name: (
                    concatenate_tensor_records(tensors)
                    if field.name == "elixir"
                    else _pad_dim_one_concatenated(
                        tensors, target, -1 if field.name in _NEGATIVE_PADDING_FIELDS else 0
                    )
                )
                for field in fields(collection)
                for tensors in (tuple(getattr(child, field.name) for child in children),)
            }
        )
    return type(first)(**values)


def _narrow_dim_one(value: _RecordT, length: int) -> _RecordT:
    """Restore a variable collection's original row capacity."""

    if not is_dataclass(value):
        raise TypeError("V4 cache records must be dataclasses")
    values: dict[str, object] = {}
    for item in fields(value):
        child = getattr(value, item.name)
        if not isinstance(child, Tensor):
            raise TypeError(f"unsupported variable cached field {item.name}: {type(child)!r}")
        values[item.name] = child if child.ndim < 2 else child.narrow(1, 0, length)
    return type(value)(**values)


def _restore_dynamic_observation(
    value: UniversalSemanticBatchV4, *, active_effect_count: int, relation_edge_count: int, candidate_count: int
) -> UniversalSemanticBatchV4:
    counts = (active_effect_count, relation_edge_count, candidate_count)
    return replace(
        value,
        **{
            name: _narrow_dim_one(getattr(value, name), count)
            for name, count in zip(_DYNAMIC_COUNTS, counts, strict=True)
        },
    )


def _pad_sequence_dynamic_observations(
    sequence: ILSequenceV4, *, active_effect_count: int, relation_edge_count: int, candidate_count: int
) -> ILSequenceV4:
    return replace(
        sequence,
        observations=tuple(
            _pad_dynamic_observation(
                observation,
                active_effect_count=active_effect_count,
                relation_edge_count=relation_edge_count,
                candidate_count=candidate_count,
            )
            for observation in sequence.observations
        ),
    )


def _record_tensor_items(value: TensorRecordV4, *, prefix: str = "") -> tuple[tuple[str, Tensor], ...]:
    result: list[tuple[str, Tensor]] = []

    def visit(current: object, path: str) -> None:
        if isinstance(current, Tensor):
            result.append((path, current))
            return
        if isinstance(current, TensorRecordV4) and is_dataclass(current):
            for item in fields(current):
                child_path = f"{path}.{item.name}" if path else item.name
                visit(getattr(current, item.name), child_path)
            return
        raise TypeError(f"unsupported cached tensor tree value {type(current)!r}")

    visit(value, prefix)
    return tuple(result)


def _result_metadata(result: ReplayProductionResultV4) -> dict[str, object]:
    return {
        "replay_tag": result.replay_tag,
        "completed": result.completed,
        "failure_reason": result.failure_reason,
        "first_untrusted_tick": result.first_untrusted_tick,
        "simulated_end_tick": result.simulated_end_tick,
        "source_end_tick": result.source_end_tick,
        "expert_action_count": result.expert_action_count,
        "executed_expert_action_count": result.executed_expert_action_count,
        "winner_matches": result.winner_matches,
        "crowns_match": result.crowns_match,
        "retained_frame_count": result.retained_frame_count,
        "decision_frame_count": result.decision_frame_count,
    }


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{os.getpid()}.tmp")


def _atomic_write(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(destination: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(destination, _canonical_json(payload) + b"\n")


def write_il_cache_shard(
    destination: str | Path,
    results: Sequence[ReplayProductionResultV4],
    *,
    shard_id: str,
    contract: Mapping[str, object],
    compression_level: int = 1,
    storage_float_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    """Pack live producer results into one checksummed Zstandard shard."""

    try:
        import zstandard
        from safetensors.torch import save as save_safetensors
    except ImportError as error:
        raise RuntimeError("V4 IL caching requires safetensors and zstandard") from error
    if not -5 <= compression_level <= 22:
        raise ValueError("Zstandard compression level must be in -5..22")

    sequences: list[ILSequenceV4] = []
    descriptors: list[ILCacheSequenceDescriptorV1] = []
    offset = 0
    for result in results:
        if result.sequences is None:
            continue
        for owner, sequence in enumerate(result.sequences):
            sequence.validate()
            if sequence.batch_size != 1:
                raise ValueError("cache writer requires uncollated batch-size-one sequences")
            length = sequence.time_steps
            descriptors.append(
                ILCacheSequenceDescriptorV1(
                    sequence_id=sequence.sequence_id,
                    replay_tag=result.replay_tag,
                    owner=owner,
                    offset=offset,
                    length=length,
                    natural_time=sequence.natural_time,
                    gate_loss_mask_present=sequence.gate_loss_mask is not None,
                    value_loss_mask_present=sequence.value_loss_mask is not None,
                )
            )
            sequences.append(sequence)
            offset += length
    if not sequences:
        raise ValueError("cache shard needs at least one retained IL sequence")

    observations = tuple(observation for sequence in sequences for observation in sequence.observations)
    active_effect_counts = tuple(int(observation.active_effects.mask.shape[1]) for observation in observations)
    relation_edge_counts = tuple(int(observation.relation_edges.mask.shape[1]) for observation in observations)
    candidate_counts = tuple(int(observation.candidates.mask.shape[1]) for observation in observations)
    active_effect_capacity = max(active_effect_counts)
    relation_edge_capacity = max(relation_edge_counts)
    candidate_capacity = max(candidate_counts)
    packed_observations = _concatenate_padded_observations(
        observations,
        active_effect_count=active_effect_capacity,
        relation_edge_count=relation_edge_capacity,
        candidate_count=candidate_capacity,
    )
    packed_actions = concatenate_tensor_records(tuple(action for sequence in sequences for action in sequence.actions))
    if not isinstance(packed_observations, UniversalSemanticBatchV4):
        raise TypeError("cached observations have an unexpected record type")
    if not isinstance(packed_actions, ActionSequenceV4):
        raise TypeError("cached actions have an unexpected record type")
    if storage_float_dtype is not None:
        packed_observations = packed_observations.to_storage("cpu", float_dtype=storage_float_dtype)

    tensors: dict[str, Tensor] = {}
    _flatten_record(packed_observations, prefix="observations", destination=tensors)
    _flatten_record(packed_actions, prefix="actions", destination=tensors)
    for name in IL_TIME_FIELDS:
        tensors[name] = torch.cat(
            [sequence.valid_mask if (value := getattr(sequence, name)) is None else value for sequence in sequences],
            dim=0,
        ).contiguous()
    tensors["sequence_offsets"] = torch.tensor(
        [descriptor.offset for descriptor in descriptors] + [offset], dtype=torch.long
    )
    for name, counts in (
        ("active_effect_counts", active_effect_counts),
        ("relation_edge_counts", relation_edge_counts),
        ("candidate_counts", candidate_counts),
    ):
        tensors[name] = torch.tensor(counts, dtype=torch.int32)
    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        if not bool(finite.all()):
            nonfinite = int((~finite).sum().item())
            raise ValueError(f"cache tensor {name} contains {nonfinite} non-finite values after storage packing")

    manifest = {
        "schema": IL_CACHE_SHARD_SCHEMA,
        "shard_id": str(shard_id),
        "contract": dict(contract),
        "sequences": [asdict(descriptor) for descriptor in descriptors],
        "results": [_result_metadata(result) for result in results],
        "owner_frames": offset,
        "match_decision_frames": offset // 2,
        "dynamic_capacities": {
            "active_effects": active_effect_capacity,
            "relation_edges": relation_edge_capacity,
            "candidates": candidate_capacity,
        },
        "tensor_count": len(tensors),
    }
    manifest_payload = _canonical_json(manifest)
    tensor_payload = save_safetensors(tensors)
    container = _CONTAINER_HEADER.pack(_CONTAINER_MAGIC, len(manifest_payload)) + manifest_payload + tensor_payload
    compressor = zstandard.ZstdCompressor(level=compression_level, threads=1)
    compressed = compressor.compress(container)

    data_path = Path(destination)
    if not data_path.name.endswith(".cril.zst"):
        raise ValueError("cache shard destination must end with .cril.zst")
    summary_path = data_path.with_suffix(".json")
    summary: dict[str, object] = {
        "schema": IL_CACHE_SUMMARY_SCHEMA,
        "shard_schema": IL_CACHE_SHARD_SCHEMA,
        "shard_id": str(shard_id),
        "data_file": data_path.name,
        "compressed_sha256": _sha256(compressed),
        "container_sha256": _sha256(container),
        "compressed_bytes": len(compressed),
        "container_bytes": len(container),
        "compression_ratio": round(len(container) / max(1, len(compressed)), 6),
        "replay_results": len(results),
        "sequences": len(sequences),
        "owner_frames": offset,
        "match_decision_frames": offset // 2,
        "completed_replays": sum(result.completed for result in results),
        "failed_replays": sum(not result.completed for result in results),
        "zero_frame_failures": sum(not result.completed and result.decision_frame_count == 0 for result in results),
        "failures_before_40s": sum(
            not result.completed and result.decision_frame_count > 0 and result.simulated_end_tick < 800
            for result in results
        ),
        "unexpected_unknown_failures": sum(
            not result.completed and "unknown" in str(result.failure_reason or "").lower() for result in results
        ),
        "winner_comparable_replays": sum(result.winner_matches is not None for result in results),
        "winner_mismatches": sum(result.winner_matches is False for result in results),
        "crowns_comparable_replays": sum(result.crowns_match is not None for result in results),
        "crowns_mismatches": sum(result.crowns_match is False for result in results),
        "expert_actions": sum(result.expert_action_count for result in results),
        "executed_expert_actions": sum(result.executed_expert_action_count for result in results),
        "decision_frames": sum(result.decision_frame_count for result in results),
    }
    _atomic_write(data_path, compressed)
    _atomic_write_json(summary_path, summary)
    return summary


class ILCacheShardV1:
    """One decompressed in-memory shard with vectorized recurrent batch loads."""

    def __init__(
        self,
        *,
        path: Path,
        manifest: Mapping[str, object],
        tensors: Mapping[str, Tensor],
        summary: Mapping[str, object],
        config: ModelConfigV4 | None = None,
    ) -> None:
        self.config = ModelConfigV4() if config is None else config
        raw_descriptors = manifest.get("sequences")
        if not isinstance(raw_descriptors, list):
            raise ValueError("cache manifest has no sequence descriptors")
        self.path = path
        self.manifest = dict(manifest)
        self.summary = dict(summary)
        self.descriptors = tuple(
            ILCacheSequenceDescriptorV1(**item) for item in raw_descriptors if isinstance(item, dict)
        )
        if len(self.descriptors) != len(raw_descriptors):
            raise ValueError("cache sequence descriptor is not an object")
        self.observations = _restore_record(UniversalSemanticBatchV4, prefix="observations", tensors=tensors)
        self.legacy_pocket_migration = _uses_legacy_conservative_pocket(manifest)
        self.actions = _restore_record(ActionSequenceV4, prefix="actions", tensors=tensors)
        self.legacy_dropped_expert_mask = torch.zeros(int(self.actions.gate.shape[0]), dtype=torch.bool)
        self.legacy_dropped_expert_frames = 0
        if self.legacy_pocket_migration:
            _upgrade_legacy_pocket_placement(self.observations)
            _upgrade_legacy_bridge_placement(self.observations)
        # Both ``sequence`` and ``load_batch`` read from ``self.observations``,
        # so zeroing once here covers every cached-observation consumer.
        self.pending_ticks_neutralized = bool(self.config.neutralize_pending_ticks_remaining) and (
            _neutralize_pending_ticks_remaining(self.observations)
        )
        for name in (
            *IL_TIME_FIELDS,
            "sequence_offsets",
            "active_effect_counts",
            "relation_edge_counts",
            "candidate_counts",
        ):
            setattr(self, name, tensors[name])
        if self.legacy_pocket_migration:
            self.gate_loss_mask = self.gate_loss_mask.clone()
            self.legacy_dropped_expert_mask = _sanitize_legacy_illegal_expert_targets(
                self.observations, self.actions, self.gate_loss_mask
            ).cpu()
            self.legacy_dropped_expert_frames = int(self.legacy_dropped_expert_mask.sum())
        self._validate_layout()

    @property
    def owner_frames(self) -> int:
        return int(self.episode_start.shape[0])

    def _validate_layout(self) -> None:
        frames = self.owner_frames
        if self.observations.batch_size != frames:
            raise ValueError("cache observation frame count disagrees with masks")
        if int(self.actions.gate.shape[0]) != frames:
            raise ValueError("cache action frame count disagrees with masks")
        for label in IL_TIME_FIELDS:
            value = getattr(self, label)
            if tuple(value.shape) != (frames, 1):
                raise ValueError(f"cache {label} must have shape [F,1]")
        dynamic_capacities = self.manifest.get("dynamic_capacities")
        if not isinstance(dynamic_capacities, dict):
            raise ValueError("cache manifest has no dynamic capacities")
        for value, label, capacity in (
            (self.active_effect_counts, "active_effect_counts", self.observations.active_effects.mask.shape[1]),
            (self.relation_edge_counts, "relation_edge_counts", self.observations.relation_edges.mask.shape[1]),
            (self.candidate_counts, "candidate_counts", self.observations.candidates.mask.shape[1]),
        ):
            if tuple(value.shape) != (frames,):
                raise ValueError(f"cache {label} must have shape [F]")
            if torch.any((value < 0) | (value > int(capacity))):
                raise ValueError(f"cache {label} is outside its padded capacity")
        expected_capacities = {
            "active_effects": int(self.observations.active_effects.mask.shape[1]),
            "relation_edges": int(self.observations.relation_edges.mask.shape[1]),
            "candidates": int(self.observations.candidates.mask.shape[1]),
        }
        if dynamic_capacities != expected_capacities:
            raise ValueError("cache dynamic capacities disagree with its tensors")
        expected_offsets = [descriptor.offset for descriptor in self.descriptors]
        expected_offsets.append(frames)
        if self.sequence_offsets.tolist() != expected_offsets:
            raise ValueError("cache tensor offsets disagree with the manifest")
        for index, descriptor in enumerate(self.descriptors):
            if descriptor.owner not in (0, 1) or descriptor.length <= 0:
                raise ValueError("cache sequence descriptor is invalid")
            if descriptor.offset + descriptor.length != expected_offsets[index + 1]:
                raise ValueError("cache sequence descriptors are not contiguous")

    def sequence(self, index: int) -> ILSequenceV4:
        """Reconstruct one exact owner sequence, primarily for fidelity checks."""

        descriptor = self.descriptors[index]
        observations = tuple(
            _restore_dynamic_observation(
                self.observations.narrow_batch(descriptor.offset + step, 1),
                active_effect_count=int(self.active_effect_counts[descriptor.offset + step]),
                relation_edge_count=int(self.relation_edge_counts[descriptor.offset + step]),
                candidate_count=int(self.candidate_counts[descriptor.offset + step]),
            )
            for step in range(descriptor.length)
        )
        actions = tuple(self.actions.narrow_batch(descriptor.offset + step, 1) for step in range(descriptor.length))
        frame_slice = slice(descriptor.offset, descriptor.offset + descriptor.length)
        return ILSequenceV4(
            observations=observations,
            actions=actions,
            episode_start=self.episode_start[frame_slice],
            valid_mask=self.valid_mask[frame_slice],
            returns=self.returns[frame_slice],
            gate_loss_mask=(self.gate_loss_mask[frame_slice] if descriptor.gate_loss_mask_present else None),
            value_loss_mask=(self.value_loss_mask[frame_slice] if descriptor.value_loss_mask_present else None),
            sequence_id=descriptor.sequence_id,
            natural_time=descriptor.natural_time,
        )

    def _chunk_gather(
        self, references: Sequence[ILCacheChunkRefV1], *, time_steps: int
    ) -> tuple[Tensor, Tensor, list[ILCacheSequenceDescriptorV1], list[int]]:
        """Time-major ``[T*B]`` frame indices (short chunks repeat their last frame) and the ``[T,B]`` mask."""

        if not references or time_steps <= 0:
            raise ValueError("cache batch needs references and positive time steps")
        base_offsets: list[int] = []
        valid_steps: list[int] = []
        selected_descriptors: list[ILCacheSequenceDescriptorV1] = []
        for reference in references:
            descriptor = self.descriptors[reference.sequence_index]
            if (
                reference.start < 0
                or reference.steps <= 0
                or reference.steps > time_steps
                or reference.start + reference.steps > descriptor.length
            ):
                raise ValueError("cache chunk reference is outside its sequence")
            base_offsets.append(descriptor.offset + reference.start)
            valid_steps.append(reference.steps)
            selected_descriptors.append(descriptor)
        bases = torch.tensor(base_offsets, dtype=torch.long)
        steps = torch.arange(time_steps, dtype=torch.long)[:, None]
        available = torch.tensor(valid_steps, dtype=torch.long)[None, :]
        local_steps = torch.minimum(steps, available - 1)
        indices = (bases[None, :] + local_steps).reshape(-1)
        padding_mask = steps < available
        return indices, padding_mask, selected_descriptors, valid_steps

    def _chunk_time_fields(
        self, indices: Tensor, padding_mask: Tensor, selected_descriptors: Sequence[ILCacheSequenceDescriptorV1]
    ) -> dict[str, Tensor | None]:
        time_steps, batch_size = (int(size) for size in padding_mask.shape)
        time_tensors: dict[str, Tensor | None] = {}
        for name in IL_TIME_FIELDS:
            value = getattr(self, name).index_select(0, indices).reshape(time_steps, batch_size)
            value = (
                torch.where(padding_mask, value, torch.zeros_like(value)) if name == "returns" else value & padding_mask
            )
            if name.endswith("_loss_mask") and not any(
                getattr(item, f"{name}_present") for item in selected_descriptors
            ):
                value = None
            time_tensors[name] = value
        return time_tensors

    def load_batch(
        self,
        references: Sequence[ILCacheChunkRefV1],
        *,
        time_steps: int,
        device: torch.device | str | None = None,
        model_float_dtype: torch.dtype = torch.float32,
    ) -> LoadedILBatchV1:
        """Materialize one time-major recurrent batch with one tensor gather."""

        indices, padding_mask, selected_descriptors, valid_steps = self._chunk_gather(references, time_steps=time_steps)
        batch_size = len(references)
        observations = self.observations.index_select(indices)
        actions = self.actions.index_select(indices)
        time_tensors = {}
        for name, value in self._chunk_time_fields(indices, padding_mask, selected_descriptors).items():
            time_tensors[name] = value if device is None or value is None else value.to(device)
        if device is not None:
            observations = observations.to_model_input(device, float_dtype=model_float_dtype)
            actions = actions.to(device)

        sequence = ILSequenceV4(
            observations=tuple(observations.narrow_batch(step * batch_size, batch_size) for step in range(time_steps)),
            actions=tuple(actions.narrow_batch(step * batch_size, batch_size) for step in range(time_steps)),
            **time_tensors,
            sequence_id="+".join(item.sequence_id for item in selected_descriptors),
            natural_time=all(item.natural_time for item in selected_descriptors),
        )
        return LoadedILBatchV1(sequence=sequence, owner_frames=sum(valid_steps))


# Inflate straight into the destination buffer in slices this large, hashing
# each slice while it is still cache-hot.  Every zstd/SHA-256 call releases the
# GIL, so the Python loop around them holds it for microseconds per slice.
_INFLATE_SLICE_BYTES = 16 << 20
# Tensor data starts on this boundary inside the shard buffer so every dtype
# view is aligned (safetensors itself aligns tensor offsets per dtype).
_PAYLOAD_ALIGNMENT = 64
_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    name: dtype
    for name, dtype in (
        ("F64", torch.float64),
        ("F32", torch.float32),
        ("F16", torch.float16),
        ("BF16", torch.bfloat16),
        ("I64", torch.int64),
        ("I32", torch.int32),
        ("I16", torch.int16),
        ("I8", torch.int8),
        ("U8", torch.uint8),
        ("BOOL", torch.bool),
        ("U16", getattr(torch, "uint16", None)),
        ("U32", getattr(torch, "uint32", None)),
        ("U64", getattr(torch, "uint64", None)),
        ("F8_E4M3", getattr(torch, "float8_e4m3fn", None)),
        ("F8_E5M2", getattr(torch, "float8_e5m2", None)),
    )
    if dtype is not None
}
SHARD_LOAD_MODES = ("process", "thread")


def _read_exact(reader: object, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = getattr(reader, "read")(remaining)
        if not chunk:
            raise ValueError("cache container is truncated before its manifest")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _byte_view(buffer: Tensor) -> memoryview:
    """A writable memoryview over a contiguous CPU uint8 tensor (caller keeps ``buffer`` alive)."""

    size = int(buffer.numel())
    if size == 0:
        return memoryview(bytearray())
    return memoryview((ctypes.c_ubyte * size).from_address(buffer.data_ptr())).cast("B")


def _private_buffer(size: int) -> Tensor:
    return torch.empty(size, dtype=torch.uint8)


def _shared_buffer(size: int) -> Tensor:
    """A uint8 tensor allocated directly in shared memory (no copy when sent to another process)."""

    storage = torch.UntypedStorage._new_shared(size)
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (size,), (1,))


def _payload_data_offset(compressed: bytes, container_bytes: int) -> int | None:
    """Container offset of the first tensor byte, read from the first few KiB of the stream.

    Only used to align the destination buffer; any problem here returns
    ``None`` and the full, verified parse reports it with the usual error.
    """

    try:
        import zstandard

        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            magic, manifest_length = _CONTAINER_HEADER.unpack(_read_exact(reader, _CONTAINER_HEADER.size))
            if magic != _CONTAINER_MAGIC or manifest_length >= container_bytes:
                return None
            _read_exact(reader, manifest_length)
            (header_length,) = struct.unpack("<Q", _read_exact(reader, 8))
    except Exception:  # noqa: BLE001 - alignment hint only
        return None
    offset = _CONTAINER_HEADER.size + manifest_length + 8 + header_length
    return offset if offset <= container_bytes else None


def _inflate_into(compressed: bytes, destination: memoryview, hasher: object | None) -> None:
    import zstandard

    size = len(destination)
    position = 0
    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
        while position < size:
            count = reader.readinto(destination[position : min(size, position + _INFLATE_SLICE_BYTES)])
            if not count:
                break
            if hasher is not None:
                getattr(hasher, "update")(destination[position : position + count])
            position += count
        if position != size or reader.read(1):
            raise ValueError("cache container byte count mismatch")


@dataclass(frozen=True, slots=True)
class _ILCacheTensorSpec:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int  # absolute byte offset inside the shard buffer
    nbytes: int


def _safetensors_specs(container: memoryview, start: int, base: int) -> tuple[_ILCacheTensorSpec, ...]:
    """Validate a safetensors payload at ``container[start:]`` like ``safetensors.deserialize``."""

    total = len(container)
    if total - start < 8:
        raise ValueError("cache safetensors payload is truncated")
    (header_length,) = struct.unpack_from("<Q", container, start)
    data_start = start + 8 + header_length
    if data_start > total:
        raise ValueError("cache safetensors header length is invalid")
    header = json.loads(bytes(container[start + 8 : data_start]))
    if not isinstance(header, dict):
        raise ValueError("cache safetensors header is not an object")
    specs: list[_ILCacheTensorSpec] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"cache safetensors entry {name!r} is invalid")
        dtype = _SAFETENSORS_DTYPES.get(str(info.get("dtype")))
        if dtype is None:
            raise ValueError(f"cache safetensors entry {name!r} has unsupported dtype {info.get('dtype')!r}")
        shape = tuple(int(item) for item in info["shape"])
        begin, end = (int(item) for item in info["data_offsets"])
        numel = reduce(lambda left, right: left * right, shape, 1)
        if begin < 0 or end < begin or end - begin != numel * dtype.itemsize:
            raise ValueError(f"cache safetensors entry {name!r} has invalid offsets")
        specs.append(_ILCacheTensorSpec(name, dtype, shape, base + data_start + begin, end - begin))
    expected = 0
    for spec in sorted(specs, key=lambda item: item.offset):
        if spec.offset - base - data_start != expected:
            raise ValueError("cache safetensors tensors are not contiguous")
        expected += spec.nbytes
    if data_start + expected != total:
        raise ValueError("cache safetensors data length mismatch")
    return tuple(specs)


def _tensor_from_spec(buffer: Tensor, spec: _ILCacheTensorSpec) -> Tensor:
    if spec.nbytes == 0:
        return torch.empty(spec.shape, dtype=spec.dtype)
    view = buffer.narrow(0, spec.offset, spec.nbytes)
    if spec.offset % spec.dtype.itemsize:
        view = view.clone()  # unaligned writer: copy once into an aligned allocation
    return view.view(spec.dtype).view(spec.shape)


@dataclass(frozen=True, slots=True)
class _DecodedILCacheShard:
    """A verified shard whose tensors all live in one aligned ``uint8`` buffer.

    This is what a loader process returns: one buffer (one shared-memory
    segment, one fd under the ``file_descriptor`` strategy) plus metadata.
    """

    data_path: str
    summary: dict[str, object]
    manifest: dict[str, object]
    buffer: Tensor
    specs: tuple[_ILCacheTensorSpec, ...]

    def build(self, config: ModelConfigV4 | None = None) -> ILCacheShardV1:
        tensors = {spec.name: _tensor_from_spec(self.buffer, spec) for spec in self.specs}
        return ILCacheShardV1(
            path=Path(self.data_path), manifest=self.manifest, tensors=tensors, summary=self.summary, config=config
        )


def _decode_il_cache_shard(
    summary_path: str | Path, *, verify: bool, allocate: Callable[[int], Tensor] = _private_buffer
) -> _DecodedILCacheShard:
    """Verify and inflate one shard without materialising Python ``bytes`` of its payload.

    Same checks, order and error messages as the original reader (inflate to
    ``bytes``, slice, ``safetensors.torch.load``), whose slice copy and
    per-tensor ``bytes`` objects held the GIL for ~0.6 s per 1.8 GB shard.
    Here zstd writes straight into ``allocate``'s buffer and SHA-256 runs over
    it slice by slice, both with the GIL released, and every tensor is a view
    into that buffer.  Tensor bytes are identical.
    """

    try:
        import zstandard  # noqa: F401
    except ImportError as error:
        raise RuntimeError("V4 IL caching requires safetensors and zstandard") from error
    if sys.byteorder != "little":
        raise RuntimeError("V4 IL cache loading requires a little-endian host")

    summary_file = Path(summary_path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema") != IL_CACHE_SUMMARY_SCHEMA:
        raise ValueError("unsupported V4 IL cache summary")
    data_name = summary.get("data_file")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise ValueError("cache summary data file must be a local basename")
    data_path = summary_file.parent / data_name
    compressed = data_path.read_bytes()
    if verify and _sha256(compressed) != summary.get("compressed_sha256"):
        raise ValueError("cache compressed SHA-256 mismatch")
    container_bytes = int(summary["container_bytes"])
    data_offset = _payload_data_offset(compressed, container_bytes)
    pad = 0 if data_offset is None else -data_offset % _PAYLOAD_ALIGNMENT
    buffer = allocate(pad + container_bytes)
    container = _byte_view(buffer)[pad:]
    hasher = hashlib.sha256() if verify else None
    _inflate_into(compressed, container, hasher)
    del compressed
    if hasher is not None and hasher.hexdigest() != summary.get("container_sha256"):
        raise ValueError("cache container SHA-256 mismatch")
    if len(container) < _CONTAINER_HEADER.size:
        raise ValueError("cache container is truncated")
    magic, manifest_length = _CONTAINER_HEADER.unpack_from(container)
    if magic != _CONTAINER_MAGIC:
        raise ValueError("cache container magic mismatch")
    manifest_start = _CONTAINER_HEADER.size
    manifest_end = manifest_start + manifest_length
    if manifest_end >= len(container):
        raise ValueError("cache manifest length is invalid")
    manifest = json.loads(bytes(container[manifest_start:manifest_end]))
    if not isinstance(manifest, dict) or manifest.get("schema") != IL_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported V4 IL cache shard")
    specs = _safetensors_specs(container, manifest_end, pad)
    return _DecodedILCacheShard(
        data_path=str(data_path), summary=summary, manifest=manifest, buffer=buffer, specs=specs
    )


def load_il_cache_shard(
    summary_path: str | Path, *, verify: bool = True, config: ModelConfigV4 | None = None
) -> ILCacheShardV1:
    """Verify, decompress, and load one shard into CPU memory.

    ``config`` (default ``ModelConfigV4()``) selects observation-contract
    migrations applied on load, e.g. ``neutralize_pending_ticks_remaining``.
    The heavy work (zstd, SHA-256) runs with the GIL released and no payload
    copy is made, so a background thread loading shards barely contends with
    the training thread.
    """

    return _decode_il_cache_shard(summary_path, verify=verify).build(config)


def _default_sharing_strategy() -> str:
    """``file_descriptor`` where available and the fd limit is sane, else ``file_system``.

    Each loaded shard is one shared-memory segment, so the descriptor strategy
    needs one fd per resident shard in this process (plus one transient dup
    per shard in flight).  A soft ``RLIMIT_NOFILE`` below 4096 is raised toward
    the hard limit; below 1024 after that, ``file_system`` is used instead.
    """

    strategies = torch.multiprocessing.get_all_sharing_strategies()
    if "file_descriptor" not in strategies:
        return torch.multiprocessing.get_sharing_strategy()
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        wanted = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
        if soft != resource.RLIM_INFINITY and soft < wanted:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
            soft = wanted
        if soft != resource.RLIM_INFINITY and soft < 1024:
            return "file_system"
    except (ImportError, OSError, ValueError):
        return "file_system"
    return "file_descriptor"


def shared_memory_free_bytes() -> int | None:
    """Free bytes of ``/dev/shm`` (where both torch sharing strategies allocate); ``None`` if unknown."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return None
    return int(stats.f_bavail) * int(stats.f_frsize)


def _loader_process_init(sharing_strategy: str) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    torch.set_num_threads(1)


def _loader_process_ready() -> int:
    return os.getpid()


def _decode_il_cache_shard_shared(summary_path: str, verify: bool) -> _DecodedILCacheShard:
    return _decode_il_cache_shard(summary_path, verify=verify, allocate=_shared_buffer)


class ILCacheProcessLoader:
    """Decode shards in spawned worker processes into shared memory.

    Workers verify SHA-256 and inflate exactly like ``load_il_cache_shard``,
    but into a shared-memory buffer that reaches this process without a copy;
    only the ``ILCacheShardV1`` construction (tensor views) runs here, so no
    shard-loading work competes for this interpreter's GIL.  ``spawn`` keeps
    CUDA/NCCL state out of the workers; they start lazily (or via ``warm``)
    and live until ``close``.
    """

    def __init__(self, workers: int, *, sharing_strategy: str | None = None, start_method: str = "spawn") -> None:
        if workers <= 0:
            raise ValueError("cache loader processes must be positive")
        self.workers = workers
        self.sharing_strategy = sharing_strategy or _default_sharing_strategy()
        self.start_method = start_method
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=torch.multiprocessing.get_context(start_method),
            initializer=_loader_process_init,
            initargs=(self.sharing_strategy,),
        )

    def warm(self, timeout: float | None = 300.0) -> int:
        """Start every worker and wait until each answers; returns the distinct worker count."""

        futures = [self._executor.submit(_loader_process_ready) for _ in range(self.workers)]
        return len({future.result(timeout=timeout) for future in futures})

    def submit(self, summary_path: str | Path, *, verify: bool = True) -> Future[_DecodedILCacheShard]:
        return self._executor.submit(_decode_il_cache_shard_shared, str(summary_path), verify)

    def load(
        self, summary_paths: Sequence[str | Path], *, verify: bool = True, config: ModelConfigV4 | None = None
    ) -> tuple[ILCacheShardV1, ...]:
        futures = [self.submit(path, verify=verify) for path in summary_paths]
        try:
            return tuple(future.result().build(config) for future in futures)
        finally:
            for future in futures:
                future.cancel()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> ILCacheProcessLoader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_il_cache_shards(
    summary_paths: Sequence[str | Path],
    *,
    verify: bool = True,
    workers: int = 1,
    config: ModelConfigV4 | None = None,
    process_loader: ILCacheProcessLoader | None = None,
) -> tuple[ILCacheShardV1, ...]:
    """Load independent shards in input order, optionally in parallel.

    With ``process_loader`` the shards are decoded by its worker processes
    (whose pool size then bounds the parallelism); otherwise by up to
    ``workers`` threads of this process.  Both give byte-identical tensors.
    """

    paths = tuple(summary_paths)
    if not paths:
        raise ValueError("cache shard loading needs at least one summary")
    if workers <= 0:
        raise ValueError("cache shard loading workers must be positive")
    if process_loader is not None:
        return process_loader.load(paths, verify=verify, config=config)
    if workers == 1 or len(paths) == 1:
        return tuple(load_il_cache_shard(path, verify=verify, config=config) for path in paths)
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
        return tuple(executor.map(lambda path: load_il_cache_shard(path, verify=verify, config=config), paths))


def load_il_cache_manifest(summary_path: str | Path) -> dict[str, object]:
    """Read only a shard manifest without inflating its tensor payload.

    Dataset planning needs replay tags, owner sequence indices, and lengths for
    every shard.  Streaming just the container header and JSON manifest keeps
    that preflight proportional to metadata instead of the multi-terabyte
    uncompressed tensor corpus.
    """

    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("V4 IL caching requires zstandard") from error

    summary_file = Path(summary_path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema") != IL_CACHE_SUMMARY_SCHEMA:
        raise ValueError("unsupported V4 IL cache summary")
    data_name = summary.get("data_file")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise ValueError("cache summary data file must be a local basename")

    data_path = summary_file.parent / data_name
    with data_path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            header = _read_exact(reader, _CONTAINER_HEADER.size)
            magic, manifest_length = _CONTAINER_HEADER.unpack(header)
            if magic != _CONTAINER_MAGIC or manifest_length <= 0:
                raise ValueError("cache container header is invalid")
            manifest = json.loads(_read_exact(reader, manifest_length))
    if not isinstance(manifest, dict) or manifest.get("schema") != IL_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported V4 IL cache shard")
    return manifest


def collate_loaded_il_batches(
    batches: Sequence[LoadedILBatchV1], *, device: torch.device | str | None = None
) -> LoadedILBatchV1:
    """Join equal-time batches from different cache shards losslessly."""

    if not batches:
        raise ValueError("cross-shard IL collation needs at least one batch")
    time_steps = batches[0].sequence.time_steps
    if any(batch.sequence.time_steps != time_steps for batch in batches):
        raise ValueError("cross-shard IL batches must have equal time lengths")
    capacities = {
        count_name: max(int(getattr(batch.sequence.observations[0], name).mask.shape[1]) for batch in batches)
        for name, count_name in _DYNAMIC_COUNTS.items()
    }
    padded = tuple(_pad_sequence_dynamic_observations(batch.sequence, **capacities) for batch in batches)
    sequence = collate_il_sequences(padded)
    if device is not None:
        sequence = sequence.to(device)
    return LoadedILBatchV1(sequence=sequence, owner_frames=sum(batch.owner_frames for batch in batches))


_IL_CHUNK_TENSOR_FIELDS = IL_TIME_FIELDS + IL_ANNOTATION_FIELDS


def _map_record_tensors(value: _RecordT, transform: Callable[[Tensor], Tensor]) -> _RecordT:
    return type(value)(
        **{
            item.name: transform(child) if isinstance(child, Tensor) else _map_record_tensors(child, transform)
            for item in fields(value)
            for child in (getattr(value, item.name),)
        }
    )


@dataclass(frozen=True, slots=True)
class CollatedILChunkV1:
    """One cross-shard ``[T,B]`` chunk held as flat time-major ``[T*B]`` records.

    ``sequence()`` returns the ``ILSequenceV4`` whose per-step records are
    zero-copy views of these records, so moving a chunk to a device costs one
    copy per field instead of one per field and time step.
    """

    observations: UniversalSemanticBatchV4
    actions: ActionSequenceV4
    time_fields: Mapping[str, Tensor | None]
    time_steps: int
    batch_size: int
    sequence_id: str
    natural_time: bool
    owner_frames: int
    # Host-side chunk plan: per-lane distinct leading frames (``ref.steps``;
    # later frames repeat the last one) and ``episode_start[1:].any()``.
    lane_steps: tuple[int, ...] = ()
    mid_chunk_reset: bool | None = None

    def sequence(self, *, plan: bool = False) -> ILSequenceV4:
        """The chunk as an ``ILSequenceV4``; ``plan=True`` returns an ``ILChunkSequenceV4`` carrying the host plan."""

        batch_size = self.batch_size
        observations = tuple(
            self.observations.narrow_batch(step * batch_size, batch_size) for step in range(self.time_steps)
        )
        actions = tuple(self.actions.narrow_batch(step * batch_size, batch_size) for step in range(self.time_steps))
        if plan:
            return ILChunkSequenceV4(
                observations=observations,
                actions=actions,
                **self.time_fields,
                sequence_id=self.sequence_id,
                natural_time=self.natural_time,
                lane_steps=self.lane_steps if len(self.lane_steps) == batch_size else None,
                mid_chunk_reset=self.mid_chunk_reset,
            )
        return ILSequenceV4(
            observations=observations,
            actions=actions,
            **self.time_fields,
            sequence_id=self.sequence_id,
            natural_time=self.natural_time,
        )

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            *(tensor for _name, tensor in _record_tensor_items(self.observations)),
            *(tensor for _name, tensor in _record_tensor_items(self.actions)),
            *(tensor for tensor in self.time_fields.values() if tensor is not None),
        )

    def to_model_input(
        self, device: torch.device | str, *, float_dtype: torch.dtype = torch.float32, non_blocking: bool = False
    ) -> CollatedILChunkV1:
        """Field-for-field equal to ``sequence().to(device)``.

        Floating observations cross in their storage dtype and are widened on
        the destination (an exact conversion), so a pinned chunk moves
        asynchronously at storage width.
        """

        if not float_dtype.is_floating_point:
            raise ValueError("V4 model input dtype must be floating point")

        def observation(item: Tensor) -> Tensor:
            moved = item.to(device, non_blocking=non_blocking)
            return moved.to(float_dtype) if moved.is_floating_point() else moved

        def plain(item: Tensor) -> Tensor:
            return item.to(device, non_blocking=non_blocking)

        return replace(
            self,
            observations=_map_record_tensors(self.observations, observation),
            actions=_map_record_tensors(self.actions, plain),
            time_fields={name: None if value is None else plain(value) for name, value in self.time_fields.items()},
        )


def _gather_rows(source: Tensor, indices: Tensor, *, pin_memory: bool) -> Tensor:
    if not pin_memory:
        return source.index_select(0, indices)
    output = torch.empty((int(indices.numel()), *source.shape[1:]), dtype=source.dtype, pin_memory=True)
    return torch.index_select(source, 0, indices, out=output)


# Observation collections whose padded rows can be cut, and the mask bounding
# each field's row dimension.  Not trimmed: opponent cards (positional slots),
# towers/own cards/previous actions (fixed widths) and candidates: the
# decoder's shadow features average ``used_candidate``/``legal`` over the
# candidate dimension (``ShadowCandidateLegality.shadow_features``), so the
# candidate width itself is a model input.
_TRIMMABLE_COLLECTIONS = ("groups", "active_effects", "events", "relation_edges")


def _trim_mask_name(collection: str, field: str) -> str | None:
    if collection not in _TRIMMABLE_COLLECTIONS or field == "elixir":
        return None
    return "child_mask" if collection == "groups" and field.startswith("child_") else "mask"


def _trimmed_widths(
    shards: Sequence[ILCacheShardV1], gathers: Sequence[tuple[Tensor, Tensor, object, object]]
) -> dict[tuple[str, str], int]:
    """Per ``(collection, mask)``: 1 + the last row any gathered frame uses (at least 1).

    Every model input past that row is canonical padding behind a false mask
    in every frame of the chunk, so slicing it off leaves every masked
    reduction, gather address (rows are referenced only below a set mask bit)
    and token address (groups are the last token block) unchanged.
    """

    widths: dict[tuple[str, str], int] = {}
    for collection in _TRIMMABLE_COLLECTIONS:
        for mask_name in ("mask", "child_mask") if collection == "groups" else ("mask",):
            width = 0
            for shard, (indices, *_rest) in zip(shards, gathers, strict=True):
                used = getattr(getattr(shard.observations, collection), mask_name).index_select(0, indices).any(dim=0)
                rows = used.nonzero()
                if rows.numel():
                    width = max(width, int(rows[-1, 0]) + 1)
            widths[(collection, mask_name)] = max(1, width)
    return widths


def collate_il_cache_chunk(
    shards: Sequence[ILCacheShardV1],
    references: Sequence[Sequence[ILCacheChunkRefV1]],
    *,
    time_steps: int,
    pin_memory: bool = False,
    trim_padding: bool = False,
) -> CollatedILChunkV1:
    """``collate_loaded_il_batches`` of each shard's ``load_batch``, one gather per field.

    Every field is gathered once per shard and joined lane-wise into one
    time-major ``[T*B]`` tensor, instead of padding and concatenating every
    time step of every shard.  Values, dtypes, shapes, lane order and metadata
    match the two-stage path exactly.  ``pin_memory`` allocates the joined
    tensors page-locked so ``to_model_input(..., non_blocking=True)`` is
    asynchronous.

    ``trim_padding`` cuts the padded row dimension of battle groups, children,
    active effects, events and relation edges to the chunk's widest real use
    (``_trimmed_widths``) before gathering, so the padding is neither copied
    nor computed on; model outputs are unchanged up to float rounding.
    """

    if not shards or len(shards) != len(references):
        raise ValueError("cross-shard IL collation needs one reference list per shard")
    gathers = [shard._chunk_gather(refs, time_steps=time_steps) for shard, refs in zip(shards, references, strict=True)]
    lane_counts = [len(refs) for refs in references]
    batch_size = sum(lane_counts)
    single = len(shards) == 1
    capacities = {
        name: max(int(getattr(shard.observations, name).mask.shape[1]) for shard in shards) for name in _DYNAMIC_COUNTS
    }
    widths = _trimmed_widths(shards, gathers) if trim_padding else {}

    def join(prefix: str, records: Sequence[TensorRecordV4]) -> dict[str, Tensor]:
        items = [dict(_record_tensor_items(record, prefix=prefix)) for record in records]
        if any(item.keys() != items[0].keys() for item in items):
            raise TypeError("cannot concatenate different V4 record types")
        joined: dict[str, Tensor] = {}
        for key in items[0]:
            parts = key.split(".")
            target = capacities.get(parts[1]) if len(parts) == 3 and parts[2] != "elixir" else None
            mask_name = (
                _trim_mask_name(parts[1], parts[2]) if widths and prefix == "observations" and len(parts) == 3 else None
            )
            width = None if mask_name is None else widths[(parts[1], mask_name)]
            if width is not None:
                target = width
            padding_value = -1 if parts[-1] in _NEGATIVE_PADDING_FIELDS else 0
            pieces: list[Tensor] = []
            for shard_items, (indices, _mask, _descriptors, _steps) in zip(items, gathers, strict=True):
                source = shard_items[key]
                if width is not None and source.ndim >= 2 and int(source.shape[1]) > width:
                    source = source.narrow(1, 0, width)
                pads = target is not None and source.ndim >= 2 and int(source.shape[1]) < target
                piece = _gather_rows(source, indices, pin_memory=pin_memory and single and not pads)
                if target is not None:
                    piece = _pad_dim_one(piece, target, padding_value, pin_memory=pin_memory and single)
                pieces.append(piece)
            if single:
                joined[key] = pieces[0]
                continue
            output = torch.empty(
                (time_steps, batch_size, *pieces[0].shape[1:]),
                dtype=reduce(torch.promote_types, (piece.dtype for piece in pieces)),
                pin_memory=pin_memory,
            )
            torch.cat(
                [
                    piece.view(time_steps, lanes, *piece.shape[1:])
                    for piece, lanes in zip(pieces, lane_counts, strict=True)
                ],
                dim=1,
                out=output,
            )
            joined[key] = output.view(time_steps * batch_size, *output.shape[2:])
        return joined

    observation_tensors = join("observations", [shard.observations for shard in shards])
    observations = _restore_record(type(shards[0].observations), prefix="observations", tensors=observation_tensors)
    action_tensors = join("actions", [shard.actions for shard in shards])
    actions = _restore_record(type(shards[0].actions), prefix="actions", tensors=action_tensors)
    shard_time_fields = [
        shard._chunk_time_fields(indices, mask, descriptors)
        for shard, (indices, mask, descriptors, _steps) in zip(shards, gathers, strict=True)
    ]
    time_fields: dict[str, Tensor | None] = {}
    for name in _IL_CHUNK_TENSOR_FIELDS:
        values = [fields_.get(name) for fields_ in shard_time_fields]
        if all(value is None for value in values):
            time_fields[name] = None
            continue
        # An absent loss mask means "every valid frame" (as in collate_il_sequences).
        value = torch.cat(
            [fields_["valid_mask"] if value is None else value for fields_, value in zip(shard_time_fields, values)],
            dim=1,
        )
        time_fields[name] = value.pin_memory() if pin_memory else value
    valid = time_fields["valid_mask"]
    for name in ("gate_loss_mask", "value_loss_mask"):
        loss_mask = time_fields[name]
        if loss_mask is not None and valid is not None and torch.any(loss_mask & ~valid):
            raise ValueError(f"{name} must be a subset of valid frames")
    returns = time_fields["returns"]
    if returns is None or not returns.is_floating_point() or not torch.isfinite(returns).all():
        raise ValueError("IL returns must be finite floating [T,B]")
    shard_ids = ("+".join(item.sequence_id for item in descriptors) for _indices, _mask, descriptors, _steps in gathers)
    episode_start = time_fields["episode_start"]
    return CollatedILChunkV1(
        observations=observations,
        actions=actions,
        time_fields=time_fields,
        time_steps=time_steps,
        batch_size=batch_size,
        sequence_id="+".join(shard_id or f"sequence-{index}" for index, shard_id in enumerate(shard_ids)),
        natural_time=all(item.natural_time for _indices, _mask, descriptors, _steps in gathers for item in descriptors),
        owner_frames=sum(sum(steps) for _indices, _mask, _descriptors, steps in gathers),
        lane_steps=tuple(int(step) for _indices, _mask, _descriptors, steps in gathers for step in steps),
        mid_chunk_reset=None if episode_start is None else bool(episode_start[1:].any()),
    )


def write_il_cache_index(
    destination: str | Path, summaries: Sequence[Mapping[str, object]], *, contract: Mapping[str, object]
) -> dict[str, object]:
    """Write a canonical root index for independently verified shards."""

    shards = [dict(summary) for summary in summaries]
    payload: dict[str, object] = {
        "schema": IL_CACHE_INDEX_SCHEMA,
        "contract": dict(contract),
        "shards": shards,
        "shard_count": len(shards),
        "replay_results": sum(int(item["replay_results"]) for item in shards),
        "sequences": sum(int(item["sequences"]) for item in shards),
        "owner_frames": sum(int(item["owner_frames"]) for item in shards),
        "match_decision_frames": sum(int(item["match_decision_frames"]) for item in shards),
        "compressed_bytes": sum(int(item["compressed_bytes"]) for item in shards),
        "container_bytes": sum(int(item["container_bytes"]) for item in shards),
    }
    for key in (
        "completed_replays",
        "failed_replays",
        "zero_frame_failures",
        "failures_before_40s",
        "unexpected_unknown_failures",
        "winner_comparable_replays",
        "winner_mismatches",
        "crowns_comparable_replays",
        "crowns_mismatches",
        "expert_actions",
        "executed_expert_actions",
        "decision_frames",
    ):
        payload[key] = sum(int(item.get(key, 0)) for item in shards)
    digest = _sha256(_canonical_json(payload))
    payload["index_sha256"] = digest
    _atomic_write_json(Path(destination), payload)
    return payload


def load_il_cache_index(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != IL_CACHE_INDEX_SCHEMA:
        raise ValueError("unsupported V4 IL cache index")
    expected = payload.get("index_sha256")
    unsigned = dict(payload)
    unsigned.pop("index_sha256", None)
    if not isinstance(expected, str) or _sha256(_canonical_json(unsigned)) != expected:
        raise ValueError("V4 IL cache index SHA-256 mismatch")
    return payload


def assert_il_sequences_exact(expected: ILSequenceV4, actual: ILSequenceV4) -> None:
    """Fail on any cache mutation of tensor values, dtypes, shapes, or metadata."""

    expected.validate()
    actual.validate()
    for label in ("sequence_id", "natural_time", "time_steps", "batch_size"):
        if getattr(expected, label) != getattr(actual, label):
            raise AssertionError(f"cached IL {label} differs")
    for field in ("observations", "actions"):
        for step, (left, right) in enumerate(zip(getattr(expected, field), getattr(actual, field), strict=True)):
            left_items = dict(_record_tensor_items(left))
            right_items = dict(_record_tensor_items(right))
            if left_items.keys() != right_items.keys():
                raise AssertionError(f"cached {field} {step} fields differ")
            for name, left_tensor in left_items.items():
                right_tensor = right_items[name]
                if (
                    left_tensor.dtype != right_tensor.dtype
                    or left_tensor.shape != right_tensor.shape
                    or not torch.equal(left_tensor, right_tensor)
                ):
                    raise AssertionError(f"cached {field} {step}.{name} differs")
    for name in IL_TIME_FIELDS:
        left = getattr(expected, name)
        right = getattr(actual, name)
        if (left is None) != (right is None):
            raise AssertionError(f"cached IL {name} presence differs")
        if left is not None and right is not None:
            if left.dtype != right.dtype or left.shape != right.shape:
                raise AssertionError(f"cached IL {name} contract differs")
            if not torch.equal(left, right):
                raise AssertionError(f"cached IL {name} differs")
