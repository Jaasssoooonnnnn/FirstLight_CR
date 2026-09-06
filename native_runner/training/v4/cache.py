"""Lossless sharded cache for model-ready V4 imitation sequences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Mapping, Sequence, TypeVar, get_type_hints

import torch
from torch import Tensor

from .imitation import ILSequenceV4, IL_TIME_FIELDS, collate_il_sequences
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


def _pad_dim_one(value: Tensor, target: int, padding_value: object) -> Tensor:
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
    padded = torch.full(shape, padding_value, dtype=value.dtype, device=value.device)
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
    packed_observations = concatenate_tensor_records(
        tuple(
            _pad_dynamic_observation(
                observation,
                active_effect_count=active_effect_capacity,
                relation_edge_count=relation_edge_capacity,
                candidate_count=candidate_capacity,
            )
            for observation in observations
        )
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
    ) -> None:
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

    def load_batch(
        self,
        references: Sequence[ILCacheChunkRefV1],
        *,
        time_steps: int,
        device: torch.device | str | None = None,
        model_float_dtype: torch.dtype = torch.float32,
    ) -> LoadedILBatchV1:
        """Materialize one time-major recurrent batch with one tensor gather."""

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
        batch_size = len(references)
        bases = torch.tensor(base_offsets, dtype=torch.long)
        steps = torch.arange(time_steps, dtype=torch.long)[:, None]
        available = torch.tensor(valid_steps, dtype=torch.long)[None, :]
        local_steps = torch.minimum(steps, available - 1)
        indices = (bases[None, :] + local_steps).reshape(-1)
        padding_mask = steps < available

        observations = self.observations.index_select(indices)
        actions = self.actions.index_select(indices)
        time_tensors = {}
        for name in IL_TIME_FIELDS:
            value = getattr(self, name).index_select(0, indices).reshape(time_steps, batch_size)
            value = (
                torch.where(padding_mask, value, torch.zeros_like(value)) if name == "returns" else value & padding_mask
            )
            if name.endswith("_loss_mask") and not any(
                getattr(item, f"{name}_present") for item in selected_descriptors
            ):
                value = None
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


def load_il_cache_shard(summary_path: str | Path, *, verify: bool = True) -> ILCacheShardV1:
    """Verify, decompress, and load one shard into CPU memory."""

    try:
        import zstandard
        from safetensors.torch import load as load_safetensors
    except ImportError as error:
        raise RuntimeError("V4 IL caching requires safetensors and zstandard") from error

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
    decompressor = zstandard.ZstdDecompressor()
    container = decompressor.decompress(compressed, max_output_size=int(summary["container_bytes"]))
    if len(container) != int(summary["container_bytes"]):
        raise ValueError("cache container byte count mismatch")
    if verify and _sha256(container) != summary.get("container_sha256"):
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
    manifest = json.loads(container[manifest_start:manifest_end])
    if not isinstance(manifest, dict) or manifest.get("schema") != IL_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported V4 IL cache shard")
    tensors = load_safetensors(container[manifest_end:])
    return ILCacheShardV1(path=data_path, manifest=manifest, tensors=tensors, summary=summary)


def load_il_cache_shards(
    summary_paths: Sequence[str | Path], *, verify: bool = True, workers: int = 1
) -> tuple[ILCacheShardV1, ...]:
    """Load independent shards in input order, optionally in parallel."""

    paths = tuple(summary_paths)
    if not paths:
        raise ValueError("cache shard loading needs at least one summary")
    if workers <= 0:
        raise ValueError("cache shard loading workers must be positive")
    if workers == 1 or len(paths) == 1:
        return tuple(load_il_cache_shard(path, verify=verify) for path in paths)
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
        return tuple(executor.map(lambda path: load_il_cache_shard(path, verify=verify), paths))


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

    def read_exact(reader: object, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = getattr(reader, "read")(remaining)
            if not chunk:
                raise ValueError("cache container is truncated before its manifest")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    data_path = summary_file.parent / data_name
    with data_path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            header = read_exact(reader, _CONTAINER_HEADER.size)
            magic, manifest_length = _CONTAINER_HEADER.unpack(header)
            if magic != _CONTAINER_MAGIC or manifest_length <= 0:
                raise ValueError("cache container header is invalid")
            manifest = json.loads(read_exact(reader, manifest_length))
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
