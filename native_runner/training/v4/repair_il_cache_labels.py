"""Repair alignment-dropped expert labels in existing V4 IL cache shards.

The producer stores an expert window as WAIT with ``gate_loss_mask`` False
when ``build_expert_action_batch`` cannot align it.  Before delay-aware
legality, a card touched late in its five-tick window whose candidate was
affordable only after elixir regenerated was dropped this way.  This tool
re-runs the current (delay-aware) alignment against each such frame's STORED
candidates and observation, and writes a repaired copy of every shard:

* the frame's ``ActionSequenceV4`` row becomes the aligned ACT label (with the
  touch's delay bin) and its ``gate_loss_mask`` bit is set;
* the next frame's ``previous_action`` record is rewritten exactly as
  ``UniversalObservationTensorizerV4.record_action`` would have written it;
* every other tensor stays bit-identical.

Frames that still fail, or that were dropped for reasons a label rewrite cannot
fix, are left untouched and counted per reason.  Notably, a card whose
candidate row is ABSENT from the stored candidate set (the native action mask
omits hand slots it cannot afford at the decision tick, so the tensorizer
never emitted the row) is counted as ``candidate_absent`` and cannot be
repaired offline.  Expert ability activations are never repaired: the producer
does not register a dropped ability as a pending command, so later
observations would disagree with a restored label.  Source actions come from
the uncalibrated replay (deal calibration needs a native engine and only
rewrites hand slots), so only lanes whose supervised card plays all agree with
the uncalibrated hand slots are repaired, and the slot must match exactly.

The tool always aligns with delay-aware legality enabled explicitly.
Repaired labels are legal only under that rule, which is OFF by default
(``decoding.DELAY_AWARE_LEGALITY_DEFAULT = False``); teacher-force a repaired
cache only with it enabled, i.e. after the full C1 fix
(artifacts/il-arch-study-20261023/datacheck/C1-full-fix-design.md).
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from ...contracts import ActionKind
from ...perspective import PerspectiveTransformV1
from .cache import (
    IL_CACHE_SUMMARY_SCHEMA,
    IL_CACHE_SHARD_SCHEMA,
    ILCacheShardV1,
    _CONTAINER_HEADER,
    _CONTAINER_MAGIC,
    _atomic_write,
    _atomic_write_json,
    _canonical_json,
    _sha256,
    load_il_cache_index,
    write_il_cache_index,
)
from .config import ModelConfigV4
from .decoding import ShadowCandidateLegality, elixir_multiplier_from_match_scalars
from .expert import (
    FIRST_POLICY_DECISION_TICK,
    POLICY_DECISION_TICKS,
    ExpertActionAlignmentError,
    TimedExpertActionV4,
    build_expert_action_batch,
    group_expert_action_windows,
    replay_expert_actions,
)
from .tensors import (
    ActionSequenceV4,
    CANDIDATE_ABILITY,
    CANDIDATE_DEPLOY,
    GATE_ACT,
    GATE_WAIT,
    TARGET_GRID,
    UniversalSemanticBatchV4,
)


LABEL_REPAIR_SCHEMA = "v4-il-cache-label-repair.v1"
ACTION_FIELDS = ("gate", "micro_action_count", "candidate_index", "candidate_uid", "target_cell", "delay_offset_bin")
PREVIOUS_ACTION_FIELDS = (
    "gate",
    "micro_action_count",
    "variant",
    "visible_card_vocab_id",
    "effective_card_vocab_id",
    "effective_form",
    "ability_vocab_id",
    "target_cell",
    "delay_offset_bin",
    "source_position",
    "source_mask",
    "action_mask",
)
_SUMMARY_CONTAINER_FIELDS = (
    "compressed_sha256",
    "container_sha256",
    "compressed_bytes",
    "container_bytes",
    "compression_ratio",
)

ExpertActionsFn = Callable[[Mapping[str, Any]], Sequence[TimedExpertActionV4]]
PayloadForTagFn = Callable[[str], Mapping[str, Any]]


def payload_expert_actions(payload: Mapping[str, Any]) -> tuple[TimedExpertActionV4, ...]:
    """Producer path from a source payload to timed native expert actions.

    ``prepare_collected_replay`` applies the producer's ``data_i`` owner/side
    mapping and native coordinates.  The producer additionally calibrates the
    opening deal against a native engine (``calibrate_collected_replay_deal``);
    that only rewrites PLAY_CARD hand slots, so repair is restricted to lanes
    whose supervised card plays prove the uncalibrated slots agree.
    """

    from ...royaleapi_replay import prepare_collected_replay

    return replay_expert_actions(prepare_collected_replay(payload).replay)


def _tool_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("repair_il_cache_labels.py", "decoding.py", "expert.py")
    }


# --------------------------------------------------------------------------
# Raw container I/O (bit-exact; the loader's in-memory migrations are only
# used for legality decisions, never written back).


@dataclass(slots=True)
class RawShardV1:
    summary: dict[str, Any]
    compressed: bytes
    manifest: dict[str, Any]
    tensors: dict[str, Tensor]


def read_raw_shard(summary_path: str | Path) -> RawShardV1:
    import zstandard
    from safetensors.torch import load as load_safetensors

    summary_file = Path(summary_path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema") != IL_CACHE_SUMMARY_SCHEMA:
        raise ValueError(f"unsupported V4 IL cache summary {summary_file}")
    data_name = summary.get("data_file")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise ValueError("cache summary data file must be a local basename")
    compressed = (summary_file.parent / data_name).read_bytes()
    if _sha256(compressed) != summary.get("compressed_sha256"):
        raise ValueError(f"cache compressed SHA-256 mismatch for {summary_file}")
    container = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=int(summary["container_bytes"]))
    if len(container) != int(summary["container_bytes"]) or _sha256(container) != summary.get("container_sha256"):
        raise ValueError(f"cache container mismatch for {summary_file}")
    magic, manifest_length = _CONTAINER_HEADER.unpack_from(container)
    if magic != _CONTAINER_MAGIC:
        raise ValueError("cache container magic mismatch")
    manifest_end = _CONTAINER_HEADER.size + manifest_length
    manifest = json.loads(container[_CONTAINER_HEADER.size : manifest_end])
    if not isinstance(manifest, dict) or manifest.get("schema") != IL_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported V4 IL cache shard")
    tensors = dict(load_safetensors(container[manifest_end:]))
    return RawShardV1(summary=summary, compressed=compressed, manifest=manifest, tensors=tensors)


def _summary_label_repair(label_repair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": LABEL_REPAIR_SCHEMA,
        "repaired_frames": int(label_repair["repaired_frames"]),
        "attempted_frames": int(label_repair["attempted_frames"]),
        "per_reason_counts": dict(label_repair["per_reason_counts"]),
    }


def write_raw_shard(
    summary_path: Path,
    *,
    source: RawShardV1,
    manifest: Mapping[str, Any],
    tensors: Mapping[str, Tensor],
    compression_level: int,
) -> dict[str, Any]:
    import zstandard
    from safetensors.torch import save as save_safetensors

    manifest_payload = _canonical_json(dict(manifest))
    container = (
        _CONTAINER_HEADER.pack(_CONTAINER_MAGIC, len(manifest_payload))
        + manifest_payload
        + save_safetensors({name: tensor.contiguous() for name, tensor in tensors.items()})
    )
    compressed = zstandard.ZstdCompressor(level=compression_level, threads=1).compress(container)
    summary = dict(source.summary)
    summary.update(
        {
            "compressed_sha256": _sha256(compressed),
            "container_sha256": _sha256(container),
            "compressed_bytes": len(compressed),
            "container_bytes": len(container),
            "compression_ratio": round(len(container) / max(1, len(compressed)), 6),
        }
    )
    label_repair = manifest.get("label_repair")
    if isinstance(label_repair, Mapping):
        summary["label_repair"] = _summary_label_repair(label_repair)
    _atomic_write(summary_path.parent / str(summary["data_file"]), compressed)
    _atomic_write_json(summary_path, summary)
    return summary


# --------------------------------------------------------------------------
# Per-frame repair.


def _classify_alignment_error(error: ExpertActionAlignmentError) -> str:
    message = str(error)
    if "matched 0 legal candidates" in message:
        return "candidate_absent"
    if "matched" in message and "legal candidates" in message:
        return "candidate_ambiguous"
    if "unaffordable at its touch delay bin" in message:
        return "elixir_unaffordable_at_touch_bin"
    if "compound action is illegal" in message:
        # Unaffordable at every bin of the window, or a compound conflict.
        return "illegal_under_shadow_state"
    if "GRID target" in message or "placement anchor" in message:
        return "target_illegal"
    if "five-tick decision window" in message or "delay vocabulary" in message:
        return "window_mismatch"
    return "other_alignment_error"


def _frame_batch(shard: ILCacheShardV1, frame: int) -> UniversalSemanticBatchV4:
    return shard.observations.narrow_batch(frame, 1)


def _label_agreement(
    batch: UniversalSemanticBatchV4, actions: ActionSequenceV4, frame: int, window: Sequence[TimedExpertActionV4]
) -> tuple[bool, int, int]:
    """Return (label matches window, card plays checked, card slot mismatches)."""

    count = int(actions.micro_action_count[frame])
    if count != len(window):
        return False, 0, 0
    candidates = batch.candidates
    cards = slot_mismatches = 0
    for step, timed in enumerate(window):
        index = int(actions.candidate_index[frame, step])
        if not 0 <= index < candidates.mask.shape[1] or not bool(candidates.mask[0, index]):
            return False, 0, 0
        variant = int(candidates.variant[0, index])
        if timed.action.kind is ActionKind.PLAY_CARD:
            if variant != CANDIDATE_DEPLOY or int(candidates.native_visible_card_id[0, index]) != timed.action.card_id:
                return False, 0, 0
            cards += 1
            slot_mismatches += int(candidates.native_hand_slot[0, index]) != timed.action.hand_slot
        elif variant != CANDIDATE_ABILITY:
            return False, 0, 0
    return True, cards, slot_mismatches


def _card_costs(shard: ILCacheShardV1) -> dict[int, float]:
    """Cheapest observed DEPLOY cost per native card ID in this shard."""

    candidates = shard.observations.candidates
    deploy = candidates.mask & (candidates.variant == CANDIDATE_DEPLOY)
    result: dict[int, float] = {}
    for card_id, cost in zip(
        candidates.native_visible_card_id[deploy].tolist(), candidates.cost[deploy].tolist(), strict=True
    ):
        result[int(card_id)] = min(float(cost), result.get(int(card_id), float("inf")))
    return result


def _card_precheck(
    batch: UniversalSemanticBatchV4,
    window: Sequence[TimedExpertActionV4],
    *,
    decision_tick: int,
    card_costs: Mapping[int, float],
    config: ModelConfigV4,
) -> str | None:
    """Explain a card the stored candidate set cannot represent, else ``None``."""

    candidates = batch.candidates
    deploy = candidates.mask[0] & (candidates.variant[0] == CANDIDATE_DEPLOY)
    for timed in window:
        rows = (deploy & (candidates.native_visible_card_id[0] == timed.action.card_id)).nonzero().flatten()
        if rows.numel() == 0:
            # Not a legal candidate at d: the native mask omits a hand slot
            # it cannot afford at the decision tick (or the card is not in
            # hand); the row was never tensorized, so it cannot be restored.
            # Sub-classify with the card's cost seen elsewhere in the shard:
            # "affordable_at_touch" rows are what a delay-aware rebuild would
            # recover.
            cost = card_costs.get(int(timed.action.card_id or -1))
            elixir = float(candidates.elixir[0])
            if cost is None or cost <= elixir + 1e-6:
                return "candidate_absent_other"
            multiplier = float(elixir_multiplier_from_match_scalars(batch.match_scalars)[0])
            shadow = ShadowCandidateLegality(candidates, config, elixir_multiplier=multiplier, delay_aware=True)
            delay_ms = (timed.source_tick - decision_tick) * 50
            if delay_ms not in config.delay_offset_ms:
                return "candidate_absent_other"
            bin_index = config.delay_offset_ms.index(delay_ms)
            if cost <= float(shadow.elixir_at_bins()[0, bin_index]) + 1e-6:
                return "candidate_absent_affordable_at_touch"
            return "candidate_absent_unaffordable_at_touch"
        if all(int(candidates.native_hand_slot[0, row]) != timed.action.hand_slot for row in rows.tolist()):
            # The card sits in another slot than the one the engine executed:
            # the simulated hand diverged from the source.
            return "hand_slot_mismatch"
    return None


def validate_repaired_label(batch: UniversalSemanticBatchV4, sequence: ActionSequenceV4, config: ModelConfigV4) -> None:
    """Re-check one ACT label with the exact teacher-forcing legality rules.

    Mirrors ``UniversalCardPolicyV4._decode(validate=True)`` with forced
    actions: candidate legality, UID, GRID placement and delay-bin masks under
    delay-aware ``ShadowCandidateLegality``.
    """

    candidates = batch.candidates
    sequence.validate(config, candidate_count=int(candidates.mask.shape[1]))
    if int(sequence.gate[0]) != GATE_ACT:
        raise ValueError("repaired label is not ACT")
    shadow = ShadowCandidateLegality(
        candidates,
        config,
        elixir_multiplier=elixir_multiplier_from_match_scalars(batch.match_scalars),
        delay_aware=True,
    )
    active = torch.ones(1, dtype=torch.bool)
    for step in range(int(sequence.micro_action_count[0])):
        index = sequence.candidate_index[:, step]
        legal = shadow.candidate_mask()
        if not bool(legal[0, int(index[0])]):
            raise ValueError("stored candidate is illegal under shadow state")
        if int(candidates.uid[0, int(index[0])]) != int(sequence.candidate_uid[0, step]):
            raise ValueError("stored candidate UID disagrees with its row")
        target = sequence.target_cell[:, step]
        if int(candidates.target_mode[0, int(index[0])]) == TARGET_GRID:
            if int(target[0]) < 0 or not bool(shadow.selected_placement(index).flatten(1)[0, int(target[0])]):
                raise ValueError("stored target cell is illegal")
        elif int(target[0]) != -1:
            raise ValueError("non-GRID candidate must use target cell -1")
        delay = sequence.delay_offset_bin[:, step]
        if not bool(shadow.delay_offset_mask(step=step, candidate_index=index)[0, int(delay[0])]):
            raise ValueError("stored delay offsets must preserve action order")
        shadow.apply(active, index, target, delay, step=step)


def _write_previous_action(
    tensors: dict[str, Tensor], *, frame: int, source_frame: int, sequence: ActionSequenceV4
) -> None:
    """Rewrite ``frame``'s previous-action record from ``source_frame``'s label.

    Matches ``UniversalObservationTensorizerV4.record_action`` for DEPLOY
    micro-actions (abilities are never repaired, so source position stays 0).
    """

    count = int(sequence.micro_action_count[0])
    prefix = "observations.previous_action."
    candidate_prefix = "observations.candidates."
    tensors[prefix + "gate"][frame] = int(sequence.gate[0])
    tensors[prefix + "micro_action_count"][frame] = count
    for step in range(count):
        row = int(sequence.candidate_index[0, step])
        for name in ("variant", "visible_card_vocab_id", "effective_card_vocab_id", "effective_form", "ability_vocab_id"):
            tensors[prefix + name][frame, step] = tensors[candidate_prefix + name][source_frame, row]
        tensors[prefix + "target_cell"][frame, step] = int(sequence.target_cell[0, step])
        tensors[prefix + "delay_offset_bin"][frame, step] = int(sequence.delay_offset_bin[0, step])
        tensors[prefix + "action_mask"][frame, step] = True


def _previous_action_is_wait(tensors: Mapping[str, Tensor], frame: int) -> bool:
    prefix = "observations.previous_action."
    return (
        int(tensors[prefix + "gate"][frame]) == GATE_WAIT
        and int(tensors[prefix + "micro_action_count"][frame]) == 0
        and not bool(tensors[prefix + "action_mask"][frame].any())
    )


@dataclass(slots=True)
class ShardRepairV1:
    manifest: dict[str, Any]
    tensors: dict[str, Tensor]
    repaired_frames: tuple[int, ...]
    previous_action_frames: tuple[int, ...]
    attempted_frames: int
    reasons: Counter[str]
    source_errors: dict[str, str]


def repair_shard_tensors(
    raw: RawShardV1,
    *,
    expert_actions_for_tag: Callable[[str], Sequence[TimedExpertActionV4]],
    config: ModelConfigV4 | None = None,
) -> ShardRepairV1:
    """Compute repaired tensors for one shard (pure; writes nothing)."""

    actual = config or ModelConfigV4()
    # The loader view applies on-load observation migrations exactly as
    # training does; legality is decided on that view.  Writes go to clones
    # of the raw tensors only.
    shard = ILCacheShardV1(path=Path("."), manifest=raw.manifest, tensors=raw.tensors, summary=raw.summary, config=actual)
    tensors = dict(raw.tensors)
    for name in ("gate_loss_mask", *(f"actions.{field}" for field in ACTION_FIELDS)):
        tensors[name] = raw.tensors[name].clone()
    for field in PREVIOUS_ACTION_FIELDS:
        tensors[f"observations.previous_action.{field}"] = raw.tensors[f"observations.previous_action.{field}"].clone()
    # Labels are judged on the stored (raw) tensors; the loader may relabel
    # legacy shards in memory, which must never leak into the written copy.
    actions = ActionSequenceV4(**{field: raw.tensors[f"actions.{field}"] for field in ACTION_FIELDS})
    valid = raw.tensors["valid_mask"][:, 0]
    gate_loss = raw.tensors["gate_loss_mask"][:, 0]

    reasons: Counter[str] = Counter()
    repaired: list[int] = []
    previous_frames: list[int] = []
    attempted = 0
    source_errors: dict[str, str] = {}
    card_costs: dict[int, float] | None = None
    windows_by_tag: dict[str, Mapping[tuple[int, int], Sequence[TimedExpertActionV4]] | None] = {}
    for descriptor in shard.descriptors:
        start, length = descriptor.offset, descriptor.length
        dropped = [
            row
            for row in range(length)
            if bool(valid[start + row]) and not bool(gate_loss[start + row]) and int(actions.gate[start + row]) == GATE_WAIT
        ]
        if not dropped:
            continue
        attempted += len(dropped)
        if descriptor.replay_tag not in windows_by_tag:
            try:
                timed = tuple(
                    item
                    for item in expert_actions_for_tag(descriptor.replay_tag)
                    if item.source_tick >= FIRST_POLICY_DECISION_TICK
                )
                windows_by_tag[descriptor.replay_tag] = group_expert_action_windows(timed)
            except Exception as error:
                windows_by_tag[descriptor.replay_tag] = None
                source_errors[descriptor.replay_tag] = f"{type(error).__name__}: {error}"[:300]
        windows = windows_by_tag[descriptor.replay_tag]
        if windows is None:
            reasons["source_payload_unavailable"] += len(dropped)
            continue
        owner = int(descriptor.owner)

        def decision_tick(row: int) -> int:
            return FIRST_POLICY_DECISION_TICK + POLICY_DECISION_TICKS * row

        # Lane sanity: every stored ACT label must be its window's source
        # actions (else the frame->tick mapping or the source replay is not
        # the one this lane was produced from).  The producer aligned card
        # plays by their native-CALIBRATED hand slot; offline we only have the
        # uncalibrated slots, so repair only lanes where every supervised card
        # play's slot agrees with the stored candidate (calibration was the
        # identity there) and at least one card play proves it.
        consistent = True
        checked_cards = slot_mismatches = 0
        for row in range(length):
            frame = start + row
            if int(actions.gate[frame]) != GATE_ACT:
                continue
            window = windows.get((decision_tick(row), owner), ())
            matches, cards, mismatches = _label_agreement(_frame_batch(shard, frame), actions, frame, window)
            if not matches:
                consistent = False
                break
            checked_cards += cards
            slot_mismatches += mismatches
        if not consistent:
            reasons["lane_timeline_mismatch"] += len(dropped)
            continue
        if slot_mismatches:
            reasons["lane_hand_slots_uncalibrated"] += len(dropped)
            continue
        if not checked_cards:
            reasons["lane_hand_slots_unverified"] += len(dropped)
            continue

        perspective = PerspectiveTransformV1(actor_owner=owner)
        for row in dropped:
            frame = start + row
            window = windows.get((decision_tick(row), owner), ())
            if not window:
                reasons["no_source_action_in_window"] += 1
                continue
            if any(item.action.kind is not ActionKind.PLAY_CARD for item in window):
                reasons["ability_not_repairable"] += 1
                continue
            if row + 1 < length and not _previous_action_is_wait(raw.tensors, frame + 1):
                reasons["next_previous_action_not_wait"] += 1
                continue
            batch = _frame_batch(shard, frame)
            if card_costs is None:
                card_costs = _card_costs(shard)
            precheck = _card_precheck(
                batch, window, decision_tick=decision_tick(row), card_costs=card_costs, config=actual
            )
            if precheck is not None:
                reasons[precheck] += 1
                continue
            try:
                sequence = build_expert_action_batch(
                    batch,
                    (window,),
                    decision_ticks_by_row=(decision_tick(row),),
                    perspectives=(perspective,),
                    config=actual,
                    validate=True,
                    delay_aware=True,
                )
                validate_repaired_label(batch, sequence, actual)
            except ExpertActionAlignmentError as error:
                reasons[_classify_alignment_error(error)] += 1
                continue
            except ValueError:
                reasons["repaired_label_failed_validation"] += 1
                continue
            for field in ACTION_FIELDS:
                tensors[f"actions.{field}"][frame] = getattr(sequence, field)[0]
            tensors["gate_loss_mask"][frame, 0] = True
            if row + 1 < length:
                _write_previous_action(tensors, frame=frame + 1, source_frame=frame, sequence=sequence)
                previous_frames.append(frame + 1)
            repaired.append(frame)
            reasons["repaired"] += 1
    manifest = dict(raw.manifest)
    manifest["label_repair"] = {
        "schema": LABEL_REPAIR_SCHEMA,
        "delay_aware_legality": True,
        "repaired_frames": len(repaired),
        "attempted_frames": attempted,
        "per_reason_counts": dict(sorted(reasons.items())),
        "source_sha256": _tool_hashes(),
    }
    return ShardRepairV1(
        manifest=manifest,
        tensors=tensors,
        repaired_frames=tuple(repaired),
        previous_action_frames=tuple(previous_frames),
        attempted_frames=attempted,
        reasons=reasons,
        source_errors=source_errors,
    )


def _changed_rows(left: Tensor, right: Tensor) -> set[int]:
    if left.ndim == 0:
        return set() if torch.equal(left, right) else {-1}
    differs = (left != right).reshape(left.shape[0], -1).any(dim=1)
    return set(differs.nonzero(as_tuple=False).flatten().tolist())


def verify_repaired_shard(
    source: RawShardV1,
    output_summary: Path,
    *,
    repaired_frames: Sequence[int],
    previous_action_frames: Sequence[int],
    config: ModelConfigV4 | None = None,
) -> None:
    """Reload a written shard and prove it differs from its source only as intended."""

    actual = config or ModelConfigV4()
    written = read_raw_shard(output_summary)
    repaired = set(int(item) for item in repaired_frames)
    previous_rows = set(int(item) for item in previous_action_frames)
    if set(written.tensors) != set(source.tensors):
        raise AssertionError("repaired shard tensor names differ from the source")
    action_names = {"gate_loss_mask", *(f"actions.{field}" for field in ACTION_FIELDS)}
    previous_names = {f"observations.previous_action.{field}" for field in PREVIOUS_ACTION_FIELDS}
    for name, expected in source.tensors.items():
        value = written.tensors[name]
        if value.dtype != expected.dtype or value.shape != expected.shape:
            raise AssertionError(f"repaired shard tensor {name} changed dtype or shape")
        changed = _changed_rows(expected, value)
        allowed = repaired if name in action_names else previous_rows if name in previous_names else set()
        if not changed <= allowed:
            raise AssertionError(f"repaired shard tensor {name} changed outside repaired frames")
    manifest = dict(written.manifest)
    manifest.pop("label_repair", None)
    original = dict(source.manifest)
    original.pop("label_repair", None)
    if manifest != original:
        raise AssertionError("repaired shard manifest changed outside label_repair")
    shard = ILCacheShardV1(
        path=output_summary, manifest=written.manifest, tensors=written.tensors, summary=written.summary, config=actual
    )
    shard.actions.validate(actual, candidate_count=int(shard.observations.candidates.mask.shape[1]))
    for frame in sorted(repaired):
        if not bool(shard.gate_loss_mask[frame, 0]) or int(shard.actions.gate[frame]) != GATE_ACT:
            raise AssertionError(f"repaired frame {frame} is not a supervised ACT label")
        validate_repaired_label(_frame_batch(shard, frame), shard.actions.narrow_batch(frame, 1), actual)
    for frame in sorted(previous_rows):
        shard.observations.narrow_batch(frame, 1).previous_action.validate(
            actual, batch_size=1, card_vocab_size=1 << 31, ability_vocab_size=1 << 31
        )


# --------------------------------------------------------------------------
# Source payloads.


class ParquetPayloadSourceV1:
    """Random access to source payloads by replay tag via a row-group index."""

    def __init__(self, locations: Mapping[str, tuple[str, int, int]], *, cached_row_groups: int = 4) -> None:
        self.locations = locations
        self._cache: OrderedDict[tuple[str, int], tuple[list[Any], list[Any]]] = OrderedDict()
        self._limit = cached_row_groups

    @staticmethod
    def build_locations(roots: Sequence[Path], tags: set[str] | None = None) -> dict[str, tuple[str, int, int]]:
        import pyarrow.parquet as parquet

        result: dict[str, tuple[str, int, int]] = {}
        for root in roots:
            replay_root = Path(root) / "replays"
            files = sorted(replay_root.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no replay Parquet files under {replay_root}")
            for path in files:
                handle = parquet.ParquetFile(path)
                for group in range(handle.num_row_groups):
                    column = handle.read_row_group(group, columns=["replay_tag"]).column(0).to_pylist()
                    for row, tag in enumerate(column):
                        key = str(tag)
                        if tags is not None and key not in tags:
                            continue
                        if key in result:
                            raise ValueError(f"replay tag {key} occurs more than once in the source roots")
                        result[key] = (str(path), group, row)
        return result

    def __call__(self, tag: str) -> Mapping[str, Any]:
        import pyarrow.parquet as parquet

        location = self.locations.get(tag)
        if location is None:
            raise KeyError(f"replay tag {tag} is absent from the source roots")
        path, group, row = location
        key = (path, group)
        cached = self._cache.get(key)
        if cached is None:
            table = parquet.ParquetFile(path).read_row_group(group, columns=["replay_tag", "payload_json"])
            cached = (table.column("replay_tag").to_pylist(), table.column("payload_json").to_pylist())
            self._cache[key] = cached
            while len(self._cache) > self._limit:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        if str(cached[0][row]) != tag:
            raise ValueError(f"source row for {tag} moved")
        payload = json.loads(str(cached[1][row]))
        if not isinstance(payload, dict):
            raise ValueError(f"source payload for {tag} is not an object")
        return payload


# --------------------------------------------------------------------------
# Shard driver.


def repair_shard(
    summary_path: str | Path,
    output_summary_path: str | Path,
    *,
    payload_for_tag: PayloadForTagFn,
    expert_actions: ExpertActionsFn = payload_expert_actions,
    config: ModelConfigV4 | None = None,
    dry_run: bool = False,
    verify: bool = False,
    compression_level: int = -5,
) -> dict[str, Any]:
    """Repair one shard into ``output_summary_path`` and return its report.

    ``payload_for_tag`` and ``expert_actions`` are the pluggable source step
    (tests replace them with stubs).  An already-repaired shard whose pass
    repairs nothing is copied byte-for-byte, so reruns are idempotent.
    """

    source_path = Path(summary_path).resolve()
    output_path = Path(output_summary_path).resolve()
    if output_path.parent == source_path.parent:
        raise ValueError("refusing to write repaired shards into the source directory")
    raw = read_raw_shard(source_path)
    result = repair_shard_tensors(
        raw, expert_actions_for_tag=lambda tag: expert_actions(payload_for_tag(tag)), config=config
    )
    label_repair = result.manifest["label_repair"]
    report: dict[str, Any] = {
        "shard_id": raw.summary.get("shard_id"),
        "attempted_frames": result.attempted_frames,
        "repaired_frames": len(result.repaired_frames),
        "per_reason_counts": label_repair["per_reason_counts"],
        "source_errors": result.source_errors,
        "dry_run": dry_run,
    }
    if dry_run:
        report["summary"] = raw.summary
        return report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_name = str(raw.summary["data_file"])
    if not result.repaired_frames:
        # Nothing to rewrite: keep the (multi-GB) container byte-identical.
        # A first pass records its counts in the summary JSON only; a shard
        # that already carries a repair record is copied verbatim (idempotent).
        shutil.copyfile(source_path.parent / data_name, output_path.parent / data_name)
        if "label_repair" in raw.manifest or "label_repair" in raw.summary:
            shutil.copyfile(source_path, output_path)
            summary = dict(raw.summary)
        else:
            summary = dict(raw.summary)
            summary["label_repair"] = _summary_label_repair(result.manifest["label_repair"])
            _atomic_write_json(output_path, summary)
        report["copied_unchanged"] = True
    else:
        summary = write_raw_shard(
            output_path, source=raw, manifest=result.manifest, tensors=result.tensors, compression_level=compression_level
        )
        report["copied_unchanged"] = False
    if verify and report["copied_unchanged"]:
        if _sha256((output_path.parent / data_name).read_bytes()) != raw.summary["compressed_sha256"]:
            raise AssertionError("unchanged shard copy is not byte-identical")
        report["verified"] = True
    elif verify:
        verify_repaired_shard(
            raw,
            output_path,
            repaired_frames=result.repaired_frames,
            previous_action_frames=result.previous_action_frames,
            config=config,
        )
        report["verified"] = True
    report["summary"] = summary
    return report


_WORKER_SOURCE: ParquetPayloadSourceV1 | None = None


def _worker_initialize(locations: Mapping[str, tuple[str, int, int]]) -> None:
    global _WORKER_SOURCE
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_SOURCE = ParquetPayloadSourceV1(locations)


def _worker_repair(
    summary_path: str, output_summary_path: str, dry_run: bool, verify: bool, compression_level: int
) -> dict[str, Any]:
    if _WORKER_SOURCE is None:
        raise RuntimeError("repair worker has no payload source")
    started = time.perf_counter()
    report = repair_shard(
        summary_path,
        output_summary_path,
        payload_for_tag=_WORKER_SOURCE,
        dry_run=dry_run,
        verify=verify,
        compression_level=compression_level,
    )
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    return report


def _select_shards(index: Mapping[str, Any], patterns: Sequence[str]) -> list[dict[str, Any]]:
    shards = [dict(item) for item in index["shards"]]
    if not patterns:
        return shards
    return [
        item
        for item in shards
        if any(
            fnmatch.fnmatch(str(item.get("shard_id")), pattern)
            or fnmatch.fnmatch(str(item.get("summary_file")), pattern)
            for pattern in patterns
        )
    ]


def _summary_file(shard: Mapping[str, Any]) -> str:
    name = shard.get("summary_file")
    if isinstance(name, str) and Path(name).name == name:
        return name
    return str(Path(str(shard["data_file"])).with_suffix(".json"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--additional-source-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", action="append", default=[], help="glob(s) over shard_id/summary_file; comma lists ok")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compression-level", type=int, default=-5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0 or not -5 <= args.compression_level <= 22:
        raise ValueError("invalid --workers or --compression-level")
    patterns = [part for item in args.shards for part in item.split(",") if part]
    output_root = args.output_root.resolve()

    jobs: list[tuple[str, Path, Path, str]] = []
    plans: list[tuple[Path, str, dict[str, Any], list[dict[str, Any]], bool]] = []
    needed_tags: set[str] = set()
    for index_path in args.index:
        index_path = index_path.resolve()
        index = load_il_cache_index(index_path)
        node = index_path.parent.name
        output_dir = output_root / node
        if output_dir == index_path.parent:
            raise ValueError("--output-root must not contain the source cache directory")
        selected = _select_shards(index, patterns)
        plans.append((index_path, node, index, selected, len(selected) == len(index["shards"])))
        contract = index.get("contract", {})
        for item in contract.get("selected_replays", ()) if isinstance(contract, Mapping) else ():
            if isinstance(item, Mapping) and isinstance(item.get("replay_tag"), str):
                needed_tags.add(str(item["replay_tag"]))
        for shard in selected:
            name = _summary_file(shard)
            jobs.append((node, index_path.parent / name, output_dir / name, str(shard["shard_id"])))

    started = time.perf_counter()
    roots = [args.source_root, *args.additional_source_root]
    locations = ParquetPayloadSourceV1.build_locations(roots, needed_tags or None)
    print(
        json.dumps(
            {
                "event": "label-repair-start",
                "shards": len(jobs),
                "source_tags": len(locations),
                "missing_source_tags": len(needed_tags - set(locations)) if needed_tags else None,
                "index_elapsed_s": round(time.perf_counter() - started, 3),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    reports: dict[tuple[str, str], dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, initializer=_worker_initialize, initargs=(locations,)
    ) as executor:
        futures = {
            executor.submit(_worker_repair, str(source), str(output), args.dry_run, args.verify, args.compression_level): (
                node,
                shard_id,
            )
            for node, source, output, shard_id in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            node, shard_id = futures[future]
            report = future.result()
            reports[(node, shard_id)] = report
            totals.update(report["per_reason_counts"])
            totals["attempted_frames"] += int(report["attempted_frames"])
            print(
                json.dumps(
                    {
                        "event": "label-repair-shard",
                        "node": node,
                        "shard": shard_id,
                        "done": done,
                        "total": len(jobs),
                        "attempted_frames": report["attempted_frames"],
                        "repaired_frames": report["repaired_frames"],
                        "copied_unchanged": report.get("copied_unchanged"),
                        "source_errors": report.get("source_errors") or None,
                        "elapsed_s": report.get("elapsed_s"),
                        "wall_s": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for index_path, node, index, selected, complete in plans:
        node_reports = [reports[(node, str(shard["shard_id"]))] for shard in selected]
        node_reasons: Counter[str] = Counter()
        for report in node_reports:
            node_reasons.update(report["per_reason_counts"])
        written = False
        if complete and not args.dry_run:
            summaries = []
            for shard, report in zip(selected, node_reports, strict=True):
                summary = dict(shard)
                summary.update({key: report["summary"][key] for key in _SUMMARY_CONTAINER_FIELDS})
                if "label_repair" in report["summary"]:
                    summary["label_repair"] = report["summary"]["label_repair"]
                summaries.append(summary)
            contract = dict(index["contract"])
            contract["label_repair"] = {
                "schema": LABEL_REPAIR_SCHEMA,
                "note": "alignment-dropped expert labels re-aligned with delay-aware legality; "
                "train with decoding.DELAY_AWARE_LEGALITY_DEFAULT = True",
                "source_index": str(index_path),
                "source_index_sha256": index.get("index_sha256"),
                "repaired_frames": sum(int(item["repaired_frames"]) for item in node_reports),
                "attempted_frames": sum(int(item["attempted_frames"]) for item in node_reports),
                "per_reason_counts": dict(sorted(node_reasons.items())),
                "source_sha256": _tool_hashes(),
            }
            write_il_cache_index(output_root / node / "index.json", summaries, contract=contract)
            written = True
        print(
            json.dumps(
                {
                    "event": "label-repair-node",
                    "node": node,
                    "shards": len(selected),
                    "index_written": written,
                    "attempted_frames": sum(int(item["attempted_frames"]) for item in node_reports),
                    "repaired_frames": sum(int(item["repaired_frames"]) for item in node_reports),
                    "per_reason_counts": dict(sorted(node_reasons.items())),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "label-repair-done",
                "wall_s": round(time.perf_counter() - started, 3),
                "totals": dict(sorted(totals.items())),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
