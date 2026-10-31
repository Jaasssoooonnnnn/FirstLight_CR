"""Process-parallel resident self-play with centrally batched GPU inference."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait as wait_futures
import ctypes
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
import math
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Iterator, Mapping, Sequence, cast
import weakref

import torch
from torch import Tensor

from ...battle_env import TICKS_PER_SECOND, _elixir_generated_raw
from ...contracts import ActionKind, ActionV1
from ...match_factory import NATIVE_GAMEPLAY_END_TICK
from .cluster_self_play import (
    ClusterCollectionV4,
    ClusterMatchAssignmentV4,
    _model_key,
    _reset_engine,
    _scatter_storage_rows,
    _stored_tensor,
    _terminal_report,
    build_resident_engine_v4,
    concatenate_padded_tensor_records_v4,
)
from .decoding import ShadowCandidateLegality, decode_action_sequence_v4, elixir_multiplier_from_match_scalars
from .expert import POLICY_DECISION_TICKS
from .policy_session import _fixed_cuda_graph_batch_v4
from .matchmaking import POLICY_IL
from .king_activation_reward import (
    apply_fireball_king_activation_penalty_v4,
    apply_king_activation_reward_v4,
    configure_king_tower_shaping_v4,
    validate_king_reward_parameters_v4,
)
from .model import UniversalCardPolicyV4
from .ppo import StoredRolloutV4
from .ppo_runtime import ppo_environment_config_v4, validate_reward_parameters_v4
from .remote_worker_transport import RemoteWorkerConnection, ReceivedRowPackedTensorRecord
from .tensors import (
    ActionSequenceV4,
    ActiveEffectSetV4,
    GATE_ACT,
    GATE_WAIT,
    RecurrentPolicyStateV4,
    RelationEdgesV4,
    UniversalSemanticBatchV4,
    concatenate_tensor_records,
    tensor_padding_value,
)


_PPO_CUDA_GRAPH_ROW_BUCKET = 32
_INFERENCE_BATCH_WAIT_SECONDS = 0.002
# Policies whose rollout graphs replay concurrently need distinct capture
# streams (cuBLAS bakes the per-(handle, stream) workspace into each graph) and
# distinct CUDA RNG states (replay refills the state's seed/offset tensors).
# Policies beyond this many share a slot and therefore one replay stream.
_ROLLOUT_ISOLATION_SLOTS_V4 = 8


def _credit_hog_deployments_v4(
    events: Sequence[Any],
    *,
    owner: int,
    seen: set[tuple[int, str]],
    reward_per_deployment: float,
    accrued_reward: float,
    episode_cap: float,
    opening_reward_per_deployment: float | None = None,
) -> tuple[int, float, int | None]:
    """Credit attested plays once; optional absolute battle-time [10, 30)s rate."""

    if owner not in (0, 1) or any(
        not math.isfinite(value) or value < 0.0 for value in (reward_per_deployment, accrued_reward, episode_cap)
    ):
        raise ValueError("hog deployment reward inputs must be finite and nonnegative")
    if opening_reward_per_deployment is not None and (
        not math.isfinite(opening_reward_per_deployment) or opening_reward_per_deployment < 0.0
    ):
        raise ValueError("hog opening reward must be finite and nonnegative")
    ticks: list[int] = []
    for event in events:
        if (
            event.event_type != "action_executed"
            or event.owner != owner
            or event.card_id != 26000021
            or event.data.get("kind") != ActionKind.PLAY_CARD.value
        ):
            continue
        action_id = event.data.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        identity = (owner, action_id)
        if identity in seen:
            continue
        seen.add(identity)
        ticks.append(int(event.tick))
    opening_count = (
        sum(10 * TICKS_PER_SECOND <= tick < 30 * TICKS_PER_SECOND for tick in ticks)
        if opening_reward_per_deployment is not None
        else 0
    )
    # Use executed-event ticks, not observation time, segment-relative time,
    # or the displayed overtime clock. The window REPLACES the general rate.
    requested_bonus = (len(ticks) - opening_count) * reward_per_deployment
    if opening_reward_per_deployment is not None:
        requested_bonus += opening_count * opening_reward_per_deployment
    bonus = min(requested_bonus, max(0.0, episode_cap - accrued_reward))
    return len(ticks), bonus, min(ticks) if ticks else None


def _first_hog_timing_bonus_v4(
    *,
    first_hog_tick: int | None,
    first_hog_was_already_deployed: bool,
    maximum_bonus: float,
    start_seconds: float,
    deadline_seconds: float,
) -> float:
    """Return a one-time bonus that decays linearly from start to deadline."""

    values = (maximum_bonus, start_seconds, deadline_seconds)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("first-Hog timing reward inputs must be finite and nonnegative")
    if deadline_seconds <= start_seconds:
        raise ValueError("first-Hog deadline must be after its reward start")
    if first_hog_was_already_deployed or first_hog_tick is None:
        return 0.0
    start_tick = start_seconds * TICKS_PER_SECOND
    deadline_tick = deadline_seconds * TICKS_PER_SECOND
    if not start_tick <= first_hog_tick < deadline_tick:
        return 0.0
    return maximum_bonus * (deadline_tick - first_hog_tick) / (deadline_tick - start_tick)


def _first_hog_deadline_penalty_v4(
    *, before_tick: int, after_tick: int, first_hog_tick: int | None, deadline_seconds: float, penalty: float
) -> float:
    """Charge once on the transition crossing the deadline without a prior Hog."""

    if any(not math.isfinite(value) or value < 0.0 for value in (deadline_seconds, penalty)):
        raise ValueError("first-Hog deadline penalty inputs must be nonnegative")
    if after_tick < before_tick:
        raise ValueError("first-Hog deadline transition moved backwards")
    deadline_tick = deadline_seconds * TICKS_PER_SECOND
    missed = first_hog_tick is None or first_hog_tick >= deadline_tick
    return penalty if before_tick < deadline_tick <= after_tick and missed else 0.0


def _first_hog_deferral_penalty_v4(
    *,
    tick: int,
    first_hog_tick: int | None,
    hog_playable: bool,
    submitted_hog: bool,
    own_elixir: float,
    gate_wait: bool,
    penalty_per_decision: float,
    accrued_penalty: float,
    episode_cap: float,
    start_seconds: float,
    deadline_seconds: float,
) -> float:
    """Penalize declining a playable Hog or idling full while cycling to it."""

    values = (own_elixir, penalty_per_decision, accrued_penalty, episode_cap, start_seconds, deadline_seconds)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("first-Hog deferral inputs must be finite and nonnegative")
    if deadline_seconds <= start_seconds:
        raise ValueError("first-Hog deadline must be after its reward start")
    in_window = start_seconds * TICKS_PER_SECOND <= tick < deadline_seconds * TICKS_PER_SECOND
    if not in_window or first_hog_tick is not None or submitted_hog:
        return 0.0
    should_penalize = hog_playable or (not hog_playable and own_elixir >= 10.0 - 1e-6 and gate_wait)
    if not should_penalize:
        return 0.0
    return min(penalty_per_decision, max(0.0, episode_cap - accrued_penalty))


def _first_successful_card_tick_v4(events: Sequence[Any], *, owner: int, current: int | None) -> int | None:
    """Return the first attested own card execution without reclassifying abilities."""
    if owner not in (0, 1):
        raise ValueError("card telemetry owner must be 0 or 1")
    ticks = [
        int(event.tick)
        for event in events
        if event.event_type == "action_executed"
        and event.owner == owner
        and event.data.get("kind") == ActionKind.PLAY_CARD.value
    ]
    observed = min(ticks) if ticks else None
    if current is None:
        return observed
    return current if observed is None else min(current, observed)


def _newly_wasted_elixir_v4(*, before_elixir: float, after_elixir: float, generated_elixir: float) -> float:
    """Exact newly discarded elixir for one observed engine transition."""
    if any(not math.isfinite(value) or value < 0.0 for value in (before_elixir, after_elixir, generated_elixir)):
        raise ValueError("elixir telemetry inputs must be finite and nonnegative")
    if after_elixir < 10.0 - 1e-6:
        return 0.0
    return generated_elixir if before_elixir >= 10.0 - 1e-6 else max(0.0, before_elixir + generated_elixir - 10.0)


def _frozen_opponent_opening_force_owner_v4(
    *,
    current_owners: tuple[int, ...],
    active_actions: tuple[int, int],
    tick: int,
    after_seconds: int | None,
    policy_matchup: str | None = None,
    il_after_seconds: int | None = None,
) -> int | None:
    """Force a frozen opponent's first ACT, with an optional IL-only timeout."""

    if il_after_seconds is not None:
        if il_after_seconds <= 0:
            raise ValueError("IL opening timeout must be positive")
        if policy_matchup == POLICY_IL:
            after_seconds = il_after_seconds
    if after_seconds is None:
        return None
    if after_seconds <= 0 or tick < 0:
        raise ValueError("frozen-opponent opening timeout must be positive")
    if current_owners == (0, 1):
        return None
    if len(current_owners) != 1 or current_owners[0] not in (0, 1):
        raise ValueError("frozen-opponent opening force needs one current owner")
    opponent_owner = 1 - current_owners[0]
    if active_actions[opponent_owner] > 0 or tick < after_seconds * TICKS_PER_SECOND:
        return None
    return opponent_owner


def _deck_rule_opening_force_owner_v4(
    *,
    active_actions: tuple[int, int],
    deck_average_elixir: tuple[float, float],
    deck_ids: tuple[str, str],
    tick: int,
    force_tick: int | None,
    already_forced_owner: int | None = None,
) -> int | None:
    """Model-blind, role-blind opening force used by head-to-head evaluation.

    Opt-in only (``force_tick=None`` disables it). At or after ``force_tick``,
    if NEITHER owner has submitted a non-WAIT action, request one ACT from the
    owner holding the lower-average-elixir deck. Equal averages fall back to
    the lexicographically smaller deck id, then to owner 0 for a mirror, so the
    choice depends only on the decks. Because schedules keep decks with seats
    and swap the models, every model receives every forced deck equally.
    """

    if force_tick is None:
        return None
    if force_tick <= 0 or tick < 0:
        raise ValueError("deck-rule opening force tick must be positive")
    if len(active_actions) != 2 or len(deck_average_elixir) != 2 or len(deck_ids) != 2:
        raise ValueError("deck-rule opening force needs two owners")
    if any(not math.isfinite(value) or value <= 0.0 for value in deck_average_elixir):
        raise ValueError("deck-rule opening force needs positive deck average elixir")
    if already_forced_owner is not None:
        return None
    if tick < force_tick or active_actions[0] > 0 or active_actions[1] > 0:
        return None
    first, second = (float(value) for value in deck_average_elixir)
    if abs(first - second) > 1e-9:
        return 0 if first < second else 1
    if deck_ids[0] != deck_ids[1]:
        return 0 if deck_ids[0] < deck_ids[1] else 1
    return 0


def _scaled_terminal_rewards_v4(reward: Mapping[int, float], terminal: Any, scale: float) -> dict[int, float]:
    """Scale only the terminal outcome; leave potential shaping untouched."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("terminal reward scale must be finite and positive")
    return {
        owner: float(reward[owner]) + ((scale - 1.) * float(terminal.result_by_owner[owner])
                                     if terminal.ended else 0.)
        for owner in (0, 1)
    }


def _elixir_overflow_penalty_transition_v4(
    *,
    personal_wasted_elixir: float,
    personal_accrued_cost: float,
    unilateral_wasted_elixir: float,
    unilateral_accrued_cost: float,
    before_elixir: float,
    after_elixir: float,
    generated_elixir: float,
    both_full: bool,
    personal_coefficient: float,
    personal_grace_elixir: float,
    unilateral_coefficient: float,
    step_penalty_cap: float,
) -> tuple[float, float, float, float, float, float]:
    """Advance independent personal-overflow and unilateral-overflow clocks."""

    values = (
        personal_wasted_elixir,
        personal_accrued_cost,
        unilateral_wasted_elixir,
        unilateral_accrued_cost,
        before_elixir,
        after_elixir,
        generated_elixir,
        personal_coefficient,
        personal_grace_elixir,
        unilateral_coefficient,
        step_penalty_cap,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("elixir overflow penalty inputs must be finite and nonnegative")
    if step_penalty_cap <= 0.0:
        raise ValueError("elixir overflow step penalty cap must be positive")
    if after_elixir < 10.0 - 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    newly_wasted = _newly_wasted_elixir_v4(
        before_elixir=before_elixir, after_elixir=after_elixir, generated_elixir=generated_elixir
    )
    next_personal_wasted = personal_wasted_elixir + newly_wasted
    personal_target_cost = personal_coefficient * max(0.0, next_personal_wasted - personal_grace_elixir)
    personal_delta = max(0.0, personal_target_cost - personal_accrued_cost)
    if both_full:
        next_unilateral_wasted = 0.0
        unilateral_accrued_cost = 0.0
        unilateral_delta = 0.0
    else:
        next_unilateral_wasted = unilateral_wasted_elixir + newly_wasted
        unilateral_target_cost = unilateral_coefficient * next_unilateral_wasted * next_unilateral_wasted
        unilateral_delta = max(0.0, unilateral_target_cost - unilateral_accrued_cost)
    combined_delta = personal_delta + unilateral_delta
    scale = 1.0 if combined_delta <= step_penalty_cap else step_penalty_cap / combined_delta
    personal_penalty = personal_delta * scale
    unilateral_penalty = unilateral_delta * scale
    return (
        next_personal_wasted,
        personal_accrued_cost + personal_penalty,
        next_unilateral_wasted,
        unilateral_accrued_cost + unilateral_penalty,
        personal_penalty,
        unilateral_penalty,
    )


def _await_inference_batch_v4(futures, *, wait_seconds: float, gather_all: bool = False):
    """Coalesce already in-flight engine frames without dropping or advancing any lane."""
    ready, _ = wait_futures(tuple(futures), return_when=FIRST_COMPLETED)
    deadline = time.perf_counter() + wait_seconds
    while not (gather_all and len(ready) == len(futures)):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        extra, _ = wait_futures(tuple(future for future in futures if future not in ready),
                                timeout=remaining, return_when=FIRST_COMPLETED)
        if not extra:
            break
        ready.update(extra)
    return ready


def _ppo_cuda_graph_row_capacity_v4(rows: int) -> int:
    """Use powers of two for small policy fragments and 32-row tiles above."""

    if rows <= 0:
        raise ValueError("PPO CUDA Graph row count must be positive")
    if rows <= _PPO_CUDA_GRAPH_ROW_BUCKET:
        return 1 << (rows - 1).bit_length()
    return _PPO_CUDA_GRAPH_ROW_BUCKET * ((rows + _PPO_CUDA_GRAPH_ROW_BUCKET - 1) // _PPO_CUDA_GRAPH_ROW_BUCKET)


def _copy_tensor_record_(destination: object, source: object, path: str = "batch") -> None:
    if isinstance(destination, Tensor):
        if not isinstance(source, Tensor) or destination.shape != source.shape:
            source_shape = source.shape if isinstance(source, Tensor) else type(source)
            raise RuntimeError(f"async shared tensor shape changed at {path}: {destination.shape} != {source_shape}")
        destination.copy_(source)
        return
    if is_dataclass(destination):
        if type(source) is not type(destination):
            raise TypeError("async shared record type changed")
        for item in fields(destination):
            _copy_tensor_record_(getattr(destination, item.name), getattr(source, item.name), f"{path}.{item.name}")
        return
    if source != destination:
        raise ValueError("async shared non-tensor field changed")


class _RowPackedTensorRecordV4(ReceivedRowPackedTensorRecord):
    """Represent one nested batch with a small number of row-major buffers."""

    def __init__(
        self,
        template: object,
        *,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
        model_float_dtype: torch.dtype | None = None,
        copy_source: bool = True,
    ) -> None:
        source_tensors: list[tuple[torch.dtype, int, int, Tensor]] = []
        widths: dict[torch.dtype, int] = {}
        batch_size: int | None = None

        def describe(value: object) -> object:
            nonlocal batch_size
            if isinstance(value, Tensor):
                if value.ndim <= 0:
                    raise ValueError("row-packed tensor records require batch axes")
                rows = int(value.shape[0])
                if batch_size is None:
                    batch_size = rows
                elif rows != batch_size:
                    raise ValueError("row-packed tensor record batch size changed")
                if rows <= 0:
                    raise ValueError("row-packed tensor records cannot be empty")
                width = int(value.numel()) // rows
                offset = widths.get(value.dtype, 0)
                widths[value.dtype] = offset + width
                source_tensors.append((value.dtype, offset, width, value))
                return ("tensor", value.dtype, tuple(int(size) for size in value.shape[1:]), offset, width)
            if is_dataclass(value) and not isinstance(value, type):
                return (
                    "dataclass",
                    type(value),
                    tuple((item.name, describe(getattr(value, item.name))) for item in fields(value)),
                )
            if isinstance(value, tuple):
                return ("tuple", tuple(describe(row) for row in value))
            if isinstance(value, list):
                return ("list", tuple(describe(row) for row in value))
            if isinstance(value, dict):
                return ("dict", tuple((describe(key), describe(row)) for key, row in value.items()))
            return ("value", value)

        self.schema = describe(template)
        if batch_size is None:
            raise ValueError("row-packed tensor record contains no tensors")
        self.batch_size = batch_size
        self.source_dtypes = tuple(widths)
        actual_device = torch.device(device)
        self.buffer_dtypes = {
            dtype: (model_float_dtype if model_float_dtype is not None and dtype.is_floating_point else dtype)
            for dtype in self.source_dtypes
        }
        self.buffers = {
            dtype: torch.empty(
                (batch_size, widths[dtype]),
                dtype=self.buffer_dtypes[dtype],
                device=actual_device,
                pin_memory=pin_memory,
            )
            for dtype in self.source_dtypes
        }
        self.record = self._restore()
        if copy_source:
            for dtype, offset, width, source in source_tensors:
                self.buffers[dtype][:, offset : offset + width].copy_(source.reshape(batch_size, width))

    def empty_rows(self, rows: int) -> "_RowPackedTensorRecordV4":
        if rows <= 0:
            raise ValueError("row-packed storage rows must be positive")
        return self._from_buffers(
            self, {dtype: buffer.new_empty((rows, int(buffer.shape[1]))) for dtype, buffer in self.buffers.items()}
        )

    def index_copy_(self, indices: Tensor, source: "_RowPackedTensorRecordV4") -> "_RowPackedTensorRecordV4":
        if not self.compatible(source):
            raise ValueError("row-packed storage schema changed")
        for dtype, buffer in self.buffers.items():
            buffer.index_copy_(0, indices, source.buffers[dtype])
        return self


def _pad_packed_rollout_tail_v4(
    storage: _RowPackedTensorRecordV4,
    last_observation: UniversalSemanticBatchV4,
    valid_mask: Tensor,
    *,
    scratch_bytes: int = 32 * 1024 * 1024,
) -> bool:
    """Copy the same final lane rows into padding, using bounded dtype buffers.

    Return False without changing storage if an unexpected schema expansion
    requires the original per-field scatter. Valid rows are never modified.
    """
    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool or valid_mask.device.type != "cpu":
        raise ValueError("packed tail padding requires a CPU bool [T,B] mask")
    if scratch_bytes <= 0:
        raise ValueError("packed tail scratch budget must be positive")
    steps, lanes = valid_mask.shape
    if steps <= 0 or lanes <= 0 or storage.batch_size != steps * lanes:
        raise ValueError("packed tail storage dimensions differ from mask")
    missing = (~valid_mask).reshape(-1).nonzero(as_tuple=False).flatten()
    if not missing.numel():
        return True
    # Preserve tensor_padding_value rules for smaller final-row schemas. The
    # scratch record has the full trajectory schema; every lane is overwritten.
    scratch = storage.empty_rows(lanes)
    padded_last = _scatter_storage_rows(
        scratch.record, last_observation, torch.arange(lanes), total_rows=lanes, clone_previous=False
    )
    source = _RowPackedTensorRecordV4(padded_last)
    if not storage.compatible(source):
        return False
    for dtype, target in storage.buffers.items():
        last = source.buffers[dtype]
        rows_per_chunk = max(1, scratch_bytes // max(1, last.shape[1] * last.element_size()))
        for indices in missing.split(rows_per_chunk):
            target.index_copy_(0, indices, last.index_select(0, indices.remainder(lanes)))
    return True


def _concatenate_row_packed_records_v4(values: Sequence[Any]) -> _RowPackedTensorRecordV4 | None:
    normalized = tuple(values)
    if not normalized or any(not normalized[0].compatible(value) for value in normalized[1:]):
        return None
    return _RowPackedTensorRecordV4._from_buffers(
        normalized[0],
        {
            dtype: torch.cat(tuple(value.buffers[dtype] for value in normalized), dim=0)
            for dtype in normalized[0].source_dtypes
        },
    )


class _ModelPackedArenaV4:
    """Assemble received dtype buffers directly into reusable model partitions.

    No per-packet nested record is reconstructed and no combined batch is
    gathered a second time by model. Padding repeats row zero, exactly as the
    legacy graph path does. Callers must copy learner rows before arena reuse.
    """

    def __init__(self, *, pin_memory: bool = True) -> None:
        self.pin_memory = pin_memory
        self.partitions: dict[tuple[str, int], _RowPackedTensorRecordV4] = {}

    def assemble(
        self, values: Sequence[tuple[Any, int]], rows_by_model: Mapping[str, Sequence[int]],
    ) -> dict[str, _RowPackedTensorRecordV4] | None:
        if not values or any(not values[0][0].compatible(value) for value, _ in values[1:]):
            return None
        template = values[0][0]
        source_rows: list[tuple[Any, int]] = []
        for value, length in values:
            if length <= 0 or length > value.batch_size:
                raise ValueError("model partition length is outside the packet")
            source_rows.extend((value, row) for row in range(length))
        result: dict[str, _RowPackedTensorRecordV4] = {}
        for key, rows in rows_by_model.items():
            if not rows:
                continue
            capacity = _ppo_cuda_graph_row_capacity_v4(len(rows))
            cache_key = (key, capacity)
            packed = self.partitions.get(cache_key)
            if packed is None or not packed.compatible(template):
                packed = _RowPackedTensorRecordV4._from_buffers(template, {
                    dtype: torch.empty((capacity, buffer.shape[1]), dtype=buffer.dtype,
                                       pin_memory=self.pin_memory)
                    for dtype, buffer in template.buffers.items()
                })
                self.partitions[cache_key] = packed
            # Coalesce adjacent rows from the same packet before the dtype cats.
            spans: list[tuple[Any, int, int]] = []
            for index in rows:
                if index < 0 or index >= len(source_rows):
                    raise ValueError("model partition row is outside the received batch")
                value, row = source_rows[index]
                if spans and spans[-1][0] is value and spans[-1][1] + spans[-1][2] == row:
                    previous, start, count = spans[-1]
                    spans[-1] = (previous, start, count + 1)
                else:
                    spans.append((value, row, 1))
            for dtype, destination in packed.buffers.items():
                torch.cat(tuple(value.buffers[dtype].narrow(0, start, count)
                                for value, start, count in spans), dim=0,
                          out=destination[:len(rows)])
                if len(rows) < capacity:
                    destination[len(rows):].copy_(destination[:1].expand(capacity - len(rows), -1))
            result[key] = packed
        return result


class _PackedCudaGraphBatchV4:
    """Stage a nested CPU record with one pinned transfer per dtype."""

    def __init__(self, template: UniversalSemanticBatchV4, device: torch.device) -> None:
        self.host_packed = _RowPackedTensorRecordV4(template, device="cpu", pin_memory=True, copy_source=False)
        self.device_packed = _RowPackedTensorRecordV4(
            template, device=device, model_float_dtype=torch.float32, copy_source=False
        )
        self.host_batch = cast(UniversalSemanticBatchV4, self.host_packed.record)
        self.device_batch = cast(UniversalSemanticBatchV4, self.device_packed.record)

    def stage(self, source: UniversalSemanticBatchV4, packed_source: _RowPackedTensorRecordV4 | None = None) -> None:
        if packed_source is not None and self.host_packed.compatible(packed_source):
            # Model-partitioned arenas are already pinned and live until all
            # model streams finish. Avoid the second host-to-host staging copy.
            if all(buffer.is_pinned() for buffer in packed_source.buffers.values()):
                for dtype, buffer in packed_source.buffers.items():
                    self.device_packed.buffers[dtype].copy_(buffer, non_blocking=True)
                return
            for dtype, host_buffer in self.host_packed.buffers.items():
                host_buffer.copy_(packed_source.buffers[dtype])
        else:
            _copy_tensor_record_(self.host_batch, source, "cuda_graph.host_batch")
        for dtype, host_buffer in self.host_packed.buffers.items():
            self.device_packed.buffers[dtype].copy_(host_buffer, non_blocking=True)


def _empty_storage_rows(value: object, rows: int) -> object:
    """Create fixed-width storage that callers will completely populate."""

    if isinstance(value, Tensor):
        return value.new_empty((rows, *value.shape[1:]))
    if is_dataclass(value):
        return type(value)(
            **{item.name: _empty_storage_rows(getattr(value, item.name), rows) for item in fields(value)}
        )
    return value


def _pad_relation_edges(batch: UniversalSemanticBatchV4, capacity: int) -> UniversalSemanticBatchV4:
    current = int(batch.relation_edges.mask.shape[1])
    if capacity < current:
        raise ValueError("relation capacity is smaller than the current frame")
    if capacity == current:
        return batch

    def pad(tensor: Tensor) -> Tensor:
        output = tensor.new_zeros((int(tensor.shape[0]), capacity))
        output[:, :current] = tensor
        return output

    return replace(
        batch,
        relation_edges=RelationEdgesV4(
            source=pad(batch.relation_edges.source),
            target=pad(batch.relation_edges.target),
            relation_type=pad(batch.relation_edges.relation_type),
            mask=pad(batch.relation_edges.mask),
        ),
    )


def _pad_active_effects(batch: UniversalSemanticBatchV4, capacity: int) -> UniversalSemanticBatchV4:
    current = int(batch.active_effects.mask.shape[1])
    if capacity < current:
        raise ValueError("active-effect capacity is smaller than the current frame")
    if capacity == current:
        return batch

    def pad(tensor: Tensor, fill: int = 0) -> Tensor:
        output = tensor.new_full((int(tensor.shape[0]), capacity, *tensor.shape[2:]), fill)
        output[:, :current] = tensor
        return output

    return replace(
        batch,
        active_effects=ActiveEffectSetV4(
            effect_vocab_id=pad(batch.active_effects.effect_vocab_id),
            parent_type=pad(batch.active_effects.parent_type),
            parent_index=pad(batch.active_effects.parent_index, -1),
            source_owner_type=pad(batch.active_effects.source_owner_type),
            runtime_features=pad(batch.active_effects.runtime_features),
            mask=pad(batch.active_effects.mask),
        ),
    )


def _pad_batch_rows(value: object, capacity: int, fill: int = 0) -> object:
    if isinstance(value, Tensor):
        current = int(value.shape[0])
        if current > capacity:
            raise ValueError("async actor batch exceeds its resident capacity")
        if current == capacity:
            return value
        output = value.new_full((capacity, *value.shape[1:]), fill)
        output[:current] = value
        return output
    if is_dataclass(value):
        return type(value)(
            **{
                item.name: _pad_batch_rows(getattr(value, item.name), capacity, tensor_padding_value(value, item.name))
                for item in fields(value)
            }
        )
    return value


def _action_to_wire(action: ActionSequenceV4) -> tuple[object, ...]:
    if action.gate.shape != (1,):
        raise ValueError("async action wire requires exactly one actor")
    return (
        int(action.gate[0]),
        int(action.micro_action_count[0]),
        tuple(int(value) for value in action.candidate_index[0]),
        tuple(int(value) for value in action.candidate_uid[0]),
        tuple(int(value) for value in action.target_cell[0]),
        tuple(int(value) for value in action.delay_offset_bin[0]),
    )


def _pack_action_batch_to_cpu(actions: ActionSequenceV4, length: int) -> tuple[ActionSequenceV4, Tensor]:
    """Transfer one packed action matrix without Python scalar materialization."""

    batch_size = int(actions.gate.shape[0])
    micro_actions = int(actions.candidate_index.shape[1])
    if length <= 0 or length > batch_size:
        raise ValueError("packed action transfer length is outside the batch")
    selected = cast(ActionSequenceV4, actions.narrow_batch(0, length))
    with torch.inference_mode(False):
        packed = (
            torch.cat(
                (
                    selected.gate[:, None],
                    selected.micro_action_count[:, None],
                    selected.candidate_index,
                    selected.candidate_uid,
                    selected.target_cell,
                    selected.delay_offset_bin,
                ),
                dim=1,
            )
            .detach()
            .to(device="cpu")
        )
        if torch.is_inference(packed):
            packed = packed.clone()
    cursor = 2
    candidate_index = packed[:, cursor : cursor + micro_actions]
    cursor += micro_actions
    candidate_uid = packed[:, cursor : cursor + micro_actions]
    cursor += micro_actions
    target_cell = packed[:, cursor : cursor + micro_actions]
    cursor += micro_actions
    delay_offset_bin = packed[:, cursor : cursor + micro_actions]
    stored = ActionSequenceV4(
        gate=packed[:, 0],
        micro_action_count=packed[:, 1],
        candidate_index=candidate_index,
        candidate_uid=candidate_uid,
        target_cell=target_cell,
        delay_offset_bin=delay_offset_bin,
    )
    return stored, packed


def _pack_model_actions_to_cpu_v4(
    values: Mapping[str, tuple[ActionSequenceV4, int]],
) -> dict[str, tuple[ActionSequenceV4, Tensor]]:
    """One D2H action transfer for all policies, preserving each policy's row order."""
    matrices: list[Tensor] = []
    lengths: list[tuple[str, int]] = []
    for key, (actions, length) in values.items():
        if length <= 0 or length > actions.gate.shape[0]:
            raise ValueError("packed action transfer length is outside the batch")
        matrices.append(torch.cat((
            actions.gate[:length, None], actions.micro_action_count[:length, None],
            actions.candidate_index[:length], actions.candidate_uid[:length],
            actions.target_cell[:length], actions.delay_offset_bin[:length],
        ), dim=1))
        lengths.append((key, length))
    if not matrices:
        return {}
    with torch.inference_mode(False):
        combined = torch.cat(matrices, dim=0).detach().to(device="cpu")
        if torch.is_inference(combined):
            combined = combined.clone()
    result: dict[str, tuple[ActionSequenceV4, Tensor]] = {}
    offset = 0
    for key, length in lengths:
        packed = combined.narrow(0, offset, length)
        result[key] = (_action_batch_from_packed(packed), packed)
        offset += length
    return result


def _action_batch_from_packed(value: object) -> ActionSequenceV4:
    if not isinstance(value, Tensor):
        raise TypeError("async packed action payload changed type")
    if value.device.type != "cpu" or value.dtype != torch.long or value.ndim != 2:
        raise TypeError("async packed action tensor must be a CPU int64 matrix")
    width = int(value.shape[1])
    if width < 6 or (width - 2) % 4:
        raise ValueError("async packed action tensor width is invalid")
    micro_actions = (width - 2) // 4
    cursor = 2
    candidate_index = value[:, cursor : cursor + micro_actions]
    cursor += micro_actions
    candidate_uid = value[:, cursor : cursor + micro_actions]
    cursor += micro_actions
    target_cell = value[:, cursor : cursor + micro_actions]
    cursor += micro_actions
    delay_offset_bin = value[:, cursor : cursor + micro_actions]
    return ActionSequenceV4(
        gate=value[:, 0],
        micro_action_count=value[:, 1],
        candidate_index=candidate_index,
        candidate_uid=candidate_uid,
        target_cell=target_cell,
        delay_offset_bin=delay_offset_bin,
    )


def _actions_from_packed(actor_rows: object, value: object) -> dict[int, ActionSequenceV4]:
    if not isinstance(actor_rows, tuple):
        raise TypeError("async packed action actor rows changed type")
    batch = _action_batch_from_packed(value)
    if int(batch.gate.shape[0]) != len(actor_rows):
        raise ValueError("async packed action row count changed")
    normalized = tuple(int(row) for row in actor_rows)
    if len(normalized) != len(set(normalized)):
        raise ValueError("async packed action actor rows are not unique")
    return {
        local_actor: cast(ActionSequenceV4, batch.narrow_batch(row, 1)) for row, local_actor in enumerate(normalized)
    }


@dataclass(frozen=True, slots=True)
class AsyncResidentEngineSpecV4:
    engine_index: int
    cpu: int
    host: str = "127.0.0.1"
    base_port: int = 28_000
    lanes: int = 8
    timeout: float = 90.0
    capture_mode: str = "compact"
    worker_host: str | None = None
    worker_port: int | None = None
    worker_cpus: tuple[int, ...] | None = None
    worker_nice: int = 0
    worker_switch_interval_ms: float = 5.0
    diagnostic_timing: bool = False
    segment_decision_steps: int | None = None
    force_frozen_opponent_opening_after_seconds: int | None = None
    force_il_opponent_opening_after_seconds: int | None = None
    # Training deadline for both owners independently, in all matchup types.
    force_all_owners_opening_after_seconds: int | None = None
    # Opt-in evaluation knobs (defaults preserve PPO behaviour exactly).
    # Absolute tick of the symmetric deck-rule opening force; see
    # _deck_rule_opening_force_owner_v4. Exclusive with the two fields above.
    force_deck_rule_opening_tick: int | None = None
    # Override EnvironmentConfigV1.warmup_ticks (PPO default 130; IL uses 90).
    environment_warmup_ticks: int | None = None
    # Skip the per-actor Hog-playability ShadowCandidateLegality pass. Only
    # allowed when every Hog/first-Hog shaping coefficient is zero.
    skip_shaping_telemetry: bool = False
    # Only the PPO trainer opts into simulated online command latency.
    # Inference / h2h specs leave this None, preserving their own timing.
    training_command_delay_jitter_ticks: int | None = None


def _configure_ppo_training_command_delay_v4(engine: Any, jitter_ticks: int | None) -> None:
    if jitter_ticks is not None and (
        isinstance(jitter_ticks, bool) or not isinstance(jitter_ticks, int) or not 0 <= jitter_ticks <= 3
    ):
        raise ValueError("PPO command-delay jitter must be None or an integer in [0, 3]")
    if jitter_ticks is not None:
        for environment in engine.environments:
            environment.simulated_card_delay_jitter_ticks = jitter_ticks


@dataclass(frozen=True, slots=True)
class _ActorIdentityV4:
    engine_index: int
    local_actor: int
    model_key: str
    current: bool


def _inference_model_key(actor: _ActorIdentityV4) -> str:
    """Keep frozen opponents on their dedicated inference models."""

    return actor.model_key


def _match_shadow_legality_v4(batch: UniversalSemanticBatchV4, config: Any) -> ShadowCandidateLegality:
    """Shadow legality at the batch's real elixir regeneration rate (1x/2x/3x)."""

    return ShadowCandidateLegality(
        batch.candidates, config, elixir_multiplier=elixir_multiplier_from_match_scalars(batch.match_scalars)
    )


@dataclass(frozen=True, slots=True)
class _RolloutIsolationV4:
    """Per-policy resources that let its rollout graphs replay concurrently."""

    slot: int
    capture_stream: torch.cuda.Stream
    generator: torch.Generator


class _RolloutIsolationSlotV4:
    def __init__(self, capture_stream: torch.cuda.Stream) -> None:
        self.capture_stream = capture_stream
        self.owners: weakref.WeakSet[UniversalCardPolicyV4] = weakref.WeakSet()


_ROLLOUT_ISOLATION_LOCK_V4 = threading.Lock()
_ROLLOUT_ISOLATION_SLOT_TABLE_V4: dict[int, list[_RolloutIsolationSlotV4]] = {}
_ROLLOUT_ISOLATION_GENERATORS_CREATED_V4 = [0]


def _cuda_device_index_v4(device: torch.device) -> int:
    return int(device.index) if device.index is not None else int(torch.cuda.current_device())


def _rollout_isolation_v4(model: UniversalCardPolicyV4, device: torch.device) -> _RolloutIsolationV4:
    """Bind a policy to its own RNG state and a capture stream distinct from other live policies.

    Graphs captured on one stream share that stream's cuBLAS workspace, and graphs
    sampling from one generator share its seed/offset tensors; replaying such graphs
    on different streams at once races on both. The binding is permanent for the
    model object because both resources are baked into its captured graphs.
    """

    index = _cuda_device_index_v4(device)
    records = model.__dict__.setdefault("_ppo_rollout_cuda_isolation_v4", {})
    if not isinstance(records, dict):
        raise RuntimeError("PPO rollout isolation cache changed type")
    record = records.get(index)
    if record is not None:
        if not isinstance(record, _RolloutIsolationV4):
            raise RuntimeError("PPO rollout isolation entry changed type")
        return record
    with _ROLLOUT_ISOLATION_LOCK_V4:
        slots = _ROLLOUT_ISOLATION_SLOT_TABLE_V4.setdefault(index, [])
        free = [slot for slot, entry in enumerate(slots) if not len(entry.owners)]
        if free:
            slot = free[0]
        elif len(slots) < _ROLLOUT_ISOLATION_SLOTS_V4:
            taken = {entry.capture_stream.cuda_stream for entry in slots}
            if torch.cuda.graph.default_capture_stream is not None:
                taken.add(torch.cuda.graph.default_capture_stream.cuda_stream)
            # The high-priority pool is otherwise unused here; skip handles the
            # pool's round robin has already handed to another slot.
            for _ in range(64):
                stream = torch.cuda.Stream(device=torch.device("cuda", index), priority=-1)
                if stream.cuda_stream not in taken:
                    break
            else:
                raise RuntimeError("could not reserve a distinct rollout capture stream")
            slots.append(_RolloutIsolationSlotV4(stream))
            slot = len(slots) - 1
        else:
            slot = min(range(len(slots)), key=lambda item: len(slots[item].owners))
        slots[slot].owners.add(model)
        sequence = _ROLLOUT_ISOLATION_GENERATORS_CREATED_V4[0]
        _ROLLOUT_ISOLATION_GENERATORS_CREATED_V4[0] = sequence + 1
        base_seed = int(torch.cuda.default_generators[index].initial_seed())
        generator = torch.Generator(device=torch.device("cuda", index))
        generator.manual_seed((base_seed + 0x9E37_79B9_7F4A_7C15 * (sequence + 1)) % (1 << 63))
        record = _RolloutIsolationV4(slot=slot, capture_stream=slots[slot].capture_stream, generator=generator)
    records[index] = record
    return record


@contextmanager
def _rollout_rng_scope_v4(device: torch.device, generator: torch.Generator | None) -> Iterator[None]:
    """Route default-generator sampling (and graph RNG registration) to ``generator``."""

    if generator is None:
        yield
        return
    default = torch.cuda.default_generators[_cuda_device_index_v4(device)]
    saved = default.graphsafe_get_state()
    default.graphsafe_set_state(generator)
    try:
        yield
    finally:
        default.graphsafe_set_state(saved)


class _StochasticCudaGraphV4:
    """Fixed-shape PPO sampler whose replay advances CUDA RNG state."""

    def __init__(
        self,
        device: torch.device,
        *,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        effect_capacity: int = 512,
        relation_capacity: int = 2048,
        graph_pool: object | None = None,
        isolation: _RolloutIsolationV4 | None = None,
    ) -> None:
        if effect_capacity <= 0 or relation_capacity <= 0:
            raise ValueError("PPO CUDA Graph capacities must be positive")
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.effect_capacity = effect_capacity
        self.relation_capacity = relation_capacity
        self.graph_pool = graph_pool
        # None keeps the process default generator and torch's shared default
        # capture stream (single-policy collection, bitwise unchanged).
        self.isolation = isolation
        self.graph: torch.cuda.CUDAGraph | None = None
        self.batch: UniversalSemanticBatchV4 | None = None
        self.packed_batch: _PackedCudaGraphBatchV4 | None = None
        self.state: RecurrentPolicyStateV4 | None = None
        self.episode_start: Tensor | None = None
        self.output: Any | None = None

    def _sample(
        self,
        model: UniversalCardPolicyV4,
        batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: Tensor,
    ) -> Any:
        return model.sample_for_ppo_rollout(batch, state, episode_start=episode_start, validate=False)

    def _capture(
        self,
        model: UniversalCardPolicyV4,
        host_batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: Tensor,
        packed_host_batch: _RowPackedTensorRecordV4 | None,
    ) -> Any:
        fixed_batch = _fixed_cuda_graph_batch_v4(
            host_batch, effect_capacity=self.effect_capacity, relation_capacity=self.relation_capacity
        )
        self.packed_batch = _PackedCudaGraphBatchV4(fixed_batch, self.device)
        self.packed_batch.stage(fixed_batch, packed_host_batch)
        self.batch = self.packed_batch.device_batch
        self.state = RecurrentPolicyStateV4(hidden=state.hidden.detach().clone(), cell=state.cell.detach().clone())
        self.episode_start = episode_start.detach().clone()
        isolation = self.isolation
        with _rollout_rng_scope_v4(self.device, None if isolation is None else isolation.generator):
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warmup_stream):
                with torch.autocast(
                    device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
                ):
                    for _ in range(3):
                        self._sample(model, self.batch, self.state, self.episode_start)
            torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            # capture_begin registers the default generator's *current* state, so
            # an isolated graph's replays only ever touch its policy's RNG state.
            with torch.cuda.graph(
                self.graph,
                pool=self.graph_pool,
                stream=None if isolation is None else isolation.capture_stream,
                capture_error_mode="thread_local",
            ):
                with torch.autocast(
                    device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
                ):
                    self.output = self._sample(model, self.batch, self.state, self.episode_start)
            # capture_begin refilled the RNG state's seed/offset tensors on the
            # capture stream; order that before this stream's replay refill.
            capture_stream = torch.cuda.graph.default_capture_stream if isolation is None else isolation.capture_stream
            if capture_stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        if self.output is None:
            raise RuntimeError("PPO CUDA Graph capture produced no output")
        self.graph.replay()
        return self.output

    def run(
        self,
        model: UniversalCardPolicyV4,
        host_batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: Tensor,
        packed_host_batch: _RowPackedTensorRecordV4 | None = None,
    ) -> Any:
        if self.graph is None:
            return self._capture(model, host_batch, state, episode_start, packed_host_batch)
        if (
            self.batch is None
            or self.packed_batch is None
            or self.state is None
            or self.episode_start is None
            or self.output is None
        ):
            raise RuntimeError("PPO CUDA Graph storage is incomplete")
        fixed_batch = _fixed_cuda_graph_batch_v4(
            host_batch, effect_capacity=self.effect_capacity, relation_capacity=self.relation_capacity
        )
        self.packed_batch.stage(fixed_batch, packed_host_batch)
        _copy_tensor_record_(self.state, state, "cuda_graph.state")
        self.episode_start.copy_(episode_start)
        self.graph.replay()
        return self.output


class _IndexedStateCudaGraphV4(_StochasticCudaGraphV4):
    """Capture recurrent gather/scatter with inference instead of launching it per batch."""

    def __init__(self, device: torch.device, *, state_bank: RecurrentPolicyStateV4,
                 actor_capacity: int, capacity: int, **kwargs: Any) -> None:
        super().__init__(device, **kwargs)
        self.state_bank = state_bank
        self.actor_capacity = actor_capacity
        self.capacity = capacity
        with torch.cuda.device(device):
            self.host_control = torch.empty((capacity, 3), dtype=torch.long, pin_memory=True)
        self.device_control = torch.empty((capacity, 3), dtype=torch.long, device=device)

    def prepare(self, rows: Sequence[int], starts: Sequence[bool]) -> None:
        if not rows or len(rows) != len(starts) or len(rows) > self.capacity:
            raise ValueError("indexed graph control rows changed shape")
        if any(row < 0 or row >= self.actor_capacity for row in rows):
            raise ValueError("indexed graph actor row is outside its state bank")
        controls = [(row, row, int(start)) for row, start in zip(rows, starts, strict=True)]
        # Padding reads row zero as before, but writes to distinct scratch rows;
        # it must never overwrite a real actor's next state.
        controls += [(rows[0], self.actor_capacity + offset, 0)
                     for offset in range(self.capacity - len(rows))]
        self.host_control.copy_(torch.tensor(controls, dtype=torch.long))

    def _sample(self, model, batch, state, episode_start):
        reads = self.device_control[:, 0]
        writes = self.device_control[:, 1]
        selected = self.state_bank.index_select(reads)
        starts = self.device_control[:, 2] != 0
        output = super()._sample(model, batch, selected, starts)
        # Capture records these operations; warmup must not advance the bank.
        if torch.cuda.is_current_stream_capturing():
            self.state_bank.hidden.index_copy_(0, writes, output.next_state.hidden.to(self.state_bank.hidden.dtype))
            self.state_bank.cell.index_copy_(0, writes, output.next_state.cell.to(self.state_bank.cell.dtype))
        return output

    def run_indexed(self, model, host_batch, packed_host_batch):
        self.device_control.copy_(self.host_control, non_blocking=True)
        if self.graph is None:
            dummy_state = self.state_bank.narrow_batch(0, self.capacity)
            dummy_start = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
            return self._capture(model, host_batch, dummy_state, dummy_start, packed_host_batch)
        if self.packed_batch is None or self.output is None:
            raise RuntimeError("indexed graph storage is incomplete")
        fixed_batch = _fixed_cuda_graph_batch_v4(
            host_batch, effect_capacity=self.effect_capacity, relation_capacity=self.relation_capacity)
        self.packed_batch.stage(fixed_batch, packed_host_batch)
        self.graph.replay()
        return self.output


def _rollout_state_bank_v4(model, key: str, actor_capacity: int, device: torch.device) -> RecurrentPolicyStateV4:
    banks = model.__dict__.setdefault("_ppo_rollout_state_banks_v4", {})
    bank_key = (device.index, key, actor_capacity)
    bank = banks.get(bank_key)
    if bank is None:
        bank = model.initial_state(actor_capacity + _ppo_cuda_graph_row_capacity_v4(actor_capacity), device=device)
        banks[bank_key] = bank
    bank.hidden.zero_()
    bank.cell.zero_()
    return bank


def _shared_rollout_graph_pool_v4(model: UniversalCardPolicyV4, device: torch.device) -> object:
    """Share storage between mutually exclusive rollout graph variants."""

    pools = model.__dict__.setdefault("_ppo_rollout_cuda_graph_pool_v4", {})
    if not isinstance(pools, dict):
        raise RuntimeError("PPO rollout CUDA Graph pool cache changed type")
    key = device.index
    pool = pools.get(key)
    if pool is None:
        with torch.cuda.device(device):
            pool = torch.cuda.graph_pool_handle()
        pools[key] = pool
    return pool


def _persistent_rollout_graph_v4(
    model: UniversalCardPolicyV4,
    device: torch.device,
    capacity: int,
    *,
    effect_capacity: int = 512,
    relation_capacity: int = 2048,
    isolated: bool = False,
    state_bank: RecurrentPolicyStateV4 | None = None,
    actor_capacity: int = 0,
    state_bank_key: str = "",
) -> _StochasticCudaGraphV4 | None:
    """Reuse captured rollout graphs after in-place PPO parameter updates.

    ``isolated`` graphs may replay concurrently with other policies' isolated
    graphs on other streams; shared (default) graphs must not.
    """

    cache = model.__dict__.setdefault("_ppo_rollout_cuda_graph_cache_v4", {})
    if not isinstance(cache, dict):
        raise RuntimeError("PPO rollout CUDA Graph cache changed type")
    isolated = bool(isolated) and device.type == "cuda"
    key = (
        device.index,
        int(capacity),
        int(effect_capacity),
        int(relation_capacity),
        float(model.ppo_gate_temperature),
        float(model.ppo_action_temperature),
        float(model.ppo_continue_temperature),
    ) + ((True,) if isolated else ())
    # A captured graph bakes the model's delay policy in (head-to-head sets it
    # per model); the default ``sample`` keeps the historical key.
    delay_policy = getattr(model, "delay_policy", "sample")
    if delay_policy != "sample":
        key = key + (f"delay_policy={delay_policy}",)
    if state_bank is not None:
        key = key + ("indexed-state", state_bank_key, actor_capacity,
                     state_bank.hidden.data_ptr(), state_bank.cell.data_ptr())
    graph = cache.get(key)
    if graph is None:
        if bool(model.__dict__.get("_ppo_rollout_cuda_graph_cache_frozen_v4", False)):
            return None
        graph_type = _StochasticCudaGraphV4 if state_bank is None else _IndexedStateCudaGraphV4
        indexed_options = {} if state_bank is None else {
            "state_bank": state_bank, "actor_capacity": actor_capacity, "capacity": capacity}
        graph = graph_type(
            device,
            effect_capacity=effect_capacity,
            relation_capacity=relation_capacity,
            graph_pool=(_shared_rollout_graph_pool_v4(model, device) if device.type == "cuda" else None),
            isolation=_rollout_isolation_v4(model, device) if isolated else None,
            **indexed_options,
        )
        cache[key] = graph
    if not isinstance(graph, _StochasticCudaGraphV4):
        raise RuntimeError("PPO rollout CUDA Graph cache entry changed type")
    return graph


def _worker_actor_rows(
    spec: AsyncResidentEngineSpecV4, assignments: Sequence[ClusterMatchAssignmentV4]
) -> tuple[_ActorIdentityV4, ...]:
    result = []
    for match_index, assignment in enumerate(assignments):
        for owner in (0, 1):
            key = _model_key(assignment, owner)
            result.append(
                _ActorIdentityV4(
                    engine_index=spec.engine_index,
                    local_actor=match_index * 2 + owner,
                    model_key=key,
                    current=key == "current",
                )
            )
    return tuple(result)


def _engine_worker(
    connection: RemoteWorkerConnection,
    spec: AsyncResidentEngineSpecV4,
    assignments: tuple[ClusterMatchAssignmentV4, ...],
    gamma_per_decision: float,
    shaping_beta: float,
    mutual_elixir_overflow_penalty: float,
    mutual_elixir_overflow_grace: float,
    unilateral_elixir_overflow_penalty: float,
    elixir_overflow_step_penalty_cap: float,
    policy_state_id: str,
    validate_tensors: bool,
    max_decision_steps: int,
    hog_deploy_reward: float = 0.0,
    hog_deploy_reward_episode_cap: float = 0.1,
    hog_deploy_opening_reward: float | None = None,
    first_hog_timing_reward_max: float = 0.0,
    first_hog_timing_start_seconds: float = 10.0,
    first_hog_timing_deadline_seconds: float = 30.0,
    first_hog_missed_deadline_penalty: float = 0.0,
    first_hog_deferral_penalty: float = 0.0,
    first_hog_deferral_penalty_episode_cap: float = 0.0,
    fireball_king_activation_penalty: float = 0.0,
    king_activation_penalty: float = 0.0,
    king_activation_scope: str = "any",
    king_shaping_mode: str = "legacy",
    king_potential_weight: float = 3.0,
    terminal_reward_scale: float = 1.0,
) -> None:
    """Own one native endpoint and all stateful tensorizers on one CPU."""

    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM) != 0:
        raise OSError("failed to bind async worker lifetime to its parent")
    if os.getppid() == 1:
        raise RuntimeError("async worker parent exited during startup")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not 0 <= spec.worker_nice <= 19:
        raise ValueError("async worker nice must be between 0 and 19")
    if spec.worker_nice:
        os.nice(spec.worker_nice)
    if not 0.1 <= spec.worker_switch_interval_ms <= 1_000.0:
        raise ValueError("async worker switch interval must be between 0.1 and 1000 ms")
    sys.setswitchinterval(spec.worker_switch_interval_ms / 1_000.0)
    worker_cpus = spec.worker_cpus or (int(spec.cpu),)
    if not worker_cpus or int(spec.cpu) not in worker_cpus or len(worker_cpus) != len(set(worker_cpus)):
        raise ValueError("async worker CPU group is invalid")
    os.sched_setaffinity(0, set(worker_cpus))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    environment_config = ppo_environment_config_v4(gamma_per_decision=gamma_per_decision, shaping_beta=shaping_beta)
    if spec.environment_warmup_ticks is not None:
        if spec.environment_warmup_ticks <= 0:
            raise ValueError("environment warmup override must be positive")
        environment_config = replace(environment_config, warmup_ticks=int(spec.environment_warmup_ticks))
    if spec.force_deck_rule_opening_tick is not None:
        if spec.force_deck_rule_opening_tick <= 0:
            raise ValueError("deck-rule opening force tick must be positive")
        if (
            spec.force_frozen_opponent_opening_after_seconds is not None
            or spec.force_il_opponent_opening_after_seconds is not None
        ):
            raise ValueError("deck-rule opening force cannot be combined with role-based opening force")
    if spec.force_all_owners_opening_after_seconds is not None:
        if spec.force_all_owners_opening_after_seconds <= 0:
            raise ValueError("all-owner opening timeout must be positive")
        if any(value is not None for value in (
            spec.force_deck_rule_opening_tick,
            spec.force_frozen_opponent_opening_after_seconds,
            spec.force_il_opponent_opening_after_seconds,
        )):
            raise ValueError("all-owner training opening cannot be combined with legacy opening modes")
    if spec.skip_shaping_telemetry and (
        hog_deploy_reward
        or hog_deploy_opening_reward is not None
        or first_hog_timing_reward_max
        or first_hog_missed_deadline_penalty
        or first_hog_deferral_penalty
        or first_hog_deferral_penalty_episode_cap
    ):
        raise ValueError("shaping telemetry can only be skipped when Hog shaping is disabled")
    engine = build_resident_engine_v4(
        spec.engine_index,
        environment_config=environment_config,
        host=spec.host,
        base_port=spec.base_port,
        lanes=spec.lanes,
        timeout=spec.timeout,
        capture_mode=spec.capture_mode,
    )
    _configure_ppo_training_command_delay_v4(engine, spec.training_command_delay_jitter_ticks)
    validate_king_reward_parameters_v4(
        king_activation_penalty=king_activation_penalty,
        king_activation_scope=king_activation_scope,
        king_shaping_mode=king_shaping_mode,
        king_potential_weight=king_potential_weight,
    )
    configure_king_tower_shaping_v4(
        engine, king_shaping_mode=king_shaping_mode, king_potential_weight=king_potential_weight
    )
    executor = ThreadPoolExecutor(max_workers=spec.lanes, thread_name_prefix=f"ppo-native-{spec.engine_index}")
    started = time.perf_counter()
    segment_decision_steps = spec.segment_decision_steps
    if segment_decision_steps is None or segment_decision_steps <= 0:
        raise ValueError("async segment decision steps must be positive")
    if (
        spec.force_frozen_opponent_opening_after_seconds is not None
        and spec.force_frozen_opponent_opening_after_seconds <= 0
    ):
        raise ValueError("frozen-opponent opening timeout must be positive")
    if spec.force_il_opponent_opening_after_seconds is not None and spec.force_il_opponent_opening_after_seconds <= 0:
        raise ValueError("IL opening timeout must be positive")
    validate_reward_parameters_v4(locals())
    segment_end_decision_steps = segment_decision_steps
    tensorize_seconds = 0.0
    native_seconds = 0.0
    frame_send_seconds = 0.0
    action_wait_seconds = 0.0
    action_decode_seconds = 0.0
    last_action_decode_seconds = 0.0
    last_native_step_seconds = 0.0
    last_advance_seconds = 0.0
    last_native_channel_timing: dict[str, float] = {}
    last_batches: dict[int, UniversalSemanticBatchV4] = {}
    current_actor_ids = frozenset(
        match_index * 2 + int(owner)
        for match_index, assignment in enumerate(assignments)
        for owner in assignment.matchup.current_owners
    )
    if not current_actor_ids:
        raise RuntimeError("async worker has no current-policy actors")
    expected_actions: tuple[int, ...] = ()
    shared_batch: UniversalSemanticBatchV4 | None = None
    shared_packed_batch: _RowPackedTensorRecordV4 | None = None
    relation_capacity = 0
    effect_capacity = 0
    max_relation_count = 0
    max_effect_count = 0

    def step_match(index: int, decoded: Mapping[int, tuple[Any, ...]]) -> Any:
        return runtimes[index].environment.step(decoded, record_trace=False, policy_features_only=True)

    def frame_packet(
        rewards: tuple[tuple[int, float, bool, bool], ...], *, include_segment_boundary: bool = False
    ) -> dict[str, object] | None:
        nonlocal tensorize_seconds, expected_actions, shared_batch
        nonlocal shared_packed_batch
        nonlocal relation_capacity, effect_capacity
        nonlocal max_relation_count, max_effect_count
        nonlocal last_action_decode_seconds
        nonlocal last_native_step_seconds, last_advance_seconds
        nonlocal last_native_channel_timing
        tensor_started = time.perf_counter()
        current_actor_rows: list[int] = []
        current_batches: list[UniversalSemanticBatchV4] = []
        opponent_actor_rows: list[int] = []
        opponent_batches: list[UniversalSemanticBatchV4] = []
        force_act_actor_rows: list[int] = []
        for match_index, runtime in enumerate(runtimes):
            if (
                not runtime.active
                or int(runtime.observations[0].tick) >= NATIVE_GAMEPLAY_END_TICK
                or (
                    segment_end_decision_steps is not None
                    and runtime.decision_steps >= segment_end_decision_steps
                    and not include_segment_boundary
                )
            ):
                continue
            force_owner = (
                None
                if include_segment_boundary
                else _frozen_opponent_opening_force_owner_v4(
                    current_owners=runtime.assignment.matchup.current_owners,
                    active_actions=runtime.active_actions,
                    tick=int(runtime.observations[0].tick),
                    after_seconds=(spec.force_frozen_opponent_opening_after_seconds),
                    policy_matchup=runtime.assignment.matchup.policy_matchup,
                    il_after_seconds=(spec.force_il_opponent_opening_after_seconds),
                )
            )
            if force_owner is None and not include_segment_boundary:
                force_owner = _deck_rule_opening_force_owner_v4(
                    active_actions=runtime.active_actions,
                    deck_average_elixir=runtime.deck_average_elixir,
                    deck_ids=(runtime.assignment.matchup.deck0.deck_id, runtime.assignment.matchup.deck1.deck_id),
                    tick=int(runtime.observations[0].tick),
                    force_tick=spec.force_deck_rule_opening_tick,
                    already_forced_owner=runtime.opening_forced_owner,
                )
            force_owners = set() if force_owner is None else {force_owner}
            if spec.force_all_owners_opening_after_seconds is not None and not include_segment_boundary:
                deadline = spec.force_all_owners_opening_after_seconds * TICKS_PER_SECOND
                if int(runtime.observations[0].tick) >= deadline:
                    force_owners.update(
                        owner for owner in (0, 1)
                        if runtime.active_actions[owner] == 0
                        and runtime.opening_forced_ticks_by_owner[owner] is None
                    )
            for owner in (0, 1):
                local_actor = match_index * 2 + owner
                batch = runtime.tensorizers[owner].tensorize(runtime.observations[owner], validate=validate_tensors)
                last_batches[local_actor] = batch
                if local_actor in current_actor_ids:
                    current_actor_rows.append(local_actor)
                    current_batches.append(batch)
                else:
                    opponent_actor_rows.append(local_actor)
                    opponent_batches.append(batch)
                if owner in force_owners:
                    force_act_actor_rows.append(local_actor)
        actor_rows = [*current_actor_rows, *opponent_actor_rows]
        batches = [*current_batches, *opponent_batches]
        expected_actions = tuple(actor_rows)
        if not batches:
            tensorize_seconds += time.perf_counter() - tensor_started
            return None
        combined = cast(UniversalSemanticBatchV4, concatenate_padded_tensor_records_v4(tuple(batches)))
        combined = cast(UniversalSemanticBatchV4, _pad_batch_rows(combined, spec.lanes * 2))
        relation_count = int(combined.relation_edges.mask.shape[1])
        effect_count = int(combined.active_effects.mask.shape[1])
        max_relation_count = max(max_relation_count, relation_count)
        max_effect_count = max(max_effect_count, effect_count)
        replacement_batch = None
        needs_replacement = shared_batch is None or relation_count > relation_capacity or effect_count > effect_capacity
        if needs_replacement:
            relation_capacity = max(160, 1 << (relation_count - 1).bit_length())
            effect_capacity = max(64, 1 << (effect_count - 1).bit_length())
        combined = _pad_relation_edges(combined, relation_capacity)
        combined = _pad_active_effects(combined, effect_capacity)
        combined = cast(UniversalSemanticBatchV4, combined.to_storage("cpu", float_dtype=torch.float16))
        if needs_replacement:
            shared_packed_batch = _RowPackedTensorRecordV4(combined)
            shared_batch = cast(UniversalSemanticBatchV4, shared_packed_batch.record)
            replacement_batch = shared_batch
        else:
            assert shared_batch is not None and shared_packed_batch is not None
            _copy_tensor_record_(shared_batch, combined)
        packet: dict[str, object] = {
            "kind": "frame",
            "engine_index": spec.engine_index,
            "actor_rows": expected_actions,
            "storage_actor_rows": tuple(current_actor_rows),
            "force_act_actor_rows": tuple(force_act_actor_rows),
            "rewards": rewards,
            "remote_storage_from_batch": True,
        }
        if replacement_batch is not None:
            packet["batch"] = replacement_batch
        if shared_packed_batch is None:
            raise RuntimeError("remote PPO frame has no row-packed batch")
        packet["row_packed_batch"] = shared_packed_batch
        tensor_duration = time.perf_counter() - tensor_started
        tensorize_seconds += tensor_duration
        if spec.diagnostic_timing:
            packet["worker_round_timing"] = {
                "action_decode_seconds": last_action_decode_seconds,
                "native_step_seconds": last_native_step_seconds,
                "advance_seconds": last_advance_seconds,
                "tensorize_seconds": tensor_duration,
            }
            packet["native_channel_timing"] = last_native_channel_timing
        return packet

    def advance(
        action_rows: Mapping[int, ActionSequenceV4] | None, forced_actor_rows: frozenset[int] = frozenset()
    ) -> tuple[tuple[int, float, bool, bool], ...]:
        nonlocal native_seconds, last_native_step_seconds, last_advance_seconds
        nonlocal last_native_channel_timing
        advance_started = time.perf_counter()
        policy_active = {
            index: (
                runtime.active
                and int(runtime.observations[0].tick) < NATIVE_GAMEPLAY_END_TICK
                and (segment_end_decision_steps is None or runtime.decision_steps < segment_end_decision_steps)
            )
            for index, runtime in enumerate(runtimes)
        }
        decoded_by_match: dict[int, dict[int, tuple[Any, ...]]] = {
            index: {} for index, runtime in enumerate(runtimes) if runtime.active
        }
        submitted_by_match: dict[int, dict[int, set[str]]] = {
            index: {} for index, runtime in enumerate(runtimes) if runtime.active
        }
        first_hog_step_penalty_by_match: dict[int, list[float]] = {
            index: [0.0, 0.0] for index, runtime in enumerate(runtimes) if runtime.active
        }
        if action_rows is not None:
            if tuple(sorted(action_rows)) != tuple(sorted(expected_actions)):
                raise RuntimeError("async engine received an incomplete action batch")
            if not forced_actor_rows.issubset(action_rows):
                raise RuntimeError("forced ACT command rows are outside the batch")
            for local_actor in expected_actions:
                match_index, owner = divmod(local_actor, 2)
                runtime = runtimes[match_index]
                action = action_rows[local_actor]
                batch = last_batches[local_actor]
                tensorizer = runtime.tensorizers[owner]
                tensorizer.record_action(action, batch, row=0, validate=False)
                try:
                    decoded = decode_action_sequence_v4(
                        action,
                        batch.candidates,
                        row=0,
                        observation=runtime.observations[owner],
                        catalog=tensorizer.catalog,
                        deck=tensorizer.deck,
                        card_costs=tensorizer.card_costs,
                        ability_id_by_vocab_id=(tensorizer.ability_id_by_vocab_id),
                        horizontal_mirror=(tensorizer.perspective.horizontal_mirror),
                        hand_slot_permutation=(tensorizer.perspective.hand_slot_permutation),
                        config=tensorizer.config,
                        base_latency_ticks=1,
                        validate=False,
                    )
                except ValueError as error:
                    if "illegal under V4 shadow state" not in str(error):
                        raise
                    candidates = batch.candidates
                    raise ValueError(
                        f"{error}; engine={spec.engine_index}; "
                        f"local_actor={local_actor}; tick="
                        f"{runtime.observations[owner].tick}; "
                        f"action={_action_to_wire(action)!r}; "
                        f"candidate_mask={candidates.mask[0].tolist()!r}; "
                        f"candidate_uid={candidates.uid[0].tolist()!r}; "
                        f"candidate_variant={candidates.variant[0].tolist()!r}; "
                        f"candidate_cost={candidates.cost[0].tolist()!r}; "
                        f"candidate_own_row="
                        f"{candidates.own_card_row[0].tolist()!r}; "
                        f"candidate_exclusion="
                        f"{candidates.exclusion_group_id[0].tolist()!r}; "
                        f"candidate_target_mode="
                        f"{candidates.target_mode[0].tolist()!r}; "
                        f"candidate_visible_id="
                        f"{candidates.native_visible_card_id[0].tolist()!r}; "
                        f"elixir={float(candidates.elixir[0])!r}"
                    ) from error
                identified = tuple(
                    replace(
                        item,
                        action_id=(
                            item.action_id
                            or (
                                f"ppo-{policy_state_id[:8]}-"
                                f"{runtime.assignment.matchup.sequence}-"
                                f"{owner}-{runtime.observations[0].tick}-{micro}"
                            )
                        ),
                    )
                    for micro, item in enumerate(decoded.actions)
                )
                decoded_by_match[match_index][owner] = identified
                submitted_by_match[match_index][owner] = {item.action_id for item in identified}
                action_tick = int(runtime.observations[0].tick)
                nonwait_submitted = any(item.kind != ActionKind.WAIT for item in identified)
                first_nonwait = list(runtime.first_nonwait_action_tick)
                if nonwait_submitted and first_nonwait[owner] is None:
                    first_nonwait[owner] = action_tick
                runtime.first_nonwait_action_tick = (first_nonwait[0], first_nonwait[1])
                current_first_card = runtime.first_successful_card_tick[owner]
                own_elixir = float(batch.candidates.elixir[0])
                submitted_card = any(item.kind == ActionKind.PLAY_CARD for item in identified)
                if current_first_card is None and own_elixir >= 10.0 - 1e-6 and not submitted_card:
                    deferred = list(runtime.full_elixir_deferred_decision_ticks_before_first_card)
                    deferred[owner] += 1
                    runtime.full_elixir_deferred_decision_ticks_before_first_card = (deferred[0], deferred[1])
                if spec.skip_shaping_telemetry:
                    hog_playable = False
                else:
                    shadow = _match_shadow_legality_v4(batch, tensorizer.config)
                    playable = shadow.candidate_mask()[0]
                    hog_playable = bool(
                        (playable & (batch.candidates.native_visible_card_id[0] == 26000021)).any().item()
                    )
                submitted_hog = any(
                    item.kind == ActionKind.PLAY_CARD and item.card_id == 26000021 for item in identified
                )
                if owner in runtime.assignment.matchup.current_owners:
                    deferral_totals = list(runtime.first_hog_deferral_penalty_total)
                    deferral_penalty = _first_hog_deferral_penalty_v4(
                        tick=action_tick,
                        first_hog_tick=runtime.first_hog_deploy_tick[owner],
                        hog_playable=hog_playable,
                        submitted_hog=submitted_hog,
                        own_elixir=own_elixir,
                        gate_wait=int(action.gate[0]) == GATE_WAIT,
                        penalty_per_decision=first_hog_deferral_penalty,
                        accrued_penalty=deferral_totals[owner],
                        episode_cap=first_hog_deferral_penalty_episode_cap,
                        start_seconds=first_hog_timing_start_seconds,
                        deadline_seconds=first_hog_timing_deadline_seconds,
                    )
                    deferral_totals[owner] += deferral_penalty
                    runtime.first_hog_deferral_penalty_total = (deferral_totals[0], deferral_totals[1])
                    first_hog_step_penalty_by_match[match_index][owner] = deferral_penalty
                if hog_playable and runtime.first_hog_deploy_tick[owner] is None:
                    first_playable = list(runtime.first_hog_playable_tick)
                    if first_playable[owner] is None:
                        first_playable[owner] = action_tick
                    runtime.first_hog_playable_tick = (first_playable[0], first_playable[1])
                    opportunities = list(runtime.hog_playable_decision_ticks_before_first_hog)
                    opportunities[owner] += 1
                    runtime.hog_playable_decision_ticks_before_first_hog = (opportunities[0], opportunities[1])
                    if not submitted_hog:
                        deferred_hog = list(runtime.hog_playable_deferred_ticks_before_first_hog)
                        deferred_hog[owner] += 1
                        runtime.hog_playable_deferred_ticks_before_first_hog = (deferred_hog[0], deferred_hog[1])
                        if action_tick < 30 * TICKS_PER_SECOND:
                            opening_deferred = list(runtime.opening_hog_playable_deferred_ticks)
                            opening_deferred[owner] += 1
                            runtime.opening_hog_playable_deferred_ticks = (opening_deferred[0], opening_deferred[1])
                counts = list(runtime.active_actions)
                counts[owner] += sum(item.kind != ActionKind.WAIT for item in identified)
                runtime.active_actions = (counts[0], counts[1])
                if int(action.gate[0]) == GATE_ACT:
                    act_steps = list(runtime.act_steps)
                    act_steps[owner] += 1
                    runtime.act_steps = (act_steps[0], act_steps[1])
                    if int(action.micro_action_count[0]) == 2:
                        second_steps = list(runtime.second_action_steps)
                        second_steps[owner] += 1
                        runtime.second_action_steps = (second_steps[0], second_steps[1])
                if local_actor in forced_actor_rows:
                    if int(action.gate[0]) != GATE_ACT or not identified:
                        raise RuntimeError("opening force did not produce ACT")
                    forced_ticks = list(runtime.opening_forced_ticks_by_owner)
                    if forced_ticks[owner] is not None:
                        raise RuntimeError("opening ACT was forced more than once for the same owner")
                    forced_ticks[owner] = int(runtime.observations[0].tick)
                    runtime.opening_forced_ticks_by_owner = (forced_ticks[0], forced_ticks[1])
                    if runtime.opening_forced_owner is None:
                        runtime.opening_forced_owner = owner
                        runtime.opening_forced_tick = forced_ticks[owner]

        for match_index, runtime in enumerate(runtimes):
            if not runtime.active or policy_active[match_index]:
                continue
            decoded_by_match[match_index] = {
                owner: (ActionV1.wait(owner, ticks=POLICY_DECISION_TICKS, metadata={"synthetic_policy_sleep": True}),)
                for owner in (0, 1)
            }
            submitted_by_match[match_index] = {0: set(), 1: set()}

        active_ids = tuple(runtime.env_id for runtime in runtimes if runtime.active)
        if not active_ids:
            return ()
        channel_profile_before = engine.coordinator.profile() if spec.diagnostic_timing else {}
        engine.coordinator.begin_round(active_ids)
        native_started = time.perf_counter()
        futures = {
            executor.submit(step_match, index, decoded_by_match[index]): index
            for index, runtime in enumerate(runtimes)
            if runtime.active
        }
        results: dict[int, Any] = {}
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException as error:
            engine.coordinator.abort_round(error)
            for future in futures:
                future.cancel()
            raise
        else:
            engine.coordinator.finish_round()
        last_native_step_seconds = time.perf_counter() - native_started
        native_seconds += last_native_step_seconds
        if spec.diagnostic_timing:
            channel_profile_after = engine.coordinator.profile()
            last_native_channel_timing = {
                name.removesuffix("_ns") + "_seconds": (
                    float(channel_profile_after[name]) - float(channel_profile_before[name])
                )
                / 1_000_000_000.0
                for name in ("send_ns", "request_ack_ns", "receive_ns", "decode_ns", "total_ns")
            }
            last_native_channel_timing["outside_channel_seconds"] = max(
                0.0, last_native_step_seconds - last_native_channel_timing["total_seconds"]
            )

        updates: list[tuple[int, float, bool, bool]] = []
        for match_index, result in results.items():
            runtime = runtimes[match_index]
            next_observations, reward, terminated, truncated, _info = result
            before_observations = runtime.observations
            before_tick = int(before_observations[0].tick)
            after_tick = int(next_observations[0].tick)
            generated_elixir = _elixir_generated_raw(before_tick, after_tick) / 10_000.0
            before_elixir = tuple(
                float(
                    next(player for player in before_observations[owner].players if player.owner == owner).elixir_exact
                )
                for owner in (0, 1)
            )
            after_elixir = tuple(
                float(next(player for player in next_observations[owner].players if player.owner == owner).elixir_exact)
                for owner in (0, 1)
            )
            both_full = all(value >= 10.0 - 1e-6 for value in after_elixir)
            personal_wasted = list(runtime.personal_elixir_overflow_wasted)
            personal_accrued = list(runtime.personal_elixir_overflow_cost_accrued)
            unilateral_wasted = list(runtime.unilateral_elixir_overflow_wasted)
            unilateral_accrued = list(runtime.unilateral_elixir_overflow_cost_accrued)
            totals = list(runtime.elixir_overflow_penalty_total)
            personal_totals = list(runtime.personal_elixir_overflow_penalty_total)
            unilateral_totals = list(runtime.unilateral_elixir_overflow_penalty_total)
            personal_grace_crossings = list(runtime.personal_elixir_overflow_grace_crossings)
            cumulative_wasted = list(runtime.cumulative_elixir_overflow_wasted)
            opening_wasted = list(runtime.opening_elixir_overflow_wasted)
            cumulative_unilateral_wasted = list(runtime.cumulative_unilateral_elixir_overflow_wasted)
            max_continuous_wasted = list(runtime.max_continuous_elixir_overflow_wasted)
            first_overflow_tick = list(runtime.first_elixir_overflow_tick)
            adjusted_reward = (
                _scaled_terminal_rewards_v4(reward, next_observations[0].terminal, terminal_reward_scale)
                if terminal_reward_scale != 1.0 else dict(reward)
            )
            for owner in (0, 1):
                newly_wasted = _newly_wasted_elixir_v4(
                    before_elixir=before_elixir[owner],
                    after_elixir=after_elixir[owner],
                    generated_elixir=generated_elixir,
                )
                previous_personal_wasted = personal_wasted[owner]
                (
                    personal_wasted[owner],
                    personal_accrued[owner],
                    unilateral_wasted[owner],
                    unilateral_accrued[owner],
                    personal_penalty,
                    unilateral_penalty,
                ) = _elixir_overflow_penalty_transition_v4(
                    personal_wasted_elixir=personal_wasted[owner],
                    personal_accrued_cost=personal_accrued[owner],
                    unilateral_wasted_elixir=unilateral_wasted[owner],
                    unilateral_accrued_cost=unilateral_accrued[owner],
                    before_elixir=before_elixir[owner],
                    after_elixir=after_elixir[owner],
                    generated_elixir=generated_elixir,
                    both_full=both_full,
                    personal_coefficient=mutual_elixir_overflow_penalty,
                    personal_grace_elixir=mutual_elixir_overflow_grace,
                    unilateral_coefficient=(unilateral_elixir_overflow_penalty),
                    step_penalty_cap=elixir_overflow_step_penalty_cap,
                )
                penalty = personal_penalty + unilateral_penalty
                cumulative_wasted[owner] += newly_wasted
                if after_tick <= 30 * TICKS_PER_SECOND:
                    opening_wasted[owner] += newly_wasted
                if not both_full:
                    cumulative_unilateral_wasted[owner] += newly_wasted
                max_continuous_wasted[owner] = max(max_continuous_wasted[owner], personal_wasted[owner])
                if newly_wasted > 0.0 and first_overflow_tick[owner] is None:
                    first_overflow_tick[owner] = after_tick
                totals[owner] += penalty
                personal_totals[owner] += personal_penalty
                unilateral_totals[owner] += unilateral_penalty
                if previous_personal_wasted <= mutual_elixir_overflow_grace < personal_wasted[owner]:
                    personal_grace_crossings[owner] += 1
                adjusted_reward[owner] = float(adjusted_reward[owner]) - penalty
            runtime.personal_elixir_overflow_wasted = (personal_wasted[0], personal_wasted[1])
            runtime.personal_elixir_overflow_cost_accrued = (personal_accrued[0], personal_accrued[1])
            runtime.unilateral_elixir_overflow_wasted = (unilateral_wasted[0], unilateral_wasted[1])
            runtime.unilateral_elixir_overflow_cost_accrued = (unilateral_accrued[0], unilateral_accrued[1])
            runtime.elixir_overflow_wasted = tuple(personal_wasted[owner] for owner in (0, 1))
            runtime.elixir_overflow_penalty_total = (totals[0], totals[1])
            runtime.personal_elixir_overflow_penalty_total = (personal_totals[0], personal_totals[1])
            runtime.unilateral_elixir_overflow_penalty_total = (unilateral_totals[0], unilateral_totals[1])
            runtime.personal_elixir_overflow_grace_crossings = (
                personal_grace_crossings[0],
                personal_grace_crossings[1],
            )
            runtime.cumulative_elixir_overflow_wasted = (cumulative_wasted[0], cumulative_wasted[1])
            runtime.opening_elixir_overflow_wasted = (opening_wasted[0], opening_wasted[1])
            runtime.cumulative_unilateral_elixir_overflow_wasted = (
                cumulative_unilateral_wasted[0],
                cumulative_unilateral_wasted[1],
            )
            runtime.max_continuous_elixir_overflow_wasted = (max_continuous_wasted[0], max_continuous_wasted[1])
            runtime.first_elixir_overflow_tick = (first_overflow_tick[0], first_overflow_tick[1])
            runtime.observations = next_observations
            if policy_active[match_index]:
                runtime.decision_steps += 1
                if runtime.decision_steps > max_decision_steps:
                    raise RuntimeError("async native match exceeded its policy horizon")
            for owner in (0, 1):
                rejected = 0
                for event in next_observations[owner].events:
                    if (
                        event.event_type == "action_rejected"
                        and event.owner == owner
                        and event.data.get("action_id") in submitted_by_match[match_index][owner]
                    ):
                        identity = (
                            int(event.tick),
                            owner,
                            event.card_id,
                            event.data.get("action_id"),
                            event.data.get("requested_tick"),
                            event.data.get("expected_execution_tick"),
                        )
                        if identity in runtime.seen_rejections:
                            continue
                        runtime.seen_rejections.add(identity)
                        rejected += 1
                        runtime.rejection_details.append(
                            {
                                "tick": int(event.tick),
                                "owner": owner,
                                "card_id": event.card_id,
                                "entity_id": event.entity_id,
                                "position": (tuple(event.position) if event.position is not None else None),
                                "data": event.to_dict()["data"],
                            }
                        )
                counts = list(runtime.rejected_actions)
                counts[owner] += rejected
                runtime.rejected_actions = (counts[0], counts[1])
            apply_fireball_king_activation_penalty_v4(
                runtime, before_observations, next_observations, adjusted_reward, fireball_king_activation_penalty
            )
            apply_king_activation_reward_v4(
                runtime,
                before_observations,
                next_observations,
                adjusted_reward,
                king_activation_penalty,
                king_activation_scope,
            )
            hog_counts = list(runtime.hog_deployments)
            hog_totals = list(runtime.hog_deploy_reward_total)
            first_hog_timing_totals = list(runtime.first_hog_timing_reward_total)
            first_hog_deadline_totals = list(runtime.first_hog_deadline_penalty_total)
            hog_first_ticks = list(runtime.first_hog_deploy_tick)
            for owner in (0, 1):
                successful_cards = list(runtime.first_successful_card_tick)
                successful_cards[owner] = _first_successful_card_tick_v4(
                    next_observations[owner].events, owner=owner, current=successful_cards[owner]
                )
                runtime.first_successful_card_tick = (successful_cards[0], successful_cards[1])
                first_hog_was_already_deployed = hog_first_ticks[owner] is not None
                count, deploy_bonus, first_tick = _credit_hog_deployments_v4(
                    next_observations[owner].events,
                    owner=owner,
                    seen=runtime.seen_hog_deployments,
                    reward_per_deployment=(
                        hog_deploy_reward if owner in runtime.assignment.matchup.current_owners else 0.0
                    ),
                    accrued_reward=hog_totals[owner],
                    episode_cap=hog_deploy_reward_episode_cap,
                    opening_reward_per_deployment=(
                        hog_deploy_opening_reward if owner in runtime.assignment.matchup.current_owners else None
                    ),
                )
                hog_counts[owner] += count
                hog_totals[owner] += deploy_bonus
                timing_bonus = _first_hog_timing_bonus_v4(
                    first_hog_tick=first_tick,
                    first_hog_was_already_deployed=first_hog_was_already_deployed,
                    maximum_bonus=(
                        first_hog_timing_reward_max if owner in runtime.assignment.matchup.current_owners else 0.0
                    ),
                    start_seconds=first_hog_timing_start_seconds,
                    deadline_seconds=first_hog_timing_deadline_seconds,
                )
                first_hog_timing_totals[owner] += timing_bonus
                if first_tick is not None and hog_first_ticks[owner] is None:
                    hog_first_ticks[owner] = first_tick
                deadline_penalty = _first_hog_deadline_penalty_v4(
                    before_tick=before_tick,
                    after_tick=after_tick,
                    first_hog_tick=hog_first_ticks[owner],
                    deadline_seconds=first_hog_timing_deadline_seconds,
                    penalty=(
                        first_hog_missed_deadline_penalty if owner in runtime.assignment.matchup.current_owners else 0.0
                    ),
                )
                first_hog_deadline_totals[owner] += deadline_penalty
                deferral_penalty = first_hog_step_penalty_by_match[match_index][owner]
                adjusted_reward[owner] = (
                    float(adjusted_reward[owner]) + deploy_bonus + timing_bonus - deadline_penalty - deferral_penalty
                )
            runtime.hog_deployments = (hog_counts[0], hog_counts[1])
            runtime.hog_deploy_reward_total = (hog_totals[0], hog_totals[1])
            runtime.first_hog_timing_reward_total = (first_hog_timing_totals[0], first_hog_timing_totals[1])
            runtime.first_hog_deadline_penalty_total = (first_hog_deadline_totals[0], first_hog_deadline_totals[1])
            runtime.first_hog_deploy_tick = (hog_first_ticks[0], hog_first_ticks[1])
            for owner in runtime.assignment.matchup.current_owners:
                updates.append(
                    (
                        match_index * 2 + owner,
                        float(adjusted_reward[owner]),
                        bool(terminated[owner]),
                        bool(truncated[owner]),
                    )
                )
            if bool(terminated[0]) or bool(truncated[0]):
                runtime.active = False
        last_advance_seconds = time.perf_counter() - advance_started
        return tuple(updates)

    segment_started = started
    segment_batch_profile_start: Mapping[str, object] = {}

    def worker_timing(reset_duration: float) -> dict[str, float]:
        batch_profile = engine.coordinator.profile()

        def profile_delta(name: str) -> int:
            return int(batch_profile.get(name, 0)) - int(segment_batch_profile_start.get(name, 0))

        return {
            "total_seconds": time.perf_counter() - segment_started,
            "reset_seconds": reset_duration,
            "tensorize_seconds": tensorize_seconds,
            "native_step_seconds": native_seconds,
            "frame_send_seconds": frame_send_seconds,
            "action_wait_seconds": action_wait_seconds,
            "action_decode_seconds": action_decode_seconds,
            "batch_channel_total_seconds": profile_delta("total_ns") / 1e9,
            "batch_channel_request_ack_seconds": (profile_delta("request_ack_ns") / 1e9),
            "batch_channel_receive_seconds": profile_delta("receive_ns") / 1e9,
            "batch_channel_decode_seconds": profile_delta("decode_ns") / 1e9,
            "batch_channel_response_mib": (profile_delta("response_bytes") / (1024.0 * 1024.0)),
            "batch_channel_decoded_response_mib": (profile_delta("decoded_response_bytes") / (1024.0 * 1024.0)),
            "max_relation_count": float(max_relation_count),
            "max_effect_count": float(max_effect_count),
        }

    def reset_worker_segment_timing() -> None:
        nonlocal segment_started, segment_batch_profile_start
        nonlocal tensorize_seconds, native_seconds, frame_send_seconds
        nonlocal action_wait_seconds, action_decode_seconds
        nonlocal max_relation_count, max_effect_count
        segment_started = time.perf_counter()
        segment_batch_profile_start = engine.coordinator.profile()
        tensorize_seconds = 0.0
        native_seconds = 0.0
        frame_send_seconds = 0.0
        action_wait_seconds = 0.0
        action_decode_seconds = 0.0
        max_relation_count = 0
        max_effect_count = 0

    try:
        engine.prepare()
        reset_started = time.perf_counter()
        runtimes = _reset_engine(engine, assignments, environment_config)
        reset_seconds = time.perf_counter() - reset_started
        engine.coordinator.start_session()
        segment_started = time.perf_counter()
        segment_batch_profile_start = engine.coordinator.profile()
        pending_rewards: tuple[tuple[int, float, bool, bool], ...] = ()
        packet = frame_packet(pending_rewards)
        while True:
            at_segment_boundary = (
                segment_end_decision_steps is not None
                and any(runtime.active for runtime in runtimes)
                and all(
                    not runtime.active or runtime.decision_steps >= segment_end_decision_steps for runtime in runtimes
                )
            )
            while packet is None and any(runtime.active for runtime in runtimes) and not at_segment_boundary:
                pending_rewards = tuple((*pending_rewards, *advance(None)))
                packet = frame_packet(pending_rewards)
                at_segment_boundary = (
                    segment_end_decision_steps is not None
                    and any(runtime.active for runtime in runtimes)
                    and all(
                        not runtime.active or runtime.decision_steps >= segment_end_decision_steps
                        for runtime in runtimes
                    )
                )
            if packet is None and at_segment_boundary:
                packet = frame_packet(pending_rewards, include_segment_boundary=True)
                if packet is None:
                    raise RuntimeError("async segment boundary has no active frame")
                packet["segment_boundary"] = True
                packet["timing"] = worker_timing(reset_seconds)
                frame_send_started = time.perf_counter()
                connection.send(packet)
                frame_send_seconds += time.perf_counter() - frame_send_started
                command = connection.recv()
                if not isinstance(command, Mapping):
                    raise RuntimeError("async segment resume command changed type")
                kind = command.get("kind")
                if kind == "stop":
                    break
                if kind != "resume":
                    raise RuntimeError("async segment expected resume or stop")
                next_policy_state_id = command.get("policy_state_id")
                if not isinstance(next_policy_state_id, str) or not next_policy_state_id:
                    raise RuntimeError("async segment resume policy id changed type")
                policy_state_id = next_policy_state_id
                segment_end_decision_steps += segment_decision_steps
                pending_rewards = ()
                reset_seconds = 0.0
                reset_worker_segment_timing()
                packet = frame_packet(pending_rewards)
                continue
            if packet is None:
                reports = tuple(_terminal_report(runtime) for runtime in runtimes)
                for report in reports:
                    report["terminal_reward_scale"] = terminal_reward_scale
                    report["terminal_outcome_reward"] = tuple(terminal_reward_scale * v for v in report["result"])
                connection.send(
                    {
                        "kind": "done",
                        "engine_index": spec.engine_index,
                        "rewards": pending_rewards,
                        "reports": reports,
                        "timing": worker_timing(reset_seconds),
                    }
                )
                break
            frame_send_started = time.perf_counter()
            connection.send(packet)
            frame_send_seconds += time.perf_counter() - frame_send_started
            action_wait_started = time.perf_counter()
            command = connection.recv()
            action_wait_seconds += time.perf_counter() - action_wait_started
            if not isinstance(command, Mapping):
                raise RuntimeError("async engine received an invalid command")
            action_decode_started = time.perf_counter()
            kind = command.get("kind")
            if kind == "act_packed":
                actions = _actions_from_packed(command.get("actor_rows"), command.get("packed_actions"))
                raw_forced_actor_rows = command.get("forced_act_actor_rows", ())
                if not isinstance(raw_forced_actor_rows, tuple):
                    raise RuntimeError("forced ACT payload changed type")
                forced_actor_rows = frozenset(int(value) for value in raw_forced_actor_rows)
            else:
                raise RuntimeError("async engine received an invalid command")
            last_action_decode_seconds = time.perf_counter() - action_decode_started
            action_decode_seconds += last_action_decode_seconds
            pending_rewards = advance(actions, forced_actor_rows)
            packet = frame_packet(pending_rewards)
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "error",
                    "engine_index": spec.engine_index,
                    "error": repr(error),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:EPISODE_FAILURE_MESSAGE_CHARS],
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        engine.close()
        connection.close()


EPISODE_FAILURE_MESSAGE_CHARS = 300


def episode_failure_records_v4(
    engine_index: int,
    assignments: Sequence[ClusterMatchAssignmentV4],
    packet: Mapping[str, object],
    *,
    lanes_offset: int = 0,
) -> tuple[dict[str, object], ...]:
    """One record per match the failed engine worker was running (single slot: one).

    The worker dies with the exception, so every match it owned is lost.
    """

    error_type = str(packet.get("error_type") or str(packet.get("error", "")).split("(", 1)[0] or "unknown")
    message = str(packet.get("error_message") or packet.get("error") or "")[:EPISODE_FAILURE_MESSAGE_CHARS]
    rows = []
    for lane, assignment in enumerate(assignments):
        matchup = assignment.matchup
        rows.append(
            {
                "time": time.time(),
                "engine_index": int(engine_index),
                "lane": int(lanes_offset + lane),
                "error_type": error_type,
                "error_message": message,
                "sequence": int(matchup.sequence),
                "category": str(matchup.category),
                "policy_matchup": str(matchup.policy_matchup),
                "current_owners": tuple(int(owner) for owner in matchup.current_owners),
                "deck0_id": matchup.deck0.deck_id,
                "deck1_id": matchup.deck1.deck_id,
                "tower_troop_ids": (int(matchup.deck0.tower_troop_id), int(matchup.deck1.tower_troop_id)),
                "battle_level": int(matchup.battle_level),
                "seed": int(matchup.seed),
                "opponent_checkpoint_id": (
                    None if assignment.opponent is None else str(assignment.opponent.checkpoint_id)
                ),
            }
        )
    return tuple(rows)


@torch.inference_mode()
def collect_async_cluster_wave_v4(
    specs: Sequence[AsyncResidentEngineSpecV4],
    assignments: Sequence[ClusterMatchAssignmentV4],
    current_model: UniversalCardPolicyV4,
    anchor_model: UniversalCardPolicyV4,
    history_models: Mapping[str, UniversalCardPolicyV4],
    *,
    gamma_per_decision: float,
    shaping_beta: float,
    mutual_elixir_overflow_penalty: float,
    mutual_elixir_overflow_grace: float,
    unilateral_elixir_overflow_penalty: float,
    elixir_overflow_step_penalty_cap: float,
    policy_state_id: str,
    max_decision_steps: int,
    validate_tensors: bool = False,
    segment_decision_steps: int,
    segment_callback: Callable[[ClusterCollectionV4], str | None],
    hog_deploy_reward: float = 0.0,
    hog_deploy_reward_episode_cap: float = 0.1,
    hog_deploy_opening_reward: float | None = None,
    first_hog_timing_reward_max: float = 0.0,
    first_hog_timing_start_seconds: float = 10.0,
    first_hog_timing_deadline_seconds: float = 30.0,
    first_hog_missed_deadline_penalty: float = 0.0,
    first_hog_deferral_penalty: float = 0.0,
    first_hog_deferral_penalty_episode_cap: float = 0.0,
    fireball_king_activation_penalty: float = 0.0,
    king_activation_penalty: float = 0.0,
    king_activation_scope: str = "any",
    king_shaping_mode: str = "legacy",
    king_potential_weight: float = 3.0,
    terminal_reward_scale: float = 1.0,
    indexed_state_graphs: bool = False,
    adaptive_batching: bool = False,
    cache_static_memory: bool = False,
    tolerate_episode_failures: bool = False,
) -> ClusterCollectionV4:
    """Collect recurrent PPO segments from remote resident engines.

    ``tolerate_episode_failures``: when an engine worker fails (an
    ``"error"`` packet such as a ``BattleEnvError``, or a lost connection),
    its matches are dropped instead of failing the wave.  Every learner frame
    the dropped matches stored in the current segment is invalidated, so the
    lane disappears from the segment's rollout (``select_lanes``) and no
    partial episode reaches the learner.  The engine leaves the wave; its
    lane starts a fresh match with the next wave's assignment.  The records
    are returned as ``timing["episode_failures"]`` with
    ``timing["episode_failure_count"]``; the caller enforces a rate limit.
    """

    normalized_specs = tuple(replace(spec, segment_decision_steps=segment_decision_steps) for spec in specs)
    if not normalized_specs or max_decision_steps <= 0 or segment_decision_steps <= 0:
        raise ValueError("async collection requires engines and a nonnegative wait")
    validate_reward_parameters_v4(locals())
    expected = sum(spec.lanes for spec in normalized_specs)
    if len(assignments) != expected:
        raise ValueError("async assignments do not fill every resident lane")
    by_engine: dict[int, tuple[ClusterMatchAssignmentV4, ...]] = {}
    cursor = 0
    for spec in normalized_specs:
        by_engine[spec.engine_index] = tuple(assignments[cursor : cursor + spec.lanes])
        cursor += spec.lanes
    actors = tuple(
        actor for spec in normalized_specs for actor in _worker_actor_rows(spec, by_engine[spec.engine_index])
    )

    actors_by_key = {
        key: tuple(actor for actor in actors if _inference_model_key(actor) == key)
        for key in sorted({_inference_model_key(actor) for actor in actors})
    }
    models: dict[str, UniversalCardPolicyV4] = {
        "current": current_model,
        "anchor": anchor_model,
        **{f"history:{key}": value for key, value in history_models.items()},
    }
    missing = set(actors_by_key).difference(models)
    if missing:
        raise ValueError(f"async collection lacks actor models: {sorted(missing)}")
    device = next(current_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for model in models.values():
        model.eval()
        if next(model.parameters()).device != device:
            raise ValueError("async actor models must share one GPU")
    if cache_static_memory:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for model in dict.fromkeys(models.values()):
                model.refresh_rollout_static_memory()
    fixed_row = {
        (_inference_model_key(actor), actor.engine_index, actor.local_actor): row
        for key, group in actors_by_key.items()
        for row, actor in enumerate(group)
    }
    states = {
        key: (_rollout_state_bank_v4(models[key], key, len(actors), device) if indexed_state_graphs
              else models[key].initial_state(len(group), device=device))
        for key, group in actors_by_key.items()
    }
    # Preserve the original single-policy fast path. Capture remains serialized
    # by _StochasticCudaGraphV4, while independent policy Graph replays can
    # overlap once their shapes have been captured. Overlapping replays are only
    # safe for isolated graphs (own RNG state and capture stream per policy);
    # policies that share an isolation slot (or a model object) share a stream.
    model_streams: dict[str, torch.cuda.Stream] = {}
    if len(actors_by_key) > 1:
        slot_streams: dict[int, torch.cuda.Stream] = {}
        for key in actors_by_key:
            slot = _rollout_isolation_v4(models[key], device).slot
            if slot not in slot_streams:
                slot_streams[slot] = torch.cuda.Stream(device=device)
            model_streams[key] = slot_streams[slot]
    current_actors = tuple(actor for actor in actors if actor.current)
    initial_state = current_model.initial_state(len(current_actors), device=device).to_storage(
        "cpu", float_dtype=torch.float32
    )
    current_row_by_actor = {(actor.engine_index, actor.local_actor): row for row, actor in enumerate(current_actors)}
    sync_observations: list[UniversalSemanticBatchV4 | None] = []
    sync_actions: list[ActionSequenceV4 | None] = []
    sync_log_probs: list[Tensor] = []
    sync_values: list[Tensor] = []
    sync_rewards: list[Tensor] = []
    sync_terminated: list[Tensor] = []
    sync_truncated: list[Tensor] = []
    sync_valid: list[Tensor] = []
    sync_forced_gate: list[Tensor] = []
    sync_episode_start: list[Tensor] = []
    deferred_packed_observations: list[_RowPackedTensorRecordV4] = []
    deferred_actions: list[ActionSequenceV4] = []
    deferred_log_probs: list[Tensor] = []
    deferred_values: list[Tensor] = []
    deferred_flat_indices: list[Tensor] = []
    sync_last_transition = [-1 for _ in current_actors]
    sync_actor_steps = [0 for _ in current_actors]
    sync_last_observation_rows: list[tuple[UniversalSemanticBatchV4, int] | None] = [None for _ in current_actors]
    sync_last_action_rows: list[tuple[ActionSequenceV4, int] | None] = [None for _ in current_actors]
    sync_boundary_next_value = torch.zeros(len(current_actors))
    sync_boundary_next_value_valid = torch.zeros(len(current_actors), dtype=torch.bool)
    actor_started: set[tuple[int, int]] = set()
    segment_episode_start = {(actor.engine_index, actor.local_actor): True for actor in current_actors}

    if any(spec.worker_host is None or spec.worker_port is None for spec in normalized_specs):
        raise ValueError("production PPO requires remote workers for every engine")
    parents: dict[int, Any] = {}
    shared_batches: dict[int, UniversalSemanticBatchV4] = {}
    model_packed_arena = _ModelPackedArenaV4()
    started = time.perf_counter()
    inference_seconds = 0.0
    assembly_seconds = 0.0
    model_seconds = 0.0
    action_seconds = 0.0
    storage_seconds = 0.0
    command_seconds = 0.0
    ready_wait_seconds = 0.0
    receive_seconds = 0.0
    receive_pipeline_wait_seconds = 0.0
    receive_pipeline_thread_seconds = 0.0
    reports: list[dict[str, object]] = []
    timings: list[Mapping[str, float]] = []
    episode_failures: list[dict[str, object]] = []
    failed_engines: set[int] = set()
    pending: set[int] = set()
    live_engines: set[int] = set()
    engine_packet_counts = {spec.engine_index: 0 for spec in normalized_specs}
    engine_actor_rows = {spec.engine_index: 0 for spec in normalized_specs}
    engine_first_packet_elapsed: dict[int, float] = {}
    inference_batches = 0
    inference_frame_packets = 0
    inference_actor_rows = 0
    inference_max_frame_packets = 0
    packed_relation_capacity = 0
    packed_effect_capacity = 0
    packed_schema_expansions = 0
    direct_row_packed_batches = 0
    direct_row_packed_fallbacks = 0
    deferred_storage_finalize_seconds = 0.0
    rollout_finalize_seconds = 0.0
    packed_tail_padding_seconds = 0.0
    collector_affinity = os.sched_getaffinity(0)
    # Keep exactly one blocking receive outstanding per remote engine. Frames
    # can arrive while the main thread runs GPU inference and rollout storage.
    receive_executor = ThreadPoolExecutor(max_workers=len(normalized_specs), thread_name_prefix="ppo-remote-receive")

    def drop_failed_engine(engine_index: int, packet: Mapping[str, object]) -> None:
        """Forget every match of a failed engine and every frame it stored this segment."""

        episode_failures.extend(episode_failure_records_v4(engine_index, by_engine[engine_index], packet))
        failed_engines.add(engine_index)
        pending.discard(engine_index)
        live_engines.discard(engine_index)
        for actor in current_actors:
            if actor.engine_index != engine_index:
                continue
            row = current_row_by_actor[(actor.engine_index, actor.local_actor)]
            for valid in sync_valid:
                valid[row] = False
            for forced in sync_forced_gate:
                forced[row] = False
            for started_row in sync_episode_start:
                started_row[row] = False
            for terminated_row in sync_terminated:
                terminated_row[row] = False
            for truncated_row in sync_truncated:
                truncated_row[row] = False
            sync_last_transition[row] = -1
            sync_boundary_next_value_valid[row] = False
        try:
            parents[engine_index].close()
        except Exception:
            pass

    def apply_reward_updates(engine_index: int, updates: object) -> None:
        if engine_index in failed_engines:
            return
        if not isinstance(updates, tuple):
            raise TypeError("async reward update payload changed type")
        for local_actor, reward, terminated, truncated in updates:
            row = current_row_by_actor[(engine_index, int(local_actor))]
            transition = sync_last_transition[row]
            if transition < 0:
                raise RuntimeError("native reward arrived before a policy transition")
            sync_rewards[transition][row] += float(reward)
            sync_terminated[transition][row] |= bool(terminated)
            sync_truncated[transition][row] |= bool(truncated)

    def store_current_iteration(
        entries: tuple[_ActorIdentityV4, ...],
        stored_observations: UniversalSemanticBatchV4,
        packed_observations: _RowPackedTensorRecordV4,
        stored_actions: ActionSequenceV4,
        stored_forced_gate: Tensor,
        stored_log_prob: Tensor,
        stored_value: Tensor,
    ) -> float:
        if stored_forced_gate.dtype != torch.bool or tuple(stored_forced_gate.shape) != (len(entries),):
            raise ValueError("stored forced gate mask must be bool [B]")
        storage_started = time.perf_counter()
        flat_indices: list[int] = []
        for source_row, actor in enumerate(entries):
            storage_row = current_row_by_actor[(actor.engine_index, actor.local_actor)]
            transition = sync_actor_steps[storage_row]
            while len(sync_observations) <= transition:
                sync_observations.append(None)
                sync_actions.append(None)
                sync_log_probs.append(torch.zeros(len(current_actors)))
                sync_values.append(torch.zeros(len(current_actors)))
                sync_rewards.append(torch.zeros(len(current_actors)))
                sync_terminated.append(torch.zeros(len(current_actors), dtype=torch.bool))
                sync_truncated.append(torch.zeros(len(current_actors), dtype=torch.bool))
                sync_valid.append(torch.zeros(len(current_actors), dtype=torch.bool))
                sync_forced_gate.append(torch.zeros(len(current_actors), dtype=torch.bool))
                sync_episode_start.append(torch.zeros(len(current_actors), dtype=torch.bool))
            sync_valid[transition][storage_row] = True
            sync_forced_gate[transition][storage_row] = stored_forced_gate[source_row]
            if transition == 0:
                sync_episode_start[transition][storage_row] = segment_episode_start[
                    (actor.engine_index, actor.local_actor)
                ]
            sync_last_transition[storage_row] = transition
            sync_actor_steps[storage_row] += 1
            flat_indices.append(transition * len(current_actors) + storage_row)
        deferred_packed_observations.append(packed_observations)
        deferred_actions.append(stored_actions)
        deferred_log_probs.append(stored_log_prob)
        deferred_values.append(stored_value)
        deferred_flat_indices.append(torch.tensor(flat_indices, dtype=torch.long))
        for row, actor in enumerate(entries):
            storage_row = current_row_by_actor[(actor.engine_index, actor.local_actor)]
            sync_last_observation_rows[storage_row] = (stored_observations, row)
            sync_last_action_rows[storage_row] = (stored_actions, row)
        return time.perf_counter() - storage_started

    def apply_segment_boundary(engine_index: int, packet: Mapping[str, object]) -> None:
        """Bootstrap the final transition without advancing carried RNN state."""

        nonlocal assembly_seconds, model_seconds
        assembly_started = time.perf_counter()
        replacement_batch = packet.get("batch")
        if replacement_batch is not None:
            if not isinstance(replacement_batch, UniversalSemanticBatchV4):
                raise RuntimeError("async segment boundary batch changed type")
            shared_batches[engine_index] = replacement_batch
        batch = shared_batches.get(engine_index)
        actor_rows = packet.get("actor_rows")
        if batch is None or not isinstance(actor_rows, tuple):
            raise RuntimeError("async segment boundary tensor payload is invalid")
        selected: list[tuple[int, _ActorIdentityV4]] = []
        for source_row, local_actor in enumerate(actor_rows):
            actor = actor_by_engine_local[(engine_index, int(local_actor))]
            if actor.current:
                selected.append((source_row, actor))
        if not selected:
            raise RuntimeError("async segment boundary has no current actors")
        selected = [
            (source_row, actor)
            for source_row, actor in selected
            if sync_last_transition[current_row_by_actor[(actor.engine_index, actor.local_actor)]] >= 0
        ]
        # A lane can terminate before producing a learner transition in this
        # segment.  Its boundary observation is still present in the engine
        # batch, but there is no transition to bootstrap or train from.
        if not selected:
            assembly_seconds += time.perf_counter() - assembly_started
            return
        source_indices = torch.tensor([source_row for source_row, _actor in selected], dtype=torch.long)
        with torch.inference_mode(False):
            model_batch = cast(UniversalSemanticBatchV4, batch.index_select(source_indices))
        state_indices = torch.tensor(
            [fixed_row[("current", actor.engine_index, actor.local_actor)] for _source_row, actor in selected],
            dtype=torch.long,
            device=device,
        )
        state = states["current"].index_select(state_indices)
        assembly_seconds += time.perf_counter() - assembly_started
        model_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            context = current_model.forward(
                model_batch.to_model_input(device),
                state,
                episode_start=torch.zeros(len(selected), dtype=torch.bool, device=device),
                validate=validate_tensors,
            )
        values = _stored_tensor(context.value, dtype=torch.float32)
        model_seconds += time.perf_counter() - model_started
        for value_row, (_source_row, actor) in enumerate(selected):
            storage_row = current_row_by_actor[(actor.engine_index, actor.local_actor)]
            transition = sync_last_transition[storage_row]
            sync_boundary_next_value[storage_row] = values[value_row]
            sync_boundary_next_value_valid[storage_row] = True
            sync_truncated[transition][storage_row] = True

    def finalize_segment_rollout() -> StoredRolloutV4:
        nonlocal storage_seconds, deferred_storage_finalize_seconds, packed_tail_padding_seconds
        packed_storage = None
        if deferred_packed_observations:
            finalize_started = time.perf_counter()
            flat_indices = torch.cat(deferred_flat_indices)
            packed_updates = _concatenate_row_packed_records_v4(deferred_packed_observations)
            if packed_updates is None:
                padded_updates = cast(
                    UniversalSemanticBatchV4,
                    concatenate_padded_tensor_records_v4(
                        tuple(packed.record for packed in deferred_packed_observations)
                    ),
                )
                packed_updates = _RowPackedTensorRecordV4(padded_updates)
            flat_rows = len(sync_valid) * len(current_actors)
            packed_storage = packed_updates.empty_rows(flat_rows)
            packed_storage.index_copy_(flat_indices, packed_updates)
            action_updates = cast(ActionSequenceV4, concatenate_tensor_records(tuple(deferred_actions)))
            action_storage = cast(ActionSequenceV4, _empty_storage_rows(action_updates, flat_rows))
            action_storage = cast(
                ActionSequenceV4,
                _scatter_storage_rows(
                    action_storage, action_updates, flat_indices, total_rows=flat_rows, clone_previous=False
                ),
            )
            log_prob_storage = torch.zeros(flat_rows)
            log_prob_storage.index_copy_(0, flat_indices, torch.cat(deferred_log_probs))
            value_storage = torch.zeros(flat_rows)
            value_storage.index_copy_(0, flat_indices, torch.cat(deferred_values))
            actors_per_step = len(current_actors)
            sync_observations[:] = [
                cast(
                    UniversalSemanticBatchV4,
                    packed_storage.narrow_rows(transition * actors_per_step, actors_per_step).record,
                )
                for transition in range(len(sync_valid))
            ]
            sync_actions[:] = [
                cast(ActionSequenceV4, action_storage.narrow_batch(transition * actors_per_step, actors_per_step))
                for transition in range(len(sync_valid))
            ]
            sync_log_probs[:] = list(log_prob_storage.reshape(len(sync_valid), actors_per_step).unbind(0))
            sync_values[:] = list(value_storage.reshape(len(sync_valid), actors_per_step).unbind(0))
            deferred_storage_finalize_seconds = time.perf_counter() - finalize_started
            storage_seconds += deferred_storage_finalize_seconds
        if not sync_observations:
            raise RuntimeError("rectangular async collection stored no decisions")
        lane_has_data = torch.stack(sync_valid).any(dim=0)
        if any(
            (observation_row is None or action_row is None) and bool(lane_has_data[row])
            for row, (observation_row, action_row) in enumerate(
                zip(sync_last_observation_rows, sync_last_action_rows, strict=True)
            )
        ):
            raise RuntimeError("rectangular async collection lacks padding rows")
        # A dropped (failed) lane may never have stored a frame: pad it with any
        # stored row; the lane is removed below by ``select_lanes``.
        fallback_observation_row = next((item for item in sync_last_observation_rows if item is not None), None)
        fallback_action_row = next((item for item in sync_last_action_rows if item is not None), None)
        if fallback_observation_row is None or fallback_action_row is None:
            raise RuntimeError("rectangular async collection lacks padding rows")
        last_observation_rows = tuple(
            (fallback_observation_row if item is None else item)[0].narrow_batch(
                (fallback_observation_row if item is None else item)[1], 1
            )
            for item in sync_last_observation_rows
        )
        last_action_rows = tuple(
            (fallback_action_row if item is None else item)[0].narrow_batch(
                (fallback_action_row if item is None else item)[1], 1
            )
            for item in sync_last_action_rows
        )
        sync_last_observation = cast(
            UniversalSemanticBatchV4, concatenate_padded_tensor_records_v4(last_observation_rows)
        )
        sync_last_action = cast(ActionSequenceV4, concatenate_tensor_records(last_action_rows))
        valid_mask = torch.stack(sync_valid)
        tail_started = time.perf_counter()
        packed_tail_filled = packed_storage is not None and _pad_packed_rollout_tail_v4(
            packed_storage, sync_last_observation, valid_mask
        )
        packed_tail_padding_seconds = time.perf_counter() - tail_started
        final_observations: list[UniversalSemanticBatchV4] = []
        final_actions: list[ActionSequenceV4] = []
        for transition, valid in enumerate(sync_valid):
            observation = sync_observations[transition]
            action = sync_actions[transition]
            if observation is None or action is None:
                raise RuntimeError("rectangular rollout contains an empty step")
            missing_indices = (~valid).nonzero(as_tuple=False).flatten()
            if missing_indices.numel() and not packed_tail_filled:
                observation = cast(
                    UniversalSemanticBatchV4,
                    _scatter_storage_rows(
                        observation,
                        sync_last_observation.index_select(missing_indices),
                        missing_indices,
                        total_rows=len(current_actors),
                        clone_previous=False,
                    ),
                )
            if missing_indices.numel():
                action = cast(
                    ActionSequenceV4,
                    _scatter_storage_rows(
                        action,
                        sync_last_action.index_select(missing_indices),
                        missing_indices,
                        total_rows=len(current_actors),
                        clone_previous=False,
                    ),
                )
            final_observations.append(observation)
            final_actions.append(action)
        behavior_value = torch.stack(sync_values)
        terminated = torch.stack(sync_terminated)
        next_value = torch.zeros_like(behavior_value)
        if len(sync_values) > 1:
            next_value[:-1] = torch.where(
                terminated[:-1] | ~valid_mask[1:], torch.zeros_like(behavior_value[1:]), behavior_value[1:]
            )
        for storage_row in sync_boundary_next_value_valid.nonzero(as_tuple=False).flatten():
            row = int(storage_row)
            transition = sync_last_transition[row]
            if transition >= 0:
                next_value[transition, row] = sync_boundary_next_value[row]
        rollout = StoredRolloutV4(
            observations=tuple(final_observations),
            packed_observation_rows=packed_storage,
            actions=tuple(final_actions),
            episode_start=torch.stack(sync_episode_start),
            valid_mask=valid_mask,
            forced_gate_mask=torch.stack(sync_forced_gate),
            behavior_log_prob=torch.stack(sync_log_probs),
            behavior_value=behavior_value,
            next_value=next_value,
            rewards=torch.stack(sync_rewards),
            terminated=terminated,
            truncated=torch.stack(sync_truncated),
            initial_state=cast(RecurrentPolicyStateV4, initial_state),
            gate_temperature=current_model.ppo_gate_temperature,
            action_temperature=current_model.ppo_action_temperature,
            continue_temperature=current_model.ppo_continue_temperature,
            policy_state_id=policy_state_id,
            lane_keys=tuple((actor.engine_index, actor.local_actor) for actor in current_actors),
        )
        active_lanes = valid_mask.any(dim=0).nonzero(as_tuple=False).flatten()
        if not active_lanes.numel():
            raise RuntimeError("async segment contains no learner lanes")
        if int(active_lanes.numel()) != rollout.batch_size:
            rollout = rollout.select_lanes(active_lanes)
        rollout.validate()
        return rollout

    def build_segment_collection(rollout: StoredRolloutV4, *, elapsed_seconds: float) -> ClusterCollectionV4:
        if not timings:
            raise RuntimeError("async segment has no worker timings")
        return ClusterCollectionV4(
            rollout=rollout,
            reports=tuple(reports),
            timing={
                "total_seconds": elapsed_seconds,
                "reset_seconds": max(float(row["reset_seconds"]) for row in timings),
                "tensorize_seconds": max(float(row["tensorize_seconds"]) for row in timings),
                "inference_seconds": inference_seconds,
                "assembly_seconds": assembly_seconds,
                "model_seconds": model_seconds,
                "action_seconds": action_seconds,
                "storage_seconds": storage_seconds,
                "deferred_storage_finalize_seconds": (deferred_storage_finalize_seconds),
                "rollout_finalize_seconds": rollout_finalize_seconds,
                "packed_tail_padding_seconds": packed_tail_padding_seconds,
                "packed_relation_capacity": float(packed_relation_capacity),
                "packed_effect_capacity": float(packed_effect_capacity),
                "packed_schema_expansions": float(packed_schema_expansions),
                "packed_storage_schema_fallbacks": 0.0,
                "direct_row_packed_batches": float(direct_row_packed_batches),
                "direct_row_packed_fallbacks": float(direct_row_packed_fallbacks),
                "command_seconds": command_seconds,
                "ready_wait_seconds": ready_wait_seconds,
                "receive_seconds": receive_seconds,
                "receive_pipeline_wait_seconds": receive_pipeline_wait_seconds,
                "receive_pipeline_thread_seconds": receive_pipeline_thread_seconds,
                "frame_send_seconds": max(float(row["frame_send_seconds"]) for row in timings),
                "action_wait_seconds": max(float(row["action_wait_seconds"]) for row in timings),
                "action_decode_seconds": max(float(row["action_decode_seconds"]) for row in timings),
                "native_step_seconds": max(float(row["native_step_seconds"]) for row in timings),
                "batch_channel_total_seconds": max(float(row["batch_channel_total_seconds"]) for row in timings),
                "batch_channel_request_ack_seconds": max(
                    float(row["batch_channel_request_ack_seconds"]) for row in timings
                ),
                "batch_channel_receive_seconds": max(float(row["batch_channel_receive_seconds"]) for row in timings),
                "batch_channel_decode_seconds": max(float(row["batch_channel_decode_seconds"]) for row in timings),
                "batch_channel_response_mib": max(float(row["batch_channel_response_mib"]) for row in timings),
                "batch_channel_decoded_response_mib": max(
                    float(row["batch_channel_decoded_response_mib"]) for row in timings
                ),
                "matches": float(len(reports)),
                "matches_per_minute": (60.0 * len(reports) / max(elapsed_seconds, 1e-9)),
                "owner_frames": float(rollout.valid_mask.sum()),
                "owner_frames_per_second": (float(rollout.valid_mask.sum()) / max(elapsed_seconds, 1e-9)),
                "episode_start_count": float(rollout.episode_start.sum()),
                "truncated_count": float(rollout.truncated.sum()),
                "boundary_bootstrap_count": float(sync_boundary_next_value_valid.sum()),
                "initial_hidden_abs_max": float(rollout.initial_state.hidden.abs().max()),
                "live_engines": float(len(live_engines)),
                "configured_engines": float(len(normalized_specs)),
                "configured_matches": float(len(assignments)),
                "episode_failure_count": float(len(episode_failures)),
                "episode_failures": tuple(dict(row) for row in episode_failures),  # type: ignore[dict-item]
                "inference_batches": float(inference_batches),
                "inference_mean_frame_packets": (inference_frame_packets / max(inference_batches, 1)),
                "inference_mean_actor_rows": (inference_actor_rows / max(inference_batches, 1)),
                "inference_max_frame_packets": float(inference_max_frame_packets),
                "max_relation_count": max(float(row["max_relation_count"]) for row in timings),
                "max_effect_count": max(float(row["max_effect_count"]) for row in timings),
            },
        )

    def reset_parent_segment(next_policy_state_id: str) -> None:
        nonlocal policy_state_id, initial_state, started
        nonlocal inference_seconds, assembly_seconds, model_seconds
        nonlocal action_seconds, storage_seconds, command_seconds
        nonlocal ready_wait_seconds, receive_seconds
        nonlocal receive_pipeline_wait_seconds, receive_pipeline_thread_seconds
        nonlocal inference_batches, inference_frame_packets, inference_actor_rows
        nonlocal inference_max_frame_packets, packet_count
        nonlocal packed_schema_expansions
        nonlocal direct_row_packed_batches, direct_row_packed_fallbacks
        nonlocal deferred_storage_finalize_seconds, segment_episode_start
        nonlocal rollout_finalize_seconds, packed_tail_padding_seconds
        policy_state_id = next_policy_state_id
        current_state_indices = torch.tensor(
            [fixed_row[("current", actor.engine_index, actor.local_actor)] for actor in current_actors],
            dtype=torch.long,
            device=device,
        )
        initial_state = (
            states["current"].index_select(current_state_indices).to_storage("cpu", float_dtype=torch.float32)
        )
        sync_observations.clear()
        sync_actions.clear()
        sync_log_probs.clear()
        sync_values.clear()
        sync_rewards.clear()
        sync_terminated.clear()
        sync_truncated.clear()
        sync_valid.clear()
        sync_forced_gate.clear()
        sync_episode_start.clear()
        deferred_packed_observations.clear()
        deferred_actions.clear()
        deferred_log_probs.clear()
        deferred_values.clear()
        deferred_flat_indices.clear()
        for row in range(len(current_actors)):
            sync_last_transition[row] = -1
            sync_actor_steps[row] = 0
        sync_boundary_next_value.zero_()
        sync_boundary_next_value_valid.zero_()
        segment_episode_start = {
            (actor.engine_index, actor.local_actor): ((actor.engine_index, actor.local_actor) not in actor_started)
            for actor in current_actors
        }
        reports.clear()
        timings.clear()
        episode_failures.clear()
        started = time.perf_counter()
        inference_seconds = 0.0
        assembly_seconds = 0.0
        model_seconds = 0.0
        action_seconds = 0.0
        storage_seconds = 0.0
        command_seconds = 0.0
        ready_wait_seconds = 0.0
        receive_seconds = 0.0
        receive_pipeline_wait_seconds = 0.0
        receive_pipeline_thread_seconds = 0.0
        inference_batches = 0
        inference_frame_packets = 0
        inference_actor_rows = 0
        inference_max_frame_packets = 0
        packet_count = 0
        packed_schema_expansions = 0
        direct_row_packed_batches = 0
        direct_row_packed_fallbacks = 0
        deferred_storage_finalize_seconds = 0.0
        rollout_finalize_seconds = 0.0
        packed_tail_padding_seconds = 0.0
        for engine_index in engine_packet_counts:
            engine_packet_counts[engine_index] = 0
            engine_actor_rows[engine_index] = 0
        engine_first_packet_elapsed.clear()

    try:
        for spec in normalized_specs:
            assert spec.worker_host is not None and spec.worker_port is not None
            parent = RemoteWorkerConnection.connect(spec.worker_host, spec.worker_port, timeout=spec.timeout)
            parent.send(
                {
                    "kind": "start",
                    "spec": spec,
                    "assignments": by_engine[spec.engine_index],
                    "gamma_per_decision": gamma_per_decision,
                    "shaping_beta": shaping_beta,
                    "terminal_reward_scale": terminal_reward_scale,
                    "fireball_king_activation_penalty": fireball_king_activation_penalty,
                    "king_activation_penalty": king_activation_penalty,
                    "king_activation_scope": king_activation_scope,
                    "king_shaping_mode": king_shaping_mode,
                    "king_potential_weight": king_potential_weight,
                    "hog_deploy_reward": hog_deploy_reward,
                    "hog_deploy_reward_episode_cap": hog_deploy_reward_episode_cap,
                    "hog_deploy_opening_reward": hog_deploy_opening_reward,
                    "first_hog_timing_reward_max": first_hog_timing_reward_max,
                    "first_hog_timing_start_seconds": (first_hog_timing_start_seconds),
                    "first_hog_timing_deadline_seconds": (first_hog_timing_deadline_seconds),
                    "first_hog_missed_deadline_penalty": (first_hog_missed_deadline_penalty),
                    "first_hog_deferral_penalty": first_hog_deferral_penalty,
                    "first_hog_deferral_penalty_episode_cap": (first_hog_deferral_penalty_episode_cap),
                    "mutual_elixir_overflow_penalty": (mutual_elixir_overflow_penalty),
                    "mutual_elixir_overflow_grace": (mutual_elixir_overflow_grace),
                    "unilateral_elixir_overflow_penalty": (unilateral_elixir_overflow_penalty),
                    "elixir_overflow_step_penalty_cap": (elixir_overflow_step_penalty_cap),
                    "policy_state_id": policy_state_id,
                    "validate_tensors": validate_tensors,
                    "max_decision_steps": max_decision_steps,
                }
            )
            parents[spec.engine_index] = parent
            pending.add(spec.engine_index)
            live_engines.add(spec.engine_index)

        connection_to_engine = {connection: engine_index for engine_index, connection in parents.items()}

        def receive(connection: Any) -> tuple[Any, object, float]:
            receive_started = time.perf_counter()
            packet = connection.recv()
            return connection, packet, time.perf_counter() - receive_started

        receive_futures = {
            receive_executor.submit(receive, connection): connection for connection in connection_to_engine
        }
        packet_count = 0
        progress_packet_interval = 100 * len(normalized_specs)
        actor_by_engine_local = {(actor.engine_index, actor.local_actor): actor for actor in actors}

        def fetch_and_prepare_batch() -> dict[str, Any] | None:
            """Receive and assemble the next inference batch."""

            nonlocal ready_wait_seconds, receive_seconds
            nonlocal receive_pipeline_wait_seconds
            nonlocal receive_pipeline_thread_seconds
            nonlocal assembly_seconds, direct_row_packed_batches
            nonlocal direct_row_packed_fallbacks
            nonlocal packed_relation_capacity, packed_effect_capacity
            nonlocal packed_schema_expansions
            while pending:
                ready_wait_started = time.perf_counter()
                wait_budget = _INFERENCE_BATCH_WAIT_SECONDS
                if adaptive_batching:
                    # Spend at most one observed GPU batch's latency waiting
                    # for a fuller batch, capped at 25 ms. Native time only
                    # advances after actions, so this changes scheduling alone.
                    wait_budget = min(0.025, max(wait_budget, model_seconds / max(inference_batches, 1)))
                ready_futures = _await_inference_batch_v4(
                    receive_futures, wait_seconds=wait_budget, gather_all=adaptive_batching)
                ready_wait_seconds += time.perf_counter() - ready_wait_started

                receive_started = time.perf_counter()

                def received_or_failure(future: Any) -> tuple[Any, object, float]:
                    try:
                        return future.result()
                    except Exception as error:
                        if not tolerate_episode_failures:
                            raise
                        return (
                            receive_futures[future],
                            {
                                "kind": "error",
                                "error": repr(error),
                                "error_type": type(error).__name__,
                                "error_message": f"worker connection lost: {error}"[:EPISODE_FAILURE_MESSAGE_CHARS],
                            },
                            0.0,
                        )

                received_with_durations = tuple(received_or_failure(future) for future in ready_futures)
                for future in ready_futures:
                    del receive_futures[future]
                received = tuple((connection, packet) for connection, packet, _duration in received_with_durations)
                receive_pipeline_thread_seconds += sum(
                    duration for _connection, _packet, duration in received_with_durations
                )
                receive_pipeline_wait_seconds += time.perf_counter() - ready_wait_started
                receive_seconds += time.perf_counter() - receive_started

                frames: list[tuple[int, Mapping[str, object]]] = []
                deferred_reward_updates: list[tuple[int, object]] = []
                for connection, packet in received:
                    engine_index = connection_to_engine[connection]
                    if not isinstance(packet, Mapping):
                        raise RuntimeError("async worker packet changed type")
                    kind = packet.get("kind")
                    if kind == "error":
                        if tolerate_episode_failures:
                            drop_failed_engine(engine_index, packet)
                            continue
                        raise RuntimeError(
                            f"async engine {engine_index} failed: {packet.get('error')}\n{packet.get('traceback')}"
                        )
                    deferred_reward_updates.append((engine_index, packet.get("rewards", ())))
                    if kind == "frame" and packet.get("segment_boundary") is True:
                        timing = packet.get("timing")
                        if not isinstance(timing, Mapping):
                            raise RuntimeError("async segment timing is invalid")
                        apply_segment_boundary(engine_index, packet)
                        timings.append(cast(Mapping[str, float], timing))
                        pending.remove(engine_index)
                        continue
                    if kind == "done":
                        raw_reports = packet.get("reports")
                        timing = packet.get("timing")
                        if not isinstance(raw_reports, tuple) or not isinstance(timing, Mapping):
                            raise RuntimeError("async terminal packet is invalid")
                        reports.extend(cast(tuple[dict[str, object], ...], raw_reports))
                        timings.append(cast(Mapping[str, float], timing))
                        pending.remove(engine_index)
                        live_engines.remove(engine_index)
                        connection.close()
                        continue
                    if kind != "frame":
                        raise RuntimeError("async worker emitted an unknown packet")
                    frames.append((engine_index, packet))

                if not frames:
                    return {"frames": (), "deferred_reward_updates": deferred_reward_updates}

                frames.sort(key=lambda item: item[0])
                assembly_started = time.perf_counter()
                active_actors: list[_ActorIdentityV4] = []
                active_force_flags: list[bool] = []
                storage_actors: list[_ActorIdentityV4] = []
                frame_batches: list[UniversalSemanticBatchV4] = []
                frame_packed_batches: list[Any] = []
                frame_lengths: list[int] = []
                for engine_index, packet in frames:
                    actor_rows = packet.get("actor_rows")
                    storage_actor_rows = packet.get("storage_actor_rows")
                    force_act_actor_rows = packet.get("force_act_actor_rows")
                    replacement_batch = packet.get("batch")
                    row_packed_batch = packet.get("row_packed_batch")
                    if replacement_batch is not None:
                        if not isinstance(replacement_batch, UniversalSemanticBatchV4):
                            raise RuntimeError("async shared batch changed type")
                        shared_batches[engine_index] = replacement_batch
                    batch = shared_batches.get(engine_index)
                    if (
                        not isinstance(actor_rows, tuple)
                        or not isinstance(storage_actor_rows, tuple)
                        or not isinstance(force_act_actor_rows, tuple)
                        or batch is None
                    ):
                        raise RuntimeError("async frame tensor payload is invalid")
                    force_act_set = {int(value) for value in force_act_actor_rows}
                    if not force_act_set.issubset({int(value) for value in actor_rows}):
                        raise RuntimeError("forced ACT rows are outside the frame")
                    engine_packet_counts[engine_index] += 1
                    engine_actor_rows[engine_index] += len(actor_rows)
                    engine_first_packet_elapsed.setdefault(engine_index, time.perf_counter() - started)
                    active_actors.extend(
                        actor_by_engine_local[(engine_index, int(local_actor))] for local_actor in actor_rows
                    )
                    active_force_flags.extend(int(local_actor) in force_act_set for local_actor in actor_rows)
                    storage_actors.extend(
                        actor_by_engine_local[(engine_index, int(local_actor))] for local_actor in storage_actor_rows
                    )
                    frame_batches.append(batch)
                    frame_lengths.append(len(actor_rows))
                    if row_packed_batch is not None:
                        if not hasattr(row_packed_batch, "narrow_rows"):
                            raise RuntimeError("async row-packed frame changed type")
                        frame_packed_batches.append(row_packed_batch)

                entries_by_model: dict[str, list[_ActorIdentityV4]] = {key: [] for key in actors_by_key}
                source_rows_by_model: dict[str, list[int]] = {key: [] for key in actors_by_key}
                for source_row, actor in enumerate(active_actors):
                    inference_key = _inference_model_key(actor)
                    entries_by_model[inference_key].append(actor)
                    source_rows_by_model[inference_key].append(source_row)
                if storage_actors != entries_by_model.get("current", []):
                    raise RuntimeError("async current-policy storage row order changed")

                with torch.inference_mode(False):
                    model_packed_batches = (
                        model_packed_arena.assemble(tuple(zip(frame_packed_batches, frame_lengths)), source_rows_by_model)
                        if len(frame_packed_batches) == len(frame_batches) else None
                    )
                if model_packed_batches is not None:
                    direct_row_packed_batches += 1
                    assembly_seconds += time.perf_counter() - assembly_started
                    return {
                        "frames": frames, "deferred_reward_updates": deferred_reward_updates,
                        "active_packed_batch": None, "model_packed_batches": model_packed_batches,
                        "entries_by_model": entries_by_model, "source_rows_by_model": source_rows_by_model,
                        "active_force_flags": active_force_flags, "actor_rows": len(active_actors),
                    }

                frame_batches = [
                    batch if length == batch.batch_size else batch.narrow_batch(0, length)
                    for batch, length in zip(frame_batches, frame_lengths)
                ]
                frame_packed_batches = [
                    batch.narrow_rows(0, length) for batch, length in zip(frame_packed_batches, frame_lengths)
                ] if len(frame_packed_batches) == len(frame_lengths) else []

                with torch.inference_mode(False):
                    direct_packed_batch = (
                        _concatenate_row_packed_records_v4(frame_packed_batches)
                        if len(frame_packed_batches) == len(frame_batches)
                        else None
                    )
                    if direct_packed_batch is None:
                        direct_row_packed_fallbacks += 1
                        active_batch = cast(
                            UniversalSemanticBatchV4, concatenate_padded_tensor_records_v4(tuple(frame_batches))
                        )
                    else:
                        direct_row_packed_batches += 1
                        active_batch = cast(UniversalSemanticBatchV4, direct_packed_batch.record)
                    if direct_packed_batch is None:
                        next_relation_capacity = max(
                            packed_relation_capacity, int(active_batch.relation_edges.mask.shape[1])
                        )
                        next_effect_capacity = max(
                            packed_effect_capacity, int(active_batch.active_effects.mask.shape[1])
                        )
                        if (
                            next_relation_capacity != packed_relation_capacity
                            or next_effect_capacity != packed_effect_capacity
                        ):
                            packed_schema_expansions += 1
                        packed_relation_capacity = next_relation_capacity
                        packed_effect_capacity = next_effect_capacity
                        active_batch = _pad_relation_edges(active_batch, packed_relation_capacity)
                        active_batch = _pad_active_effects(active_batch, packed_effect_capacity)
                    active_packed_batch = (
                        direct_packed_batch
                        if direct_packed_batch is not None
                        else _RowPackedTensorRecordV4(active_batch)
                    )
                    active_batch = cast(UniversalSemanticBatchV4, active_packed_batch.record)

                entries_by_model: dict[str, list[_ActorIdentityV4]] = {key: [] for key in actors_by_key}
                source_rows_by_model: dict[str, list[int]] = {key: [] for key in actors_by_key}
                for source_row, actor in enumerate(active_actors):
                    inference_key = _inference_model_key(actor)
                    entries_by_model[inference_key].append(actor)
                    source_rows_by_model[inference_key].append(source_row)
                current_storage_entries = entries_by_model.get("current", [])
                if storage_actors != current_storage_entries:
                    raise RuntimeError("async current-policy storage row order changed")
                assembly_seconds += time.perf_counter() - assembly_started
                return {
                    "frames": frames,
                    "deferred_reward_updates": deferred_reward_updates,
                    "active_batch": active_batch,
                    "active_packed_batch": active_packed_batch,
                    "entries_by_model": entries_by_model,
                    "source_rows_by_model": source_rows_by_model,
                    "active_force_flags": active_force_flags,
                    "actor_rows": len(active_actors),
                }
            return None

        final_segment_collection: ClusterCollectionV4 | None = None
        while pending or bool(sync_valid):
            if not pending:
                segment_elapsed = time.perf_counter() - started
                finalize_started = time.perf_counter()
                segment_rollout = finalize_segment_rollout()
                rollout_finalize_seconds = time.perf_counter() - finalize_started
                final_segment_collection = build_segment_collection(segment_rollout, elapsed_seconds=segment_elapsed)
                next_policy_state_id = segment_callback(final_segment_collection)
                if next_policy_state_id is None or not live_engines:
                    for engine_index in tuple(live_engines):
                        parents[engine_index].send({"kind": "stop"})
                    break
                if not isinstance(next_policy_state_id, str) or not (next_policy_state_id):
                    raise RuntimeError("async segment callback returned no policy id")
                reset_parent_segment(next_policy_state_id)
                pending.update(live_engines)
                for engine_index in tuple(live_engines):
                    connection = parents[engine_index]
                    connection.send({"kind": "resume", "policy_state_id": next_policy_state_id})
                    future = receive_executor.submit(receive, connection)
                    receive_futures[future] = connection
                continue
            prepared = fetch_and_prepare_batch()
            if prepared is None:
                break

            frames = cast(list[tuple[int, Mapping[str, object]]], prepared["frames"])
            deferred_reward_updates = cast(list[tuple[int, object]], prepared["deferred_reward_updates"])
            if not frames:
                for engine_index, updates in deferred_reward_updates:
                    apply_reward_updates(engine_index, updates)
                continue

            inference_batches += 1
            inference_frame_packets += len(frames)
            inference_max_frame_packets = max(inference_max_frame_packets, len(frames))
            inference_actor_rows += int(prepared["actor_rows"])
            inference_started = time.perf_counter()
            active_packed_batch = cast(_RowPackedTensorRecordV4, prepared["active_packed_batch"])
            model_packed_batches = prepared.get("model_packed_batches")
            entries_by_model = cast(dict[str, list[_ActorIdentityV4]], prepared["entries_by_model"])
            source_rows_by_model = cast(dict[str, list[int]], prepared["source_rows_by_model"])
            active_force_flags = cast(list[bool], prepared["active_force_flags"])

            outputs: dict[str, Any] = {}
            current_storage_packed: _RowPackedTensorRecordV4 | None = None
            force_context_by_key: dict[
                str, tuple[UniversalSemanticBatchV4, RecurrentPolicyStateV4, Tensor, Tensor]
            ] = {}
            model_started = time.perf_counter()
            coordinator_stream = torch.cuda.current_stream(device) if model_streams else None
            active_model_keys: list[str] = []
            inflight_model_resources: list[object] = []
            for key, entries in entries_by_model.items():
                if not entries:
                    continue
                with torch.inference_mode(False):
                    if model_packed_batches is not None:
                        partition = model_packed_batches[key]
                        # Rollout storage retains observations until the update;
                        # own these rows before commands can overwrite receive buffers.
                        if key == "current":
                            active_model_packed = _RowPackedTensorRecordV4._from_buffers(partition, {
                                dtype: buffer[:len(entries)].clone() for dtype, buffer in partition.buffers.items()
                            })
                        else:
                            active_model_packed = partition.narrow_rows(0, len(entries))
                    else:
                        active_batch_indices = torch.tensor(source_rows_by_model[key], dtype=torch.long)
                        active_model_packed = active_packed_batch.index_select(active_batch_indices)
                    active_model_batch = active_model_packed.record
                    if key == "current":
                        current_storage_packed = active_model_packed
                state_rows = [fixed_row[(key, actor.engine_index, actor.local_actor)] for actor in entries]
                start_rows = [(actor.engine_index, actor.local_actor) not in actor_started for actor in entries]
                has_force_request = any(active_force_flags[row] for row in source_rows_by_model[key])
                active_indices = None
                active_state = None
                if not indexed_state_graphs or has_force_request:
                    active_indices = torch.tensor(state_rows, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
                    active_state = states[key].index_select(active_indices)
                model_capacity = _ppo_cuda_graph_row_capacity_v4(len(entries))
                if len(entries) < model_capacity:
                    pad_rows = torch.cat(
                        (
                            torch.arange(len(entries), dtype=torch.long),
                            torch.zeros(model_capacity - len(entries), dtype=torch.long),
                        )
                    )
                    packed_batch = (
                        model_packed_batches[key] if model_packed_batches is not None
                        else active_model_packed.index_select(pad_rows)
                    )
                    batch = packed_batch.record
                    model_state = None
                    if not indexed_state_graphs:
                        state_pad = torch.cat(
                            (torch.arange(len(entries), device=device),
                             torch.zeros(model_capacity - len(entries), dtype=torch.long, device=device)))
                        model_state = active_state.index_select(state_pad)
                else:
                    packed_batch = model_packed_batches[key] if model_packed_batches is not None else active_model_packed
                    batch = packed_batch.record
                    model_state = active_state
                episode_start = None
                if not indexed_state_graphs or has_force_request:
                    episode_start = torch.tensor(
                        start_rows + [False] * (model_capacity - len(entries)),
                        dtype=torch.bool, pin_memory=True).to(device, non_blocking=True)
                graph = _persistent_rollout_graph_v4(
                    models[key],
                    device,
                    model_capacity,
                    effect_capacity=int(batch.active_effects.mask.shape[1]),
                    relation_capacity=int(batch.relation_edges.mask.shape[1]),
                    isolated=bool(model_streams),
                    state_bank=states[key] if indexed_state_graphs else None,
                    actor_capacity=len(actors) if indexed_state_graphs else 0,
                    state_bank_key=key,
                )
                if graph is None:
                    raise RuntimeError("production rollout CUDA graph is unavailable")
                if indexed_state_graphs:
                    if not isinstance(graph, _IndexedStateCudaGraphV4):
                        raise RuntimeError("indexed rollout graph changed type")
                    graph.prepare(state_rows, start_rows)
                model_stream = model_streams.get(key)
                if model_stream is not None:
                    if coordinator_stream is None:
                        raise RuntimeError("parallel model coordinator is unavailable")
                    model_stream.wait_stream(coordinator_stream)
                with torch.cuda.stream(model_stream) if model_stream is not None else nullcontext():
                    if indexed_state_graphs:
                        output = graph.run_indexed(models[key], batch, packed_batch)
                        selected_next = None
                    else:
                        output = graph.run(models[key], batch, model_state, episode_start, packed_batch)
                        selected_next = cast(RecurrentPolicyStateV4, output.next_state.narrow_batch(0, len(entries)))
                        selected_next = RecurrentPolicyStateV4(
                            hidden=selected_next.hidden.to(dtype=states[key].hidden.dtype),
                            cell=selected_next.cell.to(dtype=states[key].cell.dtype),
                        )
                        states[key].hidden.index_copy_(0, active_indices, selected_next.hidden)
                        states[key].cell.index_copy_(0, active_indices, selected_next.cell)
                if model_stream is not None:
                    # Coordinator-owned scratch tensors must outlive asynchronous
                    # consumption on the model stream; otherwise the CUDA allocator
                    # may recycle active_indices before index_copy_ reads it.
                    inflight_model_resources.append(
                        (batch, packed_batch, active_indices, model_state, episode_start, selected_next)
                    )
                    active_model_keys.append(key)
                actor_started.update((actor.engine_index, actor.local_actor) for actor in entries)
                outputs[key] = output
                requested_force = torch.tensor(
                    [active_force_flags[source_row] for source_row in source_rows_by_model[key]], dtype=torch.bool
                )
                if requested_force.any():
                    legal_force = (
                        _match_shadow_legality_v4(active_model_batch, models[key].config)
                        .candidate_mask()
                        .any(dim=-1)
                    )
                    requested_force &= legal_force.cpu()
                if requested_force.any():
                    if active_state is None or episode_start is None:
                        raise RuntimeError("forced ACT lost its pre-inference recurrent state")
                    force_context_by_key[key] = (
                        active_model_batch,
                        active_state,
                        episode_start[: len(entries)],
                        requested_force,
                    )
            if coordinator_stream is not None:
                for key in active_model_keys:
                    coordinator_stream.wait_stream(model_streams[key])
                coordinator_stream.synchronize()
            del inflight_model_resources

            forced_rows_by_key: dict[str, Tensor] = {}
            for key, (host_batch, source_state, source_episode_start, requested_force) in force_context_by_key.items():
                output = outputs[key]
                requested_indices = requested_force.nonzero(as_tuple=False).flatten().to(device)
                original_wait = output.actions.gate.index_select(0, requested_indices) == GATE_WAIT
                forced_indices = requested_indices.index_select(0, original_wait.nonzero(as_tuple=False).flatten())
                if not forced_indices.numel():
                    continue
                host_indices = forced_indices.cpu()
                with torch.inference_mode(False):
                    forced_batch = cast(UniversalSemanticBatchV4, host_batch.index_select(host_indices)).to_model_input(
                        device
                    )
                forced_state = source_state.index_select(forced_indices)
                forced_episode_start = source_episode_start.index_select(0, forced_indices)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    forced_context = models[key].forward(
                        forced_batch, forced_state, episode_start=forced_episode_start, validate=validate_tensors
                    )
                    forced_output = models[key].sample_after_preselected_act(
                        forced_batch, forced_context, validate=validate_tensors
                    )
                if forced_output.action_components is None:
                    raise RuntimeError("forced ACT output lacks head components")
                for item in fields(ActionSequenceV4):
                    getattr(output.actions, item.name).index_copy_(
                        0, forced_indices, getattr(forced_output.actions, item.name)
                    )
                output.log_prob.index_copy_(
                    0, forced_indices, forced_output.log_prob - forced_output.action_components.gate_log_prob
                )
                forced_rows_by_key[key] = host_indices
            model_seconds += time.perf_counter() - model_started

            action_started = time.perf_counter()
            command_actor_rows: dict[int, list[int]] = {engine_index: [] for engine_index, _packet in frames}
            command_action_rows: dict[int, list[Tensor]] = {engine_index: [] for engine_index, _packet in frames}
            command_forced_actor_rows: dict[int, list[int]] = {engine_index: [] for engine_index, _packet in frames}
            stored_actions_by_key: dict[str, ActionSequenceV4] = {}
            stored_forced_gate_by_key: dict[str, Tensor] = {}
            host_actions_by_key = _pack_model_actions_to_cpu_v4({
                key: (outputs[key].actions, len(entries)) for key, entries in entries_by_model.items() if entries
            })
            for key, entries in entries_by_model.items():
                if not entries:
                    continue
                stored_actions, packed_actions = host_actions_by_key[key]
                stored_actions_by_key[key] = stored_actions
                forced_rows = set(int(value) for value in forced_rows_by_key.get(key, torch.empty(0, dtype=torch.long)))
                stored_forced_gate = torch.zeros(len(entries), dtype=torch.bool)
                if forced_rows:
                    stored_forced_gate[torch.tensor(sorted(forced_rows), dtype=torch.long)] = True
                stored_forced_gate_by_key[key] = stored_forced_gate
                for row, actor in enumerate(entries):
                    command_actor_rows[actor.engine_index].append(actor.local_actor)
                    command_action_rows[actor.engine_index].append(packed_actions[row])
                    if row in forced_rows:
                        command_forced_actor_rows[actor.engine_index].append(actor.local_actor)
            action_seconds += time.perf_counter() - action_started
            command_started = time.perf_counter()
            for engine_index, actor_rows in command_actor_rows.items():
                packed_rows = command_action_rows[engine_index]
                if not packed_rows:
                    raise RuntimeError("async engine command has no action rows")
                parents[engine_index].send(
                    {
                        "kind": "act_packed",
                        "actor_rows": tuple(actor_rows),
                        "packed_actions": torch.stack(packed_rows),
                        "forced_act_actor_rows": tuple(command_forced_actor_rows[engine_index]),
                    }
                )
                connection = parents[engine_index]
                future = receive_executor.submit(receive, connection)
                receive_futures[future] = connection
            command_seconds += time.perf_counter() - command_started

            # Copy graph-owned scalar outputs before materializing the exact
            # on-policy rows; the next graph replay may reuse those buffers.
            for engine_index, updates in deferred_reward_updates:
                apply_reward_updates(engine_index, updates)
            current_entries = tuple(entries_by_model.get("current", ()))
            if current_entries:
                current_output = outputs.get("current")
                current_actions = stored_actions_by_key.get("current")
                current_forced_gate = stored_forced_gate_by_key.get("current")
                if any(
                    item is None
                    for item in (current_output, current_actions, current_forced_gate, current_storage_packed)
                ):
                    raise RuntimeError("async current-policy storage is incomplete")
                stored_observations = current_storage_packed.record
                stored_packed_observations = current_storage_packed
                stored_log_prob = _stored_tensor(current_output.log_prob[: len(current_entries)], dtype=torch.float32)
                stored_value = _stored_tensor(current_output.value[: len(current_entries)], dtype=torch.float32)
                storage_seconds += store_current_iteration(
                    current_entries,
                    stored_observations,
                    stored_packed_observations,
                    current_actions,
                    current_forced_gate,
                    stored_log_prob,
                    stored_value,
                )
            inference_seconds += time.perf_counter() - inference_started
            packet_count += len(frames)
            if packet_count % progress_packet_interval < len(frames):
                progress_elapsed = time.perf_counter() - started
                print(
                    {
                        "event": "async_cluster_progress",
                        "rank": int(os.environ.get("RANK", "0")),
                        "engine_packets": packet_count,
                        "active_engines": len(pending),
                        "elapsed_seconds": progress_elapsed,
                        "inference_batches": inference_batches,
                        "inference_mean_frame_packets": (inference_frame_packets / max(inference_batches, 1)),
                        "inference_mean_actor_rows": (inference_actor_rows / max(inference_batches, 1)),
                        "ready_wait_seconds": ready_wait_seconds,
                        "receive_seconds": receive_seconds,
                        "receive_pipeline_wait_seconds": (receive_pipeline_wait_seconds),
                        "receive_pipeline_thread_seconds": (receive_pipeline_thread_seconds),
                        "assembly_seconds": assembly_seconds,
                        "model_seconds": model_seconds,
                        "action_seconds": action_seconds,
                        "command_seconds": command_seconds,
                        "storage_seconds": storage_seconds,
                        "packed_relation_capacity": packed_relation_capacity,
                        "packed_effect_capacity": packed_effect_capacity,
                        "packed_schema_expansions": packed_schema_expansions,
                        "packed_storage_schema_fallbacks": 0,
                        "direct_row_packed_batches": direct_row_packed_batches,
                        "direct_row_packed_fallbacks": direct_row_packed_fallbacks,
                    },
                    flush=True,
                )
    finally:
        for connection in parents.values():
            try:
                connection.close()
            except OSError:
                pass
        receive_executor.shutdown(wait=True, cancel_futures=True)
        os.sched_setaffinity(0, collector_affinity)
    if final_segment_collection is None:
        raise RuntimeError("segmented collection produced no rollout")
    return final_segment_collection
