"""Dtype-packed, time-major PPO input cache; no sampling or loss changes."""
from __future__ import annotations

import time
from typing import cast

import torch
from torch import Tensor

from .ppo import StoredRolloutV4
from .remote_worker_transport import ReceivedRowPackedTensorRecord
from .tensors import ActionSequenceV4, UniversalSemanticBatchV4, _map_tensors, concatenate_tensor_records


class PackedRolloutDeviceCacheV4:
    """Keep whole-wave observations on the GPU and gather a chunk per dtype.

    If the byte budget is insufficient, gather from the same packed CPU
    buffers and transfer just the selected chunk. CPU tensors remain owned by
    the rollout. Both paths preserve time-major order and canonical padding.
    """

    def __init__(self, rollout: StoredRolloutV4, device: torch.device, *, budget_bytes: int) -> None:
        started = time.perf_counter()
        source = rollout.packed_observation_rows
        if not isinstance(source, ReceivedRowPackedTensorRecord):
            raise TypeError("device rollout cache requires packed observations")
        if source.batch_size != rollout.time_steps * rollout.batch_size:
            raise ValueError("packed rollout dimensions do not match T*B")
        self.template = source
        self.device = device
        self.lanes = rollout.batch_size
        actions = cast(ActionSequenceV4, concatenate_tensor_records(rollout.actions))
        action_tensors = [getattr(actions, name) for name in (
            'gate', 'micro_action_count', 'candidate_index', 'candidate_uid', 'target_cell', 'delay_offset_bin')]
        required = sum(value.numel() * value.element_size() for value in (*source.buffers.values(), *action_tensors))
        self.resident = device.type == 'cuda' and required <= budget_bytes
        storage_device = device if self.resident else torch.device('cpu')
        self.buffers = {dtype: value.to(storage_device, non_blocking=self.resident)
                        for dtype, value in source.buffers.items()}
        self.actions = cast(ActionSequenceV4, _map_tensors(
            actions, lambda value: value.to(storage_device, non_blocking=self.resident)))
        self.row_numbers = torch.arange(source.batch_size, device=storage_device).reshape(rollout.time_steps, self.lanes)
        if self.resident:
            torch.cuda.current_stream(device).synchronize()
        self.device_bytes = required if self.resident else 0
        self.prefetch_seconds = time.perf_counter() - started

    def chunk(self, start: int, end: int, lanes: Tensor) -> tuple[UniversalSemanticBatchV4, ActionSequenceV4]:
        if lanes.ndim != 1 or lanes.dtype != torch.long or not 0 <= start < end <= self.row_numbers.shape[0]:
            raise ValueError("invalid packed rollout chunk")
        storage_lanes = lanes.to(self.row_numbers.device)
        indices = self.row_numbers[start:end].index_select(1, storage_lanes).reshape(-1)
        selected = {dtype: value.index_select(0, indices) for dtype, value in self.buffers.items()}
        # These local CPU references stay alive through the single wait below.
        model_buffers = {dtype: value.to(self.device,
            dtype=torch.float32 if value.is_floating_point() else value.dtype,
            non_blocking=self.device.type == 'cuda') for dtype, value in selected.items()}
        packed = ReceivedRowPackedTensorRecord._from_buffers(self.template, model_buffers)
        # The legacy concatenation gives each field a dense batch stride.
        # Restore that layout once here; otherwise downstream kernels repeatedly
        # materialize strided views of the much wider dtype-packed row.
        observations = cast(UniversalSemanticBatchV4, _map_tensors(packed.record, lambda value: value.contiguous()))
        selected_actions = cast(ActionSequenceV4, _map_tensors(self.actions, lambda value: value.index_select(0, indices)))
        actions = cast(ActionSequenceV4, _map_tensors(selected_actions,
            lambda value: value.to(self.device, non_blocking=self.device.type == 'cuda')))
        if self.device.type == 'cuda' and not self.resident:
            torch.cuda.current_stream(self.device).synchronize()
        return observations, actions
