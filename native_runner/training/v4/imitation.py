"""Chronological teacher-forced imitation learning for policy V4."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .learning import RecurrentEvaluationV4
from .tensors import ActionSequenceV4, GATE_ACT, UniversalSemanticBatchV4, concatenate_tensor_records


DEFAULT_IL_GATE_ACT_WEIGHT = 8.0
DEFAULT_IL_DELAY_NEIGHBOR_WEIGHT = 0.2
IL_TIME_FIELDS = ("episode_start", "valid_mask", "returns", "gate_loss_mask", "value_loss_mask")


@dataclass(slots=True)
class ILSequenceV4:
    """One natural-time sequence (or a chronological batch of sequences)."""

    observations: tuple[UniversalSemanticBatchV4, ...]
    actions: tuple[ActionSequenceV4, ...]
    episode_start: Tensor
    valid_mask: Tensor
    returns: Tensor
    gate_loss_mask: Tensor | None = None
    value_loss_mask: Tensor | None = None
    sequence_id: str = ""
    natural_time: bool = True

    @property
    def time_steps(self) -> int:
        return len(self.observations)

    @property
    def batch_size(self) -> int:
        return self.observations[0].batch_size if self.observations else 0

    def validate(self) -> None:
        time_steps = self.time_steps
        if time_steps == 0 or len(self.actions) != time_steps:
            raise ValueError("IL observations/actions need the same nonzero length")
        batch_size = self.batch_size
        if any(item.batch_size != batch_size for item in self.observations):
            raise ValueError("IL observation batch size changed within a sequence")
        shape = (time_steps, batch_size)
        for value, label in ((self.episode_start, "episode_start"), (self.valid_mask, "valid_mask")):
            if value.dtype != torch.bool or tuple(value.shape) != shape:
                raise ValueError(f"IL {label} must be bool [T,B]")
        for loss_mask, label in ((self.gate_loss_mask, "gate_loss_mask"), (self.value_loss_mask, "value_loss_mask")):
            if loss_mask is not None:
                if loss_mask.dtype != torch.bool or tuple(loss_mask.shape) != shape:
                    raise ValueError(f"IL {label} must be bool [T,B]")
                if torch.any(loss_mask & ~self.valid_mask):
                    raise ValueError(f"{label} must be a subset of valid frames")
        if tuple(self.returns.shape) != shape or not self.returns.is_floating_point():
            raise ValueError("IL returns must be floating [T,B]")
        if not torch.isfinite(self.returns).all():
            raise ValueError("IL returns must be finite")
        for action in self.actions:
            if tuple(action.gate.shape) != (batch_size,):
                raise ValueError("IL action batch size changed within a sequence")

    def to(self, device: torch.device | str) -> "ILSequenceV4":
        return replace(
            self,
            observations=tuple(item.to_model_input(device) for item in self.observations),
            actions=tuple(item.to(device) for item in self.actions),
            **{name: None if (value := getattr(self, name)) is None else value.to(device) for name in IL_TIME_FIELDS},
        )


def _collate_il_sequences(sequences: Sequence[ILSequenceV4], *, pad: bool) -> ILSequenceV4:
    if not sequences:
        raise ValueError("IL collation needs at least one sequence")
    for sequence in sequences:
        sequence.validate()
    time_steps = max(sequence.time_steps for sequence in sequences)
    if not pad and any(sequence.time_steps != time_steps for sequence in sequences):
        raise ValueError("collated IL sequences must have equal time lengths")

    time_tensors = {}
    for name in IL_TIME_FIELDS:
        values = [getattr(sequence, name) for sequence in sequences]
        if all(value is None for value in values):
            time_tensors[name] = None
            continue
        padded = []
        for sequence, value in zip(sequences, values, strict=True):
            value = sequence.valid_mask if value is None else value
            missing = time_steps - sequence.time_steps
            if missing:
                value = torch.cat((value, value.new_zeros((missing, sequence.batch_size))))
            padded.append(value)
        time_tensors[name] = torch.cat(padded, dim=1)

    records = {
        name: tuple(
            concatenate_tensor_records(
                tuple(getattr(sequence, name)[min(step, sequence.time_steps - 1)] for sequence in sequences)
            )
            for step in range(time_steps)
        )
        for name in ("observations", "actions")
    }
    result = ILSequenceV4(
        **records,
        **time_tensors,
        sequence_id="+".join(sequence.sequence_id or f"sequence-{index}" for index, sequence in enumerate(sequences)),
        natural_time=all(sequence.natural_time for sequence in sequences),
    )
    result.validate()
    return result


def collate_il_sequences(sequences: Sequence[ILSequenceV4]) -> ILSequenceV4:
    """Batch equal-length chronological sequences without shuffling frames."""
    return _collate_il_sequences(sequences, pad=False)


def collate_padded_il_sequences(sequences: Sequence[ILSequenceV4]) -> ILSequenceV4:
    """Repeat final observation/action for padding; clear all padded loss masks."""
    return _collate_il_sequences(sequences, pad=True)


def slice_il_sequence(sequence: ILSequenceV4, start: int, stop: int) -> ILSequenceV4:
    """Take a chronological TBPTT window while preserving every mask."""
    sequence.validate()
    if not 0 <= start < stop <= sequence.time_steps:
        raise ValueError("IL slice must be a non-empty in-range interval")
    return replace(
        sequence,
        observations=sequence.observations[start:stop],
        actions=sequence.actions[start:stop],
        **{name: None if (value := getattr(sequence, name)) is None else value[start:stop] for name in IL_TIME_FIELDS},
        sequence_id=f"{sequence.sequence_id}[{start}:{stop}]",
    )


@dataclass(frozen=True, slots=True)
class ILLossWeightsV4:
    gate: float = 1.0
    candidate: float = 1.0
    target: float = 1.0
    delay: float = 1.0
    continue_action: float = 1.0
    value: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0 for value in (getattr(self, item.name) for item in fields(self))
        ):
            raise ValueError("IL loss weights must be finite and non-negative")


@dataclass(slots=True)
class ILLossV4:
    total: Tensor
    gate: Tensor
    candidate: Tensor
    target: Tensor
    delay: Tensor
    continue_action: Tensor
    value: Tensor
    gate_count: Tensor
    candidate_count: Tensor
    target_count: Tensor
    delay_count: Tensor
    continue_count: Tensor
    value_count: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach().cpu()) for name in (item.name for item in fields(self))}


def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if tuple(values.shape) != tuple(mask.shape):
        raise ValueError("loss values and mask must have the same shape")
    weight = mask.to(values.dtype)
    count = weight.sum()
    mean = (values * weight).sum() / count.clamp_min(1.0)
    return mean, count


def _masked_gate_mean(values: Tensor, mask: Tensor, gates: Tensor, *, act_weight: float) -> tuple[Tensor, Tensor]:
    if tuple(values.shape) != tuple(mask.shape) or tuple(values.shape) != tuple(gates.shape):
        raise ValueError("gate loss values, mask, and labels must have the same shape")
    if not math.isfinite(act_weight) or act_weight <= 0.0:
        raise ValueError("gate ACT weight must be finite and positive")
    valid = mask.to(values.dtype)
    class_weight = torch.where(gates == GATE_ACT, torch.full_like(values, act_weight), torch.ones_like(values))
    count = valid.sum()
    mean = (values * valid * class_weight).sum() / count.clamp_min(1.0)
    return mean, count


def _delay_neighbor_smoothed_nll(
    log_probs: Tensor, legal_mask: Tensor, target_bins: Tensor, *, neighbor_weight: float
) -> Tensor:
    if log_probs.ndim != target_bins.ndim + 1:
        raise ValueError("delay log-probs must add one vocabulary dimension")
    if tuple(log_probs.shape) != tuple(legal_mask.shape):
        raise ValueError("delay log-probs and legality mask must have the same shape")
    if tuple(log_probs.shape[:-1]) != tuple(target_bins.shape):
        raise ValueError("delay labels and log-probs have incompatible shapes")
    if legal_mask.dtype != torch.bool:
        raise TypeError("delay legality mask must be bool")
    if not math.isfinite(neighbor_weight) or neighbor_weight < 0.0 or neighbor_weight > 1.0 / 3.0:
        raise ValueError("delay neighbor weight must be finite and in [0, 1/3]")

    vocabulary = torch.arange(log_probs.shape[-1], device=target_bins.device)
    distance = (vocabulary - target_bins.clamp_min(0)[..., None]).abs()
    center_weight = 1.0 - 2.0 * neighbor_weight
    target_weights = (distance == 0).to(log_probs.dtype) * center_weight
    target_weights = target_weights + (
        (distance == 1).to(log_probs.dtype) * legal_mask.to(log_probs.dtype) * neighbor_weight
    )
    target_weights = target_weights / target_weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(log_probs.dtype).eps
    )
    return -(target_weights * log_probs).sum(dim=-1)


def imitation_loss(
    sequence: ILSequenceV4,
    evaluation: RecurrentEvaluationV4,
    *,
    weights: ILLossWeightsV4 | None = None,
    value_loss: str = "huber",
    gate_act_weight: float = DEFAULT_IL_GATE_ACT_WEIGHT,
    delay_neighbor_weight: float = DEFAULT_IL_DELAY_NEIGHBOR_WEIGHT,
) -> ILLossV4:
    """Normalize every action head by its own valid expert population."""

    sequence.validate()
    actual_weights = weights or ILLossWeightsV4()
    time_steps, batch_size = sequence.valid_mask.shape
    if tuple(evaluation.value.shape) != (time_steps, batch_size):
        raise ValueError("IL evaluation has the wrong [T,B] shape")
    components = evaluation.components
    max_micro_actions = sequence.actions[0].candidate_index.shape[1]
    micro_counts = torch.stack([item.micro_action_count for item in sequence.actions])
    micro_mask = (
        torch.arange(max_micro_actions, device=micro_counts.device)[None, None, :] < micro_counts[:, :, None]
    ) & sequence.valid_mask[:, :, None]
    target_cells = torch.stack([item.target_cell for item in sequence.actions])
    delay_bins = torch.stack([item.delay_offset_bin for item in sequence.actions])
    grid_mask = micro_mask & (target_cells >= 0)
    gates = torch.stack([item.gate for item in sequence.actions])
    act_mask = sequence.valid_mask & (gates == GATE_ACT)
    gate_mask = sequence.valid_mask if sequence.gate_loss_mask is None else sequence.gate_loss_mask

    gate, gate_count = _masked_gate_mean(-components.gate_log_prob, gate_mask, gates, act_weight=gate_act_weight)
    candidate, candidate_count = _masked_mean(-components.candidate_log_prob, micro_mask)
    target, target_count = _masked_mean(-components.target_log_prob, grid_mask)
    delay, delay_count = _masked_mean(
        _delay_neighbor_smoothed_nll(
            components.delay_offset_log_probs,
            components.delay_offset_legal_mask,
            delay_bins,
            neighbor_weight=delay_neighbor_weight,
        ),
        micro_mask,
    )
    continuation, continue_count = _masked_mean(-components.continue_log_prob, act_mask)
    if value_loss == "huber":
        raw_value_loss = F.smooth_l1_loss(
            evaluation.value, sequence.returns.to(evaluation.value.dtype), reduction="none"
        )
    elif value_loss == "mse":
        raw_value_loss = (evaluation.value - sequence.returns.to(evaluation.value.dtype)).square()
    else:
        raise ValueError("value_loss must be 'huber' or 'mse'")
    value, value_count = _masked_mean(
        raw_value_loss, (sequence.valid_mask if sequence.value_loss_mask is None else sequence.value_loss_mask)
    )

    total = (
        actual_weights.gate * gate
        + actual_weights.candidate * candidate
        + actual_weights.target * target
        + actual_weights.delay * delay
        + actual_weights.continue_action * continuation
        + actual_weights.value * value
    )
    return ILLossV4(
        total=total,
        gate=gate,
        candidate=candidate,
        target=target,
        delay=delay,
        continue_action=continuation,
        value=value,
        gate_count=gate_count,
        candidate_count=candidate_count,
        target_count=target_count,
        delay_count=delay_count,
        continue_count=continue_count,
        value_count=value_count,
    )
