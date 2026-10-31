"""Score V4 imitation checkpoints on a held-out replay split with stateful replay.

Evaluation mirrors ``train_imitation_cache``: every selected (replay, owner)
lane is replayed once from its first frame, in ``time_steps`` chunks, carrying
the LSTM state between chunks.  Every head uses the exact population mask that
``imitation_loss`` uses, but without the gate ACT class weight and without the
delay neighbor smoothing, so the reported NLLs are plain held-out likelihoods.

Shards are loaded once per batch group.  Each chunk is collated once and then
scored by every checkpoint (weights are swapped in place on one model), each
checkpoint carrying its own recurrent state.  Under ``torch.distributed.run``
ranks split shards by owner frames and all-reduce float64 sums; rank 0 writes.

Illegal labels.  A supervised label that the checkpoint's own legality mask
rules out has (numerically) zero probability: its masked logit is
``finfo(float32).min``, so its log-prob is about -3.4e38 and a handful of such
frames dominate the head mean (a decision-tick-legality checkpoint scored on a
touch-time-legality cache reports NLLs near 1e37).  ``illegal_label_policy``
selects how they are treated, per head (gate, candidate, target, delay,
continue):

* ``skip`` (default): the frame is left out of that head's NLL/accuracy sums
  and counts.  With no illegal label the statistics are unchanged.
* ``count``: the historical arithmetic (the label's huge or infinite NLL is
  summed), for reproducing earlier rows.
* ``error``: raise, listing the per-head counts.

Every policy reports ``<head>_illegal_label_count`` and
``<head>_illegal_label_frac`` (illegal labels over the head's supervised
labels before any exclusion).  A head whose NLL sum is non-finite is reported
as ``Infinity``/``NaN`` rather than failing the whole run.

Sharpness (opt-in, ``--report-entropy``).  Each row additionally carries, per
head (gate, candidate, target, delay, continue) and over exactly that head's
NLL population: the mean entropy of the head's distribution
(``<head>_entropy``; the target head is the masked spatial distribution over
grid cells), the mean probability of its most likely choice
(``<head>_top1_prob``) and the share of frames where that probability exceeds
0.9 (``<head>_top1_gt09_frac``); plus ``gate_entropy_act`` /
``gate_entropy_wait`` on expert ACT / WAIT frames.  Distributions are the
teacher-forced ones the NLLs score (the candidate/target/delay of a second
micro-action condition on the expert's first).  The top-1 probabilities are
read by temporarily wrapping the model module's ``_distribution`` factory
(the model file is unchanged); the recorded distributions are aligned with
the returned components and checked against their entropies.  Off by
default, and the default rows are unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F

from . import model as _model_module
from .cache import SHARD_LOAD_MODES
from .checkpoint import model_config_from_contract, read_actor_critic_checkpoint, restore_actor_critic_checkpoint
from .config import ModelConfigV4
from .factory import build_production_model_v4
from .imitation import ILSequenceV4
from .learning import RecurrentEvaluationV4, evaluate_recurrent_sequence
from .model import UniversalCardPolicyV4
from .tensors import GATE_ACT, GATE_WAIT, RecurrentPolicyStateV4
from .train_imitation_cache import (
    ShardTrainingPlanV1,
    _autocast_dtype,
    _broadcast_training_plan,
    _group_chunks,
    _GroupShardLoader,
    build_training_plan,
    lane_slices,
    sequence_plan_options,
    slice_sequence_lanes,
)


EVALUATION_SCHEMA = "v4-il-heldout-evaluation.v1"

ILLEGAL_LABEL_POLICIES = ("error", "skip", "count")
# Heads whose supervised label can be ruled out by the model's legality mask.
_LEGALITY_HEADS = ("gate", "candidate", "target", "delay", "continue")
# ``masked_logits`` fills illegal entries with ``finfo(float32).min`` (every
# categorical is normalized in fp32), so an illegal label's log-prob is about
# -3.4e38.  A legal label cannot come within orders of magnitude of half that.
_ILLEGAL_LOG_PROB = 0.5 * torch.finfo(torch.float32).min

# Order of the float64 statistic vector accumulated per checkpoint.
_SUM_FIELDS = (
    "owner_frames",
    "valid_frames",
    "gate_count",
    "gate_act_count",
    "gate_wait_count",
    "gate_nll_all_sum",
    "gate_nll_act_sum",
    "gate_nll_wait_sum",
    "gate_model_act_sum",
    "gate_act_correct_sum",
    "gate_wait_correct_sum",
    "candidate_count",
    "candidate_nll_sum",
    "candidate_top1_sum",
    "target_count",
    "target_nll_sum",
    "target_exact_sum",
    "target_within1_sum",
    "delay_count",
    "delay_nll_sum",
    "delay_top1_sum",
    "continue_count",
    "continue_nll_sum",
    "continue_top1_sum",
    "value_count",
    "value_huber_sum",
    "dropped_legacy_expert_frames",
    *(f"{head}_illegal_label_count" for head in _LEGALITY_HEADS),
)
_SUM_INDEX = {name: index for index, name in enumerate(_SUM_FIELDS)}

# (metric name, numerator, denominator)
_MEAN_METRICS = (
    ("gate_nll_all", "gate_nll_all_sum", "gate_count"),
    ("gate_nll_act", "gate_nll_act_sum", "gate_act_count"),
    ("gate_nll_wait", "gate_nll_wait_sum", "gate_wait_count"),
    ("gate_act_rate_expert", "gate_act_count", "gate_count"),
    ("gate_act_rate_model", "gate_model_act_sum", "gate_count"),
    ("gate_act_top1", "gate_act_correct_sum", "gate_act_count"),
    ("gate_wait_top1", "gate_wait_correct_sum", "gate_wait_count"),
    ("candidate_nll", "candidate_nll_sum", "candidate_count"),
    ("candidate_top1", "candidate_top1_sum", "candidate_count"),
    ("target_nll", "target_nll_sum", "target_count"),
    ("target_exact", "target_exact_sum", "target_count"),
    ("target_within1", "target_within1_sum", "target_count"),
    ("delay_nll", "delay_nll_sum", "delay_count"),
    ("delay_top1", "delay_top1_sum", "delay_count"),
    ("continue_nll", "continue_nll_sum", "continue_count"),
    ("continue_top1", "continue_top1_sum", "continue_count"),
    ("value_huber", "value_huber_sum", "value_count"),
)
_COUNT_FIELDS = (
    "owner_frames",
    "valid_frames",
    "gate_count",
    "gate_act_count",
    "gate_wait_count",
    "candidate_count",
    "target_count",
    "delay_count",
    "continue_count",
    "value_count",
    "dropped_legacy_expert_frames",
    *(f"{head}_illegal_label_count" for head in _LEGALITY_HEADS),
)
# Head population count each illegal-label count belongs to.
_LEGALITY_HEAD_COUNTS = {
    "gate": "gate_count",
    "candidate": "candidate_count",
    "target": "target_count",
    "delay": "delay_count",
    "continue": "continue_count",
}

# Opt-in sharpness report (``report_entropy``): float64 sums per checkpoint.
ENTROPY_HEADS = ("gate", "candidate", "target", "delay", "continue")
TOP1_CONFIDENT_PROB = 0.9
_ENTROPY_SUM_FIELDS = (
    *(f"{head}_{stat}" for head in ENTROPY_HEADS for stat in ("entropy_sum", "top1_prob_sum", "top1_gt09_sum")),
    "gate_entropy_act_sum",
    "gate_entropy_wait_sum",
)
_ENTROPY_SUM_INDEX = {name: index for index, name in enumerate(_ENTROPY_SUM_FIELDS)}
# Head population each sharpness mean divides by (the head's NLL population).
_ENTROPY_POPULATIONS = {
    "gate": "gate_count",
    "candidate": "candidate_count",
    "target": "target_count",
    "delay": "delay_count",
    "continue": "continue_count",
    "gate_act": "gate_act_count",
    "gate_wait": "gate_wait_count",
}
# Categoricals one sparse forced evaluation builds: the all-row gate, then on
# the ACT rows ``_decode``'s gate, (candidate, target, delay) of micro-action 0,
# continue, and (candidate, target, delay) of micro-action 1.
_DECODE_DISTRIBUTIONS = 8


@dataclass(frozen=True, slots=True)
class Top1LogProbsV1:
    """Log-probability of each head's most likely choice, shaped like the components."""

    gate: Tensor
    candidate: Tensor
    target: Tensor
    delay: Tensor
    continue_: Tensor


class DistributionRecorderV1:
    """Record ``(max log-prob, entropy)`` of every categorical the model builds.

    Wraps ``native_runner.training.v4.model._distribution`` for the duration of
    the ``with`` block and restores it afterwards; the returned distributions
    are the original objects, so the evaluation itself is unchanged.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Tensor, Tensor]] = []
        self._original: object | None = None

    def __enter__(self) -> "DistributionRecorderV1":
        original = _model_module._distribution
        if getattr(original, "_distribution_recorder_v1", False):
            raise RuntimeError("distribution recorders cannot be nested")
        calls = self.calls

        def recording(logits: Tensor, *, mask: Tensor | None, temperature: float, validate: bool):  # type: ignore[no-untyped-def]
            distribution = original(logits, mask=mask, temperature=temperature, validate=validate)
            # Categorical.logits are normalized log-probs; masked entries sit at finfo.min.
            calls.append((distribution.logits.max(dim=-1).values.detach(), distribution.entropy().detach()))
            return distribution

        recording._distribution_recorder_v1 = True  # type: ignore[attr-defined]
        self._original = original
        _model_module._distribution = recording  # type: ignore[assignment]
        return self

    def __exit__(self, *exc_info: object) -> None:
        _model_module._distribution = self._original  # type: ignore[assignment]


def _entropies_match(recorded: Tensor, expected: Tensor) -> bool:
    return tuple(recorded.shape) == tuple(expected.shape) and bool(
        torch.allclose(recorded.float(), expected.float(), rtol=1e-4, atol=1e-5)
    )


def top1_log_probs(
    calls: Sequence[tuple[Tensor, Tensor]], sequence: ILSequenceV4, evaluation: RecurrentEvaluationV4
) -> Top1LogProbsV1:
    """Align one fused evaluation's recorded categoricals with its ``[T,B]`` components.

    The all-row gate maps one-to-one onto the time-major ``T*B`` rows; the ACT
    rows' decoder distributions map onto the expert-ACT rows the sparse
    evaluator decoded (restricted to ``valid_mask`` under compact-tail
    decoding), identified by matching the recorded entropies with the
    returned components.  Raises when the layout is not recognized.
    """

    components = evaluation.components
    if components.gate_entropy.dim() != 2:
        raise ValueError("top-1 alignment needs [T,B] components")
    time_steps, batch_size = (int(value) for value in components.gate_entropy.shape)
    rows = time_steps * batch_size
    micro = int(components.candidate_entropy.shape[-1])
    if micro != 2:
        raise ValueError("top-1 alignment assumes two micro-actions")
    if not calls:
        raise RuntimeError("the evaluation built no categorical distribution")
    gate_top1, gate_entropy = calls[0]
    if tuple(gate_top1.shape) != (rows,) or not _entropies_match(gate_entropy, components.gate_entropy.flatten()):
        raise RuntimeError("recorded gate distribution does not match the evaluated components")
    device = gate_top1.device
    candidate = torch.zeros(rows, micro, dtype=torch.float32, device=device)
    target = torch.zeros_like(candidate)
    delay = torch.zeros_like(candidate)
    continue_ = torch.zeros(rows, dtype=torch.float32, device=device)
    if len(calls) == 1 + _DECODE_DISTRIBUTIONS:
        decoded_rows = int(calls[2][0].shape[0])
        act_rows = torch.stack([item.gate for item in sequence.actions]).to(device).flatten() == GATE_ACT
        valid_rows = sequence.valid_mask.to(device).flatten()
        first_candidate_entropy = components.candidate_entropy.flatten(0, 1)[:, 0]
        indices = None
        for row_mask in (act_rows & valid_rows, act_rows):
            option = row_mask.nonzero(as_tuple=False).flatten()
            if option.numel() == decoded_rows and _entropies_match(calls[2][1], first_candidate_entropy[option]):
                indices = option
                break
        if indices is None:
            raise RuntimeError("cannot align the decoded ACT rows with the evaluated components")
        for step, (candidate_call, target_call, delay_call) in enumerate(((2, 3, 4), (6, 7, 8))):
            candidate[indices, step] = calls[candidate_call][0].float()
            target[indices, step] = calls[target_call][0].float()
            delay[indices, step] = calls[delay_call][0].float()
        continue_[indices] = calls[5][0].float()
    elif len(calls) != 1:
        raise RuntimeError(
            f"the evaluation built {len(calls)} categoricals; expected 1 (no ACT row) or {1 + _DECODE_DISTRIBUTIONS}"
        )
    shape = (time_steps, batch_size)
    return Top1LogProbsV1(
        gate=gate_top1.float().reshape(shape),
        candidate=candidate.reshape(*shape, micro),
        target=target.reshape(*shape, micro),
        delay=delay.reshape(*shape, micro),
        continue_=continue_.reshape(shape),
    )


@dataclass(frozen=True, slots=True)
class CheckpointInfoV1:
    path: Path
    checkpoint_id: str
    update_step: int
    training_stage: str
    next_epoch: int | None


@dataclass(frozen=True, slots=True)
class HeadMasksV1:
    """The exact ``imitation_loss`` populations for one ``[T,B]`` chunk."""

    gate: Tensor
    gate_act: Tensor
    gate_wait: Tensor
    micro: Tensor
    grid: Tensor
    act: Tensor
    value: Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_partitions(
    plans: Sequence[ShardTrainingPlanV1], world_size: int
) -> tuple[tuple[ShardTrainingPlanV1, ...], ...]:
    """Greedy longest-first assignment of whole shards to ranks by owner frames."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    bins: list[list[ShardTrainingPlanV1]] = [[] for _ in range(world_size)]
    frames = [0] * world_size
    for plan in sorted(plans, key=lambda item: (-item.owner_frames, str(item.summary_path))):
        target = min(range(world_size), key=lambda rank: (frames[rank], rank))
        bins[target].append(plan)
        frames[target] += plan.owner_frames
    return tuple(tuple(items) for items in bins)


def _local_groups(
    plans: Sequence[ShardTrainingPlanV1], shards_per_batch: int
) -> tuple[tuple[ShardTrainingPlanV1, ...], ...]:
    # Recurrent chunks run to the longest lane in a group, so group shards of
    # similar length.  The order is deterministic; it does not change results.
    ordered = sorted(plans, key=lambda plan: (max(plan.sequence_lengths), str(plan.summary_path)))
    return tuple(
        tuple(ordered[start : start + shards_per_batch]) for start in range(0, len(ordered), shards_per_batch)
    )


def head_masks(sequence: ILSequenceV4) -> HeadMasksV1:
    """Rebuild the per-head masks used by ``imitation_loss``."""

    max_micro_actions = sequence.actions[0].candidate_index.shape[1]
    micro_counts = torch.stack([item.micro_action_count for item in sequence.actions])
    micro = (
        torch.arange(max_micro_actions, device=micro_counts.device)[None, None, :] < micro_counts[:, :, None]
    ) & sequence.valid_mask[:, :, None]
    target_cells = torch.stack([item.target_cell for item in sequence.actions])
    gates = torch.stack([item.gate for item in sequence.actions])
    gate = sequence.valid_mask if sequence.gate_loss_mask is None else sequence.gate_loss_mask
    return HeadMasksV1(
        gate=gate,
        gate_act=gate & (gates == GATE_ACT),
        gate_wait=gate & (gates == GATE_WAIT),
        micro=micro,
        grid=micro & (target_cells >= 0),
        act=sequence.valid_mask & (gates == GATE_ACT),
        value=sequence.valid_mask if sequence.value_loss_mask is None else sequence.value_loss_mask,
    )


def _masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    if tuple(values.shape) != tuple(mask.shape):
        raise ValueError("statistic values and mask must have the same shape")
    # ``where`` (not multiplication) so unselected non-finite entries cannot leak.
    return torch.where(mask, values.to(torch.float64), torch.zeros((), dtype=torch.float64, device=values.device)).sum()


def _count(mask: Tensor) -> Tensor:
    return mask.sum(dtype=torch.float64)


def _check_illegal_label_policy(policy: str) -> None:
    if policy not in ILLEGAL_LABEL_POLICIES:
        raise ValueError(f"illegal_label_policy must be one of {ILLEGAL_LABEL_POLICIES}")


def _head_populations(
    sequence: ILSequenceV4, evaluation: RecurrentEvaluationV4, masks: HeadMasksV1, illegal_label_policy: str
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return ``(illegal labels per head, NLL population per head)``.

    Population keys: gate, gate_act, gate_wait, candidate, target, delay, continue.
    """

    components = evaluation.components
    expert_delay = torch.stack([item.delay_offset_bin for item in sequence.actions])
    # Labels the model's own legality mask rules out, per head.
    expert_delay_legal = components.delay_offset_legal_mask.gather(-1, expert_delay.clamp_min(0)[..., None])[..., 0]
    illegal = {
        "gate": masks.gate_act & (components.gate_log_prob.float() <= _ILLEGAL_LOG_PROB),
        "candidate": masks.micro & (components.candidate_log_prob.float() <= _ILLEGAL_LOG_PROB),
        "target": masks.grid & (components.target_log_prob.float() <= _ILLEGAL_LOG_PROB),
        "delay": masks.micro & ~expert_delay_legal,
        "continue": masks.act & (components.continue_log_prob.float() <= _ILLEGAL_LOG_PROB),
    }
    if illegal_label_policy == "count":
        populations = {
            "gate": masks.gate,
            "gate_act": masks.gate_act,
            "candidate": masks.micro,
            "target": masks.grid,
            "delay": masks.micro,
            "continue": masks.act,
        }
    else:
        # WAIT is always legal, so only ACT gate labels leave the gate population.
        populations = {
            "gate": masks.gate & ~illegal["gate"],
            "gate_act": masks.gate_act & ~illegal["gate"],
            "candidate": masks.micro & ~illegal["candidate"],
            "target": masks.grid & ~illegal["target"],
            "delay": masks.micro & ~illegal["delay"],
            "continue": masks.act & ~illegal["continue"],
        }
    populations["gate_wait"] = masks.gate_wait
    return illegal, populations


def entropy_statistics(
    sequence: ILSequenceV4,
    evaluation: RecurrentEvaluationV4,
    top1: Top1LogProbsV1,
    *,
    illegal_label_policy: str = "skip",
) -> Tensor:
    """Return the float64 ``_ENTROPY_SUM_FIELDS`` vector for one evaluated chunk.

    Every head uses the same population as its NLL in ``chunk_statistics``.
    """

    _check_illegal_label_policy(illegal_label_policy)
    masks = head_masks(sequence)
    _illegal, populations = _head_populations(sequence, evaluation, masks, illegal_label_policy)
    components = evaluation.components
    per_head = {
        "gate": (components.gate_entropy, top1.gate),
        "candidate": (components.candidate_entropy, top1.candidate),
        "target": (components.target_entropy, top1.target),
        "delay": (components.delay_offset_entropy, top1.delay),
        "continue": (components.continue_entropy, top1.continue_),
    }
    values: dict[str, Tensor] = {}
    for head in ENTROPY_HEADS:
        entropy, log_top1 = per_head[head]
        population = populations[head]
        top1_prob = log_top1.float().exp()
        values[f"{head}_entropy_sum"] = _masked_sum(entropy.float(), population)
        values[f"{head}_top1_prob_sum"] = _masked_sum(top1_prob, population)
        values[f"{head}_top1_gt09_sum"] = _count(population & (top1_prob > TOP1_CONFIDENT_PROB))
    values["gate_entropy_act_sum"] = _masked_sum(components.gate_entropy.float(), populations["gate_act"])
    values["gate_entropy_wait_sum"] = _masked_sum(components.gate_entropy.float(), populations["gate_wait"])
    if tuple(values) != _ENTROPY_SUM_FIELDS:
        raise RuntimeError("entropy statistic vector order drifted from _ENTROPY_SUM_FIELDS")
    return torch.stack(tuple(values.values()))


def summarize_entropy_statistics(sums: Sequence[float], counts: Mapping[str, object]) -> dict[str, object]:
    """Means of a reduced ``_ENTROPY_SUM_FIELDS`` vector over the heads' NLL populations."""

    if len(sums) != len(_ENTROPY_SUM_FIELDS):
        raise ValueError("entropy statistic vector has the wrong length")

    def mean(name: str, population: str) -> float | None:
        count = float(counts[_ENTROPY_POPULATIONS[population]])  # type: ignore[arg-type]
        return sums[_ENTROPY_SUM_INDEX[name]] / count if count > 0 else None

    result: dict[str, object] = {}
    for head in ENTROPY_HEADS:
        result[f"{head}_entropy"] = mean(f"{head}_entropy_sum", head)
        result[f"{head}_top1_prob"] = mean(f"{head}_top1_prob_sum", head)
        result[f"{head}_top1_gt09_frac"] = mean(f"{head}_top1_gt09_sum", head)
    result["gate_entropy_act"] = mean("gate_entropy_act_sum", "gate_act")
    result["gate_entropy_wait"] = mean("gate_entropy_wait_sum", "gate_wait")
    return result


def chunk_statistics(
    sequence: ILSequenceV4,
    evaluation: RecurrentEvaluationV4,
    *,
    board_width: int,
    owner_frames: int,
    illegal_label_policy: str = "skip",
) -> Tensor:
    """Return the float64 ``_SUM_FIELDS`` vector for one evaluated chunk.

    A label is illegal for a head when the model's own mask excluded it: its
    log-prob is the ``masked_logits`` sentinel (gate ACT with no legal
    candidate, candidate, target cell, continue=SECOND with no remaining
    candidate), or, for the delay head, its bin is outside
    ``delay_offset_legal_mask``.  Except under ``count`` such frames leave the
    head's population (``error`` is raised by the caller from the counts).
    """

    if evaluation.argmax is None:
        raise ValueError("held-out evaluation needs argmax choices (collect_argmax=True)")
    if board_width <= 0:
        raise ValueError("board_width must be positive")
    _check_illegal_label_policy(illegal_label_policy)
    masks = head_masks(sequence)
    components = evaluation.components
    argmax = evaluation.argmax
    device = sequence.valid_mask.device
    micro_counts = torch.stack([item.micro_action_count for item in sequence.actions])
    expert_uid = torch.stack([item.candidate_uid for item in sequence.actions])
    expert_cell = torch.stack([item.target_cell for item in sequence.actions])
    expert_delay = torch.stack([item.delay_offset_bin for item in sequence.actions])
    candidate_uid = torch.stack([item.candidates.uid for item in sequence.observations])

    # Gate.  Under a forced gate=ACT prefix the model's own gate choice is its argmax.
    gate_nll = -components.gate_log_prob.float()
    model_act = argmax.gate_argmax == GATE_ACT

    # Candidate: compare stable UIDs, never padded positions.
    candidate_nll = -components.candidate_log_prob.float()
    predicted_uid = candidate_uid.gather(-1, argmax.candidate_argmax.clamp_min(0))
    candidate_hit = predicted_uid == expert_uid

    # Target cell on the 18-wide row-major grid (cell = y * W + x).
    target_nll = -components.target_log_prob.float()
    predicted_cell = argmax.target_argmax.clamp_min(0)
    safe_cell = expert_cell.clamp_min(0)
    chebyshev = torch.maximum(
        (predicted_cell.remainder(board_width) - safe_cell.remainder(board_width)).abs(),
        (
            torch.div(predicted_cell, board_width, rounding_mode="floor")
            - torch.div(safe_cell, board_width, rounding_mode="floor")
        ).abs(),
    )

    # Delay: plain NLL of the expert bin after renormalizing over legal bins.
    delay_log_probs = components.delay_offset_log_probs.float().masked_fill(
        ~components.delay_offset_legal_mask, float("-inf")
    )
    delay_log_probs = torch.log_softmax(delay_log_probs, dim=-1)
    delay_nll = -delay_log_probs.gather(-1, expert_delay.clamp_min(0)[..., None])[..., 0]

    # Continue: expert continues exactly when it played two micro-actions.
    continue_nll = -components.continue_log_prob.float()
    continue_hit = argmax.continue_argmax == (micro_counts == 2).to(torch.long)

    value_huber = F.smooth_l1_loss(
        evaluation.value.float(), sequence.returns.to(device=device, dtype=torch.float32), reduction="none"
    )

    illegal, populations = _head_populations(sequence, evaluation, masks, illegal_label_policy)
    gate_mask, gate_act_mask = populations["gate"], populations["gate_act"]
    candidate_mask, target_mask = populations["candidate"], populations["target"]
    delay_mask, continue_mask = populations["delay"], populations["continue"]

    values = {
        "owner_frames": torch.tensor(float(owner_frames), dtype=torch.float64, device=device),
        "valid_frames": _count(sequence.valid_mask),
        "gate_count": _count(gate_mask),
        "gate_act_count": _count(gate_act_mask),
        "gate_wait_count": _count(masks.gate_wait),
        "gate_nll_all_sum": _masked_sum(gate_nll, gate_mask),
        "gate_nll_act_sum": _masked_sum(gate_nll, gate_act_mask),
        "gate_nll_wait_sum": _masked_sum(gate_nll, masks.gate_wait),
        "gate_model_act_sum": _count(gate_mask & model_act),
        "gate_act_correct_sum": _count(gate_act_mask & model_act),
        "gate_wait_correct_sum": _count(masks.gate_wait & ~model_act),
        "candidate_count": _count(candidate_mask),
        "candidate_nll_sum": _masked_sum(candidate_nll, candidate_mask),
        "candidate_top1_sum": _count(candidate_mask & candidate_hit),
        "target_count": _count(target_mask),
        "target_nll_sum": _masked_sum(target_nll, target_mask),
        "target_exact_sum": _count(target_mask & (predicted_cell == expert_cell)),
        "target_within1_sum": _count(target_mask & (chebyshev <= 1)),
        "delay_count": _count(delay_mask),
        "delay_nll_sum": _masked_sum(delay_nll, delay_mask),
        "delay_top1_sum": _count(delay_mask & (argmax.delay_argmax == expert_delay)),
        "continue_count": _count(continue_mask),
        "continue_nll_sum": _masked_sum(continue_nll, continue_mask),
        "continue_top1_sum": _count(continue_mask & continue_hit),
        "value_count": _count(masks.value),
        "value_huber_sum": _masked_sum(value_huber, masks.value),
        "dropped_legacy_expert_frames": torch.zeros((), dtype=torch.float64, device=device),
        **{f"{head}_illegal_label_count": _count(illegal[head]) for head in _LEGALITY_HEADS},
    }
    if tuple(values) != _SUM_FIELDS:
        raise RuntimeError("statistic vector order drifted from _SUM_FIELDS")
    return torch.stack(tuple(values.values()))


def summarize_statistics(sums: Sequence[float], *, illegal_label_policy: str = "skip") -> dict[str, object]:
    """Convert a reduced ``_SUM_FIELDS`` vector into counts and means.

    Counts must be finite.  A non-finite NLL/Huber sum makes that metric
    ``inf``/``nan`` (``Infinity``/``NaN`` in JSON) instead of failing the row.
    ``<head>_illegal_label_frac`` divides by the head's supervised labels
    before exclusion: its count plus the illegal labels, except under
    ``count``, where the illegal labels are already in the count.
    """

    if len(sums) != len(_SUM_FIELDS):
        raise ValueError("statistic vector has the wrong length")
    _check_illegal_label_policy(illegal_label_policy)
    if not all(math.isfinite(sums[_SUM_INDEX[name]]) for name in _COUNT_FIELDS):
        raise FloatingPointError("held-out evaluation produced a non-finite count")
    result: dict[str, object] = {name: int(round(sums[_SUM_INDEX[name]])) for name in _COUNT_FIELDS}
    for name, numerator, denominator in _MEAN_METRICS:
        count = sums[_SUM_INDEX[denominator]]
        result[name] = sums[_SUM_INDEX[numerator]] / count if count > 0 else None
    for head in _LEGALITY_HEADS:
        illegal_count = int(result[f"{head}_illegal_label_count"])  # type: ignore[call-overload]
        population = int(result[_LEGALITY_HEAD_COUNTS[head]])  # type: ignore[call-overload]
        if illegal_label_policy != "count":
            population += illegal_count
        result[f"{head}_illegal_label_frac"] = illegal_count / population if population > 0 else None
    candidate_nll, target_nll = result["candidate_nll"], result["target_nll"]
    result["decision_nll"] = (
        None
        if candidate_nll is None or target_nll is None
        else float(candidate_nll) + float(target_nll)  # type: ignore[arg-type]
    )
    return result


def _checkpoint_info(path: Path, payload: Mapping[str, object]) -> CheckpointInfoV1:
    extra = payload.get("extra")
    next_epoch = extra.get("next_epoch") if isinstance(extra, Mapping) else None
    return CheckpointInfoV1(
        path=path,
        checkpoint_id=str(payload["checkpoint_id"]),
        update_step=int(payload["update_step"]),  # type: ignore[arg-type]
        training_stage=str(payload["training_stage"]),
        next_epoch=None if next_epoch is None else int(next_epoch),
    )


def _model_for_config(
    config: ModelConfigV4, *, template: UniversalCardPolicyV4 | None, device: torch.device
) -> UniversalCardPolicyV4:
    """Build the evaluation model for one checkpoint architecture.

    Without a template this is the production model; an injected template
    (tests) is reused when its config matches and otherwise rebuilt on its own
    catalogs with the checkpoint's config.
    """

    if template is None:
        return build_production_model_v4(device=device, config=config)
    if template.config == config:
        return template.to(device)
    return UniversalCardPolicyV4(
        template.catalog,
        ability_catalog=template.ability_catalog,
        entity_archetype_catalog=template.entity_archetype_catalog,
        effect_catalog=template.effect_catalog,
        mechanic_profile_catalog=template.mechanic_profile_catalog,
        config=config,
    ).to(device)


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _write_json_line(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True) + "\n")


def evaluate_checkpoints(
    index_paths: Sequence[Path],
    tags_path: Path,
    checkpoint_paths: Sequence[Path],
    *,
    device: torch.device | str,
    time_steps: int = 32,
    shards_per_batch: int = 8,
    load_workers: int = 6,
    manifest_workers: int = 16,
    autocast: str = "bfloat16",
    validate: bool = False,
    verify_cache: bool = True,
    output_path: Path | None = None,
    model: UniversalCardPolicyV4 | None = None,
    progress: bool = False,
    fused_sequence: bool = True,
    micro_batches: int = 1,
    preload_next_group: bool = True,
    preload_workers_mode: str = "thread",
    prefetch_chunks: int = 2,
    illegal_label_policy: str = "skip",
    trim_padding: bool = False,
    compact_tail: bool = False,
    report_entropy: bool = False,
) -> list[dict[str, object]]:
    """Evaluate every checkpoint on the replay tags; return one row per checkpoint.

    ``trim_padding`` and ``compact_tail`` are the trainer's chunk speed-ups
    (padded collection rows cut per chunk; repeated tail frames encoded once
    and only valid rows decoded); every statistic is masked by ``valid_mask``
    subsets, so rows are unchanged up to float rounding.

    ``micro_batches`` scores each chunk's lanes in that many contiguous
    slices (each carrying its own lanes' recurrent state); the statistics
    are sums, so the rows are unchanged while the peak working set of one
    forward shrinks with the slice.

    Every checkpoint is scored with the ``ModelConfigV4`` stored in its
    contract (missing keys take current defaults).  ``model`` defaults to
    ``build_production_model_v4(device=device, config=...)``; tests pass a
    small-catalog model, which is rebuilt on its own catalogs whenever a
    checkpoint's config differs from it.  When ``torch.distributed`` is initialized the
    rank-0 plan is broadcast, shards are partitioned by owner frames, and all
    sums are all-reduced; only rank 0 appends to ``output_path`` and prints.

    As in training, the next group's shards are read while the current group
    is scored (``preload_next_group``; decoded by threads, or by worker
    processes for ``preload_workers_mode="process"``) and up to ``prefetch_chunks`` chunks are
    gathered and copied ahead on a background thread, so the short forward
    passes do not wait on cache I/O.  Rows are unchanged.

    ``illegal_label_policy`` (``skip``/``count``/``error``) handles supervised
    labels the checkpoint's own legality mask rules out (see the module
    docstring); every row reports the per-head illegal-label counts.
    ``validate=True`` rejects such labels inside the model before any policy
    applies.

    ``report_entropy`` adds the per-head sharpness fields (module docstring);
    it needs ``fused_sequence``.
    """

    if time_steps <= 0 or shards_per_batch <= 0 or load_workers <= 0 or manifest_workers <= 0 or micro_batches <= 0:
        raise ValueError(
            "time_steps, shards_per_batch, load_workers, manifest_workers, and micro_batches must be positive"
        )
    if prefetch_chunks < 0:
        raise ValueError("prefetch_chunks must be non-negative")
    _check_illegal_label_policy(illegal_label_policy)
    if report_entropy and not fused_sequence:
        raise ValueError("report_entropy needs fused_sequence (one recorded evaluation per chunk)")
    if preload_workers_mode not in SHARD_LOAD_MODES:
        raise ValueError(f"preload_workers_mode must be one of {SHARD_LOAD_MODES}")
    if autocast not in ("none", "bfloat16", "float16"):
        raise ValueError("autocast must be 'none', 'bfloat16', or 'float16'")
    if not index_paths:
        raise ValueError("at least one cache index is required")
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    resolved_checkpoints = tuple(Path(path).resolve() for path in checkpoint_paths)
    if len(set(resolved_checkpoints)) != len(resolved_checkpoints):
        raise ValueError("checkpoint paths must be unique")
    resolved_indexes = tuple(Path(path).resolve() for path in index_paths)
    resolved_tags = Path(tags_path).resolve()
    device = torch.device(device)
    autocast_dtype = _autocast_dtype(autocast)
    rank, world_size = _distributed_context()
    started = time.perf_counter()

    if world_size > 1:
        plans, dataset_metadata = _broadcast_training_plan(rank, resolved_indexes, resolved_tags, manifest_workers)
    else:
        plans, dataset_metadata = build_training_plan(resolved_indexes, resolved_tags, manifest_workers=manifest_workers)
    if not plans:
        raise ValueError("replay tag file selected no cache shards")
    local_plans = balanced_partitions(plans, world_size)[rank]
    groups = _local_groups(local_plans, shards_per_batch)

    # Each checkpoint is evaluated with the ModelConfigV4 stored in its
    # contract.  One model is built per distinct config; checkpoints sharing a
    # config swap weights in place on that model.
    models: dict[ModelConfigV4, UniversalCardPolicyV4] = {}
    loaded_by_config: dict[ModelConfigV4, int] = {}
    configs: list[ModelConfigV4] = []
    infos: list[CheckpointInfoV1] = []
    snapshots: list[dict[str, Tensor]] = []
    for checkpoint_index, path in enumerate(resolved_checkpoints):
        payload = read_actor_critic_checkpoint(path, map_location="cpu")
        contract = payload.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"checkpoint has no model/catalog contract: {path}")
        config = model_config_from_contract(contract)
        if config not in models:
            models[config] = _model_for_config(config, template=model, device=device)
            models[config].eval()
        policy = models[config]
        restore_actor_critic_checkpoint(payload, policy)
        infos.append(_checkpoint_info(path, payload))
        configs.append(config)
        loaded_by_config[config] = checkpoint_index
        del payload
        if len(resolved_checkpoints) > 1:
            snapshots.append({name: value.detach().clone() for name, value in policy.state_dict().items()})

    sums = torch.zeros(len(infos), len(_SUM_FIELDS), dtype=torch.float64, device=device)
    entropy_sums = torch.zeros(len(infos), len(_ENTROPY_SUM_FIELDS), dtype=torch.float64, device=device)
    eval_seconds = [0.0] * len(infos)
    cache_load_seconds = 0.0
    board_width = int(configs[0].board_width)
    dropped_index = _SUM_INDEX["dropped_legacy_expert_frames"]

    shard_loader = _GroupShardLoader(
        [[plan.summary_path for plan in group_plans] for group_plans in groups],
        workers=load_workers,
        preload_next=preload_next_group,
        mode=preload_workers_mode,
        verify=verify_cache,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
    )
    try:
        with torch.no_grad():
            for group_index, group_plans in enumerate(groups):
                load_started = time.perf_counter()
                shards, _preloaded = shard_loader.get(group_index)
                cache_load_seconds += time.perf_counter() - load_started
                dropped = sum(
                    int(
                        shard.legacy_dropped_expert_mask[
                            shard.descriptors[index].offset : shard.descriptors[index].offset
                            + shard.descriptors[index].length
                        ].sum()
                    )
                    for shard, plan in zip(shards, group_plans, strict=True)
                    for index in plan.sequence_indices
                )
                sums[:, dropped_index] += float(dropped)
                batch_size = sum(len(plan.sequence_indices) for plan in group_plans)
                states: list[RecurrentPolicyStateV4] = [
                    models[config].initial_state(batch_size, device=device) for config in configs
                ]
                max_length = max(length for plan in group_plans for length in plan.sequence_lengths)
                chunk_count = math.ceil(max_length / time_steps)
                chunks = _group_chunks(
                    shards,
                    group_plans,
                    chunk_count=chunk_count,
                    time_steps=time_steps,
                    device=device,
                    dummy=False,
                    prefetch_chunks=prefetch_chunks,
                    trim_padding=trim_padding,
                    plan=True,
                )
                for _chunk_index in range(chunk_count):
                    sequence, owner_frames = next(chunks)
                    for checkpoint_index in range(len(infos)):
                        config = configs[checkpoint_index]
                        policy = models[config]
                        if loaded_by_config[config] != checkpoint_index:
                            policy.load_state_dict(snapshots[checkpoint_index], strict=True)
                            loaded_by_config[config] = checkpoint_index
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        step_started = time.perf_counter()
                        slices = lane_slices(batch_size, micro_batches)
                        final_states: list[RecurrentPolicyStateV4] = []
                        for slice_index, (lane_start, lane_stop) in enumerate(slices):
                            lanes = sequence if len(slices) == 1 else slice_sequence_lanes(sequence, lane_start, lane_stop)
                            lane_state = states[checkpoint_index]
                            if len(slices) > 1:
                                lane_state = lane_state.narrow_batch(lane_start, lane_stop - lane_start)
                            recorder = DistributionRecorderV1() if report_entropy else None
                            with (recorder if recorder is not None else contextlib.nullcontext()), torch.autocast(
                                device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
                            ):
                                evaluation = evaluate_recurrent_sequence(
                                    policy,
                                    lanes.observations,
                                    lanes.actions,
                                    lanes.episode_start,
                                    initial_state=lane_state,
                                    gate_temperature=1.0,
                                    action_temperature=1.0,
                                    continue_temperature=1.0,
                                    validate=validate,
                                    preencode_observations=True,
                                    collect_argmax=True,
                                    fused=fused_sequence,
                                    **sequence_plan_options(lanes, compact_tail=compact_tail),
                                )
                            final_states.append(evaluation.final_state)
                            # Statistics are sums; the chunk's owner frames are counted once.
                            sums[checkpoint_index] += chunk_statistics(
                                lanes,
                                evaluation,
                                board_width=board_width,
                                owner_frames=owner_frames if slice_index == 0 else 0,
                                illegal_label_policy=illegal_label_policy,
                            )
                            if recorder is not None:
                                entropy_sums[checkpoint_index] += entropy_statistics(
                                    lanes,
                                    evaluation,
                                    top1_log_probs(recorder.calls, lanes, evaluation),
                                    illegal_label_policy=illegal_label_policy,
                                )
                                del recorder
                            del evaluation
                        states[checkpoint_index] = (
                            final_states[0]
                            if len(final_states) == 1
                            else RecurrentPolicyStateV4(
                                hidden=torch.cat([item.hidden for item in final_states]),
                                cell=torch.cat([item.cell for item in final_states]),
                            )
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        eval_seconds[checkpoint_index] += time.perf_counter() - step_started
                getattr(chunks, "close")()
                del chunks, states
                # Freed on the loader thread, ahead of its next preload.
                shard_loader.release(shards)
                del shards
                if progress:
                    print(
                        json.dumps(
                            {
                                "event": "eval_progress",
                                "rank": rank,
                                "group": group_index + 1,
                                "groups": len(groups),
                                "elapsed_s": round(time.perf_counter() - started, 3),
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        shard_loader.close()

    timings = torch.tensor([*eval_seconds, cache_load_seconds], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        if report_entropy:
            dist.all_reduce(entropy_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
    reduced = sums.cpu().tolist()
    reduced_entropy = entropy_sums.cpu().tolist()
    reduced_timings = timings.cpu().tolist()
    elapsed_s = time.perf_counter() - started

    expected_owner_frames = int(dataset_metadata["selected_owner_frames"])  # type: ignore[call-overload]
    summaries = [
        summarize_statistics(sums_row, illegal_label_policy=illegal_label_policy) for sums_row in reduced
    ]
    if illegal_label_policy == "error":
        offending = [
            f"{info.path}: "
            + ", ".join(
                f"{head}={statistics[f'{head}_illegal_label_count']}/"
                f"{statistics[_LEGALITY_HEAD_COUNTS[head]] + statistics[f'{head}_illegal_label_count']}"  # type: ignore[operator]
                for head in _LEGALITY_HEADS
            )
            for info, statistics in zip(infos, summaries, strict=True)
            if any(statistics[f"{head}_illegal_label_count"] for head in _LEGALITY_HEADS)
        ]
        if offending:
            raise ValueError(
                "supervised labels are illegal under the checkpoint's own legality mask "
                "(illegal/supervised per head; use illegal_label_policy skip or count): " + "; ".join(offending)
            )
    rows: list[dict[str, object]] = []
    for checkpoint_index, info in enumerate(infos):
        statistics = summaries[checkpoint_index]
        if statistics["owner_frames"] != expected_owner_frames:
            raise RuntimeError(
                f"evaluated {statistics['owner_frames']} owner frames, plan selected {expected_owner_frames}"
            )
        rows.append(
            {
                "schema": EVALUATION_SCHEMA,
                "event": "imitation_eval",
                "checkpoint": str(info.path),
                "checkpoint_id": info.checkpoint_id,
                "update_step": info.update_step,
                "training_stage": info.training_stage,
                "next_epoch": info.next_epoch,
                "model_config": asdict(configs[checkpoint_index]),
                "tags_path": str(resolved_tags),
                "tags_sha256": dataset_metadata["train_tags_sha256"],
                "cache_indexes": [str(path) for path in resolved_indexes],
                "cache_index_sha256": dataset_metadata["cache_index_sha256"],
                "replays": int(dataset_metadata["selected_replays"]),  # type: ignore[call-overload]
                "lanes": int(dataset_metadata["selected_owner_sequences"]),  # type: ignore[call-overload]
                "shards": int(dataset_metadata["selected_shards"]),  # type: ignore[call-overload]
                "overridden_replay_copies": dataset_metadata["overridden_replay_copies"],
                "time_steps": time_steps,
                "shards_per_batch": shards_per_batch,
                "micro_batches": micro_batches,
                "autocast": autocast,
                "validate": validate,
                "illegal_label_policy": illegal_label_policy,
                "world_size": world_size,
                "device_type": device.type,
                "checkpoint_eval_s": round(reduced_timings[checkpoint_index], 3),
                "cache_load_s": round(reduced_timings[-1], 3),
                "elapsed_s": round(elapsed_s, 3),
                **statistics,
            }
        )
        if report_entropy:
            rows[-1]["report_entropy"] = True
            rows[-1].update(summarize_entropy_statistics(reduced_entropy[checkpoint_index], statistics))
    if rank == 0:
        for row in rows:
            if output_path is not None:
                _write_json_line(output_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--tags", type=Path, required=True, help="held-out replay-tag file, one tag per line")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL file; one row per checkpoint is appended")
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--shards-per-batch", type=int, default=8)
    parser.add_argument(
        "--micro-batches",
        type=int,
        default=1,
        help="score each chunk's lanes in this many slices (same statistics, smaller forward working set)",
    )
    parser.add_argument("--load-workers", type=int, default=6)
    parser.add_argument("--manifest-workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--autocast", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fused-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate each chunk with one fused LSTM call and one ACT-row decode (--no-fused-sequence: stepped)",
    )
    parser.add_argument("--verify-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preload-next-group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read the next shard group while the current one is scored (two groups resident)",
    )
    parser.add_argument(
        "--preload-workers-mode",
        choices=SHARD_LOAD_MODES,
        default="thread",
        help="decode preloaded shards in --load-workers threads (default) or spawned processes (shared memory)",
    )
    parser.add_argument(
        "--prefetch-chunks",
        type=int,
        default=2,
        help="chunks a background thread gathers and copies ahead; 0 = on the scoring thread",
    )
    parser.add_argument(
        "--illegal-label-policy",
        choices=ILLEGAL_LABEL_POLICIES,
        default="skip",
        help="supervised labels the checkpoint's own legality mask rules out: leave them out of that head's "
        "statistics (skip), keep the historical arithmetic (count), or fail (error); counts are always reported",
    )
    parser.add_argument(
        "--trim-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cut each chunk's padded group/child/effect/event/edge rows to its widest real use "
        "(rows unchanged up to float rounding)",
    )
    parser.add_argument(
        "--compact-tail-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode the repeated frames past a lane's end once and decode only valid rows "
        "(rows unchanged up to float rounding)",
    )
    parser.add_argument(
        "--report-entropy",
        action="store_true",
        help="also report per-head entropy, mean top-1 probability and the share of frames with top-1 > 0.9 "
        "(teacher-forced distributions, same populations as the NLLs; off by default)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif distributed and device.type != "cpu":
        raise ValueError("distributed evaluation supports only cuda or cpu devices")
    torch.set_num_threads(args.torch_threads)
    if distributed:
        if device.type == "cuda":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend="gloo")
    output = args.output.resolve()
    try:
        if not distributed or dist.get_rank() == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
        evaluate_checkpoints(
            args.index,
            args.tags,
            args.checkpoint,
            device=device,
            time_steps=args.time_steps,
            shards_per_batch=args.shards_per_batch,
            load_workers=args.load_workers,
            manifest_workers=args.manifest_workers,
            autocast=args.autocast,
            validate=args.validate,
            verify_cache=args.verify_cache,
            output_path=output,
            progress=True,
            fused_sequence=args.fused_sequence,
            micro_batches=args.micro_batches,
            preload_next_group=args.preload_next_group,
            preload_workers_mode=args.preload_workers_mode,
            prefetch_chunks=args.prefetch_chunks,
            illegal_label_policy=args.illegal_label_policy,
            trim_padding=args.trim_padding,
            compact_tail=args.compact_tail_frames,
            report_entropy=args.report_entropy,
        )
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
