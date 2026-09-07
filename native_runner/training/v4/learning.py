"""Shared recurrent sequence utilities for V4 IL and PPO.

The engine-facing producer keeps observations in chronological order.  This
module deliberately accepts a sequence of complete V4 batches instead of a
flat image-style tensor so the LSTM, public tracker state, and previous-action
input are never shuffled apart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Sequence, cast

import torch
from torch import Tensor

from .model import UniversalCardPolicyV4
from .tensors import (
    ActionEvaluationComponentsV4,
    ActionSequenceV4,
    EncodedObservationV4,
    RecurrentPolicyStateV4,
    UniversalSemanticBatchV4,
    concatenate_padded_tensor_records,
)


@dataclass(slots=True)
class RecurrentEvaluationV4:
    """Stacked policy evaluation for one chronological ``[T,B]`` sequence."""

    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    components: ActionEvaluationComponentsV4
    final_state: RecurrentPolicyStateV4


def _validate_time_batch_mask(value: Tensor, *, time_steps: int, batch_size: int, label: str) -> None:
    if value.dtype != torch.bool or tuple(value.shape) != (time_steps, batch_size):
        raise ValueError(f"{label} must be bool [T,B]")


def evaluate_recurrent_sequence(
    model: UniversalCardPolicyV4,
    observations: Sequence[UniversalSemanticBatchV4],
    actions: Sequence[ActionSequenceV4],
    episode_start: Tensor,
    *,
    initial_state: RecurrentPolicyStateV4 | None = None,
    gate_temperature: float | None = None,
    action_temperature: float | None = None,
    continue_temperature: float | None = None,
    validate: bool = True,
    preencode_observations: bool = False,
    encoded_observations: Sequence[EncodedObservationV4] | None = None,
) -> RecurrentEvaluationV4:
    """Teacher-force actions; callers detach state between TBPTT chunks."""

    time_steps = len(observations)
    if time_steps == 0 or len(actions) != time_steps:
        raise ValueError("observations and actions need the same nonzero length")
    batch_size = observations[0].batch_size
    if any(item.batch_size != batch_size for item in observations):
        raise ValueError("all sequence observations must have the same batch size")
    _validate_time_batch_mask(episode_start, time_steps=time_steps, batch_size=batch_size, label="episode_start")
    if preencode_observations and encoded_observations is not None:
        raise ValueError("pass preencoding mode or encoded observations, not both")

    device = observations[0].match_scalars.device
    if episode_start.device != device:
        raise ValueError("episode_start and observations must share a device")
    state = initial_state or model.initial_state(batch_size, device=device)
    encoded_sequence = None if encoded_observations is None else tuple(encoded_observations)
    if encoded_sequence is not None and (
        len(encoded_sequence) != time_steps
        or any(int(item.scene_state.shape[0]) != batch_size for item in encoded_sequence)
    ):
        raise ValueError("encoded observations must preserve [T,B]")
    evaluation_observations = observations
    if preencode_observations:
        flat_observations = cast(UniversalSemanticBatchV4, concatenate_padded_tensor_records(tuple(observations)))
        encoded_flat = model.encode_observation(flat_observations, validate=validate)
        evaluation_observations = tuple(
            flat_observations.narrow_batch(step * batch_size, batch_size) for step in range(time_steps)
        )
        encoded_sequence = tuple(encoded_flat.narrow_batch(step * batch_size, batch_size) for step in range(time_steps))
    log_probs: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []
    components: list[ActionEvaluationComponentsV4] = []
    for step, (observation, expert_action) in enumerate(zip(evaluation_observations, actions, strict=True)):
        evaluate = model.evaluate_actions if encoded_sequence is None else model.evaluate_encoded_actions
        arguments = (observation,) if encoded_sequence is None else (observation, encoded_sequence[step])
        output = evaluate(
            *arguments,
            state,
            expert_action,
            episode_start=episode_start[step],
            gate_temperature=gate_temperature,
            action_temperature=action_temperature,
            continue_temperature=continue_temperature,
            validate=validate,
        )
        if output.action_components is None:
            raise RuntimeError("V4 action evaluation did not return head components")
        log_probs.append(output.log_prob)
        entropies.append(output.entropy)
        values.append(output.value)
        components.append(output.action_components)
        state = output.next_state

    return RecurrentEvaluationV4(
        log_prob=torch.stack(log_probs),
        entropy=torch.stack(entropies),
        value=torch.stack(values),
        components=ActionEvaluationComponentsV4(
            **{
                item.name: torch.stack([getattr(component, item.name) for component in components])
                for item in fields(ActionEvaluationComponentsV4)
            }
        ),
        final_state=state,
    )


def discounted_returns(
    rewards: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    truncation_bootstrap_value: Tensor | None = None,
) -> Tensor:
    """Return targets with terminal and time-limit semantics kept distinct.

    ``gamma`` is applied once per 250 ms / five-tick decision step.  A true
    terminal never bootstraps.  A time-limit truncation uses the matching value
    in ``truncation_bootstrap_value`` and stops return propagation across the
    reset boundary.
    """

    if rewards.ndim != 2 or not rewards.is_floating_point():
        raise ValueError("rewards must be floating [T,B]")
    time_steps, batch_size = rewards.shape
    for value, label in ((terminated, "terminated"), (truncated, "truncated")):
        _validate_time_batch_mask(value, time_steps=time_steps, batch_size=batch_size, label=label)
    if torch.any(terminated & truncated):
        raise ValueError("a transition cannot be both terminated and truncated")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0,1]")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")

    if truncation_bootstrap_value is None:
        truncation_values = torch.zeros_like(rewards)
    else:
        if tuple(truncation_bootstrap_value.shape) != tuple(rewards.shape):
            raise ValueError("truncation bootstrap values must have shape [T,B]")
        truncation_values = truncation_bootstrap_value.to(device=rewards.device, dtype=rewards.dtype)
    if torch.any(truncated & ~torch.isfinite(truncation_values)):
        raise ValueError("truncated transitions need finite bootstrap values")

    running = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)

    result = torch.empty_like(rewards)
    for step in range(time_steps - 1, -1, -1):
        continuation = torch.where(
            terminated[step], torch.zeros_like(running), torch.where(truncated[step], truncation_values[step], running)
        )
        running = rewards[step] + float(gamma) * continuation
        result[step] = running
    return result


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute PPO GAE without leaking traces across episode boundaries."""

    if (
        rewards.ndim != 2
        or tuple(values.shape) != tuple(rewards.shape)
        or tuple(next_values.shape) != tuple(rewards.shape)
    ):
        raise ValueError("rewards, values, and next_values must share [T,B]")
    time_steps, batch_size = rewards.shape
    for value, label in ((terminated, "terminated"), (truncated, "truncated")):
        _validate_time_batch_mask(value, time_steps=time_steps, batch_size=batch_size, label=label)
    if torch.any(terminated & truncated):
        raise ValueError("a transition cannot be both terminated and truncated")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0,1]")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be finite and in [0,1]")
    if not all(item.is_floating_point() for item in (rewards, values, next_values)):
        raise ValueError("GAE tensors must be floating point")
    if not all(torch.isfinite(item).all() for item in (rewards, values, next_values)):
        raise ValueError("GAE tensors must be finite")

    not_terminal = (~terminated).to(rewards.dtype)
    deltas = rewards + float(gamma) * not_terminal * next_values - values
    trace_continues = (~(terminated | truncated)).to(rewards.dtype)
    advantage = torch.empty_like(rewards)
    running = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
    for step in range(time_steps - 1, -1, -1):
        running = deltas[step] + (float(gamma) * float(gae_lambda) * trace_continues[step] * running)
        advantage[step] = running
    return advantage, advantage + values


def trusted_decision_mask(
    decision_ticks: Tensor,
    first_untrusted_tick: Tensor,
    *,
    decision_ticks_per_step: int = 5,
    discard_tail_ticks: int = 200,
) -> Tensor:
    """Drop the final ten seconds before replay failure or early termination.

    At 20 Hz, ten seconds is 200 native ticks.  A frame remains trusted only
    when its complete five-tick decision window ends before that tail begins.
    ``first_untrusted_tick`` is one scalar per batch lane.
    """

    if decision_ticks.ndim != 2 or first_untrusted_tick.ndim != 1:
        raise ValueError("decision ticks must be [T,B] and cutoff ticks [B]")
    if decision_ticks.shape[1] != first_untrusted_tick.shape[0]:
        raise ValueError("decision and cutoff batch sizes differ")
    if decision_ticks_per_step <= 0 or discard_tail_ticks < 0:
        raise ValueError("decision stride must be positive and tail non-negative")
    return decision_ticks + int(decision_ticks_per_step) <= first_untrusted_tick[None] - int(discard_tail_ticks)
