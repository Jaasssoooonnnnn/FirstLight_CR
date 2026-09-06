"""Synchronous multi-GPU PPO updates for recurrent rollout segments."""

from __future__ import annotations

import math
import time
from typing import Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .model import UniversalCardPolicyV4
from .ppo import PPOConfigV4, StoredRolloutV4
from .tensors import (
    ActionSequenceV4,
    UniversalSemanticBatchV4,
    concatenate_padded_tensor_records,
    concatenate_tensor_records,
)


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _sum(value: Tensor) -> Tensor:
    _rank, world_size = _world()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _maximum(value: Tensor) -> Tensor:
    _rank, world_size = _world()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def _allreduce_gradients(parameters: Sequence[nn.Parameter], *, world_size: int) -> None:
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("distributed PPO has no trainable parameters")
    gradients = [
        (
            parameter.grad
            if parameter.grad is not None
            else torch.zeros_like(parameter, memory_format=torch.preserve_format)
        )
        for parameter in trainable
    ]
    if world_size > 1:
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for parameter, gradient in zip(trainable, gradients, strict=True):
            count = parameter.numel()
            reduced = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = reduced.clone()
            else:
                parameter.grad.copy_(reduced)
            offset += count
    else:
        for parameter, gradient in zip(trainable, gradients, strict=True):
            if parameter.grad is None:
                parameter.grad = gradient


def _global_normalize_advantages(advantages: Tensor, valid_mask: Tensor, *, device: torch.device) -> Tensor:
    selected = advantages[valid_mask].to(device=device, dtype=torch.float64)
    statistics = torch.tensor(
        [
            float(selected.sum()) if selected.numel() else 0.0,
            float(selected.square().sum()) if selected.numel() else 0.0,
            float(selected.numel()),
        ],
        dtype=torch.float64,
        device=device,
    )
    _sum(statistics)
    count = statistics[2]
    if float(count) <= 1.0:
        return advantages
    mean = statistics[0] / count
    variance = (statistics[1] / count - mean.square()).clamp_min(0.0)
    return (advantages - mean.cpu().to(advantages.dtype)) / (variance.sqrt().cpu().to(advantages.dtype) + 1e-8)


def _global_rollout_statistics(
    rollout: StoredRolloutV4, advantages: Tensor, returns: Tensor, *, device: torch.device
) -> dict[str, float]:
    valid = rollout.valid_mask
    reward = rollout.rewards[valid].double()
    advantage = advantages[valid].double()
    target_return = returns[valid].double()
    behavior_value = rollout.behavior_value[valid].double()
    error = behavior_value - target_return

    def moments(value: Tensor) -> tuple[float, float]:
        return float(value.sum()), float(value.square().sum())

    reward_sum, reward_square_sum = moments(reward)
    advantage_sum, advantage_square_sum = moments(advantage)
    return_sum, return_square_sum = moments(target_return)
    value_sum, _value_square_sum = moments(behavior_value)
    error_sum, error_square_sum = moments(error)
    statistics = torch.tensor(
        [
            float(valid.sum()),
            reward_sum,
            reward_square_sum,
            advantage_sum,
            advantage_square_sum,
            return_sum,
            return_square_sum,
            value_sum,
            error_sum,
            error_square_sum,
        ],
        dtype=torch.float64,
        device=device,
    )
    _sum(statistics)
    count = max(float(statistics[0]), 1.0)

    def mean_std(total: Tensor, square_total: Tensor) -> tuple[float, float]:
        mean = total / count
        variance = (square_total / count - mean.square()).clamp_min(0.0)
        return float(mean), float(variance.sqrt())

    reward_mean, reward_std = mean_std(statistics[1], statistics[2])
    advantage_mean, advantage_std = mean_std(statistics[3], statistics[4])
    return_mean, return_std = mean_std(statistics[5], statistics[6])
    error_mean = statistics[8] / count
    error_variance = (statistics[9] / count - error_mean.square()).clamp_min(0.0)
    return_variance = statistics[6] / count - (statistics[5] / count).square()
    explained_variance = 1.0 - float(error_variance / return_variance) if float(return_variance) > 1e-12 else 0.0
    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "return_mean": return_mean,
        "return_std": return_std,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "behavior_value_mean": float(statistics[7] / count),
        "value_explained_variance": explained_variance,
    }


def _check_temperatures(model: UniversalCardPolicyV4, rollout: StoredRolloutV4) -> None:
    for actual, expected, label in (
        (model.ppo_gate_temperature, rollout.gate_temperature, "gate"),
        (model.ppo_action_temperature, rollout.action_temperature, "action"),
        (model.ppo_continue_temperature, rollout.continue_temperature, "continue"),
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"distributed PPO {label} temperature changed")


def _balanced_lane_permutation(valid_mask: Tensor, *, minibatch_size: int, generator: torch.Generator) -> Tensor:
    """Shuffle lanes while balancing valid-frame work across minibatches."""

    if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("PPO lane balancing requires a boolean [T,B] mask")
    batch_size = int(valid_mask.shape[1])
    if batch_size <= 0 or minibatch_size <= 0:
        raise ValueError("PPO lane balancing requires positive batch sizes")
    batch_count = math.ceil(batch_size / minibatch_size)
    random_order = torch.randperm(batch_size, generator=generator).tolist()
    lane_lengths = valid_mask.sum(dim=0).tolist()
    longest_first = sorted(random_order, key=lambda lane: -int(lane_lengths[lane]))
    capacities = [min(minibatch_size, batch_size - index * minibatch_size) for index in range(batch_count)]
    bins: list[list[int]] = [[] for _ in capacities]
    loads = [0 for _ in capacities]
    for lane in longest_first:
        available = [index for index, capacity in enumerate(capacities) if len(bins[index]) < capacity]
        target = min(available, key=lambda index: (loads[index], index))
        bins[target].append(lane)
        loads[target] += int(lane_lengths[lane])
    return torch.tensor([lane for batch in bins for lane in batch], dtype=torch.long)


def distributed_ppo_update_v4(
    model: UniversalCardPolicyV4,
    optimizer: torch.optim.Optimizer,
    rollout: StoredRolloutV4,
    *,
    epochs: int,
    lane_minibatch_size: int,
    lane_microbatch_size: int | None = None,
    sequence_chunk_steps: int,
    config: PPOConfigV4 | None = None,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    expected_policy_state_id: str | None = None,
    validate: bool = False,
) -> list[dict[str, float]]:
    """Train synchronously across ranks with chunked recurrent backpropagation.

    Every rank owns a disjoint rollout shard. Gradients are weighted by the
    global valid-frame count before one all-reduce, so uneven terminal lengths
    and a one-lane shard imbalance do not bias the update. Model parameters are
    fixed while all time chunks of a minibatch are evaluated; hidden state is
    carried forward and detached only at chunk boundaries (ordinary TBPTT).
    """

    rollout.validate()
    if rollout.time_steps > 1 and torch.any(rollout.episode_start[1:]):
        raise ValueError("distributed PPO lanes must contain one episode each")
    if rollout.time_steps > 1 and torch.any(rollout.valid_mask[1:] & ~rollout.valid_mask[:-1]):
        raise ValueError("distributed PPO lane validity must be one contiguous prefix")
    if expected_policy_state_id is not None and (rollout.policy_state_id != expected_policy_state_id):
        raise ValueError("distributed PPO rollout behavior identity differs")
    actual_microbatch_size = lane_minibatch_size if lane_microbatch_size is None else int(lane_microbatch_size)
    if (
        epochs <= 0
        or lane_minibatch_size <= 0
        or actual_microbatch_size <= 0
        or actual_microbatch_size > lane_minibatch_size
        or sequence_chunk_steps <= 0
    ):
        raise ValueError("distributed PPO batch settings must be positive")
    _check_temperatures(model, rollout)
    actual = config or PPOConfigV4()
    rank, world_size = _world()
    device = next(model.parameters()).device
    advantages, returns = rollout.advantages_and_returns(gamma=actual.gamma, gae_lambda=actual.gae_lambda)
    rollout_statistics = _global_rollout_statistics(rollout, advantages, returns, device=device)
    advantages = _global_normalize_advantages(advantages, rollout.valid_mask, device=device)

    local_batches = math.ceil(rollout.batch_size / lane_minibatch_size)
    batch_count_tensor = torch.tensor(local_batches, device=device, dtype=torch.long)
    _maximum(batch_count_tensor)
    global_batches = int(batch_count_tensor)
    parameters = tuple(model.parameters())
    metrics: list[dict[str, float]] = []
    stop_for_kl = False

    for epoch in range(epochs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0x51F0_0000 + epoch * 4099 + rank)
        permutation = _balanced_lane_permutation(
            rollout.valid_mask, minibatch_size=lane_minibatch_size, generator=generator
        )
        for minibatch in range(global_batches):
            start_lane = minibatch * lane_minibatch_size
            indices = permutation[start_lane : start_lane + lane_minibatch_size]
            has_local_data = bool(indices.numel())
            if has_local_data:
                local_valid = (
                    rollout.valid_mask.index_select(1, indices.to(rollout.valid_mask.device))
                    .sum()
                    .to(device=device, dtype=torch.float64)
                )
            else:
                local_valid = torch.zeros((), device=device, dtype=torch.float64)
            global_valid = local_valid.clone()
            _sum(global_valid)
            if float(global_valid) <= 0.0:
                continue

            optimizer.zero_grad(set_to_none=True)
            sums = torch.zeros(8, device=device, dtype=torch.float64)
            evaluated_frames = 0
            dense_frames = 0
            selection_seconds = 0.0
            assembly_seconds = 0.0
            gradient_sync_seconds = 0.0
            optimizer_step_seconds = 0.0
            forward_backward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            forward_backward_seconds = 0.0
            if has_local_data:
                for microbatch_start in range(0, int(indices.numel()), actual_microbatch_size):
                    selection_started = time.perf_counter()
                    microbatch_indices = indices[microbatch_start : microbatch_start + actual_microbatch_size]
                    lane_indices = None if int(microbatch_indices.numel()) == rollout.batch_size else microbatch_indices

                    def select_time(value: Tensor) -> Tensor:
                        if lane_indices is None:
                            return value
                        return value.index_select(1, lane_indices.to(value.device))

                    selected_batch_size = int(microbatch_indices.numel())
                    selected_valid_mask = select_time(rollout.valid_mask)
                    selected_episode_start = select_time(rollout.episode_start)
                    selected_behavior_log_prob = select_time(rollout.behavior_log_prob)
                    selected_behavior_value = select_time(rollout.behavior_value)
                    selected_forced_gate = (
                        torch.zeros_like(selected_valid_mask)
                        if rollout.forced_gate_mask is None
                        else select_time(rollout.forced_gate_mask)
                    )
                    selected_initial_state = (
                        rollout.initial_state
                        if lane_indices is None
                        else rollout.initial_state.index_select(lane_indices.to(rollout.initial_state.hidden.device))
                    )
                    selected_advantages = select_time(advantages)
                    selected_returns = select_time(returns)
                    state = selected_initial_state.to(device)
                    selection_seconds += time.perf_counter() - selection_started

                    for time_start in range(0, rollout.time_steps, sequence_chunk_steps):
                        time_end = min(rollout.time_steps, time_start + sequence_chunk_steps)
                        chunk_steps = time_end - time_start
                        chunk_valid = selected_valid_mask[time_start:time_end]
                        active_lanes = chunk_valid.any(dim=0).nonzero(as_tuple=False).flatten()
                        if not active_lanes.numel():
                            break
                        all_lanes_active = int(active_lanes.numel()) == selected_batch_size
                        evaluated_frames += chunk_steps * int(active_lanes.numel())
                        dense_frames += chunk_steps * selected_batch_size
                        assembly_started = time.perf_counter()
                        whole_rollout_active = lane_indices is None and all_lanes_active
                        if whole_rollout_active:
                            chunk_observations = tuple(rollout.observations[time_start:time_end])
                            chunk_actions = tuple(rollout.actions[time_start:time_end])
                        else:
                            source_lanes = (
                                active_lanes if lane_indices is None else lane_indices.index_select(0, active_lanes)
                            )
                            chunk_observations = tuple(
                                observation.index_select(source_lanes.to(observation.match_scalars.device))
                                for observation in rollout.observations[time_start:time_end]
                            )
                            chunk_actions = tuple(
                                action.index_select(source_lanes.to(action.gate.device))
                                for action in rollout.actions[time_start:time_end]
                            )
                        flat_observations = concatenate_padded_tensor_records(chunk_observations)
                        if not isinstance(flat_observations, UniversalSemanticBatchV4):
                            raise TypeError("PPO observation concatenation changed type")
                        flat_observations = flat_observations.to_model_input(device)
                        flat_actions = concatenate_tensor_records(chunk_actions)
                        if not isinstance(flat_actions, ActionSequenceV4):
                            raise TypeError("PPO action concatenation changed type")
                        flat_actions = flat_actions.to(device)
                        episode_start = selected_episode_start[time_start:time_end]
                        if not all_lanes_active:
                            episode_start = episode_start.index_select(1, active_lanes)
                        episode_start = episode_start.to(device)
                        active_lanes_device = active_lanes.to(device)
                        chunk_state = state if all_lanes_active else state.index_select(active_lanes_device)
                        conditioned_gate_mask = selected_forced_gate[time_start:time_end]
                        if not all_lanes_active:
                            conditioned_gate_mask = conditioned_gate_mask.index_select(1, active_lanes)
                        conditioned_gate_mask = conditioned_gate_mask.to(device)
                        assembly_seconds += time.perf_counter() - assembly_started

                        cuda_start = None
                        cuda_end = None
                        cpu_forward_started = time.perf_counter()
                        if device.type == "cuda":
                            cuda_start = torch.cuda.Event(enable_timing=True)
                            cuda_end = torch.cuda.Event(enable_timing=True)
                            cuda_start.record()
                        with torch.autocast(
                            device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
                        ):
                            encoded_flat = model.encode_observation(flat_observations, validate=validate)
                            sequence_evaluation = model.evaluate_encoded_action_sequence(
                                flat_observations,
                                encoded_flat,
                                chunk_state,
                                flat_actions,
                                episode_start,
                                time_steps=chunk_steps,
                                gate_temperature=rollout.gate_temperature,
                                action_temperature=rollout.action_temperature,
                                continue_temperature=rollout.continue_temperature,
                                conditioned_gate_mask=conditioned_gate_mask,
                                validate=validate,
                            )
                            (evaluated_log_prob, evaluated_entropy, evaluated_value, final_state) = sequence_evaluation
                            valid = chunk_valid
                            if not all_lanes_active:
                                valid = valid.index_select(1, active_lanes)
                            valid = valid.to(device)
                            advantage = selected_advantages[time_start:time_end]
                            if not all_lanes_active:
                                advantage = advantage.index_select(1, active_lanes)
                            advantage = advantage.to(device=device, dtype=evaluated_log_prob.dtype)
                            old_log_prob = selected_behavior_log_prob[time_start:time_end]
                            if not all_lanes_active:
                                old_log_prob = old_log_prob.index_select(1, active_lanes)
                            old_log_prob = old_log_prob.to(device=device, dtype=evaluated_log_prob.dtype)
                            log_ratio = evaluated_log_prob - old_log_prob
                            ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                            unclipped = ratio * advantage
                            clipped = ratio.clamp(1.0 - actual.clip_range, 1.0 + actual.clip_range) * advantage
                            policy_values = -torch.minimum(unclipped, clipped)

                            target_return = selected_returns[time_start:time_end]
                            if not all_lanes_active:
                                target_return = target_return.index_select(1, active_lanes)
                            target_return = target_return.to(device=device, dtype=evaluated_value.dtype)
                            if actual.value_clip_range is None:
                                raw_value = (evaluated_value - target_return).square()
                            else:
                                old_value = selected_behavior_value[time_start:time_end]
                                if not all_lanes_active:
                                    old_value = old_value.index_select(1, active_lanes)
                                old_value = old_value.to(device=device, dtype=evaluated_value.dtype)
                                clipped_value = old_value + (evaluated_value - old_value).clamp(
                                    -actual.value_clip_range, actual.value_clip_range
                                )
                                raw_value = torch.maximum(
                                    (evaluated_value - target_return).square(), (clipped_value - target_return).square()
                                )
                            weight = valid.to(evaluated_log_prob.dtype)
                            policy_sum = (policy_values * weight).sum()
                            value_sum = 0.5 * (raw_value * weight).sum()
                            entropy_sum = (evaluated_entropy * weight).sum()
                            loss_sum = policy_sum + actual.value_coefficient * value_sum
                            if actual.entropy_coefficient:
                                loss_sum = loss_sum - actual.entropy_coefficient * entropy_sum
                            loss = loss_sum * (world_size / global_valid.to(policy_sum.dtype))
                        loss.backward()
                        if cuda_end is not None and cuda_start is not None:
                            cuda_end.record()
                            forward_backward_events.append((cuda_start, cuda_end))
                        else:
                            forward_backward_seconds += time.perf_counter() - cpu_forward_started
                        final_state = final_state.detach()
                        if all_lanes_active:
                            state = final_state
                        else:
                            with torch.no_grad():
                                state.hidden.index_copy_(0, active_lanes_device, final_state.hidden)
                                state.cell.index_copy_(0, active_lanes_device, final_state.cell)
                        with torch.no_grad():
                            sums[:6] += torch.stack(
                                (
                                    policy_sum.detach().double(),
                                    value_sum.detach().double(),
                                    entropy_sum.detach().double(),
                                    ((-log_ratio) * weight).sum().detach().double(),
                                    (((ratio - 1.0).abs() > actual.clip_range).to(weight.dtype) * weight)
                                    .sum()
                                    .detach()
                                    .double(),
                                    weight.sum().detach().double(),
                                )
                            )
            sums[6] = evaluated_frames
            sums[7] = dense_frames
            gradient_sync_started = time.perf_counter()
            _allreduce_gradients(parameters, world_size=world_size)
            gradient_sync_seconds = time.perf_counter() - gradient_sync_started
            gradient_norm = 0.0
            if actual.max_grad_norm is not None:
                gradient_norm = float(nn.utils.clip_grad_norm_(parameters, actual.max_grad_norm))
            optimizer_step_started = time.perf_counter()
            optimizer.step()
            optimizer_step_seconds = time.perf_counter() - optimizer_step_started

            for event_start, event_end in forward_backward_events:
                forward_backward_seconds += event_start.elapsed_time(event_end) / 1000.0

            _sum(sums)
            count = max(float(sums[5]), 1.0)
            row = {
                "total": (
                    float(sums[0])
                    + actual.value_coefficient * float(sums[1])
                    - actual.entropy_coefficient * float(sums[2])
                )
                / count,
                "policy": float(sums[0]) / count,
                "value": float(sums[1]) / count,
                "entropy": float(sums[2]) / count,
                "approx_kl": float(sums[3]) / count,
                "clip_fraction": float(sums[4]) / count,
                "valid_count": float(sums[5]),
                "evaluated_frame_count": float(sums[6]),
                "dense_frame_count": float(sums[7]),
                "evaluated_frame_fraction": (float(sums[6]) / max(float(sums[7]), 1.0)),
                "rollout_selection_seconds": selection_seconds,
                "rollout_assembly_seconds": assembly_seconds,
                "forward_backward_seconds": forward_backward_seconds,
                "gradient_sync_seconds": gradient_sync_seconds,
                "optimizer_step_seconds": optimizer_step_seconds,
                "gradient_norm": gradient_norm,
                **rollout_statistics,
            }
            metrics.append(row)
            if actual.target_kl is not None and row["approx_kl"] > actual.target_kl:
                stop_for_kl = True
                break
        if stop_for_kl:
            break
    return metrics
