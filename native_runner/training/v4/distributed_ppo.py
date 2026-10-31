"""Synchronous multi-GPU PPO updates for recurrent rollout segments."""

from __future__ import annotations

import math
import os
import time
from typing import Sequence, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .model import UniversalCardPolicyV4
from .ppo import PPOConfigV4, StoredRolloutV4
from .ppo_anchor import anchor_kl_terms, value_only_parameter_names
from .ppo_device_rollout import PackedRolloutDeviceCacheV4
from .ppo_update_profile import profile_scope, profile_step, profile_update
from .tensors import (
    GATE_ACT,
    ActionSequenceV4,
    RecurrentPolicyStateV4,
    UniversalSemanticBatchV4,
    TensorRecordV4,
    _map_tensors,
    concatenate_padded_tensor_records,
    concatenate_tensor_records,
)


def _ppo_transfer_record_v4(
    record: TensorRecordV4, device: torch.device, *, model_input: bool = False,
) -> TensorRecordV4:
    """Transfer a chunk with one host wait, preserving storage/model dtypes.

    The immutable source record stays alive until all copies have completed.
    This avoids a CUDA synchronization for every nested observation field.
    The wait remains inside assembly timing, as in the synchronous path.
    """
    result = _map_tensors(record, lambda value: value.to(
        device=device,
        dtype=torch.float32 if model_input and value.is_floating_point() else None,
        non_blocking=device.type == "cuda",
    ))
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    return cast(TensorRecordV4, result)


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


@torch.no_grad()
def _reference_rollout_components_v4(
    reference: UniversalCardPolicyV4, rollout: StoredRolloutV4, initial_state: RecurrentPolicyStateV4,
    *, microbatch_size: int, chunk_steps: int, autocast_dtype: torch.dtype | None,
    validate: bool = False, device_rollout: PackedRolloutDeviceCacheV4 | None = None,
) -> tuple[Tensor, RecurrentPolicyStateV4]:
    """Replay each reference lane once, including lanes PPO may early-stop before.

    Components are immutable across PPO epochs. Reuse resident observation
    chunks when available; only the small component/state outputs survive.
    """
    device = next(reference.parameters()).device
    components = torch.zeros(rollout.time_steps, rollout.batch_size, 5, device=device)
    source = initial_state.to(device)
    final = RecurrentPolicyStateV4(hidden=source.hidden.clone(), cell=source.cell.clone())
    for start in range(0, rollout.batch_size, microbatch_size):
        lanes = torch.arange(start, min(rollout.batch_size, start + microbatch_size))
        lane_device = lanes.to(device)
        state = final.index_select(lane_device)
        valid = rollout.valid_mask.index_select(1, lanes.to(rollout.valid_mask.device))
        for time_start in range(0, rollout.time_steps, chunk_steps):
            time_end = min(rollout.time_steps, time_start + chunk_steps)
            active = valid[time_start:time_end].any(dim=0).nonzero(as_tuple=False).flatten()
            if not active.numel():
                break
            source_lanes = lanes.index_select(0, active.cpu())
            active_device = active.to(device)
            if device_rollout is not None:
                observations, actions = device_rollout.chunk(time_start, time_end, source_lanes)
            else:
                observations = concatenate_padded_tensor_records(tuple(
                    row.index_select(source_lanes.to(row.match_scalars.device))
                    for row in rollout.observations[time_start:time_end]
                ))
                observations = cast(UniversalSemanticBatchV4, _ppo_transfer_record_v4(
                    observations, device, model_input=True))
                actions = concatenate_tensor_records(tuple(
                    row.index_select(source_lanes.to(row.gate.device)) for row in rollout.actions[time_start:time_end]
                ))
                actions = cast(ActionSequenceV4, _ppo_transfer_record_v4(actions, device))
            episode_start = rollout.episode_start[time_start:time_end].index_select(
                1, source_lanes.to(rollout.episode_start.device)).to(device)
            forced = (torch.zeros_like(episode_start) if rollout.forced_gate_mask is None else
                      rollout.forced_gate_mask[time_start:time_end].index_select(
                          1, source_lanes.to(rollout.forced_gate_mask.device)).to(device))
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                encoded = reference.encode_observation(observations, validate=validate)
                evaluated = reference.evaluate_encoded_action_sequence(
                    observations, encoded, state.index_select(active_device), actions, episode_start,
                    time_steps=time_end - time_start, gate_temperature=rollout.gate_temperature,
                    action_temperature=rollout.action_temperature, continue_temperature=rollout.continue_temperature,
                    conditioned_gate_mask=forced, return_log_prob_components=True, resets_checked=True,
                    validate=validate,
                )
            components[time_start:time_end].index_copy_(1, source_lanes.to(device), evaluated[4])
            state.hidden.index_copy_(0, active_device, evaluated[3].hidden)
            state.cell.index_copy_(0, active_device, evaluated[3].cell)
        final.hidden.index_copy_(0, lane_device, state.hidden)
        final.cell.index_copy_(0, lane_device, state.cell)
    return components, final


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
    reference_model: UniversalCardPolicyV4 | None = None,
    reference_initial_state: RecurrentPolicyStateV4 | None = None,
    reference_final_state_out: RecurrentPolicyStateV4 | None = None,
    anchor_kl_coefficient: float = 0.0,
    anchor_continue_kl_coefficient: float = 0.0,
    value_only: bool = False,
    device_rollout_budget_gib: float = 0.0,
) -> list[dict[str, float]]:
    """Train synchronously across ranks with chunked recurrent backpropagation.

    Every rank owns a disjoint rollout shard. Gradients are weighted by the
    global valid-frame count before one all-reduce, so uneven terminal lengths
    and a one-lane shard imbalance do not bias the update. Model parameters are
    fixed while all time chunks of a minibatch are evaluated; hidden state is
    carried forward and detached only at chunk boundaries (ordinary TBPTT).

    ``reference_model`` adds the IL anchor ``beta * (KL_gate + KL_cand +
    KL_target) + beta_c * KL_cont`` (see ``ppo_anchor``), normalized like the
    policy loss.  The frozen reference is evaluated without gradients on the
    same trajectory at the rollout's actual temperatures. Continuing episodes
    require their own frozen-reference initial state, never the learner state.
    The optional final-state buffer is filled before optimization/early stopping.

    ``value_only`` (critic warmup) freezes every parameter that reaches the
    actor: only the value-only modules receive gradients; the policy and
    anchor terms are reported but not optimized.
    """

    rollout.validate()
    if not math.isfinite(device_rollout_budget_gib) or device_rollout_budget_gib < 0:
        raise ValueError("PPO device rollout budget must be finite and nonnegative")
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
    anchor_enabled = reference_model is not None
    if reference_model is not None:
        if reference_model is model:
            raise ValueError("IL anchor reference must be a separate frozen model")
        if reference_model.delay_policy != model.delay_policy:
            raise ValueError("IL anchor reference and learner delay policies differ")
        lanes_with_data = rollout.valid_mask.any(dim=0)
        if reference_initial_state is None:
            if torch.any(lanes_with_data & ~rollout.episode_start[0].to(lanes_with_data.device)):
                raise ValueError("continuing IL anchor lanes require their own reference initial state")
            if reference_final_state_out is not None:
                reference_initial_state = reference_model.initial_state(rollout.batch_size)
        for state in (reference_initial_state, reference_final_state_out):
            if state is not None and (
                tuple(state.hidden.shape) != (rollout.batch_size, reference_model.config.lstm_hidden_dim)
                or state.cell.shape != state.hidden.shape
                or not torch.isfinite(state.hidden).all() or not torch.isfinite(state.cell).all()
            ):
                raise ValueError("IL anchor reference state has invalid shape or values")
        for coefficient in (anchor_kl_coefficient, anchor_continue_kl_coefficient):
            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise ValueError("IL anchor KL coefficients must be finite and nonnegative")
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
    frozen_for_value_only: list[nn.Parameter] = []
    # cuDNN RNN backward is only defined in training mode (fused LSTM in
    # evaluate_encoded_action_sequence passes ``self.training``).  The policy
    # has no dropout/batch norm, so the mode never changes a log-prob; the
    # caller's mode is restored afterwards (rollouts run in eval()).
    was_training = model.training
    model.train()
    if value_only:
        value_names = set(value_only_parameter_names(model))
        for name, parameter in model.named_parameters():
            if name not in value_names and parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen_for_value_only.append(parameter)
    try:
        return _distributed_ppo_epochs(
            model,
            optimizer,
            rollout,
            epochs=epochs,
            lane_minibatch_size=lane_minibatch_size,
            actual_microbatch_size=actual_microbatch_size,
            sequence_chunk_steps=sequence_chunk_steps,
            actual=actual,
            autocast_dtype=autocast_dtype,
            validate=validate,
            reference_model=reference_model,
            reference_initial_state=reference_initial_state,
            reference_final_state_out=reference_final_state_out,
            anchor_kl_coefficient=float(anchor_kl_coefficient),
            anchor_continue_kl_coefficient=float(anchor_continue_kl_coefficient),
            value_only=bool(value_only),
            anchor_enabled=anchor_enabled,
            advantages=advantages,
            returns=returns,
            rollout_statistics=rollout_statistics,
            global_batches=global_batches,
            parameters=parameters,
            metrics=metrics,
            device_rollout_budget_gib=device_rollout_budget_gib,
        )
    finally:
        for parameter in frozen_for_value_only:
            parameter.requires_grad_(True)
        model.train(was_training)


def _ppo_device_rollout_budget_bytes_v4(
    device: torch.device, requested_gib: float, *, world_size: int
) -> int:
    budget = int(requested_gib * 2**30)
    if device.type == "cuda":
        # Driver free memory excludes unused blocks still held by PyTorch.
        # Reclaim those before applying the conservative half-free limit;
        # otherwise the previous wave's input cache can force the next wave
        # onto the CPU despite ample reclaimable memory. Live CUDA Graph
        # allocations stay owned by their graphs and are not released here.
        torch.cuda.empty_cache()
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        sharing = max(1, math.ceil(local_world / max(1, torch.cuda.device_count())))
        free, _total = torch.cuda.mem_get_info(device)
        budget = min(budget, free // sharing // 2)
    return budget


@profile_update
def _distributed_ppo_epochs(
    model: UniversalCardPolicyV4,
    optimizer: torch.optim.Optimizer,
    rollout: StoredRolloutV4,
    *,
    epochs: int,
    lane_minibatch_size: int,
    actual_microbatch_size: int,
    sequence_chunk_steps: int,
    actual: PPOConfigV4,
    autocast_dtype: torch.dtype | None,
    validate: bool,
    reference_model: UniversalCardPolicyV4 | None,
    reference_initial_state: RecurrentPolicyStateV4 | None,
    reference_final_state_out: RecurrentPolicyStateV4 | None,
    anchor_kl_coefficient: float,
    anchor_continue_kl_coefficient: float,
    value_only: bool,
    anchor_enabled: bool,
    advantages: Tensor,
    returns: Tensor,
    rollout_statistics: dict[str, float],
    global_batches: int,
    parameters: tuple[nn.Parameter, ...],
    metrics: list[dict[str, float]],
    device_rollout_budget_gib: float,
) -> list[dict[str, float]]:
    rank, world_size = _world()
    device = next(model.parameters()).device
    stop_for_kl = False
    cache_reference = (
        reference_model is not None and epochs > 1
        and not any(parameter.requires_grad for parameter in reference_model.parameters())
        and os.environ.get("CR_PPO_CACHE_REFERENCE", "1") != "0"
    )
    reference_components_cache = None
    # Stored actions are immutable. Determine the same sorted ACT row indices
    # on the host before launching either model, avoiding CUDA nonzero's host
    # rendezvous after each expensive encoder forward. Debug switch preserves
    # the original path for parity tests and performance comparisons.
    gate_rows_cpu = (
        torch.stack([action.gate.detach().cpu() for action in rollout.actions])
        if os.environ.get("CR_PPO_PRECOMPUTE_ACT_INDICES", "1") != "0" else None
    )
    device_rollout = None
    training_data: dict[str, Tensor] = {}
    if device_rollout_budget_gib > 0 and rollout.packed_observation_rows is not None:
        budget = _ppo_device_rollout_budget_bytes_v4(device, device_rollout_budget_gib, world_size=world_size)
        device_rollout = PackedRolloutDeviceCacheV4(rollout, device, budget_bytes=budget)
        prefetch_started = time.perf_counter()
        training_data = {
            name: value.to(device, non_blocking=device.type == "cuda") for name, value in {
                "valid": rollout.valid_mask,
                "episode_start": rollout.episode_start,
                "behavior_log_prob": rollout.behavior_log_prob,
                "behavior_value": rollout.behavior_value,
                "forced_gate": (torch.zeros_like(rollout.valid_mask) if rollout.forced_gate_mask is None
                                else rollout.forced_gate_mask),
                "advantages": advantages,
                "returns": returns,
            }.items()
        }
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
            device_rollout.device_bytes += sum(value.numel() * value.element_size() for value in training_data.values())
        device_rollout.prefetch_seconds += time.perf_counter() - prefetch_started

    reference_precomputed = reference_model is not None and reference_initial_state is not None
    if reference_precomputed:
        reference_components_cache, final_reference_state = _reference_rollout_components_v4(
            reference_model, rollout, reference_initial_state,
            microbatch_size=actual_microbatch_size, chunk_steps=sequence_chunk_steps,
            autocast_dtype=autocast_dtype, validate=validate, device_rollout=device_rollout,
        )
        cache_reference = True
        if reference_final_state_out is not None:
            with torch.no_grad():
                reference_final_state_out.hidden.copy_(final_reference_state.hidden)
                reference_final_state_out.cell.copy_(final_reference_state.cell)

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
            # gate, candidate, target, continue KL; learner/reference P(SECOND); ACT rows.
            anchor_sums = torch.zeros(7, device=device, dtype=torch.float64)
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
                    device_lanes = None if lane_indices is None or not training_data else lane_indices.to(device)

                    def select_training(name: str, source: Tensor) -> Tensor:
                        if name not in training_data:
                            return select_time(source)
                        value = training_data[name]
                        return value if device_lanes is None else value.index_select(1, device_lanes)

                    selected_valid_device = select_training("valid", rollout.valid_mask)
                    selected_episode_start = select_training("episode_start", rollout.episode_start)
                    selected_behavior_log_prob = select_training("behavior_log_prob", rollout.behavior_log_prob)
                    selected_behavior_value = select_training("behavior_value", rollout.behavior_value)
                    selected_forced_gate = (
                        torch.zeros_like(selected_valid_mask)
                        if rollout.forced_gate_mask is None
                        else select_time(rollout.forced_gate_mask)
                    )
                    if "forced_gate" in training_data:
                        selected_forced_gate = select_training("forced_gate", selected_forced_gate)
                    selected_initial_state = (
                        rollout.initial_state
                        if lane_indices is None
                        else rollout.initial_state.index_select(lane_indices.to(rollout.initial_state.hidden.device))
                    )
                    selected_advantages = select_training("advantages", advantages)
                    selected_returns = select_training("returns", returns)
                    state = selected_initial_state.to(device)
                    # Zero state: every lane starts its episode at step 0 (checked above).
                    reference_state = (
                        None
                        if reference_model is None or reference_precomputed or (cache_reference and epoch > 0)
                        else reference_model.initial_state(selected_batch_size, device=device)
                    )
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
                        if device_rollout is not None:
                            source_lanes = (
                                active_lanes if lane_indices is None else lane_indices.index_select(0, active_lanes)
                            )
                            with profile_scope("ppo.rollout_chunk"):
                                flat_observations, flat_actions = device_rollout.chunk(time_start, time_end, source_lanes)
                        elif whole_rollout_active:
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
                        if device_rollout is None:
                            flat_observations = concatenate_padded_tensor_records(chunk_observations)
                            if not isinstance(flat_observations, UniversalSemanticBatchV4):
                                raise TypeError("PPO observation concatenation changed type")
                            flat_observations = cast(UniversalSemanticBatchV4,
                                _ppo_transfer_record_v4(flat_observations, device, model_input=True))
                            flat_actions = concatenate_tensor_records(chunk_actions)
                            if not isinstance(flat_actions, ActionSequenceV4):
                                raise TypeError("PPO action concatenation changed type")
                            flat_actions = cast(ActionSequenceV4, _ppo_transfer_record_v4(flat_actions, device))
                        active_lanes_device = active_lanes.to(device)
                        episode_start = selected_episode_start[time_start:time_end]
                        if not all_lanes_active:
                            episode_start = episode_start.index_select(1, active_lanes_device if episode_start.is_cuda else active_lanes)
                        episode_start = episode_start.to(device)
                        chunk_state = state if all_lanes_active else state.index_select(active_lanes_device)
                        chunk_reference_state = (
                            None
                            if reference_state is None
                            else reference_state
                            if all_lanes_active
                            else reference_state.index_select(active_lanes_device)
                        )
                        conditioned_gate_mask = selected_forced_gate[time_start:time_end]
                        if not all_lanes_active:
                            conditioned_gate_mask = conditioned_gate_mask.index_select(
                                1, active_lanes_device if conditioned_gate_mask.is_cuda else active_lanes)
                        conditioned_gate_mask = conditioned_gate_mask.to(device)
                        forced_act_indices = None
                        if gate_rows_cpu is not None:
                            source_lanes_cpu = (
                                active_lanes if lane_indices is None else lane_indices.index_select(0, active_lanes)
                            )
                            forced_act_indices = (
                                gate_rows_cpu[time_start:time_end].index_select(1, source_lanes_cpu.cpu())
                                .reshape(-1).eq(GATE_ACT).nonzero(as_tuple=False).flatten()
                                .to(device, non_blocking=device.type == "cuda")
                            )
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
                                return_log_prob_components=anchor_enabled,
                                resets_checked=True,
                                forced_act_indices=forced_act_indices,
                                validate=validate,
                            )
                            (evaluated_log_prob, evaluated_entropy, evaluated_value, final_state) = sequence_evaluation[
                                :4
                            ]
                            anchor_terms = None
                            reference_final_state = None
                            if reference_model is not None:
                                reference_lanes = (
                                    active_lanes if lane_indices is None else lane_indices.index_select(0, active_lanes)
                                ).to(device)
                                if reference_precomputed or (cache_reference and epoch > 0):
                                    assert reference_components_cache is not None
                                    reference_components = reference_components_cache[time_start:time_end].index_select(
                                        1, reference_lanes)
                                else:
                                    assert chunk_reference_state is not None
                                    with torch.no_grad():
                                        reference_encoded = reference_model.encode_observation(
                                            flat_observations, validate=validate
                                        )
                                        reference_evaluation = reference_model.evaluate_encoded_action_sequence(
                                            flat_observations,
                                            reference_encoded,
                                            chunk_reference_state,
                                            flat_actions,
                                            episode_start,
                                            time_steps=chunk_steps,
                                            gate_temperature=rollout.gate_temperature,
                                            action_temperature=rollout.action_temperature,
                                            continue_temperature=rollout.continue_temperature,
                                            conditioned_gate_mask=conditioned_gate_mask,
                                            return_log_prob_components=True,
                                            resets_checked=True,
                                            forced_act_indices=forced_act_indices,
                                            validate=validate,
                                        )
                                    reference_final_state = reference_evaluation[3]
                                    reference_components = reference_evaluation[4]
                                    if cache_reference:
                                        if reference_components_cache is None:
                                            reference_components_cache = reference_components.new_zeros(
                                                rollout.time_steps, rollout.batch_size, 5)
                                        reference_components_cache[time_start:time_end].index_copy_(
                                            1, reference_lanes, reference_components.detach())
                                lanes = int(episode_start.shape[1])
                                anchor_terms = anchor_kl_terms(
                                    sequence_evaluation[4],
                                    reference_components,
                                    gate=flat_actions.gate.reshape(chunk_steps, lanes),
                                    micro_action_count=flat_actions.micro_action_count.reshape(chunk_steps, lanes),
                                )
                            valid = selected_valid_device[time_start:time_end]
                            if not all_lanes_active:
                                valid = valid.index_select(1, active_lanes_device if valid.is_cuda else active_lanes)
                            valid = valid.to(device)
                            advantage = selected_advantages[time_start:time_end]
                            if not all_lanes_active:
                                advantage = advantage.index_select(1, active_lanes_device if advantage.is_cuda else active_lanes)
                            advantage = advantage.to(device=device, dtype=evaluated_log_prob.dtype)
                            old_log_prob = selected_behavior_log_prob[time_start:time_end]
                            if not all_lanes_active:
                                old_log_prob = old_log_prob.index_select(1, active_lanes_device if old_log_prob.is_cuda else active_lanes)
                            old_log_prob = old_log_prob.to(device=device, dtype=evaluated_log_prob.dtype)
                            log_ratio = evaluated_log_prob - old_log_prob
                            ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                            unclipped = ratio * advantage
                            clipped = ratio.clamp(1.0 - actual.clip_range, 1.0 + actual.clip_range) * advantage
                            policy_values = -torch.minimum(unclipped, clipped)

                            target_return = selected_returns[time_start:time_end]
                            if not all_lanes_active:
                                target_return = target_return.index_select(1, active_lanes_device if target_return.is_cuda else active_lanes)
                            target_return = target_return.to(device=device, dtype=evaluated_value.dtype)
                            if actual.value_clip_range is None:
                                raw_value = (evaluated_value - target_return).square()
                            else:
                                old_value = selected_behavior_value[time_start:time_end]
                                if not all_lanes_active:
                                    old_value = old_value.index_select(1, active_lanes_device if old_value.is_cuda else active_lanes)
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
                            if value_only:
                                loss_sum = actual.value_coefficient * value_sum
                            else:
                                loss_sum = policy_sum + actual.value_coefficient * value_sum
                                if actual.entropy_coefficient:
                                    loss_sum = loss_sum - actual.entropy_coefficient * entropy_sum
                            anchor_row_sums = None
                            if anchor_terms is not None:
                                anchor_weight = weight.float()
                                anchor_row_sums = torch.stack(
                                    [
                                        (anchor_terms[name] * anchor_weight).sum()
                                        for name in (
                                            "gate",
                                            "candidate",
                                            "target",
                                            "continue",
                                            "learner_second_prob",
                                            "reference_second_prob",
                                            "act",
                                        )
                                    ]
                                )
                                if not value_only:
                                    anchor_loss = anchor_kl_coefficient * anchor_row_sums[:3].sum() + (
                                        anchor_continue_kl_coefficient * anchor_row_sums[3]
                                    )
                                    loss_sum = loss_sum + anchor_loss.to(loss_sum.dtype)
                            loss = loss_sum * (world_size / global_valid.to(policy_sum.dtype))
                        with profile_scope("ppo.backward"):
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
                        if reference_final_state is not None:
                            assert reference_state is not None
                            if all_lanes_active:
                                reference_state = reference_final_state
                            else:
                                reference_state.hidden.index_copy_(0, active_lanes_device, reference_final_state.hidden)
                                reference_state.cell.index_copy_(0, active_lanes_device, reference_final_state.cell)
                        if anchor_row_sums is not None:
                            anchor_sums += anchor_row_sums.detach().double()
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
            with profile_scope("ppo.gradient_sync"):
                _allreduce_gradients(parameters, world_size=world_size)
            gradient_sync_seconds = time.perf_counter() - gradient_sync_started
            gradient_norm = 0.0
            if actual.max_grad_norm is not None:
                gradient_norm = float(nn.utils.clip_grad_norm_(parameters, actual.max_grad_norm))
            optimizer_step_started = time.perf_counter()
            with profile_scope("ppo.optimizer_step"):
                optimizer.step()
            optimizer_step_seconds = time.perf_counter() - optimizer_step_started

            for event_start, event_end in forward_backward_events:
                forward_backward_seconds += event_start.elapsed_time(event_end) / 1000.0

            _sum(sums)
            if anchor_enabled:
                _sum(anchor_sums)
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
                "rollout_device_cache_gib": 0.0 if device_rollout is None else device_rollout.device_bytes / 2**30,
                "rollout_device_prefetch_seconds": 0.0 if device_rollout is None else device_rollout.prefetch_seconds,
                "rollout_packed_chunks": float(device_rollout is not None),
                "rollout_device_observations_resident": float(device_rollout is not None and device_rollout.resident),
                **rollout_statistics,
            }
            if value_only:
                row["value_only"] = 1.0
            if anchor_enabled:
                act_count = max(float(anchor_sums[6]), 1.0)
                kl_gate, kl_candidate, kl_target, kl_continue = (float(anchor_sums[index]) / count for index in range(4))
                anchor_loss_value = (
                    0.0
                    if value_only
                    else anchor_kl_coefficient * (kl_gate + kl_candidate + kl_target)
                    + anchor_continue_kl_coefficient * kl_continue
                )
                row.update(
                    {
                        "anchor_kl_gate": kl_gate,
                        "anchor_kl_candidate": kl_candidate,
                        "anchor_kl_target": kl_target,
                        "anchor_kl_continue": kl_continue,
                        # The adaptive coefficient controls this sum (continue excluded).
                        "anchor_kl": kl_gate + kl_candidate + kl_target,
                        "anchor_kl_all_heads": kl_gate + kl_candidate + kl_target + kl_continue,
                        "anchor_kl_coefficient": anchor_kl_coefficient,
                        "anchor_continue_kl_coefficient": anchor_continue_kl_coefficient,
                        "anchor_loss": anchor_loss_value,
                        "anchor_learner_second_prob": float(anchor_sums[4]) / act_count,
                        "anchor_reference_second_prob": float(anchor_sums[5]) / act_count,
                        "anchor_act_count": float(anchor_sums[6]),
                    }
                )
                row["total"] += anchor_loss_value
            metrics.append(row)
            profile_step()
            if actual.target_kl is not None and row["approx_kl"] > actual.target_kl:
                stop_for_kl = True
                break
        if stop_for_kl:
            break
    return metrics
