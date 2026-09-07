"""Recurrent rollout storage and optimizer configuration for policy V4."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import torch
from torch import Tensor

from .learning import generalized_advantage_estimate
from .tensors import ActionSequenceV4, GATE_ACT, RecurrentPolicyStateV4, UniversalSemanticBatchV4


def _time_batch(value: Tensor, shape: tuple[int, int], label: str) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{label} must have shape [T,B]")


@dataclass(slots=True)
class StoredRolloutV4:
    """On-policy data retained for repeated PPO epochs.

    Observations are materialized once at collection time.  PPO minibatches
    never rerun the battle engine, and minibatching selects whole recurrent
    lanes rather than shuffled frames.
    """

    observations: tuple[UniversalSemanticBatchV4, ...]
    actions: tuple[ActionSequenceV4, ...]
    episode_start: Tensor
    valid_mask: Tensor
    behavior_log_prob: Tensor
    behavior_value: Tensor
    next_value: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    initial_state: RecurrentPolicyStateV4
    gate_temperature: float
    action_temperature: float
    continue_temperature: float
    policy_state_id: str
    forced_gate_mask: Tensor | None = None

    @property
    def time_steps(self) -> int:
        return len(self.observations)

    @property
    def batch_size(self) -> int:
        return self.observations[0].batch_size if self.observations else 0

    def validate(self) -> None:
        time_steps = self.time_steps
        if time_steps == 0 or len(self.actions) != time_steps:
            raise ValueError("rollout observations/actions need equal nonzero length")
        batch_size = self.batch_size
        if any(item.batch_size != batch_size for item in self.observations):
            raise ValueError("rollout observation batch size changed over time")
        shape = (time_steps, batch_size)
        for value, label in (
            (self.episode_start, "episode_start"),
            (self.valid_mask, "valid_mask"),
            (self.terminated, "terminated"),
            (self.truncated, "truncated"),
        ):
            _time_batch(value, shape, label)
            if value.dtype != torch.bool:
                raise ValueError(f"{label} must be boolean")
        if self.forced_gate_mask is not None:
            _time_batch(self.forced_gate_mask, shape, "forced_gate_mask")
            if self.forced_gate_mask.dtype != torch.bool:
                raise ValueError("forced_gate_mask must be boolean")
            if torch.any(self.forced_gate_mask & ~self.valid_mask):
                raise ValueError("forced gate rows must be valid transitions")
            for step, action in enumerate(self.actions):
                if torch.any(self.forced_gate_mask[step] & (action.gate.to(self.forced_gate_mask.device) != GATE_ACT)):
                    raise ValueError("forced gate rows must contain ACT actions")
        for value, label in (
            (self.behavior_log_prob, "behavior_log_prob"),
            (self.behavior_value, "behavior_value"),
            (self.next_value, "next_value"),
            (self.rewards, "rewards"),
        ):
            _time_batch(value, shape, label)
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{label} must be finite floating point")
        if torch.any(self.terminated & self.truncated):
            raise ValueError("rollout transition cannot terminate and truncate")
        if (
            self.initial_state.hidden.ndim != 2
            or self.initial_state.hidden.shape[0] != batch_size
            or tuple(self.initial_state.cell.shape) != tuple(self.initial_state.hidden.shape)
        ):
            raise ValueError("rollout initial recurrent state has the wrong shape")
        if not self.policy_state_id:
            raise ValueError("rollout must bind the behavior policy state")
        for value, label in (
            (self.gate_temperature, "gate temperature"),
            (self.action_temperature, "action temperature"),
            (self.continue_temperature, "continue temperature"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"rollout {label} must be finite and positive")

    def select_lanes(self, indices: Tensor) -> "StoredRolloutV4":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("rollout lane indices must be long [N]")

        def select_time(value: Tensor) -> Tensor:
            return value.index_select(1, indices.to(value.device))

        return replace(
            self,
            observations=tuple(item.index_select(indices.to(item.match_scalars.device)) for item in self.observations),
            actions=tuple(item.index_select(indices.to(item.gate.device)) for item in self.actions),
            initial_state=self.initial_state.index_select(indices.to(self.initial_state.hidden.device)),
            **{
                name: select_time(value)
                for name, value in ((item.name, getattr(self, item.name)) for item in fields(self))
                if isinstance(value, Tensor)
            },
        )

    def advantages_and_returns(self, *, gamma: float, gae_lambda: float) -> tuple[Tensor, Tensor]:
        self.validate()
        return generalized_advantage_estimate(
            self.rewards,
            self.behavior_value,
            self.next_value,
            self.terminated,
            self.truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )


@dataclass(frozen=True, slots=True)
class PPOConfigV4:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float | None = 1.0
    target_kl: float | None = None

    def __post_init__(self) -> None:
        bounded = ((self.gamma, "gamma"), (self.gae_lambda, "gae_lambda"))
        for value, label in bounded:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"PPO {label} must be in [0,1]")
        if not math.isfinite(self.clip_range) or self.clip_range <= 0.0:
            raise ValueError("PPO clip_range must be positive")
        if self.value_clip_range is not None and (
            not math.isfinite(self.value_clip_range) or self.value_clip_range <= 0.0
        ):
            raise ValueError("PPO value_clip_range must be positive or None")
        if any(not math.isfinite(value) or value < 0.0 for value in (self.value_coefficient, self.entropy_coefficient)):
            raise ValueError("PPO loss coefficients must be non-negative")
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl <= 0.0):
            raise ValueError("PPO target_kl must be positive or None")
