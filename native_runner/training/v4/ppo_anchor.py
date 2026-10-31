"""RLHF-style IL anchor for V4 PPO: per-head KL(pi_theta || pi_ref) and its adaptive coefficient.

The reference policy is a frozen IL checkpoint (strong-v1) evaluated with no
gradient on exactly the learner's trajectory, from a zero LSTM state at each
lane's episode start, with independent reference recurrence carried between
segments of the same resident episode.

Per decision node the KL is summed over four heads (the delay head is never
part of it; under ``delay_policy="earliest"`` it is not a policy choice):

* gate and continue are binary, so the selected action's log-probability under
  each policy recovers the whole two-way distribution: ``p(other) = 1 - p(a)``.
  When the other class is masked ``p(a) = 1`` and the KL is exactly 0.  This is
  the analytic KL(pi_theta || pi_ref), differentiable in theta.
* candidate (<= 6 classes) and target (576 cells) only expose the selected
  action's log-probability without a model API change, so they use the k3
  estimator ``(r - 1) - log r`` with ``r = pi_ref(a) / pi_theta(a)``: its value is
  an unbiased (non-negative) estimate of KL(pi_theta || pi_ref) under
  pi_theta-sampled actions, and its gradient equals the gradient of the forward
  KL(pi_ref || pi_theta) (mode covering, the right direction against forgetting).
  The analytic version needs masked candidate/target log-probs from the model
  (``ActionEvaluationComponentsV4``), which is outside the trainer's files.

All four terms score learner and reference at the rollout's actual positive
temperatures. No gate ACT-weight correction is required for KL; if explicitly
requested, it is applied to learner/reference weights before their comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .tensors import GATE_ACT, GATE_WAIT


ANCHOR_KL_SCHEMA_V4 = "v4-ppo-il-anchor-kl.v1"
ANCHOR_CONTINUE_MODES = ("tied", "fixed")
GATE_LOGIT_OFFSET_EXTRA_KEY = "gate_logit_offset_applied"
_GATE_OFFSET_ATTRIBUTE = "_ppo_gate_logit_offset_applied_v4"
# Modules that only feed the critic (see UniversalCardPolicyV4._policy_context).
VALUE_ONLY_MODULE_NAMES = frozenset(
    (
        "value_head",
        "value_context_norm",
        "scene_value_skip",
        "spatial_value_skip",
        "scalar_value_skip",
        "event_value_skip",
    )
)
_LOG_RATIO_CLAMP = 20.0


def _log1mexp(log_p: Tensor) -> Tensor:
    """Stable ``log(1 - exp(log_p))`` for ``log_p <= 0``; ``-inf``-free (clamped)."""

    log_p = log_p.clamp_max(0.0)
    tiny = torch.finfo(log_p.dtype).tiny
    near_one = log_p > -math.log(2.0)
    return torch.where(
        near_one,
        torch.log((-torch.expm1(log_p)).clamp_min(tiny)),
        torch.log1p(-torch.exp(log_p).clamp_max(1.0 - torch.finfo(log_p.dtype).eps)),
    )


def binary_kl_from_selected_log_prob(learner_log_prob: Tensor, reference_log_prob: Tensor) -> Tensor:
    """KL(pi_theta || pi_ref) of a two-way head from the selected action's log-probs.

    ``p(a)`` and ``1 - p(a)`` are the whole distribution of a binary head.  A row
    whose other class is masked (``p(a) = 1`` under both policies, log-prob 0)
    gives exactly 0.  Gradients flow only through ``learner_log_prob``.
    """

    learner = learner_log_prob.float()
    reference = reference_log_prob.detach().float()
    p_selected = learner.exp()
    selected_term = p_selected * (learner - reference)
    learner_other = _log1mexp(learner)
    reference_other = _log1mexp(reference)
    p_other = (1.0 - p_selected).clamp_min(0.0)
    other_term = p_other * (learner_other - reference_other)
    # A masked other class is impossible under both policies: no contribution.
    other_term = torch.where(learner >= 0.0, torch.zeros_like(other_term), other_term)
    return (selected_term + other_term).clamp_min(0.0)


def sampled_kl_k3(learner_log_prob: Tensor, reference_log_prob: Tensor) -> Tensor:
    """k3 estimator ``(r - 1) - log r`` with ``r = pi_ref(a) / pi_theta(a)``; 0 where both are 0."""

    log_ratio = (reference_log_prob.detach().float() - learner_log_prob.float()).clamp(
        -_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP
    )
    return torch.expm1(log_ratio) - log_ratio


def anchor_kl_terms(
    learner_components: Tensor,
    reference_components: Tensor,
    *,
    gate: Tensor,
    micro_action_count: Tensor,
) -> dict[str, Tensor]:
    """Per-row KL terms from ``[..., 5]`` log-prob components (gate, cand, target, delay, continue).

    The delay column (index 3) is ignored.  Also returns the learner's and the
    reference's probability of SECOND on ACT rows (0 elsewhere, and 0 when
    SECOND is masked) for monitoring the continue head.
    """

    if learner_components.shape != reference_components.shape or learner_components.shape[-1] != 5:
        raise ValueError("anchor KL needs matching [...,5] log-prob components")
    if gate.shape != learner_components.shape[:-1] or micro_action_count.shape != gate.shape:
        raise ValueError("anchor KL action fields must match the component rows")
    act = gate == GATE_ACT
    second = act & (micro_action_count == 2)
    learner_continue = learner_components[..., 4].float()
    reference_continue = reference_components[..., 4].float()
    learner_p = learner_continue.exp()
    reference_p = reference_continue.detach().exp()
    zero = torch.zeros_like(learner_p)
    return {
        "gate": binary_kl_from_selected_log_prob(learner_components[..., 0], reference_components[..., 0]),
        "candidate": sampled_kl_k3(learner_components[..., 1], reference_components[..., 1]),
        "target": sampled_kl_k3(learner_components[..., 2], reference_components[..., 2]),
        "continue": binary_kl_from_selected_log_prob(learner_continue, reference_continue),
        "learner_second_prob": torch.where(second, learner_p, torch.where(act, 1.0 - learner_p, zero)).detach(),
        "reference_second_prob": torch.where(second, reference_p, torch.where(act, 1.0 - reference_p, zero)),
        "act": act.to(learner_p.dtype),
    }


@dataclass(frozen=True, slots=True)
class AnchorKLConfigV4:
    """IL-anchor coefficients.  ``continue_mode="tied"`` keeps beta_c / beta fixed (one adaptive scalar)."""

    initial_coefficient: float = 0.05
    target_kl: float = 0.02
    initial_continue_coefficient: float = 0.25
    continue_mode: str = "tied"
    horizon_updates: float = 4.0
    minimum_coefficient: float = 0.005
    maximum_coefficient: float = 1.0
    adaptive: bool = True
    # The first N controller updates (counted after the critic warmup, which
    # never calls the controller) may only raise beta: at the start KL ~ 0
    # would otherwise walk beta down to its minimum before the policy moves.
    no_decrease_updates: int = 0

    def __post_init__(self) -> None:
        values = (
            self.initial_coefficient,
            self.target_kl,
            self.initial_continue_coefficient,
            self.horizon_updates,
            self.minimum_coefficient,
            self.maximum_coefficient,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("anchor KL parameters must be finite")
        if self.initial_coefficient <= 0.0 or self.target_kl <= 0.0 or self.horizon_updates <= 0.0:
            raise ValueError("anchor KL coefficient, target and horizon must be positive")
        if self.initial_continue_coefficient < 0.0:
            raise ValueError("anchor continue KL coefficient cannot be negative")
        if not 0.0 < self.minimum_coefficient <= self.initial_coefficient <= self.maximum_coefficient:
            raise ValueError("anchor KL coefficient must satisfy 0 < min <= initial <= max")
        if self.continue_mode not in ANCHOR_CONTINUE_MODES:
            raise ValueError(f"anchor continue mode must be one of {ANCHOR_CONTINUE_MODES}")
        if not isinstance(self.no_decrease_updates, int) or self.no_decrease_updates < 0:
            raise ValueError("anchor no-decrease update count must be a nonnegative integer")

    def contract(self) -> dict[str, Any]:
        return {
            "schema": ANCHOR_KL_SCHEMA_V4,
            **asdict(self),
            "heads": ("gate", "candidate", "target", "continue"),
            "gate_continue_estimator": "analytic-binary-from-selected-log-prob",
            "candidate_target_estimator": "k3",
            "delay": "excluded",
            "controller": "ziegler-proportional; error=clip(kl/target-1,+-0.2); beta*=1+error/horizon",
            "controlled_kl": "gate+candidate+target per valid frame (continue excluded)",
            "reference_state": "zero LSTM state at each lane's episode start",
        }


class AdaptiveKLControllerV4:
    """Ziegler et al. (2019) proportional controller over PPO updates.

    ``beta <- beta * (1 + clip(KL / target - 1, -0.2, 0.2) / horizon)``, clamped
    to ``[min, max]``; during the first ``no_decrease_updates`` calls the
    error is floored at 0 (beta may rise, never fall).  ``tied`` continue coefficient = beta * (initial ratio).
    """

    def __init__(self, config: AnchorKLConfigV4) -> None:
        self.config = config
        self.value = float(config.initial_coefficient)
        self.updates = 0

    @property
    def continue_value(self) -> float:
        config = self.config
        if config.continue_mode == "fixed":
            return float(config.initial_continue_coefficient)
        return self.value * config.initial_continue_coefficient / config.initial_coefficient

    def update(self, observed_kl: float) -> float:
        if not math.isfinite(observed_kl) or observed_kl < 0.0:
            raise ValueError("observed anchor KL must be finite and nonnegative")
        config = self.config
        self.updates += 1
        if config.adaptive:
            error = min(0.2, max(-0.2, observed_kl / config.target_kl - 1.0))
            if self.updates <= config.no_decrease_updates:
                error = max(0.0, error)
            self.value = min(
                config.maximum_coefficient,
                max(config.minimum_coefficient, self.value * (1.0 + error / config.horizon_updates)),
            )
        return self.value

    def state_dict(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "value": self.value, "updates": self.updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> bool:
        """Restore if the stored config equals the current one; return whether it did."""

        if dict(state.get("config") or {}) != asdict(self.config):
            return False
        value = float(state["value"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("stored anchor KL coefficient is invalid")
        self.value = value
        self.updates = int(state.get("updates", 0))
        return True


def resolve_gate_logit_offset(spec: str | float, payload: Mapping[str, Any]) -> float:
    """``auto`` -> ``-ln(extra.gate_act_weight)`` (0 when absent); else a finite float."""

    if isinstance(spec, str) and spec.strip().lower() == "auto":
        extra = payload.get("extra")
        weight = None if not isinstance(extra, Mapping) else extra.get("gate_act_weight")
        if weight is None:
            return 0.0
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("checkpoint gate_act_weight must be positive")
        return -math.log(weight)
    value = float(spec)
    if not math.isfinite(value):
        raise ValueError("gate logit offset must be finite")
    return value


def stored_gate_logit_offset(payload: Mapping[str, Any]) -> float | None:
    """Offset already baked into a PPO-written checkpoint, or ``None`` for an IL checkpoint."""

    extra = payload.get("extra")
    if not isinstance(extra, Mapping) or GATE_LOGIT_OFFSET_EXTRA_KEY not in extra:
        return None
    return float(extra[GATE_LOGIT_OFFSET_EXTRA_KEY])


def apply_gate_logit_offset_once(model: nn.Module, payload: Mapping[str, Any], spec: str | float) -> float:
    """Add the ACT-logit correction to the gate bias exactly once (same edit as head-to-head).

    A checkpoint written by PPO already carries the correction in its gate bias
    (``extra.gate_logit_offset_applied``) and is left unchanged.  Returns the
    total offset now baked into ``model``.
    """

    if model.__dict__.get(_GATE_OFFSET_ATTRIBUTE) is not None:
        raise RuntimeError("gate logit offset was already applied to this model")
    stored = stored_gate_logit_offset(payload)
    if stored is not None:
        model.__dict__[_GATE_OFFSET_ATTRIBUTE] = stored
        return stored
    offset = resolve_gate_logit_offset(spec, payload)
    gate = model.gate_head[-1]  # type: ignore[index]
    if not isinstance(gate, nn.Linear) or gate.out_features != 2 or gate.bias is None:
        raise TypeError("gate head must end in a two-way Linear layer with bias")
    if GATE_ACT == GATE_WAIT:
        raise RuntimeError("gate ACT/WAIT indices collide")
    if offset:
        with torch.no_grad():
            gate.bias[GATE_ACT] += float(offset)
    model.__dict__[_GATE_OFFSET_ATTRIBUTE] = float(offset)
    return float(offset)


def value_only_parameter_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(name for name, _ in model.named_parameters() if name.split(".", 1)[0] in VALUE_ONLY_MODULE_NAMES)
    if not names:
        raise ValueError("model has no value-only parameters")
    return names
