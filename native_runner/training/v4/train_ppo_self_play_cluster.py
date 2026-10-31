"""Multi-GPU, multi-host resident PPO self-play trainer for distributed training."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time
import uuid
from typing import Any, Mapping, Sequence
from queue import Empty, Queue

import torch
import torch.distributed as dist

from ...battle_env import DEFAULT_KING_POTENTIAL_WEIGHT, KING_SHAPING_MODES
from ...match_factory import NATIVE_GAMEPLAY_END_TICK
from .async_cluster_self_play import AsyncResidentEngineSpecV4, collect_async_cluster_wave_v4
from .checkpoint import (
    CHECKPOINT_SCHEMA_V4,
    checkpoint_contract,
    model_config_from_contract,
    read_actor_critic_checkpoint,
    restore_actor_critic_checkpoint,
    save_actor_critic_checkpoint,
)
from .cluster_self_play import ClusterCollectionV4, ClusterMatchAssignmentV4
from .deck_pool import DeckIdentityV4, DeckPoolManifestV4
from .decoding import DELAY_POLICIES, DELAY_POLICY_EARLIEST, DELAY_POLICY_SAMPLE, ShadowCandidateLegality
from .distributed_ppo import distributed_ppo_update_v4
from .imitation import ILLossWeightsV4
from .king_activation_reward import (
    DEFAULT_KING_ACTIVATION_PENALTY,
    DEFAULT_KING_ACTIVATION_SCOPE,
    KING_ACTIVATION_SCOPES,
    validate_king_reward_parameters_v4,
)
from .ppo_anchor import (
    ANCHOR_CONTINUE_MODES,
    GATE_LOGIT_OFFSET_EXTRA_KEY,
    AdaptiveKLControllerV4,
    AnchorKLConfigV4,
    apply_gate_logit_offset_once,
)
from .ppo_expert_bc import HogExpertSamplerV4, distributed_expert_bc_update_v4, expert_bc_contract_v4
from .factory import build_production_model_v4
from .league import LeagueEntryV4, PolicyLeagueV4
from .policy_session import LoadedPolicyV4, load_policy_v4
from .matchmaking import (
    BATTLE_LEVELS,
    MATCH_LEARNER_FOCUS,
    MATCH_LEARNER_HARD,
    MATCH_LEARNER_WEIGHTED,
    POLICY_HISTORY,
    POLICY_IL,
    DeckMatchmakerV4,
    FixedLearnerOpponentDeckSamplerV4,
    MatchmakerConfigV4,
)
from .model import UniversalCardPolicyV4
from .ppo import PPOConfigV4, StoredRolloutV4
from .ppo_reference_state import ReferenceStateCacheV4
from .ppo_runtime import ppo_environment_config_v4, ppo_ruleset_id_v4, validate_reward_parameters_v4
from .tensors import GATE_ACT, GATE_WAIT


DEFAULT_GATE_TEMPERATURE_START = 0.20
DEFAULT_GATE_TEMPERATURE_END = 0.20
DEFAULT_GATE_TEMPERATURE_ANNEAL_UPDATES = 1
DEFAULT_ACTION_TEMPERATURE = 1.0
DEFAULT_CONTINUE_TEMPERATURE = 2.0
ANCHOR_TEMPERATURE = 1.0
# Training uses the EnvironmentConfigV1 warmup; evaluation has separate knobs.
DEFAULT_ENVIRONMENT_WARMUP_TICKS = 130
DEFAULT_CRITIC_WARMUP_ITERS = 10
DEFAULT_OPENING_FORCE_AFTER_SECONDS = 15
RESUME_CONTRACT_SCHEMA = "v4-ppo-strict-resume.v2"
HISTORY_CANDIDATES_PER_BATCH = 8
SPECIAL_TOWER_RUNTIME_IDS = frozenset((159_000_002, 159_000_004))
WANDB_PROJECT = "cr-ai-ppo"


def _gate_temperature_for_update(update_step: int, *, start: float, end: float, anneal_updates: int) -> float:
    """Return the behavior temperature for one 1-indexed rollout update."""

    if update_step <= 0 or anneal_updates <= 0:
        raise ValueError("gate temperature schedule steps must be positive")
    if not math.isfinite(start) or not math.isfinite(end) or start <= 0.0 or end <= 0.0 or end > start:
        raise ValueError("gate temperatures must satisfy 0 < end <= start")
    if anneal_updates == 1:
        return float(end)
    progress = min(1.0, (update_step - 1) / (anneal_updates - 1))
    return float(start + (end - start) * progress)


def _set_temperatures(
    model: UniversalCardPolicyV4,
    *,
    gate_temperature: float = DEFAULT_GATE_TEMPERATURE_START,
    continue_temperature: float = DEFAULT_CONTINUE_TEMPERATURE,
) -> None:
    model.set_ppo_gate_temperature(gate_temperature)
    model.set_ppo_action_temperature(DEFAULT_ACTION_TEMPERATURE)
    model.set_ppo_continue_temperature(continue_temperature)


def _rollout_behavior_counts(rollout: StoredRolloutV4, model: UniversalCardPolicyV4) -> torch.Tensor:
    """Count gate eligibility and sampled downstream choices on CPU."""

    from .decoding import elixir_multiplier_from_match_scalars

    delay_bins = len(model.config.delay_offset_ms)
    counts = torch.zeros(10 + delay_bins, dtype=torch.long)
    forced_gate_mask = rollout.forced_gate_mask
    for step, (observation, action) in enumerate(zip(rollout.observations, rollout.actions, strict=True)):
        valid = rollout.valid_mask[step].cpu()
        forced_gate = torch.zeros_like(valid) if forced_gate_mask is None else forced_gate_mask[step].cpu()
        shadow = ShadowCandidateLegality(
            observation.candidates,
            model.config,
            elixir_multiplier=elixir_multiplier_from_match_scalars(observation.match_scalars),
        )
        legal_act = shadow.candidate_mask().any(dim=-1).cpu()
        eligible = valid & legal_act
        actual_act = valid & (action.gate.cpu() == GATE_ACT)
        original_wait = (action.gate.cpu() == GATE_WAIT) | forced_gate
        if torch.any(forced_gate & (~eligible | ~actual_act)):
            raise RuntimeError("forced ACT rollout telemetry is inconsistent")
        counts[0] += valid.sum()
        counts[1] += eligible.sum()
        counts[2] += (eligible & original_wait).sum()
        counts[3] += forced_gate.sum()
        counts[4] += actual_act.sum()
        micro_action_count = action.micro_action_count.cpu()
        counts[5] += (actual_act & (micro_action_count == 1)).sum()
        counts[6] += (actual_act & (micro_action_count == 2)).sum()
        shadow.apply(
            actual_act,
            action.candidate_index[:, 0].cpu(),
            action.target_cell[:, 0].cpu(),
            action.delay_offset_bin[:, 0].cpu(),
            step=0,
        )
        continue_eligible = actual_act & shadow.candidate_mask().any(dim=-1)
        continue_second = actual_act & (micro_action_count == 2)
        if torch.any(continue_second & ~continue_eligible):
            raise RuntimeError("sampled SECOND action had no legal opportunity")
        counts[7] += continue_eligible.sum()
        counts[8] += (continue_eligible & ~continue_second).sum()
        active_micro = (
            torch.arange(model.config.max_micro_actions, dtype=torch.long)[None] < micro_action_count[:, None]
        ) & valid[:, None]
        delay_values = action.delay_offset_bin.cpu()[active_micro]
        if delay_values.numel():
            delay_counts = torch.bincount(delay_values, minlength=delay_bins)
            counts[9] += delay_values.numel()
            counts[10:] += delay_counts[:delay_bins]
    return counts


def _rollout_behavior_metrics(counts: torch.Tensor, *, delay_offset_ms: Sequence[int]) -> dict[str, int | float]:
    if counts.dtype != torch.long or counts.ndim != 1:
        raise TypeError("rollout behavior counts must be a 1D int64 tensor")
    if counts.numel() != 10 + len(delay_offset_ms) or torch.any(counts < 0):
        raise ValueError("rollout behavior counts have the wrong shape")

    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator) / max(float(denominator), 1.0)

    eligible = int(counts[1])
    eligible_wait = int(counts[2])
    act = int(counts[4])
    continue_stop = int(counts[5])
    continue_second = int(counts[6])
    continue_eligible = int(counts[7])
    continue_eligible_stop = int(counts[8])
    delay_total = int(counts[9])
    if continue_stop + continue_second != act:
        raise RuntimeError("continue counts do not partition ACT rows")
    if continue_eligible_stop + continue_second != continue_eligible:
        raise RuntimeError("eligible continue counts do not partition opportunities")
    if int(counts[10:].sum()) != delay_total:
        raise RuntimeError("delay counts do not partition sampled actions")
    result: dict[str, int | float] = {
        "behavior_valid_ticks": int(counts[0]),
        "act_eligible_ticks": eligible,
        "eligible_wait_ticks": eligible_wait,
        "eligible_wait_rate": fraction(eligible_wait, eligible),
        "eligible_act_rate": fraction(eligible - eligible_wait, eligible),
        "forced_opening_act_ticks": int(counts[3]),
        "sampled_act_ticks": act,
        "continue_stop_count": continue_stop,
        "continue_second_count": continue_second,
        "continue_stop_fraction": fraction(continue_stop, act),
        "continue_second_fraction": fraction(continue_second, act),
        "continue_eligible_ticks": continue_eligible,
        "continue_eligible_stop_count": continue_eligible_stop,
        "continue_eligible_stop_rate": fraction(continue_eligible_stop, continue_eligible),
        "continue_eligible_second_rate": fraction(continue_second, continue_eligible),
        "delay_action_count": delay_total,
    }
    for index, milliseconds in enumerate(delay_offset_ms):
        count = int(counts[10 + index])
        result[f"delay_{milliseconds}ms_count"] = count
        result[f"delay_{milliseconds}ms_fraction"] = fraction(count, delay_total)
    return result


def _model_for_payload(payload: Mapping[str, Any], *, device: torch.device) -> UniversalCardPolicyV4:
    """Build the architecture the checkpoint was trained with (``contract.model_config``).

    Same construction as ``policy_session.load_policy_v4``: strong-v1 uses CNN
    128x5 / LSTM 768; keys absent from older checkpoints fall back to defaults.
    """

    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint has no model/catalog contract")
    return build_production_model_v4(device=device, config=model_config_from_contract(contract))


def _load_contract_model(
    path: str | Path,
    *,
    device: torch.device,
    delay_policy: str,
    gate_logit_offset: str | float,
    optimizer_factory: Any | None = None,
) -> tuple[UniversalCardPolicyV4, dict[str, Any], torch.optim.Optimizer | None]:
    """read -> build from contract -> restore -> delay policy -> gate correction (all before capture)."""

    if delay_policy not in DELAY_POLICIES:
        raise ValueError(f"delay policy must be one of {DELAY_POLICIES}")
    payload = read_actor_critic_checkpoint(path, map_location=device)
    model = _model_for_payload(payload, device=device)
    optimizer = None if optimizer_factory is None else optimizer_factory(model)
    payload = restore_actor_critic_checkpoint(payload, model, optimizer=optimizer)
    model.set_delay_policy(delay_policy)
    applied = apply_gate_logit_offset_once(model, payload, gate_logit_offset)
    model.__dict__["_ppo_gate_logit_offset_v4"] = applied
    return model, payload, optimizer


def _gate_logit_offset_of(model: UniversalCardPolicyV4) -> float:
    return float(model.__dict__.get("_ppo_gate_logit_offset_v4", 0.0))


def _load_frozen_model(
    path: str | Path,
    *,
    device: torch.device,
    delay_policy: str = DELAY_POLICY_SAMPLE,
    gate_logit_offset: str | float = 0.0,
    gate_temperature: float = DEFAULT_GATE_TEMPERATURE_START,
    continue_temperature: float = DEFAULT_CONTINUE_TEMPERATURE,
    sampling: bool = False,
) -> tuple[UniversalCardPolicyV4, dict[str, Any]]:
    """Frozen KL reference, calibrated to the learner's training distribution.

    Battle opponents use _load_interface_opponent_model instead: training gate
    corrections and temperatures must not leak into deployed argmax inference.
    """

    model, payload, _optimizer = _load_contract_model(
        path, device=device, delay_policy=delay_policy, gate_logit_offset=gate_logit_offset
    )
    _set_temperatures(model, gate_temperature=gate_temperature, continue_temperature=continue_temperature)
    model.set_ppo_rollout_sampling(bool(sampling))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _load_interface_opponent_model(path: str | Path, *, device: torch.device) -> LoadedPolicyV4:
    """Use the launch_interface.cmd default loader and deterministic decoding.

    Keep checkpoint gate weights and temperatures exactly as live inference
    loads them, including any correction already baked into a PPO snapshot.
    The launcher's default delay policy is earliest; learner settings are not
    opponent inference settings.
    """

    loaded = load_policy_v4(path, device=device, delay_policy=DELAY_POLICY_EARLIEST)
    loaded.model.set_ppo_rollout_sampling(False)
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    return loaded


def _training_opening_force_options(args: argparse.Namespace) -> dict[str, int | None]:
    if args.opening_force_after_seconds <= 0:
        raise ValueError("training opening timeout must be positive")
    if args.il_opening_force_after_seconds is not None and args.il_opening_force_after_seconds <= 0:
        raise ValueError("IL opening timeout must be positive")
    if args.il_opening_force_after_seconds not in (None, args.opening_force_after_seconds):
        raise ValueError("all training owners share --opening-force-after-seconds; IL timeout must match")
    return {"force_all_owners_opening_after_seconds": args.opening_force_after_seconds}


def _resolve_temperatures(args: argparse.Namespace, *, anchor_enabled: bool) -> tuple[float, float, float]:
    """Use positive behavior temperatures; KL scores both models at these temperatures."""
    defaults = ((ANCHOR_TEMPERATURE,) * 3 if anchor_enabled else
                (DEFAULT_GATE_TEMPERATURE_START, DEFAULT_GATE_TEMPERATURE_END, DEFAULT_CONTINUE_TEMPERATURE))
    values = tuple(default if value is None else float(value) for default, value in zip(defaults, (
        args.gate_temperature_start, args.gate_temperature_end, args.continue_temperature,
    ), strict=True))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("behavior temperatures must be finite and positive")
    return values


def _anchor_kl_config(args: argparse.Namespace) -> AnchorKLConfigV4 | None:
    if args.anchor_checkpoint is None:
        return None
    coefficient = float(args.anchor_kl_coef)
    continue_coefficient = (
        5.0 * coefficient if args.anchor_continue_kl_coef is None else float(args.anchor_continue_kl_coef)
    )
    return AnchorKLConfigV4(
        initial_coefficient=coefficient,
        target_kl=float(args.anchor_target_kl),
        initial_continue_coefficient=continue_coefficient,
        continue_mode=str(args.anchor_continue_kl_mode),
        horizon_updates=float(args.anchor_kl_horizon),
        minimum_coefficient=float(args.anchor_kl_coef_min),
        maximum_coefficient=float(args.anchor_kl_coef_max),
        adaptive=not bool(args.anchor_kl_fixed),
        no_decrease_updates=int(args.anchor_kl_no_decrease_updates),
    )


def _king_reward_parameters(args: argparse.Namespace) -> dict[str, object]:
    """The collector/worker defaults are neutral, so always pass all four."""

    return {
        "king_activation_penalty": float(args.king_activation_penalty),
        "king_activation_scope": str(args.king_activation_scope),
        "king_shaping_mode": str(args.king_shaping_mode),
        "king_potential_weight": float(args.king_potential_weight),
    }


# Keys of the strict part of the resume contract: architecture / observation /
# action contract and the data contract that the restored matchmaker, sampler,
# league and rank RNG states depend on.  Everything else (reward, IL anchor,
# opponent mix, PPO hyperparameters, critic warmup) may change on resume and is
# only recorded.
STRICT_RESUME_KEYS = (
    "schema",
    "learner_contract",
    "delay_policy",
    "gate_logit_offset",
    "behavior_temperatures",
    "gate_temperature_schedule",
    "deck_pool",
    # engine_count is recorded only: CPU nodes may join or leave between resumes.
    "resident_slots_per_engine",
    "world_size",
    "environment_warmup_ticks",
    "decision_ticks",
    "battle_levels",
    "battle_level_split",
    "seed",
    "learner_deck_id",
    "fixed_learner_opponent_mix",
    "focus_tail_current_focus_percent",
    "expert_bc",
)


def _split_resume_contract(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    strict = {key: contract.get(key) for key in STRICT_RESUME_KEYS}
    loose = {key: value for key, value in contract.items() if key not in STRICT_RESUME_KEYS}
    return strict, loose


def _check_resume_contract(stored: object, current: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Strictly compare architecture/observation/action and data keys; return loose changes."""

    if not isinstance(stored, Mapping):
        raise ValueError("resume checkpoint has no training contract")
    if stored.get("schema") != current.get("schema"):
        raise ValueError(
            f"resume checkpoint contract schema {stored.get('schema')!r} != {current.get('schema')!r} "
            "(pre-v2 strict resumes are intentionally incompatible)"
        )
    stored_strict, stored_loose = _split_resume_contract(stored)
    current_strict, current_loose = _split_resume_contract(current)
    differences = sorted(key for key in STRICT_RESUME_KEYS if stored_strict[key] != current_strict[key])
    if differences:
        raise ValueError(f"resume checkpoint architecture/data contract changed: {differences}")
    return {
        key: {"checkpoint": stored_loose.get(key), "current": current_loose.get(key)}
        for key in sorted(set(stored_loose).union(current_loose))
        if stored_loose.get(key) != current_loose.get(key)
    }


def _checkpoint_identity(path: str | Path) -> tuple[str, int]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_V4:
        raise ValueError("unsupported fixed-opponent checkpoint")
    checkpoint_id = str(payload.get("checkpoint_id", ""))
    update_step = int(payload.get("update_step", -1))
    if not checkpoint_id or update_step < 0:
        raise ValueError("fixed-opponent checkpoint identity is invalid")
    return checkpoint_id, update_step


def _json_log(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = tuple(rows[0])
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _combined_mean_std(
    rows: Sequence[Mapping[str, object]], *, mean_key: str, std_key: str, weight_key: str
) -> tuple[float, float]:
    total_weight = sum(float(row[weight_key]) for row in rows)
    if total_weight <= 0.0:
        return 0.0, 0.0
    total = sum(float(row[weight_key]) * float(row[mean_key]) for row in rows)
    square_total = sum(float(row[weight_key]) * (float(row[std_key]) ** 2 + float(row[mean_key]) ** 2) for row in rows)
    mean = total / total_weight
    variance = max(0.0, square_total / total_weight - mean * mean)
    return mean, math.sqrt(variance)


KING_ACTIVATION_SOURCE_FAMILIES = ("spell", "troop", "building", "other", "unknown")


def _king_activation_batch_metrics(reports: Sequence[Mapping[str, object]]) -> dict[str, int | float]:
    """Generalized king-activation statistics, indexed by the learner owner."""

    owners = [
        (report, int(owner)) for report in reports if "king_activations" in report for owner in report["current_owners"]
    ]
    if not owners:
        return {}
    count = len(owners)
    result: dict[str, int | float] = {
        "batch_performance/king_activation_rate": sum(
            int(report["king_activations"][owner]) for report, owner in owners  # type: ignore[index]
        )
        / count,
        "batch_performance/king_activated_against_rate": sum(
            int(report["king_activations"][1 - owner]) for report, owner in owners  # type: ignore[index]
        )
        / count,
        "batch_performance/king_activation_reward_mean": sum(
            float(report["king_activation_reward_total"][owner]) for report, owner in owners  # type: ignore[index]
        )
        / count,
        "batch_performance/king_activation_unattributed_rate": sum(
            int(report["king_activation_source_unattributed"][owner]) for report, owner in owners  # type: ignore[index]
        )
        / count,
    }
    ticks = [
        report["king_activation_tick"][owner]  # type: ignore[index]
        for report, owner in owners
        if int(report["king_activations"][owner]) and report["king_activation_tick"][owner] is not None  # type: ignore[index]
    ]
    if ticks:
        result["batch_performance/king_activation_tick_mean"] = sum(float(tick) for tick in ticks) / len(ticks)
    families = {family: 0 for family in KING_ACTIVATION_SOURCE_FAMILIES}
    for report, owner in owners:
        for detail in report.get("king_activation_details", ()) or ():  # type: ignore[union-attr]
            if isinstance(detail, Mapping) and int(detail.get("owner", -1)) == owner:
                family = str(detail.get("source_family", "unknown"))
                families[family if family in families else "unknown"] += 1
    for family, value in families.items():
        result[f"batch_performance/king_activation_source/{family}"] = value
    return result


def _batch_wandb_metrics(
    segments: Sequence[Mapping[str, object]], reports: Sequence[Mapping[str, object]]
) -> dict[str, int | float]:
    """Collapse one synchronized match wave into one comparable W&B point."""

    if not segments:
        raise ValueError("batch visualization requires at least one segment")
    batch_index = int(segments[0]["batch_index"])
    if any(int(row["batch_index"]) != batch_index for row in segments):
        raise ValueError("batch visualization crossed a match-wave boundary")
    if not bool(segments[-1]["batch_complete"]):
        raise ValueError("batch visualization requires a completed match wave")

    weight_key = "learner_owner_frames"
    total_frames = sum(int(row[weight_key]) for row in segments)
    if total_frames <= 0:
        raise ValueError("batch visualization has no learner frames")

    def weighted(key: str) -> float:
        return sum(int(row[weight_key]) * float(row[key]) for row in segments) / total_frames

    return_mean, return_std = _combined_mean_std(
        segments, mean_key="return_mean", std_key="return_std", weight_key=weight_key
    )
    reward_mean, reward_std = _combined_mean_std(
        segments, mean_key="reward_mean", std_key="reward_std", weight_key=weight_key
    )
    advantage_mean, advantage_std = _combined_mean_std(
        segments, mean_key="advantage_mean", std_key="advantage_std", weight_key=weight_key
    )
    behavior_value_mean = weighted("behavior_value_mean")

    error_total = 0.0
    error_square_total = 0.0
    for row in segments:
        weight = int(row[weight_key])
        segment_return_mean = float(row["return_mean"])
        segment_return_variance = float(row["return_std"]) ** 2
        segment_error_mean = float(row["behavior_value_mean"]) - segment_return_mean
        segment_error_variance = max(0.0, (1.0 - float(row["value_explained_variance"])) * segment_return_variance)
        error_total += weight * segment_error_mean
        error_square_total += weight * (segment_error_variance + segment_error_mean**2)
    error_mean = error_total / total_frames
    error_variance = max(0.0, error_square_total / total_frames - error_mean**2)
    return_variance = return_std**2
    explained_variance = 1.0 - error_variance / return_variance if return_variance > 1e-12 else 0.0

    eligible = sum(int(row["act_eligible_ticks"]) for row in segments)
    eligible_wait = sum(int(row["eligible_wait_ticks"]) for row in segments)
    continue_stop = sum(int(row["continue_stop_count"]) for row in segments)
    continue_second = sum(int(row["continue_second_count"]) for row in segments)
    continue_total = continue_stop + continue_second
    delay_counts = {
        milliseconds: sum(int(row[f"delay_{milliseconds}ms_count"]) for row in segments)
        for milliseconds in (0, 50, 100, 150, 200)
    }
    delay_total = sum(delay_counts.values())
    truncated = sum(int(row["collection_truncated_count"]) for row in segments)
    bootstrapped = sum(int(row["collection_boundary_bootstrap_count"]) for row in segments)

    result: dict[str, int | float] = {
        "batch/index": batch_index,
        "batch/end_update": int(segments[-1]["update"]),
        "batch/segments": len(segments),
        "batch_data_quality/learner_frames": total_frames,
        "batch_data_quality/min_segment_frames": min(int(row[weight_key]) for row in segments),
        "batch_data_quality/max_segment_frames": max(int(row[weight_key]) for row in segments),
        "batch_data_quality/rejected_actions": sum(int(row["native_rejected_actions"]) for row in segments),
        "batch_data_quality/boundary_bootstrap_coverage": (bootstrapped / truncated if truncated else 1.0),
        "batch_data_quality/initial_lstm_abs_max": max(
            float(row["collection_initial_hidden_abs_max"]) for row in segments
        ),
        "batch_critic/value_loss": weighted("value"),
        "batch_critic/target_return_mean": return_mean,
        "batch_critic/target_return_std": return_std,
        "batch_critic/behavior_value_mean": behavior_value_mean,
        "batch_critic/explained_variance": explained_variance,
        "batch_performance/train_reward_mean": reward_mean,
        "batch_performance/train_reward_std": reward_std,
        "batch_optimization/policy_loss": weighted("policy"),
        "batch_optimization/total_loss": weighted("total"),
        "batch_optimization/entropy": weighted("entropy"),
        "batch_optimization/approx_kl": weighted("approx_kl"),
        "batch_optimization/clip_fraction": weighted("clip_fraction"),
        "batch_optimization/advantage_mean": advantage_mean,
        "batch_optimization/advantage_std": advantage_std,
        "batch_exploration/act_eligible_ticks": eligible,
        "batch_exploration/eligible_wait_ticks": eligible_wait,
        "batch_exploration/eligible_wait_rate": (eligible_wait / max(eligible, 1)),
        "batch_exploration/eligible_act_rate": ((eligible - eligible_wait) / max(eligible, 1)),
        "batch_exploration/forced_opening_act_ticks": sum(int(row["forced_opening_act_ticks"]) for row in segments),
        "batch_exploration/continue_stop_count": continue_stop,
        "batch_exploration/continue_second_count": continue_second,
        "batch_exploration/continue_stop_fraction": (continue_stop / max(continue_total, 1)),
        "batch_exploration/continue_second_fraction": (continue_second / max(continue_total, 1)),
    }
    for milliseconds, count in delay_counts.items():
        result[f"batch_exploration/delay_{milliseconds}ms_count"] = count
        result[f"batch_exploration/delay_{milliseconds}ms_fraction"] = count / max(delay_total, 1)
    if "episode_failure_rate" in segments[-1]:
        result["batch_data_quality/episode_failure_rate"] = float(segments[-1]["episode_failure_rate"])  # type: ignore[arg-type]
        result["batch_data_quality/episode_failures"] = int(segments[-1]["wave_episode_failures"])  # type: ignore[call-overload]
    continue_eligible = sum(int(row.get("continue_eligible_ticks", 0)) for row in segments)
    if continue_eligible:
        result["batch_exploration/continue_eligible_second_rate"] = continue_second / continue_eligible
    if all("anchor_kl" in row for row in segments):
        for key in (
            "anchor_kl",
            "anchor_kl_gate",
            "anchor_kl_candidate",
            "anchor_kl_target",
            "anchor_kl_continue",
            "anchor_loss",
            "anchor_learner_second_prob",
            "anchor_reference_second_prob",
        ):
            result[f"batch_anchor/{key.removeprefix('anchor_')}"] = weighted(key)
        result["batch_anchor/kl_coefficient"] = float(segments[-1]["anchor_kl_coefficient_next"])
        result["batch_anchor/continue_kl_coefficient"] = float(segments[-1]["anchor_continue_kl_coefficient_next"])

    report_metrics = _wandb_report_metrics(reports)
    for key, value in report_metrics.items():
        if key.startswith("performance/vs_"):
            result[f"batch_{key}"] = value
        elif key.startswith("performance/by_deck_mix/"):
            result[f"batch_{key}"] = value
        elif key.startswith("performance/scoreline/"):
            result[f"batch_{key}"] = value
        elif key in ("performance/average_game_seconds", "performance/overtime_rate"):
            result[f"batch_{key}"] = value
    result["batch_performance/completed_games"] = len(reports)
    king_owners = [
        (report, int(owner))
        for report in reports
        if "fireball_king_activations" in report
        for owner in report["current_owners"]
    ]
    if king_owners:
        for field_name in (
            "fireball_king_activations",
            "fireball_king_activation_penalty_total",
            "sleeping_enemy_king_damage_count",
            "king_activation_not_attributed_to_fireball",
        ):
            result[f"batch_performance/{field_name}_mean"] = sum(
                report[field_name][owner] for report, owner in king_owners
            ) / len(king_owners)
    result.update(_king_activation_batch_metrics(reports))
    hog_owners = [
        (report, int(owner)) for report in reports if "hog_deployments" in report for owner in report["current_owners"]
    ]
    if hog_owners:
        result["batch_performance/hog_deployments_mean"] = sum(
            report["hog_deployments"][owner] for report, owner in hog_owners
        ) / len(hog_owners)
        result["batch_performance/zero_hog_deployment_rate"] = sum(
            report["hog_deployments"][owner] == 0 for report, owner in hog_owners
        ) / len(hog_owners)
        result["batch_performance/hog_deploy_reward_mean"] = sum(
            report["hog_deploy_reward_total"][owner] for report, owner in hog_owners
        ) / len(hog_owners)
        if all("first_hog_timing_reward_total" in report for report, _ in hog_owners):
            result["batch_performance/first_hog_timing_reward_mean"] = sum(
                report["first_hog_timing_reward_total"][owner] for report, owner in hog_owners
            ) / len(hog_owners)
            result["batch_performance/first_hog_deferral_penalty_mean"] = sum(
                report["first_hog_deferral_penalty_total"][owner] for report, owner in hog_owners
            ) / len(hog_owners)
            result["batch_performance/first_hog_deadline_penalty_mean"] = sum(
                report["first_hog_deadline_penalty_total"][owner] for report, owner in hog_owners
            ) / len(hog_owners)
            result["batch_performance/first_hog_deadline_miss_rate"] = sum(
                report["first_hog_deadline_penalty_total"][owner] > 0.0 for report, owner in hog_owners
            ) / len(hog_owners)
        first_ticks = [
            report["first_hog_deploy_tick"][owner]
            for report, owner in hog_owners
            if report["first_hog_deploy_tick"][owner] is not None
        ]
        if first_ticks:
            result["batch_performance/first_hog_seconds_when_played"] = sum(first_ticks) / len(first_ticks) / 20.0
            result["batch_performance/first_hog_before_30_seconds_rate"] = sum(
                tick < 30 * 20 for tick in first_ticks
            ) / len(hog_owners)
    return result


def _init_offline_wandb(output_dir: Path, config: Mapping[str, object]) -> Any:
    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        name=output_dir.name,
        job_type="train",
        dir=str(output_dir),
        mode="offline",
        config=dict(config),
        settings=wandb.Settings(console="off"),
    )
    if run is None:
        raise RuntimeError("W&B offline initialization returned no run")
    run.define_metric("update")
    run.define_metric("*", step_metric="update")
    run.define_metric("batch/index")
    for namespace in (
        "batch_anchor/*",
        "batch_critic/*",
        "batch_data_quality/*",
        "batch_exploration/*",
        "batch_optimization/*",
        "batch_performance/*",
    ):
        run.define_metric(namespace, step_metric="batch/index")
    return run


def _rank_rng_state(device: torch.device) -> dict[str, object]:
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}


def _set_rank_rng_state(state: Mapping[str, object], *, device: torch.device) -> None:
    python_state = state.get("python")
    torch_state = state.get("torch")
    cuda_state = state.get("cuda")
    if not isinstance(torch_state, torch.Tensor) or not isinstance(cuda_state, torch.Tensor):
        raise ValueError("PPO rank RNG state is invalid")
    random.setstate(python_state)  # type: ignore[arg-type]
    torch.set_rng_state(torch_state.cpu())
    torch.cuda.set_rng_state(cuda_state.cpu(), device)


def _restore_rank_rng_state(payload: Mapping[str, object], *, rank: int, world_size: int, device: torch.device) -> None:
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("resume checkpoint lacks strict PPO state")
    states = extra.get("rank_rng_states")
    if not isinstance(states, (list, tuple)) or len(states) != world_size:
        raise ValueError("resume checkpoint rank RNG topology changed")
    state = states[rank]
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint rank RNG state is invalid")
    _set_rank_rng_state(state, device=device)


PPO_PHASES = ("rollout", "update", "critic_warmup", "idle")
DEFAULT_MAX_EPISODE_FAILURE_RATE = 0.02


def _episode_failures_from_shards(shards: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Failure records the rank collectors attached to their segment timing."""

    rows: list[dict[str, object]] = []
    for rank, shard in enumerate(shards):
        timing = shard.get("timing")
        if not isinstance(timing, Mapping):
            continue
        for row in timing.get("episode_failures", ()) or ():
            if isinstance(row, Mapping):
                rows.append({**row, "rank": rank})
    return tuple(rows)


def _episode_failure_rate(wave_failures: int, wave_matches: int, *, max_rate: float) -> tuple[float, bool]:
    """Failure rate of the current wave and whether it is within ``max_rate``."""

    if wave_matches <= 0:
        raise ValueError("episode failure rate needs a positive wave size")
    rate = float(wave_failures) / float(wave_matches)
    return rate, rate <= max_rate + 1e-12


def _write_phase(path: Path, phase: str, iteration: int) -> None:
    """Atomically publish the trainer phase for the GPU filler (``<run_dir>/phase.json``).

    ``rollout`` while the collector runs (GPU mostly idle), ``update`` /
    ``critic_warmup`` while optimizing, ``idle`` on exit.  Written to a
    temporary file in the same directory, then ``os.replace``d.
    """

    if phase not in PPO_PHASES:
        raise ValueError(f"PPO phase must be one of {PPO_PHASES}")
    payload = {"phase": phase, "iteration": int(iteration), "ts": time.time()}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_strict_checkpoint(output_dir: Path, checkpoint_path: Path) -> None:
    latest = output_dir / "latest.pt"
    temporary = output_dir / f".latest.{uuid.uuid4().hex}.tmp"
    temporary.symlink_to(checkpoint_path.relative_to(output_dir))
    temporary.replace(latest)
    for previous in (output_dir / "checkpoints").glob("update-*.pt"):
        if previous != checkpoint_path:
            previous.unlink()


def _wandb_report_metrics(reports: Sequence[Mapping[str, object]]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    if reports:
        game_seconds = tuple(min(float(report["terminal_tick"]), NATIVE_GAMEPLAY_END_TICK) / 20.0 for report in reports)
        result["performance/average_game_seconds"] = sum(game_seconds) / len(game_seconds)
        result["performance/overtime_rate"] = sum(seconds > 180.0 for seconds in game_seconds) / len(game_seconds)
        scorelines = {(1, 0): "1_0", (2, 1): "2_1", (3, 2): "3_2", (3, 1): "3_1", (3, 0): "3_0", (2, 0): "2_0"}
        scoreline_counts = {label: 0 for label in scorelines.values()}
        other_scorelines = 0
        for report in reports:
            crowns = tuple(sorted((int(value) for value in report["crowns"]), reverse=True))
            label = scorelines.get(crowns)
            if label is None:
                other_scorelines += 1
            else:
                scoreline_counts[label] += 1
        for label, count in scoreline_counts.items():
            result[f"performance/scoreline/{label}_ratio"] = count / len(reports)
        result["performance/scoreline/other_ratio"] = other_scorelines / len(reports)
    for policy, label in ((POLICY_IL, "il"), (POLICY_HISTORY, "history")):
        rows: list[tuple[float, float]] = []
        for report in reports:
            if report["policy_matchup"] != policy:
                continue
            owners = tuple(int(value) for value in report["current_owners"])
            values = tuple(float(value) for value in report["result"])
            crowns = tuple(float(value) for value in report["crowns"])
            owner = owners[0]
            rows.append((values[owner], crowns[owner] - crowns[1 - owner]))
        if rows:
            outcomes = tuple(row[0] for row in rows)
            result[f"performance/vs_{label}/games"] = len(rows)
            result[f"performance/vs_{label}/score"] = sum(
                1.0 if value > 0.0 else 0.5 if value == 0.0 else 0.0 for value in outcomes
            ) / len(outcomes)
            result[f"performance/vs_{label}/win_rate"] = sum(value > 0.0 for value in outcomes) / len(outcomes)
            result[f"performance/vs_{label}/draw_rate"] = sum(value == 0.0 for value in outcomes) / len(outcomes)
            result[f"performance/vs_{label}/mean_crown_difference"] = sum(row[1] for row in rows) / len(rows)
    for category in (
        "focus-focus",
        "focus-tail-coverage",
        "focus-tail-weighted",
        "tail-tail",
        "focus-tail-hard",
        MATCH_LEARNER_FOCUS,
        MATCH_LEARNER_WEIGHTED,
        MATCH_LEARNER_HARD,
    ):
        outcomes = []
        for report in reports:
            if report["category"] != category or report["policy_matchup"] not in (POLICY_IL, POLICY_HISTORY):
                continue
            owner = int(tuple(report["current_owners"])[0])
            outcomes.append(float(tuple(report["result"])[owner]))
        if outcomes:
            label = category.replace("-", "_")
            result[f"performance/by_deck_mix/{label}_score"] = sum(
                1.0 if value > 0.0 else 0.5 if value == 0.0 else 0.0 for value in outcomes
            ) / len(outcomes)
    return result


def _wandb_update_metrics(metric: Mapping[str, object]) -> dict[str, int | float]:
    truncated = int(metric["collection_truncated_count"])
    return {
        "update": int(metric["update"]),
        "safety/all_training_frames_used": float(metric["learner_owner_frame_fraction"]),
        "safety/rejected_actions": int(metric["native_rejected_actions"]),
        **(
            {
                "safety/episode_failure_rate": float(metric["episode_failure_rate"]),  # type: ignore[arg-type]
                "safety/wave_episode_failures": int(metric["wave_episode_failures"]),  # type: ignore[call-overload]
            }
            if "episode_failure_rate" in metric
            else {}
        ),
        "safety/boundary_bootstrap_coverage": (
            float(metric["collection_boundary_bootstrap_count"]) / truncated if truncated else 1.0
        ),
        "league/history_snapshots": int(metric["league_history_snapshots"]),
        **{
            f"anchor/{key.removeprefix('anchor_')}": float(metric[key])  # type: ignore[arg-type]
            for key in (
                "anchor_kl",
                "anchor_kl_gate",
                "anchor_kl_candidate",
                "anchor_kl_target",
                "anchor_kl_continue",
                "anchor_kl_coefficient_next",
                "anchor_continue_kl_coefficient_next",
            )
            if key in metric
        },
        **({"anchor/second_rate": float(metric["second_rate"])} if "second_rate" in metric else {}),  # type: ignore[arg-type]
    }


def _release_rollout_cuda_graphs(models: Sequence[UniversalCardPolicyV4], *, device: torch.device) -> None:
    """Release inference-only graph pools before same-GPU PPO backprop."""

    torch.cuda.synchronize(device)
    for model in dict.fromkeys(models):
        model.__dict__.pop("_ppo_rollout_cuda_graph_cache_v4", None)
        model.__dict__.pop("_ppo_rollout_cuda_graph_cache_frozen_v4", None)
        model.__dict__.pop("_ppo_rollout_cuda_graph_pool_v4", None)
    gc.collect()
    torch.cuda.empty_cache()


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or len(set(result)) != len(result):
        raise ValueError("engine readiness has an invalid control CPU list")
    return result


def _ready_worker_cpus(ready: Mapping[str, object]) -> tuple[int, ...]:
    engine_count = int(ready.get("engine_count", -1))
    raw_workers = ready.get("worker_cpus")
    if not isinstance(raw_workers, list):
        raise ValueError("PPO topology lacks worker CPUs")
    workers = tuple(int(value) for value in raw_workers)
    if len(workers) != engine_count:
        raise ValueError("trainer worker CPU topology does not match engine count")
    allowed = os.sched_getaffinity(0)
    if any(cpu not in allowed for cpu in workers):
        raise ValueError("trainer worker CPU is outside the Slurm allocation")
    return workers


def _ready_engine_hosts(ready: Mapping[str, object], *, engine_count: int) -> tuple[str, ...]:
    raw = ready.get("engine_hosts")
    if not isinstance(raw, list) or len(raw) != engine_count:
        raise ValueError("engine readiness host topology does not match engine count")
    hosts = tuple(str(value).strip() for value in raw)
    if any(not host for host in hosts):
        raise ValueError("engine readiness contains an empty host")
    return hosts


def _ready_engine_ports(ready: Mapping[str, object], *, engine_count: int) -> tuple[int, ...]:
    raw = ready.get("engine_ports")
    if not isinstance(raw, list) or len(raw) != engine_count:
        raise ValueError("engine readiness port topology does not match engine count")
    ports = tuple(int(value) for value in raw)
    if any(not 1 <= port <= 65_535 for port in ports):
        raise ValueError("engine readiness contains an invalid TCP port")
    if len(set(zip(_ready_engine_hosts(ready, engine_count=engine_count), ports, strict=True))) != engine_count:
        raise ValueError("engine readiness contains duplicate endpoints")
    return ports


def _ready_engine_worker_ports(ready: Mapping[str, object], *, engine_count: int) -> tuple[int, ...]:
    raw = ready.get("engine_worker_ports")
    if not isinstance(raw, list) or len(raw) != engine_count:
        raise ValueError("engine readiness worker ports do not match engine count")
    ports = tuple(int(value) for value in raw)
    if any(not 1 <= port <= 65_535 for port in ports):
        raise ValueError("engine readiness contains an invalid worker port")
    return ports


def _ready_engine_worker_hosts(ready: Mapping[str, object], *, engine_count: int) -> tuple[str, ...]:
    raw = ready.get("engine_worker_hosts")
    if not isinstance(raw, list) or len(raw) != engine_count:
        raise ValueError("engine readiness worker hosts do not match engine count")
    hosts = tuple(str(value).strip() for value in raw)
    if len(hosts) != engine_count or any(not host for host in hosts):
        raise ValueError("engine readiness contains an invalid worker host")
    return hosts


def _ready_rank_engine_indices(
    ready: Mapping[str, object], *, engine_count: int, world_size: int
) -> tuple[tuple[int, ...], ...]:
    raw = ready.get("rank_engine_indices")
    if not isinstance(raw, list) or len(raw) != world_size:
        raise ValueError("rank engine assignments do not match world size")
    assignments: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, list) or not row or any(not isinstance(index, int) for index in row):
            raise ValueError("rank engine assignments must be nonempty integer lists")
        assignments.append(tuple(row))
    flattened = [index for row in assignments for index in row]
    if sorted(flattened) != list(range(engine_count)):
        raise ValueError("rank engine assignments must cover every engine exactly once")
    return tuple(assignments)


def _bind_rank_cpus(
    ready: Mapping[str, object], local_rank: int, world: int, engine_indices: Sequence[int], worker_cpus: Sequence[int]
) -> tuple[int, ...]:
    control = _parse_cpu_list(str(ready["control_cpus"]))
    if len(control) < world:
        raise ValueError("not enough control CPUs for actor/learner ranks")
    groups = tuple(tuple(control[index::world]) for index in range(world))
    assigned_control = groups[local_rank]
    assigned_engines = tuple(worker_cpus[index] for index in engine_indices)
    # More engines than worker CPUs intentionally time-share cores.  Return
    # the unique affinity set so the caller does not size ATen's thread pool
    # from repeated per-engine CPU entries (192 engines previously created 25
    # threads per rank for only eight usable CPUs).
    assigned = tuple(dict.fromkeys((*assigned_control, *assigned_engines)))
    if not assigned_control or not assigned_engines:
        raise ValueError("distributed rank received no control CPU")
    os.sched_setaffinity(0, set(assigned))
    return assigned


def _balanced_assignments(
    matchmaker: DeckMatchmakerV4,
    league: PolicyLeagueV4,
    rng: random.Random,
    *,
    match_capacities: Sequence[int],
    rich_capacities: Sequence[int],
    special_towers_require_rich: bool,
    uniform_history_sampling: bool = False,
    fixed_opponent_il_max_average_elixir: float | None = None,
    deck_average_elixir_by_id: Mapping[str, float] | None = None,
    learner_deck: DeckIdentityV4 | None = None,
    learner_opponent_sampler: FixedLearnerOpponentDeckSamplerV4 | None = None,
    fixed_league: PolicyLeagueV4 | None = None,
    fixed_opponent_percent: int = 0,
    history_percent: int = 0,
) -> tuple[tuple[ClusterMatchAssignmentV4, ...], ...]:
    """Schedule one wave.

    With ``fixed_league`` (split mode) the matchmaker's history share is
    ``fixed_opponent_percent + history_percent``: each history match goes to a
    fixed opponent (uniform cycle) with probability fixed / (fixed + history)
    and otherwise to the PFSP snapshot pool in ``league``.  Before the first
    snapshot exists the snapshot share falls back to the frozen IL anchor,
    preserving the configured self-play share. The matchmaker uses the same
    IL fallback when no historical or fixed opponents exist.
    """

    split_mode = fixed_league is not None
    if split_mode:
        if uniform_history_sampling:
            raise ValueError("split fixed/snapshot mode samples snapshots by PFSP")
        if min(fixed_opponent_percent, history_percent) < 0 or fixed_opponent_percent + history_percent <= 0:
            raise ValueError("split fixed/snapshot shares must be nonnegative with a positive sum")
        if fixed_opponent_percent and not fixed_league.has_history:
            raise ValueError("fixed opponent share requires fixed opponent checkpoints")
    if not match_capacities or len(match_capacities) != len(rich_capacities):
        raise ValueError("distributed resident capacities are invalid")
    if any(not 0 <= rich <= matches for matches, rich in zip(match_capacities, rich_capacities, strict=True)):
        raise ValueError("rich resident slot count is outside one rank shard")
    if fixed_opponent_il_max_average_elixir is not None:
        if not math.isfinite(fixed_opponent_il_max_average_elixir) or fixed_opponent_il_max_average_elixir <= 0.0:
            raise ValueError("fixed-opponent deck threshold must be positive")
        if deck_average_elixir_by_id is None:
            raise ValueError("fixed-opponent deck threshold requires deck averages")
    world_size = len(match_capacities)
    total = sum(match_capacities)
    history_candidates = (
        tuple(league.entries)
        if uniform_history_sampling
        else league.sample_history_candidates(rng, limit=HISTORY_CANDIDATES_PER_BATCH)
    )
    uniform_history_cycle: list[LeagueEntryV4] = []
    fixed_cycle: list[LeagueEntryV4] = []
    candidates: list[ClusterMatchAssignmentV4] = []
    has_history = league.has_history or (split_mode and fixed_league.has_history)  # type: ignore[union-attr]
    matchups = tuple(
        matchmaker.next_match(has_history=has_history, battle_level=battle_level)
        for battle_level in matchmaker.battle_levels(total)
    )
    learner_opponents = (
        None
        if learner_opponent_sampler is None
        else learner_opponent_sampler.sample_batch(tuple(matchup.policy_matchup for matchup in matchups))
    )
    for match_index, matchup in enumerate(matchups):
        if learner_deck is not None:
            matchup = replace(
                matchup,
                deck0=(learner_deck if 0 in matchup.current_owners else matchup.deck0),
                deck1=(learner_deck if 1 in matchup.current_owners else matchup.deck1),
            )
        if learner_opponents is not None:
            if learner_deck is None or len(matchup.current_owners) != 1:
                raise RuntimeError("fixed-learner opponent sampling requires one current owner")
            category, opponent_deck = learner_opponents[match_index]
            current_owner = matchup.current_owners[0]
            matchup = replace(
                matchup,
                category=category,
                deck0=(opponent_deck if current_owner == 1 else matchup.deck0),
                deck1=(opponent_deck if current_owner == 0 else matchup.deck1),
            )
        opponent = None
        if matchup.policy_matchup == POLICY_HISTORY and split_mode:
            assert fixed_league is not None
            use_fixed = bool(fixed_opponent_percent) and (
                not history_percent or rng.randrange(fixed_opponent_percent + history_percent) < fixed_opponent_percent
            )
            use_anchor_il = False
            if use_fixed and fixed_opponent_il_max_average_elixir is not None:
                opponent_owner = 1 - matchup.current_owners[0]
                opponent_deck = (matchup.deck0, matchup.deck1)[opponent_owner]
                assert deck_average_elixir_by_id is not None
                try:
                    opponent_average_elixir = float(deck_average_elixir_by_id[opponent_deck.deck_id])
                except KeyError as error:
                    raise ValueError("fixed-opponent deck has no average-elixir value") from error
                use_anchor_il = opponent_average_elixir <= fixed_opponent_il_max_average_elixir + 1e-12
            if use_fixed and not use_anchor_il:
                if not fixed_cycle:
                    fixed_cycle = list(fixed_league.entries)
                    rng.shuffle(fixed_cycle)
                opponent = fixed_cycle.pop()
            elif not use_fixed and history_candidates:
                opponent = league.sample_history_from(rng, history_candidates)
            else:
                # An empty snapshot pool adds IL games, never self-play games.
                matchup = replace(matchup, policy_matchup=POLICY_IL)
        elif matchup.policy_matchup == POLICY_HISTORY:
            use_anchor_il = False
            if fixed_opponent_il_max_average_elixir is not None:
                if len(matchup.current_owners) != 1:
                    raise RuntimeError("fixed-opponent matchup needs one current owner")
                opponent_owner = 1 - matchup.current_owners[0]
                opponent_deck = (matchup.deck0, matchup.deck1)[opponent_owner]
                assert deck_average_elixir_by_id is not None
                try:
                    opponent_average_elixir = float(deck_average_elixir_by_id[opponent_deck.deck_id])
                except KeyError as error:
                    raise ValueError("fixed-opponent deck has no average-elixir value") from error
                use_anchor_il = opponent_average_elixir <= fixed_opponent_il_max_average_elixir + 1e-12
            if use_anchor_il:
                matchup = replace(matchup, policy_matchup=POLICY_IL)
            elif uniform_history_sampling:
                if not uniform_history_cycle:
                    uniform_history_cycle = list(history_candidates)
                    rng.shuffle(uniform_history_cycle)
                opponent = uniform_history_cycle.pop()
            else:
                opponent = league.sample_history_from(rng, history_candidates)
        candidates.append(ClusterMatchAssignmentV4(matchup, opponent))

    shards: list[list[ClusterMatchAssignmentV4]] = [[] for _ in range(world_size)]
    current_current = [0 for _ in range(world_size)]
    rich_matches = [0 for _ in range(world_size)]

    def needs_rich(assignment: ClusterMatchAssignmentV4) -> bool:
        if not special_towers_require_rich:
            return False
        matchup = assignment.matchup
        return any(deck.tower_troop_id in SPECIAL_TOWER_RUNTIME_IDS for deck in (matchup.deck0, matchup.deck1))

    rich_two = [row for row in candidates if needs_rich(row) and len(row.matchup.current_owners) == 2]
    rich_one = [row for row in candidates if needs_rich(row) and len(row.matchup.current_owners) == 1]
    ordinary_two = [row for row in candidates if not needs_rich(row) and len(row.matchup.current_owners) == 2]
    ordinary_one = [row for row in candidates if not needs_rich(row) and len(row.matchup.current_owners) == 1]
    two_count = len(rich_two) + len(ordinary_two)
    target_two = [0 for _ in range(world_size)]
    for _ in range(two_count):
        eligible = [rank for rank in range(world_size) if target_two[rank] < match_capacities[rank]]
        if not eligible:
            raise RuntimeError("current-current schedule exceeds resident capacity")
        rank = min(eligible, key=lambda item: (match_capacities[item] + target_two[item], target_two[item], item))
        target_two[rank] += 1

    def place_two(rows: Sequence[ClusterMatchAssignmentV4], *, rich: bool) -> None:
        for assignment in rows:
            eligible = [
                rank
                for rank in range(world_size)
                if current_current[rank] < target_two[rank]
                and len(shards[rank]) < match_capacities[rank]
                and (not rich or rich_matches[rank] < rich_capacities[rank])
            ]
            if not eligible:
                raise RuntimeError("cannot balance current-current resident rows")
            rank = min(eligible, key=lambda item: (current_current[item], rich_matches[item], len(shards[item]), item))
            shards[rank].append(assignment)
            current_current[rank] += 1
            rich_matches[rank] += int(rich)

    place_two(rich_two, rich=True)
    place_two(ordinary_two, rich=False)
    if current_current != target_two:
        raise RuntimeError("current-current target allocation is incomplete")

    for assignment in rich_one:
        eligible = [
            rank
            for rank in range(world_size)
            if len(shards[rank]) < match_capacities[rank] and rich_matches[rank] < rich_capacities[rank]
        ]
        if not eligible:
            raise RuntimeError("special Tower Troop schedule exceeds exact-rich resident capacity")
        rank = min(eligible, key=lambda item: (rich_matches[item], len(shards[item]), item))
        shards[rank].append(assignment)
        rich_matches[rank] += 1

    for assignment in ordinary_one:
        eligible = [rank for rank in range(world_size) if len(shards[rank]) < match_capacities[rank]]
        if not eligible:
            raise RuntimeError("ordinary resident schedule overfilled")
        rank = min(eligible, key=lambda item: (len(shards[item]), item))
        shards[rank].append(assignment)
    if any(len(shard) != capacity for shard, capacity in zip(shards, match_capacities, strict=True)):
        raise RuntimeError("distributed schedule did not fill every resident lane")
    # The first resident engines on each rank use full compressed JSON. Place
    # every special Tower Troop match in that prefix, then fill its unused
    # slots with ordinary matches and route the remainder to compact engines.
    result = []
    for rank, shard in enumerate(shards):
        special = sorted((row for row in shard if needs_rich(row)), key=lambda row: row.matchup.sequence)
        ordinary = sorted((row for row in shard if not needs_rich(row)), key=lambda row: row.matchup.sequence)
        padding = rich_capacities[rank] - len(special)
        result.append(tuple(special + ordinary[:padding] + ordinary[padding:]))
    return tuple(result)


def _broadcast_object(value: object | None, *, source: int = 0) -> object:
    payload = [value]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def _gather_objects(value: object, *, rank: int, world_size: int) -> list[object] | None:
    output = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(value, output, dst=0)
    return output


def _load_history_models(
    assignments: Sequence[ClusterMatchAssignmentV4],
    cache: dict[str, UniversalCardPolicyV4],
    *,
    device: torch.device,
) -> dict[str, UniversalCardPolicyV4]:
    required = {
        assignment.opponent.checkpoint_id: assignment.opponent
        for assignment in assignments
        if assignment.opponent is not None
    }
    for checkpoint_id, entry in required.items():
        if checkpoint_id in cache:
            continue
        loaded = _load_interface_opponent_model(entry.checkpoint_path, device=device)
        if loaded.checkpoint_id != checkpoint_id:
            raise ValueError("historical checkpoint file identity changed")
        cache[checkpoint_id] = loaded.model
    # Historical models not selected in this wave need not consume GPU memory.
    for checkpoint_id in tuple(cache):
        if checkpoint_id not in required:
            del cache[checkpoint_id]
    return {checkpoint_id: cache[checkpoint_id] for checkpoint_id in required}


DDP_BACKENDS = ("nccl", "gloo")


def _rank_cuda_device_index(
    local_rank: int, *, backend: str, device_count: int,
    rank_cuda_device: Sequence[int] | None = None,
) -> int:
    """GPU of one rank: NCCL needs one GPU per rank; gloo may stack ranks on a GPU (round robin).

    NCCL rejects two ranks on one GPU ("Duplicate GPU"); the 16.8M-parameter
    policy's gradient all-reduce is cheap enough for gloo, which then lets
    several single-threaded collector loops share one GPU.  Each rank is its
    own process, so its CUDA context, CUDA Graph pools and static buffers are
    separate by construction.
    """

    if backend not in DDP_BACKENDS:
        raise ValueError(f"DDP backend must be one of {DDP_BACKENDS}")
    if device_count <= 0 or local_rank < 0:
        raise ValueError("rank device mapping needs visible GPUs and a nonnegative local rank")
    if rank_cuda_device is not None:
        mapping = tuple(int(value) for value in rank_cuda_device)
        if local_rank >= len(mapping) or any(value < 0 or value >= device_count for value in mapping):
            raise ValueError("rank_cuda_device contains an unavailable GPU or rank")
        if backend == "nccl" and len(set(mapping)) != len(mapping):
            raise ValueError("NCCL needs one visible GPU per rank; use --ddp-backend gloo to share GPUs")
        return mapping[local_rank]
    if backend == "nccl":
        if local_rank >= device_count:
            raise ValueError("NCCL needs one visible GPU per rank; use --ddp-backend gloo to share GPUs")
        return local_rank
    return local_rank % device_count


def _validate_topology(
    ready: Mapping[str, object], *, world_size: int, backend: str = "nccl"
) -> tuple[int, int]:
    """Validate the production topology and return ``(engine_count, slots_per_engine)``.

    The slot count comes from the topology file (``resident_slots_per_engine``):
    production runs one slot per native standalone engine with
    any engine count over any number of nodes; the legacy 8-slot emulator
    topology (same schema) stays accepted.
    """

    if ready.get("schema") != "v4-ppo-multinode-topology.v1":
        raise ValueError("unsupported PPO topology schema")
    expected_job = str(ready.get("trainer_slurm_job_id", ""))
    expected_host = str(ready.get("trainer_host", "")).split(".", 1)[0]
    engine_count = int(ready.get("engine_count", -1))
    lanes = int(ready.get("resident_slots_per_engine", -1))
    visible_devices = torch.cuda.device_count()
    declared_devices = int(ready.get("cuda_devices", visible_devices))
    rank_devices = ready.get("rank_cuda_device")
    checks = {
        "slurm_job_id": bool(expected_job) and os.environ.get("SLURM_JOB_ID") == expected_job,
        "host": bool(expected_host) and socket.gethostname().split(".", 1)[0] == expected_host,
        "world_size": world_size == int(ready.get("gpus", -1)),
        # NCCL: one GPU per rank.  gloo: ranks stacked evenly on the visible GPUs.
        "visible_gpu_count": (
            visible_devices == declared_devices and (visible_devices == world_size
            if backend == "nccl"
            else visible_devices > 0 and world_size % visible_devices == 0)
        ),
        "rank_cuda_device": rank_devices is None or (
            isinstance(rank_devices, (list, tuple))
            and len(rank_devices) == world_size
            and declared_devices > 0
            and all(isinstance(value, int) and 0 <= value < declared_devices for value in rank_devices)
            and all(rank_devices.count(index) == world_size // declared_devices for index in range(declared_devices))
        ),
        "engine_count": engine_count > 0,
        "lanes": lanes >= 1,
        "decision_ticks": int(ready.get("decision_ticks", -1)) == 5,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"production PPO topology failed checks: {failed}")
    return engine_count, lanes


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-pool", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="fresh learner initialization checkpoint")
    parser.add_argument(
        "--il-anchor-checkpoint",
        type=Path,
        help="frozen IL opponent checkpoint (argmax, the current-il share); defaults to --checkpoint",
    )
    parser.add_argument(
        "--delay-policy",
        choices=DELAY_POLICIES,
        default=DELAY_POLICY_SAMPLE,
        help=(
            "delay-bin decoding for the learner, rollout and KL reference; frozen battle opponents "
            "always use the interface default earliest. Set before CUDA Graph capture; with earliest, "
            "delay never enters log-prob, entropy, ratio or KL"
        ),
    )
    parser.add_argument(
        "--gate-logit-offset",
        default="0",
        help=(
            "ACT-logit correction added once to the learner and KL reference only; "
            "'auto' = -ln(extra.gate_act_weight). Frozen battle opponents use the "
            "interface's argmax loader without an added correction. PPO snapshots already carry it"
        ),
    )
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        help="frozen IL reference for the KL anchor (strong-v1); enables loss += beta*KL(pi||pi_ref)",
    )
    parser.add_argument("--anchor-kl-coef", type=float, default=0.05, help="initial adaptive beta")
    parser.add_argument(
        "--anchor-target-kl",
        type=float,
        default=0.02,
        help="adaptive-controller target for gate+candidate+target KL in nats per valid frame",
    )
    parser.add_argument(
        "--anchor-continue-kl-coef",
        type=float,
        help="initial continue-head coefficient beta_c (default 5 * --anchor-kl-coef)",
    )
    parser.add_argument(
        "--anchor-continue-kl-mode",
        choices=ANCHOR_CONTINUE_MODES,
        default="tied",
        help="tied: beta_c keeps its initial ratio to the adaptive beta; fixed: beta_c is constant",
    )
    parser.add_argument("--anchor-kl-horizon", type=float, default=4.0, help="controller horizon in updates")
    parser.add_argument("--anchor-kl-coef-min", type=float, default=0.005)
    parser.add_argument("--anchor-kl-coef-max", type=float, default=1.0)
    parser.add_argument("--anchor-kl-fixed", action="store_true", help="disable beta adaptation")
    parser.add_argument(
        "--anchor-kl-no-decrease-updates",
        type=int,
        default=20,
        help="first N controller updates after the critic warmup may only raise beta",
    )
    parser.add_argument(
        "--critic-warmup-iters",
        type=int,
        default=DEFAULT_CRITIC_WARMUP_ITERS,
        help="first N updates train only value-only modules; the actor (and expert BC) is frozen",
    )
    parser.add_argument(
        "--environment-warmup-ticks",
        type=int,
        default=DEFAULT_ENVIRONMENT_WARMUP_TICKS,
        help="native warmup ticks before the first training decision (default 130)",
    )
    parser.add_argument(
        "--history-percent",
        type=int,
        default=0,
        help="percent of games against the PFSP snapshot pool (separate from --fixed-opponent-percent)",
    )
    parser.add_argument(
        "--opening-force-after-seconds",
        type=int,
        default=DEFAULT_OPENING_FORCE_AFTER_SECONDS,
        help=(
            "force the first legal ACT independently for each still-idle owner after this battle time; "
            "applies to both learner and opponent in every matchup, including both self-play owners"
        ),
    )
    parser.add_argument(
        "--il-opening-force-after-seconds",
        type=int,
        help=(
            "legacy compatibility option; must equal --opening-force-after-seconds because "
            "all training owners now share the same opening deadline"
        ),
    )
    parser.add_argument(
        "--self-play-percent", type=int, default=0, help="percent of games using current versus current"
    )
    parser.add_argument(
        "--fixed-opponent-percent",
        type=int,
        default=0,
        help="total percent of games assigned to --fixed-opponent-checkpoint",
    )
    parser.add_argument(
        "--fixed-opponent-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="repeatable uniformly sampled frozen opponent checkpoint",
    )
    parser.add_argument(
        "--fixed-opponent-il-max-average-elixir",
        type=float,
        help=(
            "within the fixed-opponent share, use the frozen IL anchor when "
            "the opponent deck average is at most this value; otherwise use "
            "--fixed-opponent-checkpoint"
        ),
    )
    parser.add_argument(
        "--learner-deck-id", help="optional exact deck-pool identity used for every current-policy owner"
    )
    parser.add_argument(
        "--fixed-learner-opponent-mix",
        action="store_true",
        help=(
            "with a fixed learner deck and no self-play, sample configurable "
            "focus, sqrt(uses)-weighted full-pool, and online-hard opponents"
        ),
    )
    parser.add_argument("--learner-opponent-focus-percent", type=int, default=30)
    parser.add_argument("--learner-opponent-weighted-percent", type=int, default=50)
    parser.add_argument("--learner-opponent-hard-percent", type=int, default=20)
    parser.add_argument("--hard-deck-decay-per-batch", type=float, default=0.8)
    parser.add_argument("--hard-deck-win-rate-threshold", type=float, default=0.55)
    parser.add_argument("--hard-deck-top-k", type=int, default=128)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine-ready", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument(
        "--retain-rollout-cuda-graphs", action="store_true",
        help="keep captured rollout graphs across in-place PPO updates (requires GPU memory headroom)",
    )
    parser.add_argument(
        "--rollout-device-cache-gib", type=float, default=0.0,
        help="per-rank GPU byte budget for a dtype-packed whole-rollout input cache (0 disables)",
    )
    parser.add_argument(
        "--rollout-indexed-state-graphs", action="store_true",
        help="include recurrent state gather/scatter in rollout CUDA Graphs",
    )
    parser.add_argument("--rollout-adaptive-batching", action="store_true",
                        help="coalesce in-flight frames up to one GPU batch latency, bounded at 25 ms")
    parser.add_argument("--rollout-cache-static-memory", action="store_true",
                        help="cache observation-independent effect/profile encodings for each fixed-policy wave")
    parser.add_argument("--rollout-segment-seconds", type=int, default=40)
    parser.add_argument("--snapshot-every", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9997)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--shaping-beta", type=float, default=0.05)
    parser.add_argument("--terminal-reward-scale", type=float, default=1.0,
                        help="absolute terminal win/loss reward; draw remains zero")
    parser.add_argument(
        "--personal-elixir-overflow-penalty",
        dest="personal_elixir_overflow_penalty",
        type=float,
        default=0.0,
        help="linear cost for all continuous own-elixir overflow",
    )
    parser.add_argument(
        "--personal-elixir-overflow-grace",
        dest="personal_elixir_overflow_grace",
        type=float,
        default=4.0,
        help="own wasted elixir exempt from the personal linear penalty",
    )
    parser.add_argument("--unilateral-elixir-overflow-penalty", type=float, default=0.0)
    parser.add_argument("--elixir-overflow-step-penalty-cap", type=float, default=0.1)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--hog-deploy-reward", type=float, default=0.0)
    parser.add_argument(
        "--fireball-king-activation-penalty",
        type=float,
        default=0.0,
        help="legacy Fireball-only one-sided penalty; keep 0 with --king-activation-penalty",
    )
    parser.add_argument(
        "--king-activation-penalty",
        type=float,
        default=DEFAULT_KING_ACTIVATION_PENALTY,
        help="zero-sum first enemy-king activation penalty c (activator -c, victim +c); stats recorded at c=0",
    )
    parser.add_argument("--king-activation-scope", choices=KING_ACTIVATION_SCOPES, default=DEFAULT_KING_ACTIVATION_SCOPE)
    parser.add_argument(
        "--king-shaping-mode",
        choices=KING_SHAPING_MODES,
        default="princess_gated",
        help="princess_gated: king damage is out of the tower potential while all princess towers stand",
    )
    parser.add_argument("--king-potential-weight", type=float, default=DEFAULT_KING_POTENTIAL_WEIGHT)
    parser.add_argument("--hog-deploy-reward-episode-cap", type=float, default=0.1)
    parser.add_argument(
        "--hog-deploy-opening-reward",
        type=float,
        help="replace the general Hog reward at executed battle time 10 <= t < 30 seconds",
    )
    parser.add_argument("--first-hog-timing-reward-max", type=float, default=0.0)
    parser.add_argument("--first-hog-timing-start-seconds", type=float, default=10.0)
    parser.add_argument("--first-hog-timing-deadline-seconds", type=float, default=30.0)
    parser.add_argument("--first-hog-missed-deadline-penalty", type=float, default=0.0)
    parser.add_argument("--first-hog-deferral-penalty", type=float, default=0.0)
    parser.add_argument("--first-hog-deferral-penalty-episode-cap", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--lane-minibatch-size", type=int, default=256)
    parser.add_argument("--lane-microbatch-size", type=int, default=96)
    parser.add_argument(
        "--activation-checkpointing",
        choices=("none", "default", "all"),
        default="none",
        help=(
            "learner-only recompute in the PPO backward (default = attention+spatial, all = +pair-bias); "
            "ignored under no_grad and CUDA Graph capture, so rollout inference is unchanged"
        ),
    )
    parser.add_argument("--sequence-chunk-steps", type=int, default=48)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--max-episode-failure-rate",
        type=float,
        default=DEFAULT_MAX_EPISODE_FAILURE_RATE,
        help=(
            "a failed match (e.g. BattleEnvError) is dropped and logged to episode_failures.jsonl; the run "
            "still fails when the current wave's failure rate exceeds this fraction (0 = fail on any)"
        ),
    )
    parser.add_argument("--expert-bc-manifest", type=Path)
    parser.add_argument("--expert-bc-coefficient", type=float, default=0.2)
    parser.add_argument("--expert-bc-records-per-rank", type=int, default=2)
    parser.add_argument("--expert-bc-window-steps", type=int, default=128)
    parser.add_argument("--expert-bc-hog-gate-weight", type=float, default=0.0)
    parser.add_argument("--expert-bc-hog-gate-scope", choices=("all", "proactive-first"), default="all")
    parser.add_argument("--expert-bc-hog-gate-gradient-scope", choices=("full", "gate-head-only"), default="full")
    parser.add_argument(
        "--expert-bc-delay-weight",
        type=float,
        help="BC delay-head weight (default 0 under --delay-policy earliest, else the legacy 1)",
    )
    parser.add_argument(
        "--expert-bc-continue-second-weight",
        type=float,
        help="BC SECOND class weight (default 32 under --delay-policy earliest = strong-v1 IL, else 1)",
    )
    parser.add_argument(
        "--gate-temperature-start",
        type=float,
        help=f"default {DEFAULT_GATE_TEMPERATURE_START}; default 1 with --anchor-checkpoint",
    )
    parser.add_argument(
        "--gate-temperature-end",
        type=float,
        help=f"default {DEFAULT_GATE_TEMPERATURE_END}; default 1 with --anchor-checkpoint",
    )
    parser.add_argument("--gate-temperature-anneal-updates", type=int, default=DEFAULT_GATE_TEMPERATURE_ANNEAL_UPDATES)
    parser.add_argument(
        "--continue-temperature",
        type=float,
        help=f"default {DEFAULT_CONTINUE_TEMPERATURE}; default 1 with --anchor-checkpoint",
    )
    parser.add_argument(
        "--ddp-backend",
        choices=DDP_BACKENDS,
        default="nccl",
        help="gloo allows several ranks per GPU (world size a multiple of the visible GPU count)",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate topology/model/config startup, write preflight.json and exit before collection")
    return parser


def _expert_bc_loss_weights(args: argparse.Namespace) -> ILLossWeightsV4:
    """BC weights consistent with the IL objective (strong-v1: delay 0, SECOND x32)."""

    earliest = args.delay_policy != DELAY_POLICY_SAMPLE
    delay = (0.0 if earliest else 1.0) if args.expert_bc_delay_weight is None else float(args.expert_bc_delay_weight)
    second = (
        (32.0 if earliest else 1.0)
        if args.expert_bc_continue_second_weight is None
        else float(args.expert_bc_continue_second_weight)
    )
    return ILLossWeightsV4(value=0.0, delay=delay, continue_second_weight=second)


def _resolve_opponent_mix(args: argparse.Namespace) -> dict[str, int]:
    """current-il (frozen IL argmax) / self-play / fixed checkpoints / PFSP snapshots, in percent."""

    self_play = int(args.self_play_percent)
    fixed = int(args.fixed_opponent_percent)
    history = int(args.history_percent)
    for name, value in (("self-play", self_play), ("fixed opponent", fixed), ("history", history)):
        if not 0 <= value <= 100:
            raise ValueError(f"{name} percent must be in [0, 100]")
    if fixed and not args.fixed_opponent_checkpoint:
        raise ValueError("fixed opponent percent requires at least one checkpoint")
    if args.fixed_opponent_checkpoint and not fixed:
        raise ValueError("fixed opponent checkpoints require a positive percent")
    il = 100 - self_play - fixed - history
    if il < 0:
        raise ValueError("self-play, fixed opponent and history percentages exceed 100")
    return {"current_il": il, "self_play": self_play, "fixed_opponent": fixed, "history": history}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    anchor_kl_config = _anchor_kl_config(args)
    (gate_temperature_start, gate_temperature_end, continue_temperature) = _resolve_temperatures(
        args, anchor_enabled=anchor_kl_config is not None
    )
    args.gate_temperature_start = gate_temperature_start
    args.gate_temperature_end = gate_temperature_end
    args.continue_temperature = continue_temperature
    if not math.isfinite(args.max_episode_failure_rate) or not 0.0 <= args.max_episode_failure_rate <= 1.0:
        raise ValueError("--max-episode-failure-rate must be in [0, 1]")
    if args.critic_warmup_iters < 0:
        raise ValueError("critic warmup iterations cannot be negative")
    if args.environment_warmup_ticks <= 0:
        raise ValueError("environment warmup ticks must be positive")
    validate_king_reward_parameters_v4(
        king_activation_penalty=args.king_activation_penalty,
        king_activation_scope=args.king_activation_scope,
        king_shaping_mode=args.king_shaping_mode,
        king_potential_weight=args.king_potential_weight,
    )
    expert_bc_loss_weights = _expert_bc_loss_weights(args)
    if (
        not math.isfinite(args.expert_bc_coefficient)
        or args.expert_bc_coefficient <= 0
        or min(args.expert_bc_records_per_rank, args.expert_bc_window_steps) <= 0
    ):
        raise ValueError("expert BC coefficient and dimensions must be positive")
    if not math.isfinite(args.expert_bc_hog_gate_weight) or args.expert_bc_hog_gate_weight < 0:
        raise ValueError("expert Hog gate weight must be finite and nonnegative")
    if args.expert_bc_hog_gate_weight and args.expert_bc_manifest is None:
        raise ValueError("expert Hog gate supervision requires an expert BC manifest")
    opening_force_options = _training_opening_force_options(args)
    opening_force_config = {
        "enabled": True,
        "scope": "both-owners-all-matchups",
        "after_seconds": args.opening_force_after_seconds,
        "condition": "each owner has not acted, independently of the other owner's activity",
        "once_per_owner_per_game": True,
        "force_learner": True,
        "force_fixed_opponents": True,
        "downstream_decode": "frozen opponents use interface argmax; learner keeps sampling",
    }
    il_anchor_checkpoint = args.il_anchor_checkpoint or args.checkpoint
    opponent_mix = _resolve_opponent_mix(args)
    self_play_percent = opponent_mix["self_play"]
    fixed_opponent_percent = opponent_mix["fixed_opponent"]
    history_percent = opponent_mix["history"]
    fixed_opponent_paths = tuple(args.fixed_opponent_checkpoint)
    if str(args.gate_logit_offset).strip().lower() != "auto":
        try:
            gate_offset_value = float(args.gate_logit_offset)
        except ValueError as error:
            raise ValueError("--gate-logit-offset must be 'auto' or a number") from error
        if not math.isfinite(gate_offset_value):
            raise ValueError("--gate-logit-offset must be finite")
    fixed_opponent_il_max_average_elixir = (
        None if args.fixed_opponent_il_max_average_elixir is None else float(args.fixed_opponent_il_max_average_elixir)
    )
    if fixed_opponent_il_max_average_elixir is not None:
        if not fixed_opponent_percent:
            raise ValueError("fixed-opponent deck threshold requires a positive fixed share")
        if not math.isfinite(fixed_opponent_il_max_average_elixir) or fixed_opponent_il_max_average_elixir <= 0.0:
            raise ValueError("fixed-opponent deck threshold must be positive")
    learner_deck_id = None if args.learner_deck_id is None else str(args.learner_deck_id).strip()
    if args.learner_deck_id is not None and not learner_deck_id:
        raise ValueError("learner deck ID cannot be empty")
    if args.fixed_learner_opponent_mix:
        if learner_deck_id is None:
            raise ValueError("fixed-learner opponent mix requires --learner-deck-id")
        if self_play_percent != 0:
            raise ValueError("fixed-learner opponent mix does not support self-play")
        learner_opponent_mix = (
            args.learner_opponent_focus_percent,
            args.learner_opponent_weighted_percent,
            args.learner_opponent_hard_percent,
        )
        if any(value < 0 for value in learner_opponent_mix):
            raise ValueError("fixed-learner opponent percentages cannot be negative")
        if sum(learner_opponent_mix) != 100:
            raise ValueError("fixed-learner opponent percentages must sum to 100")
        if args.learner_opponent_hard_percent <= 0:
            raise ValueError("fixed-learner hard opponent percentage must be positive")
    if not 0.0 < args.hard_deck_decay_per_batch <= 1.0:
        raise ValueError("hard deck decay per batch must be in (0, 1]")
    if not 0.0 < args.hard_deck_win_rate_threshold < 1.0:
        raise ValueError("hard deck win-rate threshold must be in (0, 1)")
    resolved_fixed_paths = tuple(path.resolve() for path in fixed_opponent_paths)
    if len(set(resolved_fixed_paths)) != len(resolved_fixed_paths):
        raise ValueError("fixed opponent checkpoint paths must be unique")
    il_opponent_percent = opponent_mix["current_il"]
    # The matchmaker's history share covers both fixed checkpoints and PFSP
    # snapshots; _balanced_assignments splits it by the two CLI shares.
    matchmaker_config = MatchmakerConfigV4(
        current_current=self_play_percent,
        current_il=il_opponent_percent,
        current_history=fixed_opponent_percent + history_percent,
        focus_tail_current_focus_percent=75,
    )
    positive = (
        args.timeout,
        args.updates,
        args.rollout_segment_seconds,
        args.snapshot_every,
        args.learning_rate,
        args.gamma,
        args.gae_lambda,
        args.clip_range,
        args.target_kl,
        args.ppo_epochs,
        args.lane_minibatch_size,
        args.lane_microbatch_size,
        args.sequence_chunk_steps,
        args.max_grad_norm,
        args.gate_temperature_start,
        args.gate_temperature_end,
        args.gate_temperature_anneal_updates,
        args.continue_temperature,
        args.hard_deck_top_k,
        args.seed,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("production PPO counts and hyperparameters must be positive")
    if args.lane_microbatch_size > args.lane_minibatch_size:
        raise ValueError("lane microbatch size must fit inside one minibatch")
    validate_reward_parameters_v4(
        {
            **vars(args),
            "mutual_elixir_overflow_penalty": args.personal_elixir_overflow_penalty,
            "mutual_elixir_overflow_grace": args.personal_elixir_overflow_grace,
        }
    )
    if args.first_hog_deferral_penalty > 0.0 and args.first_hog_deferral_penalty_episode_cap <= 0.0:
        raise ValueError("first-Hog deferral penalty requires a positive episode cap")
    _gate_temperature_for_update(
        1,
        start=args.gate_temperature_start,
        end=args.gate_temperature_end,
        anneal_updates=args.gate_temperature_anneal_updates,
    )
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed support is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ready = json.loads(args.engine_ready.read_text(encoding="utf-8"))
    if not isinstance(ready, Mapping) or ready.get("ok") is not True:
        raise RuntimeError("production PPO topology is not ready")
    device_index = _rank_cuda_device_index(
        local_rank, backend=args.ddp_backend, device_count=torch.cuda.device_count(),
        rank_cuda_device=ready.get("rank_cuda_device"),
    )
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group(backend=args.ddp_backend)
    # Establish rank-to-device identity before the first barrier so NCCL never
    # has to guess which GPU a process owns.
    device_probe = torch.zeros((), device=device)
    dist.all_reduce(device_probe)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    actor_device = device
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed + rank * 104729)
    torch.cuda.manual_seed(args.seed + rank * 104729)
    random.seed(args.seed + rank * 104729)

    ready = json.loads(args.engine_ready.read_text(encoding="utf-8"))
    if not isinstance(ready, Mapping) or ready.get("ok") is not True:
        raise ValueError("engine readiness artifact is invalid")
    engine_count, lanes = _validate_topology(ready, world_size=world_size, backend=args.ddp_backend)
    engine_hosts = _ready_engine_hosts(ready, engine_count=engine_count)
    engine_ports = _ready_engine_ports(ready, engine_count=engine_count)
    engine_worker_ports = _ready_engine_worker_ports(ready, engine_count=engine_count)
    engine_worker_hosts = _ready_engine_worker_hosts(ready, engine_count=engine_count)
    engine_cpus = _ready_worker_cpus(ready)
    rank_engine_indices = _ready_rank_engine_indices(ready, engine_count=engine_count, world_size=world_size)
    engine_indices = rank_engine_indices[rank]
    if not engine_indices:
        raise ValueError("distributed rank owns no native engine")
    learner_rank_cpus = _bind_rank_cpus(ready, local_rank, world_size, engine_indices, engine_cpus)
    # The eight control CPUs are striped across ranks. ATen performs the large
    # rollout concatenation/index-copy operations there; per-engine workers
    # independently force their Torch pools back to one thread on engine CPUs.
    torch.set_num_threads(len(learner_rank_cpus))
    collection_control_cpus = _parse_cpu_list(str(ready["control_cpus"]))[local_rank::world_size]
    if not collection_control_cpus:
        raise ValueError("async rank received no collection control CPU")
    os.sched_setaffinity(0, set(learner_rank_cpus))
    matches_per_rank = len(engine_indices) * lanes
    gathered_match_capacities: list[int | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_match_capacities, matches_per_rank)
    match_capacities = tuple(int(value) for value in gathered_match_capacities)
    rich_capacities = (0,) * world_size

    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("refusing to mix PPO into a non-empty output")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "league-checkpoints").mkdir(parents=True, exist_ok=True)
    dist.barrier()
    metrics_path = args.output_dir / "metrics.jsonl"
    phase_path = args.output_dir / "phase.json"
    matches_path = args.output_dir / "matches.jsonl"
    league_path = args.output_dir / "league.json"

    manifest = DeckPoolManifestV4.load(args.deck_pool)
    if len(manifest.decks) != 7_695 or len(manifest.focus_decks) != 20:
        raise ValueError("formal cluster PPO requires the verified 7,695/20 pool")
    deck_average_elixir_by_id = {deck.deck_id: deck.average_elixir for deck in manifest.decks}
    decks_by_id = {deck.deck_id: deck for deck in manifest.decks}
    try:
        learner_deck = None if learner_deck_id is None else decks_by_id[learner_deck_id]
    except KeyError as error:
        raise ValueError("learner deck ID is absent from the deck pool") from error
    environment_config = ppo_environment_config_v4(gamma_per_decision=args.gamma, shaping_beta=args.shaping_beta)
    # Worker-side override only (AsyncResidentEngineSpecV4.environment_warmup_ticks);
    # EnvironmentConfigV1's default stays 130.  The local copy keeps config.json's
    # ruleset identity equal to what the workers build.
    environment_config = replace(environment_config, warmup_ticks=int(args.environment_warmup_ticks))
    max_decision_steps = math.ceil(environment_config.max_battle_ticks / 5) + 4

    def make_optimizer(model: UniversalCardPolicyV4) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=0.0)

    # Learner == actor (same object).  Built from the checkpoint's contract,
    # delay policy and gate correction set before any CUDA Graph capture.
    current_model, current_payload, optimizer = _load_contract_model(
        args.checkpoint if args.resume_checkpoint is None else args.resume_checkpoint,
        device=device,
        delay_policy=args.delay_policy,
        gate_logit_offset=args.gate_logit_offset,
        # Only a strict PPO resume restores optimizer state (never the IL one).
        optimizer_factory=None if args.resume_checkpoint is None else make_optimizer,
    )
    gate_logit_offset_applied = _gate_logit_offset_of(current_model)
    # Actor batches and learner ACT batches have different row groupings.
    # BF16 location logits are unusually sensitive to those GEMM/conv shapes,
    # so keep this small head in FP32 while the rest of PPO remains autocast.
    current_model.set_ppo_location_head_fp32(True)
    current_model.set_activation_checkpointing(
        {"none": False, "default": True, "all": ("pair-bias", "attention", "spatial")}[args.activation_checkpointing]
    )
    if args.resume_checkpoint is None:
        optimizer = make_optimizer(current_model)
        _set_temperatures(
            current_model,
            gate_temperature=_gate_temperature_for_update(
                1,
                start=args.gate_temperature_start,
                end=args.gate_temperature_end,
                anneal_updates=args.gate_temperature_anneal_updates,
            ),
            continue_temperature=args.continue_temperature,
        )
        initial_update_step = 0
    else:
        assert optimizer is not None
        if current_payload.get("optimizer_state_dict") is None:
            raise ValueError("resume checkpoint lacks optimizer state")
        initial_update_step = int(current_payload["update_step"])
        if initial_update_step <= 0 or initial_update_step >= args.updates:
            raise ValueError("resume update must be between zero and --updates")
        expected_gate_temperature = _gate_temperature_for_update(
            initial_update_step + 1,
            start=args.gate_temperature_start,
            end=args.gate_temperature_end,
            anneal_updates=args.gate_temperature_anneal_updates,
        )
        if not math.isclose(current_model.ppo_gate_temperature, expected_gate_temperature, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("resume checkpoint gate temperature changed")
        if not math.isclose(
            current_model.ppo_continue_temperature, args.continue_temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("resume checkpoint continue temperature changed")
    source_checkpoint_id = str(current_payload["checkpoint_id"])
    behavior_state_id = source_checkpoint_id
    actor_model = current_model
    segment_actor_stream = torch.cuda.Stream(device=actor_device)
    actor_state_id = behavior_state_id
    reference_model_options = {
        "device": actor_device,
        "delay_policy": args.delay_policy,
        "gate_logit_offset": args.gate_logit_offset,
        "gate_temperature": args.gate_temperature_start,
        "continue_temperature": args.continue_temperature,
    }
    anchor_policy = _load_interface_opponent_model(il_anchor_checkpoint, device=actor_device)
    anchor_model, anchor_state_id = anchor_policy.model, anchor_policy.checkpoint_id
    # The calibrated KL reference must never share the deployed opponent object.
    reference_model: UniversalCardPolicyV4 | None = None
    reference_state_id: str | None = None
    if args.anchor_checkpoint is not None:
        reference_model, reference_payload = _load_frozen_model(
            args.anchor_checkpoint, sampling=False, **reference_model_options
        )
        reference_state_id = str(reference_payload["checkpoint_id"])
        if asdict(reference_model.config) != asdict(current_model.config) and rank == 0:
            print(
                {
                    "event": "ppo_anchor_architecture_differs",
                    "learner": asdict(current_model.config),
                    "reference": asdict(reference_model.config),
                },
                flush=True,
            )
    anchor_controller = None if anchor_kl_config is None else AdaptiveKLControllerV4(anchor_kl_config)
    expert_bc_sampler = (
        None
        if args.expert_bc_manifest is None
        else HogExpertSamplerV4(
            args.expert_bc_manifest,
            seed=args.seed,
            records_per_rank=args.expert_bc_records_per_rank,
            window_steps=args.expert_bc_window_steps,
        )
    )
    expert_bc_contract = (
        None
        if expert_bc_sampler is None
        else expert_bc_contract_v4(
            expert_bc_sampler,
            coefficient=args.expert_bc_coefficient,
            hog_gate_weight=args.expert_bc_hog_gate_weight,
            hog_gate_scope=args.expert_bc_hog_gate_scope,
            hog_gate_gradient_scope=args.expert_bc_hog_gate_gradient_scope,
            loss_weights=expert_bc_loss_weights,
        )
    )
    bc_identities = [None for _ in range(world_size)]
    dist.all_gather_object(bc_identities, expert_bc_contract)
    if any(item != expert_bc_contract for item in bc_identities):
        raise RuntimeError("GPU ranks loaded different expert BC contracts")
    resume_contract = {
        # Strict part (STRICT_RESUME_KEYS): architecture / observation / action
        # contract and data contract.  The rest may change on resume.
        "schema": RESUME_CONTRACT_SCHEMA,
        "learner_contract": checkpoint_contract(current_model),
        "delay_policy": args.delay_policy,
        "gate_logit_offset": gate_logit_offset_applied,
        "behavior_temperatures": {
            "action": current_model.ppo_action_temperature,
            "continue": args.continue_temperature,
        },
        "gate_temperature_schedule": (
            args.gate_temperature_start,
            args.gate_temperature_end,
            args.gate_temperature_anneal_updates,
        ),
        "environment_warmup_ticks": int(args.environment_warmup_ticks),
        "training_command_delay": {
            "server_delay_ticks": 20, "card_jitter_ticks": (0, 3),
            "distribution": "uniform_integer_per_actor_compound", "earliest_total_ticks": (21, 24),
        },
        "decision_ticks": 5,
        "anchor_checkpoint_id": anchor_state_id,
        "il_anchor_kl": (
            None
            if anchor_kl_config is None
            else {**anchor_kl_config.contract(), "reference_checkpoint_id": reference_state_id,
                  "reference_state": "per-resident-actor frozen LSTM carried across rollout segments",
                  "temperatures": "actual rollout temperatures for both learner and reference"}
        ),
        "critic_warmup_iters": int(args.critic_warmup_iters),
        "reward_parameters": _king_reward_parameters(args),
        # Content identity (strict) so a run can resume on another cluster path.
        "deck_pool": "sha256:" + hashlib.sha256(args.deck_pool.read_bytes()).hexdigest(),
        "deck_pool_path": str(args.deck_pool.resolve()),
        "engine_count": engine_count,
        "resident_slots_per_engine": lanes,
        "world_size": world_size,
        "rollout_segment_seconds": args.rollout_segment_seconds,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "shaping_beta": args.shaping_beta,
        "terminal_reward_scale": args.terminal_reward_scale,
        "hog_deploy_reward": args.hog_deploy_reward,
        **(
            {"fireball_king_activation_penalty": args.fireball_king_activation_penalty}
            if args.fireball_king_activation_penalty
            else {}
        ),
        "hog_deploy_reward_episode_cap": args.hog_deploy_reward_episode_cap,
        **(
            {"hog_deploy_opening_reward": args.hog_deploy_opening_reward}
            if args.hog_deploy_opening_reward is not None
            else {}
        ),
        "first_hog_timing_reward_max": args.first_hog_timing_reward_max,
        "first_hog_timing_start_seconds": args.first_hog_timing_start_seconds,
        "first_hog_timing_deadline_seconds": (args.first_hog_timing_deadline_seconds),
        "first_hog_missed_deadline_penalty": (args.first_hog_missed_deadline_penalty),
        "first_hog_deferral_penalty": args.first_hog_deferral_penalty,
        "first_hog_deferral_penalty_episode_cap": (args.first_hog_deferral_penalty_episode_cap),
        "personal_elixir_overflow_penalty": (args.personal_elixir_overflow_penalty),
        "personal_elixir_overflow_grace": args.personal_elixir_overflow_grace,
        "unilateral_elixir_overflow_penalty": (args.unilateral_elixir_overflow_penalty),
        "elixir_overflow_step_penalty_cap": (args.elixir_overflow_step_penalty_cap),
        "clip_range": args.clip_range,
        "target_kl": args.target_kl,
        "ppo_epochs": args.ppo_epochs,
        "lane_minibatch_size": args.lane_minibatch_size,
        "lane_microbatch_size": args.lane_microbatch_size,
        "activation_checkpointing": args.activation_checkpointing,
        "sequence_chunk_steps": args.sequence_chunk_steps,
        "max_grad_norm": args.max_grad_norm,
        "gate_temperature_start": args.gate_temperature_start,
        "gate_temperature_end": args.gate_temperature_end,
        "gate_temperature_anneal_updates": (args.gate_temperature_anneal_updates),
        "continue_temperature": args.continue_temperature,
        "seed": args.seed,
        "battle_levels": BATTLE_LEVELS,
        "battle_level_split": "exact-50-50-shuffled-per-batch",
        "history_candidates_per_batch": HISTORY_CANDIDATES_PER_BATCH,
        "policy_mix_with_history_percent": (
            matchmaker_config.current_current,
            matchmaker_config.current_il,
            matchmaker_config.current_history,
        ),
        "opponent_mix_percent": opponent_mix,
        "il_opponent_decision": "argmax",
        "frozen_opponent_inference": {
            "loader": "load_policy_v4 (launch_interface.cmd)",
            "decision": "argmax",
            "delay_policy": DELAY_POLICY_EARLIEST,
            "additional_gate_logit_offset": 0.0,
            "temperatures": "checkpoint values (no learner overrides)",
            "scope": ["current-il", "history", "fixed"],
        },
        "self_play_percent": self_play_percent,
        "fixed_opponent_percent": fixed_opponent_percent,
        "history_percent": history_percent,
        "empty_history_fallback": POLICY_IL,
        "fixed_opponent_checkpoints": tuple(str(path) for path in resolved_fixed_paths),
        "fixed_opponent_il_max_average_elixir": (fixed_opponent_il_max_average_elixir),
        "learner_deck_id": learner_deck_id,
        "fixed_learner_opponent_mix": bool(args.fixed_learner_opponent_mix),
        "fixed_learner_opponent_deck_mix_percent": (30, 50, 20),
        "hard_deck_decay_per_batch": args.hard_deck_decay_per_batch,
        "hard_deck_win_rate_threshold": args.hard_deck_win_rate_threshold,
        "hard_deck_top_k": args.hard_deck_top_k,
        "learner_opponent_focus_percent": args.learner_opponent_focus_percent,
        "learner_opponent_weighted_percent": (args.learner_opponent_weighted_percent),
        "learner_opponent_hard_percent": args.learner_opponent_hard_percent,
        "focus_tail_current_focus_percent": 75,
        "opening_force": opening_force_config,
    }
    resume_contract["expert_bc"] = expert_bc_contract
    resume_extra = current_payload.get("extra")
    resume_contract_changes: dict[str, dict[str, Any]] = {}
    anchor_controller_restored = False
    if args.resume_checkpoint is not None:
        if not isinstance(resume_extra, Mapping) or (resume_extra.get("strict_boundary") is not True):
            raise ValueError("resume checkpoint is not a strict PPO boundary")
        resume_contract_changes = _check_resume_contract(resume_extra.get("resume_contract"), resume_contract)
        if anchor_controller is not None and isinstance(resume_extra.get("anchor_kl_controller_state"), Mapping):
            # Same anchor config -> continue the adaptive beta; changed -> restart from the CLI.
            anchor_controller_restored = anchor_controller.load_state_dict(resume_extra["anchor_kl_controller_state"])
        if rank == 0 and resume_contract_changes:
            print({"event": "ppo_resume_contract_changes", "changes": resume_contract_changes}, flush=True)
        initial_batch_index = int(resume_extra["next_batch_index"])
    else:
        initial_batch_index = 1

    identities = [None for _ in range(world_size)]
    dist.all_gather_object(identities, behavior_state_id)
    if len(set(identities)) != 1:
        raise RuntimeError("GPU ranks loaded different behavior checkpoints")
    gpu_names = [None for _ in range(world_size)]
    dist.all_gather_object(
        gpu_names,
        {
            "learner_device": str(device),
            "learner_name": torch.cuda.get_device_name(device),
            "actor_device": str(actor_device),
            "actor_name": torch.cuda.get_device_name(actor_device),
        },
    )
    matchmaker = DeckMatchmakerV4(manifest, seed=args.seed, config=matchmaker_config) if rank == 0 else None
    learner_opponent_sampler = (
        FixedLearnerOpponentDeckSamplerV4(
            manifest,
            seed=args.seed ^ 0x6826_5D31,
            decay_per_batch=args.hard_deck_decay_per_batch,
            hard_win_rate_threshold=args.hard_deck_win_rate_threshold,
            hard_top_k=args.hard_deck_top_k,
            focus_percent=args.learner_opponent_focus_percent,
            weighted_percent=args.learner_opponent_weighted_percent,
            hard_percent=args.learner_opponent_hard_percent,
        )
        if rank == 0 and args.fixed_learner_opponent_mix
        else None
    )
    league = None
    league_rng = None
    fixed_league = None
    if rank == 0:
        league_rng = random.Random(args.seed ^ 0x4F31_AA09)
        # Fixed opponents live in their own league (uniform), so the PFSP
        # snapshot pool keeps growing and sampling when fixed opponents exist.
        fixed_league = PolicyLeagueV4(anchor_checkpoint_id=anchor_state_id, anchor_checkpoint_path=il_anchor_checkpoint)
        for fixed_path in resolved_fixed_paths:
            if args.resume_checkpoint is None and fixed_path == args.checkpoint.resolve():
                fixed_checkpoint_id = source_checkpoint_id
                fixed_update_step = int(current_payload["update_step"])
            else:
                fixed_checkpoint_id, fixed_update_step = _checkpoint_identity(fixed_path)
            fixed_league.add_snapshot(fixed_checkpoint_id, fixed_path, update_step=fixed_update_step)
        if args.resume_checkpoint is None:
            league = PolicyLeagueV4(anchor_checkpoint_id=anchor_state_id, anchor_checkpoint_path=il_anchor_checkpoint)
        else:
            assert isinstance(resume_extra, Mapping) and matchmaker is not None
            matchmaker_state = resume_extra.get("matchmaker_state")
            league_state = resume_extra.get("league_state")
            if not isinstance(matchmaker_state, dict) or not isinstance(league_state, Mapping):
                raise ValueError("resume checkpoint lacks matchmaker or league state")
            matchmaker.load_state_dict(matchmaker_state)
            if learner_opponent_sampler is not None:
                sampler_state = resume_extra.get("learner_opponent_sampler_state")
                if not isinstance(sampler_state, Mapping):
                    raise ValueError("resume checkpoint lacks learner opponent sampler state")
                learner_opponent_sampler.load_state_dict(sampler_state)
            league = PolicyLeagueV4.from_dict(league_state)
            if league.anchor_checkpoint_id != anchor_state_id:
                # The IL anchor is an opponent/anchor parameter: allowed to change on resume.
                print(
                    {
                        "event": "ppo_resume_il_anchor_changed",
                        "checkpoint": league.anchor_checkpoint_id,
                        "current": anchor_state_id,
                    },
                    flush=True,
                )
                league = PolicyLeagueV4(
                    anchor_checkpoint_id=anchor_state_id,
                    anchor_checkpoint_path=il_anchor_checkpoint,
                    entries=tuple(league.entries),
                )
            stored_fixed = resume_extra.get("fixed_league_state")
            if isinstance(stored_fixed, Mapping):
                previous = {item.checkpoint_id: item for item in PolicyLeagueV4.from_dict(stored_fixed).entries}
                fixed_league.entries = [previous.get(item.checkpoint_id, item) for item in fixed_league.entries]
            league_rng.setstate(resume_extra["league_rng_state"])
    history_cache: dict[str, UniversalCardPolicyV4] = {}

    async_engine_specs = tuple(
        AsyncResidentEngineSpecV4(
            engine_index=engine_index,
            cpu=engine_cpus[engine_index],
            host=engine_hosts[engine_index],
            base_port=engine_ports[engine_index] - engine_index,
            lanes=lanes,
            timeout=args.timeout,
            capture_mode="compact",
            worker_host=engine_worker_hosts[engine_index],
            worker_port=engine_worker_ports[engine_index],
            **opening_force_options,
            environment_warmup_ticks=int(args.environment_warmup_ticks),
            training_command_delay_jitter_ticks=3,
        )
        for engine_index in engine_indices
    )
    # Learner and KL reference share the training delay policy. Frozen battle
    # opponents independently retain the interface's earliest delay policy.
    for label, model in (("learner", current_model), ("reference", reference_model)):
        if model is not None and model.delay_policy != args.delay_policy:
            raise RuntimeError(f"{label} delay policy {model.delay_policy!r} != {args.delay_policy!r}")
    dist.barrier()

    wandb_run: Any | None = None
    if rank == 0:
        # The resume contract is the source of truth for model/reward/sampler
        # settings. Keep the exact CLI and runtime identity alongside it so the
        # run remains reproducible without a second hand-maintained description.
        run_config = {
            **resume_contract,
            "schema": "v4-ppo-segmented-production.v1",
            "arguments": {
                name: (
                    str(value.resolve())
                    if isinstance(value, Path)
                    else [str(item.resolve()) if isinstance(item, Path) else item for item in value]
                    if isinstance(value, list)
                    else value
                )
                for name, value in vars(args).items()
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "topology": str(args.engine_ready.resolve()),
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_id": source_checkpoint_id,
            "il_anchor_checkpoint": str(il_anchor_checkpoint.resolve()),
            "il_anchor_checkpoint_id": anchor_state_id,
            "initial_policy_checkpoint_id": behavior_state_id,
            "initial_update_step": initial_update_step,
            "learner_fixed_deck": None if learner_deck is None else learner_deck.to_dict(),
            "deck_count": len(manifest.decks),
            "focus_deck_count": len(manifest.focus_decks),
            "gpus": gpu_names,
            "parallel_matches": engine_count * lanes,
            "ruleset_id": ppo_ruleset_id_v4(environment_config),
            "decision_ticks": 5,
            "policy_gameplay_end_tick": NATIVE_GAMEPLAY_END_TICK,
            "rollout_segment_decision_steps": args.rollout_segment_seconds * 4,
            "behavior_temperatures": {
                "gate": current_model.ppo_gate_temperature,
                "action": current_model.ppo_action_temperature,
                "continue": current_model.ppo_continue_temperature,
            },
            "entropy_coefficient": 0.0,
            "inference_batch_wait_ms": 2.0,
            "inference_batch_wait_max_ms": 25.0 if args.rollout_adaptive_batching else 2.0,
            "capture_mode": "compact",
            "collector": "remote-segmented-cuda-graph",
            "expert_bc": expert_bc_contract,
            "expert_bc_loss_weights": asdict(expert_bc_loss_weights) if expert_bc_contract is not None else None,
            "anchor_checkpoint": None if args.anchor_checkpoint is None else str(args.anchor_checkpoint.resolve()),
            "anchor_reference_checkpoint_id": reference_state_id,
            "anchor_kl_controller_restored": anchor_controller_restored,
            "resume_contract_changes": resume_contract_changes,
            "learner_model_config": asdict(current_model.config),
            "gate_logit_offset_applied": gate_logit_offset_applied,
            "wandb": {"mode": "offline", "project": WANDB_PROJECT},
        }
        (args.output_dir / "config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not args.preflight_only:
            wandb_run = _init_offline_wandb(args.output_dir, run_config)

    if args.preflight_only:
        if rank == 0:
            preflight = {
                "startup_validated": True,
                "world_size": world_size,
                "rollout_segment_seconds": args.rollout_segment_seconds,
                "gate_temperature_start": args.gate_temperature_start,
                "gate_temperature_end": args.gate_temperature_end,
                "behavior_temperatures": run_config["behavior_temperatures"],
                "gate_logit_offset_applied": gate_logit_offset_applied,
                "learner_and_opponent_gate_bias_equal": torch.equal(
                    current_model.gate_head[-1].bias, anchor_model.gate_head[-1].bias),
                "reference_enabled": reference_model is not None,
                "reference_state": "independent recurrence carried across resident segments",
                "opening_force": opening_force_config,
                "gae_lambda": args.gae_lambda,
                "shaping_beta": args.shaping_beta,
                "collection_started": False,
                "optimizer_steps": 0,
            }
            (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
            print(json.dumps({"event": "ppo_preflight_complete", **preflight}), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    ppo_config = PPOConfigV4(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_clip_range=args.clip_range,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    # Stored with every PPO checkpoint: the gate correction is already in the
    # gate bias (never applied twice on reload), and the delay policy it used.
    checkpoint_policy_extra = {
        GATE_LOGIT_OFFSET_EXTRA_KEY: gate_logit_offset_applied,
        "delay_policy": args.delay_policy,
    }

    def prepare_collection(
        policy_state_id: str,
    ) -> tuple[tuple[ClusterMatchAssignmentV4, ...], dict[str, UniversalCardPolicyV4], str]:
        if rank == 0:
            assert matchmaker is not None and league is not None and league_rng is not None
            assert fixed_league is not None
            split_shares = fixed_opponent_percent + history_percent > 0
            schedule = _balanced_assignments(
                matchmaker,
                league,
                league_rng,
                match_capacities=match_capacities,
                rich_capacities=rich_capacities,
                special_towers_require_rich=False,
                fixed_opponent_il_max_average_elixir=(fixed_opponent_il_max_average_elixir),
                deck_average_elixir_by_id=deck_average_elixir_by_id,
                learner_deck=learner_deck,
                learner_opponent_sampler=learner_opponent_sampler,
                fixed_league=fixed_league if split_shares else None,
                fixed_opponent_percent=fixed_opponent_percent,
                history_percent=history_percent,
            )
        else:
            schedule = None
        schedule = _broadcast_object(schedule)
        if not isinstance(schedule, tuple) or len(schedule) != world_size:
            raise RuntimeError("distributed schedule broadcast is invalid")
        local = schedule[rank]
        if len(local) != matches_per_rank:
            raise RuntimeError("rank schedule does not fill its engines")
        rng_state = _rank_rng_state(device)
        try:
            historical = _load_history_models(
                local,
                history_cache,
                device=actor_device,
            )
        finally:
            _set_rank_rng_state(rng_state, device=device)
        return local, historical, str(policy_state_id)

    def collect_actor_wave(
        local: tuple[ClusterMatchAssignmentV4, ...],
        historical: Mapping[str, UniversalCardPolicyV4],
        policy_state_id: str,
        segment_callback: Any,
    ) -> ClusterCollectionV4:
        caller_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(learner_rank_cpus))
        actor_model.eval()
        try:
            with torch.cuda.stream(segment_actor_stream):
                result = collect_async_cluster_wave_v4(
                    async_engine_specs,
                    local,
                    actor_model,
                    anchor_model,
                    historical,
                    gamma_per_decision=args.gamma,
                    shaping_beta=args.shaping_beta,
                    terminal_reward_scale=args.terminal_reward_scale,
                    fireball_king_activation_penalty=args.fireball_king_activation_penalty,
                    indexed_state_graphs=args.rollout_indexed_state_graphs,
                    adaptive_batching=args.rollout_adaptive_batching,
                    cache_static_memory=args.rollout_cache_static_memory,
                    **_king_reward_parameters(args),
                    hog_deploy_reward=args.hog_deploy_reward,
                    hog_deploy_reward_episode_cap=args.hog_deploy_reward_episode_cap,
                    hog_deploy_opening_reward=args.hog_deploy_opening_reward,
                    first_hog_timing_reward_max=(args.first_hog_timing_reward_max),
                    first_hog_timing_start_seconds=(args.first_hog_timing_start_seconds),
                    first_hog_timing_deadline_seconds=(args.first_hog_timing_deadline_seconds),
                    first_hog_missed_deadline_penalty=(args.first_hog_missed_deadline_penalty),
                    first_hog_deferral_penalty=(args.first_hog_deferral_penalty),
                    first_hog_deferral_penalty_episode_cap=(args.first_hog_deferral_penalty_episode_cap),
                    mutual_elixir_overflow_penalty=(args.personal_elixir_overflow_penalty),
                    mutual_elixir_overflow_grace=(args.personal_elixir_overflow_grace),
                    unilateral_elixir_overflow_penalty=(args.unilateral_elixir_overflow_penalty),
                    elixir_overflow_step_penalty_cap=(args.elixir_overflow_step_penalty_cap),
                    policy_state_id=policy_state_id,
                    tolerate_episode_failures=True,
                    max_decision_steps=max_decision_steps,
                    validate_tensors=args.validate,
                    segment_decision_steps=args.rollout_segment_seconds * 4,
                    segment_callback=segment_callback,
                )
            segment_actor_stream.synchronize()
            return result
        finally:
            os.sched_setaffinity(0, caller_affinity)

    segment_executor: ThreadPoolExecutor | None = None
    segment_future: Future[ClusterCollectionV4] | None = None
    segment_collections: Queue[ClusterCollectionV4] | None = None
    segment_resumes: Queue[str | None] | None = None
    segment_assignments: tuple[ClusterMatchAssignmentV4, ...] | None = None
    segment_history_models: dict[str, UniversalCardPolicyV4] | None = None
    segment_session_can_resume = True
    reference_states = ReferenceStateCacheV4() if reference_model is not None else None
    resume_rng_pending = args.resume_checkpoint is not None

    def start_segment_session(policy_state_id: str) -> None:
        nonlocal segment_assignments, segment_history_models
        nonlocal segment_collections, segment_resumes
        nonlocal segment_executor, segment_future
        nonlocal resume_rng_pending
        if segment_executor is not None or segment_future is not None:
            raise RuntimeError("segmented collector session is already running")
        if reference_states is not None:
            reference_states.clear()
        (segment_assignments, segment_history_models, segment_policy_state_id) = prepare_collection(policy_state_id)
        if resume_rng_pending:
            _restore_rank_rng_state(current_payload, rank=rank, world_size=world_size, device=device)
            resume_rng_pending = False
        collection_queue: Queue[ClusterCollectionV4] = Queue(maxsize=1)
        resume_queue: Queue[str | None] = Queue(maxsize=1)
        segment_collections = collection_queue
        segment_resumes = resume_queue

        def publish_segment(collection: ClusterCollectionV4) -> str | None:
            collection_queue.put(collection)
            return resume_queue.get()

        segment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ppo-rank-{rank}-segment-session")
        segment_future = segment_executor.submit(
            collect_actor_wave, segment_assignments, segment_history_models, segment_policy_state_id, publish_segment
        )

    def next_segment_collection() -> ClusterCollectionV4:
        if segment_collections is None or segment_future is None:
            raise RuntimeError("segmented collector session is unavailable")
        while True:
            try:
                return segment_collections.get(timeout=0.1)
            except Empty:
                if segment_future.done():
                    segment_future.result()
                    raise RuntimeError("segmented collector exited before publishing")

    batch_index = initial_batch_index
    batch_completed_games = 0
    wave_episode_failures = 0
    episode_failures_path = args.output_dir / "episode_failures.jsonl"
    batch_collection_seconds = 0.0
    batch_end_to_end_seconds = 0.0
    batch_reports: list[Mapping[str, object]] = []
    batch_segments: list[Mapping[str, object]] = []
    run_failed = True
    phase_iteration = initial_update_step + 1

    def publish_phase(phase: str, iteration: int) -> None:
        nonlocal phase_iteration
        phase_iteration = iteration
        if rank == 0:
            _write_phase(phase_path, phase, iteration)

    try:
        publish_phase("rollout", initial_update_step + 1)
        start_segment_session(actor_state_id)
        for update_step in range(initial_update_step + 1, args.updates + 1):
            dist.barrier()
            cycle_started = time.perf_counter()
            if segment_assignments is None or segment_history_models is None:
                raise RuntimeError("segmented collector metadata is incomplete")
            collected = next_segment_collection()
            post_collection_started = time.perf_counter()
            publish_phase("critic_warmup" if update_step <= args.critic_warmup_iters else "update", update_step)
            history_models = segment_history_models
            collected_policy_state_id = collected.rollout.policy_state_id
            if not args.retain_rollout_cuda_graphs:
                _release_rollout_cuda_graphs((actor_model, anchor_model, *history_models.values()), device=actor_device)
            gathered = _gather_objects(
                {"reports": collected.reports, "timing": collected.timing}, rank=rank, world_size=world_size
            )
            if rank == 0:
                assert gathered is not None and league is not None
                reports = tuple(
                    report
                    for shard in gathered
                    for report in shard["reports"]  # type: ignore[index]
                )
                rejected = sum(sum(int(value) for value in report["rejected_actions"]) for report in reports)
                truncated = sum(bool(report["truncated"]) for report in reports)
                # Dropped matches (worker exceptions): logged, counted per wave,
                # and fatal only above --max-episode-failure-rate.
                segment_failures = _episode_failures_from_shards(gathered)  # type: ignore[arg-type]
                for row in segment_failures:
                    _json_log(episode_failures_path, {"update": update_step, "batch_index": batch_index, **row})
                wave_episode_failures += len(segment_failures)
                episode_failure_rate, failure_rate_ok = _episode_failure_rate(
                    wave_episode_failures, engine_count * lanes, max_rate=args.max_episode_failure_rate
                )
                collection_ok = (
                    all(
                        float(shard["timing"]["owner_frames"]) > 0  # type: ignore[index]
                        for shard in gathered
                    )
                    and truncated == 0
                    and rejected == 0
                    and failure_rate_ok
                )
                failure = (
                    None
                    if collection_ok
                    else {
                        "matches": len(reports),
                        "episode_failures_this_wave": wave_episode_failures,
                        "episode_failure_rate": episode_failure_rate,
                        "max_episode_failure_rate": args.max_episode_failure_rate,
                        "last_episode_failure": segment_failures[-1] if segment_failures else None,
                        "rejected_actions": rejected,
                        "truncated": truncated,
                        "rejection_details": tuple(
                            {
                                "sequence": report["sequence"],
                                "engine_index": report["engine_index"],
                                "env_id": report["env_id"],
                                "details": report["rejection_details"],
                            }
                            for report in reports
                            if report["rejection_details"]
                        ),
                    }
                )
                if collection_ok:
                    for report in reports:
                        checkpoint_id = report["opponent_checkpoint_id"]
                        if checkpoint_id is not None:
                            learner_owner = int(report["current_owners"][0])
                            assert fixed_league is not None
                            target_league = (
                                fixed_league
                                if any(item.checkpoint_id == str(checkpoint_id) for item in fixed_league.entries)
                                else league
                            )
                            target_league.record_result(str(checkpoint_id), float(report["result"][learner_owner]))
                segment_session_can_resume = all(
                    float(shard["timing"].get("live_engines", 1)) > 0  # type: ignore[index]
                    for shard in gathered
                )
            else:
                reports = ()
                collection_ok = False
                failure = None
                segment_session_can_resume = False
            collection_status = _broadcast_object((collection_ok, failure, segment_session_can_resume))
            if collection_status[0] is not True:
                raise RuntimeError(f"cluster collection failed audit: {collection_status[1]}")
            segment_session_can_resume = bool(collection_status[2])
            if rank == 0:
                assert gathered is not None
                collection_seconds_for_event = max(float(shard["timing"]["total_seconds"]) for shard in gathered)
                collection_owner_frames_for_event = sum(int(shard["timing"]["owner_frames"]) for shard in gathered)
                collection_episode_start_count = sum(
                    int(shard["timing"].get("episode_start_count", 0)) for shard in gathered
                )
                collection_truncated_count = sum(int(shard["timing"].get("truncated_count", 0)) for shard in gathered)
                collection_boundary_bootstrap_count = sum(
                    int(shard["timing"].get("boundary_bootstrap_count", 0)) for shard in gathered
                )
                collection_initial_hidden_abs_max = max(
                    float(shard["timing"].get("initial_hidden_abs_max", 0.0)) for shard in gathered
                )
                print(
                    {
                        "event": "ppo_collection_complete",
                        "update": update_step,
                        "completed_games": len(reports),
                        "collection_seconds": collection_seconds_for_event,
                        "collection_owner_frames": collection_owner_frames_for_event,
                        "collection_owner_frames_per_second": (
                            collection_owner_frames_for_event / collection_seconds_for_event
                        ),
                        "collection_episode_start_count": (collection_episode_start_count),
                        "collection_truncated_count": collection_truncated_count,
                        "collection_boundary_bootstrap_count": (collection_boundary_bootstrap_count),
                        "collection_initial_hidden_abs_max": (collection_initial_hidden_abs_max),
                        "native_rejected_actions": rejected,
                    },
                    flush=True,
                )

            learner_input_state_id = behavior_state_id
            current_model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            learner_rollout = collected.rollout
            rollout_gate_temperature = float(learner_rollout.gate_temperature)
            source_owner_frames_local = int(learner_rollout.valid_mask.sum())
            behavior_count_started = time.perf_counter()
            behavior_counts = _rollout_behavior_counts(learner_rollout, current_model).to(device)
            dist.all_reduce(behavior_counts, op=dist.ReduceOp.SUM)
            behavior_metrics = _rollout_behavior_metrics(
                behavior_counts.cpu(), delay_offset_ms=current_model.config.delay_offset_ms
            )
            behavior_count_seconds = time.perf_counter() - behavior_count_started
            pre_optimizer_seconds = time.perf_counter() - post_collection_started
            update_started = time.perf_counter()
            # Reports and timing were gathered above; the learner consumes every
            # valid lane and can release the collection wrapper before optimization.
            del collected
            gc.collect()
            # cuDNN RNN backward requires training mode.  The model has no dropout
            # or batch norm (LSTM dropout 0, Layer/GroupNorm only), so train() does
            # not change any log-prob; the rollout resumes in eval() below.
            current_model.train()
            critic_warmup = update_step <= args.critic_warmup_iters
            anchor_coefficient = 0.0 if anchor_controller is None else anchor_controller.value
            anchor_continue_coefficient = 0.0 if anchor_controller is None else anchor_controller.continue_value
            reference_initial_state = (
                None if reference_states is None else reference_states.prepare(reference_model, learner_rollout)
            )
            reference_final_state = (
                None if reference_model is None else reference_model.initial_state(learner_rollout.batch_size)
            )
            update_metrics = distributed_ppo_update_v4(
                current_model,
                optimizer,
                learner_rollout,
                epochs=args.ppo_epochs,
                lane_minibatch_size=args.lane_minibatch_size,
                lane_microbatch_size=args.lane_microbatch_size,
                sequence_chunk_steps=args.sequence_chunk_steps,
                config=ppo_config,
                expected_policy_state_id=collected_policy_state_id,
                validate=args.validate,
                reference_model=reference_model,
                reference_initial_state=reference_initial_state,
                reference_final_state_out=reference_final_state,
                anchor_kl_coefficient=anchor_coefficient,
                anchor_continue_kl_coefficient=anchor_continue_coefficient,
                value_only=critic_warmup,
                device_rollout_budget_gib=args.rollout_device_cache_gib,
            )
            anchor_metrics: dict[str, float] = {}
            if reference_states is not None:
                reference_states.commit(learner_rollout, reference_final_state)
            if anchor_controller is not None:
                kl_rows = [row for row in update_metrics if "anchor_kl" in row]
                kl_weight = sum(float(row["valid_count"]) for row in kl_rows)
                observed_kl = (
                    sum(float(row["anchor_kl"]) * float(row["valid_count"]) for row in kl_rows) / kl_weight
                    if kl_weight > 0
                    else 0.0
                )
                # The actor is frozen during critic warmup: KL stays ~0 and would
                # only drive beta down, so the controller waits for the actor.
                if kl_rows and not critic_warmup:
                    anchor_controller.update(observed_kl)
                anchor_metrics = {
                    "anchor_kl_observed": observed_kl,
                    "anchor_kl_target": anchor_controller.config.target_kl,
                    "anchor_kl_coefficient_used": anchor_coefficient,
                    "anchor_continue_kl_coefficient_used": anchor_continue_coefficient,
                    "anchor_kl_coefficient_next": anchor_controller.value,
                    "anchor_continue_kl_coefficient_next": anchor_controller.continue_value,
                    "anchor_controller_updates": anchor_controller.updates,
                }
            bc_metrics = {}
            # Expert BC moves the actor, so it waits for the critic warmup too.
            if expert_bc_sampler is not None and not critic_warmup:
                bc_metrics = distributed_expert_bc_update_v4(
                    current_model,
                    optimizer,
                    expert_bc_sampler,
                    update_step=update_step,
                    sequence_chunk_steps=args.sequence_chunk_steps,
                    coefficient=args.expert_bc_coefficient,
                    hog_gate_weight=args.expert_bc_hog_gate_weight,
                    hog_gate_scope=args.expert_bc_hog_gate_scope,
                    hog_gate_gradient_scope=args.expert_bc_hog_gate_gradient_scope,
                    max_grad_norm=args.max_grad_norm,
                    rollout=learner_rollout,
                    validate=args.validate,
                    loss_weights=expert_bc_loss_weights,
                )
                if rank == 0:
                    print({"event": "ppo_expert_bc_complete", "update": update_step, **bc_metrics}, flush=True)
            next_gate_temperature = _gate_temperature_for_update(
                update_step + 1,
                start=args.gate_temperature_start,
                end=args.gate_temperature_end,
                anneal_updates=args.gate_temperature_anneal_updates,
            )
            current_model.set_ppo_gate_temperature(next_gate_temperature)
            # Back to inference mode before the collector resumes on this model.
            current_model.eval()
            update_seconds_local = time.perf_counter() - update_started
            post_optimizer_started = time.perf_counter()
            update_seconds = torch.tensor(update_seconds_local, device=device)
            dist.all_reduce(update_seconds, op=dist.ReduceOp.MAX)
            peak_cuda = torch.tensor(
                [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(peak_cuda, op=dist.ReduceOp.MAX)
            rank_rng_states: list[object | None] = [None for _ in range(world_size)]
            dist.all_gather_object(rank_rng_states, _rank_rng_state(device))

            if rank == 0:
                assert matchmaker is not None and league is not None and league_rng is not None
                if learner_opponent_sampler is not None and not segment_session_can_resume:
                    completed_batch_reports = (*batch_reports, *reports)
                    learner_opponent_sampler.record_completed_batch(
                        tuple(
                            (
                                str(report["policy_matchup"]),
                                str(
                                    report["deck1_id"] if int(report["current_owners"][0]) == 0 else report["deck0_id"]
                                ),
                                float(report["result"][int(report["current_owners"][0])]),
                            )
                            for report in completed_batch_reports
                        )
                    )
                behavior_state_id = uuid.uuid4().hex
                if update_step % args.snapshot_every == 0:
                    league_checkpoint = args.output_dir / "league-checkpoints" / f"update-{update_step:08d}.pt"
                    league_checkpoint_id = save_actor_critic_checkpoint(
                        league_checkpoint,
                        current_model,
                        update_step=update_step,
                        training_stage="ppo-league-snapshot",
                        gamma_per_decision=args.gamma,
                        gae_lambda=args.gae_lambda,
                        extra=checkpoint_policy_extra,
                    )
                    # Snapshots always join the PFSP pool, also with fixed opponents.
                    league.add_snapshot(league_checkpoint_id, league_checkpoint, update_step=update_step)
                league.save(league_path)
                strict_boundary = not segment_session_can_resume or update_step == args.updates
                if strict_boundary:
                    checkpoint_path = args.output_dir / "checkpoints" / f"update-{update_step:08d}.pt"
                    behavior_state_id = save_actor_critic_checkpoint(
                        checkpoint_path,
                        current_model,
                        optimizer=optimizer,
                        update_step=update_step,
                        training_stage="ppo-distributed",
                        gamma_per_decision=args.gamma,
                        gae_lambda=args.gae_lambda,
                        extra={
                            **checkpoint_policy_extra,
                            "strict_boundary": True,
                            "resume_contract": resume_contract,
                            "fixed_league_state": None if fixed_league is None else fixed_league.to_dict(),
                            "anchor_kl_controller_state": (
                                None if anchor_controller is None else anchor_controller.state_dict()
                            ),
                            "rank_rng_states": rank_rng_states,
                            "matchmaker_state": matchmaker.state_dict(),
                            "learner_opponent_sampler_state": (
                                None if learner_opponent_sampler is None else learner_opponent_sampler.state_dict()
                            ),
                            "league_state": league.to_dict(),
                            "league_rng_state": league_rng.getstate(),
                            "next_batch_index": batch_index + 1,
                        },
                    )
                    _publish_strict_checkpoint(args.output_dir, checkpoint_path)
            behavior_state_id = str(_broadcast_object(behavior_state_id if rank == 0 else None))
            actor_state_id = behavior_state_id
            dist.barrier()
            cycle_seconds_local = time.perf_counter() - cycle_started
            post_optimizer_seconds = time.perf_counter() - post_optimizer_started
            cycle_seconds = torch.tensor(cycle_seconds_local, device=device)
            dist.all_reduce(cycle_seconds, op=dist.ReduceOp.MAX)

            owner_frames = torch.tensor(source_owner_frames_local, device=device, dtype=torch.long)
            dist.all_reduce(owner_frames, op=dist.ReduceOp.SUM)
            learner_owner_frames = torch.tensor(int(learner_rollout.valid_mask.sum()), device=device, dtype=torch.long)
            dist.all_reduce(learner_owner_frames, op=dist.ReduceOp.SUM)
            learner_lanes = torch.tensor(learner_rollout.batch_size, device=device, dtype=torch.long)
            dist.all_reduce(learner_lanes, op=dist.ReduceOp.SUM)
            if rank == 0:
                total_matches = len(reports)
                batch_reports.extend(reports)
                effective_end_to_end_seconds = float(cycle_seconds)
                collection_seconds = max(
                    float(shard["timing"]["total_seconds"])  # type: ignore[index]
                    for shard in gathered  # type: ignore[union-attr]
                )
                collection_phase_seconds = {
                    f"collection_{name}": max(
                        float(shard["timing"][name])  # type: ignore[index]
                        for shard in gathered  # type: ignore[union-attr]
                    )
                    for name in (
                        "reset_seconds",
                        "tensorize_seconds",
                        "inference_seconds",
                        "native_step_seconds",
                        "batch_channel_total_seconds",
                        "batch_channel_request_ack_seconds",
                        "batch_channel_receive_seconds",
                        "batch_channel_decode_seconds",
                        "batch_channel_response_mib",
                        "batch_channel_decoded_response_mib",
                        "assembly_seconds",
                        "model_seconds",
                        "action_seconds",
                        "storage_seconds",
                        "rollout_finalize_seconds",
                        "deferred_storage_finalize_seconds",
                        "packed_tail_padding_seconds",
                        "command_seconds",
                        "inference_batches",
                        "inference_mean_frame_packets",
                        "inference_mean_actor_rows",
                        "inference_max_frame_packets",
                        "max_relation_count",
                        "max_effect_count",
                    )
                    if all(
                        name in shard["timing"]  # type: ignore[operator]
                        for shard in gathered  # type: ignore[union-attr]
                    )
                }
                batch_completed_games += total_matches
                batch_collection_seconds += collection_seconds
                batch_end_to_end_seconds += effective_end_to_end_seconds
                batch_collection_games_per_minute = 60.0 * batch_completed_games / batch_collection_seconds
                batch_end_to_end_games_per_minute = 60.0 * batch_completed_games / batch_end_to_end_seconds
                averaged = _mean_metrics(update_metrics)
                metric = {
                    "stage": "ppo",
                    "update": update_step,
                    "checkpoint_id": behavior_state_id,
                    "learner_input_checkpoint_id": learner_input_state_id,
                    "rollout_policy_state_id": collected_policy_state_id,
                    "actor_checkpoint_id_after_update": actor_state_id,
                    "rollout_gate_temperature": rollout_gate_temperature,
                    "next_gate_temperature": next_gate_temperature,
                    "completed_games_this_segment": total_matches,
                    "batch_index": batch_index,
                    "batch_completed_games": batch_completed_games,
                    "batch_progress_fraction": (batch_completed_games / (engine_count * lanes)),
                    "batch_complete": not segment_session_can_resume,
                    "batch_collection_games_per_minute": (batch_collection_games_per_minute),
                    "batch_end_to_end_games_per_minute": (batch_end_to_end_games_per_minute),
                    "owner_frames": int(owner_frames),
                    "learner_owner_frames": int(learner_owner_frames),
                    "learner_owner_frame_fraction": (float(learner_owner_frames) / max(float(owner_frames), 1.0)),
                    "learner_lanes": int(learner_lanes),
                    "collection_seconds": collection_seconds,
                    "collection_owner_frames_per_second": (float(owner_frames) / max(collection_seconds, 1e-9)),
                    "collection_episode_start_count": (collection_episode_start_count),
                    "collection_truncated_count": collection_truncated_count,
                    "collection_boundary_bootstrap_count": (collection_boundary_bootstrap_count),
                    "collection_initial_hidden_abs_max": (collection_initial_hidden_abs_max),
                    **collection_phase_seconds,
                    "episode_failure_count": len(segment_failures),
                    "wave_episode_failures": wave_episode_failures,
                    "episode_failure_rate": episode_failure_rate,
                    "native_rejected_actions": sum(
                        sum(int(value) for value in report["rejected_actions"]) for report in reports
                    ),
                    "optimizer_seconds": float(update_seconds),
                    "rank0_behavior_count_seconds": behavior_count_seconds,
                    "rank0_pre_optimizer_seconds": pre_optimizer_seconds,
                    "rank0_post_optimizer_seconds": post_optimizer_seconds,
                    "optimizer_peak_cuda_allocated_gib": (float(peak_cuda[0]) / 1024**3),
                    "optimizer_peak_cuda_reserved_gib": (float(peak_cuda[1]) / 1024**3),
                    "sequential_iteration_seconds": float(cycle_seconds),
                    "end_to_end_seconds": effective_end_to_end_seconds,
                    "end_to_end_owner_frames_per_second": (
                        float(owner_frames) / max(effective_end_to_end_seconds, 1e-9)
                    ),
                    "rollout_segment_seconds": args.rollout_segment_seconds,
                    "minibatch_updates": len(update_metrics),
                    "league_history_snapshots": len(league.entries),
                    "learner_opponent_hard_sampler": (
                        None if learner_opponent_sampler is None else learner_opponent_sampler.summary()
                    ),
                    **behavior_metrics,
                    **averaged,
                    **anchor_metrics,
                    "critic_warmup": critic_warmup,
                    "delay_policy": args.delay_policy,
                    # SECOND rate over continue-eligible ACT ticks of this rollout.
                    "second_rate": behavior_metrics["continue_eligible_second_rate"],
                    "fixed_league_entries": None if fixed_league is None else len(fixed_league.entries),
                    **bc_metrics,
                }
                batch_segments.append(metric)
                _json_log(metrics_path, metric)
                for report in reports:
                    _json_log(matches_path, {"update": update_step, **report})
                if wandb_run is None:
                    raise RuntimeError("rank 0 lost its W&B offline run")
                logging_rng_state = _rank_rng_state(device)
                try:
                    wandb_metrics = _wandb_update_metrics(metric)
                    if bool(metric["batch_complete"]):
                        wandb_metrics.update(_batch_wandb_metrics(batch_segments, batch_reports))
                    wandb_run.log(wandb_metrics, step=update_step)
                finally:
                    _set_rank_rng_state(logging_rng_state, device=device)
                print(json.dumps(metric, sort_keys=True), flush=True)
            # Release the consumed rollout before resuming the resident matches.
            del learner_rollout
            del update_metrics
            gc.collect()
            if segment_resumes is not None:
                if segment_actor_stream is not None:
                    torch.cuda.current_stream(device).synchronize()
                resume_state_id = (
                    behavior_state_id if update_step < args.updates and segment_session_can_resume else None
                )
                if resume_state_id is not None:
                    publish_phase("rollout", update_step + 1)
                segment_resumes.put(resume_state_id)
                if resume_state_id is None and segment_future is not None:
                    segment_future.result()
                    if segment_executor is not None:
                        segment_executor.shutdown(wait=True)
                    segment_future = None
                    segment_executor = None
                    segment_collections = None
                    segment_resumes = None
                    segment_assignments = None
                    segment_history_models = None
                    if update_step < args.updates:
                        batch_index += 1
                        batch_completed_games = 0
                        wave_episode_failures = 0
                        batch_collection_seconds = 0.0
                        batch_end_to_end_seconds = 0.0
                        batch_reports.clear()
                        batch_segments.clear()
                        publish_phase("rollout", update_step + 1)
                        start_segment_session(behavior_state_id)
        run_failed = False
    finally:
        if rank == 0:
            try:
                _write_phase(phase_path, "idle", phase_iteration)
            except OSError:
                pass
        if segment_resumes is not None and segment_future is not None and not (segment_future.done()):
            try:
                segment_resumes.put_nowait(None)
            except Exception:
                pass
        if segment_executor is not None:
            segment_executor.shutdown(wait=True, cancel_futures=True)
        if wandb_run is not None:
            wandb_run.finish(exit_code=int(run_failed))
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
