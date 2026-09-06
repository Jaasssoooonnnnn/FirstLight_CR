"""Multi-GPU, multi-host resident PPO self-play trainer for distributed training."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import gc
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

from ...match_factory import NATIVE_GAMEPLAY_END_TICK
from .async_cluster_self_play import AsyncResidentEngineSpecV4, collect_async_cluster_wave_v4
from .checkpoint import CHECKPOINT_SCHEMA_V4, load_actor_critic_checkpoint, save_actor_critic_checkpoint
from .cluster_self_play import ClusterCollectionV4, ClusterMatchAssignmentV4
from .deck_pool import DeckIdentityV4, DeckPoolManifestV4
from .decoding import ShadowCandidateLegality
from .distributed_ppo import distributed_ppo_update_v4
from .ppo_expert_bc import HogExpertSamplerV4, distributed_expert_bc_update_v4, expert_bc_contract_v4
from .factory import build_production_model_v4
from .league import PolicyLeagueV4
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
from .ppo_runtime import ppo_environment_config_v4, ppo_ruleset_id_v4, validate_reward_parameters_v4
from .tensors import GATE_ACT, GATE_WAIT


DEFAULT_GATE_TEMPERATURE_START = 0.20
DEFAULT_GATE_TEMPERATURE_END = 0.20
DEFAULT_GATE_TEMPERATURE_ANNEAL_UPDATES = 1
DEFAULT_ACTION_TEMPERATURE = 1.0
DEFAULT_CONTINUE_TEMPERATURE = 2.0
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

    delay_bins = len(model.config.delay_offset_ms)
    counts = torch.zeros(10 + delay_bins, dtype=torch.long)
    forced_gate_mask = rollout.forced_gate_mask
    for step, (observation, action) in enumerate(zip(rollout.observations, rollout.actions, strict=True)):
        valid = rollout.valid_mask[step].cpu()
        forced_gate = torch.zeros_like(valid) if forced_gate_mask is None else forced_gate_mask[step].cpu()
        shadow = ShadowCandidateLegality(observation.candidates, model.config)
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


def _load_frozen_model(
    path: str | Path, *, device: torch.device, gate_temperature: float = DEFAULT_GATE_TEMPERATURE_START
) -> tuple[UniversalCardPolicyV4, dict[str, Any]]:
    model = build_production_model_v4(device=device)
    payload = load_actor_critic_checkpoint(path, model, map_location=device)
    _set_temperatures(model, gate_temperature=gate_temperature)
    model.set_ppo_rollout_sampling(False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


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
        "safety/boundary_bootstrap_coverage": (
            float(metric["collection_boundary_bootstrap_count"]) / truncated if truncated else 1.0
        ),
        "league/history_snapshots": int(metric["league_history_snapshots"]),
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
) -> tuple[tuple[ClusterMatchAssignmentV4, ...], ...]:
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
    uniform_history_cycle = []
    candidates: list[ClusterMatchAssignmentV4] = []
    matchups = tuple(
        matchmaker.next_match(has_history=league.has_history, battle_level=battle_level)
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
        if matchup.policy_matchup == POLICY_HISTORY:
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
    gate_temperature: float,
) -> dict[str, UniversalCardPolicyV4]:
    required = {
        assignment.opponent.checkpoint_id: assignment.opponent
        for assignment in assignments
        if assignment.opponent is not None
    }
    for checkpoint_id, entry in required.items():
        if checkpoint_id in cache:
            continue
        model, payload = _load_frozen_model(entry.checkpoint_path, device=device, gate_temperature=gate_temperature)
        if str(payload["checkpoint_id"]) != checkpoint_id:
            raise ValueError("historical checkpoint file identity changed")
        cache[checkpoint_id] = model
    # Historical models not selected in this wave need not consume GPU memory.
    for checkpoint_id in tuple(cache):
        if checkpoint_id not in required:
            del cache[checkpoint_id]
    return {checkpoint_id: cache[checkpoint_id] for checkpoint_id in required}


def _validate_topology(ready: Mapping[str, object], *, world_size: int) -> tuple[int, int]:
    """Validate the one supported production topology and return its dimensions."""

    if ready.get("schema") != "v4-ppo-multinode-topology.v1":
        raise ValueError("unsupported PPO topology schema")
    expected_job = str(ready.get("trainer_slurm_job_id", ""))
    expected_host = str(ready.get("trainer_host", "")).split(".", 1)[0]
    engine_count = int(ready.get("engine_count", -1))
    lanes = int(ready.get("resident_slots_per_engine", -1))
    checks = {
        "slurm_job_id": bool(expected_job) and os.environ.get("SLURM_JOB_ID") == expected_job,
        "host": bool(expected_host) and socket.gethostname().split(".", 1)[0] == expected_host,
        "world_size": world_size == int(ready.get("gpus", -1)),
        "visible_gpu_count": torch.cuda.device_count() == world_size,
        "engine_count": engine_count > 0,
        "lanes": lanes == 8,
        "decision_ticks": int(ready.get("decision_ticks", -1)) == 5,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"production PPO topology failed checks: {failed}")
    return engine_count, lanes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-pool", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="fresh learner initialization checkpoint")
    parser.add_argument(
        "--il-anchor-checkpoint", type=Path, help="frozen IL opponent checkpoint; defaults to --checkpoint"
    )
    parser.add_argument(
        "--il-opening-force-after-seconds",
        type=int,
        help=(
            "force the IL anchor's first legal ACT after this battle time if "
            "it has not acted; never force the learner or fixed opponents"
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
    parser.add_argument("--rollout-segment-seconds", type=int, default=40)
    parser.add_argument("--snapshot-every", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9997)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--shaping-beta", type=float, default=0.05)
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
    parser.add_argument("--fireball-king-activation-penalty", type=float, default=0.0)
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
    parser.add_argument("--sequence-chunk-steps", type=int, default=48)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--expert-bc-manifest", type=Path)
    parser.add_argument("--expert-bc-coefficient", type=float, default=0.2)
    parser.add_argument("--expert-bc-records-per-rank", type=int, default=2)
    parser.add_argument("--expert-bc-window-steps", type=int, default=128)
    parser.add_argument("--expert-bc-hog-gate-weight", type=float, default=0.0)
    parser.add_argument("--expert-bc-hog-gate-scope", choices=("all", "proactive-first"), default="all")
    parser.add_argument("--expert-bc-hog-gate-gradient-scope", choices=("full", "gate-head-only"), default="full")
    parser.add_argument("--gate-temperature-start", type=float, default=DEFAULT_GATE_TEMPERATURE_START)
    parser.add_argument("--gate-temperature-end", type=float, default=DEFAULT_GATE_TEMPERATURE_END)
    parser.add_argument("--gate-temperature-anneal-updates", type=int, default=DEFAULT_GATE_TEMPERATURE_ANNEAL_UPDATES)
    parser.add_argument("--continue-temperature", type=float, default=DEFAULT_CONTINUE_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
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
    if args.il_opening_force_after_seconds is not None and args.il_opening_force_after_seconds <= 0:
        raise ValueError("IL opening timeout must be positive")
    opening_force_config = (
        {"enabled": False}
        if args.il_opening_force_after_seconds is None
        else {
            "enabled": True,
            "scope": "il-anchor-only",
            "after_seconds": args.il_opening_force_after_seconds,
            "condition": "IL has not acted since game start",
            "once_per_game": True,
            "force_learner": False,
            "force_fixed_opponents": False,
            "downstream_decode": "IL argmax over legal actions",
        }
    )
    il_anchor_checkpoint = args.il_anchor_checkpoint or args.checkpoint
    self_play_percent = int(args.self_play_percent)
    fixed_opponent_percent = int(args.fixed_opponent_percent)
    fixed_opponent_paths = tuple(args.fixed_opponent_checkpoint)
    if not 0 <= self_play_percent <= 100:
        raise ValueError("self-play percent must be in [0, 100]")
    if not 0 <= fixed_opponent_percent <= 100:
        raise ValueError("fixed opponent percent must be in [0, 100]")
    if fixed_opponent_percent and not fixed_opponent_paths:
        raise ValueError("fixed opponent percent requires at least one checkpoint")
    if fixed_opponent_paths and not fixed_opponent_percent:
        raise ValueError("fixed opponent checkpoints require a positive percent")
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
    il_opponent_percent = 100 - self_play_percent - fixed_opponent_percent
    if il_opponent_percent < 0:
        raise ValueError("self-play and fixed opponent percentages exceed 100")
    matchmaker_config = MatchmakerConfigV4(
        current_current=self_play_percent,
        current_il=il_opponent_percent,
        current_history=fixed_opponent_percent,
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
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
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
    engine_count, lanes = _validate_topology(ready, world_size=world_size)
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
    max_decision_steps = math.ceil(environment_config.max_battle_ticks / 5) + 4

    current_model = build_production_model_v4(device=device)
    # Actor batches and learner ACT batches have different row groupings.
    # BF16 location logits are unusually sensitive to those GEMM/conv shapes,
    # so keep this small head in FP32 while the rest of PPO remains autocast.
    current_model.set_ppo_location_head_fp32(True)
    optimizer = torch.optim.AdamW(current_model.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=0.0)
    if args.resume_checkpoint is None:
        current_payload = load_actor_critic_checkpoint(args.checkpoint, current_model, map_location=device)
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
        current_payload = load_actor_critic_checkpoint(
            args.resume_checkpoint, current_model, optimizer=optimizer, map_location=device
        )
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
    anchor_model, anchor_payload = _load_frozen_model(
        il_anchor_checkpoint, device=actor_device, gate_temperature=args.gate_temperature_start
    )
    anchor_state_id = str(anchor_payload["checkpoint_id"])
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
        )
    )
    bc_identities = [None for _ in range(world_size)]
    dist.all_gather_object(bc_identities, expert_bc_contract)
    if any(item != expert_bc_contract for item in bc_identities):
        raise RuntimeError("GPU ranks loaded different expert BC contracts")
    resume_contract = {
        "schema": "v4-ppo-strict-resume.v1",
        "anchor_checkpoint_id": anchor_state_id,
        "deck_pool": str(args.deck_pool.resolve()),
        "engine_count": engine_count,
        "resident_slots_per_engine": lanes,
        "world_size": world_size,
        "rollout_segment_seconds": args.rollout_segment_seconds,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "shaping_beta": args.shaping_beta,
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
        "self_play_percent": self_play_percent,
        "fixed_opponent_percent": fixed_opponent_percent,
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
    # Omit the optional key when disabled to preserve old no-BC strict resumes.
    if expert_bc_contract is not None:
        resume_contract["expert_bc"] = expert_bc_contract
    resume_extra = current_payload.get("extra")
    if args.resume_checkpoint is not None:
        if not isinstance(resume_extra, Mapping) or (resume_extra.get("strict_boundary") is not True):
            raise ValueError("resume checkpoint is not a strict PPO boundary")
        if resume_extra.get("resume_contract") != resume_contract:
            raise ValueError("resume checkpoint training contract changed")
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
    if rank == 0:
        league_rng = random.Random(args.seed ^ 0x4F31_AA09)
        if args.resume_checkpoint is None:
            league = PolicyLeagueV4(anchor_checkpoint_id=anchor_state_id, anchor_checkpoint_path=il_anchor_checkpoint)
            for fixed_path in resolved_fixed_paths:
                if fixed_path == args.checkpoint.resolve():
                    fixed_checkpoint_id = source_checkpoint_id
                    fixed_update_step = int(current_payload["update_step"])
                else:
                    fixed_checkpoint_id, fixed_update_step = _checkpoint_identity(fixed_path)
                league.add_snapshot(fixed_checkpoint_id, fixed_path, update_step=fixed_update_step)
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
                raise ValueError("resume checkpoint IL anchor changed")
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
            force_il_opponent_opening_after_seconds=(args.il_opening_force_after_seconds),
        )
        for engine_index in engine_indices
    )
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
            "capture_mode": "compact",
            "collector": "remote-segmented-cuda-graph",
            "expert_bc": expert_bc_contract,
            "wandb": {"mode": "offline", "project": WANDB_PROJECT},
        }
        (args.output_dir / "config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        wandb_run = _init_offline_wandb(args.output_dir, run_config)

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

    def prepare_collection(
        policy_state_id: str,
    ) -> tuple[tuple[ClusterMatchAssignmentV4, ...], dict[str, UniversalCardPolicyV4], str]:
        if rank == 0:
            assert matchmaker is not None and league is not None and league_rng is not None
            schedule = _balanced_assignments(
                matchmaker,
                league,
                league_rng,
                match_capacities=match_capacities,
                rich_capacities=rich_capacities,
                special_towers_require_rich=False,
                uniform_history_sampling=bool(resolved_fixed_paths),
                fixed_opponent_il_max_average_elixir=(fixed_opponent_il_max_average_elixir),
                deck_average_elixir_by_id=deck_average_elixir_by_id,
                learner_deck=learner_deck,
                learner_opponent_sampler=learner_opponent_sampler,
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
                local, history_cache, device=actor_device, gate_temperature=args.gate_temperature_start
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
                    fireball_king_activation_penalty=args.fireball_king_activation_penalty,
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
    resume_rng_pending = args.resume_checkpoint is not None

    def start_segment_session(policy_state_id: str) -> None:
        nonlocal segment_assignments, segment_history_models
        nonlocal segment_collections, segment_resumes
        nonlocal segment_executor, segment_future
        nonlocal resume_rng_pending
        if segment_executor is not None or segment_future is not None:
            raise RuntimeError("segmented collector session is already running")
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
    batch_collection_seconds = 0.0
    batch_end_to_end_seconds = 0.0
    batch_reports: list[Mapping[str, object]] = []
    batch_segments: list[Mapping[str, object]] = []
    run_failed = True
    try:
        start_segment_session(actor_state_id)
        for update_step in range(initial_update_step + 1, args.updates + 1):
            dist.barrier()
            cycle_started = time.perf_counter()
            if segment_assignments is None or segment_history_models is None:
                raise RuntimeError("segmented collector metadata is incomplete")
            collected = next_segment_collection()
            history_models = segment_history_models
            collected_policy_state_id = collected.rollout.policy_state_id
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
                collection_ok = (
                    all(
                        float(shard["timing"]["owner_frames"]) > 0  # type: ignore[index]
                        for shard in gathered
                    )
                    and truncated == 0
                    and rejected == 0
                )
                failure = (
                    None
                    if collection_ok
                    else {
                        "matches": len(reports),
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
                            league.record_result(str(checkpoint_id), float(report["result"][learner_owner]))
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
            behavior_counts = _rollout_behavior_counts(learner_rollout, current_model).to(device)
            dist.all_reduce(behavior_counts, op=dist.ReduceOp.SUM)
            behavior_metrics = _rollout_behavior_metrics(
                behavior_counts.cpu(), delay_offset_ms=current_model.config.delay_offset_ms
            )
            update_started = time.perf_counter()
            # Reports and timing were gathered above; the learner consumes every
            # valid lane and can release the collection wrapper before optimization.
            del collected
            gc.collect()
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
            )
            bc_metrics = {}
            if expert_bc_sampler is not None:
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
            update_seconds_local = time.perf_counter() - update_started
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
                    )
                    if not resolved_fixed_paths:
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
                            "strict_boundary": True,
                            "resume_contract": resume_contract,
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
                    "native_rejected_actions": sum(
                        sum(int(value) for value in report["rejected_actions"]) for report in reports
                    ),
                    "optimizer_seconds": float(update_seconds),
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
                        batch_collection_seconds = 0.0
                        batch_end_to_end_seconds = 0.0
                        batch_reports.clear()
                        batch_segments.clear()
                        start_segment_session(behavior_state_id)
        run_failed = False
    finally:
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
