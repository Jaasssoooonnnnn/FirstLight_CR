"""Shared resident-engine state and storage helpers for V4 PPO."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import time
from typing import Any, Mapping, Sequence, cast

import torch
from torch import Tensor

from ...battle_env import BattleEnvV1
from ...cr_native_env import NativeClashEnv, ResidentNativeClashEnv
from ...resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1
from .expert import POLICY_DECISION_TICKS
from .factory import build_episode_tensorizers_v4
from .league import LeagueEntryV4
from .matchmaking import POLICY_HISTORY, POLICY_IL, ScheduledMatchupV4
from .ppo import StoredRolloutV4
from .ppo_runtime import build_ppo_environment_v4, episode_for_matchup_v4
from .tensors import TensorRecordV4, tensor_padding_value


@dataclass(frozen=True, slots=True)
class ClusterMatchAssignmentV4:
    matchup: ScheduledMatchupV4
    opponent: LeagueEntryV4 | None = None

    def __post_init__(self) -> None:
        if self.matchup.policy_matchup == POLICY_HISTORY:
            if self.opponent is None:
                raise ValueError("historical matchup requires a league opponent")
        elif self.opponent is not None:
            raise ValueError("only historical matchups may bind a league opponent")


@dataclass(slots=True)
class ResidentEngineV4:
    engine_index: int
    port: int
    natives: tuple["RetryResidentNativeClashEnv", ...]
    coordinator: ResidentBatchCoordinatorV1
    environments: tuple[BattleEnvV1, ...]

    @property
    def lanes(self) -> int:
        return len(self.environments)

    def prepare(self) -> None:
        attestation = self.environments[0].prepare_runner_attestation()
        for environment in self.environments[1:]:
            environment.runner_attestation = attestation

    def close(self) -> None:
        self.coordinator.close_session()
        for native in self.natives:
            native.close_transport()


@dataclass(frozen=True, slots=True)
class ClusterCollectionV4:
    rollout: StoredRolloutV4
    reports: tuple[dict[str, object], ...]
    timing: dict[str, float]


@dataclass(slots=True)
class _MatchRuntimeV4:
    engine: ResidentEngineV4
    env_id: int
    environment: BattleEnvV1
    assignment: ClusterMatchAssignmentV4
    observations: dict[int, Any]
    tensorizers: tuple[Any, Any]
    active: bool = True
    decision_steps: int = 0
    started: float = 0.0
    active_actions: tuple[int, int] = (0, 0)
    first_nonwait_action_tick: tuple[int | None, int | None] = (None, None)
    first_successful_card_tick: tuple[int | None, int | None] = (None, None)
    full_elixir_deferred_decision_ticks_before_first_card: tuple[int, int] = (0, 0)
    rejected_actions: tuple[int, int] = (0, 0)
    rejection_details: list[dict[str, object]] = field(default_factory=list)
    seen_rejections: set[tuple[object, ...]] = field(default_factory=set)
    deck_average_elixir: tuple[float, float] = (0.0, 0.0)
    opening_forced_owner: int | None = None
    opening_forced_tick: int | None = None
    elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    elixir_overflow_penalty_total: tuple[float, float] = (0.0, 0.0)
    personal_elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    personal_elixir_overflow_cost_accrued: tuple[float, float] = (0.0, 0.0)
    unilateral_elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    unilateral_elixir_overflow_cost_accrued: tuple[float, float] = (0.0, 0.0)
    personal_elixir_overflow_penalty_total: tuple[float, float] = (0.0, 0.0)
    unilateral_elixir_overflow_penalty_total: tuple[float, float] = (0.0, 0.0)
    personal_elixir_overflow_grace_crossings: tuple[int, int] = (0, 0)
    cumulative_elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    opening_elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    cumulative_unilateral_elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    max_continuous_elixir_overflow_wasted: tuple[float, float] = (0.0, 0.0)
    first_elixir_overflow_tick: tuple[int | None, int | None] = (None, None)
    hog_deployments: tuple[int, int] = (0, 0)
    hog_deploy_reward_total: tuple[float, float] = (0.0, 0.0)
    first_hog_timing_reward_total: tuple[float, float] = (0.0, 0.0)
    first_hog_deferral_penalty_total: tuple[float, float] = (0.0, 0.0)
    first_hog_deadline_penalty_total: tuple[float, float] = (0.0, 0.0)
    first_hog_deploy_tick: tuple[int | None, int | None] = (None, None)
    first_hog_playable_tick: tuple[int | None, int | None] = (None, None)
    hog_playable_decision_ticks_before_first_hog: tuple[int, int] = (0, 0)
    hog_playable_deferred_ticks_before_first_hog: tuple[int, int] = (0, 0)
    opening_hog_playable_deferred_ticks: tuple[int, int] = (0, 0)
    seen_hog_deployments: set[tuple[int, str]] = field(default_factory=set)
    seen_king_activation_owners: set[int] = field(default_factory=set)
    sleeping_enemy_king_damage_count: tuple[int, int] = (0, 0)
    king_activation_not_attributed_to_fireball: tuple[int, int] = (0, 0)
    fireball_king_activations: tuple[int, int] = (0, 0)
    fireball_king_activation_penalty_total: tuple[float, float] = (0.0, 0.0)
    fireball_king_activation_details: list[dict[str, object]] = field(default_factory=list)


class RetryResidentNativeClashEnv(ResidentNativeClashEnv):
    """Resident routing that preserves the base transport retry keyword."""

    def _request(self, command: str, *, _read_retry_count: int = 2) -> dict[str, Any]:
        if self._resident_closed:
            raise RuntimeError("resident native environment is closed")
        if (
            command == "attest"
            or command.startswith("touch ")
            or command.startswith("render ")
            or command.startswith("env ")
        ):
            wire_command = command
        else:
            wire_command = f"env {self.env_id} {command}"
        return NativeClashEnv._request(self, wire_command, _read_retry_count=_read_retry_count)


def _padded_records(values: tuple[Any, ...], fill: int = 0) -> Any:
    """Concatenate dynamic V4 batches with their canonical masked padding."""

    if not values:
        raise ValueError("at least one V4 tensor record is required")
    first = values[0]
    if isinstance(first, Tensor):
        if not all(isinstance(item, Tensor) for item in values):
            raise TypeError("cannot concatenate mixed tensor fields")
        ranks = {item.ndim for item in values}
        if len(ranks) != 1:
            raise ValueError("cannot pad tensors with different ranks")
        target = tuple(max(int(item.shape[axis]) for item in values) for axis in range(1, first.ndim))
        padded = []
        for item in values:
            if tuple(item.shape[1:]) == target:
                padded.append(item)
                continue
            output = item.new_full((int(item.shape[0]), *target), fill)
            slices = (slice(None),) + tuple(slice(0, int(size)) for size in item.shape[1:])
            output[slices] = item
            padded.append(output)
        return torch.cat(tuple(padded), dim=0)
    if is_dataclass(first):
        if not all(type(item) is type(first) for item in values):
            raise TypeError("cannot concatenate different V4 record types")
        return type(first)(
            **{
                field.name: _padded_records(
                    tuple(getattr(value, field.name) for value in values), tensor_padding_value(first, field.name)
                )
                for field in fields(first)
            }
        )
    if not all(item == first for item in values[1:]):
        raise ValueError("non-tensor V4 record fields differ")
    return first


def concatenate_padded_tensor_records_v4(values: Sequence[TensorRecordV4]) -> TensorRecordV4:
    return cast(TensorRecordV4, _padded_records(tuple(values)))


def _stored_tensor(value: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    with torch.inference_mode(False):
        stored = value.detach().to(device="cpu", dtype=dtype)
        return stored.clone() if torch.is_inference(stored) else stored


def _scatter_storage_rows(
    previous: Any | None, updates: Any, indices: Tensor, *, total_rows: int, clone_previous: bool = True, fill: int = 0
) -> Any:
    """Scatter active policy rows into a fixed-width CPU storage record."""

    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("storage scatter indices must be long [N]")
    if previous is None:
        if indices.tolist() != list(range(total_rows)):
            raise RuntimeError("the first policy batch must contain every actor")
        return updates
    if isinstance(previous, Tensor):
        if not isinstance(updates, Tensor):
            raise TypeError("storage scatter tensor types differ")
        if previous.ndim != updates.ndim:
            raise ValueError("storage scatter tensor ranks differ")
        target = tuple(max(int(previous.shape[axis]), int(updates.shape[axis])) for axis in range(1, previous.ndim))

        def _pad_rows(value: Tensor, rows: int, *, clone: bool) -> Tensor:
            if tuple(value.shape[1:]) == target:
                return value.clone() if clone else value
            padded = value.new_full((rows, *target), fill)
            slices = (slice(None),) + tuple(slice(0, int(size)) for size in value.shape[1:])
            padded[slices] = value
            return padded

        result = _pad_rows(previous, int(previous.shape[0]), clone=clone_previous)
        padded_updates = _pad_rows(updates, int(updates.shape[0]), clone=False)
        result.index_copy_(0, indices.to(result.device), padded_updates.to(result.device))
        return result
    if is_dataclass(previous):
        if type(updates) is not type(previous):
            raise TypeError("storage scatter record types differ")
        return type(previous)(
            **{
                field.name: _scatter_storage_rows(
                    getattr(previous, field.name),
                    getattr(updates, field.name),
                    indices,
                    total_rows=total_rows,
                    clone_previous=clone_previous,
                    fill=tensor_padding_value(previous, field.name),
                )
                for field in fields(previous)
            }
        )
    if updates != previous:
        raise ValueError("storage scatter non-tensor fields differ")
    return previous


def build_resident_engine_v4(
    engine_index: int,
    *,
    environment_config: Any,
    host: str = "127.0.0.1",
    base_port: int = 28000,
    lanes: int = 8,
    timeout: float = 90.0,
    capture_mode: str = "compact",
) -> ResidentEngineV4:
    if engine_index < 0 or not 1 <= lanes <= 16:
        raise ValueError("resident engine index/lanes are invalid")
    if capture_mode not in {"compact", "compact-raw", "zlib-json", "json"}:
        raise ValueError("unsupported resident capture mode")
    port = int(base_port) + int(engine_index)
    natives = tuple(RetryResidentNativeClashEnv(env_id, host, port, timeout=timeout) for env_id in range(lanes))
    coordinator = ResidentBatchCoordinatorV1(
        natives,
        timeout=timeout,
        training_capture=capture_mode in {"compact", "compact-raw"},
        compressed_json=capture_mode in {"compact", "zlib-json"},
        validate_json_capture=False,
    )
    proxies = tuple(ResidentBatchNativeProxyV1(native, coordinator) for native in natives)
    environments = tuple(build_ppo_environment_v4(proxy, environment_config) for proxy in proxies)
    return ResidentEngineV4(
        engine_index=int(engine_index), port=port, natives=natives, coordinator=coordinator, environments=environments
    )


def _initial_elixir(observations: Mapping[int, Any]) -> dict[int, float]:
    return {
        owner: float(next(player for player in observations[owner].players if player.owner == owner).elixir_exact)
        for owner in (0, 1)
    }


def _reset_engine(
    engine: ResidentEngineV4, assignments: Sequence[ClusterMatchAssignmentV4], environment_config: Any
) -> tuple[_MatchRuntimeV4, ...]:
    if len(assignments) != engine.lanes:
        raise ValueError("one assignment is required for every resident lane")
    engine.coordinator.close_session()
    runtimes: list[_MatchRuntimeV4] = []
    for env_id, (environment, assignment) in enumerate(zip(engine.environments, assignments, strict=True)):
        episode = episode_for_matchup_v4(assignment.matchup, environment_config)
        observations, _info = environment.reset(
            episode,
            options={
                "render_mode": "headless",
                "decision_ticks": POLICY_DECISION_TICKS,
                "event_driven_decisions": False,
                "reuse_runner_attestation": True,
            },
        )
        if int(observations[0].tick) != environment.warmup_ticks:
            raise RuntimeError("PPO reset did not stop at the standard warmup tick")
        tensorizers = build_episode_tensorizers_v4(episode)
        elixir = _initial_elixir(observations)
        for owner, tensorizer in enumerate(tensorizers):
            tensorizer.start_episode(
                observations[owner], initial_elixir=elixir if tensorizer.tracker is not None else None
            )
        deck_average_elixir = tuple(
            sum(float(tensorizer.card_costs[card_id]) for card_id in tensorizer.deck) / len(tensorizer.deck)
            for tensorizer in tensorizers
        )
        runtimes.append(
            _MatchRuntimeV4(
                engine=engine,
                env_id=env_id,
                environment=environment,
                assignment=assignment,
                observations=observations,
                tensorizers=cast(tuple[Any, Any], tensorizers),
                started=time.perf_counter(),
                deck_average_elixir=cast(tuple[float, float], deck_average_elixir),
            )
        )
    return tuple(runtimes)


def _model_key(assignment: ClusterMatchAssignmentV4, owner: int) -> str:
    if owner in assignment.matchup.current_owners:
        return "current"
    if assignment.matchup.policy_matchup == POLICY_IL:
        return "anchor"
    if assignment.matchup.policy_matchup == POLICY_HISTORY:
        assert assignment.opponent is not None
        return f"history:{assignment.opponent.checkpoint_id}"
    raise RuntimeError("current-current matchup exposed a frozen owner")


def _terminal_report(runtime: _MatchRuntimeV4) -> dict[str, object]:
    observation = runtime.observations[0]
    terminal = observation.terminal
    crowns = tuple(int(next(p for p in observation.players if p.owner == owner).crowns) for owner in (0, 1))
    matchup = runtime.assignment.matchup
    return {
        "sequence": matchup.sequence,
        "category": matchup.category,
        "policy_matchup": matchup.policy_matchup,
        "deck0_id": matchup.deck0.deck_id,
        "deck1_id": matchup.deck1.deck_id,
        "battle_level": matchup.battle_level,
        "current_owners": matchup.current_owners,
        "opponent_checkpoint_id": (
            None if runtime.assignment.opponent is None else runtime.assignment.opponent.checkpoint_id
        ),
        "engine_index": runtime.engine.engine_index,
        "env_id": runtime.env_id,
        "decision_steps": runtime.decision_steps,
        "terminal_tick": int(observation.tick),
        "result": tuple(float(value) for value in terminal.result_by_owner),
        "crowns": crowns,
        "truncated": bool(observation.truncated),
        "active_actions": runtime.active_actions,
        "first_nonwait_action_tick": runtime.first_nonwait_action_tick,
        "first_successful_card_tick": runtime.first_successful_card_tick,
        "full_elixir_deferred_decision_ticks_before_first_card": (
            runtime.full_elixir_deferred_decision_ticks_before_first_card
        ),
        "hog_deployments": runtime.hog_deployments,
        "sleeping_enemy_king_damage_count": runtime.sleeping_enemy_king_damage_count,
        "king_activation_not_attributed_to_fireball": runtime.king_activation_not_attributed_to_fireball,
        "fireball_king_activations": runtime.fireball_king_activations,
        "fireball_king_activation_penalty_total": runtime.fireball_king_activation_penalty_total,
        "fireball_king_activation_details": runtime.fireball_king_activation_details,
        "hog_deploy_reward_total": runtime.hog_deploy_reward_total,
        "first_hog_timing_reward_total": runtime.first_hog_timing_reward_total,
        "first_hog_deferral_penalty_total": (runtime.first_hog_deferral_penalty_total),
        "first_hog_deadline_penalty_total": (runtime.first_hog_deadline_penalty_total),
        "first_hog_deploy_tick": runtime.first_hog_deploy_tick,
        "first_hog_playable_tick": runtime.first_hog_playable_tick,
        "hog_playable_decision_ticks_before_first_hog": (runtime.hog_playable_decision_ticks_before_first_hog),
        "hog_playable_deferred_ticks_before_first_hog": (runtime.hog_playable_deferred_ticks_before_first_hog),
        "opening_hog_playable_deferred_ticks": (runtime.opening_hog_playable_deferred_ticks),
        "deck_average_elixir": runtime.deck_average_elixir,
        "opening_forced_owner": runtime.opening_forced_owner,
        "opening_forced_tick": runtime.opening_forced_tick,
        "elixir_overflow_wasted": runtime.elixir_overflow_wasted,
        "elixir_overflow_penalty_total": (runtime.elixir_overflow_penalty_total),
        "personal_elixir_overflow_wasted": (runtime.personal_elixir_overflow_wasted),
        "unilateral_elixir_overflow_wasted": (runtime.unilateral_elixir_overflow_wasted),
        "personal_elixir_overflow_penalty_total": (runtime.personal_elixir_overflow_penalty_total),
        "unilateral_elixir_overflow_penalty_total": (runtime.unilateral_elixir_overflow_penalty_total),
        "personal_elixir_overflow_grace_crossings": (runtime.personal_elixir_overflow_grace_crossings),
        "cumulative_elixir_overflow_wasted": (runtime.cumulative_elixir_overflow_wasted),
        "opening_elixir_overflow_wasted": runtime.opening_elixir_overflow_wasted,
        "cumulative_unilateral_elixir_overflow_wasted": (runtime.cumulative_unilateral_elixir_overflow_wasted),
        "max_continuous_elixir_overflow_wasted": (runtime.max_continuous_elixir_overflow_wasted),
        "first_elixir_overflow_tick": runtime.first_elixir_overflow_tick,
        "rejected_actions": runtime.rejected_actions,
        "rejection_details": tuple(runtime.rejection_details),
        "elapsed_seconds": time.perf_counter() - runtime.started,
    }
