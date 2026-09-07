"""Online battle-engine producer for chronological V4 imitation sequences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ...battle_env import BattleEnvV1
from ...contracts import ActionV1
from ..replay_viewer import _direct_render_commands, _schedule_direct_render_commands
from .expert import (
    FIRST_POLICY_DECISION_TICK,
    POLICY_DECISION_TICKS,
    ExpertActionAlignmentError,
    build_expert_action_batch,
    group_expert_action_windows,
    replay_expert_actions,
    require_accepted_timeline,
    screen_expert_timeline,
)
from .imitation import ILSequenceV4
from .learning import discounted_returns, trusted_decision_mask
from .tensorizer import UniversalObservationTensorizerV4

if TYPE_CHECKING:
    from ...resident_batch_vector import ResidentBatchCoordinatorV1
    from ...royaleapi_replay import PreparedCollectedReplay


@dataclass(frozen=True, slots=True)
class ReplayProductionResultV4:
    replay_tag: str
    sequences: tuple[ILSequenceV4, ILSequenceV4] | None = field(repr=False)
    completed: bool
    failure_reason: str | None
    first_untrusted_tick: int | None
    simulated_end_tick: int
    source_end_tick: int
    expert_action_count: int
    executed_expert_action_count: int
    winner_matches: bool | None
    crowns_match: bool | None
    decision_frame_count: int = 0

    @property
    def retained_frame_count(self) -> int:
        if self.sequences is None:
            return 0
        return int(self.sequences[0].valid_mask.sum().item())


@dataclass(slots=True)
class _ILReplayLaneV4:
    environment: BattleEnvV1
    prepared: PreparedCollectedReplay
    tensorizers: tuple[UniversalObservationTensorizerV4, UniversalObservationTensorizerV4]
    expert_actions: tuple[Any, ...]
    windows: Mapping[tuple[int, int], Sequence[Any]]
    source_end_tick: int
    observations_by_owner: list[list[Any]] = field(default_factory=lambda: [[], []])
    actions_by_owner: list[list[Any]] = field(default_factory=lambda: [[], []])
    rewards_by_owner: list[list[float]] = field(default_factory=lambda: [[], []])
    terminated_rows: list[bool] = field(default_factory=list)
    truncated_rows: list[bool] = field(default_factory=list)
    decision_ticks: list[int] = field(default_factory=list)
    observations: Mapping[int, Any] | None = None
    current_tick: int = 0
    executed_expert_action_count: int = 0
    completed: bool = False
    active: bool = True
    failure_reason: str | None = None
    first_untrusted_tick: int | None = None
    winner_matches: bool | None = None
    crowns_match: bool | None = None
    pending_frame_batches: tuple[Any, Any] | None = None
    pending_frame_sequences: tuple[Any, Any] | None = None
    pending_gate_loss: tuple[bool, bool] | None = None
    gate_loss_rows: list[list[bool]] = field(default_factory=lambda: [[], []])
    replay_commands: tuple[Any, ...] = ()
    replay_schedule_sequences: tuple[int, ...] = ()
    replay_schedule_index: int = 0
    replay_schedule_reconciled_index: int = 0
    terminal_observed: bool = False

    @property
    def env_id(self) -> int:
        return int(self.environment.native.env_id)


def _wait_group(ticks: int) -> dict[int, tuple[ActionV1, ...]]:
    return {owner: (ActionV1.wait(owner, ticks=ticks),) for owner in (0, 1)}


def _build_il_lane(
    environment: BattleEnvV1, prepared: PreparedCollectedReplay, tensorizers: Sequence[UniversalObservationTensorizerV4]
) -> _ILReplayLaneV4:
    if len(tensorizers) != 2 or tuple(item.actor_owner for item in tensorizers) != (0, 1):
        raise ValueError("online IL needs owner-0 and owner-1 tensorizers")
    if any(item.config.decision_ticks != POLICY_DECISION_TICKS for item in tensorizers):
        raise ValueError("online IL tensorizers must use five-tick decisions")
    expert_actions = tuple(replay_expert_actions(prepared.replay))
    timeline_report = screen_expert_timeline(expert_actions)
    require_accepted_timeline(timeline_report, replay_id=prepared.replay_tag)
    replay_commands = _direct_render_commands(prepared.replay)
    if len(replay_commands) != len(expert_actions):
        raise RuntimeError("renderer and expert command counts differ")
    return _ILReplayLaneV4(
        environment=environment,
        prepared=prepared,
        tensorizers=(tensorizers[0], tensorizers[1]),
        expert_actions=expert_actions,
        windows=group_expert_action_windows(expert_actions),
        source_end_tick=int(prepared.replay.end_native_tick),
        replay_commands=replay_commands,
    )


def _initialize_il_lane(lane: _ILReplayLaneV4) -> None:
    execution_environment = replace(
        lane.prepared.replay.episode_config.environment, max_battle_ticks=lane.source_end_tick
    )
    if lane.environment.environment_config != execution_environment:
        raise RuntimeError(
            "training BattleEnv must match the collected replay end tick; use build_collected_replay_environment_v4"
        )
    execution_episode = replace(
        lane.prepared.replay.episode_config, environment=execution_environment, ruleset_id=lane.environment.ruleset_id
    )
    reset_options: dict[str, Any] = {
        "render_mode": "headless",
        "defer_headless_warmup": True,
        "headless_initial_step_ticks": 1,
        "allow_initial_remaining_runtime_rejections": True,
        "allow_verified_standard_layout_alias": True,
        "decision_ticks": POLICY_DECISION_TICKS,
        "event_driven_decisions": False,
    }
    observations, _info = lane.environment.reset(execution_episode, options=reset_options)
    lane.current_tick = int(observations[0].tick)
    if lane.current_tick < FIRST_POLICY_DECISION_TICK:
        observations, _reward, terminated, truncated, _info = lane.environment.step(
            _wait_group(FIRST_POLICY_DECISION_TICK - lane.current_tick), record_trace=False
        )
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("battle ended during the action-free opening warmup")
        lane.current_tick = int(observations[0].tick)
    if lane.current_tick != FIRST_POLICY_DECISION_TICK:
        raise RuntimeError(f"first IL decision expected tick {FIRST_POLICY_DECISION_TICK}, got {lane.current_tick}")
    if lane.replay_commands:
        native = lane.environment.native
        for command in lane.replay_commands:
            if command.target_tick <= lane.current_tick:
                raise RuntimeError("renderer command is not in the future after opening warmup")
        scheduled = _schedule_direct_render_commands(native, lane.replay_commands)
        lane.replay_schedule_sequences = tuple(scheduled[index] for index in range(len(lane.replay_commands)))
    initial_elixir = {
        owner: float(next(player for player in observations[owner].players if player.owner == owner).elixir_exact)
        for owner in (0, 1)
    }
    for owner, tensorizer in enumerate(lane.tensorizers):
        tensorizer.start_episode(
            observations[owner], initial_elixir=(initial_elixir if tensorizer.tracker is not None else None)
        )
    lane.observations = observations


def _prepare_il_lane(lane: _ILReplayLaneV4, *, validate_tensors: bool) -> dict[int, tuple[ActionV1, ...]]:
    if lane.observations is None:
        raise RuntimeError("IL lane has not been initialized")
    frame_batches = tuple(
        lane.tensorizers[owner].tensorize(lane.observations[owner], validate=validate_tensors) for owner in (0, 1)
    )
    frame_sequences: list[Any] = []
    gate_loss: list[bool] = []
    for owner in (0, 1):
        window = lane.windows.get((lane.current_tick, owner), ())
        try:
            sequence = build_expert_action_batch(
                frame_batches[owner],
                (window,),
                decision_ticks_by_row=(lane.current_tick,),
                perspectives=(lane.tensorizers[owner].perspective,),
                config=lane.tensorizers[owner].config,
                validate=validate_tensors,
            )
            label_is_usable = True
        except ExpertActionAlignmentError:
            if not window:
                raise
            sequence = build_expert_action_batch(
                frame_batches[owner],
                ((),),
                decision_ticks_by_row=(lane.current_tick,),
                perspectives=(lane.tensorizers[owner].perspective,),
                config=lane.tensorizers[owner].config,
                validate=validate_tensors,
            )
            label_is_usable = False
        frame_sequences.append(sequence)
        gate_loss.append(label_is_usable)
    for owner in (0, 1):
        lane.tensorizers[owner].record_action(frame_sequences[owner], frame_batches[owner], row=0, validate=False)
    lane.pending_frame_batches = (frame_batches[0], frame_batches[1])
    lane.pending_frame_sequences = (frame_sequences[0], frame_sequences[1])
    lane.pending_gate_loss = (gate_loss[0], gate_loss[1])
    return _wait_group(POLICY_DECISION_TICKS)


def _consume_renderer_schedule(lane: _ILReplayLaneV4, *, through_tick: int) -> None:
    """Advance the due-command cursor without leaving the resident data plane.

    The resident compact batch connection owns the probe's one control-server
    thread for the lifetime of a producer session.  Opening a second socket for
    a schedule receipt here would wait until that session closes.  Receipt
    validation is therefore deferred to ``_reconcile_renderer_schedule`` after
    the coordinator has closed the batch connection.
    """

    while lane.replay_schedule_index < len(lane.replay_commands):
        command = lane.replay_commands[lane.replay_schedule_index]
        if command.target_tick > through_tick:
            break
        lane.replay_schedule_index += 1


def _reconcile_renderer_schedule(lane: _ILReplayLaneV4) -> None:
    """Validate due native receipts and backfill per-row training masks."""

    while lane.replay_schedule_reconciled_index < lane.replay_schedule_index:
        index = lane.replay_schedule_reconciled_index
        command = lane.replay_commands[index]
        sequence = lane.replay_schedule_sequences[index]
        receipt = lane.environment.native.replay_schedule_status(sequence)
        state = str(receipt.get("state", ""))
        queued_at = int(receipt.get("queuedAtTick", -1))
        if state not in {"succeeded", "failed"}:
            raise RuntimeError(
                f"native replay {command.kind} has unresolved schedule state "
                f"at tick {command.target_tick}: {state or 'unknown'}"
            )
        if queued_at != command.target_tick - 1:
            raise RuntimeError(
                f"native replay {command.kind} resolved at tick {queued_at}, expected {command.target_tick - 1}"
            )
        if state == "succeeded":
            lane.executed_expert_action_count += 1
        else:
            # User-facing replay rendering keeps advancing after an exact
            # boundary-time semantic rejection.  Headless uses that same
            # scheduler contract; do not train a rejected source label.
            row = next(
                (
                    row_index
                    for row_index, decision_tick in enumerate(lane.decision_ticks)
                    if int(decision_tick) < int(command.target_tick) <= int(decision_tick) + POLICY_DECISION_TICKS
                ),
                None,
            )
            if row is None:
                raise RuntimeError(
                    f"native replay rejection has no matching IL decision row: target_tick={command.target_tick}"
                )
            lane.gate_loss_rows[int(command.owner)][row] = False
        lane.replay_schedule_reconciled_index += 1


def _terminal_fidelity(lane: _ILReplayLaneV4) -> None:
    assert lane.observations is not None
    expected_winner = lane.prepared.replay.terminal.get("winner")
    actual_winner = lane.observations[0].terminal.winner
    if expected_winner in (0, 1) or expected_winner is None:
        lane.winner_matches = actual_winner == expected_winner
    expected_crowns = {
        int(player["owner"]): int(player["crowns"])
        for player in lane.prepared.replay.terminal.get("players", ())
        if isinstance(player, Mapping) and player.get("owner") in (0, 1) and type(player.get("crowns")) is int
    }
    if set(expected_crowns) == {0, 1}:
        actual_crowns = {player.owner: player.crowns for player in lane.observations[0].players}
        lane.crowns_match = actual_crowns == expected_crowns
    fidelity_failures = []
    if lane.winner_matches is False:
        fidelity_failures.append("winner")
    if lane.crowns_match is False:
        fidelity_failures.append("crowns")
    if fidelity_failures:
        lane.completed = False
        lane.failure_reason = "terminal fidelity mismatch: " + ", ".join(fidelity_failures)
        lane.first_untrusted_tick = lane.current_tick


def _accept_il_step(
    lane: _ILReplayLaneV4, step_result: tuple[Any, Any, Any, Any, Any], *, max_frames: int | None
) -> None:
    next_observations, reward, terminated, truncated, _info = step_result
    next_tick = int(next_observations[0].tick)
    if next_tick <= lane.current_tick:
        raise RuntimeError("battle engine did not advance the IL timeline")
    _consume_renderer_schedule(lane, through_tick=next_tick)
    if lane.pending_frame_batches is None or lane.pending_frame_sequences is None or lane.pending_gate_loss is None:
        raise RuntimeError("IL lane has no pending teacher-forced frame")
    for owner in (0, 1):
        frame_batch = lane.pending_frame_batches[owner]
        lane.observations_by_owner[owner].append(frame_batch.to_storage("cpu", float_dtype=torch.float16))
        lane.actions_by_owner[owner].append(lane.pending_frame_sequences[owner])
        lane.rewards_by_owner[owner].append(float(reward[owner]))
        lane.gate_loss_rows[owner].append(lane.pending_gate_loss[owner])
    terminated_flag = bool(terminated[0])
    truncated_flag = bool(truncated[0])
    if any(bool(terminated[owner]) != terminated_flag for owner in (0, 1)):
        raise RuntimeError("battle owners disagree on terminal state")
    if any(bool(truncated[owner]) != truncated_flag for owner in (0, 1)):
        raise RuntimeError("battle owners disagree on truncation state")
    source_timeline_reached = not terminated_flag and next_tick >= lane.source_end_tick
    lane.terminated_rows.append(terminated_flag)
    lane.truncated_rows.append(truncated_flag and not source_timeline_reached)
    lane.decision_ticks.append(lane.current_tick)
    lane.observations = next_observations
    lane.current_tick = next_tick
    lane.pending_frame_batches = None
    lane.pending_frame_sequences = None
    lane.pending_gate_loss = None
    if terminated_flag:
        lane.active = False
        lane.terminal_observed = True
        lane.completed = True
        _terminal_fidelity(lane)
        if lane.replay_schedule_index < len(lane.replay_commands):
            lane.completed = False
            lane.failure_reason = "battle ended before all expert actions executed"
            lane.first_untrusted_tick = lane.current_tick
            return
        return
    if source_timeline_reached:
        lane.active = False
        if lane.replay_schedule_index < len(lane.replay_commands):
            lane.failure_reason = "source timeline ended before all expert actions"
            lane.first_untrusted_tick = lane.source_end_tick
        else:
            lane.completed = True
        return
    if truncated_flag:
        lane.active = False
        lane.failure_reason = "battle truncated before source timeline"
        lane.first_untrusted_tick = lane.current_tick
        return
    if max_frames is not None and len(lane.decision_ticks) >= max_frames:
        lane.active = False
        lane.failure_reason = "producer frame limit reached"
        lane.first_untrusted_tick = lane.current_tick
        return


def _fail_il_lane(lane: _ILReplayLaneV4, error: BaseException) -> None:
    lane.active = False
    lane.completed = False
    lane.failure_reason = f"{type(error).__name__}: {error}"
    lane.first_untrusted_tick = lane.current_tick


def _finalize_il_lane(lane: _ILReplayLaneV4, *, gamma_per_decision: float) -> ReplayProductionResultV4:
    time_steps = len(lane.decision_ticks)
    if time_steps == 0:
        return ReplayProductionResultV4(
            replay_tag=lane.prepared.replay_tag,
            sequences=None,
            completed=False,
            failure_reason=lane.failure_reason or "no IL frames were produced",
            first_untrusted_tick=lane.first_untrusted_tick,
            simulated_end_tick=lane.current_tick,
            source_end_tick=lane.source_end_tick,
            expert_action_count=len(lane.expert_actions),
            executed_expert_action_count=lane.executed_expert_action_count,
            winner_matches=lane.winner_matches,
            crowns_match=lane.crowns_match,
            decision_frame_count=0,
        )

    decision_tensor = torch.tensor(lane.decision_ticks, dtype=torch.long)[:, None]
    if lane.completed:
        valid_mask = torch.ones(time_steps, 1, dtype=torch.bool)
    else:
        if lane.first_untrusted_tick is None:
            raise RuntimeError("failed IL lane is missing its first untrusted tick")
        valid_mask = trusted_decision_mask(decision_tensor, torch.tensor([lane.first_untrusted_tick], dtype=torch.long))
    terminal = torch.tensor(lane.terminated_rows, dtype=torch.bool)[:, None]
    truncation = torch.tensor(lane.truncated_rows, dtype=torch.bool)[:, None]
    sequences: list[ILSequenceV4] = []
    completed = lane.completed
    failure_reason = lane.failure_reason
    for owner in (0, 1):
        rewards = torch.tensor(lane.rewards_by_owner[owner], dtype=torch.float32)[:, None]
        truncation_bootstrap = torch.zeros_like(rewards)
        if torch.any(truncation):
            completed = False
            failure_reason = "time-limit truncation lacks a value bootstrap"
            value_mask = torch.zeros_like(valid_mask)
        else:
            value_mask = valid_mask.clone() if completed and lane.terminal_observed else torch.zeros_like(valid_mask)
        returns = discounted_returns(
            rewards, terminal, truncation, gamma=gamma_per_decision, truncation_bootstrap_value=truncation_bootstrap
        )
        sequences.append(
            ILSequenceV4(
                observations=tuple(lane.observations_by_owner[owner]),
                actions=tuple(lane.actions_by_owner[owner]),
                episode_start=torch.tensor([[step == 0] for step in range(time_steps)], dtype=torch.bool),
                valid_mask=valid_mask.clone(),
                returns=returns,
                gate_loss_mask=(torch.tensor(lane.gate_loss_rows[owner], dtype=torch.bool)[:, None] & valid_mask),
                value_loss_mask=value_mask,
                sequence_id=f"{lane.prepared.replay_tag}:owner-{owner}",
                natural_time=True,
            )
        )
    return ReplayProductionResultV4(
        replay_tag=lane.prepared.replay_tag,
        sequences=(sequences[0], sequences[1]),
        completed=completed,
        failure_reason=failure_reason,
        first_untrusted_tick=lane.first_untrusted_tick,
        simulated_end_tick=lane.current_tick,
        source_end_tick=lane.source_end_tick,
        expert_action_count=len(lane.expert_actions),
        executed_expert_action_count=lane.executed_expert_action_count,
        winner_matches=lane.winner_matches,
        crowns_match=lane.crowns_match,
        decision_frame_count=time_steps,
    )


def produce_il_replay_batch(
    environments: Sequence[BattleEnvV1],
    prepared_replays: Sequence[PreparedCollectedReplay],
    tensorizers_by_lane: Sequence[Sequence[UniversalObservationTensorizerV4]],
    *,
    gamma_per_decision: float,
    batch_coordinator: ResidentBatchCoordinatorV1,
    replacement_prepared_replays: Sequence[PreparedCollectedReplay] = (),
    replacement_tensorizers_by_replay: Sequence[Sequence[UniversalObservationTensorizerV4]] = (),
    max_frames: int | None = None,
    validate_tensors: bool = False,
) -> tuple[ReplayProductionResultV4, ...]:
    """Replay independent matches on configurable slots of one native engine.

    Tensorizers, trackers, previous actions, and chronological buffers remain
    lane-local.  Only the equal five-tick native advance is batched.  Ended or
    failed lanes leave the next batch round without disturbing surviving lanes.
    """

    lane_count = len(environments)
    if lane_count == 0:
        raise ValueError("IL replay batch must contain at least one lane")
    if len(prepared_replays) != lane_count or len(tensorizers_by_lane) != lane_count:
        raise ValueError("IL replay batch inputs must have equal lengths")
    replacement_count = len(replacement_prepared_replays)
    if len(replacement_tensorizers_by_replay) != replacement_count:
        raise ValueError("replacement replays and tensorizers must be aligned")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")
    env_ids = tuple(int(environment.native.env_id) for environment in environments)
    if len(set(env_ids)) != lane_count:
        raise ValueError("IL replay batch requires unique resident env IDs")
    if any(getattr(environment.native, "coordinator", None) is not batch_coordinator for environment in environments):
        raise ValueError("IL batch coordinator must own every BattleEnv native proxy")

    lane_slots: list[_ILReplayLaneV4 | None] = [
        _build_il_lane(environments[index], prepared_replays[index], tensorizers_by_lane[index])
        for index in range(lane_count)
    ]
    replacements = tuple(zip(replacement_prepared_replays, replacement_tensorizers_by_replay, strict=True))
    next_replacement = 0
    schedule_index_by_lane = {id(lane): index for index, lane in enumerate(lane_slots) if lane is not None}
    finalized: list[tuple[int, ReplayProductionResultV4]] = []
    finalized_schedule_indices: set[int] = set()
    session_started = False
    executor: ThreadPoolExecutor | None = None

    def step_lane(lane: _ILReplayLaneV4, actions: dict[int, tuple[ActionV1, ...]]) -> tuple[Any, Any, Any, Any, Any]:
        try:
            return lane.environment.step(actions, record_trace=False)
        except BaseException as error:
            batch_coordinator.abort_round(error)
            raise

    def finalize_lane(lane: _ILReplayLaneV4) -> None:
        lane_id = id(lane)
        schedule_index = schedule_index_by_lane[lane_id]
        if schedule_index in finalized_schedule_indices:
            raise RuntimeError("IL replay lane was finalized twice")
        try:
            try:
                _reconcile_renderer_schedule(lane)
            except Exception as error:
                _fail_il_lane(lane, error)
            result = _finalize_il_lane(lane, gamma_per_decision=gamma_per_decision)
        finally:
            for tensorizer in lane.tensorizers:
                tensorizer.end_episode()
            finalized_schedule_indices.add(schedule_index)
        finalized.append((schedule_index, result))

    def refill_inactive_lanes() -> None:
        nonlocal next_replacement, session_started

        while True:
            inactive_indices = [index for index, lane in enumerate(lane_slots) if lane is not None and not lane.active]
            if not inactive_indices:
                break
            # Schedule receipts use the ordinary control protocol.  Close the
            # long-lived compact batch connection before finalizing any lane so
            # those receipt queries cannot block behind the batch session.
            if session_started:
                batch_coordinator.close_session()
                session_started = False
            for index in inactive_indices:
                previous = lane_slots[index]
                assert previous is not None
                native = previous.environment.native
                finalize_lane(previous)
                if next_replacement >= replacement_count:
                    lane_slots[index] = None
                    continue
                prepared, tensorizers = replacements[next_replacement]
                schedule_index = lane_count + next_replacement
                next_replacement += 1
                from .factory import build_collected_replay_environment_v4

                environment = build_collected_replay_environment_v4(prepared, native=native)
                replacement = _build_il_lane(environment, prepared, tensorizers)
                schedule_index_by_lane[id(replacement)] = schedule_index
                try:
                    _initialize_il_lane(replacement)
                except Exception as error:
                    _fail_il_lane(replacement, error)
                lane_slots[index] = replacement
        if not session_started and any(lane is not None and lane.active for lane in lane_slots):
            batch_coordinator.start_session()
            session_started = True

    try:
        for lane in lane_slots:
            assert lane is not None
            try:
                _initialize_il_lane(lane)
            except Exception as error:
                _fail_il_lane(lane, error)

        executor = ThreadPoolExecutor(max_workers=lane_count, thread_name_prefix="v4-il-slot")

        while True:
            refill_inactive_lanes()
            active_lanes = tuple(lane for lane in lane_slots if lane is not None and lane.active)
            if not active_lanes:
                break
            # Advance the native batch while the CPU tensorizes the preceding
            # captured frame. Each lane remains private until its future settles.
            round_lanes = active_lanes
            batch_coordinator.begin_round(tuple(lane.env_id for lane in round_lanes))
            round_actions = {lane.env_id: _wait_group(POLICY_DECISION_TICKS) for lane in round_lanes}
            futures = {executor.submit(step_lane, lane, round_actions[lane.env_id]): lane for lane in round_lanes}
            results: dict[int, tuple[Any, Any, Any, Any, Any]] = {}
            lane_errors: dict[int, BaseException] = {}
            for lane in round_lanes:
                try:
                    prepared_actions = _prepare_il_lane(lane, validate_tensors=validate_tensors)
                    if prepared_actions != round_actions[lane.env_id]:
                        raise RuntimeError("IL observation pass must submit WAIT only")
                except Exception as error:
                    _fail_il_lane(lane, error)
            for future, lane in futures.items():
                try:
                    results[lane.env_id] = future.result()
                except BaseException as error:
                    lane_errors[lane.env_id] = error
            native_round_succeeded = batch_coordinator.round_succeeded
            finish_error: BaseException | None = None
            try:
                batch_coordinator.finish_round()
            except BaseException as error:
                finish_error = error

            if not native_round_succeeded or finish_error is not None:
                round_error = finish_error or next(iter(lane_errors.values()), None)
                if round_error is None:
                    round_error = RuntimeError("resident batch round did not complete")
                if not isinstance(round_error, Exception):
                    raise round_error
                for lane in round_lanes:
                    _fail_il_lane(lane, round_error)
                continue

            for lane in round_lanes:
                if not lane.active:
                    continue
                lane_error = lane_errors.get(lane.env_id)
                if lane_error is not None:
                    if not isinstance(lane_error, Exception):
                        raise lane_error
                    _fail_il_lane(lane, lane_error)
                    continue
                try:
                    _accept_il_step(lane, results[lane.env_id], max_frames=max_frames)
                except Exception as error:
                    _fail_il_lane(lane, error)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if session_started:
            batch_coordinator.close_session()
        for lane in lane_slots:
            if lane is None:
                continue
            schedule_index = schedule_index_by_lane[id(lane)]
            if schedule_index in finalized_schedule_indices:
                continue
            for tensorizer in lane.tensorizers:
                tensorizer.end_episode()

    expected_results = lane_count + next_replacement
    if len(finalized) != expected_results:
        raise RuntimeError(f"IL replay stream finalized {len(finalized)} of {expected_results} lanes")
    return tuple(result for _index, result in sorted(finalized))
