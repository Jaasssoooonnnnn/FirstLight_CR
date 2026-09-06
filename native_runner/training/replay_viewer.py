"""Browse retained training matches and replay them in the stock renderer."""

from __future__ import annotations

from ..local_config import setting

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ..arena import cell_to_world
from ..battle_env import BattleEnvV1
from ..contracts import ActionKind, ActionV1
from ..cr_native_env import (
    AbilityAction,
    COMMAND_CONSUMPTION_STEPS,
    DeployAction,
    HandAction,
    NativeClashEnv,
    ResidentNativeClashEnv,
    RunnerError,
)
from ..match_factory import MatchConfig, NATIVE_FINALIZATION_DEADLINE_TICK, PRINCESS_TOWER_TROOP_ID
from ..resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1
from .replay_archive import TrainingReplayError, TrainingReplayV1


DEFAULT_REPLAY_DIRECTORY = Path("training_replays/pekka_bridge_spam_v1")
NATIVE_RENDER_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
LOGGER = logging.getLogger(__name__)
# Start ability resolution early enough to retry transient not-ready states.
# The probe resolves the one Ready controller matching the retained card hint
# and receives the source tick as an absolute tick, so host observation latency
# cannot move the command.
ABILITY_QUEUE_LEAD_TICKS = 60
HAND_QUEUE_LEAD_TICKS = 30
REPLAY_REWIND_TICKS = 5 * 20
REPLAY_CHECKPOINT_INTERVAL_TICKS = 20
REPLAY_CHECKPOINT_LIMIT = 48


@dataclass(frozen=True, slots=True)
class ReplayProgress:
    operation_index: int
    operation_count: int
    native_tick: int
    expected_tick: int
    tick_drift: int
    phase: str


@dataclass(frozen=True, slots=True)
class ReplayResult:
    stopped: bool
    final_native_tick: int
    expected_winner: int | None
    actual_winner: int | None
    max_absolute_tick_drift: int
    completion: str = "native-terminal"
    queued_action_count: int = 0
    action_count: int = 0
    queued_ability_count: int = 0
    ability_count: int = 0
    source_end_tick: int | None = None


class ReplayRewindControl:
    """Thread-safe rewind requests shared by the UI and playback worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_ticks = 0
        self._active = False
        self._message = "回退快照尚未就绪"

    def request_rewind(self, ticks: int = REPLAY_REWIND_TICKS) -> bool:
        if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 1:
            raise ValueError("rewind ticks must be a positive integer")
        with self._lock:
            if not self._active:
                return False
            self._pending_ticks += ticks
            self._message = f"已请求后退 {ticks / 20.0:g} 秒"
            return True

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    def _activate(self, *, current_tick: int, earliest_tick: int) -> None:
        with self._lock:
            self._active = True
            self._message = f"可回退范围：tick {earliest_tick}–{current_tick}"

    def _take_request(self) -> int:
        with self._lock:
            ticks = self._pending_ticks
            self._pending_ticks = 0
            return ticks

    def _report_restore(self, *, restored_tick: int, target_tick: int, earliest_tick: int) -> None:
        with self._lock:
            self._message = f"已后退至 tick {restored_tick}；目标 tick {target_tick}，最早可用 tick {earliest_tick}"

    def _deactivate(self, message: str) -> None:
        with self._lock:
            self._active = False
            self._pending_ticks = 0
            self._message = message


@dataclass(frozen=True, slots=True)
class _DirectReplayCursor:
    command_index: int
    processed: int
    queued_count: int
    queued_ability_count: int
    max_drift: int


@dataclass(frozen=True, slots=True)
class _NativeReplayCheckpoint:
    tick: int
    handle: Mapping[str, Any]
    cursor: _DirectReplayCursor


@dataclass(frozen=True, slots=True)
class _NativeReplayRestore:
    tick: int
    target_tick: int
    cursor: _DirectReplayCursor
    limited: bool


class _NativeReplayHistory:
    """Rolling native snapshots plus the matching Python command cursor."""

    def __init__(self, native: NativeClashEnv, control: ReplayRewindControl | None) -> None:
        self.native = native
        self.control = control
        self.checkpoints: list[_NativeReplayCheckpoint] = []
        self.disabled = control is None
        self.capture_failures = 0
        self.retry_after_tick = 0

    def capture(self, status: Mapping[str, Any], cursor: _DirectReplayCursor, *, force: bool = False) -> None:
        if self.disabled:
            return
        current_tick = int(status["tick"])
        if current_tick < self.retry_after_tick:
            return
        if (
            not force
            and self.checkpoints
            and current_tick < self.checkpoints[-1].tick + REPLAY_CHECKPOINT_INTERVAL_TICKS
        ):
            assert self.control is not None
            self.control._activate(current_tick=current_tick, earliest_tick=self.checkpoints[0].tick)
            return
        try:
            handle = self.native.create_snapshot()
            snapshot_tick = int(handle["tick"])
        except RunnerError as error:
            self.capture_failures += 1
            if "manager changed while processing snapshot request" in str(error) and self.capture_failures <= 5:
                self.retry_after_tick = current_tick + REPLAY_CHECKPOINT_INTERVAL_TICKS
                LOGGER.info("native replay snapshot deferred until tick %d: %s", self.retry_after_tick, error)
                return
            LOGGER.warning("native replay rewind snapshots unavailable: %s", error)
            self.disabled = True
            assert self.control is not None
            self.control._deactivate(f"回退不可用：{error}")
            self.close(deactivate=False)
            return
        except (KeyError, TypeError, ValueError) as error:
            LOGGER.warning("native replay rewind snapshots unavailable: %s", error)
            self.disabled = True
            assert self.control is not None
            self.control._deactivate(f"回退不可用：{error}")
            self.close(deactivate=False)
            return
        self.capture_failures = 0
        self.retry_after_tick = 0
        if self.checkpoints and snapshot_tick <= self.checkpoints[-1].tick:
            self._release_handle(handle)
            return
        self.checkpoints.append(_NativeReplayCheckpoint(tick=snapshot_tick, handle=dict(handle), cursor=cursor))
        while len(self.checkpoints) > REPLAY_CHECKPOINT_LIMIT:
            expired = self.checkpoints.pop(0)
            self._release_handle(expired.handle)
        assert self.control is not None
        self.control._activate(current_tick=snapshot_tick, earliest_tick=self.checkpoints[0].tick)

    def restore_requested(self, current_tick: int) -> _NativeReplayRestore | None:
        if self.disabled or self.control is None:
            return None
        requested_ticks = self.control._take_request()
        if requested_ticks < 1:
            return None
        if not self.checkpoints:
            self.control._deactivate("回退不可用：尚无 native 快照")
            return None
        target_tick = max(0, current_tick - requested_ticks)
        eligible = [checkpoint for checkpoint in self.checkpoints if checkpoint.tick <= target_tick]
        checkpoint = eligible[-1] if eligible else self.checkpoints[0]
        self.native.pause()
        restored = self.native.restore(checkpoint.handle)
        restored_tick = int(restored["tick"])
        if restored_tick != checkpoint.tick:
            raise TrainingReplayError(
                f"native rewind restored a different tick: expected {checkpoint.tick}, got {restored_tick}"
            )
        future = [item for item in self.checkpoints if item.tick > checkpoint.tick]
        self.checkpoints = [item for item in self.checkpoints if item.tick <= checkpoint.tick]
        for item in future:
            self._release_handle(item.handle)
        earliest_tick = self.checkpoints[0].tick
        limited = checkpoint.tick > target_tick
        self.control._report_restore(restored_tick=restored_tick, target_tick=target_tick, earliest_tick=earliest_tick)
        return _NativeReplayRestore(
            tick=restored_tick, target_tick=target_tick, cursor=checkpoint.cursor, limited=limited
        )

    def close(self, *, deactivate: bool = True) -> None:
        checkpoints = self.checkpoints
        self.checkpoints = []
        for checkpoint in checkpoints:
            self._release_handle(checkpoint.handle)
        if deactivate and self.control is not None:
            self.control._deactivate("回放已经结束")

    def _release_handle(self, handle: Mapping[str, Any]) -> None:
        try:
            self.native.release_snapshot(handle)
        except (ValueError, RunnerError) as error:
            LOGGER.warning("failed to release native replay snapshot: %s", error)


def _wait_while_paused(
    native: NativeClashEnv, *, pause_event: threading.Event | None, stop_event: threading.Event | None
) -> bool:
    """Pause at a replay operation boundary and resume without ending playback.

    Returns ``True`` when the caller should stop.  Keeping the wait outside
    ``BattleEnv.step`` is important: pausing the native renderer in the middle
    of a blocking step would prevent that step from reaching its target tick.
    """

    if stop_event is not None and stop_event.is_set():
        native.pause()
        return True
    if pause_event is None or not pause_event.is_set():
        return False
    native.pause()
    while pause_event.is_set():
        if stop_event is not None and stop_event.is_set():
            return True
        time.sleep(0.025)
    if stop_event is not None and stop_event.is_set():
        return True
    native.resume()
    return False


@dataclass(frozen=True, slots=True)
class _ResidentSafeCommand:
    request_tick: int
    target_tick: int
    operation_index: int
    deploy: DeployAction | None = None
    hand: HandAction | None = None
    expected_card_id: int | None = None
    ability: AbilityAction | None = None

    @property
    def kind(self) -> str:
        return "deploy" if self.deploy is not None else "ability"


@dataclass(frozen=True, slots=True)
class _DirectRenderCommand:
    """One retained policy action scheduled directly in the stock renderer."""

    request_tick: int
    target_tick: int
    operation_index: int
    owner: int
    card_id: int | None = None
    x: int | None = None
    y: int | None = None
    ability: bool = False
    ability_name_hints: tuple[str, ...] = ()
    ability_source_labels: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "ability" if self.ability else "play_card"


def _metadata_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _match_config_from_replay(replay: TrainingReplayV1) -> MatchConfig:
    """Reconstruct the exact native match inputs retained in an episode."""

    config = replay.episode_config
    tags = config.tags
    return MatchConfig(
        deck0=config.deck0,
        deck1=config.deck1,
        seed=config.seed,
        game_mode=config.game_mode,
        arena=config.arena,
        location=int(tags.get("location", 15000199)),
        level_cap=int(tags.get("level_cap", 0)),
        minimum_card_level=int(tags.get("minimum_card_level", 0)),
        deck0_form_availability=tuple(int(value) for value in tags.get("deck0_form_availability", (0,) * 8)),
        deck1_form_availability=tuple(int(value) for value in tags.get("deck1_form_availability", (0,) * 8)),
        tower_troop0_id=int(tags.get("tower_troop0_id", PRINCESS_TOWER_TROOP_ID)),
        tower_troop1_id=int(tags.get("tower_troop1_id", PRINCESS_TOWER_TROOP_ID)),
        king_tower_level=(int(tags["king_tower_level"]) if tags.get("king_tower_level") is not None else None),
        owner0_name=str(tags.get("owner0_name", "Policy-0")),
        owner1_name=str(tags.get("owner1_name", "Policy-1")),
        end_tick=int(tags.get("end_tick", 7200)),
    )


def _direct_render_commands(replay: TrainingReplayV1) -> tuple[_DirectRenderCommand, ...]:
    """Extract sparse native commands without rebuilding training tensors."""

    commands: list[_DirectRenderCommand] = []
    for operation_index, operation in enumerate(replay.operations):
        for action in operation.actions:
            target_tick = operation.start_native_tick + max(1, action.execute_offset_ticks or 1)
            if action.kind is ActionKind.PLAY_CARD:
                if action.card_id is None or action.target_grid is None:
                    raise TrainingReplayError("retained play-card action is missing card or target")
                x, y = cell_to_world(action.target_grid)
                if action.subcell_offset is not None:
                    x += int(round(action.subcell_offset[0] * 1000))
                    y += int(round(action.subcell_offset[1] * 1000))
                commands.append(
                    _DirectRenderCommand(
                        request_tick=operation.start_native_tick,
                        target_tick=target_tick,
                        operation_index=operation_index,
                        owner=action.owner,
                        card_id=action.card_id,
                        x=min(max(x, 0), 17_999),
                        y=min(max(y, 0), 31_999),
                    )
                )
            elif action.kind is ActionKind.ACTIVATE_ABILITY:
                metadata = action.metadata
                raw_hints = metadata.get("ability_runtime_hints", ())
                raw_labels = metadata.get("ability_source_names", ())
                source_keys = _metadata_strings(metadata.get("ability_source_keys", ()))
                ability_name_hints = _metadata_strings(raw_hints)
                if len(source_keys) > 1:
                    # Public RoyaleAPI rows do not identify which of several
                    # ability cards produced the marker.  An unhinted native
                    # lookup is strict: it succeeds only when exactly one
                    # controller is Ready, instead of selecting the first card
                    # whose alias happens to match.
                    ability_name_hints = ()
                commands.append(
                    _DirectRenderCommand(
                        request_tick=operation.start_native_tick,
                        target_tick=target_tick,
                        operation_index=operation_index,
                        owner=action.owner,
                        ability=True,
                        ability_name_hints=ability_name_hints,
                        ability_source_labels=_metadata_strings(raw_labels),
                    )
                )
    return tuple(sorted(commands, key=lambda item: (item.target_tick, item.request_tick, item.operation_index)))


def _schedule_direct_render_commands(
    native: NativeClashEnv, commands: Sequence[_DirectRenderCommand], *, start_index: int = 0
) -> dict[int, int]:
    """Register renderer commands without resolving live hand/source identity."""

    if not 0 <= start_index <= len(commands):
        raise ValueError("renderer schedule start index is out of range")
    if commands:
        native.clear_replay_schedule()
    if start_index == len(commands):
        return {}
    scheduled: dict[int, int] = {}
    for index in range(start_index, len(commands)):
        command = commands[index]
        if command.ability:
            receipt = native.schedule_replay_ability_at_tick(
                owner=command.owner, ability_name_hints=command.ability_name_hints, execute_tick=command.target_tick
            )
        else:
            assert command.card_id is not None
            assert command.x is not None
            assert command.y is not None
            receipt = native.schedule_replay_card_at_tick(
                owner=command.owner, card_id=command.card_id, x=command.x, y=command.y, execute_tick=command.target_tick
            )
        scheduled[index] = int(receipt["sequence"])
    return scheduled


def _native_deploy_delay(*, current_tick: int, target_tick: int) -> int:
    """Translate an observable application tick to ``deploy`` wire delay."""

    delay = int(target_tick) - int(current_tick) - COMMAND_CONSUMPTION_STEPS
    if delay < 1:
        raise TrainingReplayError(
            "native-render deploy command missed its queueing window: "
            f"target tick {target_tick}, current tick {current_tick}"
        )
    return delay


def _replay_environment(replay: TrainingReplayV1, *, native: object) -> BattleEnvV1:
    source = replay.episode_config.environment
    return BattleEnvV1(
        native=native,  # type: ignore[arg-type]
        warmup_ticks=source.warmup_ticks,
        max_battle_ticks=source.max_battle_ticks,
        entity_limit=source.entity_limit,
        event_limit=source.event_limit,
        event_window_ticks=source.event_window_ticks,
        shaping_beta=source.shaping_beta,
        shaping_tau_ticks=source.shaping_tau_ticks,
        include_native_digest=source.include_native_digest,
    )


def _renderer_has_expected_hand_card(native: NativeClashEnv, command: _ResidentSafeCommand) -> bool:
    """Check one renderer hand identity before queuing its exact action."""

    assert command.hand is not None
    assert command.expected_card_id is not None
    observation = native.observe()
    try:
        player = next(item for item in observation["players"] if int(item["owner"]) == command.hand.owner)
        card = next(item for item in player["hand"] if int(item["handIndex"]) == command.hand.hand_index)
    except StopIteration:
        # The stock scene exposes an empty hand during its opening animation.
        return False
    except (KeyError, TypeError) as error:
        raise TrainingReplayError("native renderer did not expose the expected hand slot") from error
    return int(card["cardId"]) == command.expected_card_id


def _group_actions(actions: Sequence[ActionV1]) -> dict[int, tuple[ActionV1, ...]]:
    grouped: dict[int, list[ActionV1]] = {0: [], 1: []}
    for action in actions:
        grouped[action.owner].append(action)
    if not grouped[0] or not grouped[1]:
        raise TrainingReplayError("each replay operation must contain both owners")
    return {owner: tuple(items) for owner, items in grouped.items()}


def _wait_actions(ticks: int) -> dict[int, tuple[ActionV1, ...]]:
    return {
        owner: (ActionV1.wait(owner, ticks=ticks, metadata={"native_render_replay_wait": True}),) for owner in (0, 1)
    }


def _prepare_resident_safe_commands(
    replay: TrainingReplayV1, *, host: str, port: int, resident_env_id: int
) -> tuple[MatchConfig, tuple[_ResidentSafeCommand, ...]]:
    """Resolve private native command descriptors in an isolated resident slot."""

    if not 0 <= resident_env_id < 16:
        raise ValueError("resident-safe env ID must be in 0..15")
    probe = NativeClashEnv(host, port, timeout=15.0)
    status = probe._request("multi-status")
    occupied_ids = {int(slot["envId"]) for slot in status.get("slots", ())}
    if resident_env_id in occupied_ids:
        raise TrainingReplayError(f"resident-safe scratch slot {resident_env_id} is occupied")

    native = ResidentNativeClashEnv(resident_env_id, host, port, timeout=15.0)
    coordinator = ResidentBatchCoordinatorV1((native,), timeout=30.0)
    proxy = ResidentBatchNativeProxyV1(native, coordinator)
    environment = _replay_environment(replay, native=proxy)
    # This scratch environment is used only to recover low-level native
    # descriptors for a retained visual trace.  Keep the episode/environment
    # checks, but bind implementation-only ruleset drift exactly as the
    # render-only path does below.
    if environment.ruleset_id != replay.ruleset_id:
        if replay.episode_config.environment != environment.environment_config:
            raise TrainingReplayError(
                "retained replay environment configuration does not match the current native runner"
            )
        environment.ruleset_manifest = None
        environment.ruleset_id = replay.ruleset_id
        environment._replay_render_compatibility = True
    decision_ticks = max(1, round(20.0 / replay.episode_config.decision_hz))
    commands: list[_ResidentSafeCommand] = []
    try:
        environment.reset(
            replay.episode_config,
            options={"render_mode": "headless", "decision_ticks": decision_ticks, "event_driven_decisions": False},
        )
        coordinator.start_session()
        match = environment.match_config
        if match is None:
            raise TrainingReplayError("resident-safe preparation did not retain its match config")
        for operation_index, operation in enumerate(replay.operations):
            current_tick = int(environment.raw_observation["tick"])
            if current_tick < operation.start_native_tick:
                coordinator.begin_round((resident_env_id,))
                try:
                    environment.step(_wait_actions(operation.start_native_tick - current_tick), record_trace=False)
                except BaseException as error:
                    coordinator.abort_round(error)
                    raise
                finally:
                    coordinator.finish_round()
            elif current_tick > operation.start_native_tick:
                raise TrainingReplayError(
                    "resident-safe preparation drifted before operation "
                    f"{operation_index}: expected "
                    f"{operation.start_native_tick}, got {current_tick}"
                )

            raw = environment.raw_observation
            for action in operation.actions:
                target_tick = operation.start_native_tick + max(1, action.execute_offset_ticks or 1)
                if action.kind == ActionKind.PLAY_CARD:
                    assert action.hand_slot is not None
                    assert action.target_grid is not None
                    player = next(item for item in raw["players"] if int(item["owner"]) == action.owner)
                    card = next(item for item in player["hand"] if int(item["handIndex"]) == action.hand_slot)
                    if action.card_id is not None and int(card["cardId"]) != action.card_id:
                        raise TrainingReplayError(
                            f"resident-safe card identity diverged at operation {operation_index}"
                        )
                    x, y = environment._world_target(action)
                    commands.append(
                        _ResidentSafeCommand(
                            request_tick=operation.start_native_tick,
                            target_tick=target_tick,
                            operation_index=operation_index,
                            deploy=DeployAction(
                                player_id=int(player["accountId"]),
                                card_id=int(card.get("commandCardId", card["cardId"])),
                                card_parameter=int(card["cardParameter"]),
                                x=x,
                                y=y,
                            ),
                            hand=HandAction(owner=action.owner, hand_index=action.hand_slot, x=x, y=y),
                            expected_card_id=int(card["cardId"]),
                        )
                    )
                elif action.kind == ActionKind.ACTIVATE_ABILITY:
                    assert action.source_entity is not None
                    candidates, _reasons = environment._ability_action_candidates(action.owner, raw)
                    candidate = candidates.get(action.source_entity)
                    if candidate is None:
                        raise TrainingReplayError(
                            f"resident-safe ability identity diverged at operation {operation_index}"
                        )
                    _state, entity_key = candidate
                    commands.append(
                        _ResidentSafeCommand(
                            request_tick=operation.start_native_tick,
                            target_tick=target_tick,
                            operation_index=operation_index,
                            ability=AbilityAction(*entity_key),
                        )
                    )

            coordinator.begin_round((resident_env_id,))
            try:
                environment.step(_group_actions(operation.actions), record_trace=False)
            except BaseException as error:
                coordinator.abort_round(error)
                raise
            finally:
                coordinator.finish_round()
            actual_tick = int(environment.raw_observation["tick"])
            if actual_tick != operation.end_native_tick:
                raise TrainingReplayError(
                    "resident-safe preparation drifted after operation "
                    f"{operation_index}: expected "
                    f"{operation.end_native_tick}, got {actual_tick}"
                )
        return match, tuple(commands)
    finally:
        coordinator.close_session()
        environment.close()
        try:
            native.close()
        except RunnerError:
            # If reset failed before the resident session was configured,
            # there is no slot-level close command to send.
            pass


def _play_training_replay_with_resident_slots(
    replay: TrainingReplayV1,
    *,
    host: str,
    port: int,
    resident_env_id: int,
    resident_prepare_port: int | None,
    speed: float,
    stop_event: threading.Event | None,
    pause_event: threading.Event | None,
    on_native_ready: Callable[[NativeClashEnv], None] | None,
    on_progress: Callable[[ReplayProgress], None] | None,
) -> ReplayResult:
    """Replay without destroying existing resident slots on the engine."""

    match, commands = _prepare_resident_safe_commands(
        replay,
        host=host,
        port=(port if resident_prepare_port is None else resident_prepare_port),
        resident_env_id=resident_env_id,
    )
    native = NativeClashEnv(host, port, timeout=15.0)
    native.configure_native_render(match.to_json())
    if on_native_ready is not None:
        on_native_ready(native)

    queued = 0
    maximum_lateness = 0
    resident_status = native._request("multi-status")
    resident_active = bool(resident_status.get("active") or resident_status.get("mode") == "resident-headless")
    if not resident_active:
        # A clean stock-render process can resolve each card from its own live
        # hand. This is the same path used by the interactive native overlay
        # and is required for renderer-side Hero skin binding.
        native.set_speed(float(speed))
        for command in sorted(commands, key=lambda item: (item.request_tick, item.target_tick, item.operation_index)):
            if stop_event is not None and stop_event.is_set():
                native.pause()
                status = native.status()
                return ReplayResult(
                    stopped=True,
                    final_native_tick=int(status["tick"]),
                    expected_winner=replay.terminal.get("winner"),
                    actual_winner=None,
                    max_absolute_tick_drift=maximum_lateness,
                )

            trigger_tick = max(
                0,
                command.request_tick
                - (HAND_QUEUE_LEAD_TICKS if command.hand is not None else ABILITY_QUEUE_LEAD_TICKS),
            )
            status = native.status()
            current_tick = int(status["tick"])
            while current_tick < trigger_tick:
                time.sleep(0.002)
                status = native.status()
                current_tick = int(status["tick"])
            if status.get("ended"):
                break

            if command.hand is not None:
                while not _renderer_has_expected_hand_card(native, command):
                    if current_tick >= command.target_tick:
                        raise TrainingReplayError(
                            f"native renderer hand diverged before target tick {command.target_tick}"
                        )
                    time.sleep(0.002)
                    status = native.status()
                    current_tick = int(status["tick"])
                receipt = native.queue_hand_action_at(command.hand, execute_tick=command.target_tick)
                if int(receipt["cardId"]) != command.expected_card_id:
                    raise TrainingReplayError("native renderer queued a different hand card")
            else:
                assert command.ability is not None
                while True:
                    if current_tick >= command.target_tick:
                        raise TrainingReplayError(
                            "native-render ability missed target tick "
                            f"{command.target_tick} at native tick "
                            f"{current_tick}"
                        )
                    try:
                        native.queue_ability_action_at(
                            command.ability, execute_in_ticks=(command.target_tick - current_tick)
                        )
                        break
                    except RunnerError as error:
                        if ("could not resolve one ready native champion ability source") not in str(error):
                            raise
                    time.sleep(0.002)
                    status = native.status()
                    current_tick = int(status["tick"])
                    if status.get("ended"):
                        break
                if status.get("ended"):
                    break
            lateness = max(0, current_tick - command.request_tick)
            maximum_lateness = max(maximum_lateness, lateness)
            queued += 1
            if on_progress is not None:
                on_progress(
                    ReplayProgress(
                        operation_index=queued,
                        operation_count=len(commands),
                        native_tick=current_tick,
                        expected_tick=command.target_tick,
                        tick_drift=lateness,
                        phase="queueing",
                    )
                )
    else:
        # With resident slots active, ordinary observations are deliberately
        # unavailable. Prequeue exact headless-resolved card descriptors while
        # paused, then resolve only live Champion abilities during playback.
        native.pause()
        status = native.status()
        current_tick = int(status["tick"])
        for command in (item for item in commands if item.deploy is not None):
            if stop_event is not None and stop_event.is_set():
                return ReplayResult(
                    stopped=True,
                    final_native_tick=current_tick,
                    expected_winner=replay.terminal.get("winner"),
                    actual_winner=None,
                    max_absolute_tick_drift=maximum_lateness,
                )
            assert command.deploy is not None
            native.deploy(
                command.deploy,
                delay_ticks=_native_deploy_delay(current_tick=current_tick, target_tick=command.target_tick),
            )
            queued += 1
            if on_progress is not None:
                on_progress(
                    ReplayProgress(
                        operation_index=queued,
                        operation_count=len(commands),
                        native_tick=current_tick,
                        expected_tick=command.target_tick,
                        tick_drift=0,
                        phase="prequeue",
                    )
                )

        native.set_speed(float(speed))
        native.resume()
        for command in (item for item in commands if item.ability is not None):
            if _wait_while_paused(native, pause_event=pause_event, stop_event=stop_event):
                status = native.status()
                return ReplayResult(
                    stopped=True,
                    final_native_tick=int(status["tick"]),
                    expected_winner=replay.terminal.get("winner"),
                    actual_winner=None,
                    max_absolute_tick_drift=maximum_lateness,
                )

            trigger_tick = max(0, command.request_tick - ABILITY_QUEUE_LEAD_TICKS)
            status = native.status()
            current_tick = int(status["tick"])
            if status.get("ended"):
                break
            while current_tick < trigger_tick:
                if _wait_while_paused(native, pause_event=pause_event, stop_event=stop_event):
                    status = native.status()
                    return ReplayResult(
                        stopped=True,
                        final_native_tick=int(status["tick"]),
                        expected_winner=replay.terminal.get("winner"),
                        actual_winner=None,
                        max_absolute_tick_drift=maximum_lateness,
                    )
                time.sleep(0.002)
                status = native.status()
                current_tick = int(status["tick"])
            assert command.ability is not None
            ability_queued = False
            while current_tick < command.target_tick:
                try:
                    native.queue_ability_action_at(
                        command.ability, execute_in_ticks=(command.target_tick - current_tick)
                    )
                    ability_queued = True
                    break
                except RunnerError as error:
                    if ("could not resolve one ready native champion ability source") not in str(error):
                        raise
                time.sleep(0.002)
                status = native.status()
                current_tick = int(status["tick"])
                if status.get("ended"):
                    break
            if status.get("ended"):
                break
            if not ability_queued:
                LOGGER.warning(
                    "retained ability for owner %d was unavailable at target tick %d",
                    command.ability.owner,
                    command.target_tick,
                )
            lateness = max(0, current_tick - command.request_tick)
            maximum_lateness = max(maximum_lateness, lateness)
            queued += 1
            if on_progress is not None:
                on_progress(
                    ReplayProgress(
                        operation_index=queued,
                        operation_count=len(commands),
                        native_tick=current_tick,
                        expected_tick=command.target_tick,
                        tick_drift=lateness,
                        phase=("queueing" if ability_queued else "skipped-ability"),
                    )
                )

    native.set_speed(float(speed))
    status = native.status()
    while not status.get("ended"):
        if _wait_while_paused(native, pause_event=pause_event, stop_event=stop_event):
            return ReplayResult(
                stopped=True,
                final_native_tick=int(status["tick"]),
                expected_winner=replay.terminal.get("winner"),
                actual_winner=None,
                max_absolute_tick_drift=maximum_lateness,
            )
        if int(status["tick"]) >= NATIVE_FINALIZATION_DEADLINE_TICK:
            native.pause()
            raise TrainingReplayError("resident-safe native replay did not finalize before its deadline")
        if on_progress is not None:
            on_progress(
                ReplayProgress(
                    operation_index=queued,
                    operation_count=len(commands),
                    native_tick=int(status["tick"]),
                    expected_tick=replay.end_native_tick,
                    tick_drift=0,
                    phase="native-render",
                )
            )
        time.sleep(0.025)
        status = native.status()

    expected_winner = replay.terminal.get("winner")
    actual_winner = status.get("winner")
    return ReplayResult(
        stopped=False,
        final_native_tick=int(status["tick"]),
        expected_winner=(int(expected_winner) if expected_winner in (0, 1) else None),
        actual_winner=(int(actual_winner) if actual_winner in (0, 1) else None),
        max_absolute_tick_drift=maximum_lateness,
    )


def _play_training_replay_direct(
    replay: TrainingReplayV1,
    *,
    native: NativeClashEnv,
    speed: float,
    strict_ticks: bool,
    stop_at_replay_end: bool,
    stop_event: threading.Event | None,
    pause_event: threading.Event | None,
    on_native_ready: Callable[[NativeClashEnv], None] | None,
    on_progress: Callable[[ReplayProgress], None] | None,
    rewind_control: ReplayRewindControl | None = None,
) -> ReplayResult:
    history = _NativeReplayHistory(native, rewind_control)
    try:
        return _play_training_replay_direct_core(
            replay,
            native=native,
            speed=speed,
            strict_ticks=strict_ticks,
            stop_at_replay_end=stop_at_replay_end,
            stop_event=stop_event,
            pause_event=pause_event,
            on_native_ready=on_native_ready,
            on_progress=on_progress,
            rewind_history=history,
        )
    finally:
        history.close()


def _play_training_replay_direct_core(
    replay: TrainingReplayV1,
    *,
    native: NativeClashEnv,
    speed: float,
    strict_ticks: bool,
    stop_at_replay_end: bool,
    stop_event: threading.Event | None,
    pause_event: threading.Event | None,
    on_native_ready: Callable[[NativeClashEnv], None] | None,
    on_progress: Callable[[ReplayProgress], None] | None,
    rewind_history: _NativeReplayHistory,
) -> ReplayResult:
    """Schedule the sparse retained actions directly on the native clock.

    The former dedicated-render path called ``BattleEnv.step`` once for every
    policy operation.  That rebuilt observations and tensors while the stock
    renderer continued advancing, so even 1x playback accumulated hundreds
    of ticks of drift.  A visual replay only needs the retained action times:
    resolve cards/abilities from the live renderer shortly before each target
    tick and let the native clock advance continuously. Public replay actions
    are registered semantically at startup; the probe's per-tick hook resolves
    the live hand form or newest queueable Champion at target_tick - 1, falling
    back through older live carriers when needed. The stock renderer therefore
    keeps its requested speed without per-action braking.
    """

    commands = _direct_render_commands(replay)
    native.create_native_match(_match_config_from_replay(replay))
    native.pause()
    if on_native_ready is not None:
        on_native_ready(native)

    command_index = 0
    processed = 0
    queued_count = 0
    ability_count = sum(1 for command in commands if command.ability)
    queued_ability_count = 0
    max_drift = 0
    scheduled_sequences: dict[int, int] = {}

    def cursor() -> _DirectReplayCursor:
        return _DirectReplayCursor(
            command_index=command_index,
            processed=processed,
            queued_count=queued_count,
            queued_ability_count=queued_ability_count,
            max_drift=max_drift,
        )

    def restore_cursor(restored: _DirectReplayCursor) -> None:
        nonlocal command_index
        nonlocal processed
        nonlocal queued_count
        nonlocal queued_ability_count
        nonlocal max_drift

        command_index = restored.command_index
        processed = restored.processed
        queued_count = restored.queued_count
        queued_ability_count = restored.queued_ability_count
        max_drift = restored.max_drift

    def schedule_remaining(status: Mapping[str, Any]) -> None:
        """Register semantic actions without resolving live identities yet."""

        current_tick = int(status["tick"])
        for index in range(command_index, len(commands)):
            command = commands[index]
            if command.target_tick <= current_tick:
                raise TrainingReplayError(
                    "cannot register retained replay action after its target "
                    f"tick {command.target_tick}; native tick is {current_tick}"
                )
        scheduled_sequences.clear()
        scheduled_sequences.update(_schedule_direct_render_commands(native, commands, start_index=command_index))

    def stopped_result(status: Mapping[str, Any]) -> ReplayResult:
        return ReplayResult(
            stopped=True,
            final_native_tick=int(status["tick"]),
            expected_winner=replay.terminal.get("winner"),
            actual_winner=None,
            max_absolute_tick_drift=max_drift,
            completion="user-stopped",
            queued_action_count=queued_count,
            action_count=len(commands),
            queued_ability_count=queued_ability_count,
            ability_count=ability_count,
        )

    def service_controls(status: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        paused_here = False
        while True:
            if stop_event is not None and stop_event.is_set():
                native.pause()
                return "stop", status
            restored = rewind_history.restore_requested(int(status["tick"]))
            if restored is not None:
                restore_cursor(restored.cursor)
                status = native.status()
                schedule_remaining(status)
                if on_progress is not None:
                    on_progress(
                        ReplayProgress(
                            operation_index=processed,
                            operation_count=len(commands),
                            native_tick=restored.tick,
                            expected_tick=restored.target_tick,
                            tick_drift=restored.tick - restored.target_tick,
                            phase=("rewound-limit" if restored.limited else "rewound"),
                        )
                    )
                if pause_event is None or not pause_event.is_set():
                    native.resume()
                return "rewound", status
            rewind_history.capture(status, cursor())
            if pause_event is None or not pause_event.is_set():
                if paused_here:
                    native.resume()
                return "continue", status
            if not paused_here:
                native.pause()
                paused_here = True
            time.sleep(0.025)
            status = native.status()

    def raise_early_native_terminal(status: Mapping[str, Any]) -> None:
        if not stop_at_replay_end:
            return
        tick = int(status["tick"])
        # RoyaleAPI retains a short terminal tail after the last real source
        # event.  Native may therefore finalize before ``end_native_tick``
        # even when every retained action has already been scheduled.  That
        # is a valid native-terminal completion, not evidence of divergence.
        if queued_count == len(commands):
            return
        native.pause()
        replay_tag = str(
            replay.episode_config.tags.get("royaleapi_replay_tag")
            or replay.source.get("replay_tag")
            or replay.episode_id
        )
        crowns = status.get("crownsRaw")
        crown_detail = (
            f"; reconstructed crowns {int(crowns[0])}:{int(crowns[1])}"
            if (isinstance(crowns, Sequence) and not isinstance(crowns, (str, bytes)) and len(crowns) == 2)
            else ""
        )
        raise TrainingReplayError(
            f"RoyaleAPI replay {replay_tag}: native match ended early at "
            f"tick {tick}, before source replay end tick "
            f"{replay.end_native_tick}; queued {queued_count}/"
            f"{len(commands)} actions and {queued_ability_count}/"
            f"{ability_count} abilities{crown_detail}. The reconstructed "
            "battle state diverged before all retained source actions were "
            "queued; later source actions are present but the native manager "
            "is already terminal"
        )

    if commands:
        initial_status = native.status()
        if initial_status.get("ended"):
            raise_early_native_terminal(initial_status)
        schedule_remaining(initial_status)
        rewind_history.capture(initial_status, cursor(), force=True)
    native.set_speed(float(speed))
    native.resume()

    def consume_scheduled_group(status: Mapping[str, Any]) -> bool:
        """Account for one target-tick group after native processed it."""

        nonlocal command_index
        nonlocal processed
        nonlocal queued_count
        nonlocal queued_ability_count
        nonlocal max_drift

        if command_index >= len(commands):
            return False
        current_tick = int(status["tick"])
        target_tick = commands[command_index].target_tick
        if current_tick < target_tick:
            return False
        group_end = command_index + 1
        while group_end < len(commands) and commands[group_end].target_tick == target_tick:
            group_end += 1

        for index in range(command_index, group_end):
            command = commands[index]
            sequence = scheduled_sequences.get(index)
            if sequence is None:
                native.pause()
                raise TrainingReplayError("retained replay action has no native schedule receipt")
            receipt = native.replay_schedule_status(sequence)
            state = str(receipt["state"])
            receipt_tick = int(receipt.get("queuedAtTick", -1))
            queued = state == "succeeded"
            if state == "pending":
                # The game-step hook processes the semantic action before it
                # publishes target_tick. Pending here is therefore a broken
                # scheduling contract, not ordinary host polling latency.
                queued = False
                error_name = "pending-after-target"
            else:
                error_name = str(receipt.get("error", "unknown"))
            absolute_drift = (
                abs((receipt_tick + 1) - command.target_tick)
                if receipt_tick >= 0
                else max(0, current_tick - command.target_tick)
            )
            max_drift = max(max_drift, absolute_drift)
            processed += 1
            command_index += 1
            phase = "queueing" if queued else f"skipped-{command.kind}"
            if queued:
                if receipt_tick != command.target_tick - 1:
                    native.pause()
                    raise TrainingReplayError(
                        f"native renderer resolved {command.kind} at tick "
                        f"{receipt_tick}, expected boundary "
                        f"{command.target_tick - 1}"
                    )
                queued_count += 1
                if command.ability:
                    queued_ability_count += 1
            else:
                message = (
                    f"retained {command.kind} for owner {command.owner} "
                    f"could not be queued by target tick "
                    f"{command.target_tick}: native {error_name}"
                )
                if strict_ticks:
                    native.pause()
                    raise TrainingReplayError(message)
                LOGGER.warning(message)
            if on_progress is not None:
                on_progress(
                    ReplayProgress(
                        operation_index=processed,
                        operation_count=len(commands),
                        native_tick=(receipt_tick if receipt_tick >= 0 else current_tick),
                        expected_tick=command.target_tick,
                        tick_drift=absolute_drift,
                        phase=phase,
                    )
                )
        rewind_history.capture(status, cursor())
        return True

    while True:
        while command_index < len(commands):
            status = native.status()
            if consume_scheduled_group(status):
                if status.get("ended") and command_index < len(commands):
                    raise_early_native_terminal(status)
                    break
                continue
            control_result, status = service_controls(status)
            if control_result == "stop":
                return stopped_result(status)
            if control_result == "rewound":
                continue
            if status.get("ended"):
                raise_early_native_terminal(status)
                break
            time.sleep(0.002)

        status = native.status()
        restart_commands = False
        while True:
            if status.get("ended"):
                raise_early_native_terminal(status)
                break
            control_result, status = service_controls(status)
            if control_result == "stop":
                return stopped_result(status)
            if control_result == "rewound":
                if command_index < len(commands):
                    restart_commands = True
                    break
                continue
            if stop_at_replay_end and int(status["tick"]) >= replay.end_native_tick:
                native.pause()
                expected_winner = replay.terminal.get("winner")
                return ReplayResult(
                    stopped=False,
                    final_native_tick=int(status["tick"]),
                    expected_winner=(int(expected_winner) if expected_winner in (0, 1) else None),
                    actual_winner=None,
                    max_absolute_tick_drift=max_drift,
                    completion="source-timeline",
                    queued_action_count=queued_count,
                    action_count=len(commands),
                    queued_ability_count=queued_ability_count,
                    ability_count=ability_count,
                    source_end_tick=replay.end_native_tick,
                )
            if int(status["tick"]) >= NATIVE_FINALIZATION_DEADLINE_TICK:
                native.pause()
                raise TrainingReplayError("native replay did not finalize before its deadline")
            if on_progress is not None:
                on_progress(
                    ReplayProgress(
                        operation_index=processed,
                        operation_count=len(commands),
                        native_tick=int(status["tick"]),
                        expected_tick=replay.end_native_tick,
                        tick_drift=0,
                        phase="native-render",
                    )
                )
            time.sleep(0.025)
            status = native.status()
        if restart_commands:
            continue
        break

    expected_winner = replay.terminal.get("winner")
    actual_winner = status.get("winner")
    return ReplayResult(
        stopped=False,
        final_native_tick=int(status["tick"]),
        expected_winner=(int(expected_winner) if expected_winner in (0, 1) else None),
        actual_winner=(int(actual_winner) if actual_winner in (0, 1) else None),
        max_absolute_tick_drift=max_drift,
        completion="native-terminal",
        queued_action_count=queued_count,
        action_count=len(commands),
        queued_ability_count=queued_ability_count,
        ability_count=ability_count,
    )


def play_training_replay(
    replay: TrainingReplayV1,
    *,
    host: str = "127.0.0.1",
    port: int = int(setting("CR_CONTROL_PORT", "26789")),
    speed: float = 1.0,
    strict_ticks: bool = False,
    stop_at_replay_end: bool = False,
    resident_safe_env_id: int | None = None,
    resident_safe_prepare_port: int | None = None,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    on_native_ready: Callable[[NativeClashEnv], None] | None = None,
    on_progress: Callable[[ReplayProgress], None] | None = None,
    rewind_control: ReplayRewindControl | None = None,
) -> ReplayResult:
    """Play retained policy actions in the stock native renderer."""

    if float(speed) not in NATIVE_RENDER_SPEEDS:
        raise ValueError("speed must be one of 0.25, 0.5, 1, 2, 4")
    native = NativeClashEnv(host, port, timeout=15.0)
    native.wait_ready(timeout=15.0)
    resident_status = native._request("multi-status")
    occupied = int(resident_status.get("occupied", 0))
    if rewind_control is not None and (occupied or resident_safe_env_id is not None):
        rewind_control._deactivate("回退仅支持专用 native-render 会话")
        raise TrainingReplayError("rewind requires a dedicated native-render session")
    if occupied:
        if resident_safe_env_id is not None:
            return _play_training_replay_with_resident_slots(
                replay,
                host=host,
                port=port,
                resident_env_id=resident_safe_env_id,
                resident_prepare_port=resident_safe_prepare_port,
                speed=speed,
                stop_event=stop_event,
                pause_event=pause_event,
                on_native_ready=on_native_ready,
                on_progress=on_progress,
            )
        raise TrainingReplayError(
            f"native endpoint {host}:{port} still owns {occupied} "
            "resident training matches; stop that collector or use a "
            "dedicated idle engine port before replay"
        )
    if resident_safe_env_id is not None:
        # An explicitly selected scratch slot is useful even on an otherwise
        # idle endpoint: it resolves exact packed native card descriptors from
        # a deterministic headless copy before the visual scene is configured.
        return _play_training_replay_with_resident_slots(
            replay,
            host=host,
            port=port,
            resident_env_id=resident_safe_env_id,
            resident_prepare_port=resident_safe_prepare_port,
            speed=speed,
            stop_event=stop_event,
            pause_event=pause_event,
            on_native_ready=on_native_ready,
            on_progress=on_progress,
        )
    return _play_training_replay_direct(
        replay,
        native=native,
        speed=speed,
        strict_ticks=strict_ticks,
        stop_at_replay_end=stop_at_replay_end,
        stop_event=stop_event,
        pause_event=pause_event,
        on_native_ready=on_native_ready,
        on_progress=on_progress,
        rewind_control=rewind_control,
    )
