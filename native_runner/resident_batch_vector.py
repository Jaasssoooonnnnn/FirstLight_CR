"""Batch-native coordinator and per-slot compatibility proxies."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping, Sequence

from .cr_native_env import (
    FIRST_PLAYABLE_TICK,
    AbilityAction,
    HandAction,
    ResidentNativeClashEnv,
    RunnerError,
    _decode_observed_hand_card,
    _observed_command_card_id,
)
from .resident_batch_channel import (
    BatchAbilityActionV1,
    BatchActionV1,
    BatchPlayActionV1,
    BatchStepFailureV1,
    BatchStepOutcomeV1,
    BatchStepResultV1,
    ResidentBatchChannelV1,
)
from .training_capture import TrainingAtomicCaptureV1


@dataclass(frozen=True, slots=True)
class _ExpectedReceipt:
    kind: int
    owner: int
    execute_tick: int
    argument0: int
    argument1: int
    card_id: int = 0
    card_parameter: int = 0
    deck_slot: int = -1
    cost: int = -1
    form_code: int = -1


class ResidentBatchCoordinatorV1:
    """Synchronize independent BattleEnv threads onto one native batch call."""

    def __init__(
        self,
        natives: Sequence[ResidentNativeClashEnv],
        *,
        timeout: float = 30.0,
        training_capture: bool = True,
        compressed_json: bool = False,
        validate_json_capture: bool = True,
    ) -> None:
        if not natives:
            raise ValueError("batch coordinator requires resident natives")
        env_ids = tuple(native.env_id for native in natives)
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("resident native env IDs must be unique")
        endpoint = {(native.host, native.port) for native in natives}
        if len(endpoint) != 1:
            raise ValueError("batch coordinator requires one native endpoint")
        self.natives = tuple(natives)
        self.env_ids = env_ids
        self.host, self.port = endpoint.pop()
        self.timeout = float(timeout)
        self.training_capture = bool(training_capture)
        self.compressed_json = bool(compressed_json)
        self.validate_json_capture = bool(validate_json_capture)
        self._condition = threading.Condition()
        self._channel: ResidentBatchChannelV1 | None = None
        self._last_profile: Mapping[str, int | str] = {"version": "resident-batch-channel-profile.v1", "batches": 0}
        self._expected: tuple[int, ...] = ()
        self._submissions: dict[int, tuple[tuple[BatchActionV1, ...], int]] = {}
        self._results: dict[int, BatchStepOutcomeV1] | None = None
        self._error: BaseException | None = None
        self._round_active = False

    @property
    def session_active(self) -> bool:
        return self._channel is not None

    def start_session(self) -> None:
        if self._channel is not None:
            return
        self._channel = ResidentBatchChannelV1(
            self.host,
            self.port,
            timeout=self.timeout,
            training_capture=self.training_capture,
            compressed_json=self.compressed_json,
            validate_json_capture=self.validate_json_capture,
        )

    def close_session(self) -> None:
        channel, self._channel = self._channel, None
        if channel is not None:
            self._last_profile = channel.profile()
            channel.close()
        with self._condition:
            self._round_active = False
            self._expected = ()
            self._submissions.clear()
            self._results = None
            self._error = None
            self._condition.notify_all()

    def begin_round(self, env_ids: Sequence[int]) -> None:
        normalized = tuple(int(env_id) for env_id in env_ids)
        if (
            not normalized
            or len(set(normalized)) != len(normalized)
            or any(env_id not in self.env_ids for env_id in normalized)
        ):
            raise ValueError("batch round env IDs are invalid")
        with self._condition:
            if self._channel is None:
                raise RuntimeError("batch session is not active")
            if self._round_active:
                raise RuntimeError("previous batch round is still active")
            self._expected = normalized
            self._submissions = {}
            self._results = None
            self._error = None
            self._round_active = True

    def abort_round(self, error: BaseException) -> None:
        with self._condition:
            if self._round_active and self._results is None and self._error is None:
                self._error = error
                self._condition.notify_all()

    @property
    def round_succeeded(self) -> bool:
        """Whether native results exist before lane-local post-processing."""

        with self._condition:
            return self._round_active and self._results is not None and self._error is None

    def submit(self, env_id: int, actions: Sequence[BatchActionV1], ticks: int) -> BatchStepResultV1:
        execute = False
        payload: dict[int, tuple[BatchActionV1, ...]] = {}
        with self._condition:
            if not self._round_active or env_id not in self._expected:
                raise RuntimeError("resident env is not in the active batch")
            if env_id in self._submissions:
                raise RuntimeError("resident env submitted twice")
            self._submissions[env_id] = (tuple(actions), int(ticks))
            if len(self._submissions) == len(self._expected):
                tick_values = {value[1] for value in self._submissions.values()}
                if len(tick_values) != 1:
                    self._error = RuntimeError("all resident batch slots must advance equal ticks")
                    self._condition.notify_all()
                else:
                    payload = {slot_id: self._submissions[slot_id][0] for slot_id in self._expected}
                    execute = True

        if execute:
            try:
                channel = self._channel
                if channel is None:
                    raise RuntimeError("batch session closed during a round")
                results = channel.step_partitioned(payload, ticks=int(ticks))
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
            else:
                with self._condition:
                    self._results = results
                    self._condition.notify_all()

        deadline = time.monotonic() + self.timeout
        with self._condition:
            while self._results is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._error = TimeoutError("resident batch round did not complete")
                    self._condition.notify_all()
                    break
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                raise self._error
            assert self._results is not None
            outcome = self._results[env_id]
            if isinstance(outcome, BatchStepFailureV1):
                raise outcome.error
            return outcome

    def finish_round(self) -> None:
        with self._condition:
            if not self._round_active:
                raise RuntimeError("no resident batch round is active")
            if self._results is None and self._error is None:
                raise RuntimeError("resident batch round has not completed")
            self._round_active = False
            self._expected = ()
            self._submissions.clear()
            self._results = None
            self._error = None

    def profile(self) -> Mapping[str, int | str]:
        channel = self._channel
        if channel is None:
            return self._last_profile
        return channel.profile()

    @property
    def estimated_one_way_transport_ns(self) -> int:
        """Estimate the next command's one-way transport from the last ACK."""

        current = self._channel
        current_ack = int(getattr(current, "last_request_ack_ns", 0)) if current is not None else 0
        previous_ack = int(self._last_profile.get("last_request_ack_ns", 0))
        return max(current_ack, previous_ack) // 2


class ResidentBatchNativeProxyV1:
    """NativeClashEnv-compatible facade backed by a batch coordinator."""

    resident_event_rings_rebind_on_switch = True

    def __init__(self, native: ResidentNativeClashEnv, coordinator: ResidentBatchCoordinatorV1) -> None:
        if native not in coordinator.natives:
            raise ValueError("native is not owned by this coordinator")
        self.native = native
        self.coordinator = coordinator
        self.env_id = native.env_id
        self.host = native.host
        self.port = native.port
        self.timeout = native.timeout
        self._cached_ordinary: dict[str, Any] | None = None
        self._cached_atomic: dict[str, Any] | TrainingAtomicCaptureV1 | None = None
        self._actions: list[BatchActionV1] = []
        self._expected_receipts: list[_ExpectedReceipt] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def _require_cached_ordinary(self) -> dict[str, Any]:
        if self._cached_ordinary is None:
            raise RunnerError("batch native proxy has no cached observation")
        return self._cached_ordinary

    def create_match(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.coordinator.session_active:
            raise RuntimeError("close batch session before creating a match")
        result = self.native.create_match(*args, **kwargs)
        self._cached_ordinary = result
        self._cached_atomic = None
        return result

    def step(self, ticks: int = 1) -> dict[str, Any]:
        if not self.coordinator.session_active:
            result = self.native.step(ticks)
            self._cached_ordinary = result
            self._cached_atomic = None
            return result
        actions = tuple(self._actions)
        expected = tuple(self._expected_receipts)
        self._actions.clear()
        self._expected_receipts.clear()
        result = self.coordinator.submit(self.env_id, actions, int(ticks))
        self._validate_receipts(expected, result)
        self._cached_atomic = result.capture
        self._cached_ordinary = (
            result.capture.ordinary
            if isinstance(result.capture, TrainingAtomicCaptureV1)
            else result.capture["ordinary"]
        )
        return self._cached_ordinary

    def observe(self) -> dict[str, Any]:
        if self.coordinator.session_active:
            return self._require_cached_ordinary()
        result = self.native.observe()
        self._cached_ordinary = result
        self._cached_atomic = None
        return result

    def observe_atomic(self) -> dict[str, Any] | TrainingAtomicCaptureV1:
        if self.coordinator.session_active:
            if self._cached_atomic is None:
                raise RunnerError("batch native proxy has no atomic capture")
            return self._cached_atomic
        result = self.native.observe_atomic()
        self._cached_atomic = result
        self._cached_ordinary = result["ordinary"]
        return result

    def observe_rich(self) -> dict[str, Any]:
        if self.coordinator.session_active:
            if self._cached_atomic is None:
                raise RunnerError("batch native proxy has no rich capture")
            if isinstance(self._cached_atomic, TrainingAtomicCaptureV1):
                raise RunnerError("compact training capture has no rich JSON envelope")
            return self._cached_atomic["rich"]
        return self.native.observe_rich()

    def queue_hand_action_at(
        self, action: HandAction, *, execute_tick: int | None = None, execute_in_ticks: int | None = None
    ) -> dict[str, Any]:
        if not self.coordinator.session_active:
            return self.native.queue_hand_action_at(
                action, execute_tick=execute_tick, execute_in_ticks=execute_in_ticks
            )
        if execute_tick is not None and execute_in_ticks is not None:
            raise ValueError("pass execute_tick or execute_in_ticks, not both")
        before = self._require_cached_ordinary()
        current_tick = int(before["tick"])
        applied_tick = int(execute_tick) if execute_tick is not None else current_tick + int(execute_in_ticks or 1)
        applied_tick = max(FIRST_PLAYABLE_TICK + 1, applied_tick)
        execute_offset = applied_tick - current_tick
        try:
            player = next(item for item in before["players"] if int(item["owner"]) == action.owner)
            card = next(item for item in player["hand"] if int(item["handIndex"]) == action.hand_index)
        except StopIteration as error:
            raise ValueError("selected native hand slot is not available") from error
        descriptor = _decode_observed_hand_card(card)
        self._actions.append(
            BatchPlayActionV1(
                owner=action.owner,
                hand_index=action.hand_index,
                x=action.x,
                y=action.y,
                execute_offset_ticks=execute_offset,
            )
        )
        self._expected_receipts.append(
            _ExpectedReceipt(
                kind=1,
                owner=action.owner,
                execute_tick=applied_tick,
                argument0=action.hand_index,
                argument1=action.x,
                card_id=int(card["cardId"]),
                card_parameter=descriptor.packed,
                deck_slot=descriptor.deck_slot,
                cost=descriptor.cost,
                form_code=descriptor.form_code,
            )
        )
        boundary = applied_tick - 1
        return {
            "ok": True,
            "mode": "resident-batch-headless",
            "queued": True,
            "owner": action.owner,
            "handIndex": action.hand_index,
            "cardId": int(card["cardId"]),
            "commandCardId": _observed_command_card_id(card),
            "cardParameter": descriptor.packed,
            "deckSlot": descriptor.deck_slot,
            "cost": descriptor.cost,
            "formCode": descriptor.form_code,
            "formName": descriptor.form_name,
            "queuedAtTick": current_tick,
            "commandAgeBoundaryTick": boundary,
            "executeTick": applied_tick,
        }

    def queue_ability_action_at(self, action: AbilityAction, *, execute_in_ticks: int = 1) -> dict[str, Any]:
        if not self.coordinator.session_active:
            return self.native.queue_ability_action_at(action, execute_in_ticks=execute_in_ticks)
        if not 1 <= execute_in_ticks <= 0xFFFF:
            raise ValueError("ability execute offset must be in 1..65535")
        current_tick = int(self._require_cached_ordinary()["tick"])
        self._actions.append(
            BatchAbilityActionV1(
                owner=action.owner,
                object_index=action.object_index,
                secondary_index=action.secondary_index,
                execute_offset_ticks=execute_in_ticks,
            )
        )
        self._expected_receipts.append(
            _ExpectedReceipt(
                kind=2,
                owner=action.owner,
                execute_tick=current_tick + execute_in_ticks,
                argument0=action.object_index,
                argument1=action.secondary_index,
            )
        )
        return {
            "ok": True,
            "mode": "resident-batch-headless",
            "queued": True,
            "owner": action.owner,
            "sourceEntityKey": [action.owner, action.object_index, action.secondary_index],
            "queuedAtTick": current_tick,
            "executeTick": current_tick + execute_in_ticks,
        }

    def activate_ability(self, action: AbilityAction) -> dict[str, Any]:
        return self.queue_ability_action_at(action)

    @staticmethod
    def _validate_receipts(expected: Sequence[_ExpectedReceipt], result: BatchStepResultV1) -> None:
        if len(expected) != len(result.receipts):
            raise RunnerError("resident batch receipt count changed")
        for wanted, actual in zip(expected, result.receipts, strict=True):
            comparable = (
                actual.kind,
                actual.owner,
                actual.execute_tick,
                actual.argument0,
                actual.argument1,
                actual.card_id,
                actual.card_parameter,
                actual.deck_slot,
                actual.cost,
                actual.form_code,
            )
            target = (
                wanted.kind,
                wanted.owner,
                wanted.execute_tick,
                wanted.argument0,
                wanted.argument1,
                wanted.card_id,
                wanted.card_parameter,
                wanted.deck_slot,
                wanted.cost,
                wanted.form_code,
            )
            if comparable != target:
                raise RunnerError(
                    "resident batch action receipt differs from cached "
                    f"pre-state: expected={target!r} actual={comparable!r}"
                )

    def close(self) -> None:
        self.native.close()
