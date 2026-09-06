"""Persistent batched data-plane client for resident semantic matches."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import time
from typing import Any, Mapping, Sequence
import zlib

import orjson

from .cr_native_env import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_RUNNER_RESPONSE_BYTES,
    RunnerError,
    _validate_atomic_observation,
)
from .training_capture import TrainingAtomicCaptureV1, TrainingCaptureError, decode_training_capture


_VERSION = 1
_HANDSHAKE = struct.Struct("<4sHHHH")
_HEADER = struct.Struct("<4sHHQI")
_REQUEST_ENTRY = struct.Struct("<HHI")
_RESPONSE_ENTRY = struct.Struct("<HHHHI")
_ACTION = struct.Struct("<BBHiiiI")
_RECEIPT = struct.Struct("<BBHHHiIIiiiii")
_HANDSHAKE_MAGIC = b"CRBH"
_REQUEST_MAGIC = b"CRBQ"
_RESPONSE_MAGIC = b"CRBS"
_MAX_ENVIRONMENTS = 16
_MAX_ACTIONS = 8
_MAX_EXECUTE_OFFSET_TICKS = 0xFFFF
_ACTION_PLAY = 1
_ACTION_ABILITY = 2
_ENTRY_ZLIB = 1


@dataclass(frozen=True, slots=True)
class BatchPlayActionV1:
    owner: int
    hand_index: int
    x: int
    y: int
    execute_offset_ticks: int = 1


@dataclass(frozen=True, slots=True)
class BatchAbilityActionV1:
    owner: int
    object_index: int
    secondary_index: int
    execute_offset_ticks: int = 1


BatchActionV1 = BatchPlayActionV1 | BatchAbilityActionV1


@dataclass(frozen=True, slots=True)
class BatchActionReceiptV1:
    kind: int
    owner: int
    action_index: int
    status: int
    execute_tick: int
    card_id: int
    card_parameter: int
    deck_slot: int
    cost: int
    form_code: int
    argument0: int
    argument1: int


@dataclass(frozen=True, slots=True)
class BatchStepResultV1:
    env_id: int
    receipts: tuple[BatchActionReceiptV1, ...]
    capture: dict[str, Any] | TrainingAtomicCaptureV1


@dataclass(frozen=True, slots=True)
class BatchStepFailureV1:
    env_id: int
    error: RunnerError


BatchStepOutcomeV1 = BatchStepResultV1 | BatchStepFailureV1


class ResidentBatchChannelV1:
    """One persistent connection that advances a batch of resident matches.

    Every request carries the complete fixed-tick action set for each resident
    match. Card plays and Hero Knight ability activations are applied before
    the native step, and the response returns action receipts plus one atomic
    ordinary/rich capture per match.

    The current native control server is single-client. Keep this channel open
    only while collecting a batch sequence; close it before issuing ordinary
    control/reset/performance commands.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 15.0,
        *,
        training_capture: bool = False,
        compressed_json: bool = False,
        validate_json_capture: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.training_capture = bool(training_capture)
        self.compressed_json = bool(compressed_json)
        self.validate_json_capture = bool(validate_json_capture)
        self._closed = False
        self._sequence = 0
        self._phase_validation_identity: dict[int, tuple[int, int]] = {}
        self._phase_validation_next_sequence: dict[int, int] = {}
        self._profile: dict[str, int] = {
            "connect_ns": 0,
            "batches": 0,
            "environments": 0,
            "request_bytes": 0,
            "response_bytes": 0,
            "decoded_response_bytes": 0,
            "send_ns": 0,
            "request_ack_ns": 0,
            "last_request_ack_ns": 0,
            "receive_ns": 0,
            "decode_ns": 0,
            "total_ns": 0,
        }

        started = time.perf_counter_ns()
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._socket.settimeout(self.timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        handshake_started = time.perf_counter_ns()
        self._socket.sendall(
            (
                b"batch-fast-zlib-v1\n"
                if self.training_capture and self.compressed_json
                else b"batch-fast-v1\n"
                if self.training_capture
                else b"batch-zlib-v1\n"
                if self.compressed_json
                else b"batch-v1\n"
            )
        )
        handshake = self._receive_exact(_HANDSHAKE.size)
        self._profile["last_request_ack_ns"] = time.perf_counter_ns() - handshake_started
        self._profile["connect_ns"] = time.perf_counter_ns() - started
        magic, version, status, capacity, reserved = _HANDSHAKE.unpack(handshake)
        if (
            magic != _HANDSHAKE_MAGIC
            or version != _VERSION
            or status != 0
            or capacity != _MAX_ENVIRONMENTS
            or reserved != 0
        ):
            self.close()
            raise RunnerError(
                "resident batch handshake failed: "
                f"magic={magic!r} version={version} status={status} "
                f"capacity={capacity} reserved={reserved}"
            )

    def _receive_exact(self, size: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < size:
            chunk = self._socket.recv(size - received)
            if not chunk:
                raise RunnerError("resident batch channel closed during a frame")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _validate_request(
        actions_by_env: Mapping[int, Sequence[BatchActionV1]], ticks: int
    ) -> tuple[tuple[int, tuple[BatchActionV1, ...]], ...]:
        normalized = tuple((int(env_id), tuple(actions)) for env_id, actions in actions_by_env.items())
        env_ids = tuple(env_id for env_id, _actions in normalized)
        if (
            not normalized
            or len(normalized) > _MAX_ENVIRONMENTS
            or len(set(env_ids)) != len(env_ids)
            or any(not 0 <= env_id < _MAX_ENVIRONMENTS for env_id in env_ids)
        ):
            raise ValueError("env_ids must contain 1..16 unique values from 0..15")
        if isinstance(ticks, bool) or not 1 <= int(ticks) <= 1_000_000:
            raise ValueError("ticks must be in 1..1_000_000")
        for _env_id, actions in normalized:
            if len(actions) > _MAX_ACTIONS:
                raise ValueError("one environment supports at most 8 actions")
            for action in actions:
                if action.owner not in (0, 1):
                    raise ValueError("action owner must be 0 or 1")
                offset = action.execute_offset_ticks
                if isinstance(offset, bool) or not 1 <= int(offset) <= _MAX_EXECUTE_OFFSET_TICKS:
                    raise ValueError("action execute offset must be in 1..65535")
                if isinstance(action, BatchPlayActionV1):
                    if action.hand_index < 0:
                        raise ValueError("play action hand index must be non-negative")
                elif isinstance(action, BatchAbilityActionV1):
                    tagged = action.object_index == -2 and action.secondary_index > 0
                    legacy = action.object_index >= 0 and action.secondary_index >= 0
                    if not tagged and not legacy:
                        raise ValueError("ability action source identity is invalid")
                else:
                    raise TypeError("batch actions must be play or ability actions")
        return normalized

    @staticmethod
    def _encode_action(action: BatchActionV1) -> bytes:
        if isinstance(action, BatchPlayActionV1):
            return _ACTION.pack(
                _ACTION_PLAY, action.owner, action.execute_offset_ticks, action.hand_index, action.x, action.y, 0
            )
        return _ACTION.pack(
            _ACTION_ABILITY,
            action.owner,
            action.execute_offset_ticks,
            action.object_index,
            action.secondary_index,
            0,
            0,
        )

    @staticmethod
    def _decode_receipt(payload: bytes) -> BatchActionReceiptV1:
        (
            kind,
            owner,
            action_index,
            status,
            reserved,
            execute_tick,
            card_id,
            card_parameter,
            deck_slot,
            cost,
            form_code,
            argument0,
            argument1,
        ) = _RECEIPT.unpack(payload)
        if reserved != 0:
            raise RunnerError("resident batch receipt reserved field is set")
        return BatchActionReceiptV1(
            kind=kind,
            owner=owner,
            action_index=action_index,
            status=status,
            execute_tick=execute_tick,
            card_id=card_id,
            card_parameter=card_parameter,
            deck_slot=deck_slot,
            cost=cost,
            form_code=form_code,
            argument0=argument0,
            argument1=argument1,
        )

    @staticmethod
    def _server_failure(env_id: int, raw: Mapping[str, Any]) -> RunnerError:
        detail = raw.get("error", raw)
        context = ", ".join(f"{key}={raw[key]}" for key in ("stage", "bytesUsed", "capacity") if key in raw)
        suffix = f" ({context})" if context else ""
        return RunnerError(f"resident batch env {env_id} failed: {detail}{suffix}")

    def step_partitioned(
        self, actions_by_env: Mapping[int, Sequence[BatchActionV1]], *, ticks: int = 2
    ) -> dict[int, BatchStepOutcomeV1]:
        """Return one result or explicit failure for every framed slot."""

        if self._closed:
            raise RunnerError("resident batch channel is closed")
        normalized = self._validate_request(actions_by_env, ticks)
        self._sequence += 1
        sequence = self._sequence
        entries = b"".join(
            _REQUEST_ENTRY.pack(env_id, len(actions), int(ticks))
            + b"".join(self._encode_action(action) for action in actions)
            for env_id, actions in normalized
        )
        request = _HEADER.pack(_REQUEST_MAGIC, _VERSION, 0, sequence, len(normalized)) + entries

        total_started = time.perf_counter_ns()
        phase_started = time.perf_counter_ns()
        self._socket.sendall(request)
        send_ns = time.perf_counter_ns() - phase_started

        phase_started = time.perf_counter_ns()
        response_header = self._receive_exact(_HEADER.size)
        request_ack_ns = time.perf_counter_ns() - total_started
        magic, version, status, response_sequence, count = _HEADER.unpack(response_header)
        if (
            magic != _RESPONSE_MAGIC
            or version != _VERSION
            or status != 0
            or response_sequence != sequence
            or count != len(normalized)
        ):
            raise RunnerError(
                "resident batch response header mismatch: "
                f"magic={magic!r} version={version} status={status} "
                f"sequence={response_sequence} count={count}"
            )

        payloads: list[tuple[int, int, int, tuple[BatchActionReceiptV1, ...], bytes]] = []
        response_bytes = len(response_header)
        for _index in range(count):
            entry_header = self._receive_exact(_RESPONSE_ENTRY.size)
            (env_id, entry_status, receipt_count, entry_flags, payload_size) = _RESPONSE_ENTRY.unpack(entry_header)
            if entry_flags & ~_ENTRY_ZLIB or receipt_count > _MAX_ACTIONS:
                raise RunnerError("resident batch entry header is invalid")
            if entry_flags == _ENTRY_ZLIB and not self.compressed_json:
                raise RunnerError("resident batch returned unexpected compressed JSON")
            if payload_size > MAX_RUNNER_RESPONSE_BYTES:
                raise RunnerError("resident batch entry exceeds the 16 MiB host bound")
            receipts = tuple(
                self._decode_receipt(self._receive_exact(_RECEIPT.size)) for _receipt_index in range(receipt_count)
            )
            payload = self._receive_exact(payload_size)
            response_bytes += len(entry_header) + receipt_count * _RECEIPT.size + len(payload)
            payloads.append((env_id, entry_status, entry_flags, receipts, payload))
        receive_ns = time.perf_counter_ns() - phase_started

        phase_started = time.perf_counter_ns()
        results: dict[int, BatchStepOutcomeV1] = {}
        for (expected_env_id, expected_actions), (env_id, entry_status, entry_flags, receipts, payload) in zip(
            normalized, payloads, strict=True
        ):
            if env_id != expected_env_id:
                raise RunnerError("resident batch response changed environment order")
            try:
                if entry_flags == _ENTRY_ZLIB:
                    payload = zlib.decompress(payload)
                    if len(payload) > MAX_RUNNER_RESPONSE_BYTES:
                        raise RunnerError("resident batch decompressed entry exceeds 16 MiB")
                self._profile["decoded_response_bytes"] += len(payload)
                if self.training_capture and payload.startswith(b"CRTF"):
                    capture: dict[str, Any] | TrainingAtomicCaptureV1 = decode_training_capture(payload)
                    raw: object = capture.ordinary
                else:
                    raw = orjson.loads(payload)
                    if not isinstance(raw, Mapping):
                        raise RunnerError(f"resident batch env {env_id} failed: {raw!r}")
                    if entry_status != 0 or raw.get("ok") is False:
                        raise self._server_failure(env_id, raw)
                    if self.validate_json_capture:
                        identity = (raw.get("generation"), raw.get("stateEpoch"))
                        previous_identity = self._phase_validation_identity.get(env_id)
                        phase_event_floor = (
                            self._phase_validation_next_sequence.get(env_id) if identity == previous_identity else None
                        )
                        capture = _validate_atomic_observation(dict(raw), phase_event_floor=phase_event_floor)
                        rich = capture.get("rich")
                        phase_runtime = rich.get("phaseRuntime") if isinstance(rich, Mapping) else None
                        if isinstance(phase_runtime, Mapping):
                            self._phase_validation_identity[env_id] = (
                                int(capture["generation"]),
                                int(capture["stateEpoch"]),
                            )
                            self._phase_validation_next_sequence[env_id] = int(phase_runtime["nextSequence"])
                        else:
                            self._phase_validation_identity.pop(env_id, None)
                            self._phase_validation_next_sequence.pop(env_id, None)
                    else:
                        # BattleEnv immediately binds this mapping into the typed
                        # rich snapshot, which repeats the semantic validation.
                        capture = dict(raw)
                if entry_status != 0 or not isinstance(raw, Mapping):
                    raise RunnerError(f"resident batch env {env_id} failed: {raw!r}")
                if len(receipts) != len(expected_actions) or any(
                    receipt.status != 0 or receipt.action_index != action_index
                    for action_index, receipt in enumerate(receipts)
                ):
                    raise RunnerError(f"resident batch env {env_id} returned bad action receipts")
                results[env_id] = BatchStepResultV1(env_id=env_id, receipts=receipts, capture=capture)
            except (RunnerError, TrainingCaptureError, orjson.JSONDecodeError, zlib.error) as error:
                self._phase_validation_identity.pop(env_id, None)
                self._phase_validation_next_sequence.pop(env_id, None)
                failure = (
                    error
                    if isinstance(error, RunnerError)
                    else RunnerError(f"resident batch env {env_id} response decode failed: {error}")
                )
                results[env_id] = BatchStepFailureV1(env_id, failure)
        decode_ns = time.perf_counter_ns() - phase_started

        self._profile["batches"] += 1
        self._profile["environments"] += len(normalized)
        self._profile["request_bytes"] += len(request)
        self._profile["response_bytes"] += response_bytes
        self._profile["send_ns"] += send_ns
        self._profile["request_ack_ns"] += request_ack_ns
        self._profile["last_request_ack_ns"] = request_ack_ns
        self._profile["receive_ns"] += receive_ns
        self._profile["decode_ns"] += decode_ns
        self._profile["total_ns"] += time.perf_counter_ns() - total_started
        return results

    def step(
        self, actions_by_env: Mapping[int, Sequence[BatchActionV1]], *, ticks: int = 2
    ) -> dict[int, BatchStepResultV1]:
        """Apply mixed actions and fail if any requested slot failed."""

        outcomes = self.step_partitioned(actions_by_env, ticks=ticks)
        results: dict[int, BatchStepResultV1] = {}
        for env_id, outcome in outcomes.items():
            if isinstance(outcome, BatchStepFailureV1):
                raise outcome.error
            results[env_id] = outcome
        return results

    @property
    def last_request_ack_ns(self) -> int:
        return int(self._profile["last_request_ack_ns"])

    def profile(self) -> dict[str, int | str]:
        return {"version": "resident-batch-channel-profile.v1", **self._profile}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()

    def __enter__(self) -> "ResidentBatchChannelV1":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
