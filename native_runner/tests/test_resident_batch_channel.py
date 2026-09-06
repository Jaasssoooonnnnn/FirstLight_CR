from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
import zlib

import pytest

import native_runner.resident_batch_channel as batch_module
import native_runner.resident_batch_vector as vector_module
from native_runner.cr_native_env import RunnerError
from native_runner.resident_batch_channel import (
    BatchAbilityActionV1,
    BatchPlayActionV1,
    BatchStepFailureV1,
    BatchStepResultV1,
    ResidentBatchChannelV1,
)
from native_runner.resident_batch_vector import ResidentBatchCoordinatorV1
from native_runner.tests.test_native_runtime_telemetry_api import (
    _atomic_response,
)
from native_runner.tests.test_training_capture import _capture
from native_runner.training_capture import TrainingAtomicCaptureV1


class _ScriptedSocket:
    def __init__(self, responses: bytes) -> None:
        self.responses = responses
        self.sent = b""
        self.closed = False
        self.timeout: float | None = None
        self.socket_options: list[tuple[int, int, int]] = []

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.socket_options.append((level, option, value))

    def sendall(self, payload: bytes) -> None:
        self.sent += payload

    def recv(self, maximum: int) -> bytes:
        result, self.responses = (
            self.responses[:maximum],
            self.responses[maximum:],
        )
        return result

    def shutdown(self, _how: int) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _InMemoryBatchChannel:
    instances: list["_InMemoryBatchChannel"] = []

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        training_capture: bool = False,
        compressed_json: bool = False,
        validate_json_capture: bool = True,
        allow_legacy_training_capture_v1: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.training_capture = training_capture
        self.compressed_json = compressed_json
        self.validate_json_capture = validate_json_capture
        self.allow_legacy_training_capture_v1 = (
            allow_legacy_training_capture_v1
        )
        self.closed = False
        self.requests: list[tuple[dict[int, tuple[object, ...]], int]] = []
        type(self).instances.append(self)

    def step_partitioned(
        self,
        actions_by_env: dict[int, tuple[object, ...]],
        *,
        ticks: int,
    ) -> dict[int, BatchStepResultV1]:
        self.requests.append((dict(actions_by_env), ticks))
        return {
            env_id: BatchStepResultV1(
                env_id=env_id,
                receipts=(),
                capture={
                    "ordinary": {"tick": ticks},
                    "rich": {"tick": ticks},
                },
            )
            for env_id in actions_by_env
        }

    def profile(self) -> dict[str, int | str]:
        return {
            "version": "resident-batch-channel-profile.v1",
            "batches": len(self.requests),
        }

    def close(self) -> None:
        self.closed = True


def _successful_stream(
    env_id: int,
    receipts: tuple[bytes, ...] = (),
    *,
    compressed: bool = False,
) -> bytes:
    payload = json.dumps(
        _atomic_response(),
        separators=(",", ":"),
    ).encode("utf-8")
    wire_payload = zlib.compress(payload, level=1) if compressed else payload
    return b"".join(
        (
            batch_module._HANDSHAKE.pack(
                batch_module._HANDSHAKE_MAGIC,
                1,
                0,
                16,
                0,
            ),
            batch_module._HEADER.pack(
                batch_module._RESPONSE_MAGIC,
                1,
                0,
                1,
                1,
            ),
            batch_module._RESPONSE_ENTRY.pack(
                env_id,
                0,
                len(receipts),
                batch_module._ENTRY_ZLIB if compressed else 0,
                len(wire_payload),
            ),
            *receipts,
            wire_payload,
        )
    )


def _response_frame(
    sequence: int,
    entries: tuple[tuple[int, int, bytes], ...],
) -> bytes:
    return b"".join(
        (
            batch_module._HEADER.pack(
                batch_module._RESPONSE_MAGIC,
                1,
                0,
                sequence,
                len(entries),
            ),
            *(
                batch_module._RESPONSE_ENTRY.pack(
                    env_id,
                    status,
                    0,
                    0,
                    len(payload),
                )
                + payload
                for env_id, status, payload in entries
            ),
        )
    )


def test_persistent_batch_channel_round_trip_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = _ScriptedSocket(_successful_stream(3))
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )

    with ResidentBatchChannelV1(timeout=2.0) as channel:
        captures = {key: result.capture for key, result in channel.step({3: ()}, ticks=2).items()}
        profile = channel.profile()

    assert scripted.sent == b"".join(
        (
            b"batch-v1\n",
            batch_module._HEADER.pack(
                batch_module._REQUEST_MAGIC,
                1,
                0,
                1,
                1,
            ),
            batch_module._REQUEST_ENTRY.pack(3, 0, 2),
        )
    )
    assert scripted.timeout == 2.0
    assert scripted.socket_options == [
        (batch_module.socket.IPPROTO_TCP, batch_module.socket.TCP_NODELAY, 1)
    ]
    assert scripted.closed is True
    assert captures[3]["ordinary"]["tick"] == 530
    assert profile["batches"] == 1
    assert profile["environments"] == 1
    assert profile["connect_ns"] > 0
    assert profile["total_ns"] > 0
    assert profile["request_ack_ns"] > 0
    assert profile["last_request_ack_ns"] > 0


def test_persistent_batch_channel_decompresses_full_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = _ScriptedSocket(_successful_stream(3, compressed=True))
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )

    with ResidentBatchChannelV1(compressed_json=True) as channel:
        captures = {key: result.capture for key, result in channel.step({3: ()}, ticks=5).items()}
        profile = channel.profile()

    assert scripted.sent.startswith(b"batch-zlib-v1\n")
    assert captures[3]["ordinary"]["tick"] == 530
    assert profile["decoded_response_bytes"] > profile["response_bytes"]


def test_persistent_batch_channel_decompresses_compact_training_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _capture()
    wire_payload = zlib.compress(payload, level=1)
    scripted = _ScriptedSocket(
        b"".join(
            (
                batch_module._HANDSHAKE.pack(
                    batch_module._HANDSHAKE_MAGIC,
                    1,
                    0,
                    16,
                    0,
                ),
                batch_module._HEADER.pack(
                    batch_module._RESPONSE_MAGIC,
                    1,
                    0,
                    1,
                    1,
                ),
                batch_module._RESPONSE_ENTRY.pack(
                    3,
                    0,
                    0,
                    batch_module._ENTRY_ZLIB,
                    len(wire_payload),
                ),
                wire_payload,
            )
        )
    )
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )

    with ResidentBatchChannelV1(
        training_capture=True,
        compressed_json=True,
    ) as channel:
        captures = {key: result.capture for key, result in channel.step({3: ()}, ticks=5).items()}
        profile = channel.profile()

    assert scripted.sent.startswith(b"batch-fast-zlib-v1\n")
    assert isinstance(captures[3], TrainingAtomicCaptureV1)
    assert profile["decoded_response_bytes"] > profile["response_bytes"]


def test_batch_trusted_hot_path_defers_json_validation_to_battle_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = _ScriptedSocket(_successful_stream(3, compressed=True))
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )
    monkeypatch.setattr(
        batch_module,
        "_validate_atomic_observation",
        lambda *_args, **_kwargs: pytest.fail("duplicate validation was called"),
    )

    with ResidentBatchChannelV1(
        compressed_json=True,
        validate_json_capture=False,
    ) as channel:
        captures = {key: result.capture for key, result in channel.step({3: ()}, ticks=5).items()}

    assert captures[3]["rich"]["tick"] == 530


def test_persistent_batch_channel_encodes_mixed_actions_and_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    play_receipt = batch_module._RECEIPT.pack(
        1,
        0,
        0,
        0,
        0,
        537,
        26_000_001,
        0x60000001,
        0,
        6,
        2,
        1,
        0,
    )
    ability_receipt = batch_module._RECEIPT.pack(
        2,
        0,
        1,
        0,
        0,
        535,
        0,
        0,
        -1,
        -1,
        -1,
        -2,
        500,
    )
    scripted = _ScriptedSocket(
        _successful_stream(3, (play_receipt, ability_receipt))
    )
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )

    with ResidentBatchChannelV1() as channel:
        results = channel.step(
            {
                3: (
                    BatchPlayActionV1(
                        0,
                        1,
                        12_000,
                        18_000,
                        execute_offset_ticks=7,
                    ),
                    BatchAbilityActionV1(
                        0,
                        -2,
                        500,
                        execute_offset_ticks=5,
                    ),
                )
            }
        )

    assert scripted.sent == b"".join(
        (
            b"batch-v1\n",
            batch_module._HEADER.pack(
                batch_module._REQUEST_MAGIC,
                1,
                0,
                1,
                1,
            ),
            batch_module._REQUEST_ENTRY.pack(3, 2, 2),
            batch_module._ACTION.pack(
                1,
                0,
                7,
                1,
                12_000,
                18_000,
                0,
            ),
            batch_module._ACTION.pack(
                2,
                0,
                5,
                -2,
                500,
                0,
                0,
            ),
        )
    )
    assert results[3].receipts[0].card_id == 26_000_001
    assert results[3].receipts[0].cost == 6
    assert results[3].receipts[1].argument1 == 500


def test_batch_channel_partitions_failed_lane_and_keeps_stream_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handshake = batch_module._HANDSHAKE.pack(
        batch_module._HANDSHAKE_MAGIC,
        1,
        0,
        16,
        0,
    )
    failed = json.dumps(
        {
            "ok": False,
            "error": "rich observation section failed to encode",
            "stage": "phase-runtime",
            "bytesUsed": 123,
            "capacity": 456,
        },
        separators=(",", ":"),
    ).encode()
    succeeded = json.dumps(
        _atomic_response(),
        separators=(",", ":"),
    ).encode()
    scripted = _ScriptedSocket(
        handshake
        + _response_frame(1, ((2, 1, failed), (5, 0, succeeded)))
        + _response_frame(2, ((5, 0, succeeded),))
    )
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )

    with ResidentBatchChannelV1(validate_json_capture=False) as channel:
        first = channel.step_partitioned({2: (), 5: ()}, ticks=5)
        second = channel.step({5: ()}, ticks=5)

    assert isinstance(first[2], BatchStepFailureV1)
    assert "stage=phase-runtime" in str(first[2].error)
    assert isinstance(first[5], BatchStepResultV1)
    assert second[5].capture["ordinary"]["tick"] == 530


def test_batch_phase_validation_history_is_lane_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handshake = batch_module._HANDSHAKE.pack(
        batch_module._HANDSHAKE_MAGIC,
        1,
        0,
        16,
        0,
    )

    def payload(marker: int, next_sequence: int) -> bytes:
        return json.dumps(
            {
                "ok": True,
                "generation": 1,
                "stateEpoch": 1,
                "marker": marker,
                "rich": {"phaseRuntime": {"nextSequence": next_sequence}},
            },
            separators=(",", ":"),
        ).encode()

    scripted = _ScriptedSocket(
        handshake
        + _response_frame(1, ((2, 0, payload(2, 20)), (5, 0, payload(5, 50))))
        + _response_frame(2, ((2, 0, payload(2, 21)), (5, 0, payload(5, 51))))
    )
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )
    floors: list[tuple[int, int | None]] = []

    def validate(
        raw: dict[str, object],
        *,
        phase_event_floor: int | None,
    ) -> dict[str, object]:
        floors.append((int(raw["marker"]), phase_event_floor))
        return raw

    monkeypatch.setattr(batch_module, "_validate_atomic_observation", validate)

    with ResidentBatchChannelV1() as channel:
        channel.step({2: (), 5: ()}, ticks=5)
        channel.step({2: (), 5: ()}, ticks=5)

    assert floors == [(2, None), (5, None), (2, 20), (5, 50)]




def test_batch_channel_rejects_server_handshake_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = _ScriptedSocket(
        batch_module._HANDSHAKE.pack(
            batch_module._HANDSHAKE_MAGIC,
            1,
            1,
            16,
            0,
        )
    )
    monkeypatch.setattr(
        batch_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: scripted,
    )

    with pytest.raises(RunnerError, match="handshake failed"):
        ResidentBatchChannelV1()

    assert scripted.closed is True


def test_batch_coordinator_barriers_one_request_for_all_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _InMemoryBatchChannel.instances.clear()
    monkeypatch.setattr(
        vector_module,
        "ResidentBatchChannelV1",
        _InMemoryBatchChannel,
    )
    natives = [
        SimpleNamespace(
            env_id=env_id,
            host="127.0.0.1",
            port=26789,
        )
        for env_id in (2, 5)
    ]
    coordinator = ResidentBatchCoordinatorV1(  # type: ignore[arg-type]
        natives,
        timeout=2.0,
    )
    coordinator.start_session()
    coordinator.begin_round((2, 5))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(coordinator.submit, env_id, (), 2)
            for env_id in (2, 5)
        ]
        results = [future.result(timeout=2.0) for future in futures]
    coordinator.finish_round()

    channel = _InMemoryBatchChannel.instances[0]
    assert [result.env_id for result in results] == [2, 5]
    assert channel.requests == [({2: (), 5: ()}, 2)]
    assert coordinator.profile()["batches"] == 1
    coordinator.close_session()
    assert channel.closed is True




def test_batch_coordinator_releases_all_waiters_on_tick_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _InMemoryBatchChannel.instances.clear()
    monkeypatch.setattr(
        vector_module,
        "ResidentBatchChannelV1",
        _InMemoryBatchChannel,
    )
    natives = [
        SimpleNamespace(
            env_id=env_id,
            host="127.0.0.1",
            port=26789,
        )
        for env_id in (0, 1)
    ]
    coordinator = ResidentBatchCoordinatorV1(  # type: ignore[arg-type]
        natives,
        timeout=2.0,
    )
    coordinator.start_session()
    coordinator.begin_round((0, 1))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(coordinator.submit, 0, (), 2),
            executor.submit(coordinator.submit, 1, (), 3),
        )
        for future in futures:
            with pytest.raises(RuntimeError, match="equal ticks"):
                future.result(timeout=2.0)
    coordinator.finish_round()
    coordinator.close_session()

    assert _InMemoryBatchChannel.instances[0].requests == []


def test_batch_coordinator_returns_success_when_another_lane_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PartiallyFailingChannel(_InMemoryBatchChannel):
        def step_partitioned(
            self,
            actions_by_env: dict[int, tuple[object, ...]],
            *,
            ticks: int,
        ) -> dict[int, BatchStepResultV1 | BatchStepFailureV1]:
            results = super().step_partitioned(actions_by_env, ticks=ticks)
            results[2] = BatchStepFailureV1(2, RunnerError("lane two failed"))
            return results

    monkeypatch.setattr(
        vector_module,
        "ResidentBatchChannelV1",
        PartiallyFailingChannel,
    )
    natives = [
        SimpleNamespace(env_id=env_id, host="127.0.0.1", port=26789)
        for env_id in (2, 5)
    ]
    coordinator = ResidentBatchCoordinatorV1(  # type: ignore[arg-type]
        natives,
        timeout=2.0,
    )
    coordinator.start_session()
    coordinator.begin_round((2, 5))
    with ThreadPoolExecutor(max_workers=2) as executor:
        failed = executor.submit(coordinator.submit, 2, (), 5)
        succeeded = executor.submit(coordinator.submit, 5, (), 5)
        with pytest.raises(RunnerError, match="lane two failed"):
            failed.result(timeout=2.0)
        assert succeeded.result(timeout=2.0).env_id == 5
    assert coordinator.round_succeeded is True
    coordinator.finish_round()
    coordinator.close_session()
