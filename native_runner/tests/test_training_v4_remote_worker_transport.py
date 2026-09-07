from __future__ import annotations

import socket
import signal
from dataclasses import dataclass

import pytest
import torch

from native_runner.training.v4 import remote_ppo_worker_server
from native_runner.training.v4.remote_worker_transport import RemoteWorkerConnection
from native_runner.training.v4.async_cluster_self_play import (
    _RowPackedTensorRecordV4,
)
from native_runner.training.v4.remote_ppo_worker_server import _worker_cpu_group
from native_runner.training.v4.tensors import ActionSequenceV4


def _actions(value: int, rows: int = 2) -> ActionSequenceV4:
    shape = (rows, 4)
    return ActionSequenceV4(
        gate=torch.full((rows,), value, dtype=torch.long),
        micro_action_count=torch.zeros(rows, dtype=torch.long),
        candidate_index=torch.full(shape, -1, dtype=torch.long),
        candidate_uid=torch.full(shape, -1, dtype=torch.long),
        target_cell=torch.full(shape, -1, dtype=torch.long),
        delay_offset_bin=torch.full(shape, -1, dtype=torch.long),
    )


@dataclass(frozen=True)
class _MixedRecord:
    integer: torch.Tensor
    floating: torch.Tensor


def test_remote_worker_child_drops_parent_listener_and_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    handlers: list[tuple[signal.Signals, object]] = []
    monkeypatch.setattr(remote_ppo_worker_server.os, "close", closed.append)
    monkeypatch.setattr(
        remote_ppo_worker_server.signal,
        "signal",
        lambda signum, handler: handlers.append((signum, handler)),
    )

    remote_ppo_worker_server._prepare_worker_process(91)

    assert closed == [91]
    assert handlers == [
        (signal.SIGINT, signal.SIG_DFL),
        (signal.SIGTERM, signal.SIG_DFL),
    ]


def test_remote_worker_cpu_groups_pair_complementary_engine_cpus() -> None:
    allowed = tuple(range(64))

    assert _worker_cpu_group(
        allowed,
        local_index=0,
        engine_count=64,
        width=2,
    ) == (0, 32)
    assert _worker_cpu_group(
        allowed,
        local_index=47,
        engine_count=64,
        width=2,
    ) == (47, 15)


def test_remote_worker_cpu_groups_share_a_smaller_pool_when_enabled() -> None:
    allowed = tuple(range(56, 64))

    assert _worker_cpu_group(
        allowed,
        local_index=0,
        engine_count=26,
        width=1,
        allow_sharing=True,
    ) == (56,)
    assert _worker_cpu_group(
        allowed,
        local_index=8,
        engine_count=26,
        width=1,
        allow_sharing=True,
    ) == (56,)
    assert _worker_cpu_group(
        allowed,
        local_index=25,
        engine_count=26,
        width=1,
        allow_sharing=True,
    ) == (57,)
    with pytest.raises(ValueError, match="CPU grouping"):
        _worker_cpu_group(
            allowed,
            local_index=0,
            engine_count=26,
            width=1,
        )


def test_remote_frame_transport_reuses_receiver_tensor_storage() -> None:
    left_socket, right_socket = socket.socketpair()
    sender = RemoteWorkerConnection(left_socket)
    receiver = RemoteWorkerConnection(right_socket)
    packed = _RowPackedTensorRecordV4(_actions(0))
    batch = packed.record

    sender.send(
        {
            "kind": "frame",
            "engine_index": 7,
            "actor_rows": (0, 1),
            "storage_actor_rows": (0,),
            "rewards": (),
            "row_packed_batch": packed,
        }
    )
    first = receiver.recv()
    assert isinstance(first, dict)
    received_batch = first["batch"]
    assert isinstance(received_batch, ActionSequenceV4)
    assert received_batch.gate.tolist() == [0, 0]

    batch.gate.fill_(1)
    sender.send(
        {
            "kind": "frame",
            "engine_index": 7,
            "actor_rows": (0, 1),
            "storage_actor_rows": (0,),
            "rewards": (),
            "row_packed_batch": packed,
        }
    )
    second = receiver.recv()
    assert isinstance(second, dict)
    assert "batch" not in second
    assert received_batch.gate.tolist() == [1, 1]

    sender.send({"kind": "act", "actions": ((0, (0,)),)})
    assert receiver.recv() == {"kind": "act", "actions": ((0, (0,)),)}
    sender.close()
    receiver.close()


def test_remote_frame_transport_replaces_changed_tensor_shapes() -> None:
    left_socket, right_socket = socket.socketpair()
    sender = RemoteWorkerConnection(left_socket)
    receiver = RemoteWorkerConnection(right_socket)
    sender.send(
        {
            "kind": "frame",
            "row_packed_batch": _RowPackedTensorRecordV4(_actions(0)),
        }
    )
    first = receiver.recv()
    assert isinstance(first, dict) and "batch" in first

    sender.send(
        {
            "kind": "frame",
            "row_packed_batch": _RowPackedTensorRecordV4(_actions(1, rows=3)),
        }
    )
    second = receiver.recv()
    assert isinstance(second, dict)
    assert isinstance(second["batch"], ActionSequenceV4)
    assert second["batch"].gate.tolist() == [1, 1, 1]
    sender.close()
    receiver.close()


def test_remote_frame_transport_can_derive_storage_from_the_model_batch() -> None:
    left_socket, right_socket = socket.socketpair()
    sender = RemoteWorkerConnection(left_socket)
    receiver = RemoteWorkerConnection(right_socket)
    sender.send(
        {
            "kind": "frame",
            "remote_storage_from_batch": True,
            "row_packed_batch": _RowPackedTensorRecordV4(_actions(1)),
        }
    )

    packet = receiver.recv()

    assert isinstance(packet, dict)
    assert packet["remote_storage_from_batch"] is True
    assert isinstance(packet["batch"], ActionSequenceV4)
    assert "storage_batch" not in packet
    sender.close()
    receiver.close()


def test_remote_row_packed_frame_preserves_layout_and_reuses_receiver() -> None:
    left_socket, right_socket = socket.socketpair()
    sender = RemoteWorkerConnection(left_socket)
    receiver = RemoteWorkerConnection(right_socket)
    record = _MixedRecord(
        integer=torch.arange(6, dtype=torch.long).reshape(2, 3),
        floating=torch.arange(8, dtype=torch.float16).reshape(2, 4),
    )
    packed = _RowPackedTensorRecordV4(record)

    sender.send(
        {
            "kind": "frame",
            "actor_rows": (0, 1),
            "remote_storage_from_batch": True,
            "row_packed_batch": packed,
        }
    )
    first = receiver.recv()
    assert isinstance(first, dict)
    received_packed = first["row_packed_batch"]
    received = first["batch"]
    assert isinstance(received, _MixedRecord)
    assert torch.equal(received.integer, record.integer)
    assert torch.equal(received.floating, record.floating)

    packed.record.integer.add_(10)
    packed.record.floating.add_(20)
    sender.send(
        {
            "kind": "frame",
            "actor_rows": (0, 1),
            "remote_storage_from_batch": True,
            "row_packed_batch": packed,
        }
    )
    second = receiver.recv()
    assert isinstance(second, dict)
    assert "batch" not in second
    assert second["row_packed_batch"] is received_packed
    assert torch.equal(received.integer, record.integer + 10)
    assert torch.equal(received.floating, record.floating + 20)
    sender.close()
    receiver.close()


def test_remote_packed_action_transport_preserves_actor_rows_and_matrix() -> None:
    left_socket, right_socket = socket.socketpair()
    sender = RemoteWorkerConnection(left_socket)
    receiver = RemoteWorkerConnection(right_socket)
    packed = torch.arange(54, dtype=torch.long).reshape(3, 18)

    sender.send(
        {
            "kind": "act_packed",
            "actor_rows": (7, 3, 11),
            "packed_actions": packed,
        }
    )
    received = receiver.recv()

    assert isinstance(received, dict)
    assert received["kind"] == "act_packed"
    assert received["actor_rows"] == (7, 3, 11)
    assert torch.equal(received["packed_actions"], packed)
    sender.close()
    receiver.close()
