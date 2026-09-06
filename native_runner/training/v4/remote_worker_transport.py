"""Length-framed transport for CPU-node PPO engine workers.

Frame tensors use one reusable contiguous byte buffer per record.  The first
frame carries the dataclass/tensor schema; later frames overwrite the same
receiver buffers in place throughout the resident worker session.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Mapping
import torch
from torch import Tensor


_PREFIX = struct.Struct("!cQ")
_PICKLE = b"P"
_FRAME = b"F"
_ACTION = b"A"
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_TENSOR_BYTES = 128 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray(size)
    view = memoryview(output)
    cursor = 0
    while cursor < size:
        received = sock.recv_into(view[cursor:])
        if received == 0:
            raise EOFError("remote PPO worker connection closed")
        cursor += received
    return bytes(output)


def _recv_buffer(sock: socket.socket, output: bytearray) -> None:
    view = memoryview(output)
    cursor = 0
    while cursor < len(output):
        received = sock.recv_into(view[cursor:])
        if received == 0:
            raise EOFError("remote PPO worker tensor stream closed")
        cursor += received


def _tensor_bytes(value: Tensor) -> memoryview:
    contiguous = value.detach().contiguous().view(torch.uint8).reshape(-1)
    return memoryview(contiguous.numpy())


class ReceivedRowPackedTensorRecord:
    """A received nested record backed by one row-major buffer per dtype."""

    def __init__(
        self,
        schema: object,
        batch_size: int,
        source_dtypes: tuple[torch.dtype, ...],
        buffer_dtypes: Mapping[torch.dtype, torch.dtype],
        buffers: Mapping[torch.dtype, Tensor],
    ) -> None:
        self.schema = schema
        self.batch_size = int(batch_size)
        self.source_dtypes = source_dtypes
        self.buffer_dtypes = dict(buffer_dtypes)
        self.buffers = dict(buffers)
        self.record = self._restore()

    def _restore(self) -> object:
        def restore(item: object) -> object:
            if not isinstance(item, tuple) or not item:
                raise TypeError("row-packed tensor schema is malformed")
            kind = item[0]
            if kind == "tensor":
                _kind, source_dtype, tail_shape, offset, width = item
                flat = self.buffers[source_dtype][:, int(offset) : int(offset) + int(width)]
                return flat.reshape((self.batch_size, *tail_shape))
            if kind == "dataclass":
                return item[1](**{name: restore(child) for name, child in item[2]})
            if kind == "tuple":
                return tuple(restore(child) for child in item[1])
            if kind == "list":
                return [restore(child) for child in item[1]]
            if kind == "dict":
                return {restore(key): restore(child) for key, child in item[1]}
            if kind == "value":
                return item[1]
            raise TypeError(f"unknown row-packed tensor schema node: {kind!r}")

        return restore(self.schema)

    def compatible(self, other: object) -> bool:
        return (
            getattr(other, "schema", None) == self.schema
            and getattr(other, "source_dtypes", None) == self.source_dtypes
            and getattr(other, "buffer_dtypes", None) == self.buffer_dtypes
        )

    @classmethod
    def _from_buffers(cls, template, buffers):
        result = cls.__new__(cls)
        result.schema = template.schema
        result.batch_size = int(next(iter(buffers.values())).shape[0])
        result.source_dtypes = template.source_dtypes
        result.buffer_dtypes = template.buffer_dtypes
        result.buffers = dict(buffers)
        result.record = result._restore()
        return result

    def narrow_rows(self, start: int, length: int) -> "ReceivedRowPackedTensorRecord":
        if start < 0 or length <= 0 or start + length > self.batch_size:
            raise ValueError("row-packed narrow is outside the batch")
        return self._from_buffers(
            self, {dtype: buffer.narrow(0, start, length) for dtype, buffer in self.buffers.items()}
        )

    def index_select(self, indices: Tensor) -> "ReceivedRowPackedTensorRecord":
        if indices.device.type != "cpu" or indices.dtype != torch.long:
            raise TypeError("row-packed indices must be CPU int64")
        return self._from_buffers(
            self, {dtype: buffer.index_select(0, indices) for dtype, buffer in self.buffers.items()}
        )


class RemoteWorkerConnection:
    """Socket object with the subset of multiprocessing Connection we use."""

    remote_tensor_transfer = True

    def __init__(self, sock: socket.socket) -> None:
        self._socket = sock
        self._out_row_packed: object | None = None
        self._in_row_packed_payloads: tuple[bytearray, ...] | None = None
        self._in_row_packed: ReceivedRowPackedTensorRecord | None = None
        if self._socket.family in {socket.AF_INET, socket.AF_INET6}:
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

    @classmethod
    def connect(cls, host: str, port: int, timeout: float) -> "RemoteWorkerConnection":
        sock = socket.create_connection((host, int(port)), timeout=timeout)
        sock.settimeout(None)
        return cls(sock)

    def fileno(self) -> int:
        return self._socket.fileno()

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()

    def close_local_copy(self) -> None:
        """Close one fork-inherited descriptor without shutting down its peer."""

        self._socket.close()

    def _send(self, kind: bytes, header: object, payloads=()) -> None:
        encoded = pickle.dumps(header, protocol=5)
        if len(encoded) > _MAX_HEADER_BYTES:
            raise ValueError("remote PPO message header is too large")
        self._socket.sendall(_PREFIX.pack(kind, len(encoded)))
        self._socket.sendall(encoded)
        for payload in payloads:
            self._socket.sendall(payload)

    def send(self, value: object) -> None:
        if isinstance(value, dict) and value.get("kind") == "frame":
            row_packed = value.get("row_packed_batch")
            if row_packed is not None:
                source_dtypes = getattr(row_packed, "source_dtypes", None)
                buffer_dtypes = getattr(row_packed, "buffer_dtypes", None)
                buffers = getattr(row_packed, "buffers", None)
                schema = getattr(row_packed, "schema", None)
                batch_size = getattr(row_packed, "batch_size", None)
                if (
                    not isinstance(source_dtypes, tuple)
                    or not isinstance(buffer_dtypes, dict)
                    or not isinstance(buffers, dict)
                    or not isinstance(batch_size, int)
                    or batch_size <= 0
                ):
                    raise TypeError("remote PPO row-packed frame is invalid")
                payloads = tuple(_tensor_bytes(buffers[dtype]) for dtype in source_dtypes)
                payload_sizes = tuple(len(payload) for payload in payloads)
                if sum(payload_sizes) > _MAX_TENSOR_BYTES:
                    raise ValueError("remote PPO row-packed frame is too large")
                schema_changed = (
                    self._out_row_packed is None
                    or self._out_row_packed.batch_size != batch_size
                    or not self._out_row_packed.compatible(row_packed)
                )
                self._out_row_packed = row_packed
                metadata = {
                    key: row for key, row in value.items() if key not in {"batch", "storage_batch", "row_packed_batch"}
                }
                descriptor = None
                if schema_changed:
                    descriptor = {
                        "schema": schema,
                        "batch_size": batch_size,
                        "source_dtypes": source_dtypes,
                        "buffer_dtypes": buffer_dtypes,
                        "buffer_shapes": tuple(
                            tuple(int(size) for size in buffers[dtype].shape) for dtype in source_dtypes
                        ),
                    }
                header = {"metadata": metadata, "row_packed_descriptor": descriptor, "row_packed_bytes": payload_sizes}
                self._send(_FRAME, header, payloads)
                return
            raise RuntimeError("remote PPO frame requires row-packed tensors")
        if isinstance(value, dict) and value.get("kind") == "act_packed":
            packed_actions = value.get("packed_actions")
            if not isinstance(packed_actions, Tensor):
                raise TypeError("remote PPO packed actions changed type")
            if packed_actions.device.type != "cpu" or packed_actions.dtype != torch.long or packed_actions.ndim != 2:
                raise TypeError("remote PPO packed actions must be a CPU int64 matrix")
            packed_actions = packed_actions.detach().contiguous()
            payload = _tensor_bytes(packed_actions)
            if len(payload) > _MAX_TENSOR_BYTES:
                raise ValueError("remote PPO packed actions are too large")
            header = {key: row for key, row in value.items() if key != "packed_actions"}
            header["packed_shape"] = tuple(packed_actions.shape)
            header["packed_bytes"] = len(payload)
            self._send(_ACTION, header, (payload,))
            return
        self._send(_PICKLE, value)

    def recv(self) -> object:
        kind, header_size = _PREFIX.unpack(_recv_exact(self._socket, _PREFIX.size))
        if header_size > _MAX_HEADER_BYTES:
            raise ValueError("remote PPO message header exceeds its bound")
        header = pickle.loads(_recv_exact(self._socket, int(header_size)))
        if kind == _PICKLE:
            return header
        if kind == _ACTION:
            if not isinstance(header, dict):
                raise TypeError("remote PPO packed action header is invalid")
            shape = header.pop("packed_shape", None)
            payload_size = int(header.pop("packed_bytes", -1))
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(int(dimension) < 0 for dimension in shape)
                or payload_size < 0
                or payload_size > _MAX_TENSOR_BYTES
            ):
                raise ValueError("remote PPO packed action shape is invalid")
            expected_size = 8
            for dimension in shape:
                expected_size *= int(dimension)
            if payload_size != expected_size:
                raise ValueError("remote PPO packed action byte count changed")
            payload = bytearray(payload_size)
            _recv_buffer(self._socket, payload)
            header["packed_actions"] = torch.frombuffer(payload, dtype=torch.long, count=payload_size // 8).reshape(
                shape
            )
            return header
        if kind != _FRAME or not isinstance(header, dict):
            raise ValueError("remote PPO message kind is invalid")
        row_packed_descriptor = header.get("row_packed_descriptor")
        row_packed_sizes = header.get("row_packed_bytes")
        if row_packed_descriptor is not None or row_packed_sizes is not None:
            replacement = row_packed_descriptor is not None
            if replacement:
                if not isinstance(row_packed_descriptor, dict):
                    raise TypeError("remote PPO row-packed descriptor is invalid")
                source_dtypes = row_packed_descriptor.get("source_dtypes")
                buffer_dtypes = row_packed_descriptor.get("buffer_dtypes")
                buffer_shapes = row_packed_descriptor.get("buffer_shapes")
                batch_size = row_packed_descriptor.get("batch_size")
                schema = row_packed_descriptor.get("schema")
                if (
                    not isinstance(source_dtypes, tuple)
                    or not isinstance(buffer_dtypes, dict)
                    or not isinstance(buffer_shapes, tuple)
                    or len(buffer_shapes) != len(source_dtypes)
                    or not isinstance(batch_size, int)
                    or batch_size <= 0
                ):
                    raise TypeError("remote PPO row-packed descriptor changed type")
                payloads: list[bytearray] = []
                tensors: dict[torch.dtype, Tensor] = {}
                for source_dtype, shape in zip(source_dtypes, buffer_shapes, strict=True):
                    if (
                        not isinstance(source_dtype, torch.dtype)
                        or source_dtype not in buffer_dtypes
                        or not isinstance(shape, tuple)
                        or len(shape) != 2
                        or int(shape[0]) != batch_size
                        or int(shape[1]) <= 0
                    ):
                        raise TypeError("remote PPO row-packed buffer changed type")
                    buffer_dtype = buffer_dtypes[source_dtype]
                    if not isinstance(buffer_dtype, torch.dtype):
                        raise TypeError("remote PPO row-packed dtype changed type")
                    count = int(shape[0]) * int(shape[1])
                    payload = bytearray(count * torch.empty((), dtype=buffer_dtype).element_size())
                    payloads.append(payload)
                    tensors[source_dtype] = torch.frombuffer(payload, dtype=buffer_dtype, count=count).reshape(shape)
                if sum(len(payload) for payload in payloads) > _MAX_TENSOR_BYTES:
                    raise ValueError("remote PPO row-packed frame is too large")
                self._in_row_packed_payloads = tuple(payloads)
                self._in_row_packed = ReceivedRowPackedTensorRecord(
                    schema, batch_size, source_dtypes, buffer_dtypes, tensors
                )
            if self._in_row_packed_payloads is None or self._in_row_packed is None:
                raise RuntimeError("remote PPO row-packed frame arrived before schema")
            if not isinstance(row_packed_sizes, tuple) or tuple(
                len(payload) for payload in self._in_row_packed_payloads
            ) != tuple(int(size) for size in row_packed_sizes):
                raise ValueError("remote PPO row-packed byte count changed")
            for payload in self._in_row_packed_payloads:
                _recv_buffer(self._socket, payload)
            metadata = header.get("metadata")
            if not isinstance(metadata, dict):
                raise TypeError("remote PPO frame metadata is invalid")
            metadata["row_packed_batch"] = self._in_row_packed
            if replacement:
                metadata["batch"] = self._in_row_packed.record
            return metadata
        raise ValueError("remote PPO frame requires a row-packed descriptor")
