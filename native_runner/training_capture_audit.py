"""Diagnostic comparison of compact training captures with full JSON observations.

Enabled only when the engine runs with CR_TRAINING_AUDIT=1 (every compact capture then
also carries the same frame's atomic JSON observation) and the host sets
CR_TRAINING_AUDIT_LOG to a JSONL path.  Each frame's compact ordinary dict and rich
snapshot are compared with the JSON path's ordinary dict and bind_rich_telemetry()
snapshot; differences are aggregated by field path and appended to the log.
"""

from __future__ import annotations

import atexit
import dataclasses
import os
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

import orjson

from .rich_telemetry_adapter import bind_rich_telemetry

AUDIT_MARKER = b"\nCRAUDIT\n"
_INDEX = re.compile(r"\[[^\]]*\]")


def split_audit_payload(payload: bytes) -> tuple[bytes, bytes | None]:
    marker = payload.find(AUDIT_MARKER)
    if marker < 0:
        return payload, None
    return payload[:marker], payload[marker + len(AUDIT_MARKER) :]


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _normalize(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _diff(left: Any, right: Any, path: str, out: list[tuple[str, Any, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            _diff(left.get(key, "<absent>"), right.get(key, "<absent>"), f"{path}.{key}", out)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append((f"{path}#len", len(left), len(right)))
        for index, (a, b) in enumerate(zip(left, right)):
            _diff(a, b, f"{path}[{index}]", out)
    elif left != right:
        out.append((path, left, right))


class TrainingCaptureAuditor:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self.frames = 0
        self.frames_with_diff = 0
        self.paths: Counter[str] = Counter()
        self.examples: dict[str, tuple[Any, Any]] = {}
        atexit.register(self.flush)

    def record(self, env_id: int, compact_ordinary: Mapping[str, Any], compact_snapshot: Any, json_bytes: bytes) -> None:
        raw = orjson.loads(json_bytes)
        ordinary, rich = raw["ordinary"], raw["rich"]
        json_snapshot = bind_rich_telemetry(dict(ordinary), rich)
        differences: list[tuple[str, Any, Any]] = []
        # objects_by_key / objects_by_native_object_id are indexes of `objects`.
        compact = _normalize(compact_snapshot)
        expected = _normalize(json_snapshot)
        for name in ("objects_by_key", "objects_by_native_object_id"):
            compact.pop(name, None)
            expected.pop(name, None)
        _diff(compact, expected, "snapshot", differences)
        _diff(_normalize(compact_ordinary), _normalize(ordinary), "ordinary", differences)
        self.frames += 1
        if differences:
            self.frames_with_diff += 1
        for path, a, b in differences:
            key = _INDEX.sub("[]", path)
            self.paths[key] += 1
            if key not in self.examples:
                self.examples[key] = (repr(a)[:300], repr(b)[:300], int(ordinary.get("tick", -1)), env_id)
        if self.frames % 100 == 0:
            self.flush()

    def flush(self) -> None:
        summary = {
            "pid": os.getpid(),
            "frames": self.frames,
            "frames_with_diff": self.frames_with_diff,
            "paths": dict(self.paths.most_common()),
            "examples": {key: self.examples[key] for key in self.paths},
        }
        with open(f"{self.log_path}.{os.getpid()}.json", "wb") as handle:
            handle.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


_AUDITOR: TrainingCaptureAuditor | None = None


def auditor() -> TrainingCaptureAuditor | None:
    global _AUDITOR
    path = os.environ.get("CR_TRAINING_AUDIT_LOG")
    if path and _AUDITOR is None:
        _AUDITOR = TrainingCaptureAuditor(path)
    return _AUDITOR
