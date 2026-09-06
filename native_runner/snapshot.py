"""Serialized action operations shared by collected and training replays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING
from .contracts import ActionV1

if TYPE_CHECKING:
    from .battle_env import ReplayOperationV1


class SnapshotError(RuntimeError):
    """Raised when snapshot identity or deterministic replay diverges."""


@dataclass(frozen=True, slots=True)
class SnapshotOperationV1:
    actions: tuple[ActionV1, ...]
    advance_ticks: int
    requested_advance_ticks: int
    start_native_tick: int
    end_native_tick: int

    @classmethod
    def from_runtime(cls, operation: "ReplayOperationV1") -> "SnapshotOperationV1":
        return cls(
            actions=tuple(operation.actions),
            advance_ticks=int(operation.advance_ticks),
            requested_advance_ticks=int(
                operation.requested_advance_ticks
                if operation.requested_advance_ticks is not None
                else operation.advance_ticks
            ),
            start_native_tick=int(operation.start_native_tick),
            end_native_tick=int(operation.end_native_tick),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [item.to_dict() for item in self.actions],
            "advance_ticks": self.advance_ticks,
            "requested_advance_ticks": self.requested_advance_ticks,
            "start_native_tick": self.start_native_tick,
            "end_native_tick": self.end_native_tick,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SnapshotOperationV1":
        actions = tuple(ActionV1.from_mapping(item) for item in value.get("actions", ()))
        if {item.owner for item in actions} != {0, 1}:
            raise SnapshotError("snapshot operation must contain both owners")
        return cls(
            actions=actions,
            advance_ticks=int(value.get("advance_ticks", value["end_native_tick"] - value["start_native_tick"])),
            requested_advance_ticks=int(value["requested_advance_ticks"]),
            start_native_tick=int(value["start_native_tick"]),
            end_native_tick=int(value["end_native_tick"]),
        )
