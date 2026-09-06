from __future__ import annotations

from typing import Any

import pytest

from native_runner.cr_native_env import NativeClashEnv


class StubNativeClashEnv(NativeClashEnv):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[str] = []

    def _request(self, command: str) -> dict[str, Any]:
        self.requests.append(command)
        if command.startswith("speed "):
            return {
                "ok": True,
                "mode": "native-render",
                "speed": float(command.removeprefix("speed ")),
                "paused": False,
            }
        if command == "pause":
            return {
                "ok": True,
                "mode": "native-render",
                "speed": 1,
                "paused": True,
            }
        if command == "resume":
            return {
                "ok": True,
                "mode": "native-render",
                "speed": 1,
                "paused": False,
            }
        if command.startswith("advance-native "):
            ticks = int(command.removeprefix("advance-native "))
            return {
                "ok": True,
                "mode": "native-render",
                "requestedTicks": ticks,
                "targetTick": 130 + ticks,
                "tick": 130 + ticks,
                "ended": False,
                "paused": True,
            }
        if command == "touch status":
            return {
                "ok": True,
                "mode": "native-render-touch",
                "selectedOwner": 0,
                "selectedHandIndex": 3,
            }
        raise AssertionError(f"unexpected request: {command}")


@pytest.mark.parametrize(
    ("multiplier", "wire_value"),
    [(0.25, "0.25"), (0.5, "0.5"), (1, "1"), (2.0, "2"), (4, "4")],
)
def test_native_render_speed_protocol(multiplier: float, wire_value: str) -> None:
    env = StubNativeClashEnv()

    result = env.set_speed(multiplier)

    assert result["mode"] == "native-render"
    assert result["speed"] == float(multiplier)
    assert env.requests == [f"speed {wire_value}"]


@pytest.mark.parametrize(
    "multiplier",
    [0, 0.1, 0.75, 1.5, 8, -1, True, "2", None],
)
def test_native_render_speed_rejects_unsupported_values(multiplier: object) -> None:
    env = StubNativeClashEnv()

    with pytest.raises(ValueError, match="0.25, 0.5, 1, 2, 4"):
        env.set_speed(multiplier)  # type: ignore[arg-type]

    assert env.requests == []


def test_native_pause_and_resume_forward_native_status() -> None:
    env = StubNativeClashEnv()

    paused = env.pause()
    resumed = env.resume()

    assert paused["paused"] is True
    assert resumed["paused"] is False
    assert env.requests == ["pause", "resume"]


def test_native_render_exact_advance_uses_one_control_receipt() -> None:
    env = StubNativeClashEnv()

    result = env.advance_native_render(5)

    assert result["tick"] == 135
    assert result["paused"] is True
    assert env.requests == ["advance-native 5"]


@pytest.mark.parametrize("ticks", (0, -1, 1_000_001, True, 1.5))
def test_native_render_exact_advance_rejects_invalid_ticks(ticks: object) -> None:
    env = StubNativeClashEnv()

    with pytest.raises((TypeError, ValueError)):
        env.advance_native_render(ticks)  # type: ignore[arg-type]

    assert env.requests == []


def test_native_touch_status_uses_diagnostic_protocol() -> None:
    env = StubNativeClashEnv()

    result = env.touch_status()

    assert result["selectedOwner"] == 0
    assert result["selectedHandIndex"] == 3
    assert env.requests == ["touch status"]
