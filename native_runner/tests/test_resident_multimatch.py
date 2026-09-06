from __future__ import annotations

from typing import Any

import pytest

from native_runner.cr_native_env import ResidentNativeClashEnv, RunnerError


class _RecordingResidentClient(ResidentNativeClashEnv):
    def __init__(self, env_id: int) -> None:
        super().__init__(env_id)
        self.commands: list[str] = []

    def _raw_request(
        self,
        command: str,
        *,
        _read_retry_count: int = 2,
    ) -> dict[str, Any]:
        del _read_retry_count
        self.commands.append(command)
        return {"ok": True, "command": command}


def test_resident_client_prefixes_slot_commands_only() -> None:
    env = _RecordingResidentClient(3)

    assert env.status()["command"] == "env 3 status"
    assert env.step(7)["command"] == "env 3 step 7"
    assert env._request("attest")["command"] == "attest"
    assert (
        env._request("env 3 status", _read_retry_count=1)["command"]
        == "env 3 status"
    )


def test_resident_client_rejects_invalid_slot_and_closes_once() -> None:
    with pytest.raises(ValueError, match="0..15"):
        _RecordingResidentClient(16)

    env = _RecordingResidentClient(2)
    env.close()
    env.close()
    assert env.commands == ["env 2 close"]
    with pytest.raises(RunnerError, match="closed"):
        env.status()
