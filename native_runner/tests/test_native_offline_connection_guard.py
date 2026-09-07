from __future__ import annotations

from pathlib import Path


PROBE = (
    Path(__file__).resolve().parents[1]
    / "probe"
    / "cr_replay_probe.cpp"
)


def test_offline_connection_guard_survives_runner_mode_changes() -> None:
    source = PROBE.read_text(encoding="utf-8")

    assert "g_offline_control_session_active{false}" in source
    assert source.count(
        "g_offline_control_session_active.store(true"
    ) >= 2
    assert (
        "if (g_offline_control_session_active.load(std::memory_order_acquire) &&"
        in source
    )
    assert '\\"offlineConnectionErrorsSuppressed\\":%llu' in source
