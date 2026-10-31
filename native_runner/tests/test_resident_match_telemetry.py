"""Resident multi-match slots must observe exactly what a single-match process observes.

Observations are built from telemetry history (retained rings, causal markers,
lifetime tables, per-object caches). When that history was process-global, every
slot switch reset it, so with several resident slots each match only kept its
latest five-tick step and PPO saw different observations than IL and inference.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from native_runner.paths import PACKAGE_ROOT

PROBE = PACKAGE_ROOT / "probe"
TELEMETRY_MODULES = (
    "combat_event_telemetry.inc",
    "visibility_runtime_telemetry.inc",
    "phase_runtime_telemetry.inc",
    "special_movement_runtime_telemetry.inc",
    "action_movement_runtime_telemetry.inc",
    "character_state_runtime_telemetry.inc",
    "remaining_runtime_telemetry.inc",
    "tower_troop_runtime_telemetry.inc",
)
# Storage that carries one match's history from one native step to the next.
PER_MATCH_STORAGE = re.compile(
    r"^(?!struct|using|constexpr|thread_local)(?=[^\s(/])[^(;=]*?"
    r"(RingSlot|EpochRing<|CausalMarker|RuntimeCache|LifetimeEntry|PendingSpawnAttach|PendingActionSpawn"
    r"|EligibilityCacheEntry)\b[^(;]*\bg_\w+",
    re.MULTILINE,
)


def _body(source: str, signature: str) -> str:
    start = source.index(signature)
    return source[start : source.index("\n}\n", start)]


def test_per_match_telemetry_storage_lives_only_in_match_structs() -> None:
    for name in TELEMETRY_MODULES:
        source = (PROBE / name).read_text(encoding="utf-8")
        assert PER_MATCH_STORAGE.findall(source) == [], name
        assert re.search(r"^std::atomic<std::uint64_t> g_\w*next_sequence", source, re.MULTILINE) is None, name


def test_bind_match_telemetry_rebinds_every_module() -> None:
    probe = (PROBE / "cr_replay_probe.cpp").read_text(encoding="utf-8")
    members = re.findall(r"^  (\w+)MatchTelemetry (\w+);$", _body(probe, "struct MatchTelemetry {"), re.MULTILINE)
    assert len(members) == len(TELEMETRY_MODULES)
    binder = _body(probe, "void bind_match_telemetry(MatchTelemetry *telemetry) {")
    for _, member in members:
        assert f"g_{member}_match =" in binder
        assert f"&telemetry->{member}" in binder
        assert f"&g_{member}_default_match" in binder


def test_resident_slot_switch_rebinds_instead_of_resetting_history() -> None:
    resident = (PROBE / "resident_multimatch.inc").read_text(encoding="utf-8")
    bind = _body(resident, "bool resident_bind_slot(std::int32_t slot_id) {")
    assert "bind_match_telemetry(g_resident_slot_telemetry[slot_id].get());" in bind
    for reset in ("begin_", "end_combat_event_epoch", "restore_tower_troop_runtime_snapshot", "telemetry_ready = false"):
        assert reset not in bind
    construct = _body(resident, "bool resident_construct_slot(")
    assert construct.index("bind_match_telemetry(g_resident_slot_telemetry[slot_id].get());") < construct.index(
        "game_state_load_hook(manager"
    )
    # Compact deltas follow each slot's own rings across switches.
    assert "g_resident_training_family_cursors[kResidentMaxSlots]" in resident
    assert "g_resident_training_family_cursors[g_resident_bound_slot]" in resident


def test_remaining_runtime_epoch_refresh_only_touches_the_bound_match(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for the remaining-runtime host test")
    executable = tmp_path / "test_remaining_runtime_telemetry_compile.exe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            str(PROBE / "test_remaining_runtime_telemetry_compile.cpp"),
            f"-I{PROBE}",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=PACKAGE_ROOT,
    )
    subprocess.run([str(executable)], check=True)
