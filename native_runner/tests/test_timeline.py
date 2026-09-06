from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from native_runner.timeline import (
    GAME_MODE_CLASS_ID,
    discover_game_modes_root,
    discover_logic_root,
    load_game_mode_names,
    load_game_mode_timeline,
    load_timelines,
)


def _write_game_modes(root: Path, *, mode: str, timeline: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "game_modes.csv").write_text(
        "Name,BattleTimeline\n"
        "string,string\n"
        f"{mode},{timeline}\n",
        encoding="utf-8",
    )


def _write_timeline(root: Path, *, timeline: str, starting_elixir: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "battle_timelines.csv").write_text(
        "Name,StartingElixir,SectionLength,SectionType,SectionFlags,"
        "ElixirRateLength,ElixirFullBarMS,ElixirRateVisible,ElixirNotifyChange\n"
        "string,int,int,string,string,int,int,int,bool\n"
        f"{timeline},{starting_elixir},180,REGULATION,,180,28000,1,false\n",
        encoding="utf-8",
    )


def _write_source_pair(
    root: Path,
    *,
    label: str,
    starting_elixir: int,
    compressed: bool = False,
) -> None:
    timeline = f"{label}Timeline"
    _write_game_modes(root, mode=f"{label}Mode", timeline=timeline)
    _write_timeline(root, timeline=timeline, starting_elixir=starting_elixir)
    if compressed:
        sc_compression = pytest.importorskip("sc_compression")
        for name in ("game_modes.csv", "battle_timelines.csv"):
            path = root / name
            path.write_bytes(
                sc_compression.compress(
                    path.read_bytes(),
                    sc_compression.Signatures.LZMA,
                )
            )


def test_workspace_runtime_overlay_pair_has_priority(tmp_path: Path) -> None:
    current = tmp_path / "runtime-update" / "csv_logic"
    captured = (
        tmp_path
        / "captures"
        / "decompressed_assets"
        / "runtime_99.999.99"
        / "runtime-update"
        / "csv_logic"
    )
    release = (
        tmp_path
        / "captures"
        / "decompressed_assets"
        / "nr_99.999.99_release_test"
        / "assets"
        / "csv_logic"
    )
    _write_source_pair(
        current,
        label="Current",
        starting_elixir=7,
        compressed=True,
    )
    _write_source_pair(captured, label="Captured", starting_elixir=6)
    _write_source_pair(release, label="Release", starting_elixir=5)

    assert discover_logic_root(tmp_path) == current.resolve()
    assert discover_game_modes_root(tmp_path) == current.resolve()

    loaded = load_game_mode_timeline(GAME_MODE_CLASS_ID, tmp_path)
    assert loaded.game_mode_name == "CurrentMode"
    assert loaded.timeline.name == "CurrentTimeline"
    assert loaded.timeline.starting_elixir == 7
    assert Path(loaded.game_modes_source).parent == current.resolve()
    assert Path(loaded.timeline.source_path).parent == current.resolve()
    assert loaded.game_modes_sha256 == hashlib.sha256(
        (current / "game_modes.csv").read_bytes()
    ).hexdigest()
    assert loaded.timeline.source_sha256 == hashlib.sha256(
        (current / "battle_timelines.csv").read_bytes()
    ).hexdigest()


def test_incomplete_runtime_overlays_fall_back_to_one_release_pair(
    tmp_path: Path,
) -> None:
    current = tmp_path / "runtime-update" / "csv_logic"
    stale_capture = (
        tmp_path
        / "captures"
        / "decompressed_assets"
        / "runtime_15.535.84"
        / "runtime-update"
        / "csv_logic"
    )
    release = (
        tmp_path
        / "captures"
        / "decompressed_assets"
        / "nr_15.535.13_release_test"
        / "assets"
        / "csv_logic"
    )
    _write_game_modes(current, mode="CurrentOnly", timeline="MissingCurrentTimeline")
    _write_game_modes(
        stale_capture,
        mode="StaleOnly",
        timeline="MissingStaleTimeline",
    )
    _write_source_pair(release, label="Release", starting_elixir=5)

    assert discover_logic_root(tmp_path) == release.resolve()
    assert discover_game_modes_root(tmp_path) == release.resolve()

    loaded = load_game_mode_timeline(GAME_MODE_CLASS_ID, tmp_path)
    assert loaded.game_mode_name == "ReleaseMode"
    assert loaded.timeline.name == "ReleaseTimeline"
    assert Path(loaded.game_modes_source).parent == release.resolve()
    assert Path(loaded.timeline.source_path).parent == release.resolve()


def test_complete_captured_runtime_pair_is_the_first_fallback(tmp_path: Path) -> None:
    captured = (
        tmp_path
        / "captures"
        / "decompressed_assets"
        / "runtime_15.535.84"
        / "runtime-update"
        / "csv_logic"
    )
    release = (
        tmp_path
        / "captures"
        / "decompressed_assets"
        / "nr_15.535.13_release_test"
        / "assets"
        / "csv_logic"
    )
    _write_source_pair(captured, label="Captured", starting_elixir=6)
    _write_source_pair(release, label="Release", starting_elixir=5)

    assert discover_logic_root(tmp_path) == captured.resolve()
    assert discover_game_modes_root(tmp_path) == captured.resolve()

    loaded = load_game_mode_timeline(GAME_MODE_CLASS_ID, tmp_path)
    assert loaded.game_mode_name == "CapturedMode"
    assert loaded.timeline.name == "CapturedTimeline"
    assert Path(loaded.game_modes_source).parent == captured.resolve()
    assert Path(loaded.timeline.source_path).parent == captured.resolve()


def test_explicit_csv_logic_source_remains_authoritative(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit" / "csv_logic"
    _write_source_pair(explicit, label="Explicit", starting_elixir=4)
    _write_source_pair(
        tmp_path / "runtime-update" / "csv_logic",
        label="Workspace",
        starting_elixir=7,
    )

    assert discover_logic_root(explicit) == explicit.resolve()
    assert discover_game_modes_root(explicit) == explicit.resolve()
    assert load_game_mode_names(explicit)[GAME_MODE_CLASS_ID] == (
        "ExplicitMode",
        "ExplicitTimeline",
    )
    assert load_timelines(explicit)["ExplicitTimeline"].starting_elixir == 4
