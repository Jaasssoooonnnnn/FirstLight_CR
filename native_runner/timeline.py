"""Patch-derived game-mode and battle-timeline registry.

The Supercell CSV format uses blank-name continuation rows for array fields.
This loader preserves those rows and maps the record index to the game's
72,000,000-based global game-mode IDs instead of hard-coding elixir timing.
"""

from __future__ import annotations

from .paths import WORKSPACE_ROOT

import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Iterable

try:  # Runtime CSV assets may retain Supercell compression containers.
    from sc_compression import decompress as _sc_decompress
except ImportError:  # pragma: no cover - plaintext-only minimal hosts.
    _sc_decompress = None


GAME_MODE_CLASS_ID = 72_000_000
TICKS_PER_SECOND = 20
TICK_MS = 50


class TimelineError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TimelineSectionV1:
    length_seconds: int
    section_type: str
    flags: tuple[str, ...] = ()

    @property
    def length_ticks(self) -> int:
        return self.length_seconds * TICKS_PER_SECOND


@dataclass(frozen=True, slots=True)
class ElixirRateV1:
    length_seconds: int
    full_bar_ms: int
    visible_rate: int | None = None
    notify_change: bool = False

    @property
    def length_ticks(self) -> int:
        return self.length_seconds * TICKS_PER_SECOND

    @property
    def raw_per_tick(self) -> float:
        return 100_000.0 * TICK_MS / self.full_bar_ms


@dataclass(frozen=True, slots=True)
class BattleTimelineV1:
    name: str
    starting_elixir: float
    sections: tuple[TimelineSectionV1, ...]
    elixir_rates: tuple[ElixirRateV1, ...]
    source_path: str
    source_sha256: str
    version: str = "battle-timeline.v1"

    def __post_init__(self) -> None:
        if not self.name or not self.sections or not self.elixir_rates:
            raise TimelineError("timeline requires a name, sections, and elixir rates")
        if not 0 <= self.starting_elixir <= 10:
            raise TimelineError("starting elixir must be in [0,10]")

    @property
    def maximum_tick(self) -> int:
        return sum(item.length_ticks for item in self.sections)

    def phase(self, tick: int) -> tuple[str, float]:
        current = max(0, int(tick))
        cursor = 0
        phase = self.sections[-1].section_type.lower()
        for section in self.sections:
            end = cursor + section.length_ticks
            if current < end:
                phase = section.section_type.lower()
                break
            cursor = end
        else:
            phase = "tiebreak"
        rate = self.elixir_rate(current)
        base_visible = self.elixir_rates[0].visible_rate
        if base_visible and rate.visible_rate:
            multiplier = float(rate.visible_rate) / float(base_visible)
        else:
            multiplier = rate.raw_per_tick / self.elixir_rates[0].raw_per_tick
        return phase, multiplier

    def elixir_rate(self, tick: int) -> ElixirRateV1:
        current = max(0, int(tick))
        cursor = 0
        for rate in self.elixir_rates:
            end = cursor + rate.length_ticks
            if current < end:
                return rate
            cursor = end
        return self.elixir_rates[-1]

    def next_elixir_boundary(self, tick: int) -> int | None:
        current = max(0, int(tick))
        cursor = 0
        for rate in self.elixir_rates:
            cursor += rate.length_ticks
            if current < cursor:
                return cursor
        return None

    def generated_raw(self, start_tick: int, end_tick: int) -> float:
        cursor = max(0, int(start_tick))
        stop = max(cursor, int(end_tick))
        total = 0.0
        while cursor < stop:
            boundary = self.next_elixir_boundary(cursor)
            segment_end = stop if boundary is None else min(stop, boundary)
            total += (segment_end - cursor) * self.elixir_rate(cursor).raw_per_tick
            cursor = segment_end
        return total

    def ticks_for_raw(self, tick: int, deficit_raw: int) -> int:
        if deficit_raw <= 0:
            return 1
        cursor = max(0, int(tick))
        elapsed = 0
        remaining = float(deficit_raw)
        while True:
            rate = self.elixir_rate(cursor).raw_per_tick
            boundary = self.next_elixir_boundary(cursor)
            required = int(math.ceil(remaining / rate))
            if boundary is None or required <= boundary - cursor:
                return max(1, elapsed + required)
            available = boundary - cursor
            remaining -= available * rate
            elapsed += available
            cursor = boundary


@dataclass(frozen=True, slots=True)
class GameModeTimelineV1:
    game_mode_id: int
    game_mode_name: str
    timeline: BattleTimelineV1
    game_modes_source: str
    game_modes_sha256: str
    version: str = "game-mode-timeline.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_csv(value: bytes) -> bool:
    try:
        prefix = value[:256]
        prefix.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if b"\x00" in prefix:
        return False
    return prefix.lstrip(b"\xef\xbb\xbf\r\n\t ").startswith((b'"Name",', b"Name,"))


@lru_cache(maxsize=32)
def _asset_bytes_cached(name: str, size: int, modified_ns: int) -> bytes | None:
    del size, modified_ns  # Values invalidate the path-based cache key.
    try:
        raw = Path(name).read_bytes()
    except OSError:
        return None
    if _looks_like_csv(raw):
        return raw
    if _sc_decompress is None:
        return None
    try:
        plain, _signature, _version = _sc_decompress(raw)
    except Exception:
        return None
    return plain if _looks_like_csv(plain) else None


def _asset_bytes(path: Path) -> bytes | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file() or stat.st_size == 0:
        return None
    return _asset_bytes_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _has_timeline_source_pair(root: Path) -> bool:
    return _asset_bytes(root / "battle_timelines.csv") is not None and _asset_bytes(root / "game_modes.csv") is not None


def _natural_source_version(path: Path) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", path.as_posix()))


def discover_logic_root(workspace_root: str | Path | None = None) -> Path:
    source = Path(workspace_root or WORKSPACE_ROOT).resolve()
    # Callers may pass a decoded csv_logic directory directly.  Keep that
    # explicit source authoritative instead of treating it as a workspace.
    if workspace_root is not None and _has_timeline_source_pair(source):
        return source

    workspace = source
    runtime = workspace / "runtime-update" / "csv_logic"
    if _has_timeline_source_pair(runtime):
        return runtime

    runtime_candidates = sorted(
        (
            item / "runtime-update" / "csv_logic"
            for item in (workspace / "captures" / "decompressed_assets").glob("runtime_*")
        ),
        key=_natural_source_version,
        reverse=True,
    )
    for candidate in runtime_candidates:
        if _has_timeline_source_pair(candidate):
            return candidate

    candidates = sorted(
        (
            item / "assets" / "csv_logic"
            for item in (workspace / "captures" / "decompressed_assets").glob("nr_*_release_*")
        ),
        key=_natural_source_version,
        reverse=True,
    )
    for candidate in candidates:
        if _has_timeline_source_pair(candidate):
            return candidate
    raise FileNotFoundError("no decoded battle_timelines.csv + game_modes.csv pair found")


def discover_game_modes_root(workspace_root: str | Path | None = None) -> Path:
    """Use the game-mode map paired with the effective battle timelines."""

    return discover_logic_root(workspace_root)


def _rows(path: Path) -> Iterable[dict[str, str]]:
    plain = _asset_bytes(path)
    if plain is None:
        raise TimelineError(f"timeline asset is not readable CSV: {path}")
    with io.StringIO(plain.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        # The first body row contains Supercell column types.
        next(reader, None)
        yield from reader


def load_timelines(logic_root: str | Path) -> dict[str, BattleTimelineV1]:
    root = Path(logic_root).resolve()
    path = root / "battle_timelines.csv"
    result: dict[str, BattleTimelineV1] = {}
    current_name: str | None = None
    starting = 0.0
    sections: list[TimelineSectionV1] = []
    rates: list[ElixirRateV1] = []

    def publish() -> None:
        nonlocal current_name, starting, sections, rates
        if current_name is not None:
            result[current_name] = BattleTimelineV1(
                name=current_name,
                starting_elixir=starting,
                sections=tuple(sections),
                elixir_rates=tuple(rates),
                source_path=path.as_posix(),
                source_sha256=_sha256(path),
            )

    for row in _rows(path):
        name = (row.get("Name") or "").strip()
        if name:
            publish()
            current_name = name
            starting = float(row.get("StartingElixir") or 0)
            sections = []
            rates = []
        if current_name is None:
            continue
        section_length = (row.get("SectionLength") or "").strip()
        section_type = (row.get("SectionType") or "").strip()
        if section_length and section_type:
            sections.append(
                TimelineSectionV1(
                    int(section_length), section_type, tuple(filter(None, (row.get("SectionFlags") or "").split(";")))
                )
            )
        rate_length = (row.get("ElixirRateLength") or "").strip()
        full_bar = (row.get("ElixirFullBarMS") or "").strip()
        if rate_length and full_bar:
            rates.append(
                ElixirRateV1(
                    int(rate_length),
                    int(full_bar),
                    int(row["ElixirRateVisible"]) if row.get("ElixirRateVisible") else None,
                    (row.get("ElixirNotifyChange") or "").lower() == "true",
                )
            )
    publish()
    return result


def load_game_mode_names(logic_root: str | Path) -> dict[int, tuple[str, str]]:
    root = Path(logic_root).resolve()
    path = root / "game_modes.csv"
    result: dict[int, tuple[str, str]] = {}
    record_index = 0
    for row in _rows(path):
        name = (row.get("Name") or "").strip()
        if not name:
            continue
        timeline = (row.get("BattleTimeline") or "").strip()
        if not timeline:
            raise TimelineError(f"game mode {name} has no BattleTimeline")
        result[GAME_MODE_CLASS_ID + record_index] = (name, timeline)
        record_index += 1
    return result


@lru_cache(maxsize=8)
def load_game_mode_timeline(game_mode_id: int, workspace_root: str | Path | None = None) -> GameModeTimelineV1:
    root = None
    if workspace_root is not None:
        try:
            root = discover_logic_root(workspace_root)
        except FileNotFoundError:
            pass
    if root is None:
        from .competitive_data import load_competitive_data

        facts = load_competitive_data("runtime")
        try:
            mode_name, timeline_name = facts["game_modes"][str(game_mode_id)]
        except KeyError as error:
            raise TimelineError(f"unknown game mode ID {game_mode_id}") from error
        value = dict(facts["timelines"][timeline_name])
        value["sections"] = tuple(TimelineSectionV1(**dict(item)) for item in value["sections"])
        value["elixir_rates"] = tuple(ElixirRateV1(**dict(item)) for item in value["elixir_rates"])
        return GameModeTimelineV1(
            int(game_mode_id), mode_name, BattleTimelineV1(**value),
            facts["game_modes_source"], facts["game_modes_sha256"],
        )
    # Discovery selects one complete source pair.  Keep both identity-bearing
    # files on that effective root even if workspace contents change mid-load.
    game_modes_root = root
    game_modes_path = game_modes_root / "game_modes.csv"
    modes = load_game_mode_names(game_modes_root)
    try:
        mode_name, timeline_name = modes[int(game_mode_id)]
    except KeyError as error:
        raise TimelineError(f"unknown game mode ID {game_mode_id}") from error
    timelines = load_timelines(root)
    try:
        timeline = timelines[timeline_name]
    except KeyError as error:
        raise TimelineError(f"game mode {mode_name} references missing timeline {timeline_name}") from error
    return GameModeTimelineV1(
        game_mode_id=int(game_mode_id),
        game_mode_name=mode_name,
        timeline=timeline,
        game_modes_source=game_modes_path.as_posix(),
        game_modes_sha256=_sha256(game_modes_path),
    )
