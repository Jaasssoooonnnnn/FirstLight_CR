#!/usr/bin/env python3
"""Clickable Windows overlay for the stock Null's Royale native renderer.

This module does not redraw the battle.  It follows the MuMu
configured offline window, places a nearly invisible mouse-capture layer over
the 9:16 Android viewport, and draws feedback in a separate click-through
layer.  Clicking a hand then the arena, or dragging a hand onto the arena,
submits a real :class:`HandAction` to the native runner.

The companion panel is deliberately separate from ``viewer.py`` and the HTML
viewer.  It talks only to the loopback native control port and never uses the
normal MuMu/ADB endpoint.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any, Callable, Iterable, Mapping

from .arena import OccupiedFootprintV1, card_placement_legal_world, native_building_footprint
from .building_placement_profiles import building_placement_profile
from .card_specs import build_card_catalog
from .cr_native_env import AbilityAction, COMMAND_CONSUMPTION_STEPS, HandAction, NativeClashEnv, RunnerError
from .local_config import setting
from .normal_form_evidence import NORMAL_MODE_HERO_FORM_TO_BASE_CARD


TARGET_WINDOW_TITLE = setting("CR_VM_NAME")
ALLOWED_WINDOW_TITLES = (TARGET_WINDOW_TITLE,) if TARGET_WINDOW_TITLE else ()
TARGET_PROCESS_NAME = setting("CR_OVERLAY_PROCESS", "MuMuNxDevice.exe")
DEFAULT_RUNNER_HOST = setting("CR_CONTROL_HOST", "127.0.0.1")
DEFAULT_RUNNER_PORT = int(setting("CR_CONTROL_PORT", "26789"))

# The stock renderer is a 1080x1920 portrait surface.  These measurements are
# from captures/native_render_offline_final.png.  They are used only as host
# UI hit regions; gameplay still goes through the structured native API.
SOURCE_WIDTH = 1080
SOURCE_HEIGHT = 1920
CARD_CENTERS_X = (211, 354, 497, 640)
TOP_CARD_CENTER_Y = 170
BOTTOM_CARD_CENTER_Y = 1749
CARD_HALF_WIDTH = 67
CARD_HALF_HEIGHT = 90
# The visible arena decorations extend beyond the logical 18x32 placement
# grid.  Keep a generous click catchment, but invert the renderer projection
# against the measured grid boundaries.  Using the decorative bounds for both
# jobs made the outermost columns unreachable and compressed the back row.
ARENA_HIT_SOURCE_RECT = (54, 282, 1026, 1666)
WORLD_PROJECTION_SOURCE_RECT = (111, 430, 967, 1620)
WORLD_WIDTH = 18_000
WORLD_HEIGHT = 32_000
CELL_UNITS = 1_000
DEPLOY_MIN_X = 500
DEPLOY_MAX_X = 17_500
DEPLOY_MIN_Y = 500
DEPLOY_MAX_Y = 31_500
SPEED_CHOICES = (0.25, 0.5, 1.0, 2.0, 4.0)


def native_speed(payload: Any) -> float | None:
    """Return a finite native-render speed without trusting external JSON."""
    if not isinstance(payload, dict):
        return None
    try:
        value = float(payload.get("speed"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class OverlayError(RuntimeError):
    """Raised when the isolated native-render overlay cannot attach safely."""


@dataclass(frozen=True, slots=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError(f"invalid rectangle: {self}")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in ("left", "top", "right", "bottom", "width", "height")}


@dataclass(frozen=True, slots=True)
class CardHit:
    owner: int
    visual_slot: int
    hand_index: int
    source_rect: Rect


@dataclass(frozen=True, slots=True)
class Selection:
    owner: int
    visual_slot: int
    hand_index: int
    card_id: int
    cost: int | None
    name: str


@dataclass(frozen=True, slots=True)
class PlayResult:
    outcome: str
    selection: Selection
    world_x: int
    world_y: int
    detail: str


@dataclass(frozen=True, slots=True)
class AbilityControl:
    """Current presentation and exact activation identity for one owner."""

    owner: int
    text: str
    ability_name: str | None = None
    source_entity_key: tuple[int, int, int] | None = None

    @property
    def ready(self) -> bool:
        return self.source_entity_key is not None


def largest_portrait_viewport(container: Rect) -> Rect:
    """Fit a bottom-anchored 9:16 viewport inside a MuMu client rectangle."""

    target_aspect = SOURCE_WIDTH / SOURCE_HEIGHT
    if container.width / container.height > target_aspect:
        height = container.height
        width = max(1, int(round(height * target_aspect)))
        left = container.left + (container.width - width) // 2
        return Rect(left, container.top, left + width, container.bottom)
    width = container.width
    height = max(1, int(round(width / target_aspect)))
    top = container.bottom - height
    return Rect(container.left, top, container.right, container.bottom)


def _source_card_rect(center_x: int, center_y: int) -> Rect:
    return Rect(
        center_x - CARD_HALF_WIDTH, center_y - CARD_HALF_HEIGHT, center_x + CARD_HALF_WIDTH, center_y + CARD_HALF_HEIGHT
    )


def all_card_hits() -> tuple[CardHit, ...]:
    """Return the eight stock spectator HUD card hit regions.

    The upper hand is drawn in reverse selector order by the stock replay HUD;
    the lower hand is drawn in ordinary selector order.  Live observation of
    both default decks confirms this mapping.
    """

    return tuple(
        CardHit(owner, slot, 3 - slot if owner == 0 else slot, _source_card_rect(center_x, center_y))
        for slot, center_x in enumerate(CARD_CENTERS_X)
        for owner, center_y in enumerate((TOP_CARD_CENTER_Y, BOTTOM_CARD_CENTER_Y))
    )


CARD_HITS = all_card_hits()
ARENA_RECT = Rect(*ARENA_HIT_SOURCE_RECT)
WORLD_PROJECTION_RECT = Rect(*WORLD_PROJECTION_SOURCE_RECT)


def model_hud_mask_rects(human_owner: int) -> tuple[Rect, Rect]:
    """Cover the opposing hand/next card and elixir without hiding the arena clock."""

    if human_owner == 1:
        return Rect(0, 0, SOURCE_WIDTH, 85), Rect(0, 80, 715, 270)
    if human_owner == 0:
        return Rect(0, 1830, SOURCE_WIDTH, SOURCE_HEIGHT), Rect(0, 1650, 715, 1840)
    raise ValueError("human owner must be 0 or 1")


def viewport_to_source(viewport: Rect, local_x: float, local_y: float) -> tuple[float, float]:
    return (local_x / viewport.width * SOURCE_WIDTH, local_y / viewport.height * SOURCE_HEIGHT)


def source_to_viewport(viewport: Rect, source_x: float, source_y: float) -> tuple[float, float]:
    return (source_x / SOURCE_WIDTH * viewport.width, source_y / SOURCE_HEIGHT * viewport.height)


def source_rect_to_viewport(viewport: Rect, source_rect: Rect) -> Rect:
    left, top = source_to_viewport(viewport, source_rect.left, source_rect.top)
    right, bottom = source_to_viewport(viewport, source_rect.right, source_rect.bottom)
    return Rect(round(left), round(top), round(right), round(bottom))


def hit_card_source(source_x: float, source_y: float) -> CardHit | None:
    return next((hit for hit in CARD_HITS if hit.source_rect.contains(source_x, source_y)), None)


def arena_source_to_world(source_x: float, source_y: float) -> tuple[int, int]:
    """Map a stock-renderer arena pixel to the nearest native grid cell.

    The stock renderer shows owner 0 at the top and owner 1 at the bottom.
    Native observations use the same Y direction: owner 0's towers are near
    Y=3000/6500 and owner 1's near Y=29000/25500.  Screen and native Y must
    therefore increase together.  Inverting this axis sends every placement
    into the opponent's half, where the engine clamps it to the river line.

    ``ARENA_RECT`` is the forgiving mouse catchment around the visible arena;
    ``WORLD_PROJECTION_RECT`` is the measured 18x32 logical grid.  Keeping the
    two separate is important: the stock scene draws side walls, tower staging
    space and HUD decoration outside the actual grid.  Clicks in that padding
    clamp to the nearest outer cell instead of being ignored.

    Native replay commands use 1000-unit cell centers (500..17500 and
    500..31500).  Snapping here matches stock placement, avoids edge rejection
    from arbitrary continuous values, and makes the final row behind either
    king tower directly reachable.
    """

    if not ARENA_RECT.contains(source_x, source_y):
        raise ValueError("point is outside the calibrated native arena")

    cells = []
    for source, source_minimum, source_extent, world_extent in (
        (source_x, WORLD_PROJECTION_RECT.left, WORLD_PROJECTION_RECT.width, WORLD_WIDTH),
        (source_y, WORLD_PROJECTION_RECT.top, WORLD_PROJECTION_RECT.height, WORLD_HEIGHT),
    ):
        raw_world = (source - source_minimum) / source_extent * world_extent
        cell_count = world_extent // CELL_UNITS
        cell_index = math.floor(raw_world / CELL_UNITS)
        cell_index = min(cell_count - 1, max(0, cell_index))
        cells.append(cell_index * CELL_UNITS + CELL_UNITS // 2)
    return cells[0], cells[1]


def nearest_legal_placement(
    observation: Mapping[str, Any], card: Mapping[str, Any], owner: int,
    requested_x: int, requested_y: int, card_specs: Mapping[int, Any],
) -> tuple[int, int]:
    """Snap a manual click to a legal tile while preserving its tower side."""

    from .battle_env import TOWER_LAYOUT

    objects = observation.get("objects")
    if not isinstance(objects, list) or observation.get("truncated"):
        return requested_x, requested_y
    visible_id = int(card["cardId"])
    command_id = int(card.get("commandCardId") or visible_id)
    effective_id = NORMAL_MODE_HERO_FORM_TO_BASE_CARD.get(command_id, command_id)
    spec = card_specs.get(effective_id)
    if spec is None and visible_id != 28_000_006:  # Mirror must use its effective card.
        spec = card_specs.get(visible_id)
    if spec is None:
        return requested_x, requested_y
    raw_form = card.get("formCode")
    if raw_form is None and card.get("cardParameter") is not None:
        raw_form = int(card["cardParameter"]) & 0xF
    form_code = int(raw_form or 0)
    form = "evolution" if form_code == 1 else "base"

    tower_layout = {(tower_owner, x, y): (kind, 4 if kind == "king" else 3)
                    for _id, tower_owner, kind, x, y in TOWER_LAYOUT}
    live_towers: set[tuple[int, int, int]] = set()
    tower_footprints: list[OccupiedFootprintV1] = []
    own_tower_footprints: list[OccupiedFootprintV1] = []
    building_footprints: list[OccupiedFootprintV1] = []
    for item in objects:
        if item.get("null") or float(item.get("hp", 0) or 0) <= 0:
            continue
        position = (int(item.get("owner", -1)), int(item.get("x", -1)), int(item.get("y", -1)))
        tower = tower_layout.get(position)
        if tower is not None and float(item.get("maxHp", 0) or 0) > 0:
            kind, size = tower
            live_towers.add(position)
            tower_footprint = OccupiedFootprintV1(position[1], position[2], size, size, f"tower:{kind}")
            tower_footprints.append(tower_footprint)
            if position[0] == owner:
                own_tower_footprints.append(tower_footprint)
            continue
        object_id = int(item.get("cardId", 0) or 0)
        object_spec = card_specs.get(NORMAL_MODE_HERO_FORM_TO_BASE_CARD.get(object_id, object_id))
        if object_spec is None or object_spec.kind.value != "building":
            continue
        profile = building_placement_profile(object_spec.card_id)
        if profile is not None and profile.deploy_target_ready and not profile.stationary_collision_rectangle:
            continue
        footprint = native_building_footprint(object_spec)
        if footprint is not None:
            building_footprints.append(OccupiedFootprintV1(
                position[1], position[2], footprint.width_tiles, footprint.height_tiles, f"building:{object_id}"
            ))
    destroyed_lanes = tuple(
        kind.removeprefix("princess_")
        for _id, tower_owner, kind, x, y in TOWER_LAYOUT
        if tower_owner != owner and kind.startswith("princess_") and (tower_owner, x, y) not in live_towers
    )
    footprint = native_building_footprint(spec, form=form) if spec.kind.value == "building" else None
    offset_x, offset_y = footprint.model_subcell_offset if footprint is not None else (0.0, 0.0)
    sign = 1 if owner == 0 else -1
    own_zone = spec.target_schema.placement_mask_key in {"own_deployment_zone", "own_territory", "own_territory_area"}
    across_river = own_zone and (
        (owner == 0 and requested_y >= 17 * CELL_UNITS)
        or (owner == 1 and requested_y < 15 * CELL_UNITS)
    )
    clicked_tower = next(
        (tower for tower in own_tower_footprints
         if tower.source.startswith("tower:")
         and abs(requested_x - tower.center_x_units) * 2 < tower.width_tiles * CELL_UNITS
         and abs(requested_y - tower.center_y_units) * 2 < tower.height_tiles * CELL_UNITS),
        None,
    )

    def placement_order(position: tuple[int, int]) -> tuple[int, int, int, int, int]:
        dx, dy = position[0] - requested_x, position[1] - requested_y
        if across_river:
            return abs(dx), abs(dy), 0, position[1], position[0]
        distance = dx * dx + dy * dy
        if clicked_tower is not None:
            center = clicked_tower.center_y_units
            half_height = clicked_tower.height_tiles * CELL_UNITS / 2
            behind = requested_y >= center if owner == 1 else requested_y <= center
            preferred = (
                position[1] >= center + half_height if owner == 1 else position[1] <= center - half_height
            ) if behind else (
                position[1] <= center - half_height if owner == 1 else position[1] >= center + half_height
            )
            return 0 if preferred else 1, distance, abs(dx), position[1], position[0]
        return distance, abs(dx), 0, position[1], position[0]

    candidates = sorted(
        (
            (x * CELL_UNITS + CELL_UNITS // 2 + round(offset_x * sign * CELL_UNITS),
             y * CELL_UNITS + CELL_UNITS // 2 + round(offset_y * sign * CELL_UNITS))
            for y in range(WORLD_HEIGHT // CELL_UNITS)
            for x in range(WORLD_WIDTH // CELL_UNITS)
        ),
        key=placement_order,
    )
    for position in candidates:
        if card_placement_legal_world(
            spec, owner, position, form=form, destroyed_enemy_princess_lanes=destroyed_lanes,
            active_tower_footprints=tower_footprints, occupied_building_footprints=building_footprints,
        ):
            return position
    return requested_x, requested_y


def _load_card_names() -> dict[int, str]:
    from .card_names import CARD_NAMES

    return {card_id: f"{chinese} / {english}" for card_id, (chinese, english) in CARD_NAMES.items()}


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    GA_ROOT = 2
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_NOACTIVATE = 0x08000000
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOACTIVATE = 0x0010
    HWND_TOPMOST = -1

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _window_text(hwnd: int) -> str:
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        return buffer.value

    def _process_name(hwnd: int) -> str:
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id.value)
        if not process:
            return ""
        try:
            capacity = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(capacity.value)
            if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(capacity)):
                return ""
            return os.path.basename(buffer.value)
        finally:
            kernel32.CloseHandle(process)

    def _screen_rect(hwnd: int) -> Rect | None:
        raw = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(raw)):
            return None
        if raw.right <= raw.left or raw.bottom <= raw.top:
            return None
        return Rect(raw.left, raw.top, raw.right, raw.bottom)

    def _client_screen_rect(hwnd: int) -> Rect | None:
        raw = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(raw)):
            return None
        top_left = wintypes.POINT(raw.left, raw.top)
        bottom_right = wintypes.POINT(raw.right, raw.bottom)
        if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
            return None
        if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
            return None
        if bottom_right.x <= top_left.x or bottom_right.y <= top_left.y:
            return None
        return Rect(top_left.x, top_left.y, bottom_right.x, bottom_right.y)

    def _enum_windows(callback: Callable[[int], None], parent: int | None = None) -> None:
        def bridge(hwnd: int, _parameter: int) -> bool:
            callback(int(hwnd))
            return True

        native_callback = EnumWindowsProc(bridge)
        if parent is None:
            user32.EnumWindows(native_callback, 0)
        else:
            user32.EnumChildWindows(parent, native_callback, 0)

    def _set_dpi_awareness() -> None:
        # PER_MONITOR_AWARE_V2 keeps Tk geometry and Win32 rectangles in the
        # same physical-pixel coordinate space. Ignore "already set" errors.
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            try:
                user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass

    def _root_hwnd(widget: Any) -> int:
        widget.update_idletasks()
        return int(user32.GetAncestor(int(widget.winfo_id()), GA_ROOT))

    def _add_extended_styles(widget: Any, styles: int) -> int:
        hwnd = _root_hwnd(widget)
        current = int(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE))
        user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, current | styles)
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        return hwnd


else:

    def _set_dpi_awareness() -> None:
        return


class MumuWindowLocator:
    """Find only the configured offline MuMu render window."""

    def __init__(self, window_title: str = TARGET_WINDOW_TITLE) -> None:
        if window_title not in ALLOWED_WINDOW_TITLES:
            raise OverlayError("native overlay requires the offline CR_VM_NAME configured in .env")
        self.window_title = window_title

    def find_window(self) -> int | None:
        if os.name != "nt":
            raise OverlayError("native overlay requires Windows")
        candidates: list[int] = []

        def inspect(hwnd: int) -> None:
            if not user32.IsWindowVisible(hwnd):
                return
            if _window_text(hwnd) != self.window_title:
                return
            if _process_name(hwnd).casefold() != TARGET_PROCESS_NAME.casefold():
                return
            candidates.append(hwnd)

        _enum_windows(inspect)
        if len(candidates) > 1:
            raise OverlayError(f"multiple exact {self.window_title!r} MuMu windows are visible")
        return candidates[0] if candidates else None

    def viewport(self, hwnd: int) -> tuple[Rect, str]:
        if os.name != "nt":
            raise OverlayError("native overlay requires Windows")
        if user32.IsIconic(hwnd):
            raise OverlayError("configured VM window is minimized")
        client = _client_screen_rect(hwnd)
        if client is None:
            raise OverlayError("cannot read configured VM client rectangle")

        child_rects: list[Rect] = []

        def inspect(child: int) -> None:
            if not user32.IsWindowVisible(child):
                return
            rect = _screen_rect(child)
            if rect is None or rect.width < 200 or rect.height < 300:
                return
            aspect_error = abs(rect.width / rect.height - SOURCE_WIDTH / SOURCE_HEIGHT)
            if aspect_error <= 0.02 and rect.area <= client.area * 1.02:
                child_rects.append(rect)

        _enum_windows(inspect, hwnd)
        if child_rects:
            return max(child_rects, key=lambda item: item.area), "portrait-child"
        return largest_portrait_viewport(client), "client-aspect-fit"

    @staticmethod
    def foreground_matches(hwnd: int, *overlay_hwnds: int | None) -> bool:
        foreground = int(user32.GetForegroundWindow())
        foreground_root = int(user32.GetAncestor(foreground, GA_ROOT))
        target_root = int(user32.GetAncestor(hwnd, GA_ROOT))
        return foreground_root in {target_root, *(item for item in overlay_hwnds if item is not None)}


def _player(observation: dict[str, Any], owner: int) -> dict[str, Any]:
    try:
        return next(item for item in observation["players"] if int(item["owner"]) == owner)
    except (KeyError, StopIteration) as error:
        raise RunnerError(f"native observation has no owner {owner}") from error


def _card_in_hand(player: dict[str, Any], hand_index: int) -> dict[str, Any]:
    try:
        return next(item for item in player["hand"] if int(item["handIndex"]) == hand_index)
    except (KeyError, StopIteration) as error:
        raise RunnerError(f"native hand has no selector index {hand_index}") from error


def ability_control(rich_observation: dict[str, Any] | None, owner: int) -> AbilityControl:
    """Resolve one fail-closed ability button from rich native telemetry."""

    side = "上方" if owner == 0 else "下方"
    if rich_observation is None:
        return AbilityControl(owner, f"{side}技能：遥测不可用")
    try:
        runtime = _player(rich_observation, owner).get("abilityRuntime")
    except RunnerError:
        return AbilityControl(owner, f"{side}技能：遥测不可用")
    if not isinstance(runtime, list) or not runtime:
        return AbilityControl(owner, f"{side}技能：未上场")

    ready: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
    for item in runtime:
        if not isinstance(item, dict):
            continue
        keys = item.get("championEntityKeys")
        if (
            item.get("available") is True
            and item.get("buttonStateLabel") in {"Ready", "LimitedAvailability"}
            and isinstance(keys, list)
            and len(keys) == 1
            and isinstance(keys[0], list)
            and len(keys[0]) == 3
        ):
            key = tuple(int(value) for value in keys[0])
            if key[0] == owner:
                ready.append((item, key))
    if len(ready) == 1:
        item, key = ready[0]
        name = str(item.get("actionDataName") or "主动技能")
        return AbilityControl(owner, f"{side}技能：可用", name, key)
    if len(ready) > 1:
        return AbilityControl(owner, f"{side}技能：来源不唯一")

    item = runtime[0] if isinstance(runtime[0], dict) else {}
    name = str(item.get("actionDataName") or "主动技能")
    keys = item.get("championEntityKeys")
    if not isinstance(keys, list) or not keys:
        return AbilityControl(owner, f"{side}技能：未上场", name)
    cooldown = item.get("remainingCooldownMs")
    if isinstance(cooldown, int) and cooldown > 0:
        return AbilityControl(owner, f"{side}技能：冷却 {cooldown / 1000:.1f}s", name)
    state = str(item.get("buttonStateLabel") or "不可用")
    return AbilityControl(owner, f"{side}技能：{state}", name)


class NativeOverlayApp:
    """Tk/Win32 host overlay that leaves Null's Royale rendering untouched."""

    TRANSPARENT_COLOR = "#010203"

    def __init__(
        self, env: NativeClashEnv, *, poll_milliseconds: int = 200, window_title: str = TARGET_WINDOW_TITLE, owner: int | None = None
    ) -> None:
        if os.name != "nt":
            raise OverlayError("native overlay requires Windows")
        import tkinter as tk

        self.tk = tk
        self.env = env
        self.owner = owner
        self.poll_milliseconds = poll_milliseconds
        self.locator = MumuWindowLocator(window_title)
        self.card_names = _load_card_names()
        self.card_specs = build_card_catalog().by_id
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cr-native-ui")
        self.poll_future: Future[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, str | None]] | None = None
        self.action_future: Future[Any] | None = None
        self.action_kind: str | None = None
        self.latest_observation: dict[str, Any] | None = None
        self.latest_status: dict[str, Any] | None = None
        self.latest_rich_observation: dict[str, Any] | None = None
        self.latest_ability_error: str | None = None
        self.selection: Selection | None = None
        self.drag_selection: Selection | None = None
        self.pending_plays: deque[tuple[Selection, int, int]] = deque()
        self.target_hwnd: int | None = None
        self.viewport: Rect | None = None
        self.viewport_source = "unresolved"
        self.last_layout_check = 0.0
        self.last_poll = 0.0
        self.last_crosshair: tuple[float, float, float] | None = None
        self._mask_items: list[int] = []
        self._mask_viewport_size: tuple[int, int] | None = None
        self.closed = False

        self.root = tk.Tk()
        self.root.title("CR Native Controls")
        self.root.configure(bg="#151b25")
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-toolwindow", True)
        except tk.TclError:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(value=f"正在定位 {window_title}…")
        self.selection_var = tk.StringVar(value="先点上方或下方的一张牌")
        self.mask_model_hud_var = tk.BooleanVar(value=owner is not None)
        self.status_label = self._label(self.status_var, row=0, color="#d7e3f4", font="Segoe UI", pady=(7, 1))
        self.selection_label = self._label(
            self.selection_var, row=1, color="#ffd43b", font="Microsoft YaHei UI", pady=(0, 5)
        )
        self.speed_buttons: dict[float, Any] = {}
        for column, speed in enumerate(SPEED_CHOICES):
            self.speed_buttons[speed] = self._button(
                f"{speed:g}×",
                lambda value=speed: self.request_speed(value),
                row=2,
                column=column,
                padx=(7 if column == 0 else 2, 2),
                width=5,
                bg="#263447",
                fg="#f5f7fa",
                activebackground="#3b82f6",
                font=("Segoe UI Semibold", 9),
            )
        self.pause_button = self._button(
            "暂停",
            self.request_pause_toggle,
            row=2,
            column=5,
            padx=(4, 7),
            width=6,
            bg="#8b3a3a",
            activebackground="#c24141",
        )
        self.ability_buttons: dict[int, Any] = {}
        for owner, side in enumerate(("上方", "下方")):
            self.ability_buttons[owner] = self._button(
                f"{side}技能：未上场",
                lambda value=owner: self.request_ability(value),
                row=3,
                column=owner * 3,
                columnspan=3,
                sticky="ew",
                padx=(7, 3) if owner == 0 else (3, 7),
                width=18,
                bg="#3f3f46",
                activebackground="#7c3aed",
                state="disabled",
            )
        if self.owner is not None:
            self.mask_model_hud_checkbox = tk.Checkbutton(
                self.root,
                text="遮住模型手牌和圣水",
                variable=self.mask_model_hud_var,
                command=self._toggle_model_hud_mask,
                bg="#151b25",
                fg="#d7e3f4",
                activebackground="#151b25",
                activeforeground="#d7e3f4",
                selectcolor="#263447",
                font=("Microsoft YaHei UI", 9),
                anchor="w",
            )
            self.mask_model_hud_checkbox.grid(row=4, column=0, columnspan=6, sticky="w", padx=8, pady=(0, 8))

        # The capture layer is alpha=1%; it receives clicks while changing the
        # native picture by less than a single 8-bit brightness step.
        self.capture = self._layer("black", "-alpha", 0.01)
        self.capture.bind("<Button-1>", self._on_viewport_click)
        self.capture.bind("<B1-Motion>", self._on_viewport_drag)
        self.capture.bind("<ButtonRelease-1>", self._on_viewport_release)

        # Feedback is fully transparent except for Canvas primitives and is
        # always click-through, so the capture layer below owns hit handling.
        self.feedback = self._layer(
            self.TRANSPARENT_COLOR, "-transparentcolor", self.TRANSPARENT_COLOR, styles=WS_EX_TRANSPARENT
        )
        self.canvas = tk.Canvas(self.feedback, bg=self.TRANSPARENT_COLOR, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.control_hwnd = _root_hwnd(self.root)
        self.capture_hwnd = _root_hwnd(self.capture)
        self.feedback_hwnd = _root_hwnd(self.feedback)
        self.root.geometry(f"370x{160 if self.owner is not None else 130}+30+30")
        self.root.after(20, self._tick)

    def _label(self, variable: Any, *, row: int, color: str, font: str, pady: Any) -> Any:
        label = self.tk.Label(self.root, textvariable=variable, bg="#151b25", fg=color, anchor="w", font=(font, 9))
        label.grid(row=row, column=0, columnspan=6, sticky="ew", padx=9, pady=pady)
        return label

    def _button(
        self,
        text: str,
        command: Callable[[], None],
        *,
        row: int,
        column: int,
        padx: Any,
        columnspan: int = 1,
        sticky: str = "",
        **options: Any,
    ) -> Any:
        button = self.tk.Button(
            self.root,
            text=text,
            command=command,
            **{
                "fg": "white",
                "activeforeground": "white",
                "relief": "flat",
                "bd": 0,
                "font": ("Microsoft YaHei UI", 9),
                **options,
            },
        )
        button.grid(row=row, column=column, columnspan=columnspan, sticky=sticky, padx=padx, pady=(0, 8))
        return button

    def _layer(self, color: str, attribute: str, value: Any, *, styles: int = 0) -> Any:
        layer = self.tk.Toplevel(self.root)
        layer.overrideredirect(True)
        layer.configure(bg=color)
        layer.attributes("-topmost", True)
        layer.attributes(attribute, value)
        _add_extended_styles(layer, WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | styles)
        layer.withdraw()
        return layer

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for layer in (self.capture, self.feedback):
            layer.withdraw()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

    def _set_message(self, text: str, *, error: bool = False) -> None:
        self.selection_var.set(text)
        self.selection_label.configure(fg="#ff6b6b" if error else "#ffd43b")

    def _toggle_model_hud_mask(self) -> None:
        self._refresh_native_labels()
        self._draw_feedback()

    def _resolve_layout(self) -> None:
        now = time.monotonic()
        if now - self.last_layout_check < 0.15:
            return
        self.last_layout_check = now
        hwnd = self.locator.find_window()
        self.target_hwnd = hwnd
        try:
            if hwnd is None:
                raise OverlayError(f"等待 {self.locator.window_title} 窗口")
            self.viewport, self.viewport_source = self.locator.viewport(hwnd)
        except OverlayError as error:
            self.viewport = None
            self.status_var.set(str(error))
        viewport = self.viewport
        if viewport is None or not self.locator.foreground_matches(
            hwnd, self.control_hwnd, self.capture_hwnd, self.feedback_hwnd
        ):
            for layer in (self.capture, self.feedback):
                layer.withdraw()
            return

        geometry = f"{viewport.width}x{viewport.height}+{viewport.left}+{viewport.top}"
        for layer in (self.capture, self.feedback):
            layer.geometry(geometry)
            layer.deiconify()
            layer.lift()
        self.root.lift()
        self._dock_control(viewport)

    def _dock_control(self, viewport: Rect) -> None:
        self.root.update_idletasks()
        width = max(370, self.root.winfo_reqwidth())
        height = max(160 if self.owner is not None else 130, self.root.winfo_reqheight())
        virtual_left = int(user32.GetSystemMetrics(76))
        virtual_top = int(user32.GetSystemMetrics(77))
        virtual_right = virtual_left + int(user32.GetSystemMetrics(78))
        virtual_bottom = virtual_top + int(user32.GetSystemMetrics(79))
        gap = 8
        if viewport.right + gap + width <= virtual_right:
            left = viewport.right + gap
        elif viewport.left - gap - width >= virtual_left:
            left = viewport.left - gap - width
        else:
            left = min(max(viewport.left, virtual_left), virtual_right - width)
        top = min(max(viewport.top, virtual_top), virtual_bottom - height)
        self.root.geometry(f"{width}x{height}+{left}+{top}")

    def _submit_poll(self) -> None:
        if self.poll_future is not None or self.action_future is not None:
            return
        self.poll_future = self.executor.submit(self._poll_native)

    def _poll_native(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, str | None]:
        observation = self.env.observe()
        status = self.env.status()
        try:
            rich = self.env.observe_rich()
            ability_error = None
        except Exception as error:
            # Keep card play available when rich telemetry fails; only active
            # ability controls should fail closed.
            rich = None
            ability_error = str(error)
        return observation, status, rich, ability_error

    def _consume_poll(self) -> None:
        future = self.poll_future
        if future is None or not future.done():
            return
        self.poll_future = None
        try:
            observation, status, rich, ability_error = future.result()
        except Exception as error:
            self.latest_observation = None
            self.latest_status = None
            self.latest_rich_observation = None
            self.latest_ability_error = str(error)
            self.status_var.set(f"native runner 未连接：{error}")
            return
        self.latest_observation = observation
        self.latest_status = status
        self.latest_rich_observation = rich
        self.latest_ability_error = ability_error
        self._refresh_native_labels()
        self._validate_selection()
        self._dispatch_pending_play()

    def _dispatch_pending_play(self) -> None:
        if not self.pending_plays or self.action_future is not None:
            return
        observation, status = self.latest_observation, self.latest_status
        if observation is None or status is None:
            return
        if observation.get("ended") or status.get("mode") != "native-render":
            self.pending_plays.clear()
            self._set_message("对局已结束或 native-render 会话已丢失，连续下牌已取消", error=True)
            return
        if int(observation.get("queuedCommands", 0)) > 0:
            return
        selection, world_x, world_y = self.pending_plays.popleft()
        self._set_message(f"提交 {selection.name} · O{selection.owner} · ({world_x}, {world_y})")
        self._submit_action("play", self._run_play, selection, world_x, world_y)

    def _queue_or_submit_play(self, selection: Selection, world_x: int, world_y: int) -> None:
        observation = self.latest_observation or {}
        if self.action_future is not None or self.pending_plays or int(observation.get("queuedCommands", 0)) > 0:
            self.pending_plays.append((selection, world_x, world_y))
            self.last_poll = 0.0
            self._set_message(f"已缓冲 {selection.name} · 等前一张牌完成后自动下牌")
            return
        self._set_message(f"提交 {selection.name} · O{selection.owner} · ({world_x}, {world_y})")
        self._submit_action("play", self._run_play, selection, world_x, world_y)

    def _refresh_native_labels(self) -> None:
        if self.latest_observation is None or self.latest_status is None:
            return
        status = self.latest_status
        observation = self.latest_observation
        mode = str(status.get("mode", "unknown"))
        tick = int(observation.get("tick", status.get("tick", 0)))
        speed = native_speed(status)
        paused = bool(status.get("paused", False))
        rendering = not bool(status.get("renderSuppressed", False))
        if mode != "native-render":
            suffix = "native-render 会话已丢失"
        elif paused:
            suffix = "暂停"
        elif speed is None:
            suffix = "倍速状态未知"
        else:
            suffix = f"{speed:g}×"
        if not rendering:
            suffix += " · 渲染关闭"
        if observation.get("ended"):
            suffix += " · 已结束"
        self.status_var.set(f"{mode} · tick {tick} · {suffix}")
        allowed = mode == "native-render" and rendering
        for value, button in self.speed_buttons.items():
            selected = not paused and speed is not None and math.isclose(speed, value, rel_tol=1e-7)
            button.configure(
                state="normal" if allowed and self.owner is None else "disabled",
                bg="#2563eb" if selected else "#263447",
                relief="sunken" if selected else "flat",
            )
        self.pause_button.configure(
            state="normal" if allowed and self.owner is None else "disabled",
            text="继续" if paused else "暂停",
            bg="#287a4b" if paused else "#8b3a3a",
        )

        for owner, button in self.ability_buttons.items():
            control = ability_control(self.latest_rich_observation, owner)
            hidden = self.owner is not None and owner != self.owner and self.mask_model_hud_var.get()
            button.configure(
                text=f"{'上方' if owner == 0 else '下方'}技能：已隐藏" if hidden else control.text,
                state="normal" if allowed and control.ready and (self.owner is None or owner == self.owner) else "disabled",
                bg="#7c3aed" if control.ready and not hidden else "#3f3f46",
            )

    def _validate_selection(self) -> None:
        if self.selection is None or self.latest_observation is None:
            return
        try:
            current = _card_in_hand(_player(self.latest_observation, self.selection.owner), self.selection.hand_index)
        except RunnerError:
            current = None
        if current is None or int(current["cardId"]) != self.selection.card_id:
            self.selection = None
            self._set_message("手牌已轮换，请重新选择")

    def _on_viewport_click(self, event: Any) -> None:
        self.drag_selection = None
        viewport = self.viewport
        observation = self.latest_observation
        status = self.latest_status
        if viewport is None or observation is None or status is None:
            self._set_message("native runner 尚未就绪", error=True)
            return
        if status.get("mode") != "native-render":
            self._set_message("当前不是 native-render 模式", error=True)
            return
        source_x, source_y = viewport_to_source(viewport, event.x, event.y)
        hit = hit_card_source(source_x, source_y)
        if hit is not None:
            if self._select_card(hit):
                self.drag_selection = self.selection
            self._draw_feedback()
            return
        if not ARENA_RECT.contains(source_x, source_y):
            return
        if self.selection is None:
            self._set_message("请先点一张上方或下方手牌", error=True)
            return
        try:
            world_x, world_y = arena_source_to_world(source_x, source_y)
        except ValueError as error:
            self._set_message(str(error), error=True)
            return
        selected = self.selection
        self.last_crosshair = (source_x, source_y, time.monotonic())
        self._queue_or_submit_play(selected, world_x, world_y)
        self._draw_feedback()

    def _on_viewport_drag(self, event: Any) -> None:
        if self.drag_selection is None or self.viewport is None:
            return
        source_x, source_y = viewport_to_source(self.viewport, event.x, event.y)
        if ARENA_RECT.contains(source_x, source_y):
            self.last_crosshair = (source_x, source_y, time.monotonic())

    def _on_viewport_release(self, event: Any) -> None:
        dragged = self.drag_selection
        self.drag_selection = None
        if dragged is None or dragged != self.selection or self.viewport is None:
            return
        source_x, source_y = viewport_to_source(self.viewport, event.x, event.y)
        if ARENA_RECT.contains(source_x, source_y):
            self._on_viewport_click(event)

    def _select_card(self, hit: CardHit) -> bool:
        if self.owner is not None and hit.owner != self.owner:
            self._set_message("这一方由模型控制", error=True)
            return False
        if self.latest_observation is None:
            self._set_message("还没有原生手牌状态", error=True)
            return False
        if self.latest_observation.get("ended"):
            self._set_message("对局已结束", error=True)
            return False
        try:
            player = _player(self.latest_observation, hit.owner)
            card = _card_in_hand(player, hit.hand_index)
        except RunnerError as error:
            self._set_message(str(error), error=True)
            return False
        cost_value = card.get("cost")
        cost = int(cost_value) if cost_value is not None else None
        card_id = int(card["cardId"])
        name = self.card_names.get(card_id, f"卡牌 {card_id}")
        self.selection = Selection(
            owner=hit.owner,
            visual_slot=hit.visual_slot,
            hand_index=hit.hand_index,
            card_id=card_id,
            cost=cost,
            name=name,
        )
        cost_text = "?" if cost is None else str(cost)
        side = "上方" if hit.owner == 0 else "下方"
        hint = "再点竞技场"
        if cost is not None and cost * 10_000 > int(player.get("elixirRaw", 0)):
            hint = "已预选；圣水足够后再点竞技场"
        self._set_message(f"已选 {side} O{hit.owner} · {name} · {cost_text}费；{hint}")
        return True

    def _run_play(self, selection: Selection, world_x: int, world_y: int) -> PlayResult:
        if self.owner is not None and selection.owner != self.owner:
            raise RunnerError("这一方由模型控制")
        before = self.env.observe()
        player = _player(before, selection.owner)
        card = _card_in_hand(player, selection.hand_index)
        if int(card["cardId"]) != selection.card_id:
            raise RunnerError("原生手牌已经变化，请重新选择")
        if card.get("cost") is not None and int(card["cost"]) * 10_000 > int(player["elixirRaw"]):
            raise RunnerError("当前圣水不足")

        world_x, world_y = nearest_legal_placement(
            before, card, selection.owner, world_x, world_y, self.card_specs
        )

        hand_action = HandAction(owner=selection.owner, hand_index=selection.hand_index, x=world_x, y=world_y)
        if self.owner is not None:
            # Human vs AI: target the same next-tick input plus 20-tick native
            # command interval used for the model's immediate play.
            self.env.queue_hand_action_at(
                hand_action,
                execute_tick=int(before["tick"]) + COMMAND_CONSUMPTION_STEPS,
                clamp_late=True,
            )
        else:
            self.env.play_immediate(hand_action)
        deadline = time.monotonic() + 1.25
        last = before
        while time.monotonic() < deadline:
            time.sleep(0.04)
            last = self.env.observe()
            current = next((card for card in _player(last, selection.owner)["hand"]
                            if int(card["handIndex"]) == selection.hand_index), None)
            if current is None or int(current["cardId"]) != selection.card_id:
                return PlayResult("accepted", selection, world_x, world_y, "native hand rotated")
            status = self.env.status()
            if bool(status.get("paused", False)):
                return PlayResult("queued", selection, world_x, world_y, "renderer is paused")
            if int(last.get("queuedCommands", 0)) == 0 and time.monotonic() + 0.9 < deadline:
                break
        if int(last.get("queuedCommands", 0)) > 0:
            return PlayResult("queued", selection, world_x, world_y, "native command remains queued")
        return PlayResult("rejected", selection, world_x, world_y, "hand did not rotate")

    def request_speed(self, speed: float) -> None:
        if self.owner is not None or speed not in SPEED_CHOICES or self.action_future is not None:
            return
        self._set_message(f"正在切换到 {speed:g}×…")
        self._submit_action("speed", self._run_set_speed, speed)

    def _run_set_speed(self, speed: float) -> dict[str, Any]:
        return self.env.set_speed(speed)

    def _submit_action(self, kind: str, function: Callable[..., Any], *args: Any) -> None:
        self.action_kind = kind
        self.action_future = self.executor.submit(function, *args)

    def request_pause_toggle(self) -> None:
        if self.owner is not None:
            return
        if self.action_future is not None:
            return
        paused = bool((self.latest_status or {}).get("paused", False))
        self._set_message("正在继续…" if paused else "正在暂停…")
        self._submit_action("resume" if paused else "pause", self.env.resume if paused else self.env.pause)

    def request_ability(self, owner: int) -> None:
        if (self.owner is not None and owner != self.owner) or owner not in (0, 1) or self.action_future is not None:
            return
        control = ability_control(self.latest_rich_observation, owner)
        if not control.ready:
            detail = self.latest_ability_error or control.text
            self._set_message(detail, error=True)
            return
        self.selection = None
        self._set_message(f"正在激活 {control.text}")
        self._submit_action(f"ability:{owner}", self._run_ability, owner)

    def _run_ability(self, owner: int) -> dict[str, Any]:
        # Resolve the exact live entity key at click time rather than trusting
        # the last UI poll.
        control = ability_control(self.env.observe_rich(), owner)
        if not control.ready or control.source_entity_key is None:
            raise RunnerError(control.text)
        if self.owner is not None and owner != self.owner:
            raise RunnerError("这一方由模型控制")
        action = AbilityAction(*control.source_entity_key)
        result = self.env.activate_ability(action)
        return {**result, "owner": owner, "abilityName": control.ability_name}

    def _consume_action(self) -> None:
        future = self.action_future
        if future is None or not future.done():
            return
        kind = self.action_kind
        self.action_future = None
        self.action_kind = None
        try:
            result = future.result()
        except Exception as error:
            self._set_message(str(error), error=True)
            self.last_poll = 0.0
            return
        if kind == "play" and isinstance(result, PlayResult):
            if result.outcome == "accepted":
                if self.selection == result.selection:
                    self.selection = None
                self._set_message(f"下牌成功 · {result.selection.name} · ({result.world_x}, {result.world_y})")
            elif result.outcome == "queued":
                if self.selection == result.selection:
                    self.selection = None
                self._set_message(f"已排队 · {result.selection.name}")
            else:
                self._set_message("原生引擎拒绝该落点；可直接换位置再点", error=True)
        elif kind == "speed":
            speed = native_speed(result)
            if speed is None:
                speed = native_speed(self.latest_status)
            if speed is None:
                self._set_message("倍速切换失败：native-render 会话已丢失", error=True)
            else:
                self._set_message(f"倍速已切换为 {speed:g}×")
        elif kind == "pause":
            self._set_message("已暂停；仍可选牌和落点")
        elif kind == "resume":
            self._set_message("已继续")
        elif kind is not None and kind.startswith("ability:"):
            owner = int(result["owner"])
            side = "上方" if owner == 0 else "下方"
            self._set_message(f"{side}勇者骑士技能已提交；下一原生 tick 生效")
        self.last_poll = 0.0

    def _draw_feedback(self) -> None:
        viewport = self.viewport
        if viewport is None:
            return
        self.canvas.delete("feedback")
        # Thin permanent hit boxes make the native cards visibly clickable.
        for hit in CARD_HITS:
            rect = source_rect_to_viewport(viewport, hit.source_rect)
            color = "#ff8787" if hit.owner == 0 else "#74c0fc"
            width = 2
            if (
                self.selection is not None
                and hit.owner == self.selection.owner
                and hit.hand_index == self.selection.hand_index
            ):
                color = "#ffe066"
                width = 5
            self._draw_rectangle(rect, color, width)

        if self.selection is not None:
            arena = source_rect_to_viewport(viewport, WORLD_PROJECTION_RECT)
            self._draw_rectangle(arena, "#ffe066", 2, dash=(8, 6))
        if self.last_crosshair is not None:
            source_x, source_y, created = self.last_crosshair
            if time.monotonic() - created <= 0.9:
                x, y = source_to_viewport(viewport, source_x, source_y)
                self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline="#ffe066", width=3, tags="feedback")
                self.canvas.create_line(x - 15, y, x + 15, y, fill="#ffe066", width=2, tags="feedback")
                self.canvas.create_line(x, y - 15, x, y + 15, fill="#ffe066", width=2, tags="feedback")
            else:
                self.last_crosshair = None
        self._sync_model_hud_mask(viewport)

    def _sync_model_hud_mask(self, viewport: Rect) -> None:
        if self.owner is None or not self.mask_model_hud_var.get():
            if self._mask_items:
                self.canvas.delete("model-mask")
                self._mask_items.clear()
                self._mask_viewport_size = None
            return
        rects = tuple(source_rect_to_viewport(viewport, item) for item in model_hud_mask_rects(self.owner))
        if not self._mask_items:
            self._mask_items = [
                self.canvas.create_rectangle(
                    rect.left, rect.top, rect.right, rect.bottom,
                    fill="#10151f", outline="", tags="model-mask",
                )
                for rect in rects
            ]
            self._mask_viewport_size = (viewport.width, viewport.height)
        elif self._mask_viewport_size != (viewport.width, viewport.height):
            for item, rect in zip(self._mask_items, rects):
                self.canvas.coords(item, rect.left, rect.top, rect.right, rect.bottom)
            self._mask_viewport_size = (viewport.width, viewport.height)
        self.canvas.tag_raise("model-mask")

    def _draw_rectangle(self, rect: Rect, color: str, width: int, **options: Any) -> None:
        self.canvas.create_rectangle(
            rect.left, rect.top, rect.right, rect.bottom, outline=color, width=width, tags="feedback", **options
        )

    def _tick(self) -> None:
        if self.closed:
            return
        try:
            self._resolve_layout()
            self._consume_action()
            self._consume_poll()
            now = time.monotonic()
            if now - self.last_poll >= self.poll_milliseconds / 1000.0:
                self.last_poll = now
                self._submit_poll()
            self._draw_feedback()
        except Exception as error:
            self.status_var.set(f"overlay 错误：{error}")
        self.root.after(20, self._tick)


def diagnose(env: NativeClashEnv, *, window_title: str = TARGET_WINDOW_TITLE) -> dict[str, Any]:
    if os.name != "nt":
        raise OverlayError("native overlay requires Windows")
    locator = MumuWindowLocator(window_title)
    hwnd = locator.find_window()
    if hwnd is None:
        raise OverlayError(f"cannot find exact MuMu window {window_title!r}")
    viewport, source = locator.viewport(hwnd)
    status = env.status()
    observation = env.observe()
    return {
        "ok": status.get("mode") == "native-render",
        "windowTitle": window_title,
        "windowHandle": f"0x{hwnd:x}",
        "viewport": viewport.to_dict(),
        "viewportSource": source,
        "mode": status.get("mode"),
        "renderSuppressed": bool(status.get("renderSuppressed", False)),
        "speed": status.get("speed", 1.0),
        "paused": bool(status.get("paused", False)),
        "tick": observation.get("tick"),
        "upperHand": {"owner": 0, "leftToRightHandIndex": [3, 2, 1, 0]},
        "lowerHand": {"owner": 1, "leftToRightHandIndex": [0, 1, 2, 3]},
        "arenaSourceRect": ARENA_RECT.to_dict(),
        "worldProjectionSourceRect": WORLD_PROJECTION_RECT.to_dict(),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-host", default=DEFAULT_RUNNER_HOST)
    parser.add_argument("--runner-port", type=int, default=DEFAULT_RUNNER_PORT)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--owner", type=int, choices=(0, 1), help="human-controlled side in an offline model match")
    parser.add_argument(
        "--window-title",
        choices=ALLOWED_WINDOW_TITLES,
        default=TARGET_WINDOW_TITLE,
        help="exact isolated MuMu harness window to follow",
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="print the hard-locked window/viewport/native status and exit"
    )
    args = parser.parse_args(argv)
    if args.runner_host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("native overlay accepts only a loopback runner host")
    if not 50 <= args.poll_ms <= 2000:
        parser.error("--poll-ms must be in 50..2000")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    _set_dpi_awareness()
    env = NativeClashEnv(args.runner_host, args.runner_port, timeout=3.0)
    if args.diagnose:
        report = diagnose(env, window_title=args.window_title)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] and not report["renderSuppressed"] else 1
    app = NativeOverlayApp(env, poll_milliseconds=args.poll_ms, window_title=args.window_title, owner=args.owner)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
