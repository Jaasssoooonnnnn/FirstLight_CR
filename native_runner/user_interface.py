"""User-facing FirstLight CR launcher for matches and native-render replays.

The interface deliberately separates read-only preflight checks from the
destructive ``start_offline.ps1`` lifecycle.  Resource pressure is a soft
warning.  The interactive worker is separate from the training worker;
occupied resident matches, native clients, and unknown native-render sessions
on the interactive endpoint remain hard conflicts.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .paths import REPOSITORY_ROOT, WORKSPACE_ROOT
from .card_names import CARD_NAMES
from .local_config import setting

PACKAGE_ROOT = Path(__file__).resolve().parent
MUMU_MANAGER = Path(setting("CR_MUMU_MANAGER", "MuMuManager.exe"))
START_OFFLINE_SCRIPT = PACKAGE_ROOT / "start_offline.ps1"
VM_INDEX = int(setting("CR_VM_INDEX", "-1"))
VM_NAME = setting("CR_VM_NAME")
ADB_SERIAL = setting("CR_ADB_SERIAL")
CONTROL_HOST = setting("CR_CONTROL_HOST", "127.0.0.1")
CONTROL_PORT = int(setting("CR_CONTROL_PORT", "26789"))
CONTROL_PORTS = (CONTROL_PORT,)
MEMORY_WARNING_PERCENT = 85.0
MEMORY_WARNING_AVAILABLE_GIB = 4.0
GPU_WARNING_PERCENT = 80.0
VRAM_WARNING_PERCENT = 90.0
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
ERROR_ALREADY_EXISTS = 183

FORM_BASE = "基础"
FORM_EVOLUTION = "觉醒"
FORM_HERO = "精英（英雄）"
FORM_TO_MASK = {FORM_BASE: 0, FORM_EVOLUTION: 1, FORM_HERO: 2}
MASK_TO_FORM = {value: key for key, value in FORM_TO_MASK.items()}


def validate_model_deck_roles(forms: Sequence[int]) -> None:
    """Reject model form selections that exceed the loaded V4 input contract."""

    if len(forms) != 8 or any(int(mask) not in MASK_TO_FORM for mask in forms):
        raise ValueError("模型牌组的 8 个形态设置无效")
    evolution_count = sum(int(mask) == 1 for mask in forms)
    hero_count = sum(int(mask) == 2 for mask in forms)
    if evolution_count > 2 or hero_count > 2 or evolution_count + hero_count > 3:
        raise ValueError(
            f"模型牌组当前有 {evolution_count} 张觉醒、{hero_count} 张英雄形态；"
            "V4 模型最多支持 2 张觉醒、2 张英雄，且两类合计不超过 3 张。"
            "请把多余的形态改为“基础”后再开始。"
        )


SPEED_XBOW_DECK = (27000008, 26000000, 26000001, 27000006, 26000010, 26000084, 28000000, 28000011)
SPEED_XBOW_FORMS = (0, 2, 1, 1, 0, 0, 0, 0)
PEKKA_BRIDGE_SPAM_DECK = (26000036, 26000050, 26000062, 26000004, 26000046, 26000042, 28000000, 28000008)
PEKKA_BRIDGE_SPAM_FORMS = (1, 1, 2, 0, 0, 0, 0, 0)


LOGGER = logging.getLogger("firstlight_interface")


def _configure_logging() -> Path:
    destination = WORKSPACE_ROOT / "runs" / "interface"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "firstlight-interface.log"
    if not LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
        LOGGER.addHandler(handler)
    return path


def _run_hidden(
    arguments: Sequence[str | os.PathLike[str]], *, timeout: float, cwd: Path = REPOSITORY_ROOT
) -> subprocess.CompletedProcess[str]:
    # MuMu descendants can retain inherited stdout/stderr after the launcher
    # exits. Files avoid waiting for pipe EOF, including on the timeout path.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run(
            [os.fspath(item) for item in arguments],
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            stdout.read().decode("utf-8", errors="replace"),
            stderr.read().decode("utf-8", errors="replace"),
        )


def _run_powershell(script: str, *, timeout: float = 15.0) -> str:
    command = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8\n$ProgressPreference='SilentlyContinue'\n" + script
    )
    result = _run_hidden(("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command), timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "PowerShell 状态查询失败")
    return result.stdout.strip()


def _json_rows(text: str) -> tuple[dict[str, Any], ...]:
    if not text.strip():
        return ()
    value = json.loads(text)
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, list):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    raise ValueError("状态命令返回了无效 JSON")


@dataclass(frozen=True, slots=True)
class VmStatus:
    found: bool
    running: bool
    ready: bool
    name: str | None
    adb_serial: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    used_percent: float
    available_gib: float
    total_gib: float


@dataclass(frozen=True, slots=True)
class GpuStatus:
    available: bool
    utilization_percent: float | None
    utilization_samples: tuple[float, ...]
    memory_used_mib: float | None
    memory_total_mib: float | None
    memory_used_percent: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    pid: int
    command_line: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    vm: VmStatus
    memory: MemoryStatus
    gpu: GpuStatus
    training_processes: tuple[ProcessStatus, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def query_vm_status(*, vm_index: int = VM_INDEX, vm_name: str = VM_NAME) -> VmStatus:
    if not MUMU_MANAGER.is_file():
        return VmStatus(
            found=False,
            running=False,
            ready=False,
            name=None,
            adb_serial=None,
            detail=f"未找到 MuMuManager：{MUMU_MANAGER}",
        )
    result = _run_hidden((MUMU_MANAGER, "info", "--vmindex", str(vm_index)), timeout=15.0)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return VmStatus(False, False, False, None, None, detail)
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return VmStatus(False, False, False, None, None, f"MuMu 状态不是有效 JSON：{error}")
    name = str(info.get("name") or "")
    index = int(info.get("index", -1))
    found = name == vm_name and index == vm_index
    running = bool(info.get("is_process_started"))
    ready = running and info.get("player_state") == "start_finished"
    adb_serial = None
    if info.get("adb_host_ip") and info.get("adb_port"):
        adb_serial = f"{info['adb_host_ip']}:{int(info['adb_port'])}"
    if not found:
        detail = f"MuMu #{vm_index} 不是预期的 {vm_name}"
    elif ready:
        detail = f"{vm_name} 已就绪"
    elif running:
        detail = f"{vm_name} 正在启动"
    else:
        detail = f"{vm_name} 未启动"
    return VmStatus(found, running, ready, name or None, adb_serial, detail)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def query_memory_status() -> MemoryStatus:
    value = _MemoryStatusEx()
    value.dwLength = ctypes.sizeof(value)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("无法读取 Windows 内存状态")
    return MemoryStatus(
        used_percent=float(value.dwMemoryLoad),
        available_gib=float(value.ullAvailPhys) / (1 << 30),
        total_gib=float(value.ullTotalPhys) / (1 << 30),
    )


def query_gpu_status(*, samples: int = 5, interval_seconds: float = 0.2) -> GpuStatus:
    utilization: list[float] = []
    memory_used: float | None = None
    memory_total: float | None = None
    detail = ""
    for index in range(max(1, samples)):
        try:
            result = _run_hidden(
                (
                    "nvidia-smi.exe",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ),
                timeout=10.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return GpuStatus(False, None, (), None, None, None, f"GPU 状态不可用：{error}")
        if result.returncode != 0 or not result.stdout.strip():
            detail = (result.stderr or result.stdout).strip()
            return GpuStatus(False, None, (), None, None, None, f"GPU 状态不可用：{detail}")
        try:
            first = result.stdout.splitlines()[0]
            gpu_raw, used_raw, total_raw = (item.strip() for item in first.split(",")[:3])
            utilization.append(float(gpu_raw))
            memory_used = float(used_raw)
            memory_total = float(total_raw)
        except (IndexError, ValueError) as error:
            return GpuStatus(False, None, (), None, None, None, f"无法解析 nvidia-smi：{error}")
        if index + 1 < samples:
            time.sleep(interval_seconds)
    memory_percent = memory_used / memory_total * 100.0 if memory_used is not None and memory_total else None
    return GpuStatus(
        True,
        sum(utilization) / len(utilization),
        tuple(utilization),
        memory_used,
        memory_total,
        memory_percent,
        "GPU 状态正常",
    )


def query_processes(pattern: str) -> tuple[ProcessStatus, ...]:
    escaped = pattern.replace("'", "''")
    output = _run_powershell(
        f"""
$items = @(
  Get-CimInstance Win32_Process |
    Where-Object {{ $_.CommandLine -match '{escaped}' }} |
    Select-Object ProcessId,CommandLine
)
$items | ConvertTo-Json -Compress
"""
    )
    return tuple(
        ProcessStatus(pid=int(item["ProcessId"]), command_line=str(item.get("CommandLine") or ""))
        for item in _json_rows(output)
    )


def query_training_processes() -> tuple[ProcessStatus, ...]:
    return query_processes(r"native_runner\.training\.")


def resource_warnings(
    memory: MemoryStatus, gpu: GpuStatus, training_processes: Sequence[ProcessStatus]
) -> tuple[str, ...]:
    warnings: list[str] = []
    if memory.used_percent >= MEMORY_WARNING_PERCENT or memory.available_gib < MEMORY_WARNING_AVAILABLE_GIB:
        warnings.append(f"物理内存压力较高：已用 {memory.used_percent:.1f}%，可用 {memory.available_gib:.2f} GiB。")
    if gpu.available and gpu.utilization_samples and max(gpu.utilization_samples) >= GPU_WARNING_PERCENT:
        warnings.append(f"GPU 利用率达到 {max(gpu.utilization_samples):.0f}%（提醒阈值 80%）。")
    if gpu.available and gpu.memory_used_percent is not None and gpu.memory_used_percent >= VRAM_WARNING_PERCENT:
        warnings.append(
            "显存占用较高："
            f"{gpu.memory_used_mib:.0f}/{gpu.memory_total_mib:.0f} MiB "
            f"（{gpu.memory_used_percent:.1f}%）。"
        )
    if training_processes:
        warnings.append(
            "检测到训练程序："
            + ", ".join(f"PID {item.pid}" for item in training_processes)
            + "。继续只会进入界面，不会停止训练。"
        )
    return tuple(warnings)


def collect_preflight_report(*, check_vm: bool = True) -> PreflightReport:
    vm = query_vm_status() if check_vm else VmStatus(False, False, False, None, None, "离线 VM 将在实际操作时检测")
    memory = query_memory_status()
    gpu = query_gpu_status()
    try:
        training = query_training_processes()
    except Exception as error:
        LOGGER.warning("training process query failed: %s", error)
        training = ()
    return PreflightReport(
        vm=vm, memory=memory, gpu=gpu, training_processes=training, warnings=resource_warnings(memory, gpu, training)
    )


def _probe_native(command: str, *, port: int = CONTROL_PORT, timeout: float = 0.6) -> dict[str, Any] | None:
    try:
        with socket.create_connection((CONTROL_HOST, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall((command + "\n").encode("utf-8"))
            payload = bytearray()
            while len(payload) < 1 << 20:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                payload.extend(chunk)
                if b"\n" in chunk:
                    break
    except (OSError, TimeoutError):
        return None
    line = bytes(payload).split(b"\n", 1)[0].strip()
    if not line:
        return None
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def query_established_native_clients() -> tuple[dict[str, Any], ...]:
    ports = ",".join(str(value) for value in CONTROL_PORTS)
    output = _run_powershell(
        f"""
$ports = @({ports})
$processes = @{{}}
Get-CimInstance Win32_Process | ForEach-Object {{
  $processes[[int]$_.ProcessId] = [string]$_.CommandLine
}}
$rows = @(
  Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
    Where-Object {{ $ports -contains [int]$_.RemotePort }} |
    ForEach-Object {{
      [pscustomobject]@{{
        localPort = [int]$_.LocalPort
        remotePort = [int]$_.RemotePort
        pid = [int]$_.OwningProcess
        commandLine = $processes[[int]$_.OwningProcess]
      }}
    }}
)
$rows | ConvertTo-Json -Compress
""",
        timeout=20.0,
    )
    return _json_rows(output)


def runtime_conflicts(*, allow_owned_native_render: bool, ignored_pids: Sequence[int] = ()) -> tuple[str, ...]:
    conflicts: list[str] = []
    ignored = {os.getpid(), *(int(value) for value in ignored_pids)}
    try:
        clients = tuple(item for item in query_established_native_clients() if int(item.get("pid", -1)) not in ignored)
    except Exception as error:
        conflicts.append(f"无法确认 native 端口占用：{error}")
        clients = ()
    if clients:
        pids = sorted({int(item.get("pid", -1)) for item in clients})
        conflicts.append("native 引擎存在其他客户端连接：" + ", ".join(f"PID {value}" for value in pids) + "。")

    multi = _probe_native("multi-status")
    if multi is not None and int(multi.get("occupied", 0)) > 0:
        conflicts.append(f"native 引擎仍持有 {int(multi.get('occupied', 0))} 个 resident 对局。")
    status = _probe_native("status")
    if (
        status is not None
        and str(status.get("mode", "")).casefold() == "native-render"
        and not allow_owned_native_render
    ):
        conflicts.append("检测到一个不属于本界面的 native-render 会话，为避免覆盖已阻断。")
    return tuple(dict.fromkeys(conflicts))


def _endpoint_is_cold_ready() -> bool:
    status = _probe_native("status", timeout=1.0)
    if status is None:
        return False
    pointer = str(status.get("manager", "")).strip().casefold()
    null_manager = pointer in {"", "0", "0x0", "(nil)", "none"}
    return bool(
        status.get("ok")
        and status.get("coldReady")
        and not status.get("configured")
        and int(status.get("generation", 0)) == 0
        and null_manager
    )


def ensure_offline_runner(progress: Callable[[str], None], *, force_restart: bool = False) -> None:
    if not force_restart and _endpoint_is_cold_ready():
        progress("离线 native runner 已冷就绪，直接复用。")
        return
    if not START_OFFLINE_SCRIPT.is_file():
        raise FileNotFoundError(START_OFFLINE_SCRIPT)
    progress("正在强制重启专用 MuMu native runner…" if force_restart else "正在启动专用 MuMu 与离线 native runner…")
    result = _run_hidden(
        (
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            START_OFFLINE_SCRIPT,
            "-VmIndex",
            str(VM_INDEX),
            "-VmName",
            VM_NAME,
            "-Serial",
            ADB_SERIAL,
            "-ControlPort",
            str(CONTROL_PORT),
            "-GuestControlPort",
            setting("CR_GUEST_CONTROL_PORT", "26789"),
        ),
        timeout=180.0,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "start_offline.ps1 启动失败")
    progress("离线 native runner 已就绪。")


def discover_model_checkpoints() -> tuple[Path, ...]:
    checkpoint_root = REPOSITORY_ROOT / "checkpoints"
    if not checkpoint_root.is_dir():
        return ()
    candidates = tuple(checkpoint_root.rglob("*.pt"))
    return tuple(
        sorted(
            (path.resolve() for path in candidates if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
    )


def _calibrate_collected_replay_with_one_restart(
    prepared: Any,
    *,
    make_native: Callable[[], Any],
    restart_runner: Callable[[], None],
    progress: Callable[[str], None],
) -> tuple[Any, Any]:
    """Calibrate on one runner generation, then allow one clean restart.

    A configuration-load timeout is a runner lifecycle failure, not a deal
    mismatch.  Only the dedicated interactive runner is restarted and a second
    timeout terminates with both generations' exact diagnostics.
    """

    from .royaleapi_replay import (
        NativeRenderConfigurationLoadError,
        RoyaleAPIReplayError,
        calibrate_collected_replay_deal,
    )

    first_failure: NativeRenderConfigurationLoadError | None = None
    for runner_generation in (1, 2):
        native = make_native()
        try:
            return (calibrate_collected_replay_deal(prepared, native, on_attempt=progress), native)
        except NativeRenderConfigurationLoadError as error:
            if runner_generation == 2:
                raise RoyaleAPIReplayError(
                    f"replay {prepared.replay_tag} still failed after exactly "
                    "one dedicated runner restart; "
                    f"first failure: {first_failure}; second failure: {error}"
                ) from error
            first_failure = error
            progress(f"{error}; restarting the dedicated runner once before retrying.")
            restart_runner()
    raise AssertionError("unreachable replay calibration restart state")


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def _form_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if isinstance(item, str))
    return ()


@dataclass(frozen=True, slots=True)
class CardOption:
    card_id: int
    name: str
    supports_evolution: bool
    supports_hero: bool

    @property
    def display(self) -> str:
        chinese, english = CARD_NAMES[self.card_id]
        label = f"{chinese} / {english}"
        return f"{label}  [{self.card_id}]"

    @property
    def forms(self) -> tuple[str, ...]:
        output = [FORM_BASE]
        if self.supports_evolution:
            output.append(FORM_EVOLUTION)
        if self.supports_hero:
            output.append(FORM_HERO)
        return tuple(output)


def load_card_options() -> tuple[CardOption, ...]:
    from .card_specs import build_card_catalog

    payload = build_card_catalog().to_dict()
    options: list[CardOption] = []
    for spec in payload.get("specs", ()):
        if not isinstance(spec, Mapping):
            continue
        attributes = spec.get("attributes", {})
        attributes = attributes if isinstance(attributes, Mapping) else {}
        if _flag(attributes.get("NotVisible")) or _flag(attributes.get("NotInUse")):
            continue
        card_id = int(spec["card_id"])
        names = _form_names(attributes.get("EvolvedSpells"))
        lowered = tuple(item.casefold() for item in names)
        supports_evolution = bool(
            spec.get("evolution")
            or attributes.get("resolved_evolution_form")
            or any("_ev" in item or "evolution" in item for item in lowered if "hero" not in item)
        )
        supports_hero = bool(
            spec.get("ability_ids") or attributes.get("HeroAbility") or any("hero" in item for item in lowered)
        )
        options.append(
            CardOption(
                card_id=card_id,
                name=str(spec.get("name") or card_id),
                supports_evolution=supports_evolution,
                supports_hero=supports_hero,
            )
        )
    options.sort(key=lambda item: (item.display.casefold(), item.card_id))
    return tuple(options)


@dataclass(frozen=True, slots=True)
class MatchPreset:
    name: str
    deck0: tuple[int, ...]
    deck1: tuple[int, ...]
    forms0: tuple[int, ...]
    forms1: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SavedDeck:
    name: str
    deck: tuple[int, ...]
    forms: tuple[int, ...]


def load_saved_decks(path: Path) -> dict[str, SavedDeck]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("decks"), list):
            raise ValueError("文件格式不正确")
        decks: dict[str, SavedDeck] = {}
        for item in data["decks"]:
            if not isinstance(item, dict):
                raise ValueError("牌组条目不正确")
            name, deck, forms = item.get("name"), item.get("deck"), item.get("forms")
            if not isinstance(name, str) or not name.strip() or name != name.strip() or name in decks:
                raise ValueError("牌组名称不正确或重复")
            if (
                not isinstance(deck, list) or len(deck) != 8
                or any(type(card_id) is not int for card_id in deck) or len(set(deck)) != 8
                or not isinstance(forms, list) or len(forms) != 8
                or any(type(mask) is not int or mask not in MASK_TO_FORM for mask in forms)
            ):
                raise ValueError(f"牌组“{name}”的卡牌或形态不正确")
            decks[name] = SavedDeck(name, tuple(deck), tuple(forms))
        return decks
    except (OSError, ValueError) as error:
        raise ValueError(f"无法读取自定义牌组 {path}：{error}") from error


def write_saved_decks(path: Path, decks: Mapping[str, SavedDeck]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix="custom-decks-", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(
                {"version": 1, "decks": [asdict(decks[name]) for name in sorted(decks)]},
                output, ensure_ascii=False, indent=2,
            )
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


MATCH_PRESETS = (
    MatchPreset(
        "P.E.K.K.A Bridge Spam (Evo Ram/Ghost + Hero Magic Archer)",
        PEKKA_BRIDGE_SPAM_DECK,
        PEKKA_BRIDGE_SPAM_DECK,
        PEKKA_BRIDGE_SPAM_FORMS,
        PEKKA_BRIDGE_SPAM_FORMS,
    ),
    MatchPreset(
        "骑士速弩内战（精英骑士 + 觉醒弓箭手/电塔）",
        SPEED_XBOW_DECK,
        SPEED_XBOW_DECK,
        SPEED_XBOW_FORMS,
        SPEED_XBOW_FORMS,
    ),
    MatchPreset("基础速弩内战（全部基础形态）", SPEED_XBOW_DECK, SPEED_XBOW_DECK, (0,) * 8, (0,) * 8),
)


@dataclass(frozen=True, slots=True)
class ReplayRun:
    label: str
    directory: Path


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    path: Path
    completed_match_index: int
    seed: int
    result: str
    ticks: int
    created_at: str


def discover_replay_runs() -> tuple[ReplayRun, ...]:
    def directories(parent: Path) -> list[Path]:
        if not parent.is_dir():
            return []
        return sorted((item for item in parent.iterdir() if item.is_dir()), key=lambda item: item.name.casefold())

    candidates = []
    for root, category in (
        (WORKSPACE_ROOT / "training_replays", "正式训练"),
        (PACKAGE_ROOT / "training_replays", "初始化/旧训练"),
    ):
        candidates.extend((item, f"{category} · {item.name}") for item in directories(root))
    for run in directories(WORKSPACE_ROOT / "runs"):
        candidates.append((run / "replays", f"正式训练 · {run.name}"))
        candidates.extend((child / "replays", f"正式训练 · {run.name} / {child.name}") for child in directories(run))
    output: list[ReplayRun] = []
    seen: set[Path] = set()
    for directory, label in candidates:
        if not directory.is_dir() or not any(directory.glob("match-*.crr.json.gz")):
            continue
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(ReplayRun(label=label, directory=resolved))
    return tuple(output)


def load_replay_entries(directory: Path) -> tuple[ReplayEntry, ...]:
    from .training.replay_archive import TrainingReplayError, TrainingReplayV1, list_training_replays

    output: list[ReplayEntry] = []
    for path in list_training_replays(directory):
        try:
            replay = TrainingReplayV1.load(path)
        except TrainingReplayError as error:
            LOGGER.warning("skipping replay %s: %s", path, error)
            continue
        winner = replay.terminal.get("winner")
        result = "上方 P0 胜" if winner == 0 else "下方 P1 胜" if winner == 1 else "平局"
        output.append(
            ReplayEntry(
                path=path.resolve(),
                completed_match_index=replay.completed_match_index,
                seed=replay.episode_config.seed,
                result=result,
                ticks=replay.end_native_tick - replay.start_native_tick,
                created_at=datetime.fromtimestamp(replay.created_at_ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M"),
            )
        )
    return tuple(output)


class DeckEditor:
    def __init__(
        self, parent: Any, *, title: str, options: Sequence[CardOption],
        on_save: Callable[[DeckEditor], None], on_load: Callable[[DeckEditor], None],
        on_delete: Callable[[DeckEditor], None],
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.options = tuple(options)
        self.by_display = {item.display: item for item in self.options}
        self.by_id = {item.card_id: item for item in self.options}
        self.frame = ttk.LabelFrame(parent, text=title, padding=10)
        saved = ttk.Frame(self.frame)
        saved.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(saved, text="自定义牌组").pack(side="left", padx=(0, 6))
        self.saved_deck_var = tk.StringVar()
        self.saved_deck_box = ttk.Combobox(saved, textvariable=self.saved_deck_var, state="readonly", width=17)
        self.saved_deck_box.pack(side="left", fill="x", expand=True, padx=(0, 6))
        for label, command in (("载入", on_load), ("保存", on_save), ("删除", on_delete)):
            ttk.Button(saved, text=label, command=lambda action=command: action(self)).pack(side="left", padx=(0, 4))
        for column, label in enumerate(("槽位", "卡牌", "形态")):
            ttk.Label(self.frame, text=label, style="Muted.TLabel").grid(
                row=1, column=column, padx=(0, 8 if column < 2 else 0), sticky="w"
            )
        self.card_vars: list[Any] = []
        self.card_boxes: list[Any] = []
        self.selected_cards = [""] * 8
        self.form_vars: list[Any] = []
        self.form_boxes: list[Any] = []
        values = tuple(item.display for item in self.options)
        for slot in range(8):
            ttk.Label(self.frame, text=str(slot + 1)).grid(row=slot + 2, column=0, padx=(0, 8), pady=3, sticky="e")
            for column, choices, width, variables in (
                (1, values, 31, self.card_vars),
                (2, (FORM_BASE,), 14, self.form_vars),
            ):
                variable = tk.StringVar(value=FORM_BASE if column == 2 else "")
                box = ttk.Combobox(
                    self.frame, textvariable=variable, values=choices,
                    state="normal" if column == 1 else "readonly", width=width,
                )
                box.grid(row=slot + 2, column=column, padx=(0, 8 if column == 1 else 0), pady=3, sticky="ew")
                variables.append(variable)
                if column == 1:
                    self.card_boxes.append(box)
                    variable.trace_add("write", lambda *_args, index=slot: self._filter_card(index))
                    box.bind("<FocusIn>", lambda _event, widget=box: widget.selection_range(0, "end"))
                    box.bind(
                        "<FocusOut>",
                        lambda _event, index=slot: self.frame.after_idle(lambda: self._restore_card(index)),
                    )
                    box.bind("<<ComboboxSelected>>", lambda _event, index=slot: self._card_changed(index))
                else:
                    self.form_boxes.append(box)
        self.frame.columnconfigure(1, weight=1)

    def _filter_card(self, index: int) -> None:
        query = self.card_vars[index].get().strip().casefold()
        if self.card_vars[index].get() in self.by_display:
            query = ""
        values = tuple(
            item.display
            for item in self.options
            if not query or any(query in name.casefold() for name in CARD_NAMES[item.card_id])
        )
        self.card_boxes[index].configure(values=values)

    def _restore_card(self, index: int) -> None:
        focus = str(self.frame.tk.call("focus"))
        if focus.startswith(f"{self.card_boxes[index]}.popdown"):
            return
        if self.card_vars[index].get() in self.by_display:
            self._card_changed(index)
        else:
            self.card_vars[index].set(self.selected_cards[index])

    def _card_changed(self, index: int) -> None:
        option = self.by_display.get(self.card_vars[index].get())
        if option is not None:
            self.selected_cards[index] = option.display
        forms = option.forms if option is not None else (FORM_BASE,)
        self.form_boxes[index].configure(values=forms)
        if self.form_vars[index].get() not in forms:
            self.form_vars[index].set(FORM_BASE)

    def set_deck(self, deck: Sequence[int], forms: Sequence[int]) -> None:
        if len(deck) != 8 or len(forms) != 8:
            raise ValueError("牌组和形态必须各有八项")
        for index, (card_id, mask) in enumerate(zip(deck, forms)):
            option = self.by_id.get(int(card_id))
            if option is None:
                raise ValueError(f"卡牌 {card_id} 不在可见目录中")
            self.card_vars[index].set(option.display)
            self._card_changed(index)
            form = MASK_TO_FORM.get(int(mask))
            if form is None or form not in option.forms:
                raise ValueError(f"{option.name} 不支持形态掩码 {mask}")
            self.form_vars[index].set(form)

    def values(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        options: list[CardOption] = []
        masks: list[int] = []
        for slot, (card_var, form_var) in enumerate(zip(self.card_vars, self.form_vars), start=1):
            option = self.by_display.get(card_var.get())
            if option is None:
                raise ValueError(f"第 {slot} 个槽位尚未选择卡牌")
            form = form_var.get()
            if form not in option.forms:
                raise ValueError(f"{option.name} 不支持“{form}”形态")
            options.append(option)
            masks.append(FORM_TO_MASK[form])
        ids = tuple(item.card_id for item in options)
        if len(set(ids)) != 8:
            raise ValueError("同一方牌组不能包含重复卡牌")
        return ids, tuple(masks)


class CRHarnessInterface:
    def __init__(self, root: Any, *, preflight: PreflightReport, log_path: Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.log_path = log_path
        self.card_options = load_card_options()
        user_data = Path(os.environ["LOCALAPPDATA"]) / "FirstLight_CR" if os.environ.get("LOCALAPPDATA") else Path.home() / ".firstlight_cr"
        self.saved_decks_path = user_data / "custom_decks.json"
        self.saved_decks_error: ValueError | None = None
        try:
            self.saved_decks = load_saved_decks(self.saved_decks_path)
        except ValueError as error:
            self.saved_decks = {}
            self.saved_decks_error = error
        self.deck_editors: list[DeckEditor] = []
        self.busy = False
        self.owns_native_session = False
        self.native: Any | None = None
        self.overlay_process: subprocess.Popen[Any] | None = None
        self.overlay_log: Any | None = None
        self.replay_native: Any | None = None
        self.replay_stop_event = threading.Event()
        self.replay_pause_event = threading.Event()
        self.replay_rewind_control: Any | None = None
        self.replay_entries: dict[str, ReplayEntry] = {}
        self.replay_runs: dict[str, ReplayRun] = {}
        self.collected_replay_entries: dict[str, Any] = {}
        self.active_replay_kind: str | None = None
        self.vm3_resetting = False
        self.model_stop_requested = threading.Event()
        self.model_active = False
        self.model_match_kind = "human"
        self.model_after_stop: Callable[[], None] | None = None
        self.model_process: subprocess.Popen[Any] | None = None
        self.model_log: Any | None = None
        self.model_artifact_dir: Path | None = None

        root.title("FirstLight CR 控制台")
        if os.name == "nt":
            root.iconbitmap(default=str(REPOSITORY_ROOT / "docs/assets/firstlight-cr.ico"))
        else:
            self.window_icon = tk.PhotoImage(file=str(REPOSITORY_ROOT / "docs/assets/firstlight-cr.png"))
            root.iconphoto(True, self.window_icon)
        root.geometry("1180x840")
        root.minsize(1080, 760)
        root.configure(bg="#eef2f7")
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#eef2f7")
        style.configure("Header.TFrame", background="#14243a")
        style.configure(
            "HeaderTitle.TLabel", background="#14243a", foreground="#ffffff", font=("Microsoft YaHei UI", 18, "bold")
        )
        style.configure("Muted.TLabel", foreground="#5d6b7b")
        style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="#ffffff",
            background="#2563eb",
            padding=(14, 8),
        )
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])
        style.configure("Danger.TButton", foreground="#991b1b")
        style.configure("TNotebook.Tab", padding=(18, 8))

        self._build_header()
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(12, 8))
        self.model_tab = ttk.Frame(notebook, padding=12)
        self.duel_tab = ttk.Frame(notebook, padding=12)
        self.match_tab = ttk.Frame(notebook, padding=12)
        self.replay_tab = ttk.Frame(notebook, padding=12)
        self.collected_replay_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.model_tab, text="1  Human VS AI")
        notebook.add(self.duel_tab, text="2  AI VS AI")
        notebook.add(self.match_tab, text="3  手动 native 对局")
        notebook.add(self.replay_tab, text="4  训练回放")
        notebook.add(self.collected_replay_tab, text="5  采集回放")
        self._build_model_tab()
        self._build_duel_tab()
        self._build_match_tab()
        self._refresh_saved_deck_choices()
        self._build_replay_tab()
        self._build_collected_replay_tab()
        self._build_footer()
        if self.saved_decks_error is not None:
            self._show_error(self.saved_decks_error)

    def _row(self, parent: Any, title: str = "", **layout: Any) -> Any:
        from tkinter import ttk

        frame = ttk.LabelFrame(parent, text=title, padding=10) if title else ttk.Frame(parent)
        frame.pack(fill="x", **layout)
        return frame

    def _buttons(self, parent: Any, specs: Sequence[tuple[Any, ...]]) -> None:
        """Build an action bar while retaining named controls for state updates."""
        from tkinter import ttk

        for name, label, command, role in specs:
            options = {"style": "Primary.TButton"} if role == "primary" else {}
            if role in {"stop", "danger"}:
                options["state"] = "disabled"
            if role == "danger":
                options["style"] = "Danger.TButton"
            button = ttk.Button(parent, text=label, command=command, **options)
            button.pack(side="right" if role == "right" else "left", padx=(0, 8))
            if name:
                setattr(self, name, button)

    def _input(
        self,
        parent: Any,
        name: str,
        *,
        value: str = "",
        values: Sequence[str] | None = None,
        command: Callable[[], Any] | None = None,
        **options: Any,
    ) -> Any:
        import tkinter as tk
        from tkinter import ttk

        variable = tk.StringVar(value=value)
        setattr(self, name + "_var", variable)
        if values is None:
            widget = ttk.Entry(parent, textvariable=variable, **options)
            event = "<Return>"
        else:
            widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", **options)
            event = "<<ComboboxSelected>>"
        if command:
            widget.bind(event, lambda _event: command())
        return widget

    def _status(
        self, parent: Any, name: str, value: str = "", *, align: str = "e", wraplength: int = 0, **layout: Any
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        variable = tk.StringVar(value=value)
        setattr(self, name + "_var", variable)
        ttk.Label(parent, textvariable=variable, anchor=align, wraplength=wraplength, justify="left").pack(**layout)

    def _note(self, parent: Any, text: str, *, wraplength: int = 0, **layout: Any) -> None:
        from tkinter import ttk

        ttk.Label(parent, text=text, style="Muted.TLabel", wraplength=wraplength, justify="left").pack(**layout)

    def _build_header(self) -> None:
        from tkinter import ttk

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 14))
        header.pack(fill="x")
        title = ttk.Frame(header, style="Header.TFrame")
        title.pack(side="left", fill="y")
        ttk.Label(title, text="FirstLight CR 控制台", style="HeaderTitle.TLabel").pack(anchor="w")

    def _build_model_tab(self) -> None:
        from tkinter import ttk

        settings = self._row(self.model_tab, "离线模型对局")
        ttk.Label(settings, text="Checkpoint").pack(side="left")
        self.model_checkpoint_box = self._input(settings, "model_checkpoint", values=(), width=65)
        self.model_checkpoint_box.pack(side="left", fill="x", expand=True, padx=8)
        self._buttons(
            settings,
            (
                (None, "选择文件…", self.choose_model_checkpoint, ""),
            ),
        )
        decks = ttk.Frame(self.model_tab)
        decks.pack(fill="both", expand=True, pady=(10, 8))
        preset = MATCH_PRESETS[0]
        for owner, title in enumerate(("模型卡组", "你的卡组")):
            editor = DeckEditor(
                decks, title=title, options=self.card_options,
                on_save=self.save_custom_deck, on_load=self.load_custom_deck, on_delete=self.delete_custom_deck,
            )
            editor.frame.pack(side="left", fill="both", expand=True, padx=(6 if owner else 0, 0 if owner else 6))
            self.deck_editors.append(editor)
            editor.set_deck(
                preset.deck0 if owner == 0 else preset.deck1, preset.forms0 if owner == 0 else preset.forms1
            )
            setattr(self, f"model_deck{owner}_editor", editor)
        actions = self._row(self.model_tab)
        self._buttons(
            actions,
            (
                ("start_model_button", "开始模型对局", self.start_model_match, "primary"),
                ("stop_model_button", "停止对局", self.stop_model_match, "stop"),
                (None, "Reset 离线 VM 进程", self.reset_vm3, "right"),
            ),
        )
        self._status(
            self.model_tab, "model_status", "选择模型与双方牌组，然后开始离线对局。", fill="x", pady=(8, 0), align="w"
        )
        self.refresh_model_checkpoints()

    def _build_duel_tab(self) -> None:
        from tkinter import ttk

        checkpoints = self._row(self.duel_tab, "双方模型")
        for owner, label in enumerate(("上方 AI Checkpoint", "下方 AI Checkpoint")):
            row = self._row(checkpoints, pady=(0 if owner == 0 else 8, 0))
            ttk.Label(row, text=label).pack(side="left")
            box = self._input(row, f"duel_checkpoint{owner}", values=(), width=65)
            box.pack(side="left", fill="x", expand=True, padx=8)
            setattr(self, f"duel_checkpoint{owner}_box", box)
            self._buttons(row, ((None, "选择文件…", lambda side=owner: self.choose_duel_checkpoint(side), ""),))
        decks = ttk.Frame(self.duel_tab)
        decks.pack(fill="both", expand=True, pady=(10, 8))
        preset = MATCH_PRESETS[0]
        for owner, title in enumerate(("上方 AI 卡组", "下方 AI 卡组")):
            editor = DeckEditor(
                decks, title=title, options=self.card_options,
                on_save=self.save_custom_deck, on_load=self.load_custom_deck, on_delete=self.delete_custom_deck,
            )
            editor.frame.pack(side="left", fill="both", expand=True, padx=(6 if owner else 0, 0 if owner else 6))
            self.deck_editors.append(editor)
            editor.set_deck(
                preset.deck0 if owner == 0 else preset.deck1, preset.forms0 if owner == 0 else preset.forms1
            )
            setattr(self, f"duel_deck{owner}_editor", editor)
        actions = self._row(self.duel_tab)
        self._buttons(actions, (
            ("start_duel_button", "开始 AI 对局", self.start_ai_duel, "primary"),
            ("stop_duel_button", "停止对局", self.stop_model_match, "stop"),
            ("force_duel_0_button", "强制上方下牌", lambda: self.force_ai_play(0), "stop"),
            ("force_duel_1_button", "强制下方下牌", lambda: self.force_ai_play(1), "stop"),
            (None, "Reset 离线 VM 进程", self.reset_vm3, "right"),
        ))
        self._status(self.duel_tab, "duel_status", "选择双方模型和卡组，然后开始对局。", fill="x", pady=(8, 0), align="w")
        self.refresh_duel_checkpoints()

    def refresh_duel_checkpoints(self) -> None:
        values = tuple(str(path) for path in discover_model_checkpoints())
        for owner in (0, 1):
            box = getattr(self, f"duel_checkpoint{owner}_box")
            variable = getattr(self, f"duel_checkpoint{owner}_var")
            box.configure(values=values)
            if variable.get() not in values:
                variable.set(values[0] if values else "")

    def choose_duel_checkpoint(self, owner: int) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root, title=f"选择{'上方' if owner == 0 else '下方'} AI Checkpoint",
            initialdir=str(REPOSITORY_ROOT / "checkpoints"),
            filetypes=(("PyTorch checkpoint", "*.pt"), ("所有文件", "*.*")),
        )
        if selected:
            box = getattr(self, f"duel_checkpoint{owner}_box")
            value = str(Path(selected).resolve())
            box.configure(values=(value, *tuple(item for item in box.cget("values") if item != value)))
            getattr(self, f"duel_checkpoint{owner}_var").set(value)

    def _build_match_tab(self) -> None:
        from tkinter import ttk

        decks = ttk.Frame(self.match_tab)
        decks.pack(fill="both", expand=True, pady=(12, 8))
        for owner, title in enumerate(("上方 · 红方（owner 0）", "下方 · 蓝方（owner 1）")):
            editor = DeckEditor(
                decks, title=title, options=self.card_options,
                on_save=self.save_custom_deck, on_load=self.load_custom_deck, on_delete=self.delete_custom_deck,
            )
            editor.frame.pack(side="left", fill="both", expand=True, padx=(6 if owner else 0, 0 if owner else 6))
            self.deck_editors.append(editor)
            setattr(self, f"deck{owner}_editor", editor)
        actions = self._row(self.match_tab)
        self._buttons(
            actions,
            (
                ("start_match_button", "启动 native 对局并打开鼠标覆盖层", self.start_match, "primary"),
                (None, "Reset VM3 进程", self.reset_vm3, ""),
            ),
        )
        preset = MATCH_PRESETS[0]
        self.deck0_editor.set_deck(preset.deck0, preset.forms0)
        self.deck1_editor.set_deck(preset.deck1, preset.forms1)

    def _replay_table(self, tab: Any, columns: Mapping[str, tuple[str, int]], start: Callable[[], None]) -> Any:
        from tkinter import ttk

        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True, pady=(12, 8))
        tree = ttk.Treeview(frame, columns=tuple(columns), show="headings", selectmode="browse")
        for name, (label, width) in columns.items():
            tree.heading(name, text=label)
            tree.column(name, width=width)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        tree.bind("<Double-1>", lambda _event: start())
        return tree

    def _build_replay_controls(self, tab: Any, *, collected: bool) -> None:
        from tkinter import ttk

        prefix = "collected_replay" if collected else "replay"
        controls = self._row(tab)
        specs = [
            (
                f"start_{prefix}_button",
                "Native 播放选中 Replay" if collected else "播放选中回放",
                self.start_collected_replay if collected else self.start_replay,
                "primary",
            ),
            (f"pause_{prefix}_button", "暂停", self.toggle_replay_pause, "stop"),
        ]
        if collected:
            specs.append(("rewind_collected_replay_button", "后退 5 秒", self.rewind_collected_replay, "stop"))
        specs.extend(
            ((f"stop_{prefix}_button", "停止", self.stop_replay, "stop"), (None, "Reset VM3 进程", self.reset_vm3, ""))
        )
        self._buttons(controls, specs)
        ttk.Label(controls, text="倍速").pack(side="left", padx=(14, 6))
        self._input(
            controls,
            prefix + "_speed",
            value="1",
            values=("0.25", "0.5", "1", "2", "4"),
            width=6,
            command=self.set_collected_replay_speed if collected else self.set_replay_speed,
        ).pack(side="left")
        self._status(controls, prefix + "_status", "尚未播放", side="right", fill="x", expand=True)

    def _build_replay_tab(self) -> None:
        from tkinter import ttk

        top = self._row(self.replay_tab, "训练与回放")
        ttk.Label(top, text="训练目录").pack(side="left")
        self.replay_run_box = self._input(top, "replay_run", values=(), width=44, command=self.refresh_replays)
        self.replay_run_box.pack(side="left", padx=8)
        self._buttons(top, ((None, "刷新回放", self.refresh_replay_runs, ""),))
        self.replay_tree = self._replay_table(
            self.replay_tab,
            {
                "match": ("完成序号", 100),
                "seed": ("Seed", 100),
                "result": ("结果", 100),
                "duration": ("记录时长", 100),
                "created": ("保存时间", 145),
                "file": ("文件", 440),
            },
            self.start_replay,
        )
        self._build_replay_controls(self.replay_tab, collected=False)
        self.refresh_replay_runs()

    def _build_collected_replay_tab(self) -> None:
        from tkinter import ttk
        from .royaleapi_replay import DEFAULT_DATASET_ROOT

        top = self._row(self.collected_replay_tab, "RoyaleAPI 采集数据集（只读）")
        dataset_row = self._row(top)
        ttk.Label(dataset_row, text="数据目录").pack(side="left")
        self._input(
            dataset_row, "collected_dataset", value=os.fspath(DEFAULT_DATASET_ROOT.resolve()), state="readonly"
        ).pack(side="left", fill="x", expand=True, padx=8)
        self._buttons(
            dataset_row,
            (
                (None, "选择目录…", self.choose_collected_replay_directory, ""),
                (None, "恢复默认", self.reset_collected_replay_directory, ""),
            ),
        )
        filter_row = self._row(top, pady=(8, 0))
        ttk.Label(filter_row, text="Replay / 玩家 Tag").pack(side="left")
        self._input(filter_row, "collected_query", width=22, command=self.refresh_collected_replays).pack(
            side="left", padx=8
        )
        ttk.Label(filter_row, text="显示").pack(side="left")
        self._input(filter_row, "collected_limit", value="500", values=("100", "500", "1000", "5000"), width=7).pack(
            side="left", padx=8
        )
        self._buttons(filter_row, ((None, "刷新列表", self.refresh_collected_replays, ""),))
        self.collected_replay_tree = self._replay_table(
            self.collected_replay_tab,
            {
                "played": ("对战时间（上海）", 185),
                "player": ("采集源玩家", 155),
                "replay": ("Replay Tag", 180),
                "storage": ("只读存储位置", 210),
            },
            self.start_collected_replay,
        )
        self._build_replay_controls(self.collected_replay_tab, collected=True)
        self.refresh_collected_replays()

    def _build_footer(self) -> None:
        import tkinter as tk

        footer = self._row(self.root, padx=14, pady=(0, 10))
        self._note(footer, f"日志：{self.log_path}", side="right")
        self.log_text = tk.Text(
            self.root,
            height=4,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            background="#ffffff",
            foreground="#334155",
            relief="flat",
            state="disabled",
        )
        self.log_text.pack(fill="x", padx=14, pady=(0, 12))

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        LOGGER.info(message)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _post(self, callback: Callable[..., Any], *args: Any) -> None:
        self.root.after(0, callback, *args)

    def _background(
        self,
        name: str,
        work: Callable[[], Any],
        done: Callable[[Any], None] | None = None,
        failed: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Keep blocking work off Tk and deliver completion/errors on its thread."""

        def run() -> None:
            try:
                result = work()
            except BaseException as error:
                LOGGER.exception("%s failed", name)
                self._post(failed or self._show_error, error)
            else:
                if done:
                    self._post(done, result)

        threading.Thread(target=run, name=f"interface-{name}", daemon=True).start()

    def _report(self, message: str, status: Any = None, *, log: bool = True) -> None:
        if status is not None:
            status.set(message)
        if log:
            self._append_log(message)

    def _progress(self, message: str) -> None:
        self._post(self._append_log, message)

    def refresh_model_checkpoints(self) -> None:
        checkpoints = discover_model_checkpoints()
        values = tuple(str(path) for path in checkpoints)
        selected = self.model_checkpoint_var.get()
        self.model_checkpoint_box.configure(values=values)
        if selected in values:
            return
        self.model_checkpoint_var.set(values[0] if values else "")
        if not values:
            self.model_status_var.set("没有找到 V4 checkpoint。")

    def choose_model_checkpoint(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="选择 V4 checkpoint",
            initialdir=str(REPOSITORY_ROOT / "checkpoints"),
            filetypes=(("PyTorch checkpoint", "*.pt"), ("所有文件", "*.*")),
        )
        if not selected:
            return
        path = Path(selected).resolve()
        values = tuple(self.model_checkpoint_box.cget("values"))
        text = str(path)
        if text not in values:
            values = (text, *values)
            self.model_checkpoint_box.configure(values=values)
        self.model_checkpoint_var.set(text)

    def _set_model_active(self, active: bool) -> None:
        self.model_active = active
        self._set_busy(active)
        self.model_checkpoint_box.configure(state="disabled" if active else "readonly")
        for owner in (0, 1):
            getattr(self, f"duel_checkpoint{owner}_box").configure(state="disabled" if active else "readonly")
            getattr(self, f"force_duel_{owner}_button").configure(state="disabled")
        self.stop_model_button.configure(state="normal" if active and self.model_match_kind == "human" else "disabled")
        self.stop_duel_button.configure(state="normal" if active and self.model_match_kind == "duel" else "disabled")

    def force_ai_play(self, owner: int) -> None:
        if self.model_match_kind != "duel" or not self.model_active or self.model_artifact_dir is None:
            return
        if self.model_process is None or self.model_process.poll() is not None:
            return
        (self.model_artifact_dir / f"force-{owner}").touch()

    def start_ai_duel(self) -> None:
        from .match_factory import MatchConfig

        if self.busy:
            return
        try:
            checkpoints = tuple(
                Path(getattr(self, f"duel_checkpoint{owner}_var").get()).resolve() for owner in (0, 1)
            )
            for checkpoint in checkpoints:
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            deck0, forms0 = self.duel_deck0_editor.values()
            deck1, forms1 = self.duel_deck1_editor.values()
            validate_model_deck_roles(forms0)
            validate_model_deck_roles(forms1)
            config = MatchConfig(
                deck0=deck0, deck1=deck1,
                deck0_form_availability=forms0, deck1_form_availability=forms1,
                seed=20260728, level_cap=11, minimum_card_level=11, king_tower_level=11,
                owner0_name="AI-0", owner1_name="AI-1",
            )
        except (ValueError, OSError) as error:
            self._show_error(error)
            return
        if not self._confirm_replace_owned_session("AI 对局") or not self._check_operation_conflicts():
            return
        self.stop_replay()
        self._terminate_overlay()
        self.model_match_kind = "duel"
        self._set_model_active(True)
        self.model_stop_requested = threading.Event()
        self.duel_status_var.set("正在准备离线 VM 和双方模型…")
        artifact = REPOSITORY_ROOT / "runs" / "interface" / f"duel-{time.time_ns()}"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "match.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
        self.model_artifact_dir = artifact

        def run():
            if not self.owns_native_session:
                ensure_offline_runner(self._progress)
            if self.model_stop_requested.is_set():
                return None, 0
            process, log = self._launch_python(
                "native_runner.training.v4.offline_duel", artifact / "console.log",
                checkpoint_0=checkpoints[0], checkpoint_1=checkpoints[1],
                match_config=artifact / "match.json", host=CONTROL_HOST, port=CONTROL_PORT,
                speed=1.0, artifact_dir=artifact,
            )
            self.model_process, self.model_log = process, log
            announced = False
            while process.poll() is None:
                if self.model_stop_requested.is_set():
                    (artifact / "stop").touch()
                if not announced and (artifact / "ready.json").is_file():
                    announced = True
                    self._post(self._model_ready, process, 0)
                time.sleep(0.1)
            return process, process.returncode

        def failed(error):
            self._set_model_active(False)
            self.duel_status_var.set(f"AI 对局失败：{error}")
            self._show_error(error)
            self._after_model_stopped()

        self._background("offline-duel", run, lambda result: self._model_finished(*result), failed)

    def start_model_match(self) -> None:
        from .match_factory import MatchConfig

        if self.busy:
            return
        try:
            checkpoint = Path(self.model_checkpoint_var.get()).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            owner = 0
            level = 11
            deck0, forms0 = self.model_deck0_editor.values()
            deck1, forms1 = self.model_deck1_editor.values()
            validate_model_deck_roles(forms0)
            config = MatchConfig(
                deck0=deck0,
                deck1=deck1,
                deck0_form_availability=forms0,
                deck1_form_availability=forms1,
                seed=20260728,
                level_cap=level,
                minimum_card_level=level,
                king_tower_level=level,
                owner0_name="Model" if owner == 0 else "Human",
                owner1_name="Model" if owner == 1 else "Human",
            )
            speed = 1.0
        except (ValueError, OSError) as error:
            self._show_error(error)
            return
        if not self._confirm_replace_owned_session("模型对局") or not self._check_operation_conflicts():
            return
        self.stop_replay()
        self._terminate_overlay()
        self.model_match_kind = "human"
        self._set_model_active(True)
        self.model_stop_requested = threading.Event()
        self.model_status_var.set("正在准备离线 VM 和模型…")
        artifact = REPOSITORY_ROOT / "runs" / "interface" / f"model-{time.time_ns()}"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "match.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
        self.model_artifact_dir = artifact

        def run():
            if not self.owns_native_session:
                ensure_offline_runner(self._progress)
            if self.model_stop_requested.is_set():
                return None, 0
            process, log = self._launch_python(
                "native_runner.training.v4.offline_agent",
                artifact / "console.log",
                checkpoint=checkpoint,
                match_config=artifact / "match.json",
                actor_owner=owner,
                host=CONTROL_HOST,
                port=CONTROL_PORT,
                speed=speed,
                deterministic=None,
                artifact_dir=artifact,
            )
            self.model_process, self.model_log = process, log
            announced = False
            while process.poll() is None:
                if self.model_stop_requested.is_set():
                    (artifact / "stop").touch()
                if not announced and (artifact / "ready.json").is_file():
                    announced = True
                    self._post(self._model_ready, process, owner)
                time.sleep(0.1)
            return process, process.returncode

        def failed(error):
            self._set_model_active(False)
            self.model_status_var.set(f"模型对局失败：{error}")
            self._show_error(error)
            self._after_model_stopped()

        self._background("offline-model", run, lambda result: self._model_finished(*result), failed)

    def _model_ready(self, process, owner: int) -> None:
        if self.model_process is not process or self.model_stop_requested.is_set():
            return
        self.owns_native_session = True
        from .cr_native_env import NativeClashEnv

        self.native = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=15.0)
        if self.model_match_kind == "duel":
            for side in (0, 1):
                getattr(self, f"force_duel_{side}_button").configure(state="normal")
            self.duel_status_var.set("双方 AI 正在对局。")
            return
        self._launch_overlay(owner=1 - owner)
        self.model_status_var.set(f"模型控制{'上方' if owner == 0 else '下方'}；你控制另一方。")

    def _model_finished(self, process, exit_code: int) -> None:
        if self.model_process is not process:
            return
        self.model_process = None
        if self.model_log is not None:
            self.model_log.close()
            self.model_log = None
        self._terminate_overlay()
        self._set_model_active(False)
        result_path = self.model_artifact_dir / "result.json"
        status = self.duel_status_var if self.model_match_kind == "duel" else self.model_status_var
        if exit_code == 0 and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            label = "对局结束" if result.get("terminated") else "对局已停止"
            decisions = result.get("decisions", 0)
            status.set(f"{label} · 模型决策 {sum(decisions) if isinstance(decisions, list) else decisions} 次")
        else:
            status.set(
                f"模型进程已退出（exit={exit_code}），日志：{self.model_artifact_dir / 'console.log'}"
            )

        self._after_model_stopped()

    def _after_model_stopped(self) -> None:
        callback, self.model_after_stop = self.model_after_stop, None
        if callback is not None:
            callback()

    def stop_model_match(self) -> None:
        self.model_stop_requested.set()
        if self.model_artifact_dir is not None:
            (self.model_artifact_dir / "stop").touch()
        (self.duel_status_var if self.model_match_kind == "duel" else self.model_status_var).set(
            "正在停止离线模型对局…"
        )

    def _wait_model_stopped(self) -> None:
        process = self.model_process
        if process is None:
            return
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def _refresh_saved_deck_choices(self) -> None:
        names = tuple(sorted(self.saved_decks, key=str.casefold))
        for editor in self.deck_editors:
            editor.saved_deck_box.configure(values=names)
            if editor.saved_deck_var.get() not in self.saved_decks:
                editor.saved_deck_var.set("")

    def _saved_decks_available(self) -> bool:
        if self.saved_decks_error is not None:
            self._show_error(self.saved_decks_error)
            return False
        return True

    def save_custom_deck(self, editor: DeckEditor) -> None:
        from tkinter import messagebox, simpledialog

        if not self._saved_decks_available():
            return
        try:
            deck, forms = editor.values()
        except ValueError as error:
            self._show_error(error)
            return
        name = simpledialog.askstring(
            "保存自定义牌组", "给当前牌组起个名字：", initialvalue=editor.saved_deck_var.get(), parent=self.root
        )
        if name is None:
            return
        name = name.strip()
        if not name or len(name) > 50 or any(ord(char) < 32 for char in name):
            self._show_error(ValueError("牌组名称需要 1–50 个字符，且不能包含换行或控制字符"))
            return
        if name in self.saved_decks and not messagebox.askyesno(
            "覆盖自定义牌组", f"“{name}”已存在，是否用当前牌组覆盖？", parent=self.root
        ):
            return
        updated = {**self.saved_decks, name: SavedDeck(name, deck, forms)}
        try:
            write_saved_decks(self.saved_decks_path, updated)
        except OSError as error:
            self._show_error(error)
            return
        self.saved_decks = updated
        self._refresh_saved_deck_choices()
        editor.saved_deck_var.set(name)
        self._append_log(f"已保存自定义牌组：{name}")

    def load_custom_deck(self, editor: DeckEditor) -> None:
        if not self._saved_decks_available():
            return
        name = editor.saved_deck_var.get()
        saved = self.saved_decks.get(name)
        if saved is None:
            self._show_error(ValueError("请先选择要载入的自定义牌组"))
            return
        try:
            editor.set_deck(saved.deck, saved.forms)
        except ValueError as error:
            self._show_error(error)
            return
        self._append_log(f"已载入自定义牌组：{name}")

    def delete_custom_deck(self, editor: DeckEditor) -> None:
        from tkinter import messagebox

        if not self._saved_decks_available():
            return
        name = editor.saved_deck_var.get()
        if name not in self.saved_decks:
            self._show_error(ValueError("请先选择要删除的自定义牌组"))
            return
        if not messagebox.askyesno("删除自定义牌组", f"确定删除“{name}”？", parent=self.root):
            return
        updated = {key: deck for key, deck in self.saved_decks.items() if key != name}
        try:
            write_saved_decks(self.saved_decks_path, updated)
        except OSError as error:
            self._show_error(error)
            return
        self.saved_decks = updated
        self._refresh_saved_deck_choices()
        self._append_log(f"已删除自定义牌组：{name}")

    def _match_config(self) -> Any:
        from .match_factory import MatchConfig

        deck0, forms0 = self.deck0_editor.values()
        deck1, forms1 = self.deck1_editor.values()
        level = 11
        return MatchConfig(
            deck0=deck0,
            deck1=deck1,
            deck0_form_availability=forms0,
            deck1_form_availability=forms1,
            seed=20260728,
            level_cap=level,
            minimum_card_level=level,
            king_tower_level=level,
            owner0_name="PEKKA-11-A",
            owner1_name="PEKKA-11-B",
        )

    def _owned_pids(self) -> tuple[int, ...]:
        if self.overlay_process is not None and self.overlay_process.poll() is None:
            return (self.overlay_process.pid,)
        return ()

    def _check_operation_conflicts(self) -> bool:
        from tkinter import messagebox

        conflicts = runtime_conflicts(
            allow_owned_native_render=self.owns_native_session, ignored_pids=self._owned_pids()
        )
        if not conflicts:
            return True
        messagebox.showerror(
            "引擎正在被占用",
            "\n\n".join(conflicts) + f"\n\n不会覆盖 {VM_NAME} 上的其他会话；训练实例不在本操作范围内。",
            parent=self.root,
        )
        return False

    def _confirm_replace_owned_session(self, operation: str) -> bool:
        from tkinter import messagebox

        if not self.owns_native_session:
            return True
        return messagebox.askyesno(
            "替换当前 native 会话",
            f"当前界面已经启动了一个 native 会话。\n\n是否用新的{operation}替换它？",
            parent=self.root,
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.start_model_button.configure(state=state)
        self.start_duel_button.configure(state=state)
        self.start_match_button.configure(state=state)
        self.start_replay_button.configure(state=state)
        self.start_collected_replay_button.configure(state=state)

    def _replay_controls(self, kind: str) -> tuple[Any, Any, Any]:
        if kind == "collected":
            return (
                self.pause_collected_replay_button,
                self.stop_collected_replay_button,
                self.collected_replay_status_var,
            )
        return (self.pause_replay_button, self.stop_replay_button, self.replay_status_var)

    def _set_replay_controls_active(self, kind: str, active: bool) -> None:
        pause, stop, _status = self._replay_controls(kind)
        pause.configure(state="normal" if active else "disabled", text="暂停")
        stop.configure(state="normal" if active else "disabled")
        if kind == "collected":
            self.rewind_collected_replay_button.configure(state="normal" if active else "disabled")

    def reset_vm3(self) -> None:
        if self.model_active:
            self.model_after_stop = self.reset_vm3
            self.stop_model_match()
            self._background("stop-model", self._wait_model_stopped)
            return
        if self.vm3_resetting:
            return
        self.vm3_resetting = True
        self.stop_model_match()
        self.stop_replay()
        self._terminate_overlay()
        self.active_replay_kind = None
        self.replay_rewind_control = None
        self.replay_native = None
        self.native = None
        self.owns_native_session = False
        self._set_replay_controls_active("training", False)
        self._set_replay_controls_active("collected", False)
        self._set_busy(True)
        self._append_log("Reset VM3：强制重启交互式 native 进程…")

        self._background(
            "reset-vm3",
            lambda: (self._wait_model_stopped(), ensure_offline_runner(self._progress, force_restart=True)),
            lambda _result: self._vm3_reset_finished(None),
            self._vm3_reset_finished,
        )

    def _vm3_reset_finished(self, error: BaseException | None) -> None:
        self.vm3_resetting = False
        self._set_busy(False)
        if error is not None:
            self._show_error(error)
            return
        self._append_log("Reset VM3 完成；可重新开始对局或回放。")

    def start_match(self) -> None:
        if self.busy:
            return
        try:
            config = self._match_config()
        except BaseException as error:
            self._show_error(error)
            return
        if not self._confirm_replace_owned_session("手动对局"):
            return
        if not self._check_operation_conflicts():
            return
        self.stop_replay()
        self.active_replay_kind = None
        self._set_busy(True)
        self._append_log("准备创建 native-render 对局…")

        def run() -> None:
            if not self.owns_native_session:
                ensure_offline_runner(self._progress)
            from .cr_native_env import NativeClashEnv

            native = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=15.0)
            native.wait_ready(timeout=20.0)
            self._progress("正在载入双方牌组和 11 级 native 场景…")
            native.create_native_match(config, wait_timeout=30.0)
            self.native = native
            self._launch_overlay()

        self._background("match-launch", run, lambda _result: self._match_ready(), self._operation_failed)

    def _pythonw(self) -> Path:
        executable = Path(sys.executable)
        candidate = executable.with_name("pythonw.exe")
        return candidate if candidate.is_file() else executable

    def _launch_python(self, module: str, log_path: Path, **options: Any) -> tuple[Any, Any]:
        """Launch every interface-owned Python process from this repository."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        command = [os.fspath(self._pythonw()), "-m", module]
        for name, value in options.items():
            command.append("--" + name.replace("_", "-"))
            if value is not None:
                command.append(str(value))
        try:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                creationflags=CREATE_NO_WINDOW,
            )
        except BaseException:
            log_handle.close()
            raise
        return process, log_handle

    def _launch_overlay(self, *, owner: int | None = None) -> None:
        if self.overlay_process is None or self.overlay_process.poll() is not None:
            self.overlay_process, self.overlay_log = self._launch_python(
                "native_runner.native_overlay",
                WORKSPACE_ROOT / "runs" / "interface" / "native-overlay.log",
                runner_host=CONTROL_HOST,
                runner_port=CONTROL_PORT,
                window_title=VM_NAME,
                **({"owner": owner} if owner is not None else {}),
            )

    def _match_ready(self) -> None:
        self.owns_native_session = True
        self._set_busy(False)
        self._append_log("对局已启动；请切到 MuMu，点击上方或下方手牌后再点击场地。")

    def refresh_replay_runs(self) -> None:
        runs = discover_replay_runs()
        self.replay_runs = {item.label: item for item in runs}
        labels = tuple(self.replay_runs)
        self.replay_run_box.configure(values=labels)
        if labels:
            if self.replay_run_var.get() not in self.replay_runs:
                self.replay_run_var.set(labels[0])
            self.refresh_replays()
        else:
            self.replay_run_var.set("")
            self.replay_tree.delete(*self.replay_tree.get_children())
            self.replay_status_var.set("没有找到训练回放")

    def refresh_replays(self) -> None:
        run = self.replay_runs.get(self.replay_run_var.get())
        self.replay_tree.delete(*self.replay_tree.get_children())
        self.replay_entries.clear()
        if run is None:
            return
        entries = load_replay_entries(run.directory)
        for entry in entries:
            item_id = self.replay_tree.insert(
                "",
                "end",
                values=(
                    entry.completed_match_index,
                    entry.seed,
                    entry.result,
                    f"{entry.ticks / 20.0:.1f} 秒",
                    entry.created_at,
                    entry.path.name,
                ),
            )
            self.replay_entries[item_id] = entry
        children = self.replay_tree.get_children()
        if children:
            self.replay_tree.selection_set(children[0])
        self.replay_status_var.set(f"{len(entries)} 个已校验回放 · {run.directory}")

    def _selected_replay(self) -> ReplayEntry | None:
        selected = self.replay_tree.selection()
        return self.replay_entries.get(selected[0]) if selected else None

    def choose_collected_replay_directory(self) -> None:
        from tkinter import filedialog

        current = Path(self.collected_dataset_var.get()).expanduser()
        initial_directory = current if current.is_dir() else WORKSPACE_ROOT / "datasets"
        selected = filedialog.askdirectory(
            title="选择 RoyaleAPI 回放数据目录",
            initialdir=os.fspath(initial_directory),
            mustexist=True,
            parent=self.root,
        )
        if not selected:
            return
        self.collected_dataset_var.set(os.fspath(Path(selected).resolve()))
        self.refresh_collected_replays()

    def reset_collected_replay_directory(self) -> None:
        from .royaleapi_replay import DEFAULT_DATASET_ROOT

        self.collected_dataset_var.set(os.fspath(DEFAULT_DATASET_ROOT.resolve()))
        self.refresh_collected_replays()

    def refresh_collected_replays(self) -> None:
        from .royaleapi_replay import DEFAULT_DATASET_ROOT, list_collected_replays

        self.collected_replay_tree.delete(*self.collected_replay_tree.get_children())
        self.collected_replay_entries.clear()
        dataset_root = Path(self.collected_dataset_var.get()).expanduser().resolve()
        using_default = dataset_root == DEFAULT_DATASET_ROOT.resolve()
        try:
            entries = list_collected_replays(
                dataset_root=dataset_root,
                limit=int(self.collected_limit_var.get()),
                query=self.collected_query_var.get(),
                personal_dataset_roots=None if using_default else (),
            )
        except BaseException as error:
            self.collected_replay_status_var.set(f"读取列表失败：{error}")
            LOGGER.exception("collected replay listing failed")
            return
        for entry in entries:
            item_id = self.collected_replay_tree.insert(
                "", "end", values=(entry.played_at_display, entry.requested_player_tag, entry.replay_tag, entry.storage)
            )
            self.collected_replay_entries[item_id] = entry
        children = self.collected_replay_tree.get_children()
        if children:
            self.collected_replay_tree.selection_set(children[0])
        suffix = f" · 筛选 {self.collected_query_var.get().strip()}" if self.collected_query_var.get().strip() else ""
        scope = "默认目录 + 个人数据集" if using_default else f"仅目录 {dataset_root.name}"
        self.collected_replay_status_var.set(f"显示 {len(entries)} 条采集回放{suffix} · {scope}")

    def _selected_collected_replay(self) -> Any | None:
        selected = self.collected_replay_tree.selection()
        return self.collected_replay_entries.get(selected[0]) if selected else None

    def start_collected_replay(self) -> None:
        self._start_replay("collected")

    def start_replay(self) -> None:
        self._start_replay("training")

    def _start_replay(self, kind: str) -> None:
        from tkinter import messagebox
        from .training.replay_viewer import ReplayRewindControl, play_training_replay

        if self.busy:
            return
        collected = kind == "collected"
        entry = self._selected_collected_replay() if collected else self._selected_replay()
        if entry is None:
            messagebox.showinfo(
                "选择 Replay" if collected else "选择回放",
                "请先选择一条 RoyaleAPI 采集回放。" if collected else "请先选择一局训练回放。",
                parent=self.root,
            )
            return
        if not self._confirm_replace_owned_session("采集回放" if collected else "回放"):
            return
        if not self._check_operation_conflicts():
            return
        self._terminate_overlay()
        self._set_busy(True)
        self.replay_stop_event = threading.Event()
        self.replay_pause_event = threading.Event()
        self.replay_rewind_control = ReplayRewindControl() if collected else None
        self.active_replay_kind = kind
        self._set_replay_controls_active(kind, True)
        _pause, _stop, status = self._replay_controls(kind)
        status.set("正在读取只读存储并推断双方初始牌序…" if collected else "正在准备 native renderer…")
        label = f"RoyaleAPI replay {entry.replay_tag}" if collected else f"训练回放 #{entry.completed_match_index}"
        self._append_log(f"准备播放 {label}…")
        speed_var = self.collected_replay_speed_var if collected else self.replay_speed_var

        def on_native_ready(native: Any) -> None:
            self._own_replay_native(native)
            self._post(status.set, "native renderer 已就绪，正在回放。")

        def on_progress(value: Any) -> None:
            rewound = collected and str(value.phase).startswith("rewound")
            if rewound:
                limited = "（已到最早可用快照）" if value.phase == "rewound-limit" else ""
                message = f"已后退至 {value.native_tick / 20.0:.1f} 秒{limited} · tick {value.native_tick}"
            else:
                noun = "动作" if collected else "操作"
                message = (
                    f"{value.phase} · {noun} {value.operation_index}/{value.operation_count} · "
                    f"tick {value.native_tick} · 漂移 {value.tick_drift:+d}"
                )
            self._post(lambda: self._report(message, status, log=rewound))

        def run() -> Any:
            if collected:
                replay = self._prepare_collected_replay(entry)
                options = dict(strict_ticks=True, stop_at_replay_end=True, rewind_control=self.replay_rewind_control)
            else:
                from .training.replay_archive import TrainingReplayV1

                # Replay preparation creates a scratch slot on a cold runner.
                ensure_offline_runner(self._progress)
                replay = TrainingReplayV1.load(entry.path)
                options = dict(resident_safe_env_id=0)
            return play_training_replay(
                replay,
                host=CONTROL_HOST,
                port=CONTROL_PORT,
                speed=float(speed_var.get()),
                stop_event=self.replay_stop_event,
                pause_event=self.replay_pause_event,
                on_native_ready=on_native_ready,
                on_progress=on_progress,
                **options,
            )

        self._background(
            f"{kind}-replay", run, lambda result: self._replay_finished(result, kind=kind), self._operation_failed
        )

    def _own_replay_native(self, native: Any) -> None:
        self.replay_native = native
        self.native = native
        self.owns_native_session = True

    def _prepare_collected_replay(self, entry: Any) -> Any:
        from .royaleapi_replay import load_collected_replay_payload, prepare_collected_replay
        from .cr_native_env import NativeClashEnv

        prepared = prepare_collected_replay(load_collected_replay_payload(entry))
        self._progress(f"已载入 {prepared.event_count} 个事件；已为双方各随机选取一组合法牌序。")
        for warning in prepared.warnings:
            self._progress(f"采集回放说明：{warning}")
        ensure_offline_runner(self._progress)
        initial_warning_count = len(prepared.warnings)

        def make_native() -> NativeClashEnv:
            candidate = NativeClashEnv(CONTROL_HOST, CONTROL_PORT, timeout=15.0)
            candidate.wait_ready(timeout=20.0)
            return candidate

        prepared, native = _calibrate_collected_replay_with_one_restart(
            prepared,
            make_native=make_native,
            restart_runner=lambda: ensure_offline_runner(self._progress, force_restart=True),
            progress=self._progress,
        )
        self._own_replay_native(native)
        for warning in prepared.warnings[initial_warning_count:]:
            self._progress(f"采集回放说明：{warning}")
        return prepared.replay

    def toggle_replay_pause(self) -> None:
        kind = self.active_replay_kind
        if not self.busy or kind is None or self.replay_stop_event.is_set():
            return
        pause, _stop, status = self._replay_controls(kind)
        if self.replay_pause_event.is_set():
            self.replay_pause_event.clear()
            pause.configure(text="暂停")
            status.set("将在下一个操作边界继续。")
        else:
            self.replay_pause_event.set()
            pause.configure(text="继续")
            status.set("将在下一个操作边界暂停。")

    def rewind_collected_replay(self) -> None:
        control = self.replay_rewind_control
        if not self.busy or self.active_replay_kind != "collected" or control is None:
            return
        if control.request_rewind():
            message = "正在后退 5 秒…"
            self.collected_replay_status_var.set(message)
            self._append_log(message)
            return
        self.collected_replay_status_var.set(control.message)

    def set_replay_speed(self) -> None:
        self._set_replay_speed(self.replay_speed_var)

    def set_collected_replay_speed(self) -> None:
        self._set_replay_speed(self.collected_replay_speed_var)

    def _set_replay_speed(self, variable: Any) -> None:
        native = self.replay_native
        if native is not None:
            value = float(variable.get())
            self._background("replay-speed", lambda: native.set_speed(value))

    def stop_replay(self) -> None:
        self.replay_pause_event.clear()
        self.replay_stop_event.set()

    def _replay_finished(self, result: Any, *, kind: str) -> None:
        self._set_busy(False)
        self._set_replay_controls_active(kind, False)
        if kind == "collected":
            self.replay_rewind_control = None
        if result.stopped:
            message = f"回放已停止并暂停在 tick {result.final_native_tick}。"
        elif kind == "collected" and getattr(result, "completion", "") == "source-timeline":
            source_end_tick = (
                result.source_end_tick
                if getattr(result, "source_end_tick", None) is not None
                else result.final_native_tick
            )
            stop_tick = f"源回放结束 tick {source_end_tick}"
            if result.final_native_tick != source_end_tick:
                stop_tick += f"（native 实际暂停 tick {result.final_native_tick}）"
            message = (
                "采集动作重建播放完成："
                f"{result.queued_action_count}/{result.action_count} 个动作，"
                f"英雄技能 {result.queued_ability_count}/"
                f"{result.ability_count}；已播放到{stop_tick}，最大 tick 漂移 "
                f"{result.max_absolute_tick_drift}。"
            )
        else:
            fidelity = "结果一致" if result.expected_winner == result.actual_winner else "结果不一致"
            message = f"回放完成：{fidelity}，最大 tick 漂移 {result.max_absolute_tick_drift}。"
            if kind == "collected":
                message = (
                    f"{message[:-1]}；动作 "
                    f"{result.queued_action_count}/{result.action_count}，"
                    f"英雄技能 {result.queued_ability_count}/"
                    f"{result.ability_count}。"
                )
        _pause, _stop, status = self._replay_controls(kind)
        status.set(message)
        self.active_replay_kind = None
        self._append_log(message)

    def _operation_failed(self, error: BaseException) -> None:
        if self.vm3_resetting:
            LOGGER.info("ignoring stale VM3 operation failure during reset: %s", error)
            return
        self._set_busy(False)
        if self.active_replay_kind == "collected":
            self.replay_rewind_control = None
        if self.active_replay_kind is not None:
            _pause, _stop, status = self._replay_controls(self.active_replay_kind)
            status.set(f"失败：{error}")
        self._set_replay_controls_active("training", False)
        self._set_replay_controls_active("collected", False)
        self.active_replay_kind = None
        self._show_error(error)

    def _show_error(self, error: BaseException) -> None:
        from tkinter import messagebox

        LOGGER.error("%s", error)
        self._append_log(f"失败：{error}")
        messagebox.showerror("CR Harness", str(error), parent=self.root)

    def _terminate_overlay(self) -> None:
        process = self.overlay_process
        self.overlay_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        if self.overlay_log is not None:
            self.overlay_log.close()
            self.overlay_log = None

    def close(self) -> None:
        from tkinter import messagebox

        if self.busy and not messagebox.askyesno("退出", "退出会停止当前对局或回放，是否继续？", parent=self.root):
            return
        if self.model_active:
            self.model_after_stop = self._close_now
            self.stop_model_match()
            self._background("stop-model", self._wait_model_stopped)
            return
        self._close_now()

    def _close_now(self) -> None:
        self.stop_replay()
        self._terminate_overlay()
        for native in (self.replay_native, self.native):
            if native is not None:
                try:
                    native.pause()
                except Exception:
                    pass
        self.root.destroy()


def _acquire_single_instance() -> int | None:
    if os.name != "nt":
        return 1
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_bool
    ctypes.set_last_error(0)
    handle = kernel32.CreateMutexW(None, False, "Local\\CRHarnessUserInterface")
    if not handle:
        raise ctypes.WinError()
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return int(handle)


def _release_single_instance(handle: int | None) -> None:
    if os.name == "nt" and handle:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_bool
        kernel32.CloseHandle(handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FirstLight CR interactive match and replay interface")
    parser.add_argument(
        "--diagnose", action="store_true", help="print read-only preflight JSON and do not open the GUI"
    )
    parser.add_argument(
        "--skip-preflight-dialog", action="store_true", help="open the GUI without the startup warning dialog"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    log_path = _configure_logging()
    try:
        report = collect_preflight_report(check_vm=args.diagnose)
    except BaseException as error:
        LOGGER.exception("preflight failed")
        if args.diagnose:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
            return 2
        report = PreflightReport(
            vm=VmStatus(False, False, False, None, None, str(error)),
            memory=MemoryStatus(0.0, 0.0, 0.0),
            gpu=GpuStatus(False, None, (), None, None, None, str(error)),
            training_processes=(),
            warnings=(f"启动检查失败：{error}",),
        )
    if args.diagnose:
        print(json.dumps({"ok": True, **report.to_dict()}, ensure_ascii=False, indent=2))
        return 0

    import tkinter as tk
    from tkinter import messagebox

    if os.name == "nt":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FirstLight.CR.Console")

    instance = _acquire_single_instance()
    if instance is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("FirstLight CR", "FirstLight CR 控制台已经打开。")
        root.destroy()
        return 0

    root = tk.Tk()
    root.withdraw()
    try:
        if not args.skip_preflight_dialog:
            if report.warnings:
                proceed = messagebox.askokcancel(
                    "系统资源提醒",
                    "\n\n".join(report.warnings) + "\n\n是否仍然进入界面？\n"
                    "继续不会停止训练；离线 VM 只在具体操作时检测。",
                    parent=root,
                )
                if not proceed:
                    root.destroy()
                    return 0
        CRHarnessInterface(root, preflight=report, log_path=log_path)
        root.deiconify()
        root.mainloop()
    finally:
        _release_single_instance(instance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
