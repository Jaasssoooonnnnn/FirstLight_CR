"""Atomically install one probe in idle emulator guests before engines start."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence


# Also support direct execution by the node launcher.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from native_runner.local_config import setting

ADB = Path(setting("CR_TRAINING_ADB", "adb"))
PACKAGE = "nullsroyale.rel.free"


def run(arguments: Sequence[str | os.PathLike[str]], *, timeout: float = 60.0) -> str:
    result = subprocess.run(
        [os.fspath(item) for item in arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(map(os.fspath, arguments))}\n{result.stdout.strip()}"
        )
    return result.stdout


def adb(serial: str, *arguments: str, timeout: float = 60.0) -> str:
    return run((ADB, "-s", serial, *arguments), timeout=timeout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha256(serial: str, path: str) -> str:
    values = adb(serial, "shell", "sha256sum", path).split()
    if len(values) < 2 or len(values[0]) != 64:
        raise RuntimeError(f"{serial} returned an invalid digest for {path}")
    return values[0].lower()


def wait_root(serial: str) -> None:
    for attempt in range(5):
        try:
            adb(serial, "wait-for-device", timeout=120.0)
            adb(serial, "root", timeout=30.0)
            break
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == 4:
                raise
            time.sleep(2.0)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            if "uid=0(root)" in adb(serial, "shell", "id", timeout=5.0):
                return
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{serial} adbd did not become root")


def installed_probe_path(serial: str) -> str:
    rows = adb(serial, "shell", "pm", "path", PACKAGE).splitlines()
    bases = [
        row.removeprefix("package:").strip()
        for row in rows
        if row.startswith("package:") and row.strip().endswith("/base.apk")
    ]
    if len(bases) != 1:
        raise RuntimeError(f"{serial} does not have exactly one base APK")
    base = bases[0]
    if not base.startswith("/data/app/") or f"/{PACKAGE}-" not in base:
        raise RuntimeError(f"{serial} returned an unsafe APK path: {base}")
    target = f"{base.rsplit('/', 1)[0]}/lib/arm64/libcrprobe.so"
    adb(serial, "shell", "test", "-f", target)
    return target


def install(serial: str, probe: Path, expected: str) -> dict[str, object]:
    wait_root(serial)
    for engine_id in range(6):
        process = f"{PACKAGE}:engine{engine_id}"
        result = subprocess.run(
            [os.fspath(ADB), "-s", serial, "shell", "pidof", process],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            raise RuntimeError(f"{serial} is not idle: {process} is running")

    target = installed_probe_path(serial)
    previous = remote_sha256(serial, target)
    metadata = adb(serial, "shell", "stat", "-c", "%u:%g:%a", target).strip()
    label_rows = adb(serial, "shell", "ls", "-Zd", target).split()
    if metadata.count(":") != 2 or not label_rows:
        raise RuntimeError(f"{serial} could not inspect probe metadata")
    label = label_rows[0]
    if previous != expected:
        stage = f"/data/local/tmp/crprobe-deploy-{expected}.so"
        candidate = f"{target}.codex-{expected[:16]}.tmp"
        backup = f"/data/local/tmp/libcrprobe-before-{previous}.so"
        adb(serial, "push", os.fspath(probe), stage, timeout=180.0)
        if remote_sha256(serial, stage) != expected:
            raise RuntimeError(f"{serial} staged probe digest differs")
        try:
            adb(serial, "shell", "test", "-f", backup)
        except RuntimeError:
            adb(serial, "shell", "cp", "-p", target, backup)
        if remote_sha256(serial, backup) != previous:
            raise RuntimeError(f"{serial} rollback probe digest differs")
        adb(serial, "shell", "rm", "-f", candidate)
        adb(serial, "shell", "cp", "-p", target, candidate)
        adb(serial, "shell", "dd", f"if={stage}", f"of={candidate}", "conv=fsync", timeout=180.0)
        user_group, mode = metadata.rsplit(":", 1)
        adb(serial, "shell", "chown", user_group, candidate)
        adb(serial, "shell", "chmod", mode, candidate)
        adb(serial, "shell", "chcon", label, candidate)
        if remote_sha256(serial, candidate) != expected:
            raise RuntimeError(f"{serial} candidate probe digest differs")
        adb(serial, "shell", "mv", "-f", candidate, target)
        adb(serial, "shell", "rm", "-f", stage)
    if remote_sha256(serial, target) != expected:
        raise RuntimeError(f"{serial} installed probe digest differs")
    return {
        "serial": serial,
        "target": target,
        "previous_sha256": previous,
        "installed_sha256": expected,
        "replaced": previous != expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--serials", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = args.probe.resolve()
    if not probe.is_file() or probe.stat().st_size < 65_536:
        raise FileNotFoundError(probe)
    if args.output.exists():
        raise FileExistsError(args.output)
    serials = tuple(item.strip() for item in args.serials.split(",") if item.strip())
    if not serials or len(set(serials)) != len(serials):
        raise ValueError("serials must be non-empty and unique")
    expected = sha256_file(probe)
    started = time.monotonic()
    guests = [install(serial, probe, expected) for serial in serials]
    report = {
        "ok": True,
        "probe_sha256": expected,
        "elapsed_s": round(time.monotonic() - started, 6),
        "guest_count": len(guests),
        "guests": guests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
