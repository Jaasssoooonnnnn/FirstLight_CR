#!/usr/bin/env python3
"""Configure isolated Android Emulator guests and attest engine endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


# Also support direct execution by the node launcher.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from native_runner.local_config import setting

ADB = Path(setting("CR_TRAINING_ADB", "adb"))
PACKAGE = "nullsroyale.rel.free"
GUEST_BASE_PORT = 26789


def adb(serial: str, *args: str, check: bool = True, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    attempts = 5 if check else 1
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        result = subprocess.run(
            [str(ADB), "-s", serial, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0 or not check:
            return result
        if attempt + 1 < attempts:
            time.sleep(0.75)
    assert result is not None
    raise RuntimeError(f"adb {serial} {' '.join(args)} failed ({result.returncode}): {result.stdout.strip()}")


def wait_root(serial: str) -> str:
    adb(serial, "wait-for-device", timeout=120.0)
    adb(serial, "root", check=False, timeout=30.0)
    deadline = time.monotonic() + 60.0
    last = ""
    while time.monotonic() < deadline:
        probe = adb(serial, "shell", "id", check=False, timeout=5.0)
        last = probe.stdout.strip()
        if probe.returncode == 0 and "uid=0(root)" in last:
            return last
        time.sleep(0.5)
    raise RuntimeError(f"{serial} adbd did not become root: {last}")


def configure_firewall(serial: str) -> dict[str, list[str]]:
    installed: dict[str, list[str]] = {}
    for family in ("iptables", "ip6tables"):
        exists = adb(serial, "shell", family, "-S", "CR_NATIVE_OFFLINE", check=False)
        if exists.returncode == 0:
            adb(serial, "shell", family, "-F", "CR_NATIVE_OFFLINE")
        else:
            adb(serial, "shell", family, "-N", "CR_NATIVE_OFFLINE")

        while True:
            jump = adb(serial, "shell", family, "-C", "OUTPUT", "-j", "CR_NATIVE_OFFLINE", check=False)
            if jump.returncode != 0:
                break
            adb(serial, "shell", family, "-D", "OUTPUT", "-j", "CR_NATIVE_OFFLINE")
        adb(serial, "shell", family, "-I", "OUTPUT", "1", "-j", "CR_NATIVE_OFFLINE")
        adb(
            serial,
            "shell",
            family,
            "-A",
            "CR_NATIVE_OFFLINE",
            "-m",
            "conntrack",
            "--ctstate",
            "RELATED,ESTABLISHED",
            "-j",
            "RETURN",
        )
        adb(serial, "shell", family, "-A", "CR_NATIVE_OFFLINE", "-o", "lo", "-j", "RETURN")
        if family == "iptables":
            adb(serial, "shell", family, "-A", "CR_NATIVE_OFFLINE", "-d", "10.0.2.2/32", "-j", "RETURN")
            adb(
                serial,
                "shell",
                family,
                "-A",
                "CR_NATIVE_OFFLINE",
                "-j",
                "REJECT",
                "--reject-with",
                "icmp-port-unreachable",
            )
        else:
            adb(
                serial,
                "shell",
                family,
                "-A",
                "CR_NATIVE_OFFLINE",
                "-j",
                "REJECT",
                "--reject-with",
                "icmp6-port-unreachable",
            )
        rules = adb(serial, "shell", family, "-S", "CR_NATIVE_OFFLINE").stdout
        installed[family] = [line for line in rules.splitlines() if line]
    return installed


def control(port: int, command: str, timeout: float = 5.0) -> dict[str, Any]:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((command + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks).decode("utf-8").strip()
    if not raw:
        raise RuntimeError(f"port {port} returned no data for {command}")
    return json.loads(raw)


def wait_endpoint(host_port: int, engine_id: int, online_cpus: int) -> dict[str, Any]:
    deadline = time.monotonic() + 180.0
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            status = control(host_port, "engine-status", timeout=2.0)
            attested = control(host_port, "attest", timeout=8.0)
            attestation = attested.get("attestation", {})
            expected_process = f"{PACKAGE}:engine{engine_id}"
            checks = {
                "ok": status.get("ok") is True,
                "engine_id": status.get("engineId") == engine_id,
                "process": status.get("processName") == expected_process,
                "port": status.get("port") == GUEST_BASE_PORT + engine_id,
                "assigned_cpu": status.get("assignedCpu") == engine_id,
                "online_cpus": int(status.get("onlineCpus", 0)) >= online_cpus,
                "production_ready": attestation.get("production_ready") is True,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(f"failed checks {failed}; status={status}; attestation={attestation}")
            control(host_port, "render off", timeout=3.0)
            return {"engine_id": engine_id, "host_port": host_port, "status": status, "attestation": attestation}
        except Exception as exc:  # readiness polling must retain the final cause
            last_error = repr(exc)
            time.sleep(0.5)
    raise RuntimeError(f"engine {engine_id} on host port {host_port} did not become ready: {last_error}")


def pin_engine_process(serial: str, endpoint: dict[str, Any]) -> None:
    """Pin every existing engine thread to its matching guest vCPU."""

    engine_id = int(endpoint["engine_id"])
    pid = int(endpoint["status"]["pid"])
    mask = f"{1 << engine_id:x}"
    adb(serial, "shell", "taskset", "-ap", mask, str(pid))
    affinity = adb(serial, "shell", "taskset", "-p", str(pid)).stdout.strip()
    reported_mask = affinity.rsplit(":", 1)[-1].strip().lower()
    if reported_mask != mask:
        raise RuntimeError(f"{serial} engine {engine_id} affinity is {reported_mask}, expected {mask}")
    status = control(int(endpoint["host_port"]), "engine-status", timeout=3.0)
    if int(status.get("currentCpu", -1)) != engine_id:
        raise RuntimeError(
            f"{serial} engine {engine_id} still runs on CPU {status.get('currentCpu')} after affinity pin"
        )
    endpoint["status"] = status
    endpoint["affinity_mask"] = mask


def configure_guest_once(serial: str, host_base: int, engines: int) -> dict[str, Any]:
    root_identity = wait_root(serial)
    nproc_text = adb(serial, "shell", "nproc").stdout.strip()
    if int(nproc_text) != engines:
        raise RuntimeError(f"{serial} exposes {nproc_text} CPUs, expected {engines}")
    package_path = adb(serial, "shell", "pm", "path", PACKAGE).stdout.strip()
    if not package_path:
        raise RuntimeError(f"{serial} does not contain {PACKAGE}")
    version = adb(serial, "shell", "dumpsys", "package", PACKAGE).stdout
    version_name = next(
        (line.split("=", 1)[1].strip() for line in version.splitlines() if "versionName=" in line), "unknown"
    )
    apk_path = next((line.split(":", 1)[1] for line in package_path.splitlines() if line.startswith("package:")), "")
    apk_sha256 = adb(serial, "shell", "sha256sum", apk_path).stdout.split()[0]

    firewall = configure_firewall(serial)
    adb(serial, "shell", "svc", "power", "stayon", "true")
    # Android's cached-app freezer otherwise suspends all but the most recently
    # launched engine process.  Control attestation can pass before the freeze,
    # then resident replay requests silently time out later.
    adb(serial, "shell", "settings", "put", "global", "cached_apps_freezer", "disabled")
    activity_settings = adb(serial, "shell", "dumpsys", "activity", "settings").stdout
    if "use_freezer=false" not in activity_settings:
        raise RuntimeError(f"{serial} failed to disable the cached-app freezer")
    adb(serial, "shell", "am", "force-stop", "--user", "0", PACKAGE)
    time.sleep(0.5)
    for engine_id in range(engines):
        host_port = host_base + engine_id
        guest_port = GUEST_BASE_PORT + engine_id
        adb(serial, "forward", "--remove", f"tcp:{host_port}", check=False)
        adb(serial, "forward", f"tcp:{host_port}", f"tcp:{guest_port}")
    # libndk_translation can race while several ARM64 processes build their
    # first translation caches in one x86_64 guest.  Cold-start and attest one
    # process at a time; once resident, all engines run concurrently.
    endpoints: list[dict[str, Any]] = []
    for engine_id in range(engines):
        component = f"{PACKAGE}/com.supercell.clashroyale.Engine{engine_id}App"
        launched = adb(serial, "shell", "am", "start", "--user", "0", "-f", "0x18000000", "-n", component).stdout
        if "Error:" in launched or "does not exist" in launched:
            raise RuntimeError(f"{serial} failed to launch {component}: {launched}")
        endpoint = wait_endpoint(host_base + engine_id, engine_id, engines)
        pin_engine_process(serial, endpoint)
        endpoints.append(endpoint)
    pids = [int(endpoint["status"]["pid"]) for endpoint in endpoints]
    if len(set(pids)) != engines:
        raise RuntimeError(f"{serial} engine PIDs are not unique: {pids}")
    return {
        "serial": serial,
        "host_base": host_base,
        "engines": engines,
        "root_identity": root_identity,
        "nproc": int(nproc_text),
        "package_version": version_name,
        "apk_sha256": apk_sha256,
        "firewall": firewall,
        "endpoints": endpoints,
    }


def configure_guest(serial: str, host_base: int, engines: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            result = configure_guest_once(serial, host_base, engines)
            result["startup_attempt"] = attempt
            return result
        except Exception as exc:
            last_error = exc
            adb(serial, "shell", "am", "force-stop", "--user", "0", PACKAGE, check=False)
            time.sleep(2.0)
    raise RuntimeError(f"{serial} failed to start all {engines} engines after 3 attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serials", required=True)
    parser.add_argument("--host-bases", required=True)
    parser.add_argument("--engines", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serials = [value.strip() for value in args.serials.split(",") if value.strip()]
    host_bases = [int(value) for value in args.host_bases.split(",") if value.strip()]
    if len(serials) != len(host_bases):
        parser.error("--serials and --host-bases must contain the same number of entries")
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(serials)) as executor:
        futures = [
            executor.submit(configure_guest, serial, host_base, args.engines)
            for serial, host_base in zip(serials, host_bases, strict=True)
        ]
        guests = [future.result() for future in futures]
    report = {
        "ok": True,
        "elapsed_s": time.monotonic() - started,
        "guest_count": len(guests),
        "engine_count": len(guests) * args.engines,
        "guests": guests,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
