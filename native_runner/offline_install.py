"""Install a locally built probe APK while retaining existing data on the device."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid
import zipfile
import hashlib

from native_runner.offline_build import PACKAGE, sha256, supported_engine

APP = "nullsroyale.rel.free"
DATA = f"/data/user/0/{APP}"
BACKUP_PREFIX = "/data/local/tmp/firstlight-install-"
LOCATIONS = {
    "ce": DATA,
    "de": f"/data/user_de/0/{APP}",
    "external": f"/sdcard/Android/data/{APP}",
    "obb": f"/sdcard/Android/obb/{APP}",
}


class Device:
    def __init__(self, adb, serial):
        self.command = [adb, "-s", serial]

    def run(self, *args, allow_failure=False):
        result = subprocess.run(self.command + list(args), capture_output=True, timeout=300)
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace").strip()
        if result.returncode and not allow_failure:
            raise RuntimeError(output)
        return output

    def shell(self, script):
        return self.run("shell", script)

    def require_offline(self):
        if self.shell("id -u") != "0" or self.shell("am get-current-user") != "0":
            raise ValueError("Installation requires root ADB and Android user 0 on the dedicated offline VM")
        for family, reject in (("iptables", "icmp-port-unreachable"), ("ip6tables", "icmp6-port-unreachable")):
            self.shell(f"{family} -C OUTPUT -j CR_NATIVE_OFFLINE")
            self.shell(f"{family} -C CR_NATIVE_OFFLINE -j REJECT --reject-with {reject}")

    def install_apk(self, path):
        return self.run("shell", f"pm install --user 0 -r {shlex.quote(path)}", allow_failure=True)


def require_success(output):
    if output.strip() != "Success":
        raise RuntimeError(output or "Android package operation did not report success")


def resource_hashes(device, directory=DATA + "/update"):
    text = device.shell(f"cd {shlex.quote(directory)} && find . -type f -exec sha256sum {{}} +")
    hashes = {}
    for line in text.splitlines():
        digest, name = line.split(None, 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not name.startswith("./"):
            raise ValueError("Invalid device resource checksum output")
        hashes[name[2:]] = digest
    return hashes


def validate_resources(hashes):
    release = supported_engine()
    if hashes.get("fingerprint.json") != release["runtime_fingerprint_sha256"]:
        raise ValueError("First open the unmodified game yourself and finish the supported resource update")
    manifest = json.loads((PACKAGE / "data/competitive/manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["source_files"].items():
        if name.startswith("runtime-update/") and hashes.get(name.removeprefix("runtime-update/")) != digest:
            raise ValueError(f"Device update is incomplete or unsupported: {name}")


def check_apk(apk):
    receipt = json.loads(apk.with_suffix(".build.json").read_text(encoding="utf-8"))
    if receipt["apk_sha256"] != sha256(apk) or receipt.get("resources") != "existing-device-update":
        raise ValueError("Build the APK with build_offline_engine.ps1 before installing")
    with zipfile.ZipFile(apk) as archive:
        if hashlib.sha256(archive.read("lib/arm64-v8a/libg.so")).hexdigest() != supported_engine()["libg_sha256"]:
            raise ValueError("Unsupported APK engine")
        if hashlib.sha256(archive.read("lib/arm64-v8a/libcrprobe.so")).hexdigest() != receipt["probe_sha256"]:
            raise ValueError("APK probe differs from its build receipt")
        if any(name.startswith("assets/firstlight/") for name in archive.namelist()):
            raise ValueError("APK must not bundle downloaded resources")


def check_prepared_device(device):
    """Check the stock installation and its downloaded content before replacement."""
    info = device.shell(f"dumpsys package {APP}")
    if not re.search(r"\bversionName=" + re.escape(supported_engine()["apk_version"]) + r"(?:\s|$)", info):
        raise ValueError("Install and update the supported original APK yourself first")
    users = re.findall(r"User (\d+):[^\n]*installed=true", info)
    if users != ["0"]:
        raise ValueError("Use a dedicated VM with this game installed only for Android user 0")
    paths = device.shell(f"pm path {APP}").splitlines()
    if len(paths) != 1 or not paths[0].startswith("package:/data/app/"):
        raise ValueError("Expected the supported single-APK installation")
    hashes = resource_hashes(device)
    validate_resources(hashes)
    return paths[0].removeprefix("package:"), hashes


def check_original_game(device, apk):
    """Verify the original APK and updated VM before building the probe APK."""
    release = supported_engine()
    if sha256(apk) != release["apk_sha256"]:
        raise ValueError("Input APK SHA-256 differs from supported_engine.json")
    installed_path, hashes = check_prepared_device(device)
    installed_sha256 = device.shell(f"sha256sum {shlex.quote(installed_path)}").split()[0]
    if installed_sha256 != release["apk_sha256"]:
        raise ValueError("Installed game APK differs from the supported original APK")
    return {
        "original_apk_verified": True,
        "apk_version": release["apk_version"],
        "runtime_content_version": release["runtime_content_version"],
        "resource_files_verified": len(hashes),
    }


def restore_data(device, backup, locations):
    uid = device.shell(f"stat -c %u {DATA}")
    if not uid.isdecimal() or int(uid) < 10000:
        raise ValueError("Invalid installed app UID")
    for name in locations:
        target = LOCATIONS[name]
        device.shell(f"mkdir -p {target} && cp -a {backup}/{name}/. {target}/")
        if name in {"ce", "de"}:
            device.shell(f"chown -R {uid}:{uid} {target} && restorecon -RF {target}")


def restore(device, backup):
    """Restore the prior APK and data from a retained device-local transaction."""
    if not re.fullmatch(re.escape(BACKUP_PREFIX) + r"[0-9a-f]{32}", backup):
        raise ValueError("Invalid Firstlight backup path")
    state = json.loads(device.shell(f"cat {backup}/state.json"))
    if state["package"] != APP or any(name not in LOCATIONS for name in state["locations"]):
        raise ValueError("Invalid backup state")
    if device.shell(f"sha256sum {backup}/previous.apk").split()[0] != state["previous_apk_sha256"]:
        raise ValueError("Previous APK backup checksum mismatch")
    if resource_hashes(device, backup + "/ce/update") != state["resource_hashes"]:
        raise ValueError("Device resource backup checksum mismatch")
    device.shell(f"am force-stop {APP}")
    if device.shell(f"pm path {APP}").startswith("package:"):
        require_success(device.shell(f"pm uninstall {APP}"))
    require_success(device.install_apk(backup + "/previous.apk"))
    restore_data(device, backup, state["locations"])
    if resource_hashes(device) != state["resource_hashes"]:
        raise ValueError("Restored resources differ from the backup")
    return {"restored": True, "backup": backup}


def install(device, apk, receipt_path):
    check_apk(apk)
    old_path, before = check_prepared_device(device)
    old_apk = shlex.quote(old_path)
    device.shell(f"am force-stop {APP}")
    backup = BACKUP_PREFIX + uuid.uuid4().hex
    print(f"Preserving existing app data on the device: {backup}", file=sys.stderr, flush=True)
    device.shell(f"mkdir -m 700 {backup} && cp {old_apk} {backup}/previous.apk")
    locations = []
    for name, path in LOCATIONS.items():
        if device.shell(f"if [ -d {path} ]; then echo present; fi") == "present":
            device.shell(f"cp -a {path} {backup}/{name}")
            locations.append(name)
    if "ce" not in locations or resource_hashes(device, backup + "/ce/update") != before:
        raise ValueError("Device backup failed resource verification; original installation retained")
    state = {
        "package": APP,
        "backup": backup,
        "locations": locations,
        "resource_hashes": before,
        "previous_apk_sha256": device.shell(f"sha256sum {backup}/previous.apk").split()[0],
        "apk_sha256": sha256(apk),
        "installed": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    device.run("push", str(receipt_path), backup + "/state.json")
    candidate = backup + "/candidate.apk"
    device.run("push", str(apk), candidate)
    if device.shell(f"sha256sum {candidate}").split()[0] != state["apk_sha256"]:
        raise ValueError("Uploaded APK checksum mismatch; original installation retained")
    changed = False
    try:
        output = device.install_apk(candidate)
        if "INSTALL_FAILED_UPDATE_INCOMPATIBLE" in output:
            print("Replacing the differently signed APK; resources stay on this device", file=sys.stderr, flush=True)
            changed = True
            require_success(device.shell(f"pm uninstall {APP}"))
            require_success(device.install_apk(candidate))
            restore_data(device, backup, locations)
        else:
            require_success(output)
            changed = True
        if resource_hashes(device) != before:
            raise ValueError("Installed app resources differ from the preserved update")
    except (Exception, KeyboardInterrupt):
        if changed:
            restore(device, backup)
        raise
    state.update(installed=True, resource_files_verified=len(before))
    receipt_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return {"installed": True, "backup": backup, "resource_files_verified": len(before), "receipt": str(receipt_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apk", type=Path)
    group.add_argument("--check-original", type=Path)
    group.add_argument("--restore")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    device = Device(args.adb, args.serial)
    device.require_offline()
    if args.check_original:
        result = check_original_game(device, args.check_original)
    elif args.restore:
        result = restore(device, args.restore)
        if args.receipt:
            state = json.loads(args.receipt.read_text(encoding="utf-8"))
            state.update(installed=False, restored=True)
            args.receipt.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    else:
        result = install(device, args.apk, args.receipt or args.apk.with_suffix(".install.json"))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
