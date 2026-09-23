"""Installation must retain resource data across signing changes and failures."""

import json

import pytest

from native_runner import offline_install as install


class Device:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def shell(self, command):
        self.calls.append(command)
        if command.startswith("dumpsys package"):
            return "versionName=15.535.13\nUser 0: installed=true"
        if command.startswith("pm path"):
            return "package:/data/app/test/base.apk"
        if command.startswith("sha256sum"):
            return "a" * 64 + "  apk"
        if command.startswith("if [ -d"):
            return "present"
        if command.startswith("stat -c %u"):
            return "10099"
        if command.startswith("pm uninstall"):
            return "Success"
        return ""

    def run(self, *args):
        self.calls.append(args)
        return ""

    def install_apk(self, path):
        self.calls.append("install " + path)
        return next(self.results)


@pytest.fixture
def setup_install(monkeypatch, tmp_path):
    hashes = {"fingerprint.json": "b" * 64, "assets.scdb": "c" * 64}
    monkeypatch.setattr(install, "check_apk", lambda apk: None)
    monkeypatch.setattr(install, "sha256", lambda apk: "a" * 64)
    monkeypatch.setattr(install, "resource_hashes", lambda *args: hashes.copy())
    monkeypatch.setattr(install, "validate_resources", lambda value: None)
    return tmp_path / "local.apk", tmp_path / "install.json", hashes


def test_signature_change_restores_device_data_with_new_uid(setup_install):
    apk, receipt, hashes = setup_install
    device = Device(["Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE]", "Success"])
    result = install.install(device, apk, receipt)
    calls = device.calls
    uninstall = calls.index("pm uninstall " + install.APP)
    assert any(isinstance(call, tuple) and call[-1].endswith("/state.json") for call in calls[:uninstall])
    assert any(isinstance(call, str) and call.startswith("cp -a ") for call in calls[:uninstall])
    assert any(isinstance(call, str) and "chown -R 10099:10099" in call for call in calls[uninstall:])
    assert all(not isinstance(call, tuple) or call[0] != "pull" for call in calls)
    assert json.loads(receipt.read_text())["resource_hashes"] == hashes
    assert result["installed"]


def test_same_signer_upgrade_does_not_uninstall(setup_install):
    apk, receipt, _ = setup_install
    device = Device(["Success"])
    install.install(device, apk, receipt)
    assert not any(isinstance(c, str) and c.startswith("pm uninstall") for c in device.calls)
    assert not any(isinstance(c, str) and "chown -R" in c for c in device.calls)


def test_unrelated_install_failure_does_not_uninstall(setup_install):
    apk, receipt, _ = setup_install
    device = Device(["Failure [INSTALL_FAILED_INSUFFICIENT_STORAGE]"])
    with pytest.raises(RuntimeError, match="INSUFFICIENT_STORAGE"):
        install.install(device, apk, receipt)
    assert "pm uninstall " + install.APP not in device.calls


def test_failure_after_signature_replacement_restores_previous_install(setup_install, monkeypatch):
    apk, receipt, _ = setup_install
    device = Device(["Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE]", "Failure [INSTALL_FAILED_INTERNAL_ERROR]"])
    restored = []
    monkeypatch.setattr(install, "restore", lambda device, backup: restored.append(backup))
    with pytest.raises(RuntimeError, match="INTERNAL_ERROR"):
        install.install(device, apk, receipt)
    assert restored == [json.loads(receipt.read_text())["backup"]]


def test_changed_resources_trigger_rollback(setup_install, monkeypatch):
    apk, receipt, hashes = setup_install
    reads = iter([hashes, hashes, {"fingerprint.json": "changed"}])
    monkeypatch.setattr(install, "resource_hashes", lambda *args: next(reads))
    restored = []
    monkeypatch.setattr(install, "restore", lambda device, backup: restored.append(backup))
    with pytest.raises(ValueError, match="resources differ"):
        install.install(Device(["Success"]), apk, receipt)
    assert len(restored) == 1


def test_incomplete_original_update_stops_before_device_changes(setup_install, monkeypatch):
    apk, receipt, _ = setup_install
    def incomplete(hashes):
        raise ValueError("incomplete update")
    monkeypatch.setattr(install, "validate_resources", incomplete)
    device = Device([])
    with pytest.raises(ValueError, match="incomplete"):
        install.install(device, apk, receipt)
    assert not receipt.exists()
    assert not any(isinstance(c, str) and c.startswith(("am force-stop", "cp ", "mkdir", "pm uninstall", "install ")) for c in device.calls)


def test_restore_rejects_arbitrary_device_path():
    with pytest.raises(ValueError, match="Invalid Firstlight backup"):
        install.restore(Device([]), "/data/user/0")


def test_check_original_game_reads_apk_and_resources_without_replacing_app(monkeypatch, tmp_path):
    release = install.supported_engine()
    apk = tmp_path / "original.apk"
    monkeypatch.setattr(install, "sha256", lambda path: release["apk_sha256"])
    monkeypatch.setattr(install, "resource_hashes", lambda device: {"fingerprint.json": "verified"})
    monkeypatch.setattr(install, "validate_resources", lambda hashes: None)

    class OriginalDevice(Device):
        def shell(self, command):
            if command.startswith("sha256sum "):
                self.calls.append(command)
                return release["apk_sha256"] + "  base.apk"
            return super().shell(command)

    device = OriginalDevice([])
    result = install.check_original_game(device, apk)
    assert result["original_apk_verified"] is True
    assert result["resource_files_verified"] == 1
    assert not any(isinstance(c, str) and c.startswith(("am force-stop", "cp ", "mkdir", "pm uninstall", "install ")) for c in device.calls)


def test_check_original_game_rejects_wrong_apk_before_device_access(monkeypatch, tmp_path):
    monkeypatch.setattr(install, "sha256", lambda path: "0" * 64)
    device = Device([])
    with pytest.raises(ValueError, match="Input APK SHA-256"):
        install.check_original_game(device, tmp_path / "wrong.apk")
    assert device.calls == []
