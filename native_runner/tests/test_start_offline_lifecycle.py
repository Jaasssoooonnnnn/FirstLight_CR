from __future__ import annotations

from native_runner.paths import PACKAGE_ROOT

from pathlib import Path


START_OFFLINE = PACKAGE_ROOT / "start_offline.ps1"
DEPLOY_PROBE = PACKAGE_ROOT / "deploy_probe.ps1"


def _source(path: Path = START_OFFLINE) -> str:
    return path.read_text(encoding="utf-8")


def _function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name} {{")
    end = source.index(f"function {next_name} {{", start)
    return source[start:end]


def test_probe_deploy_checks_configured_worker_before_adb() -> None:
    source = _source(DEPLOY_PROBE)
    assert 'Get-LocalSetting "CR_VM_INDEX"' in source
    assert 'Get-LocalSetting "CR_VM_NAME"' in source
    assert 'Get-LocalSetting "CR_ADB_SERIAL"' in source
    assert "$VmIndex -lt 0" in source

    inspect = source.index('$infoOutput = & $MuMuManager info --vmindex "$VmIndex"')
    expected_name = source.index("$expectedName = $VmName", inspect)
    identity_gate = source.index("$info.name -ne $expectedName", expected_name)
    index_gate = source.index("[int]$info.index -ne $VmIndex", identity_gate)
    adb_host_gate = source.index('$info.adb_host_ip -ne "127.0.0.1"', index_gate)
    adb_port_gate = source.index(
        '$Serial -ne "127.0.0.1:$([int]$info.adb_port)"', adb_host_gate
    )
    first_adb = source.index('Invoke-Adb -Arguments @("connect", $Serial)')
    assert (
        inspect
        < expected_name
        < identity_gate
        < index_gate
        < adb_host_gate
        < adb_port_gate
        < first_adb
    )


def test_start_offline_validates_manager_identity_and_ports_before_adb() -> None:
    source = _source()
    inspect = source.index("$info = Get-OfflineVmInfo")
    expected_name = source.index("$expectedName = $VmName", inspect)
    identity_gate = source.index("$info.name -ne $expectedName", expected_name)
    index_gate = source.index("[int]$info.index -ne $VmIndex", identity_gate)
    adb_host_gate = source.index('$info.adb_host_ip -ne "127.0.0.1"', index_gate)
    adb_port_gate = source.index(
        '$Serial -ne "127.0.0.1:$([int]$info.adb_port)"', adb_host_gate
    )
    first_adb = source.index('Invoke-Adb -Arguments @("connect", $Serial)')
    forward = source.index(
        '"-s", $Serial, "forward", "tcp:$ControlPort", "tcp:$GuestControlPort"'
    )
    assert (
        inspect
        < expected_name
        < identity_gate
        < index_gate
        < adb_host_gate
        < adb_port_gate
        < first_adb
        < forward
    )

    parameter_gate = source[
        source.index("# Validate the configured dedicated instance") :
        source.index("if (-not (Test-Path -LiteralPath $MuMuManager))")
    ]
    assert "$ControlPort -lt 1 -or $ControlPort -gt 65535" in parameter_gate
    assert "$GuestControlPort -lt 1 -or $GuestControlPort -gt 65535" in parameter_gate


def test_restart_waits_for_full_old_pid_exit_before_start_and_binds_new_pid() -> None:
    source = _source()
    capture_old = source.index("$oldPids = @(Get-PackagePids)")
    force_stop = source.index(
        '"shell", "am", "force-stop", $Package', capture_old
    )
    wait_stopped = source.index(
        "Wait-PackageStopped -PreviousPids $oldPids", force_stop
    )
    start = source.index('"shell", "am", "start", "-n"', wait_stopped)
    capture_new = source.index(
        "$newPid = Wait-NewPackagePid -PreviousPids $oldPids", start
    )
    wait_control = source.index(
        "$status = Wait-ControlReady -ExpectedPid $newPid", capture_new
    )
    assert capture_old < force_stop < wait_stopped < start < capture_new < wait_control

    stopped = _function(source, "Wait-PackageStopped", "Wait-NewPackagePid")
    assert '$currentPids.Count -eq 0' in stopped
    assert "did not fully stop" in stopped

    started = _function(
        source, "Wait-NewPackagePid", "Assert-ExpectedPackagePid"
    )
    assert '$currentPids.Count -eq 1' in started
    assert "$PreviousPids -contains $candidate" in started
    assert "ambiguous PIDs" in started


def test_control_ready_is_pid_bound_and_requires_a_fresh_unconfigured_process() -> None:
    source = _source()
    ready = source[
        source.index("function Wait-ControlReady {") :
        source.index("$info = Get-OfflineVmInfo")
    ]
    assert "param([Parameter(Mandatory)][string]$ExpectedPid)" in ready
    assert ready.count(
        "Assert-ExpectedPackagePid -ExpectedPid $ExpectedPid"
    ) == 2
    assert "$status.ok" in ready
    assert "$status.coldReady" in ready
    assert "-not [bool]$status.configured" in ready
    assert "[uint64]$status.generation -eq 0" in ready
    assert "Test-NullNativePointer -Value $status.manager" in ready

    pointer = _function(
        source, "Test-NullNativePointer", "Wait-ControlReady"
    )
    for null_pointer in ('"0"', '"0x0"', '"(nil)"'):
        assert null_pointer in pointer
