param(
    [string]$VmName = "",
    [int]$VmIndex = 0,
    [string]$Serial = "",
    [string]$Package = "nullsroyale.rel.free",
    [string]$Activity = "com.supercell.clashroyale.GameApp",
    [int]$ControlPort = 0,
    [int]$GuestControlPort = 0,
    [string]$MuMuManager = "",
    [string]$Adb = "",
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("VmIndex")) { $VmIndex = Get-LocalSetting "CR_VM_INDEX" }
if (-not $PSBoundParameters.ContainsKey("VmName")) { $VmName = Get-LocalSetting "CR_VM_NAME" }
if (-not $PSBoundParameters.ContainsKey("Serial")) { $Serial = Get-LocalSetting "CR_ADB_SERIAL" }
if (-not $PSBoundParameters.ContainsKey("ControlPort")) { $ControlPort = Get-LocalSetting "CR_CONTROL_PORT" "26789" }
if (-not $PSBoundParameters.ContainsKey("GuestControlPort")) { $GuestControlPort = Get-LocalSetting "CR_GUEST_CONTROL_PORT" "26789" }
if (-not $PSBoundParameters.ContainsKey("MuMuManager")) { $MuMuManager = Get-LocalSetting "CR_MUMU_MANAGER" }
if (-not $PSBoundParameters.ContainsKey("Adb")) { $Adb = Get-LocalSetting "CR_ADB" }

# Validate the configured dedicated instance before touching Android.
if ($VmIndex -lt 0 -or -not $VmName -or $Serial -notmatch '^127\.0\.0\.1:[1-9][0-9]*$') {
    throw "Configure a dedicated offline VM index, name and loopback ADB serial in .env."
}
if ($ControlPort -lt 1 -or $ControlPort -gt 65535 -or $GuestControlPort -lt 1 -or $GuestControlPort -gt 65535) {
    throw "Control ports must be in 1..65535."
}
if (-not (Test-Path -LiteralPath $MuMuManager)) {
    throw "Missing MuMu manager: $MuMuManager"
}
if (-not (Test-Path -LiteralPath $Adb)) {
    throw "Missing adb executable: $Adb"
}

function Invoke-MuMu {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $output = & $MuMuManager @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager failed: $($Arguments -join ' ')`n$output"
    }
    return $output
}

function Invoke-Adb {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowFailure
    )
    # Capture expected probe failures (for example a not-yet-created iptables
    # chain) so idempotent startup does not emit misleading red error lines.
    # With the script-wide ErrorActionPreference=Stop, Windows PowerShell can
    # promote a native program's redirected stderr record to a terminating
    # NativeCommandError before we can inspect LASTEXITCODE.  Downgrade only
    # around this captured native call; the explicit exit-code check below
    # remains authoritative for real failures.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Adb @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "adb failed: $($Arguments -join ' ')`n$output"
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($output -join "`n")
    }
}

function Get-OfflineVmInfo {
    $raw = Invoke-MuMu -Arguments @("info", "--vmindex", "$VmIndex")
    return ($raw -join "`n") | ConvertFrom-Json
}

function Get-PackagePids {
    $result = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "pidof", $Package
    ) -AllowFailure
    $raw = $result.Output.Trim()
    if ($result.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
        return @()
    }
    $pids = @($raw -split "\s+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($pidValue in $pids) {
        if ($pidValue -notmatch "^[1-9][0-9]*$") {
            throw "pidof returned an invalid PID for ${Package}: $raw"
        }
    }
    return @($pids | Sort-Object -Unique)
}

function Wait-PackageStopped {
    param([string[]]$PreviousPids)
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    do {
        $currentPids = @(Get-PackagePids)
        if ($currentPids.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    $previous = if ($PreviousPids.Count -eq 0) {
        "none"
    } else {
        $PreviousPids -join ","
    }
    $remaining = @(Get-PackagePids) -join ","
    throw "Package $Package did not fully stop (old PIDs=$previous, remaining=$remaining)."
}

function Wait-NewPackagePid {
    param([string[]]$PreviousPids)
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        $currentPids = @(Get-PackagePids)
        if ($currentPids.Count -eq 1) {
            $candidate = [string]$currentPids[0]
            if ($PreviousPids -contains $candidate) {
                throw "Package $Package reused an old PID ($candidate); refusing an ambiguous restart."
            }
            return $candidate
        }
        if ($currentPids.Count -gt 1) {
            throw "Package $Package started with ambiguous PIDs: $($currentPids -join ',')."
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Package $Package did not publish a new PID within 30 seconds."
}

function Assert-ExpectedPackagePid {
    param([Parameter(Mandatory)][string]$ExpectedPid)
    $currentPids = @(Get-PackagePids)
    if ($currentPids.Count -ne 1 -or [string]$currentPids[0] -ne $ExpectedPid) {
        $actual = if ($currentPids.Count -eq 0) {
            "none"
        } else {
            $currentPids -join ","
        }
        throw "Package PID changed while waiting for control readiness (expected=$ExpectedPid, actual=$actual)."
    }
}

function Test-NullNativePointer {
    param($Value)
    if ($null -eq $Value) {
        return $false
    }
    $text = ([string]$Value).Trim().ToLowerInvariant()
    return $text -eq "0" -or $text -eq "0x0" -or $text -eq "(nil)"
}

function Wait-ControlReady {
    param([Parameter(Mandatory)][string]$ExpectedPid)
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    do {
        # Check immediately before every status request.  This prevents a
        # forwarded socket owned by the previous app process from satisfying
        # readiness during Android's asynchronous force-stop teardown.
        Assert-ExpectedPackagePid -ExpectedPid $ExpectedPid
        $client = $null
        $status = $null
        try {
            $client = [Net.Sockets.TcpClient]::new()
            $connect = $client.ConnectAsync("127.0.0.1", $ControlPort)
            if (-not $connect.Wait(1000)) {
                throw "control connection timed out"
            }
            $client.ReceiveTimeout = 2000
            $stream = $client.GetStream()
            $writer = [IO.StreamWriter]::new($stream)
            $writer.NewLine = "`n"
            $writer.AutoFlush = $true
            $reader = [IO.StreamReader]::new($stream)
            $writer.WriteLine("status")
            $status = $reader.ReadLine() | ConvertFrom-Json
            $client.Dispose()
        } catch {
            if ($null -ne $client) {
                $client.Dispose()
            }
            $status = $null
        }
        if ($null -ne $status) {
            # Close the pidof/status TOCTOU window before accepting the result.
            # This assertion is intentionally outside the network-error catch:
            # a replaced process is a hard lifecycle failure, not a retry.
            Assert-ExpectedPackagePid -ExpectedPid $ExpectedPid
            $freshProcessStatus =
                $null -ne $status.PSObject.Properties["generation"] -and
                $null -ne $status.PSObject.Properties["manager"] -and
                $null -ne $status.PSObject.Properties["configured"] -and
                $status.ok -and
                $status.coldReady -and
                -not [bool]$status.configured -and
                [uint64]$status.generation -eq 0 -and
                (Test-NullNativePointer -Value $status.manager)
            if ($freshProcessStatus) {
                return $status
            }
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Native control port did not become cold-ready within 45 seconds."
}

$info = Get-OfflineVmInfo
$expectedName = $VmName
if ($info.name -ne $expectedName -or [int]$info.index -ne $VmIndex) {
    throw "MuMu index $VmIndex is not the expected isolated worker $expectedName."
}

if (-not $info.is_process_started) {
    Invoke-MuMu -Arguments @("control", "--vmindex", "$VmIndex", "launch") | Out-Null
}

$bootDeadline = [DateTime]::UtcNow.AddSeconds(90)
do {
    $info = Get-OfflineVmInfo
    # Recent MuMu builds can leave is_android_started=false even after the VM
    # reports start_finished and its ADB transport is live.  Use the manager's
    # process/state pair here; wait-for-device and the root identity check below
    # remain the authoritative Android readiness gates.
    if ($info.is_process_started -and $info.player_state -eq "start_finished") {
        break
    }
    Start-Sleep -Milliseconds 500
} while ([DateTime]::UtcNow -lt $bootDeadline)
if (-not $info.is_process_started -or $info.player_state -ne "start_finished") {
    throw "Configured VM did not finish Android startup within 90 seconds."
}
if (
    $info.adb_host_ip -ne "127.0.0.1" -or
    $Serial -ne "127.0.0.1:$([int]$info.adb_port)"
) {
    throw "ADB serial $Serial does not match isolated worker $expectedName."
}

Invoke-Adb -Arguments @("connect", $Serial) | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "root") | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
$identity = Invoke-Adb -Arguments @("-s", $Serial, "shell", "id")
if ($identity.Output -notmatch "uid=0\(root\)") {
    throw "The isolated harness adbd is not running as root."
}

foreach ($family in @("iptables", "ip6tables")) {
    $exists = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", $family, "-S", "CR_NATIVE_OFFLINE"
    ) -AllowFailure
    if ($exists.ExitCode -eq 0) {
        Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-F", "CR_NATIVE_OFFLINE"
        ) | Out-Null
    } else {
        Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-N", "CR_NATIVE_OFFLINE"
        ) | Out-Null
    }

    $jump = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", $family, "-C", "OUTPUT", "-j", "CR_NATIVE_OFFLINE"
    ) -AllowFailure
    if ($jump.ExitCode -eq 0) {
        Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-D", "OUTPUT", "-j", "CR_NATIVE_OFFLINE"
        ) | Out-Null
    }
    Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", $family, "-I", "OUTPUT", "1", "-j", "CR_NATIVE_OFFLINE"
    ) | Out-Null
    Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
        "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "RETURN"
    ) | Out-Null
    Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE", "-o", "lo", "-j", "RETURN"
    ) | Out-Null
    if ($family -eq "iptables") {
        Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
            "-d", "10.0.2.2/32", "-j", "RETURN"
        ) | Out-Null
        Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
            "-j", "REJECT", "--reject-with", "icmp-port-unreachable"
        ) | Out-Null
    } else {
        Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", $family, "-A", "CR_NATIVE_OFFLINE",
            "-j", "REJECT", "--reject-with", "icmp6-port-unreachable"
        ) | Out-Null
    }
}

# Native rendering stops when Android suspends the display. Keep only the
# isolated VM awake while it is connected to host power.
Invoke-Adb -Arguments @("-s", $Serial, "shell", "svc", "power", "stayon", "true") | Out-Null
Invoke-Adb -Arguments @(
    "-s", $Serial, "forward", "tcp:$ControlPort", "tcp:$GuestControlPort"
) | Out-Null
$oldPids = @(Get-PackagePids)
if ($PrepareOnly) {
    [pscustomobject]@{
        ok = $true
        vmIndex = $VmIndex
        serial = $Serial
        offlineIPv4 = $true
        offlineIPv6 = $true
        preparedOnly = $true
    } | ConvertTo-Json
    return
}
Invoke-Adb -Arguments @("-s", $Serial, "shell", "am", "force-stop", $Package) | Out-Null
Wait-PackageStopped -PreviousPids $oldPids
Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "am", "start", "-n", "$Package/$Activity"
) | Out-Null
$newPid = Wait-NewPackagePid -PreviousPids $oldPids

$status = Wait-ControlReady -ExpectedPid $newPid
[pscustomobject]@{
    ok = $true
    instance = $info.name
    vmIndex = $VmIndex
    serial = $Serial
    offlineIPv4 = $true
    offlineIPv6 = $true
    coldReady = [bool]$status.coldReady
    replayBootstrap = [bool]$status.replayBootstrap
    fixtureCommands = $status.fixtureCommands
    controlPort = $ControlPort
    processId = [int]$newPid
} | ConvertTo-Json
