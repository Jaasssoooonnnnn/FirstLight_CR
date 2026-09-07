param(
    [string]$VmName = "",
    [int]$VmIndex = 0,
    [string]$Serial = "",
    [string]$Package = "nullsroyale.rel.free",
    [int]$EngineCount = 0,
    [int]$SlotsPerEngine = 0,
    [int]$BasePort = 0,
    [int]$VmMemoryGb = 0,
    [bool]$ConfigureVmResources = $false,
    [string]$ExpectedApkSha256 = "",
    [string]$MuMuManager = "",
    [string]$Adb = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("VmIndex")) { $VmIndex = Get-LocalSetting "CR_VM_INDEX" }
if (-not $PSBoundParameters.ContainsKey("VmName")) { $VmName = Get-LocalSetting "CR_VM_NAME" }
if (-not $PSBoundParameters.ContainsKey("Serial")) { $Serial = Get-LocalSetting "CR_ADB_SERIAL" }
if (-not $PSBoundParameters.ContainsKey("BasePort")) { $BasePort = Get-LocalSetting "CR_CONTROL_PORT" "26789" }
if (-not $PSBoundParameters.ContainsKey("MuMuManager")) { $MuMuManager = Get-LocalSetting "CR_MUMU_MANAGER" }
if (-not $PSBoundParameters.ContainsKey("Adb")) { $Adb = Get-LocalSetting "CR_ADB" }
if (-not $PSBoundParameters.ContainsKey("EngineCount")) { $EngineCount = Get-LocalSetting "CR_ENGINE_COUNT" "1" }
if (-not $PSBoundParameters.ContainsKey("SlotsPerEngine")) { $SlotsPerEngine = Get-LocalSetting "CR_SLOTS_PER_ENGINE" "8" }
if (-not $PSBoundParameters.ContainsKey("VmMemoryGb")) { $VmMemoryGb = Get-LocalSetting "CR_VM_MEMORY_GB" }
if (-not $PSBoundParameters.ContainsKey("ConfigureVmResources")) { $ConfigureVmResources = [bool]::Parse((Get-LocalSetting "CR_CONFIGURE_VM_RESOURCES" "true")) }
if (-not $PSBoundParameters.ContainsKey("ExpectedApkSha256") -and $env:CR_EXPECTED_APK_SHA256) { $ExpectedApkSha256 = $env:CR_EXPECTED_APK_SHA256 }

if (
    $VmIndex -lt 0 -or -not $VmName -or
    $Serial -notmatch '^127\.0\.0\.1:[1-9][0-9]*$' -or
    $Package -ne "nullsroyale.rel.free"
) {
    throw "Configure a dedicated offline VM in .env."
}
if (
    $EngineCount -lt 1 -or
    $EngineCount -gt 24 -or
    $SlotsPerEngine -lt 1 -or
    $SlotsPerEngine -gt 8 -or
    $BasePort -lt 1 -or ($BasePort + $EngineCount - 1) -gt 65535 -or
    $VmMemoryGb -lt 1
) {
    throw "The cluster supports 1..24 engines x 1..8 slots, positive memory and valid contiguous ports."
}
if (
    $ExpectedApkSha256 -and
    $ExpectedApkSha256 -notmatch "^[0-9a-fA-F]{64}$"
) {
    throw "ExpectedApkSha256 must be one SHA-256 hex digest."
}
foreach ($required in @(
    $MuMuManager,
    $Adb,
    (Join-Path $PSScriptRoot "start_offline.ps1")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required file: $required"
    }
}

function Invoke-Adb {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowFailure
    )
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Adb @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "adb failed: $($Arguments -join ' ')`n$($output -join "`n")"
    }
    [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($output -join "`n").Trim()
    }
}

function Get-MuMuInfo {
    $output = & $MuMuManager info --vmindex "$VmIndex"
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager could not inspect VM index $VmIndex."
    }
    ($output -join "`n") | ConvertFrom-Json
}

function Wait-MuMuState {
    param([Parameter(Mandatory)][bool]$Running)
    $deadline = [DateTime]::UtcNow.AddSeconds(120)
    do {
        $info = Get-MuMuInfo
        if ([bool]$info.is_process_started -eq $Running) {
            if (-not $Running -or $info.player_state -eq "start_finished") {
                return $info
            }
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "MuMu did not reach running=$Running within 120 seconds."
}

function Connect-RootAdb {
    Invoke-Adb -Arguments @("connect", $Serial) | Out-Null
    Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
    Invoke-Adb -Arguments @("-s", $Serial, "root") | Out-Null
    Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
    $identity = Invoke-Adb -Arguments @("-s", $Serial, "shell", "id")
    if ($identity.Output -notmatch "uid=0\(root\)") {
        throw "The isolated harness adbd is not running as root."
    }
}

function Get-OnlineCpuCount {
    $result = Invoke-Adb -Arguments @("-s", $Serial, "shell", "nproc")
    [int]$result.Output
}

function Get-InstalledApkIdentity {
    $packagePaths = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "pm", "path", "--user", "0", $Package
    )
    $baseApks = @(
        foreach ($line in $packagePaths.Output.Split("`n")) {
            $trimmed = $line.Trim()
            if ($trimmed -match "^package:(.+/base\.apk)$") {
                $Matches[1]
            }
        }
    )
    if ($baseApks.Count -ne 1) {
        throw "Could not identify one installed base.apk for $Package."
    }
    $digest = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "sha256sum", $baseApks[0]
    )
    if ($digest.Output -notmatch "(?i)^(?<sha256>[0-9a-f]{64})\s+") {
        throw "Could not hash the installed base.apk for $Package."
    }
    $actual = $Matches.sha256.ToLowerInvariant()
    if (
        $ExpectedApkSha256 -and
        $actual -ne $ExpectedApkSha256.ToLowerInvariant()
    ) {
        throw (
            "Installed APK SHA-256 $actual does not match the required " +
            "$($ExpectedApkSha256.ToLowerInvariant())."
        )
    }
    [pscustomobject]@{
        package = $Package
        path = $baseApks[0]
        sha256 = $actual
    }
}

function Get-MuMuSetting {
    param([Parameter(Mandatory)][string]$Key)
    $output = & $MuMuManager setting --vmindex "$VmIndex" --key $Key
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager could not read $Key."
    }
    $settings = ($output -join "`n") | ConvertFrom-Json
    [string]$settings.$Key
}

function Ensure-MuMuResources {
    $online = Get-OnlineCpuCount
    $configuredCpu = [int](Get-MuMuSetting "performance_cpu.custom")
    $configuredMemory = [int][double](
        Get-MuMuSetting "performance_mem.custom"
    )
    if (
        $online -eq $EngineCount -and
        $configuredCpu -eq $EngineCount -and
        $configuredMemory -eq $VmMemoryGb
    ) {
        return
    }
    if (-not $ConfigureVmResources) {
        throw "MuMu exposes $online CPUs; $EngineCount independent lanes require at least $EngineCount."
    }

    & $MuMuManager control --vmindex "$VmIndex" shutdown | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager could not stop the isolated VM."
    }
    Wait-MuMuState -Running $false | Out-Null
    $resourceSettings = [ordered]@{
        "performance_mode" = "custom"
        "performance_cpu.custom" = "$EngineCount"
        "performance_mem.custom" = "$VmMemoryGb"
    }
    foreach ($key in $resourceSettings.Keys) {
        $value = $resourceSettings[$key]
        & $MuMuManager setting --vmindex "$VmIndex" `
            --key "$key" --value "$value" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "MuMuManager could not set $key."
        }
    }
    if (
        (Get-MuMuSetting "performance_mode") -ne "custom" -or
        [int](Get-MuMuSetting "performance_cpu.custom") -ne $EngineCount -or
        [int][double](Get-MuMuSetting "performance_mem.custom") -ne $VmMemoryGb
    ) {
        throw "MuMu did not persist the requested cluster VM resources."
    }
    & $MuMuManager control --vmindex "$VmIndex" launch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager could not restart the isolated VM."
    }
    Wait-MuMuState -Running $true | Out-Null
    Connect-RootAdb
    $online = Get-OnlineCpuCount
    if ($online -ne $EngineCount) {
        throw "MuMu exposes $online CPUs after requesting exactly $EngineCount."
    }
}

function Set-HostCpuPool {
    if (-not $env:CR_HOST_CPU_AFFINITY) { return $null }
    $vmProcesses = @(
        Get-CimInstance Win32_Process `
            -Filter "Name='MuMuVMMHeadless.exe'" |
            Where-Object {
                $_.CommandLine -match (
                    "--comment\s+MuMuPlayerGlobal-12\.0-$VmIndex(?:\s|$)"
                )
            }
    )
    if ($vmProcesses.Count -ne 1) {
        throw "Could not identify the unique MuMu VMM process for index $VmIndex."
    }
    $vmProcess = Get-Process -Id $vmProcesses[0].ProcessId
    $desired = [Convert]::ToInt64($env:CR_HOST_CPU_AFFINITY, 16)
    if ([long]$vmProcess.ProcessorAffinity -ne $desired) {
        $vmProcess.ProcessorAffinity = [IntPtr]$desired
        $vmProcess.Refresh()
    }
    $actual = [long]$vmProcess.ProcessorAffinity
    if ($actual -ne $desired) {
        throw "Could not set MuMu VMM affinity to 0x$($desired.ToString('X'))."
    }
    [pscustomobject]@{
        pid = [int]$vmProcesses[0].ProcessId
        mask = "0x$($actual.ToString('X'))"
        topology = "configured-affinity"
    }
}

function Invoke-Control {
    param(
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][string]$Command,
        [int]$TimeoutMs = 3000
    )
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $pending = $client.ConnectAsync("127.0.0.1", $Port)
        if (-not $pending.Wait($TimeoutMs)) {
            throw "control connection timed out"
        }
        $client.ReceiveTimeout = $TimeoutMs
        $client.SendTimeout = $TimeoutMs
        $stream = $client.GetStream()
        $writer = [IO.StreamWriter]::new(
            $stream,
            [Text.UTF8Encoding]::new($false),
            4096,
            $true
        )
        $writer.NewLine = "`n"
        $writer.WriteLine($Command)
        $writer.Flush()
        $reader = [IO.StreamReader]::new(
            $stream,
            [Text.UTF8Encoding]::new($false),
            $false,
            4096,
            $true
        )
        $payload = $reader.ReadToEnd().Trim()
        if (-not $payload) {
            throw "control endpoint returned an empty response"
        }
        $payload | ConvertFrom-Json
    } finally {
        $client.Dispose()
    }
}

function Wait-EngineClusterReady {
    $pendingIds = @(0..($EngineCount - 1))
    $readyById = @{}
    $lastErrors = @{}
    $deadline = [DateTime]::UtcNow.AddSeconds(150)
    do {
        foreach ($engineId in @($pendingIds)) {
            $port = $BasePort + $engineId
            $expectedProcess = "$Package`:engine$engineId"
            try {
                $status = Invoke-Control `
                    -Port $port `
                    -Command "engine-status" `
                    -TimeoutMs 750
                if (
                    -not $status.ok -or
                    [int]$status.engineId -ne $engineId -or
                    [int]$status.androidUserId -ne 0 -or
                    [string]$status.processName -ne $expectedProcess -or
                    [int]$status.port -ne (([int](Get-LocalSetting "CR_GUEST_CONTROL_PORT" "26789")) + $engineId) -or
                    [int]$status.assignedCpu -ne $engineId -or
                    [int]$status.currentCpu -ne $engineId -or
                    [int]$status.onlineCpus -lt $EngineCount
                ) {
                    throw "engine identity or affinity mismatch"
                }
                $attestation = Invoke-Control `
                    -Port $port `
                    -Command "attest" `
                    -TimeoutMs 3000
                if (-not [bool]$attestation.attestation.production_ready) {
                    throw "content runtime is not production-ready yet"
                }
                Invoke-Control `
                    -Port $port `
                    -Command "render off" `
                    -TimeoutMs 1500 | Out-Null
                $readyById[$engineId] = [pscustomobject]@{
                    engineId = $engineId
                    androidUserId = 0
                    processName = $expectedProcess
                    host = "127.0.0.1"
                    port = $port
                    pid = [int]$status.pid
                    guestCpu = [int]$status.assignedCpu
                    slots = $SlotsPerEngine
                    probeSha256 = [string](
                        $attestation.attestation.probe_sha256
                    )
                    contentFingerprintSha256 = [string](
                        $attestation.attestation.content_fingerprint_sha256
                    )
                }
                $pendingIds = @(
                    $pendingIds | Where-Object { $_ -ne $engineId }
                )
            } catch {
                $lastErrors[$engineId] = [string]$_
            }
        }
        if ($pendingIds.Count -ne 0) {
            Start-Sleep -Milliseconds 250
        }
    } while (
        $pendingIds.Count -ne 0 -and
        [DateTime]::UtcNow -lt $deadline
    )
    if ($pendingIds.Count -ne 0) {
        $details = @(
            foreach ($engineId in $pendingIds) {
                "engine$engineId=$($lastErrors[$engineId])"
            }
        )
        throw "Engine cluster was not ready: $($details -join '; ')"
    }
    @(
        for ($engineId = 0; $engineId -lt $EngineCount; ++$engineId) {
            $readyById[$engineId]
        }
    )
}

function Wait-PackageStopped {
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    do {
        $remaining = @()
        foreach ($name in @(
            $Package
        ) + @(
            for ($engineId = 0; $engineId -lt $EngineCount; ++$engineId) {
                "$Package`:engine$engineId"
            }
        )) {
            $result = Invoke-Adb -Arguments @(
                "-s", $Serial, "shell", "pidof", $name
            ) -AllowFailure
            if ($result.Output) {
                $remaining += "$name=$($result.Output)"
            }
        }
        if ($remaining.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Package processes did not stop: $($remaining -join ', ')"
}

$initialInfo = Get-MuMuInfo
if (
    $initialInfo.name -ne $VmName -or
    [int]$initialInfo.index -ne $VmIndex -or
    (
        [bool]$initialInfo.is_process_started -and
        (
            $initialInfo.adb_host_ip -ne "127.0.0.1" -or
            $Serial -ne "127.0.0.1:$([int]$initialInfo.adb_port)"
        )
    )
) {
    throw "Configured VM identity does not match the MuMu instance."
}
if (-not [bool]$initialInfo.is_process_started) {
    & $MuMuManager control --vmindex "$VmIndex" launch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager could not launch the isolated VM."
    }
    Wait-MuMuState -Running $true | Out-Null
}
$initialVmState = Get-MuMuInfo
if (-not [bool]$initialVmState.is_process_started) {
    & $MuMuManager control --vmindex "$VmIndex" launch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "MuMuManager could not recover the stopped isolated VM."
    }
    Wait-MuMuState -Running $true | Out-Null
}
Connect-RootAdb
Ensure-MuMuResources
$installedApk = Get-InstalledApkIdentity

# Reuse the established cold-start gate to install the VM-global offline
# firewall and verify that the shared app data/probe are healthy.  The default
# process is stopped immediately afterward; production work runs only in the
# manifest component processes.
& (Join-Path $PSScriptRoot "start_offline.ps1") `
    -VmIndex $VmIndex `
    -VmName $VmName `
    -Serial $Serial `
    -Package $Package `
    -Activity "com.supercell.clashroyale.GameApp" `
    -ControlPort $BasePort `
    -GuestControlPort ([int](Get-LocalSetting "CR_GUEST_CONTROL_PORT" "26789")) `
    -MuMuManager $MuMuManager `
    -Adb $Adb | Out-Null

$hostCpuPool = Set-HostCpuPool

Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "am", "force-stop", "--user", "0", $Package
) | Out-Null
Wait-PackageStopped

for ($engineId = 0; $engineId -lt $EngineCount; ++$engineId) {
    $port = $BasePort + $engineId
    Invoke-Adb -Arguments @(
        "-s", $Serial, "forward", "tcp:$port", "tcp:$(([int](Get-LocalSetting "CR_GUEST_CONTROL_PORT" "26789")) + $engineId)"
    ) | Out-Null
}

for ($engineId = 0; $engineId -lt $EngineCount; ++$engineId) {
    $component = "com.supercell.clashroyale.Engine$($engineId)App"
    $started = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "am", "start",
        "--user", "0",
        "-f", "0x18000000",
        "-n", "$Package/$component"
    )
    if (
        $started.Output -match "(?m)^Error:" -or
        $started.Output -match "does not exist"
    ) {
        throw "Could not launch engine $engineId`: $($started.Output)"
    }
}
$engines = @(Wait-EngineClusterReady)

$uniquePids = @($engines.pid | Sort-Object -Unique)
if ($uniquePids.Count -ne $EngineCount) {
    throw "Engine cluster did not produce $EngineCount unique app processes."
}
$probeHashes = @($engines.probeSha256 | Sort-Object -Unique)
$contentHashes = @($engines.contentFingerprintSha256 | Sort-Object -Unique)
if ($probeHashes.Count -ne 1 -or $contentHashes.Count -ne 1) {
    throw "Engine processes do not share one attested probe/content version."
}
foreach ($engine in $engines) {
    $status = Invoke-Control `
        -Port $engine.port `
        -Command "engine-status" `
        -TimeoutMs 1500
    if (
        -not $status.ok -or
        [int]$status.engineId -ne [int]$engine.engineId -or
        [int]$status.pid -ne [int]$engine.pid -or
        [string]$status.processName -ne [string]$engine.processName -or
        [int]$status.currentCpu -ne [int]$engine.guestCpu
    ) {
        throw "Engine $($engine.engineId) did not remain alive and isolated after all launches."
    }
    $render = Invoke-Control `
        -Port $engine.port `
        -Command "render status" `
        -TimeoutMs 1500
    if (-not $render.ok -or -not [bool]$render.renderSuppressed) {
        throw "Engine $($engine.engineId) did not retain headless rendering state."
    }
}

[pscustomobject]@{
    ok = $true
    version = "one-mumu-component-engine-cluster.v1"
    instance = $VmName
    vmIndex = $VmIndex
    serial = $Serial
    isolation = "android-zygote-component-process"
    cpuBinding = "native-control-lane-to-guest-vcpu"
    hostCpuPool = $hostCpuPool
    engineCount = $EngineCount
    slotsPerEngine = $SlotsPerEngine
    battleCapacity = $EngineCount * $SlotsPerEngine
    actorBatchCapacity = $EngineCount * $SlotsPerEngine * 2
    installedApk = $installedApk
    offlineIPv4 = $true
    offlineIPv6 = $true
    engines = $engines
} | ConvertTo-Json -Depth 5
