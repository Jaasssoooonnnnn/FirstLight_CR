param(
    [string]$VmName = "",
    [int]$VmIndex = 0,
    [string]$Serial = "",
    [string]$Package = "nullsroyale.rel.free",
    [int]$EngineCount = 0,
    [int]$BasePort = 0,
    [string]$Adb = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("VmIndex")) { $VmIndex = Get-LocalSetting "CR_VM_INDEX" }
if (-not $PSBoundParameters.ContainsKey("VmName")) { $VmName = Get-LocalSetting "CR_VM_NAME" }
if (-not $PSBoundParameters.ContainsKey("Serial")) { $Serial = Get-LocalSetting "CR_ADB_SERIAL" }
if (-not $PSBoundParameters.ContainsKey("BasePort")) { $BasePort = Get-LocalSetting "CR_CONTROL_PORT" "26789" }
if (-not $PSBoundParameters.ContainsKey("Adb")) { $Adb = Get-LocalSetting "CR_ADB" }
if (-not $PSBoundParameters.ContainsKey("EngineCount")) { $EngineCount = Get-LocalSetting "CR_ENGINE_COUNT" "1" }

if (
    $VmIndex -lt 0 -or -not $VmName -or
    $Serial -notmatch '^127\.0\.0\.1:[1-9][0-9]*$' -or
    $Package -ne "nullsroyale.rel.free" -or
    $EngineCount -lt 1 -or
    $EngineCount -gt 24 -or
    $BasePort -lt 1 -or ($BasePort + $EngineCount - 1) -gt 65535
) {
    throw "Configure a dedicated offline VM and a valid contiguous range of at most 24 ports."
}
if (-not (Test-Path -LiteralPath $Adb -PathType Leaf)) {
    throw "Missing adb: $Adb"
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

Invoke-Adb -Arguments @("connect", $Serial) | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "root") | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "wait-for-device") | Out-Null

$running = @()
foreach ($engineId in 0..($EngineCount - 1)) {
    $name = "$Package`:engine$engineId"
    $pidResult = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "pidof", $name
    ) -AllowFailure
    if ($pidResult.Output) {
        $running += [pscustomobject]@{
            engineId = $engineId
            processName = $name
            pid = [int]$pidResult.Output
        }
    }
}

Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "am", "force-stop", "--user", "0", $Package
) -AllowFailure | Out-Null
foreach ($engineId in 0..($EngineCount - 1)) {
    $port = $BasePort + $engineId
    Invoke-Adb -Arguments @(
        "-s", $Serial, "forward", "--remove", "tcp:$port"
    ) -AllowFailure | Out-Null
}

$deadline = [DateTime]::UtcNow.AddSeconds(15)
do {
    $remaining = @()
    foreach ($entry in $running) {
        $result = Invoke-Adb -Arguments @(
            "-s", $Serial, "shell", "pidof", $entry.processName
        ) -AllowFailure
        if ($result.Output) {
            $remaining += $entry.processName
        }
    }
    if ($remaining.Count -eq 0) {
        break
    }
    Start-Sleep -Milliseconds 100
} while ([DateTime]::UtcNow -lt $deadline)
if ($remaining.Count -ne 0) {
    throw "Cluster processes did not stop: $($remaining -join ', ')"
}

[pscustomobject]@{
    ok = $true
    version = "one-mumu-component-engine-cluster.v1"
    stopped = $running
    androidUsersChanged = $false
} | ConvertTo-Json -Depth 4
