param(
    [string]$Serial = "",
    [string]$Package = "nullsroyale.rel.free",
    [string]$Adb = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("Serial")) { $Serial = Get-LocalSetting "CR_ADB_SERIAL" }
if (-not $PSBoundParameters.ContainsKey("Adb")) { $Adb = Get-LocalSetting "CR_ADB" }

if (-not (Test-Path -LiteralPath $Adb)) {
    throw "Missing adb executable: $Adb"
}

$python = Get-LocalSetting "CR_PYTHON"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$renderGate = & $python -X utf8 -c "import json; from native_runner.cr_native_env import NativeClashEnv; print(json.dumps(NativeClashEnv().set_rendering(False)))"
if ($LASTEXITCODE -ne 0) {
    throw "The probe could not suppress native EGL frame submission"
}

$pidValue = (& $Adb -s $Serial shell pidof $Package).Trim()
if (-not $pidValue) {
    throw "The game process exited while disabling rendering"
}

Start-Sleep -Seconds 1
$layer = "SurfaceView[$Package/com.supercell.clashroyale.GameApp](BLAST)#0"
function Get-LatestFrameTimestamp([string[]]$Rows) {
    $timestamps = @(
        $Rows | Where-Object {
            $_ -match '^[1-9][0-9]*\s+[1-9][0-9]*\s+[1-9][0-9]*$'
        } | ForEach-Object {
            [int64]($_ -split '\s+')[0]
        }
    )
    if ($timestamps.Count -eq 0) {
        return [int64]0
    }
    return [int64](($timestamps | Measure-Object -Maximum).Maximum)
}

$latencyBefore = & $Adb -s $Serial shell "dumpsys SurfaceFlinger --latency '$layer'"
$frameTimestampBefore = Get-LatestFrameTimestamp $latencyBefore
Start-Sleep -Seconds 1
$latencyAfter = & $Adb -s $Serial shell "dumpsys SurfaceFlinger --latency '$layer'"
$frameTimestampAfter = Get-LatestFrameTimestamp $latencyAfter
if ($frameTimestampAfter -ne $frameTimestampBefore) {
    throw "The game surface timestamp advanced after native rendering was suppressed"
}

$status = & $python -X utf8 -c "import json; from native_runner.cr_native_env import NativeClashEnv; print(json.dumps(NativeClashEnv().status()))"
if ($LASTEXITCODE -ne 0) {
    throw "The native manager stopped responding after rendering was disabled"
}

[pscustomobject]@{
    Serial = $Serial
    GamePid = $pidValue
    NewGameSurfaceFrames = 0
    FrameTimestampBefore = $frameTimestampBefore
    FrameTimestampAfter = $frameTimestampAfter
    RenderGate = ($renderGate -join "")
    NativeStatus = ($status -join "")
}
