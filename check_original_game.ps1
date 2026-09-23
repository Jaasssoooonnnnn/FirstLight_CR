param([string]$InputApk = "")

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "native_runner/local_config.ps1")
if (-not $InputApk) { $InputApk = Get-LocalSetting "CR_INPUT_APK" (Join-Path $PSScriptRoot "user-provided/nulls-royale.apk") }
if (-not (Test-Path -LiteralPath $InputApk -PathType Leaf)) { throw "Missing original APK: $InputApk" }
$InputApk = (Resolve-Path -LiteralPath $InputApk).Path
$adb = Get-LocalSetting "CR_ADB"
$serial = Get-LocalSetting "CR_ADB_SERIAL"
$python = Get-LocalSetting "CR_PYTHON" "python"

# Confirm the configured MuMu identity and establish root ADB before reading game content.
& (Join-Path $PSScriptRoot "native_runner/start_offline.ps1") -PrepareOnly | Out-Null
$androidRelease = ((& $adb -s $serial shell getprop ro.build.version.release) -join "").Trim()
if ($LASTEXITCODE -ne 0 -or $androidRelease -notmatch '^12(?:\.|$)') {
    throw "Selected MuMu instance must run Android 12; reported: $androidRelease"
}
Push-Location $PSScriptRoot
try {
    & $python -m native_runner.offline_install --adb $adb --serial $serial --check-original $InputApk
    if ($LASTEXITCODE -ne 0) { throw "Original APK or downloaded game resources do not match the supported engine." }
} finally {
    Pop-Location
}
