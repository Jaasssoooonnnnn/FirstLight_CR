param([string]$Apk = "", [string]$RestoreBackup = "")

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "native_runner/local_config.ps1")
if (-not $RestoreBackup) {
    if (-not $Apk) { $Apk = Get-LocalSetting "CR_OUTPUT_APK" (Join-Path $PSScriptRoot "build/cr-ai-offline.apk") }
    if (-not (Test-Path -LiteralPath $Apk -PathType Leaf)) { throw "Build the local offline APK first: $Apk" }
    $Apk = (Resolve-Path -LiteralPath $Apk).Path
}
$adb = Get-LocalSetting "CR_ADB"
$serial = Get-LocalSetting "CR_ADB_SERIAL"
$python = Get-LocalSetting "CR_PYTHON" "python"
$startup = Join-Path $PSScriptRoot "native_runner/start_offline.ps1"

# Identity and both IP-family firewalls are established before installing any app.
& $startup -PrepareOnly | Out-Null
Push-Location $PSScriptRoot
try {
    $arguments = @("-m", "native_runner.offline_install", "--adb", $adb, "--serial", $serial)
    if ($RestoreBackup) {
        & $python @arguments --restore $RestoreBackup
        if ($LASTEXITCODE -ne 0) { throw "Could not restore the device-local backup: $RestoreBackup" }
        return
    }
    $result = & $python @arguments --apk $Apk
    if ($LASTEXITCODE -ne 0) { throw "Offline installation failed; see the device backup path and install receipt for recovery." }
    $installed = ($result -join "`n") | ConvertFrom-Json
    try {
        & $startup
    } catch {
        $startupError = $_
        & $python @arguments --restore $installed.backup --receipt $installed.receipt
        if ($LASTEXITCODE -ne 0) { throw "Startup and automatic restore failed. Retained backup: $($installed.backup). Original error: $startupError" }
        throw "Offline startup failed; previous APK and app data restored. $startupError"
    }
    Write-Host "Reused $($installed.resource_files_verified) existing resource files. Device backup: $($installed.backup)"
} finally {
    Pop-Location
}
