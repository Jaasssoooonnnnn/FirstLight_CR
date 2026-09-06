param(
    [string]$VmName = "",
    [int]$VmIndex = 0,
    [string]$Serial = "",
    [string]$Package = "nullsroyale.rel.free",
    [string]$Activity = "com.supercell.clashroyale.GameApp",
    [string]$Probe = (Join-Path $PSScriptRoot "probe\out\libcrprobe.so"),
    [string]$MuMuManager = "",
    [string]$Adb = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("VmIndex")) { $VmIndex = Get-LocalSetting "CR_VM_INDEX" }
if (-not $PSBoundParameters.ContainsKey("VmName")) { $VmName = Get-LocalSetting "CR_VM_NAME" }
if (-not $PSBoundParameters.ContainsKey("Serial")) { $Serial = Get-LocalSetting "CR_ADB_SERIAL" }
if (-not $PSBoundParameters.ContainsKey("MuMuManager")) { $MuMuManager = Get-LocalSetting "CR_MUMU_MANAGER" }
if (-not $PSBoundParameters.ContainsKey("Adb")) { $Adb = Get-LocalSetting "CR_ADB" }
if (-not $PSBoundParameters.ContainsKey("Probe") -and $env:CR_PROBE) { $Probe = $env:CR_PROBE }

# Validate the configured dedicated instance before touching Android.
if ($VmIndex -lt 0 -or -not $VmName -or $Serial -notmatch '^127\.0\.0\.1:[1-9][0-9]*$') {
    throw "Configure a dedicated offline VM index, name and loopback ADB serial in .env."
}
foreach ($required in @($MuMuManager, $Adb, $Probe)) {
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
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($output -join "`n").Trim()
    }
}

$infoOutput = & $MuMuManager info --vmindex "$VmIndex"
if ($LASTEXITCODE -ne 0) {
    throw "MuMuManager could not inspect VM index $VmIndex."
}
$info = ($infoOutput -join "`n") | ConvertFrom-Json
$expectedName = $VmName
if (
    $info.name -ne $expectedName -or
    [int]$info.index -ne $VmIndex -or
    -not [bool]$info.is_process_started
) {
    throw "$expectedName must already be running before probe deployment."
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
    $offlineJump = Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", $family, "-C", "OUTPUT", "-j", "CR_NATIVE_OFFLINE"
    ) -AllowFailure
    if ($offlineJump.ExitCode -ne 0) {
        throw "Refusing to restart the client without the $family CR_NATIVE_OFFLINE chain."
    }
}

$packagePath = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "pm", "path", $Package
)
$baseApkLine = @($packagePath.Output -split "`r?`n") |
    Where-Object { $_ -like "package:*/base.apk" } |
    Select-Object -First 1
if (-not $baseApkLine) {
    throw "Package $Package has no installed base.apk path."
}
$baseApk = $baseApkLine.Substring("package:".Length)
$separator = $baseApk.LastIndexOf("/")
if ($separator -le 0) {
    throw "Installed base.apk path is malformed: $baseApk"
}
$packageRoot = $baseApk.Substring(0, $separator)
$target = "$packageRoot/lib/arm64/libcrprobe.so"

$targetProbe = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "test", "-f", $target
) -AllowFailure
if ($targetProbe.ExitCode -ne 0) {
    throw "Installed package has no arm64 libcrprobe.so at the exact target path."
}

$localSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $Probe).Hash.ToLowerInvariant()
$localLength = (Get-Item -LiteralPath $Probe).Length
if ($localLength -lt 65536) {
    throw "Probe binary is unexpectedly small: $localLength bytes."
}
$remoteStage = "/data/local/tmp/crprobe-deploy-$localSha.so"
Invoke-Adb -Arguments @("-s", $Serial, "push", $Probe, $remoteStage) | Out-Null

$stagedSha = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "sha256sum", $remoteStage
)
if (($stagedSha.Output -split "\s+")[0].ToLowerInvariant() -ne $localSha) {
    throw "Staged probe SHA-256 does not match the host binary."
}

$oldShaResult = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "sha256sum", $target
)
$oldSha = ($oldShaResult.Output -split "\s+")[0].ToLowerInvariant()
$backup = "/data/local/tmp/libcrprobe-before-$oldSha.so"
$backupExists = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "test", "-f", $backup
) -AllowFailure
if ($backupExists.ExitCode -ne 0) {
    Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "cp", $target, $backup
    ) | Out-Null
}

# libcrprobe.so is shared by every process of an installed package. Stop the
# package in every Android user before overwriting its inode so no component
# process can retain an old probe mapping.
$userList = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "cmd", "user", "list", "-v"
)
foreach ($line in ($userList.Output -split "`r?`n")) {
    if ($line -notmatch "id=(\d+),") {
        continue
    }
    Invoke-Adb -Arguments @(
        "-s", $Serial, "shell", "am", "force-stop",
        "--user", "$($Matches[1])", $Package
    ) -AllowFailure | Out-Null
}
# dd overwrites the existing inode, preserving the installed file's ownership
# and SELinux label. The exact destination was resolved from pm path above.
Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "dd", "if=$remoteStage", "of=$target", "conv=fsync"
) | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "shell", "chmod", "0755", $target) | Out-Null
Invoke-Adb -Arguments @("-s", $Serial, "shell", "restorecon", $target) -AllowFailure | Out-Null

$installedShaResult = Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "sha256sum", $target
)
$installedSha = ($installedShaResult.Output -split "\s+")[0].ToLowerInvariant()
if ($installedSha -ne $localSha) {
    throw "Installed probe SHA-256 does not match the host binary."
}

Invoke-Adb -Arguments @(
    "-s", $Serial, "shell", "am", "start",
    "--user", "0", "-n", "$Package/$Activity"
) | Out-Null

[pscustomobject]@{
    ok = $true
    instance = $info.name
    vmIndex = $VmIndex
    serial = $Serial
    package = $Package
    target = $target
    backup = $backup
    previousSha256 = $oldSha
    installedSha256 = $installedSha
    installedBytes = $localLength
    offlineIPv4 = $true
    offlineIPv6 = $true
} | ConvertTo-Json
