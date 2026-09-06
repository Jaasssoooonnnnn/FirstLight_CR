param(
    [string]$DecodedApk = "",
    [string]$Probe = (Join-Path $PSScriptRoot "probe\out\libcrprobe.so"),
    [string]$OutputApk = (Join-Path (Split-Path -Parent $PSScriptRoot) "builds\cr_engine_cluster_signed.apk"),
    [int]$EngineCount = 0,
    [switch]$Debuggable,
    [string]$Java = "",
    [string]$Keytool = "",
    [string]$BuildTools = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "local_config.ps1")
if (-not $PSBoundParameters.ContainsKey("DecodedApk")) { $DecodedApk = Get-LocalSetting "CR_DECODED_APK" }
if (-not $PSBoundParameters.ContainsKey("Java")) { $Java = Get-LocalSetting "CR_JAVA" }
if (-not $PSBoundParameters.ContainsKey("Keytool")) { $Keytool = Get-LocalSetting "CR_KEYTOOL" }
if (-not $PSBoundParameters.ContainsKey("BuildTools")) { $BuildTools = Get-LocalSetting "CR_BUILD_TOOLS" }
if (-not $PSBoundParameters.ContainsKey("EngineCount")) { $EngineCount = Get-LocalSetting "CR_BUILD_ENGINE_COUNT" "1" }
if (-not $PSBoundParameters.ContainsKey("Probe") -and $env:CR_PROBE) { $Probe = $env:CR_PROBE }
if (-not $PSBoundParameters.ContainsKey("OutputApk") -and $env:CR_OUTPUT_APK) { $OutputApk = $env:CR_OUTPUT_APK }

if (-not $Python) { $Python = Get-LocalSetting "CR_PYTHON" "python" }
$package = "nullsroyale.rel.free"
$apktoolVersion = "2.12.1"
$apktoolSha256 = "66cf4524a4a45a7f56567d08b2c9b6ec237bcdd78cee69fd4a59c8a0243aeafa"
$apktoolUrl = "https://github.com/iBotPeaches/Apktool/releases/download/v$apktoolVersion/apktool_$apktoolVersion.jar"
$buildRoot = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "probe\out\engine-cluster-apk")
)
$sourceRoot = Join-Path $buildRoot "source"
$apktool = Join-Path $PSScriptRoot "probe\out\tools\apktool_$apktoolVersion.jar"
$zipalign = Join-Path $BuildTools "zipalign.exe"
$apksigner = Join-Path $BuildTools "apksigner.bat"
$keystore = Join-Path $PSScriptRoot "probe\crprobe-cluster.keystore"
$passwordFile = "$keystore.pass"
$unsignedApk = Join-Path $buildRoot "cluster-unsigned.apk"
$alignedApk = Join-Path $buildRoot "cluster-aligned.apk"
$signedApk = Join-Path $buildRoot "cluster-signed.apk"

if ($EngineCount -lt 1 -or $EngineCount -gt 24) {
    throw "The cluster APK supports between 1 and 24 engine processes."
}
foreach ($required in @(
    $DecodedApk,
    (Join-Path $DecodedApk "AndroidManifest.xml"),
    (Join-Path $DecodedApk "apktool.yml"),
    (Join-Path $DecodedApk "smali\com\supercell\clashroyale\GameApp.smali"),
    $Probe,
    $Java,
    $Keytool,
    $zipalign,
    $apksigner
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required cluster-build input: $required"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $apktool) | Out-Null
if (-not (Test-Path -LiteralPath $apktool -PathType Leaf)) {
    Invoke-WebRequest -Uri $apktoolUrl -OutFile $apktool
}
$actualApktoolSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $apktool
).Hash.ToLowerInvariant()
if ($actualApktoolSha256 -ne $apktoolSha256) {
    throw "Apktool $apktoolVersion SHA-256 mismatch: $actualApktoolSha256"
}
$reportedVersion = (& $Java -jar $apktool --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $reportedVersion -ne $apktoolVersion) {
    throw "Expected Apktool $apktoolVersion, got '$reportedVersion'."
}

$allowedBuildRoot = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "probe\out")
).TrimEnd("\")
if (
    -not $buildRoot.StartsWith(
        "$allowedBuildRoot\",
        [StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "Refusing to clean build directory outside probe\out: $buildRoot"
}
if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null

& robocopy $DecodedApk $sourceRoot /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "Could not copy the decoded APK tree (robocopy exit $LASTEXITCODE)."
}

$patchArgs = @((Join-Path $PSScriptRoot "offline_build.py"), "patch", "--source", $sourceRoot)
& $Python @patchArgs
if ($LASTEXITCODE -ne 0) { throw "Could not patch the pristine APK bootstrap." }

$manifestPath = Join-Path $sourceRoot "AndroidManifest.xml"
$manifestText = [IO.File]::ReadAllText($manifestPath)
$applicationAnchor = '<application android:allowBackup="true"'
$manifestAnchor = '        <meta-data android:name="io.sentry.auto-init"'
if (
    -not $manifestText.Contains($applicationAnchor) -or
    -not $manifestText.Contains($manifestAnchor)
) {
    throw "Could not find the application manifest insertion point."
}
$manifestText = $manifestText.Replace(
    $applicationAnchor,
    '<application android:name="com.supercell.clashroyale.EngineClusterApplication" android:allowBackup="true"'
)
if ($Debuggable) {
    $debuggableAnchor =
        '<application android:name="com.supercell.clashroyale.EngineClusterApplication"'
    if (-not $manifestText.Contains($debuggableAnchor)) {
        throw "Could not find the debuggable manifest insertion point."
    }
    $manifestText = $manifestText.Replace(
        $debuggableAnchor,
        "$debuggableAnchor android:debuggable=`"true`""
    )
}
$clusterActivities = [Collections.Generic.List[string]]::new()
for ($engineId = 0; $engineId -lt $EngineCount; ++$engineId) {
    $className = "Engine$($engineId)App"
    $clusterActivities.Add(
        "        <activity android:configChanges=`"keyboard|keyboardHidden|navigation|orientation|screenLayout|screenSize|smallestScreenSize`" " +
        "android:excludeFromRecents=`"true`" android:exported=`"true`" android:launchMode=`"singleTask`" " +
        "android:name=`"com.supercell.clashroyale.$className`" android:process=`":engine$engineId`" " +
        "android:screenOrientation=`"sensorPortrait`" android:taskAffinity=`"$package.engine$engineId`" " +
        "android:theme=`"@android:style/Theme.NoTitleBar.Fullscreen`" " +
        "android:windowSoftInputMode=`"adjustNothing|stateUnchanged`"/>"
    )
}
$manifestText = $manifestText.Replace(
    $manifestAnchor,
    (($clusterActivities -join "`r`n") + "`r`n" + $manifestAnchor)
)
[IO.File]::WriteAllText(
    $manifestPath,
    $manifestText,
    [Text.UTF8Encoding]::new($false)
)

$smaliDirectory = Join-Path $sourceRoot "smali\com\supercell\clashroyale"
$applicationSmali = @"
.class public Lcom/supercell/clashroyale/EngineClusterApplication;
.super Landroid/app/Application;


# direct methods
.method public constructor <init>()V
    .locals 0

    invoke-direct {p0}, Landroid/app/Application;-><init>()V

    return-void
.end method


# virtual methods
.method protected attachBaseContext(Landroid/content/Context;)V
    .locals 4

    invoke-super {p0, p1}, Landroid/app/Application;->attachBaseContext(Landroid/content/Context;)V

    sget v0, Landroid/os/Build`$VERSION;->SDK_INT:I

    const/16 v1, 0x1c

    if-lt v0, v1, :done

    invoke-static {}, Landroid/app/Application;->getProcessName()Ljava/lang/String;

    move-result-object v0

    const-string v1, ":engine"

    invoke-virtual {v0, v1}, Ljava/lang/String;->lastIndexOf(Ljava/lang/String;)I

    move-result v2

    if-ltz v2, :done

    add-int/lit8 v2, v2, 0x1

    invoke-virtual {v0, v2}, Ljava/lang/String;->substring(I)Ljava/lang/String;

    move-result-object v3

    const-string v0, "engine0"

    invoke-virtual {v3, v0}, Ljava/lang/String;->equals(Ljava/lang/Object;)Z

    move-result v0

    if-nez v0, :done

    invoke-static {v3}, Landroid/webkit/WebView;->setDataDirectorySuffix(Ljava/lang/String;)V

    :done
    return-void
.end method
"@
[IO.File]::WriteAllText(
    (Join-Path $smaliDirectory "EngineClusterApplication.smali"),
    $applicationSmali,
    [Text.UTF8Encoding]::new($false)
)
for ($engineId = 0; $engineId -lt $EngineCount; ++$engineId) {
    $className = "Engine$($engineId)App"
    $smali = @"
.class public Lcom/supercell/clashroyale/$className;
.super Lcom/supercell/clashroyale/GameApp;


# direct methods
.method public constructor <init>()V
    .locals 0

    invoke-direct {p0}, Lcom/supercell/clashroyale/GameApp;-><init>()V

    return-void
.end method
"@
    [IO.File]::WriteAllText(
        (Join-Path $smaliDirectory "$className.smali"),
        $smali,
        [Text.UTF8Encoding]::new($false)
    )
}

$probeDestination = Join-Path $sourceRoot "lib\arm64-v8a\libcrprobe.so"
New-Item -ItemType Directory -Force -Path (
    Split-Path $probeDestination
) | Out-Null
Copy-Item -LiteralPath $Probe -Destination $probeDestination -Force

if ((Test-Path $keystore) -xor (Test-Path $passwordFile)) {
    throw "Cluster signing key/password files are incomplete; keep or remove both."
}
if (-not (Test-Path -LiteralPath $keystore)) {
    $password = (
        [Guid]::NewGuid().ToString("N") +
        [Guid]::NewGuid().ToString("N")
    )
    [IO.File]::WriteAllText(
        $passwordFile,
        $password,
        [Text.UTF8Encoding]::new($false)
    )
    & $Keytool `
        -genkeypair `
        -keystore $keystore `
        -storepass $password `
        -keypass $password `
        -alias "crprobe-cluster" `
        -keyalg RSA `
        -keysize 2048 `
        -validity 36500 `
        -dname "CN=Firstlight CR Local Build" `
        -noprompt | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the local cluster signing key."
    }
}
$password = [IO.File]::ReadAllText($passwordFile).Trim()
if (-not $password) {
    throw "The local cluster signing password file is empty."
}

& $Java -jar $apktool build $sourceRoot --force --output $unsignedApk
if ($LASTEXITCODE -ne 0) {
    throw "Apktool could not build the cluster APK."
}
& $zipalign -f -p 4 $unsignedApk $alignedApk
if ($LASTEXITCODE -ne 0) {
    throw "zipalign could not align the cluster APK."
}
& $apksigner sign `
    --ks $keystore `
    --ks-key-alias "crprobe-cluster" `
    --ks-pass "pass:$password" `
    --key-pass "pass:$password" `
    --out $signedApk `
    $alignedApk
if ($LASTEXITCODE -ne 0) {
    throw "apksigner could not sign the cluster APK."
}
& $apksigner verify --verbose --print-certs $signedApk
if ($LASTEXITCODE -ne 0) {
    throw "Cluster APK signature verification failed."
}

New-Item -ItemType Directory -Force -Path (
    Split-Path $OutputApk
) | Out-Null
Copy-Item -LiteralPath $signedApk -Destination $OutputApk -Force
$output = Get-Item -LiteralPath $OutputApk
$outputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputApk).Hash
$probeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Probe).Hash

[pscustomobject]@{
    ok = $true
    version = "engine-cluster-apk.v1"
    package = $package
    engineProcesses = $EngineCount
    interactiveMainActivity = $true
    probeSha256 = $probeHash.ToLowerInvariant()
    apkSha256 = $outputHash.ToLowerInvariant()
    apkBytes = $output.Length
    outputApk = $output.FullName
} | ConvertTo-Json
