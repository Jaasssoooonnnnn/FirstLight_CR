param(
    [string]$InputApk = "",
    [string]$OutputApk = "",
    [int]$EngineCount = 0
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "native_runner/local_config.ps1")
if (-not $InputApk) { $InputApk = Get-LocalSetting "CR_INPUT_APK" }
if (-not $OutputApk) { $OutputApk = Get-LocalSetting "CR_OUTPUT_APK" (Join-Path $PSScriptRoot "build/cr-ai-offline.apk") }
if (-not $EngineCount) { $EngineCount = Get-LocalSetting "CR_BUILD_ENGINE_COUNT" "1" }
$python = Get-LocalSetting "CR_PYTHON" "python"
$java = Get-LocalSetting "CR_JAVA"
$buildTools = Get-LocalSetting "CR_BUILD_TOOLS"
$keytool = Get-LocalSetting "CR_KEYTOOL"
$ndk = Get-LocalSetting "CR_NDK_ROOT"
foreach ($required in @($InputApk, $java, $buildTools, $keytool, $ndk)) {
    if (-not $required -or -not (Test-Path -LiteralPath $required)) { throw "Missing build input/tool: '$required'. Configure .env using .env.example." }
}
$InputApk = (Resolve-Path -LiteralPath $InputApk).Path
$OutputApk = [IO.Path]::GetFullPath($OutputApk)
if ($InputApk -eq $OutputApk) { throw "OutputApk must not overwrite the user-provided original APK." }
$work = Join-Path $PSScriptRoot ("build/work-" + [Guid]::NewGuid().ToString("N"))
$decoded = Join-Path $work "decoded"
$probe = Get-LocalSetting "CR_PROBE" (Join-Path $PSScriptRoot "native_runner/probe/out/libcrprobe.so")
$builder = Join-Path $PSScriptRoot "native_runner/offline_build.py"
New-Item -ItemType Directory -Force -Path $work | Out-Null

Write-Host "[1/4] Verify the user-provided original APK"
& $python $builder check-input --input-apk $InputApk
if ($LASTEXITCODE -ne 0) { throw "Original APK verification failed." }

Write-Host "[2/4] Build the native probe from source"
& (Join-Path $PSScriptRoot "native_runner/probe/build_probe.ps1") -NdkRoot $ndk -Output $probe

Write-Host "[3/4] Decode, patch, build and sign locally"
$apktool = Join-Path $PSScriptRoot "native_runner/probe/out/tools/apktool_2.12.1.jar"
New-Item -ItemType Directory -Force -Path (Split-Path $apktool) | Out-Null
if (-not (Test-Path -LiteralPath $apktool)) {
    Invoke-WebRequest -Uri "https://github.com/iBotPeaches/Apktool/releases/download/v2.12.1/apktool_2.12.1.jar" -OutFile $apktool
}
if ((Get-FileHash -LiteralPath $apktool -Algorithm SHA256).Hash.ToLowerInvariant() -ne "66cf4524a4a45a7f56567d08b2c9b6ec237bcdd78cee69fd4a59c8a0243aeafa") {
    throw "Apktool SHA-256 mismatch."
}
& $java -jar $apktool decode $InputApk --output $decoded
if ($LASTEXITCODE -ne 0) { throw "Pristine APK decode failed." }
& (Join-Path $PSScriptRoot "native_runner/build_engine_cluster_apk.ps1") `
    -DecodedApk $decoded -Probe $probe -OutputApk $OutputApk -EngineCount $EngineCount `
    -Java $java -Keytool $keytool `
    -BuildTools $buildTools -Python $python

Write-Host "[4/4] Verify the signed APK payload"
& $python $builder verify --apk $OutputApk --probe $probe
if ($LASTEXITCODE -ne 0) { throw "Output APK verification failed." }
$settings = @"
CR_DECODED_APK=$decoded
CR_PROBE=$probe
CR_OUTPUT_APK=$OutputApk
"@
$settingsPath = Join-Path $work "generated.env"
[IO.File]::WriteAllText($settingsPath, $settings, [Text.UTF8Encoding]::new($false))
Write-Host "Built: $OutputApk"
Write-Host "Build receipt settings: $settingsPath"
Write-Host "The probe is ready at the configured/default runtime path: $probe"
