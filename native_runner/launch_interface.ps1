param([Parameter(ValueFromRemainingArguments = $true)][string[]]$InterfaceArguments)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
. (Join-Path $PSScriptRoot "local_config.ps1")
$repository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repository
if ($InterfaceArguments -contains "--diagnose") {
    $python = Get-LocalSetting "CR_PYTHON"
    & $python -X utf8 -m native_runner.user_interface @InterfaceArguments
    exit $LASTEXITCODE
}
$pythonw = Get-LocalSetting "CR_PYTHONW"
$arguments = @("-X", "utf8", "-m", "native_runner.user_interface")
if ($InterfaceArguments) { $arguments += $InterfaceArguments }
$quoted = $arguments | ForEach-Object { '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' }
Start-Process -FilePath $pythonw -ArgumentList ($quoted -join ' ') -WorkingDirectory $repository -WindowStyle Hidden
