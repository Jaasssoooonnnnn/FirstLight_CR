# Shared literal KEY=value reader. Existing process variables take precedence.
function Import-LocalEnv {
    $configurationPath = Join-Path (Split-Path -Parent $PSScriptRoot) ".env"
    if (Test-Path -LiteralPath $configurationPath -PathType Leaf) {
        foreach ($line in [IO.File]::ReadAllLines($configurationPath)) {
            $entry = $line.Trim()
            if (-not $entry -or $entry.StartsWith('#')) { continue }
            if ($entry -notmatch '^([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
                throw "Invalid .env entry; expected KEY=value"
            }
            $key = $Matches[1]
            $value = $Matches[2].Trim()
            if ($value.StartsWith('"') -or $value.StartsWith("'")) {
                if ($value.Length -lt 2 -or $value[-1] -ne $value[0]) { throw "Unmatched quote in .env" }
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($value -and $null -eq [Environment]::GetEnvironmentVariable($key, 'Process')) {
                [Environment]::SetEnvironmentVariable($key, $value, 'Process')
            }
        }
    }
}
Import-LocalEnv

function Get-LocalSetting {
    param([string]$Name, [string]$Default = "")
    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ($value) { return $value }
    if ($Default) { return $Default }
    throw "Set $Name in .env (see .env.example)."
}
