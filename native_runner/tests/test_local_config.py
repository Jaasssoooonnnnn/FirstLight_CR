from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from native_runner.local_config import load_env
from native_runner.paths import PACKAGE_ROOT, REPOSITORY_ROOT


def test_env_values_are_literal_and_process_environment_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRSTLIGHT_TEST_PATH", raising=False)
    monkeypatch.setenv("FIRSTLIGHT_TEST_OVERRIDE", "process")
    monkeypatch.delenv("FIRSTLIGHT_TEST_EMPTY", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "# Literal paths and shell metacharacters\n"
        "FIRSTLIGHT_TEST_PATH='folder with spaces/$HOME/$(not-a-command)=value'\n"
        "FIRSTLIGHT_TEST_OVERRIDE=file\nFIRSTLIGHT_TEST_EMPTY=\n",
        encoding="utf-8",
    )
    load_env(path)
    assert os.environ["FIRSTLIGHT_TEST_PATH"] == "folder with spaces/$HOME/$(not-a-command)=value"
    assert os.environ["FIRSTLIGHT_TEST_OVERRIDE"] == "process"
    assert "FIRSTLIGHT_TEST_EMPTY" not in os.environ


@pytest.mark.parametrize("line", ["NOT AN ENTRY", "KEY='unclosed", "1KEY=value"])
def test_invalid_configuration_reports_the_line(tmp_path, line):
    path = tmp_path / ".env"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.env:1:"):
        load_env(path)


def test_public_example_contains_only_blank_settings():
    entries = [line for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    assert entries
    assert all(line.partition("=")[2] == "" for line in entries)
    keys = [line.partition("=")[0] for line in entries]
    assert len(keys) == len(set(keys))


def test_powershell_reader_preserves_literal_paths_and_override(tmp_path):
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is not installed")
    package = tmp_path / "native_runner"
    package.mkdir()
    script = package / "local_config.ps1"
    shutil.copyfile(PACKAGE_ROOT / script.name, script)
    (tmp_path / ".env").write_text(
        "FIRSTLIGHT_TEST_PATH='folder with spaces/$HOME/$(not-a-command)=value'\n"
        "FIRSTLIGHT_TEST_OVERRIDE=file\n", encoding="utf-8",
    )
    env = dict(os.environ, FIRSTLIGHT_TEST_OVERRIDE="process")
    env.pop("FIRSTLIGHT_TEST_PATH", None)
    command = "$value='caller-value'; . '" + str(script).replace("'", "''") + "'; " + (
        "@((Get-LocalSetting FIRSTLIGHT_TEST_PATH), "
        "(Get-LocalSetting FIRSTLIGHT_TEST_OVERRIDE), $value) | ConvertTo-Json -Compress"
    )
    result = subprocess.run([powershell, "-NoProfile", "-Command", command],
                            env=env, capture_output=True, text=True, check=True, timeout=15)
    assert json.loads(result.stdout) == ["folder with spaces/$HOME/$(not-a-command)=value", "process", "caller-value"]


def test_powershell_entry_points_parse():
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is not installed")
    package = str(PACKAGE_ROOT).replace("'", "''")
    command = (
        f"Get-ChildItem -LiteralPath '{package}' -Recurse -Filter *.ps1 | ForEach-Object {{ "
        "$tokens=$null; $errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        "$_.FullName, [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { throw ($errors | Out-String) } }"
    )
    subprocess.run([powershell, "-NoProfile", "-Command", command],
                   capture_output=True, text=True, check=True, timeout=15)


def test_interface_launcher_passes_no_spurious_empty_argument():
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is not installed")
    launcher = str(PACKAGE_ROOT / "launch_interface.ps1").replace("'", "''")
    command = (
        "function Start-Process { param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle); "
        "$ArgumentList | ConvertTo-Json -Compress }; "
        f". '{launcher}'"
    )
    env = dict(os.environ, CR_PYTHONW="test-interpreter")
    result = subprocess.run([powershell, "-NoProfile", "-Command", command],
                            env=env, capture_output=True, text=True, check=True, timeout=15)
    assert json.loads(result.stdout) == '"-X" "utf8" "-m" "native_runner.user_interface"'
