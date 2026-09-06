from __future__ import annotations

from native_runner.paths import WORKSPACE_ROOT, PACKAGE_ROOT

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = WORKSPACE_ROOT
PROBE = PACKAGE_ROOT / "probe"
CAPTURE_TEST = PROBE / "test_combat_damage_capture.cpp"


def test_host_combat_damage_capture_policy(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for the combat damage capture test")
    executable = tmp_path / (
        "combat-damage-capture-test.exe"
        if os.name == "nt"
        else "combat-damage-capture-test"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            str(CAPTURE_TEST),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(executable)],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
