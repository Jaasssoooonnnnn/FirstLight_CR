from __future__ import annotations

from native_runner.paths import PACKAGE_ROOT
from native_runner.tests.asset_helpers import user_apk_bytes

import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest


PROBE = PACKAGE_ROOT / "probe"


def test_tower_troop_exact_build_fingerprints() -> None:
    data = user_apk_bytes("lib/arm64-v8a/libg.so")
    assert hashlib.sha256(data).hexdigest() == (
        "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783"
    )
    assert data[0xF43F10 : 0xF43F10 + 16].hex() == (
        "fd7bbea9f30b00f9fd030091f30300aa"
    )
    assert data[0xF43F3C : 0xF43F3C + 16].hex() == (
        "fd7bbca9f70b00f9f65702a9f44f03a9"
    )
    assert data[0xF47E08 : 0xF47E08 + 16].hex() == (
        "fd7bbea9f30b00f9fd030091f30300aa"
    )
    assert data[0xF47E3C : 0xF47E3C + 16].hex() == (
        "fd7bbaa9fc6f01a9fa6702a9f85f03a9"
    )


def test_tower_troop_host_layout_contract_compiles(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for the tower-troop host test")
    executable = tmp_path / "test_tower_troop_runtime_layout.exe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(PROBE / "test_tower_troop_runtime_layout.cpp"),
            f"-I{PROBE}",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=PACKAGE_ROOT,
    )
    subprocess.run([str(executable)], check=True)
