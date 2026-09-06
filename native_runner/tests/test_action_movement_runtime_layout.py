from __future__ import annotations

from native_runner.paths import PACKAGE_ROOT
from native_runner.tests.asset_helpers import user_apk_bytes

import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest


PROBE = PACKAGE_ROOT / "probe"


def test_action_movement_exact_build_fingerprints() -> None:
    data = user_apk_bytes("lib/arm64-v8a/libg.so")
    assert hashlib.sha256(data).hexdigest() == (
        "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783"
    )
    assert data[0xF49D24 : 0xF49D24 + 16].hex() == (
        "fd7bbba9fa6701a9f85f02a9f65703a9"
    )
    assert data[0xE82A34 : 0xE82A34 + 16].hex() == (
        "ffc301d1fd7b01a9fc6f02a9fa6703a9"
    )


@pytest.mark.parametrize(
    "source_name",
    (
        "test_action_movement_runtime_layout.cpp",
        "test_action_movement_runtime_telemetry_compile.cpp",
    ),
)
def test_action_movement_host_cpp_contracts_compile(
    tmp_path: Path,
    source_name: str,
) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for the action movement host test")
    executable = tmp_path / f"{Path(source_name).stem}.exe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(PROBE / source_name),
            f"-I{PROBE}",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=PACKAGE_ROOT,
    )
    subprocess.run([str(executable)], check=True)
