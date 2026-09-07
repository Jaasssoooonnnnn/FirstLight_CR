from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from native_runner.paths import PACKAGE_ROOT


@pytest.mark.parametrize("source", [
    "test_epoch_ring.cpp", "test_movement_event_json.cpp",
    "test_tower_visibility_json.cpp", "test_visibility_runtime_telemetry_compile.cpp",
])
def test_epoch_ring_retention_and_original_wire_contract(tmp_path: Path, source: str) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    probe = PACKAGE_ROOT / "probe"
    executable = tmp_path / (Path(source).stem + ".exe")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(probe / source), f"-I{probe}", "-o", str(executable)],
        check=True,
    )
    output = subprocess.check_output([str(executable)]).replace(b"\r\n", b"\n")
    # Captured before refactoring: ring overflow/epochs/future ticks, nullable
    # fields and rejection counters; tower latest selection and snapshot copy.
    # Tower hash uses the private baseline with online delta mode disabled.
    original_hashes = {
        "test_movement_event_json.cpp": "e5901a3f21a248c3eb68744c619748735b4d968ec5bb801ba3076774812d7429",
        "test_tower_visibility_json.cpp": "77e66d15adc84552e861c2859049e9e5c6adbe7f222311cc2aa9abd27a6adf15",
    }
    if source in original_hashes:
        assert hashlib.sha256(output).hexdigest() == original_hashes[source]
