"""Pin actual native serializers to the pre-refactor wire contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from native_runner.paths import PACKAGE_ROOT


def _declaration_start(source: str, name: str) -> int:
    # Identify the declaration by its symbol, independently of pointer spacing
    # and continuation-line formatting. Calls within bodies are indented.
    match = re.search(r"^\w[^\n;{}]*?\b" + re.escape(name) + r"\b", source, re.MULTILINE)
    assert match is not None, name
    return match.start()


def _block(source: str, start: str, end: str) -> str:
    return source[_declaration_start(source, start) : _declaration_start(source, end)]


def _function(source: str, name: str) -> str:
    start = _declaration_start(source, name)
    body = source.index("{", start)
    # Balanced braces also handle functions formatted on one line. Ignore
    # braces in the JSON string literals and comments inside these serializers.
    tokens = re.compile(r'''//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[{}]''')
    depth = 0
    for token in tokens.finditer(source, body):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
            if depth == 0:
                return source[start : token.end()] + "\n"
    raise AssertionError(f"unterminated native function: {name}")


def test_native_telemetry_wire_contract(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for native serializer validation")
    probe = PACKAGE_ROOT / "probe"
    combat = (probe / "combat_event_telemetry.inc").read_text(encoding="utf-8")
    phase = (probe / "phase_runtime_telemetry.inc").read_text(encoding="utf-8")
    main = (probe / "cr_replay_probe.cpp").read_text(encoding="utf-8")
    source = r'''
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstdarg>
#include <climits>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <iostream>
#include <cassert>
#include "phase_runtime_layout.h"
#include "probe_json.h"
#define LOGE(...) ((void)0)
constexpr std::int32_t kNativeObjectIdEntityKeyTag = -2;
constexpr int kPlayerCount = 2;
'''
    for name in (
        "json_boolean",
        "derive_native_entity_key",
        "format_native_entity_key_json",
        "append_json",
        "append_native_entity_key_json",
    ):
        source += _function(main, name)
    source += _block(
        combat, "kCombatEventSchema", "g_combat_active_deployment"
    )
    source += _block(combat, "combat_raw_kind_label", "read_combat_object_vector")
    source += _block(phase, "kPhaseRuntimeRingCapacity", "PhaseActionDispatch")
    source += _function(phase, "phase_raw_kind_label")
    source += _block(combat, "append_combat_deployment_context", "verify_combat_hook_fingerprints")
    source += _block(phase, "append_native_phase_snapshot_json", "verify_phase_runtime_hook_fingerprints")
    source += Path(__file__).with_name("native_telemetry_json_fixture.cpp").read_text(encoding="utf-8")
    translation_unit = tmp_path / "telemetry.cpp"
    translation_unit.write_text(source, encoding="utf-8")
    executable = tmp_path / ("telemetry.exe" if os.name == "nt" else "telemetry")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", "-I", str(probe), str(translation_unit), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    output = subprocess.check_output([str(executable)]).replace(b"\r\n", b"\n")
    # 120 successful envelope/snapshot fixtures plus five malformed ring/identity
    # cases. The C++ fixture also verifies four insufficient buffer capacities.
    # Compared byte-for-byte with the private baseline in offline mode.
    assert len(output.splitlines()) == 125
    assert hashlib.sha256(output).hexdigest() == (
        "d15bd108d0652cd1991d53fffb70663fbb4912fdbcda50041c46d3277b226d97"
    )
