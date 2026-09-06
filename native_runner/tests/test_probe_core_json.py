"""Golden bytes from the retained native record encoders before simplification."""

import hashlib
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERIALIZERS = (
    "append_native_capture_runtime_json", "append_native_threshold_relocation_json",
    "append_native_projectile_json", "append_native_entity_resource_json",
    "append_native_periodic_attack_modifier_json", "append_native_player_runtime_json",
)
DEPENDENCIES = (
    "derive_native_entity_key", "format_native_entity_key_json", "append_json",
    "append_native_entity_key_json", "valid_utf8_bytes", "append_json_utf8_string",
    "native_periodic_attack_modifier_phase_label", "native_capture_target_phase_label",
    "native_threshold_relocation_phase_label", "ability_button_state_label",
    "ability_button_state_is_queueable", "append_native_entity_reference_json",
)


def _end_brace(source: str, start: int) -> int:
    """Find a C++ block end without counting braces inside comments or literals."""
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, start):
        if match[0] == "{":
            depth += 1
        elif match[0] == "}":
            depth -= 1
            if depth == 0:
                return match.end()
    raise AssertionError("Unclosed C++ block")


def _serializer_source(source: str) -> str:
    functions = []
    for name in (*DEPENDENCIES, *SERIALIZERS):
        match = re.search(r"^.*\b" + name + r"\(", source, re.M)
        assert match is not None, name
        opening = source.index("{", match.end())
        functions.append(source[match.start():_end_brace(source, opening)])
    types: dict[str, tuple[int, str]] = {}
    pending = set(re.findall(r"\bNative\w+(?:View|Phase)\b", "\n".join(functions)))
    while pending:
        name = pending.pop()
        if name in types:
            continue
        match = re.search(r"^(?:struct|enum class) " + name + r"\b", source, re.M)
        assert match is not None, name
        end = _end_brace(source, source.index("{", match.end())) + 1
        definition = source[match.start():end]
        types[name] = (match.start(), definition)
        pending.update(re.findall(r"\bNative\w+(?:View|Phase)\b", definition))
    body = "\n".join(text for _, text in sorted(types.values())) + "\n" + "\n".join(functions)
    constants = []
    for name in sorted(set(re.findall(r"\bk[A-Z]\w+\b", body))):
        if re.search(r"constexpr[^;]*\b" + name + r"\b", body):
            continue
        match = re.search(r"^constexpr[^;]*\b" + name + r"\s*=[^;]*;", source, re.M)
        assert match is not None, name
        constants.append(match[0])
    return "\n".join(constants) + "\n" + body


def test_core_native_json_preserves_nullable_fields_precision_and_bounds(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native serialization test")
    source = (ROOT / "probe" / "cr_replay_probe.cpp").read_text(encoding="utf-8")
    headers = "\n".join(f"#include <{name}>" for name in (
        "cstdint", "cstddef", "cstring", "cstdio", "cstdarg", "climits", "initializer_list",
    )) + '\n#include "probe_json.h"\n'
    fixture = Path(__file__).with_name("probe_core_json_fixture.cpp").read_text(encoding="utf-8")
    cpp = tmp_path / "core_json.cpp"
    executable = tmp_path / "core_json.exe"
    cpp.write_text(headers + _serializer_source(source) + fixture, encoding="utf-8")
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-I", str(ROOT / "probe"), str(cpp), "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    output = subprocess.check_output([str(executable)])
    # 1,080 complete records and 4,320 exact-boundary/overflow checks, including
    # all ability phases, nullable entities, UTF-8 and escaped control bytes.
    assert len(output) == 508_076
    assert hashlib.sha256(output).hexdigest() == (
        "ed43b1a42e3a88b3fa3ca03721ae230b3fda46d3d3115fc830a2990eb535a753"
    )
