"""Hook table refactors retain exact verification and partial-install behavior."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from native_runner.paths import PACKAGE_ROOT


@pytest.mark.parametrize(
    "fixture",
    ["test_phase_runtime_layout.cpp", "test_phase_runtime_telemetry_compile.cpp"],
)
def test_phase_native_layout_and_epoch_cache(tmp_path: Path, fixture: str) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for native phase validation")
    executable = tmp_path / ("phase.exe" if os.name == "nt" else "phase")
    subprocess.run(
        [compiler, "-std=c++17", str(PACKAGE_ROOT / "probe" / fixture), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)


def test_hook_table_verification_and_installation(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable for native hook validation")
    source = tmp_path / "hooks.cpp"
    source.write_text(
        r'''
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#define LOGE(...) ((void)0)
std::vector<std::uintptr_t> installed_offsets;
std::uintptr_t failing_offset = 0;
bool sha256_file(const char*, char (&)[65]) { return false; }
bool libg_build_id(char (&)[41]) { return false; }
bool install_inline_hook(
    std::uintptr_t, std::uintptr_t offset, const std::uint8_t (&)[16],
    const void*, void**, const char*) {
    installed_offsets.push_back(offset);
    return offset != failing_offset;
}
#include "probe_hook.h"
void replacement() {}
int main() {
    std::uint8_t library[64] = {};
    const std::uint8_t first[16] = {1, 2, 3};
    const std::uint8_t second[16] = {4, 5, 6};
    const std::uint8_t third[16] = {7, 8, 9};
    std::memcpy(library + 8, first, 16);
    std::memcpy(library + 24, second, 16);
    std::memcpy(library + 40, third, 16);
    void (*original)() = nullptr;
    std::atomic<bool> first_installed{false};
    std::atomic<bool> second_installed{false};
    std::atomic<bool> third_installed{false};
    const NativeHookSpec hooks[] = {
        native_hook(8, first, &replacement, &original, "first", &first_installed),
        native_hook(24, second, &replacement, &original, "second", &second_installed),
        native_hook(40, third, &replacement, &original, "third", &third_installed),
    };
    const auto base = reinterpret_cast<std::uintptr_t>(library);
    assert(verify_native_hook_table(base, hooks, 3));
    for (std::size_t offset : {std::size_t(8), std::size_t(24), std::size_t(55)}) {
        library[offset] ^= 1;
        assert(!verify_native_hook_table(base, hooks, 3));
        library[offset] ^= 1;
    }
    assert(install_native_hook_table(base, hooks, 3));
    assert(first_installed && second_installed && third_installed);
    assert((installed_offsets == std::vector<std::uintptr_t>{8, 24, 40}));
    installed_offsets.clear();
    failing_offset = 24;
    assert(!install_native_hook_table(base, hooks, 3));
    assert(first_installed && !second_installed && third_installed);
    assert((installed_offsets == std::vector<std::uintptr_t>{8, 24, 40}));
    installed_offsets.clear();
    third_installed = false;
    assert(!install_native_hook_table(base, hooks, 3, true));
    assert(first_installed && !second_installed && !third_installed);
    assert((installed_offsets == std::vector<std::uintptr_t>{8, 24}));
}
''',
        encoding="utf-8",
    )
    executable = tmp_path / ("hooks.exe" if os.name == "nt" else "hooks")
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(PACKAGE_ROOT / "probe"), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
