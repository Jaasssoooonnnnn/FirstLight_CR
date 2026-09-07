"""Run the probe's actual source resolver against two ready controllers."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from native_runner.tests.cpp_source import cpp_function


PROBE = Path(__file__).resolve().parents[1] / "probe" / "cr_replay_probe.cpp"


def _cpp_record(source: str, declaration: str) -> str:
    """Copy these plain struct/enum declarations without altering their fields."""
    matches = list(re.finditer(rf"(?m)^[ \t]*{declaration}[^;{{]*\{{[^{{}}]*\}}\s*;", source))
    assert len(matches) == 1, f"Expected one C++ record for {declaration!r}"
    return matches[0].group()


def test_native_dual_ready_abilities_keep_the_exact_selected_source(tmp_path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native source-routing regression")
    source = PROBE.read_text(encoding="utf-8")
    definitions = [
        _cpp_record(source, r"struct\s+AbilityRequest\b"),
        _cpp_record(source, r"enum\s+class\s+AbilityResolutionStatus\b"),
        cpp_function(source, "resolve_ready_ability_request"),
        cpp_function(source, "format_ability_command"),
    ]
    assert "live_stock_ability_is_unique_queueable" not in source

    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

constexpr int kPlayerCount = 2;
constexpr int kNativeObjectIdEntityKeyTag = -2;
constexpr int kUniqueReadyAbilityEntityKeyTag = -3;
struct PlayerStateView {
    void* player = nullptr;
    std::uint32_t account_id_high = 0, account_id_low = 0;
};
struct NativeVectorView { void* data = nullptr; int count = 0; };
struct NativeEntityReferenceView {
    bool validated = true, has_value = true;
    int owner = 0, slot = -1, object_index = -1, secondary_index = 0;
    std::uint64_t native_object_id = 0;
};
struct NativeChampionControllerView {
    int controller_slot = -1, action_data_global_id = 0;
    int remaining_charges_raw = 1, button_state = 2, remaining_cooldown_ms = 0;
    const char* action_data_name = "ability";
    std::size_t action_data_name_size = 7;
    int champion_count = 1;
    NativeEntityReferenceView champions[1];
};
using Blob = std::array<unsigned char, 1024>;
Blob manager{}, roots[2]{};
std::array<unsigned char, 16> carriers[2]{};
void* objects[2] = {carriers[0].data(), carriers[1].data()};
NativeChampionControllerView controllers[2];
int world;
template<class T> T read_object_field(const void* p, std::size_t offset) {
    T value;
    std::memcpy(&value, static_cast<const unsigned char*>(p) + offset, sizeof(T));
    return value;
}
template<class T> void put(void* p, std::size_t offset, T value) {
    std::memcpy(static_cast<unsigned char*>(p) + offset, &value, sizeof(T));
}
bool process_range_is_readable(const void* p, std::size_t) { return p != nullptr; }
bool read_player_state(void* w, int owner, PlayerStateView* out) {
    if (w != &world || owner < 0 || owner > 1) return false;
    out->player = roots[owner].data();
    out->account_id_low = 71009049 + owner;
    return true;
}
bool read_native_vector(void* p, std::size_t, int, NativeVectorView* out) {
    if (p != objects) return false;
    out->data = objects;
    out->count = 2;
    return true;
}
bool read_champion_controller(const void*, int, int slot, const void* p,
    void**, int, NativeChampionControllerView* out) {
    if (slot < 1 || slot > 2 || p != &controllers[slot - 1]) return false;
    *out = controllers[slot - 1];
    return true;
}
bool champion_controller_is_exact_empty(const void*, const void*) { return false; }
bool champion_controller_is_exact_absent(const void*, const void*) { return false; }
bool ascii_case_insensitive_contains(const char*, std::size_t, const char*) {
    return false;
}
bool ability_button_state_is_queueable(int state) { return state == 2 || state == 4; }
'''
    harness += "\n" + "\n\n".join(definitions) + "\n"
    harness += r'''
int main() {
    put(manager.data(), 0xa8, static_cast<void*>(&world));
    for (int i = 0; i < 2; ++i) {
        put(roots[i].data(), 0x10, static_cast<void*>(objects));
        put(roots[0].data(), 0x3a0 + i * sizeof(void*),
            static_cast<const void*>(&controllers[i]));
        put(carriers[i].data(), 0x08, 5000100 + i);
        controllers[i].controller_slot = i + 1;
        controllers[i].action_data_global_id = 29000000 + i;
        controllers[i].champions[0].native_object_id = 5000100 + i;
        controllers[i].champions[0].slot = i;
        controllers[i].champions[0].object_index = i;
    }
    AbilityRequest selected;
    AbilityResolutionStatus status;
    auto resolve = [&](int source, int owner = 0) {
        return resolve_ready_ability_request(manager.data(), owner,
            kNativeObjectIdEntityKeyTag, source, &selected, nullptr, &status);
    };
    // Both ready: either explicitly selected carrier retains its own command.
    for (int i = 0; i < 2; ++i) {
        assert(resolve(5000100 + i));
        assert(status == AbilityResolutionStatus::Resolved);
        assert(selected.controller_slot == i + 1);
        assert(selected.action_data_global_id == 29000000 + i);
        assert(selected.champion_game_id == 5000100 + i);
        char command[256]{};
        assert(format_ability_command(selected, 4275, command, sizeof(command)));
        assert(std::string(command).find("\"cgid\":" +
            std::to_string(5000100 + i)) != std::string::npos);
    }
    // A different ready peer cannot rescue an unavailable selected source.
    controllers[0].remaining_cooldown_ms = 1000;
    assert(!resolve(5000100));
    assert(resolve(5000101));
    controllers[0].remaining_cooldown_ms = 0;
    controllers[0].remaining_charges_raw = 0;
    assert(!resolve(5000100));
    controllers[0].remaining_charges_raw = 1;
    controllers[0].button_state = 7;
    assert(!resolve(5000100));
    controllers[0].button_state = 2;
    assert(!resolve(9999999));
    assert(!resolve(5000100, 1));
    // One identity reported by two controllers remains genuinely ambiguous.
    controllers[1].champions[0].native_object_id = 5000100;
    assert(!resolve(5000100));
    assert(status == AbilityResolutionStatus::Ambiguous);
    std::puts("dual-ready exact-source routing and rejection checks passed");
}
'''
    cpp = tmp_path / "dual_ability.cpp"
    executable = tmp_path / "dual_ability.exe"
    cpp.write_text(harness, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    assert "exact-source routing and rejection checks passed" in result.stdout
