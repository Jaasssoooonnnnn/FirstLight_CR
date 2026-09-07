#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

std::uintptr_t g_libg_base = 0;
const char* g_libg_path = "";

#define LOGE(...) ((void)0)
#define LOGI(...) ((void)0)

template <typename T>
T read_object_field(const void* object, std::size_t offset) {
    T value{};
    if (object != nullptr) {
        std::memcpy(
            &value,
            static_cast<const std::uint8_t*>(object) + offset,
            sizeof(value));
    }
    return value;
}

bool process_range_is_readable(const void*, std::size_t) {
    return false;
}

bool read_bounded_libg_string_utf8(
    const void*,
    char*,
    std::size_t,
    std::size_t*) {
    return false;
}

struct CombatEntityFact {
    bool validated = false;
    bool present = false;
    bool visibility_validated = false;
    std::uint32_t native_object_id = 0;
    std::int32_t owner = 0;
    std::int32_t object_index = 0;
    std::int32_t secondary_index = 0;
    std::int32_t card_id = 0;
    std::int32_t object_kind = -1;
    std::int32_t position_x = 0;
    std::int32_t position_y = 0;
    std::int32_t invisible_count = 0;
};

struct CombatCaptureContext {
    void* game_manager = nullptr;
    void* object_manager = nullptr;
    std::uint64_t generation = 0;
    std::uint64_t state_epoch = 0;
    std::int32_t tick = -1;
};

bool current_combat_capture_context(CombatCaptureContext*) {
    return false;
}

void capture_combat_entity(
    const void*,
    void*,
    CombatEntityFact*) {}

const char* json_boolean(bool value) {
    return value ? "true" : "false";
}

bool append_json(
    char*,
    std::size_t,
    std::size_t*,
    const char*,
    ...) {
    return true;
}

bool append_json_utf8_string(
    char*,
    std::size_t,
    std::size_t*,
    const char*,
    std::size_t) {
    return true;
}

bool append_combat_entity_fact(
    char*,
    std::size_t,
    std::size_t*,
    const CombatEntityFact&) {
    return true;
}

bool sha256_file(const char*, char (&)[65]) {
    return false;
}

bool libg_build_id(char (&)[41]) {
    return false;
}

bool install_inline_hook(
    std::uintptr_t,
    std::uintptr_t,
    const std::uint8_t (&)[16],
    const void*,
    void**,
    const char*) {
    return false;
}

#include "action_movement_runtime_telemetry.inc"

int main() {
    assert(
        std::strcmp(
            action_movement_kind_label(
                ActionMovementRawKind::GoldenKnightHopLaunch),
            "golden_knight_chain_hop_launch") == 0);
    assert(
        std::strcmp(
            action_movement_stage_label(
                ActionMovementRawKind::
                    BossBanditWarpPositionCommit),
            "position_commit") == 0);

    g_action_movement_hooks_installed.store(
        true, std::memory_order_release);
    begin_action_movement_runtime_epoch(7, 11);

    ActionMovementRawRecord record;
    record.generation = 7;
    record.state_epoch = 11;
    record.tick = 25;
    record.kind = ActionMovementRawKind::GoldenKnightHopLaunch;
    record.hook_offset =
        cr_action_movement_runtime::kGoldenKnightHopOffset;
    record.caller_offset = 0xf49f10;
    record.action_data_global_id = 123;
    record.action_class_vtable_offset =
        cr_action_movement_runtime::
            kDashingAttackChainDataVtableOffset;
    record.runtime_class_vtable_offset =
        cr_action_movement_runtime::
            kDashingAttackChainRuntimeVtableOffset;
    std::strcpy(
        record.action_name,
        cr_action_movement_runtime::
            kGoldenKnightExecuteActionName);
    record.action_name_size = std::strlen(record.action_name);
    record.source_before.validated = true;
    record.source_before.present = true;
    record.source_before.native_object_id = 1;
    record.source_before.owner = 0;
    record.source_before.card_id =
        cr_action_movement_runtime::kGoldenKnightCardId;
    record.source_before.object_kind = 1;
    record.source_after = record.source_before;
    record.target.validated = true;
    record.target.present = true;
    record.target.native_object_id = 2;
    record.target.owner = 1;
    record.target.object_kind = 1;
    record.target_present = true;
    record.chain_index = 0;
    record.complete_context = true;

    assert(same_action_movement_entity(
        record.source_before, record.source_after));
    assert(publish_action_movement_record(record) == 1);
    assert(
        g_action_movement_ring.next_sequence.load(
            std::memory_order_acquire) == 2);
    return 0;
}
