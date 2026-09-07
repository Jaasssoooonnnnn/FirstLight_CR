#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include "probe_json.h"

#if defined(__aarch64__)
#include <sys/mman.h>
#endif

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

bool read_combat_object_vector(
    void*,
    void***,
    std::int32_t*) {
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

bool combat_component_type_is(const void*, const void*, std::int32_t) {
    return false;
}

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

bool append_combat_entity_fact(
    char* response, std::size_t size, std::size_t* used, const CombatEntityFact&) {
    JsonWriter json(response, size, used);
    return json.append("{}");
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

#if defined(__aarch64__)
void emit_absolute_jump(std::uint8_t*, const void*) {}
bool make_page_writable(void*, int) {
    return false;
}
#endif

#include "visibility_runtime_telemetry.inc"
#include "phase_runtime_telemetry.inc"
#include "special_movement_runtime_telemetry.inc"

int main() {
    NativePhaseSnapshot snapshot;
    (void)read_native_phase_snapshot(nullptr, 0, 0, 0, &snapshot);
    assert(phase_raw_kind_label(PhaseRawKind::AttackStart)[0] == 'a');

    auto encode_phase = [](std::uint64_t generation, std::uint64_t epoch) {
        char buffer[32768] = {};
        std::size_t used = 0;
        assert(append_phase_runtime_events_json(generation, epoch, 100, buffer, sizeof(buffer), &used));
        return std::string(buffer, used);
    };
    g_phase_runtime_hooks_installed.store(true, std::memory_order_release);
    assert(encode_phase(7, 11).find("\"complete\":false") != std::string::npos);
    begin_phase_runtime_epoch(7, 11);
    const std::string empty = encode_phase(7, 11);
    assert(empty.find("\"complete\":true") != std::string::npos);
    assert(empty.find("\"events\":[]") != std::string::npos);

    const CombatCaptureContext context{nullptr, nullptr, 7, 11, 0};
    const void* const object = reinterpret_cast<const void*>(0x1234);
    assert(update_phase_cache(
        context,
        object,
        [](PhaseRuntimeCache& cache) {
            cache.last_attack_scale_input = 50;
            cache.last_attack_scale_output = 25;
            cache.last_attack_scale_tick = 0;
        }));
    PhaseRuntimeCache copied;
    assert(copy_phase_cache(7, 11, object, &copied));
    assert(copied.last_attack_scale_input == 50);
    assert(copied.last_attack_scale_output == 25);
    assert(copied.last_attack_scale_tick == 0);
    assert(copied.last_effective_speed == kPhaseRuntimeUnset);
    assert(!copied.lock.test_and_set(std::memory_order_acquire));
    copied.lock.clear(std::memory_order_release);
    PhaseRawRecord record =
        phase_base_record(context, PhaseRawKind::AttackStart, 0xf23110, 0);
    assert(publish_phase_record(record) == 1);
    assert(encode_phase(7, 11).find("\"sequence\":1") != std::string::npos);
    g_phase_runtime_overflow_count.store(9, std::memory_order_release);
    g_phase_runtime_rejected_count.store(3, std::memory_order_release);
    begin_phase_runtime_epoch(8, 12);
    const std::string reset = encode_phase(8, 12);
    assert(reset.find("\"complete\":true") != std::string::npos);
    assert(reset.find("\"overflowCount\":0") != std::string::npos);
    assert(reset.find("\"rejectedCount\":0") != std::string::npos);
    assert(reset.find("\"events\":[]") != std::string::npos);
    PhaseRawRecord stale_record =
        phase_base_record(context, PhaseRawKind::AttackRelease, 0xf5f100, 0);
    assert(publish_phase_record(stale_record) == 0);
    assert(encode_phase(8, 12) == reset);
    PhaseRuntimeCache stale_cache;
    assert(!copy_phase_cache(7, 11, object, &stale_cache));
    assert(!copy_phase_cache(8, 12, object, &stale_cache));

    g_special_movement_hook_installed.store(
        true, std::memory_order_release);
    begin_special_movement_runtime_epoch(8, 12);
    SpecialMovementRawRecord movement_record;
    movement_record.generation = 8;
    movement_record.state_epoch = 12;
    movement_record.tick = 3;
    movement_record.hook_offset =
        cr_special_movement_runtime::kExecuteOffset;
    movement_record.caller_offset =
        cr_special_movement_runtime::kExecuteCallerReturnOffset;
    movement_record.source_before.validated = true;
    movement_record.source_before.present = true;
    movement_record.source_before.native_object_id = 10;
    movement_record.source_after =
        movement_record.source_before;
    movement_record.source_after.position_x = 500;
    movement_record.target.validated = true;
    movement_record.target.present = true;
    movement_record.target.native_object_id = 11;
    movement_record.complete_context = true;
    assert(same_special_movement_entity(
        movement_record.source_before,
        movement_record.source_after));
    assert(publish_special_movement_record(movement_record) == 1);
    return 0;
}
