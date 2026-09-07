#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <climits>
#include <string>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>


struct FakeObject {
    alignas(void*) std::uint8_t bytes[0x200] = {};
};

FakeObject* g_readable_objects[8] = {};
std::size_t g_readable_object_count = 0;
std::atomic<std::uint64_t> g_capture_combat_entity_calls{0};
std::atomic<std::uint64_t> g_current_combat_capture_context_calls{0};
std::atomic<std::uint64_t> g_process_range_is_readable_calls{0};
std::atomic<bool> g_test_area_result{true};
std::atomic<bool> g_require_membership_for_capture{false};
void* g_test_members[8] = {};
std::size_t g_test_member_count = 0;

std::uintptr_t g_libg_base = 0;
const char* g_libg_path = "";
std::atomic<bool> g_combat_hooks_attested{false};
std::atomic<bool> g_combat_hooks_installed{false};
std::atomic<bool> g_combat_hooks_enabled{false};
std::atomic<bool> g_combat_epoch_active{false};
std::atomic<std::uint64_t> g_combat_generation{0};
std::atomic<std::uint64_t> g_combat_state_epoch{0};
std::atomic<bool> g_phase_runtime_hooks_attested{false};
std::atomic<bool> g_phase_runtime_hooks_installed{false};

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

bool process_range_is_readable(const void* pointer, std::size_t size) {
    g_process_range_is_readable_calls.fetch_add(
        1, std::memory_order_relaxed);
    if (pointer == nullptr || size > sizeof(FakeObject::bytes)) {
        return false;
    }
    for (std::size_t index = 0;
         index < g_readable_object_count;
         ++index) {
        if (pointer == g_readable_objects[index]) {
            return true;
        }
    }
    return false;
}

bool read_bounded_libg_string_utf8(
    const void*,
    char* output,
    std::size_t output_size,
    std::size_t* byte_length_out) {
    if (output != nullptr && output_size > 0) {
        output[0] = '\0';
    }
    if (byte_length_out != nullptr) {
        *byte_length_out = 0;
    }
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

enum class CombatSpawnProvenanceKind : std::uint8_t {
    None = 0,
    AreaTickNestedSpawn = 1,
    ActionSpawnToLocation = 2,
};

struct CombatSpawnProvenance {
    CombatSpawnProvenanceKind kind = CombatSpawnProvenanceKind::None;
    std::uintptr_t hook_offset = 0;
    std::int32_t source_data_global_id = -1;
};

struct CombatRawRecord {
    CombatEntityFact target;
    CombatEntityFact immediate_source;
    CombatEntityFact source;
    CombatSpawnProvenance spawn_provenance;
};

using CombatRemainingSpawnObserver =
    void (*)(void*, void*, std::int32_t);
CombatRemainingSpawnObserver g_combat_remaining_spawn_observer = nullptr;

using CombatRemainingSpawnProvenanceObserver =
    void (*)(void*, void*, CombatRawRecord*);
CombatRemainingSpawnProvenanceObserver
    g_combat_remaining_spawn_provenance_observer = nullptr;

std::atomic<bool> g_test_context_active{false};
CombatCaptureContext g_test_context;

bool current_combat_capture_context(CombatCaptureContext* result) {
    g_current_combat_capture_context_calls.fetch_add(
        1, std::memory_order_relaxed);
    if (result == nullptr ||
        !g_test_context_active.load(std::memory_order_acquire)) {
        return false;
    }
    *result = g_test_context;
    return true;
}

void capture_combat_entity(
    const void* object,
    void*,
    CombatEntityFact* result) {
    if (result == nullptr) {
        return;
    }
    g_capture_combat_entity_calls.fetch_add(
        1, std::memory_order_relaxed);
    *result = {};
    if (object == nullptr) {
        result->validated = true;
        return;
    }
    if (!process_range_is_readable(object, 0xb0)) {
        return;
    }
    result->validated = true;
    if (g_require_membership_for_capture.load(
            std::memory_order_acquire)) {
        bool member = false;
        for (std::size_t index = 0;
             index < g_test_member_count;
             ++index) {
            member = member || g_test_members[index] == object;
        }
        if (!member) {
            return;
        }
    }
    result->present = true;
    result->native_object_id =
        read_object_field<std::uint32_t>(object, 0x08);
    result->object_kind = 1;
}

bool combat_try_pointer_membership(
    void*,
    const void* object,
    bool* is_member) {
    if (is_member == nullptr) {
        return false;
    }
    *is_member = false;
    for (std::size_t index = 0;
         index < g_test_member_count;
         ++index) {
        if (g_test_members[index] == object) {
            *is_member = true;
            break;
        }
    }
    return true;
}

bool read_combat_object_vector(
    void*,
    void*** objects,
    std::int32_t* count) {
    if (objects == nullptr || count == nullptr) {
        return false;
    }
    *objects = g_test_members;
    *count = static_cast<std::int32_t>(g_test_member_count);
    return true;
}

bool test_area_damage_eligible(
    void*,
    void*,
    std::int32_t) {
    return g_test_area_result.load(std::memory_order_acquire);
}

const char* json_boolean(bool value) {
    return value ? "true" : "false";
}

bool append_json(char* output, std::size_t capacity, std::size_t* used, const char* format, ...) {
    if (!output || !used || *used >= capacity) return false;
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(output + *used, capacity - *used, format, args);
    va_end(args);
    if (n < 0 || static_cast<std::size_t>(n) >= capacity - *used) return false;
    *used += n;
    return true;
}
bool append_json_utf8_string(char* output, std::size_t capacity, std::size_t* used, const char* value, std::size_t length) {
    if (!append_json(output, capacity, used, "\"")) return false;
    for (std::size_t i = 0; i < length; ++i) {
        if (value[i] == '\\' || value[i] == '"') {
            if (!append_json(output, capacity, used, "\\%c", value[i])) return false;
        } else if (!append_json(output, capacity, used, "%c", value[i])) return false;
    }
    return append_json(output, capacity, used, "\"");
}
bool append_combat_entity_fact(char* output, std::size_t capacity, std::size_t* used, const CombatEntityFact& fact) {
    if (!fact.validated || !fact.present) return append_json(output, capacity, used, "null");
    return append_json(output, capacity, used, "{\"nativeObjectId\":%u,\"owner\":%d}", fact.native_object_id, fact.owner);
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

#include "remaining_runtime_telemetry.inc"

template <typename T>
void write_fake_object(
    FakeObject* object,
    std::size_t offset,
    T value) {
    std::memcpy(object->bytes + offset, &value, sizeof(value));
}

bool test_remaining_json_contract() {
    const char* expected[] = {
        "{"
        "\"sequence\":1"
        ",\"tick\":1"
        ",\"kind\":\"area_create\""
        ",\"hookOffset\":0"
        ",\"callerOffset\":null"
        ",\"entity\":null"
        ",\"source\":null"
        ",\"target\":null"
        ",\"targetBefore\":null"
        ",\"targetAfter\":null"
        ",\"objectKind\":null"
        ",\"runtimeVtableOffset\":null"
        ",\"dataBeforeGlobalId\":null"
        ",\"dataAfterGlobalId\":null"
        ",\"expectedCharacterDataGlobalId\":null"
        ",\"expectedProjectileDataGlobalId\":null"
        ",\"configuredDataGlobalId\":null"
        ",\"resourcePreFixed\":null"
        ",\"resourcePostFixed\":null"
        ",\"resourceActualDeltaFixed\":null"
        ",\"amountArgument\":null"
        ",\"configuredAmountArgument\":null"
        ",\"resourceOwner\":null"
        ",\"areaRemainingLifeMs\":null"
        ",\"resourceCause\":\"periodic\""
        ",\"transformKind\":\"character\""
        ",\"result\":true"
        ",\"option\":false"
        ",\"committed\":false"
        ",\"resetTarget\":false"
        ",\"completeContext\":false"
        "}",
        "{"
        "\"sequence\":2"
        ",\"tick\":2"
        ",\"kind\":\"area_expire\""
        ",\"hookOffset\":18446744069720004216"
        ",\"callerOffset\":18446744043950200440"
        ",\"entity\":{\"nativeObjectId\":4294967295,\"owner\":-1}"
        ",\"source\":{\"nativeObjectId\":3,\"owner\":-1}"
        ",\"target\":{\"nativeObjectId\":5,\"owner\":-1}"
        ",\"targetBefore\":{\"nativeObjectId\":7,\"owner\":-1}"
        ",\"targetAfter\":{\"nativeObjectId\":9,\"owner\":-1}"
        ",\"objectKind\":2"
        ",\"runtimeVtableOffset\":18446744065425036920"
        ",\"dataBeforeGlobalId\":4294967295"
        ",\"dataAfterGlobalId\":2147483647"
        ",\"expectedCharacterDataGlobalId\":123"
        ",\"expectedProjectileDataGlobalId\":4294967294"
        ",\"configuredDataGlobalId\":456"
        ",\"resourcePreFixed\":2147483647"
        ",\"resourcePostFixed\":-2147483647"
        ",\"resourceActualDeltaFixed\":-1"
        ",\"amountArgument\":0"
        ",\"configuredAmountArgument\":14"
        ",\"resourceOwner\":1"
        ",\"areaRemainingLifeMs\":-100"
        ",\"resourceCause\":\"death_owner\""
        ",\"transformKind\":\"projectile\""
        ",\"result\":false"
        ",\"option\":true"
        ",\"committed\":false"
        ",\"resetTarget\":false"
        ",\"completeContext\":false"
        "}",
    };
    for (int i = 1; i <= 2; ++i) {
        RemainingRawRecord record;
        record.sequence = i;
        record.state_epoch = 17;
        record.tick = i;
        record.kind = static_cast<RemainingRawKind>(i);
        if (i % 2 == 0) {
            record.hook_offset = UINT64_C(0xffffffff12345678);
            record.caller_offset = UINT64_C(0xfffffff912345678);
            record.runtime_vtable_offset = UINT64_C(0xfffffffe12345678);
            record.object_kind = i;
            record.entity.validated = record.entity.present = true;
            record.entity.native_object_id = UINT32_MAX;
            record.entity.owner = -1;
            record.source = record.entity;
            record.source.native_object_id = 3;
            record.target = record.source;
            record.target.native_object_id = 5;
            record.target_before = record.target;
            record.target_before.native_object_id = 7;
            record.target_after = record.target;
            record.target_after.native_object_id = 9;
            record.data_before_global_id = -1;
            record.data_after_global_id = INT_MAX;
            record.expected_character_data_global_id = 123;
            record.expected_projectile_data_global_id = -2;
            record.configured_data_global_id = 456;
            record.resource_pre_fixed = INT_MAX;
            record.resource_post_fixed = INT_MIN + 1;
            record.resource_actual_delta_fixed = -1;
            record.amount_argument = 0;
            record.configured_amount_argument = 14;
            record.resource_owner = 1;
            record.area_remaining_life_ms = -100;
        }
        record.resource_cause = static_cast<cr_remaining_runtime::ResourceDeltaCause>(i % 4);
        record.transform_kind = static_cast<RemainingTransformKind>(i % 3);
        record.result = i & 1;
        record.option = i & 2;
        record.committed = i & 4;
        record.reset_target = i & 8;
        record.complete_context = i % 3 == 0;
        char output[4096] = {};
        std::size_t used = 0;
        if (!append_remaining_record_json(output, sizeof(output), &used, record) ||
            std::string(output) != expected[i - 1]) {
            return false;
        }
        const auto exact_capacity = used + 1;
        used = 0;
        if (!append_remaining_record_json(output, exact_capacity, &used, record)) {
            return false;
        }
        used = 0;
        if (append_remaining_record_json(output, exact_capacity - 1, &used, record)) {
            return false;
        }
    }
    return true;
}

int main() {
    if (!test_remaining_json_contract()) return 19;
    char response[8] = {};
    std::size_t used = 0;
    if (verify_remaining_runtime_shared_hook_dependencies()) {
        return 1;
    }
    g_combat_hooks_attested.store(true);
    g_combat_hooks_installed.store(true);
    g_phase_runtime_hooks_attested.store(true);
    g_phase_runtime_hooks_installed.store(true);
    if (!verify_remaining_runtime_shared_hook_dependencies()) {
        return 1;
    }

    FakeObject subject;
    FakeObject reused_subject;
    FakeObject source;
    g_readable_objects[0] = &subject;
    g_readable_objects[1] = &reused_subject;
    g_readable_objects[2] = &source;
    g_readable_object_count = 3;
    g_libg_base = UINT64_C(0x100000000);
    const void* const character_vtable =
        reinterpret_cast<const void*>(
            g_libg_base +
            cr_remaining_runtime::kCharacterVtableOffset);
    const void* const area_effect_vtable =
        reinterpret_cast<const void*>(
            g_libg_base +
            cr_remaining_runtime::kAreaEffectObjectVtableOffset);
    write_fake_object(
        &subject, 0x00, character_vtable);
    write_fake_object<std::uint32_t>(
        &subject, 0x08, 101);
    write_fake_object(
        &reused_subject, 0x00, character_vtable);
    write_fake_object<std::uint32_t>(
        &reused_subject, 0x08, 101);
    write_fake_object(
        &source, 0x00, area_effect_vtable);
    write_fake_object<std::uint32_t>(
        &source, 0x08, 201);
    g_test_context.game_manager =
        reinterpret_cast<void*>(UINT64_C(0x200000000));
    g_test_context.object_manager =
        reinterpret_cast<void*>(UINT64_C(0x300000000));
    g_test_context.generation = 1;
    g_test_context.state_epoch = 1;
    g_test_context.tick = 130;
    g_test_context_active.store(true, std::memory_order_release);
    g_combat_generation.store(1, std::memory_order_release);
    g_combat_state_epoch.store(1, std::memory_order_release);
    g_combat_hooks_enabled.store(true, std::memory_order_release);
    g_combat_epoch_active.store(true, std::memory_order_release);
    g_remaining_runtime_hooks_attested.store(
        true, std::memory_order_release);
    g_remaining_runtime_hooks_installed.store(
        true, std::memory_order_release);
    g_original_remaining_area_damage_eligible =
        &test_area_damage_eligible;

    if (!remaining_area_damage_eligible_hook(
            &subject, &source, 0)) {
        return 1;
    }
    const std::uint64_t context_calls_after_first =
        g_current_combat_capture_context_calls.load();
    const std::uint64_t readability_calls_after_first =
        g_process_range_is_readable_calls.load();
    const std::uint64_t captures_after_first =
        g_capture_combat_entity_calls.load();
    for (std::size_t iteration = 0;
         iteration < 10000;
         ++iteration) {
        if (!remaining_area_damage_eligible_hook(
                &subject, &source, 0)) {
            return 1;
        }
    }
    if (g_remaining_runtime_next_sequence.load() != 2 ||
        captures_after_first != 2 ||
        g_capture_combat_entity_calls.load() !=
            captures_after_first ||
        g_current_combat_capture_context_calls.load() !=
            context_calls_after_first ||
        g_process_range_is_readable_calls.load() !=
            readability_calls_after_first) {
        return 1;
    }

    g_test_area_result.store(false, std::memory_order_release);
    if (remaining_area_damage_eligible_hook(
            &subject, &source, 0) ||
        g_remaining_runtime_next_sequence.load() != 3 ||
        g_capture_combat_entity_calls.load() != 4) {
        return 1;
    }
    (void)remaining_area_damage_eligible_hook(
        &subject, &source, 0);
    if (g_remaining_runtime_next_sequence.load() != 3 ||
        g_capture_combat_entity_calls.load() != 4) {
        return 1;
    }

    (void)remaining_area_damage_eligible_hook(
        &subject, &source, 1);
    write_fake_object<std::uint32_t>(
        &source, 0x08, 202);
    (void)remaining_area_damage_eligible_hook(
        &subject, &source, 1);
    (void)remaining_area_damage_eligible_hook(
        &reused_subject, &source, 1);
    g_test_context.generation = 2;
    g_test_context.state_epoch = 2;
    g_combat_generation.store(2, std::memory_order_release);
    g_combat_state_epoch.store(2, std::memory_order_release);
    (void)remaining_area_damage_eligible_hook(
        &reused_subject, &source, 1);
    remaining_area_eligibility_cache_invalidate_pointer(
        &reused_subject);
    (void)remaining_area_damage_eligible_hook(
        &reused_subject, &source, 1);
    (void)remaining_area_damage_eligible_hook(
        &reused_subject, &source, 2);
    if (g_remaining_runtime_next_sequence.load() != 9 ||
        g_capture_combat_entity_calls.load() != 16 ||
        g_remaining_runtime_rejected_count.load() != 0) {
        return 1;
    }

    g_test_context.generation = 3;
    g_test_context.state_epoch = 3;
    g_combat_generation.store(3, std::memory_order_release);
    g_combat_state_epoch.store(3, std::memory_order_release);
    g_test_area_result.store(true, std::memory_order_release);
    std::atomic<bool> concurrent_start{false};
    std::atomic<bool> concurrent_failed{false};
    std::array<std::thread, 8> workers;
    for (std::thread& worker : workers) {
        worker = std::thread([&]() {
            while (!concurrent_start.load(std::memory_order_acquire)) {
            }
            for (std::size_t iteration = 0;
                 iteration < 1000;
                 ++iteration) {
                if (!remaining_area_damage_eligible_hook(
                        &reused_subject, &source, 2)) {
                    concurrent_failed.store(
                        true, std::memory_order_release);
                }
            }
        });
    }
    concurrent_start.store(true, std::memory_order_release);
    for (std::thread& worker : workers) {
        worker.join();
    }
    if (concurrent_failed.load(std::memory_order_acquire) ||
        g_remaining_runtime_next_sequence.load() != 10 ||
        g_capture_combat_entity_calls.load() != 18 ||
        g_remaining_runtime_rejected_count.load() != 0) {
        return 1;
    }

    g_test_context.generation = 4;
    g_test_context.state_epoch = 4;
    g_test_context.tick = 400;
    g_combat_generation.store(4, std::memory_order_release);
    g_combat_state_epoch.store(4, std::memory_order_release);
    remaining_refresh_epoch(g_test_context);
    RemainingRemovalCapture delayed;
    delayed.validated = true;
    delayed.object_pointer = &subject;
    delayed.context = g_test_context;
    delayed.entity.validated = true;
    delayed.entity.present = true;
    delayed.entity.native_object_id = 101;
    delayed.entity.object_kind = 1;
    g_test_members[0] = &subject;
    g_test_member_count = 1;
    remaining_runtime_note_confirmed_remove(delayed, false);
    remaining_runtime_reconcile_pending_removals(g_test_context);
    if (g_remaining_runtime_next_sequence.load() != 10 ||
        g_remaining_runtime_rejected_count.load() != 0) {
        return 1;
    }
    g_test_context.tick = 401;
    g_test_member_count = 0;
    remaining_runtime_reconcile_pending_removals(g_test_context);
    if (g_remaining_runtime_next_sequence.load() != 11 ||
        g_remaining_runtime_rejected_count.load() != 0) {
        return 1;
    }

    delayed.object_pointer = &reused_subject;
    delayed.context = g_test_context;
    delayed.entity.native_object_id = 101;
    g_test_members[0] = &reused_subject;
    g_test_member_count = 1;
    remaining_runtime_note_confirmed_remove(delayed, false);
    write_fake_object<std::uint32_t>(
        &reused_subject, 0x08, 102);
    g_test_context.tick = 402;
    remaining_runtime_reconcile_pending_removals(g_test_context);
    if (g_remaining_runtime_next_sequence.load() != 12 ||
        g_remaining_runtime_rejected_count.load() != 0) {
        return 1;
    }

    FakeObject child_data;
    g_readable_objects[3] = &child_data;
    g_readable_object_count = 4;
    constexpr std::uint32_t kRuntimeUpdateGlobalId = 3'587'309'677U;
    write_fake_object<std::uint32_t>(
        &child_data,
        cr_remaining_runtime::kLogicDataGlobalIdOffset,
        kRuntimeUpdateGlobalId);
    std::int32_t runtime_update_global_id =
        kRemainingRuntimeGlobalIdUnset;
    if (!remaining_data_global_id(
            &child_data, &runtime_update_global_id) ||
        static_cast<std::uint32_t>(runtime_update_global_id) !=
            kRuntimeUpdateGlobalId) {
        return 1;
    }
    write_fake_object<const void*>(
        &subject,
        cr_remaining_runtime::kObjectDataOffset,
        &child_data);
    write_fake_object<std::uint32_t>(
        &child_data,
        cr_remaining_runtime::kLogicDataGlobalIdOffset,
        34'000'051);
    g_test_context.generation = 5;
    g_test_context.state_epoch = 5;
    g_test_context.tick = 500;
    g_combat_generation.store(5, std::memory_order_release);
    g_combat_state_epoch.store(5, std::memory_order_release);
    remaining_refresh_epoch(g_test_context);
    g_test_members[0] = &source;
    g_test_members[1] = &subject;
    g_test_member_count = 2;
    RemainingPendingSpawnAttach& delayed_attach =
        g_remaining_pending_spawn_attaches[0];
    delayed_attach.active = true;
    delayed_attach.context = g_test_context;
    delayed_attach.source_pointer = &source;
    delayed_attach.source_native_object_id = 202;
    delayed_attach.child_pointer = &subject;
    delayed_attach.child_native_object_id = 101;
    delayed_attach.child_data_global_id = 34'000'051;
    remaining_runtime_reconcile_pending_spawn_attaches(
        g_test_context);
    const RemainingRawRecord attached =
        g_remaining_runtime_ring[
            (12 - 1) % kRemainingRuntimeRingCapacity].record;
    if (g_remaining_runtime_next_sequence.load() != 13 ||
        g_remaining_runtime_rejected_count.load() != 0 ||
        attached.kind != RemainingRawKind::SpawnAttach ||
        attached.source.native_object_id != 202 ||
        attached.target.native_object_id != 101 ||
        attached.configured_data_global_id != 34'000'051 ||
        !attached.committed || !attached.complete_context) {
        return 1;
    }

    g_test_context.generation = 6;
    g_test_context.state_epoch = 6;
    g_test_context.tick = 600;
    g_combat_generation.store(6, std::memory_order_release);
    g_combat_state_epoch.store(6, std::memory_order_release);
    remaining_refresh_epoch(g_test_context);
    RemainingRawRecord tracked_spawn = remaining_base_record(
        g_test_context,
        RemainingRawKind::LifetimeSpawn,
        cr_remaining_runtime::kCombatSpawnOffset);
    tracked_spawn.entity.validated = true;
    tracked_spawn.entity.present = true;
    tracked_spawn.entity.native_object_id = 101;
    tracked_spawn.entity.object_kind = 1;
    tracked_spawn.object_kind = 1;
    tracked_spawn.runtime_vtable_offset =
        cr_remaining_runtime::kCharacterVtableOffset;
    tracked_spawn.data_after_global_id = 34'000'051;
    g_test_members[0] = &subject;
    g_test_member_count = 1;
    remaining_track_committed_lifetime(
        g_test_context,
        &subject,
        tracked_spawn,
        false,
        kRemainingRuntimeUnset);
    remaining_runtime_reconcile_tracked_lifetimes(
        g_test_context);
    if (g_remaining_runtime_next_sequence.load() != 13) {
        return 1;
    }
    g_test_member_count = 0;
    g_test_context.tick = 601;
    remaining_runtime_reconcile_tracked_lifetimes(
        g_test_context);
    const RemainingRawRecord tracked_despawn =
        g_remaining_runtime_ring[
            (13 - 1) % kRemainingRuntimeRingCapacity].record;
    if (g_remaining_runtime_next_sequence.load() != 14 ||
        tracked_despawn.kind != RemainingRawKind::LifetimeDespawn ||
        tracked_despawn.entity.native_object_id != 101 ||
        tracked_despawn.tick != 601) {
        return 1;
    }

    g_test_context.generation = 7;
    g_test_context.state_epoch = 7;
    g_test_context.tick = 700;
    g_combat_generation.store(7, std::memory_order_release);
    g_combat_state_epoch.store(7, std::memory_order_release);
    remaining_refresh_epoch(g_test_context);
    g_test_members[0] = &source;
    g_test_member_count = 1;
    g_require_membership_for_capture.store(
        true, std::memory_order_release);
    (void)remaining_area_damage_eligible_hook(
        &subject, &source, 99);
    g_require_membership_for_capture.store(
        false, std::memory_order_release);
    if (g_remaining_runtime_next_sequence.load() != 14 ||
        g_remaining_runtime_rejected_count.load() != 0) {
        return 1;
    }

    // A batch may confirm at most 256 removals, then resume the same epoch.
    g_test_context.generation = 8;
    g_test_context.state_epoch = 8;
    g_test_context.tick = 800;
    g_combat_generation.store(8);
    g_combat_state_epoch.store(8);
    remaining_refresh_epoch(g_test_context);
    g_test_member_count = 0;
    for (std::size_t i = 0; i < 300; ++i) {
        auto& entry = g_remaining_tracked_lifetimes[i];
        entry.active = true;
        entry.capture.validated = true;
        entry.capture.context = g_test_context;
        entry.capture.object_pointer = reinterpret_cast<const void*>(1000 + i);
        entry.capture.entity.native_object_id = 1000 + i;
    }
    auto before = g_remaining_runtime_next_sequence.load();
    remaining_runtime_reconcile_tracked_lifetimes(g_test_context);
    if (g_remaining_runtime_next_sequence.load() != before + 256) return 11;
    for (std::size_t i = 0; i < 300; ++i) {
        if (g_remaining_tracked_lifetimes[i].active != (i >= 256)) return 12;
    }
    remaining_runtime_reconcile_tracked_lifetimes(g_test_context);
    if (g_remaining_runtime_next_sequence.load() != before + 300) return 13;
    for (std::size_t i = 0; i < 300; ++i) {
        const auto& record = g_remaining_runtime_ring[
            (before + i - 1) % kRemainingRuntimeRingCapacity].record;
        if (record.kind != RemainingRawKind::LifetimeDespawn ||
            record.entity.native_object_id != 1000 + i || record.tick != 800) return 14;
    }
    // Full pending tables reject only new identities, retaining duplicate proof.
    RemainingRemovalCapture capture;
    capture.validated = true;
    capture.context = g_test_context;
    for (std::size_t i = 0; i < kRemainingPendingRemovalCapacity; ++i) {
        capture.object_pointer = reinterpret_cast<const void*>(2000 + i);
        capture.entity.native_object_id = 2000 + i;
        remaining_runtime_note_confirmed_remove(capture, false);
    }
    auto rejected = g_remaining_runtime_rejected_count.load();
    remaining_runtime_note_confirmed_remove(capture, false);
    if (g_remaining_runtime_rejected_count.load() != rejected) return 15;
    capture.entity.native_object_id = 9999;
    remaining_runtime_note_confirmed_remove(capture, false);
    if (g_remaining_runtime_rejected_count.load() != rejected + 1) return 16;
    auto next = g_remaining_runtime_next_sequence.load();
    remaining_runtime_reconcile_pending_removals(g_test_context);
    if (g_remaining_runtime_next_sequence.load() != next + 256) return 17;
    // A new epoch drops old entries; it never emits stale removal evidence.
    g_remaining_tracked_lifetimes[0].active = true;
    g_remaining_tracked_lifetimes[0].capture = capture;
    g_test_context.state_epoch = 9;
    next = g_remaining_runtime_next_sequence.load();
    remaining_runtime_reconcile_tracked_lifetimes(g_test_context);
    if (g_remaining_tracked_lifetimes[0].active ||
        g_remaining_runtime_next_sequence.load() != next) return 18;
    mark_remaining_runtime_shared_hooks_bound(
        false, false, false, false);
    (void)append_remaining_runtime_events_json(
        0, 0, 0, response, sizeof(response), &used);
    return remaining_raw_kind_label(
               RemainingRawKind::AreaCreate)[0] == 'a' &&
            kRemainingRuntimeRingCapacity == 4096
        ? 0
        : 1;
}
