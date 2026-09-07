#include <atomic>
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

bool combat_component_type_is(
    const void*,
    const void*,
    std::int32_t) {
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

bool read_combat_object_vector(void*, void***, std::int32_t*) { return false; }

#include "visibility_runtime_telemetry.inc"

int main() {
    VisibilityMutationContext mutation;
    VisibilityTransitionFact transition;
    (void)visibility_prepare_buff_transition(
        cr_visibility_runtime::kBuffApplyOffset,
        nullptr,
        nullptr,
        &mutation);
    (void)visibility_finish_buff_transition(mutation, &transition);
    visibility_commit_buff_transition(
        false, mutation, 0);
    return 0;
}
