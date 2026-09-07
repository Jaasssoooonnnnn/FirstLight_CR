bool combat_component_type_is(const void*,const void*,int) { return false; }
#include <cstdio>
#include <cstdarg>
#define PF_X 1
#define PF_R 4
bool libg_address_has_segment_flags(const void*, unsigned long long, int) { return true; }
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


bool append_json(char* out, std::size_t cap, std::size_t* used, const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    int n = std::vsnprintf(out + *used, cap - *used, fmt, args); va_end(args);
    if (n < 0 || static_cast<std::size_t>(n) >= cap - *used) return false;
    *used += n; return true;
}
bool append_json_utf8_string(char* out,std::size_t cap,std::size_t* used,const char* value,std::size_t size) {
    return append_json(out,cap,used,"\"%.*s\"",static_cast<int>(size),value);
}
bool append_combat_entity_fact(char* out,std::size_t cap,std::size_t* used,const CombatEntityFact& value) {
    return append_json(out,cap,used,"{\"id\":%u,\"x\":%d,\"owner\":%d}",value.native_object_id,value.position_x,value.owner);
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
#include "character_state_runtime_telemetry.inc"
#include "special_movement_runtime_telemetry.inc"

template<class Fn> void emit(Fn fn,unsigned long long generation,unsigned long long epoch,int tick) {
    static char out[2097152]; std::size_t used=0;
    bool ok=fn(generation,epoch,tick,out,sizeof(out),&used);
    std::printf("%d %zu %s\n",ok,used,out);
}
int main() {
    ActionMovementRawRecord action;
    action.generation=7; action.state_epoch=11; action.tick=25;
    action.action_data_global_id=123; action.action_name_size=5;
    std::strcpy(action.action_name,"hello");
    action.source_before.card_id=26000083;
    action.source_before.native_object_id=10; action.source_before.owner=1;
    action.source_after=action.source_before; action.source_after.position_x=45;
    CharacterStateRawRecord character;
    character.generation=7; character.state_epoch=11; character.tick=25;
    character.source_before.card_id=cr_character_state_runtime::kBanditCardId;
    character.previous_state=1; character.requested_state=2; character.committed_state=2;
    SpecialMovementRawRecord special;
    special.generation=7; special.state_epoch=11; special.tick=25;
    special.source_before=action.source_before; special.source_after=action.source_after;
    special.requested_x=334; special.cooldown_after_ms=200;
    g_action_movement_hooks_installed=true;
    g_character_state_hook_installed=true;
    g_special_movement_hook_installed=true;
    for (int scenario=0;scenario<6;++scenario) {
        begin_action_movement_runtime_epoch(7,11);
        begin_character_state_runtime_epoch(7,11);
        begin_special_movement_runtime_epoch(7,11);
        int count=scenario==4 ? 2055 : scenario;
        for(int index=0;index<count;++index) {
            action.runtime_class_vtable_offset=index%2?1234:0;
            action.chain_index=index%2?3:-1; action.target_present=index%2;
            action.kind=static_cast<ActionMovementRawKind>(index%3+1);
            publish_action_movement_record(action);
            publish_character_state_record(character);
            publish_special_movement_record(special);
        }
        if(scenario==3) {
            CombatCaptureContext context; context.generation=7; context.state_epoch=11;
            reject_action_movement_record(context); reject_character_state_record(context); reject_special_movement_record(context);
        }
        auto generation=scenario==5?8:7;
        auto tick=scenario==2?24:26;
        emit(append_action_movement_runtime_events_json,generation,11,tick);
        emit(append_character_state_runtime_events_json,generation,11,tick);
        emit(append_special_movement_runtime_events_json,generation,11,tick);
    }
}
