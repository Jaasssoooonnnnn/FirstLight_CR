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



constexpr std::size_t kMaxStoredTowerTroopRuntimes = 6;
struct StoredTowerTroopRuntimeState {
    std::uint8_t kind = 0;
    std::uint32_t source_native_object_id = 0;
    std::int32_t charge_count = 0;
    std::int32_t max_charge_count = 0;
    std::int32_t recharge_elapsed_ms = 0;
    std::int32_t recharge_duration_ms = 0;
    std::int32_t start_delay_remaining_ms = 0;
    std::int32_t start_delay_duration_ms = 0;
    std::int32_t cooking_contribution = 0;
    std::int32_t contribution_needed = 0;
    std::int32_t throw_delay_remaining_ms = -1;
    std::uint32_t target_native_object_id = 0;
};


std::uintptr_t remaining_vtable_offset(const void*) { return 0; }
bool read_combat_object_vector(void*,void***,std::int32_t*) { return false; }
#include "tower_troop_runtime_telemetry.inc"
#include "visibility_runtime_telemetry.inc"

template<class Fn> void emit(Fn fn,unsigned long long generation,unsigned long long epoch,int tick) {
    static char out[2097152]; std::size_t used=0;
    bool ok=fn(generation,epoch,tick,out,sizeof(out),&used);
    std::printf("%d %zu %s\n",ok,used,out);
}
int main() {
    for(int scenario=0;scenario<8;++scenario) {
        g_visibility_runtime_transition_hooks_installed=true;
        visibility_refresh_epoch({nullptr,nullptr,7,static_cast<unsigned>(scenario+1),30});
        VisibilityRuntimeRawRecord visible;
        visible.generation=7;visible.state_epoch=scenario+1;visible.tick=30;
        visible.edge=VisibilityPhaseEdge::BecameInvisible;
        visible.buff_global_id=0xFFFFFFFF;visible.invisible_count_after=1;
        visible.subject.native_object_id=28;visible.hook_offset=0x1FFFF;visible.complete_context=true;
        for(int i=0;i<scenario;++i) publish_visibility_runtime_record(visible);
        emit(append_visibility_runtime_events_json,scenario==5?8:7,scenario+1,scenario==6?29:31);
        begin_tower_troop_runtime_epoch(7,scenario+1);
        for(int i=0;i<10;++i) {
            TowerTroopRuntimeRecord tower;
            tower.generation=7;tower.state_epoch=scenario+1;tower.tick=30;
            tower.kind=i%2?TowerTroopRuntimeKind::RoyalChef:TowerTroopRuntimeKind::DaggerDuchess;
            tower.source_native_object_id=i%3+1; tower.charge_count=i;
            tower.recharge_elapsed_ms=i*11; tower.throw_delay_remaining_ms=i%2?44:-1;
            tower.target_native_object_id=i%4?0xFFFFFFFF:0; tower.cooking_contribution=i*99;
            publish_tower_troop_runtime(tower);
            char out[4096];std::size_t used=0;
            append_tower_troop_runtime_object_json(out,sizeof(out),&used,tower);
            std::puts(out);
        }
        TowerTroopRuntimeRecord records[6]; StoredTowerTroopRuntimeState stored[6];
        auto count=latest_tower_troop_runtimes(7,scenario+1,30,records,6);
        auto stored_count=copy_tower_troop_runtime_snapshot(7,scenario+1,30,stored,6);
        assert(count==stored_count);
        for(std::size_t i=0;i<count;++i) {
            std::printf("%u %d %d %u\n",records[i].source_native_object_id,records[i].charge_count,stored[i].charge_count,stored[i].target_native_object_id);
        }
        emit(append_tower_troop_runtime_envelope_json,7,scenario+1,30);
    }
}
