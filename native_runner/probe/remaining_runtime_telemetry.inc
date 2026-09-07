// Exact-build producers for runtime mechanics which are not safely recoverable
// from ordinary entity snapshots.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc;
//   * remaining_runtime_layout.h is the sole source of exact-build offsets;
//   * cr_replay_probe.cpp supplies read_object_field, process readability,
//     capture_combat_entity, append_json, install_inline_hook, attestation,
//     executable-memory helpers and logging;
//   * link remaining_runtime_arm64.S in the Android probe build;
//   * call install_remaining_runtime_hooks(libg_base) after combat hooks;
//   * bind the shared spawn/remove, tower-target and action-dispatch callbacks
//     with mark_remaining_runtime_shared_hooks_bound().
//
// Every public ring record is pointer-free. Raw addresses exist only inside a
// bounded thread-local causal context and are never serialized.

#include "remaining_runtime_layout.h"
#include "probe_json.h"

constexpr std::size_t kRemainingRuntimeRingCapacity = 4096;
constexpr std::size_t kRemainingPendingSpawnAttachCapacity = 128;
constexpr std::size_t kRemainingRemoveDepth = 64;
constexpr std::size_t kRemainingPendingRemovalCapacity = 256;
constexpr std::size_t kRemainingLifetimeEntryCapacity = 4096;
constexpr std::size_t kRemainingActionWaitDepth = 16;
constexpr std::size_t kRemainingAreaTickSpawnDepth = 16;
constexpr std::size_t kRemainingActionSpawnDepth = 16;
constexpr std::size_t kRemainingPendingActionSpawnCapacity = 512;
constexpr std::size_t kRemainingAreaEligibilityCacheCapacity = 1024;
constexpr std::int32_t kRemainingRuntimeUnset = (-2147483647 - 1);
constexpr std::int32_t kRemainingRuntimeGlobalIdUnset = 0;
static_assert((kRemainingAreaEligibilityCacheCapacity & (kRemainingAreaEligibilityCacheCapacity - 1)) == 0);

enum class RemainingRawKind : std::uint8_t {
  AreaCreate = 1,
  AreaExpire = 2,
  ResourceDelta = 3,
  TowerAggroAcquire = 4,
  TowerAggroChange = 5,
  TowerAggroLose = 6,
  TowerActivate = 7,
  LifetimeSpawn = 8,
  LifetimeDespawn = 9,
  Transform = 10,
  SpawnAttach = 11,
  AreaDamageEligibility = 12,
  AreaRelationParent = 13,
  AreaRelationFollow = 14,
  AreaRelationRelated = 15,
  AreaActionSpawn = 16,
};

enum class RemainingTransformKind : std::uint8_t {
  None = 0,
  Character = 1,
  Projectile = 2,
};

struct RemainingRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  RemainingRawKind kind = RemainingRawKind::LifetimeSpawn;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  CombatEntityFact entity;
  CombatEntityFact source;
  CombatEntityFact target;
  CombatEntityFact target_before;
  CombatEntityFact target_after;
  std::int32_t object_kind = kRemainingRuntimeUnset;
  std::uintptr_t runtime_vtable_offset = 0;
  std::int32_t data_before_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t data_after_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t expected_character_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t expected_projectile_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t configured_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t resource_pre_fixed = kRemainingRuntimeUnset;
  std::int32_t resource_post_fixed = kRemainingRuntimeUnset;
  std::int32_t resource_actual_delta_fixed = kRemainingRuntimeUnset;
  std::int32_t amount_argument = kRemainingRuntimeUnset;
  std::int32_t configured_amount_argument = kRemainingRuntimeUnset;
  std::int32_t resource_owner = kRemainingRuntimeUnset;
  std::int32_t area_remaining_life_ms = kRemainingRuntimeUnset;
  cr_remaining_runtime::ResourceDeltaCause resource_cause = cr_remaining_runtime::ResourceDeltaCause::None;
  RemainingTransformKind transform_kind = RemainingTransformKind::None;
  bool result = false;
  bool option = false;
  bool committed = false;
  bool reset_target = false;
  bool complete_context = false;
};

struct RemainingRingSlot {
  std::atomic<std::uint64_t> published_sequence{0};
  RemainingRawRecord record;
};

// Capture this before the shared remove hook calls native code. The raw
// pointer is retained only in the bounded private pending-removal table and
// is never dereferenced unless the live object vector still contains it.
// It is never serialized.
struct RemainingRemovalCapture {
  bool validated = false;
  bool exact_area_effect = false;
  const void *object_pointer = nullptr;
  CombatCaptureContext context;
  CombatEntityFact entity;
  std::uintptr_t runtime_vtable_offset = 0;
  std::int32_t data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t area_remaining_life_ms = kRemainingRuntimeUnset;
};

struct RemainingLifetimeEntry {
  bool active = false;
  RemainingRemovalCapture capture;
};

struct RemainingPendingSpawnAttach {
  bool active = false;
  CombatCaptureContext context;
  const void *source_pointer = nullptr;
  std::uint32_t source_native_object_id = 0;
  const void *child_pointer = nullptr;
  std::uint32_t child_native_object_id = 0;
  std::int32_t child_data_global_id = kRemainingRuntimeGlobalIdUnset;
};

struct RemainingAreaEligibilityCacheEntry {
  bool valid = false;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::uint32_t subject_native_object_id = 0;
  std::uint32_t source_native_object_id = 0;
  std::int32_t option = 0;
  bool result = false;
  std::uintptr_t subject_pointer = 0;
  std::uintptr_t source_pointer = 0;
};

// Private, bounded dynamic context for an exact LogicAreaEffectObject tick.
// It is not a temporal correlation: a child is admitted only when its
// committed generic spawn occurs synchronously inside this exact native call.
struct RemainingAreaTickSpawnContext {
  bool active = false;
  CombatCaptureContext capture;
  const void *area_pointer = nullptr;
  std::uint32_t area_native_object_id = 0;
  std::int32_t area_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::uintptr_t area_vtable_offset = 0;
  CombatEntityFact area;
};

struct RemainingActionSpawnContext {
  bool active = false;
  CombatCaptureContext capture;
  const void *source_pointer = nullptr;
  std::uint32_t source_native_object_id = 0;
  CombatEntityFact source;
  std::uintptr_t source_vtable_offset = 0;
  std::int32_t source_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t action_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t expected_child_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t action_mode = -1;
};

struct RemainingPendingActionSpawn {
  bool active = false;
  CombatCaptureContext capture;
  const void *source_pointer = nullptr;
  std::uint32_t source_native_object_id = 0;
  CombatEntityFact source;
  std::uintptr_t source_vtable_offset = 0;
  std::int32_t source_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t action_data_global_id = kRemainingRuntimeGlobalIdUnset;
  const void *child_pointer = nullptr;
};

RemainingRingSlot g_remaining_runtime_ring[kRemainingRuntimeRingCapacity];
std::atomic<std::uint64_t> g_remaining_runtime_next_sequence{1};
std::atomic<std::uint64_t> g_remaining_runtime_epoch_first_sequence{1};
std::atomic<std::uint64_t> g_remaining_runtime_generation{0};
std::atomic<std::uint64_t> g_remaining_runtime_state_epoch{0};
std::atomic<std::uint64_t> g_remaining_runtime_overflow_count{0};
std::atomic<std::uint64_t> g_remaining_runtime_rejected_count{0};
std::atomic<bool> g_remaining_runtime_hooks_attested{false};
std::atomic<bool> g_remaining_runtime_independent_hooks_installed{false};
std::atomic<bool> g_remaining_runtime_area_lifecycle_bound{false};
std::atomic<bool> g_remaining_runtime_tower_aggro_bound{false};
std::atomic<bool> g_remaining_runtime_tower_activation_bound{false};
std::atomic<bool> g_remaining_runtime_lifetime_bound{false};
std::atomic<bool> g_remaining_runtime_hooks_installed{false};
std::atomic_flag g_remaining_runtime_epoch_lock = ATOMIC_FLAG_INIT;
RemainingLifetimeEntry g_remaining_pending_removals[kRemainingPendingRemovalCapacity];
std::atomic_flag g_remaining_pending_removal_lock = ATOMIC_FLAG_INIT;
RemainingLifetimeEntry g_remaining_tracked_lifetimes[kRemainingLifetimeEntryCapacity];
std::atomic_flag g_remaining_tracked_lifetime_lock = ATOMIC_FLAG_INIT;
RemainingPendingSpawnAttach g_remaining_pending_spawn_attaches[kRemainingPendingSpawnAttachCapacity];
std::atomic_flag g_remaining_pending_spawn_attach_lock = ATOMIC_FLAG_INIT;
RemainingPendingActionSpawn g_remaining_pending_action_spawns[kRemainingPendingActionSpawnCapacity];
std::atomic_flag g_remaining_pending_action_spawn_lock = ATOMIC_FLAG_INIT;
RemainingAreaEligibilityCacheEntry g_remaining_area_eligibility_cache[kRemainingAreaEligibilityCacheCapacity];
std::uint64_t g_remaining_area_eligibility_cache_generation = 0;
std::uint64_t g_remaining_area_eligibility_cache_state_epoch = 0;
std::size_t g_remaining_area_eligibility_cache_eviction_cursor = 0;
std::atomic_flag g_remaining_area_eligibility_cache_lock = ATOMIC_FLAG_INIT;

thread_local RemainingRemovalCapture g_remaining_remove_stack[kRemainingRemoveDepth];
thread_local std::size_t g_remaining_remove_depth = 0;
thread_local const void *g_remaining_action_wait_stack[kRemainingActionWaitDepth];
thread_local std::size_t g_remaining_action_wait_depth = 0;
thread_local RemainingAreaTickSpawnContext g_remaining_area_tick_spawn_stack[kRemainingAreaTickSpawnDepth];
thread_local std::size_t g_remaining_area_tick_spawn_depth = 0;
thread_local RemainingActionSpawnContext g_remaining_action_spawn_stack[kRemainingActionSpawnDepth];
thread_local std::size_t g_remaining_action_spawn_depth = 0;

using RemainingResourceDelta = void (*)(void *, std::int32_t, std::int32_t);
using RemainingTransformExecute = void (*)(void *, void *, void *, void *);
using RemainingAreaDamageEligible = bool (*)(void *, void *, std::int32_t);
using RemainingActionWaitUpdate = void (*)(void *);
using RemainingAreaEffectTick = void (*)(void *);
using RemainingActionSpawnToLocationExecute = void (*)(void *, void *, void *, void *);
using RemainingPendingObjectAdd = void (*)(void *, void *);

extern "C" {
RemainingResourceDelta g_original_remaining_resource_whole = nullptr;
RemainingResourceDelta g_original_remaining_resource_fixed = nullptr;
void *g_original_remaining_spawn_helper = nullptr;
void *g_original_remaining_action_spawn_mode1_child_create = nullptr;
}
RemainingTransformExecute g_original_remaining_transform_execute = nullptr;
RemainingAreaDamageEligible g_original_remaining_area_damage_eligible = nullptr;
RemainingActionWaitUpdate g_original_remaining_action_wait_update = nullptr;
RemainingAreaEffectTick g_original_remaining_area_effect_tick = nullptr;
RemainingActionSpawnToLocationExecute g_original_remaining_action_spawn_to_location_execute = nullptr;
RemainingPendingObjectAdd g_original_remaining_pending_object_add = nullptr;

#if defined(__aarch64__)
extern "C" void remaining_resource_whole_entry();
extern "C" void remaining_resource_fixed_entry();
extern "C" void remaining_spawn_helper_entry();
extern "C" void remaining_action_spawn_mode1_child_create_entry();
#endif

const char *remaining_raw_kind_label(RemainingRawKind kind) {
  switch (kind) {
  case RemainingRawKind::AreaCreate:
    return "area_create";
  case RemainingRawKind::AreaExpire:
    return "area_expire";
  case RemainingRawKind::ResourceDelta:
    return "resource_delta";
  case RemainingRawKind::TowerAggroAcquire:
    return "tower_aggro_acquire";
  case RemainingRawKind::TowerAggroChange:
    return "tower_aggro_change";
  case RemainingRawKind::TowerAggroLose:
    return "tower_aggro_lose";
  case RemainingRawKind::TowerActivate:
    return "tower_activate";
  case RemainingRawKind::LifetimeSpawn:
    return "lifetime_spawn";
  case RemainingRawKind::LifetimeDespawn:
    return "lifetime_despawn";
  case RemainingRawKind::Transform:
    return "transform";
  case RemainingRawKind::SpawnAttach:
    return "spawn_attach";
  case RemainingRawKind::AreaDamageEligibility:
    return "area_damage_eligibility";
  case RemainingRawKind::AreaRelationParent:
    return "area_relation_parent";
  case RemainingRawKind::AreaRelationFollow:
    return "area_relation_follow";
  case RemainingRawKind::AreaRelationRelated:
    return "area_relation_related";
  case RemainingRawKind::AreaActionSpawn:
    return "area_action_spawn";
  }
  return "unknown";
}

const char *remaining_resource_cause_label(cr_remaining_runtime::ResourceDeltaCause cause) {
  switch (cause) {
  case cr_remaining_runtime::ResourceDeltaCause::Periodic:
    return "periodic";
  case cr_remaining_runtime::ResourceDeltaCause::DeathOwner:
    return "death_owner";
  case cr_remaining_runtime::ResourceDeltaCause::DeathOpponent:
    return "death_opponent";
  case cr_remaining_runtime::ResourceDeltaCause::None:
    break;
  }
  return "none";
}

const char *remaining_transform_kind_label(RemainingTransformKind kind) {
  switch (kind) {
  case RemainingTransformKind::Character:
    return "character";
  case RemainingTransformKind::Projectile:
    return "projectile";
  case RemainingTransformKind::None:
    break;
  }
  return "none";
}

std::uintptr_t remaining_caller_offset(const void *address) {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
  return address == nullptr || raw < g_libg_base ? 0 : raw - g_libg_base;
}

std::uintptr_t remaining_vtable_offset(const void *object) {
  if (object == nullptr || !process_range_is_readable(object, sizeof(void *))) {
    return 0;
  }
  const auto raw = reinterpret_cast<std::uintptr_t>(read_object_field<const void *>(object, 0));
  return raw < g_libg_base ? 0 : raw - g_libg_base;
}

void remaining_lock(std::atomic_flag &lock) {
  while (lock.test_and_set(std::memory_order_acquire)) {
  }
}

void remaining_unlock(std::atomic_flag &lock) { lock.clear(std::memory_order_release); }

bool remaining_area_eligibility_fast_epoch(CombatCaptureContext *result) {
  if (result == nullptr) {
    return false;
  }
  *result = {};
  if (!g_remaining_runtime_hooks_attested.load(std::memory_order_acquire) ||
      !g_remaining_runtime_hooks_installed.load(std::memory_order_acquire) ||
      !g_combat_hooks_enabled.load(std::memory_order_acquire) ||
      !g_combat_epoch_active.load(std::memory_order_acquire)) {
    return false;
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    const std::uint64_t state_epoch_before = g_combat_state_epoch.load(std::memory_order_acquire);
    const std::uint64_t generation = g_combat_generation.load(std::memory_order_acquire);
    const std::uint64_t state_epoch_after = g_combat_state_epoch.load(std::memory_order_acquire);
    if (state_epoch_before == 0 || state_epoch_before != state_epoch_after || generation == 0 ||
        !g_combat_epoch_active.load(std::memory_order_acquire) ||
        !g_combat_hooks_enabled.load(std::memory_order_acquire) ||
        !g_remaining_runtime_hooks_attested.load(std::memory_order_acquire) ||
        !g_remaining_runtime_hooks_installed.load(std::memory_order_acquire)) {
      continue;
    }
    result->generation = generation;
    result->state_epoch = state_epoch_before;
    return true;
  }
  return false;
}

bool remaining_area_eligibility_fast_epoch_matches(const CombatCaptureContext &expected) {
  CombatCaptureContext current;
  return remaining_area_eligibility_fast_epoch(&current) && current.generation == expected.generation &&
         current.state_epoch == expected.state_epoch;
}

std::uint32_t remaining_attested_native_object_id(const void *object) {
  return object == nullptr ? 0 : read_object_field<std::uint32_t>(object, cr_remaining_runtime::kObjectNativeIdOffset);
}

bool remaining_fast_native_object_id(const void *object, bool allow_null, std::uint32_t *native_object_id) {
  if (native_object_id == nullptr) {
    return false;
  }
  *native_object_id = 0;
  if (object == nullptr) {
    return allow_null;
  }
  if (!process_range_is_readable(object, cr_remaining_runtime::kObjectNativeIdOffset + sizeof(std::uint32_t))) {
    return false;
  }
  const std::uint32_t value = read_object_field<std::uint32_t>(object, cr_remaining_runtime::kObjectNativeIdOffset);
  if (value == 0) {
    return false;
  }
  *native_object_id = value;
  return true;
}

bool remaining_capture_context_still_current(const CombatCaptureContext &expected) {
  CombatCaptureContext current;
  return current_combat_capture_context(&current) && current.game_manager == expected.game_manager &&
         current.object_manager == expected.object_manager && current.generation == expected.generation &&
         current.state_epoch == expected.state_epoch && current.tick == expected.tick;
}

void remaining_area_eligibility_cache_refresh_locked(const CombatCaptureContext &context) {
  if (g_remaining_area_eligibility_cache_generation == context.generation &&
      g_remaining_area_eligibility_cache_state_epoch == context.state_epoch) {
    return;
  }
  for (RemainingAreaEligibilityCacheEntry &entry : g_remaining_area_eligibility_cache) {
    entry = {};
  }
  g_remaining_area_eligibility_cache_generation = context.generation;
  g_remaining_area_eligibility_cache_state_epoch = context.state_epoch;
  g_remaining_area_eligibility_cache_eviction_cursor = 0;
}

std::size_t remaining_area_eligibility_cache_hash(std::uintptr_t subject_pointer, std::uintptr_t source_pointer,
                                                  std::int32_t option) {
  std::uint64_t value = static_cast<std::uint64_t>(subject_pointer) * UINT64_C(0x9e3779b185ebca87);
  value ^= static_cast<std::uint64_t>(source_pointer) * UINT64_C(0xc2b2ae3d27d4eb4f);
  value ^= static_cast<std::uint32_t>(option);
  value ^= value >> 33;
  return static_cast<std::size_t>(value) & (kRemainingAreaEligibilityCacheCapacity - 1);
}

bool remaining_area_eligibility_cache_key_matches(const RemainingAreaEligibilityCacheEntry &entry,
                                                  const CombatCaptureContext &context, std::uintptr_t subject_pointer,
                                                  std::uintptr_t source_pointer, std::int32_t option) {
  return entry.valid && entry.generation == context.generation && entry.state_epoch == context.state_epoch &&
         entry.subject_pointer == subject_pointer && entry.source_pointer == source_pointer && entry.option == option;
}

RemainingAreaEligibilityCacheEntry *remaining_area_eligibility_cache_find_locked(const CombatCaptureContext &context,
                                                                                 std::uintptr_t subject_pointer,
                                                                                 std::uintptr_t source_pointer,
                                                                                 std::int32_t option) {
  const std::size_t start = remaining_area_eligibility_cache_hash(subject_pointer, source_pointer, option);
  for (std::size_t step = 0; step < kRemainingAreaEligibilityCacheCapacity; ++step) {
    RemainingAreaEligibilityCacheEntry &entry =
        g_remaining_area_eligibility_cache[(start + step) & (kRemainingAreaEligibilityCacheCapacity - 1)];
    if (remaining_area_eligibility_cache_key_matches(entry, context, subject_pointer, source_pointer, option)) {
      return &entry;
    }
  }
  return nullptr;
}

RemainingAreaEligibilityCacheEntry *remaining_area_eligibility_cache_store_slot_locked(std::uintptr_t subject_pointer,
                                                                                       std::uintptr_t source_pointer,
                                                                                       std::int32_t option) {
  const std::size_t start = remaining_area_eligibility_cache_hash(subject_pointer, source_pointer, option);
  for (std::size_t step = 0; step < kRemainingAreaEligibilityCacheCapacity; ++step) {
    RemainingAreaEligibilityCacheEntry &entry =
        g_remaining_area_eligibility_cache[(start + step) & (kRemainingAreaEligibilityCacheCapacity - 1)];
    if (!entry.valid) {
      return &entry;
    }
  }
  RemainingAreaEligibilityCacheEntry &evicted =
      g_remaining_area_eligibility_cache[g_remaining_area_eligibility_cache_eviction_cursor %
                                         kRemainingAreaEligibilityCacheCapacity];
  g_remaining_area_eligibility_cache_eviction_cursor =
      (g_remaining_area_eligibility_cache_eviction_cursor + 1) % kRemainingAreaEligibilityCacheCapacity;
  return &evicted;
}

void remaining_area_eligibility_cache_invalidate_pointer(const void *object) {
  const std::uintptr_t pointer = reinterpret_cast<std::uintptr_t>(object);
  if (pointer == 0) {
    return;
  }
  remaining_lock(g_remaining_area_eligibility_cache_lock);
  for (RemainingAreaEligibilityCacheEntry &entry : g_remaining_area_eligibility_cache) {
    if (entry.valid && (entry.subject_pointer == pointer || entry.source_pointer == pointer)) {
      entry = {};
    }
  }
  remaining_unlock(g_remaining_area_eligibility_cache_lock);
}

void remaining_area_eligibility_cache_invalidate(const CombatCaptureContext &context, std::uint32_t native_object_id) {
  if (native_object_id == 0) {
    return;
  }
  remaining_lock(g_remaining_area_eligibility_cache_lock);
  if (g_remaining_area_eligibility_cache_generation == context.generation &&
      g_remaining_area_eligibility_cache_state_epoch == context.state_epoch) {
    for (RemainingAreaEligibilityCacheEntry &entry : g_remaining_area_eligibility_cache) {
      if (entry.valid &&
          (entry.subject_native_object_id == native_object_id || entry.source_native_object_id == native_object_id)) {
        entry = {};
      }
    }
  }
  remaining_unlock(g_remaining_area_eligibility_cache_lock);
}

const void *remaining_object_data(const void *object) {
  return process_range_is_readable(object, cr_remaining_runtime::kObjectDataOffset + sizeof(void *))
             ? read_object_field<const void *>(object, cr_remaining_runtime::kObjectDataOffset)
             : nullptr;
}

bool remaining_data_global_id(const void *data, std::int32_t *result) {
  if (result == nullptr || data == nullptr ||
      !process_range_is_readable(data, cr_remaining_runtime::kLogicDataGlobalIdOffset + sizeof(std::uint32_t))) {
    return false;
  }
  const std::uint32_t value = read_object_field<std::uint32_t>(data, cr_remaining_runtime::kLogicDataGlobalIdOffset);
  if (value == 0) {
    return false;
  }
  *result = static_cast<std::int32_t>(value);
  return true;
}

void remaining_reject_impl(const CombatCaptureContext *context, std::int32_t source_line);
#define remaining_reject(context) remaining_reject_impl((context), __LINE__)

bool remaining_exact_area_effect(const void *object, const CombatEntityFact &entity, std::uintptr_t *vtable_offset) {
  const std::uintptr_t actual = remaining_vtable_offset(object);
  if (vtable_offset != nullptr) {
    *vtable_offset = actual;
  }
  return entity.validated && entity.present &&
         cr_remaining_runtime::exact_area_effect_object(entity.object_kind, actual);
}

bool remaining_area_tick_spawn_context_push(void *area) {
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context) || context.object_manager == nullptr) {
    return false;
  }
  CombatEntityFact area_fact;
  capture_combat_entity(area, context.object_manager, &area_fact);
  std::uintptr_t area_vtable_offset = 0;
  if (!area_fact.validated || !area_fact.present ||
      !remaining_exact_area_effect(area, area_fact, &area_vtable_offset)) {
    return false;
  }
  const void *const area_data = remaining_object_data(area);
  std::int32_t area_data_global_id = kRemainingRuntimeGlobalIdUnset;
  if (!remaining_data_global_id(area_data, &area_data_global_id)) {
    return false;
  }
  if (g_remaining_area_tick_spawn_depth >= kRemainingAreaTickSpawnDepth) {
    remaining_reject(&context);
    return false;
  }
  RemainingAreaTickSpawnContext &pending = g_remaining_area_tick_spawn_stack[g_remaining_area_tick_spawn_depth++];
  pending.active = true;
  pending.capture = context;
  pending.area_pointer = area;
  pending.area_native_object_id = area_fact.native_object_id;
  pending.area_data_global_id = area_data_global_id;
  pending.area_vtable_offset = area_vtable_offset;
  pending.area = area_fact;
  return true;
}

void remaining_area_effect_tick_hook(void *area) {
  const bool pushed = remaining_area_tick_spawn_context_push(area);
  g_original_remaining_area_effect_tick(area);
  if (!pushed) {
    return;
  }
  --g_remaining_area_tick_spawn_depth;
  g_remaining_area_tick_spawn_stack[g_remaining_area_tick_spawn_depth] = {};
}

void remaining_refresh_epoch(const CombatCaptureContext &context) {
  if (g_remaining_runtime_generation.load(std::memory_order_acquire) == context.generation &&
      g_remaining_runtime_state_epoch.load(std::memory_order_acquire) == context.state_epoch) {
    return;
  }
  if (g_remaining_runtime_epoch_lock.test_and_set(std::memory_order_acquire)) {
    g_remaining_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (g_remaining_runtime_generation.load(std::memory_order_relaxed) != context.generation ||
      g_remaining_runtime_state_epoch.load(std::memory_order_relaxed) != context.state_epoch) {
    remaining_lock(g_remaining_pending_removal_lock);
    for (RemainingLifetimeEntry &pending : g_remaining_pending_removals) {
      pending = {};
    }
    remaining_unlock(g_remaining_pending_removal_lock);
    remaining_lock(g_remaining_pending_spawn_attach_lock);
    for (RemainingPendingSpawnAttach &pending : g_remaining_pending_spawn_attaches) {
      pending = {};
    }
    remaining_unlock(g_remaining_pending_spawn_attach_lock);
    remaining_lock(g_remaining_pending_action_spawn_lock);
    for (RemainingPendingActionSpawn &pending : g_remaining_pending_action_spawns) {
      pending = {};
    }
    remaining_unlock(g_remaining_pending_action_spawn_lock);
    remaining_lock(g_remaining_tracked_lifetime_lock);
    for (RemainingLifetimeEntry &tracked : g_remaining_tracked_lifetimes) {
      tracked = {};
    }
    remaining_unlock(g_remaining_tracked_lifetime_lock);
    const std::uint64_t first = g_remaining_runtime_next_sequence.load(std::memory_order_acquire);
    g_remaining_runtime_generation.store(context.generation, std::memory_order_release);
    g_remaining_runtime_state_epoch.store(context.state_epoch, std::memory_order_release);
    g_remaining_runtime_epoch_first_sequence.store(first, std::memory_order_release);
    g_remaining_runtime_overflow_count.store(0, std::memory_order_release);
    g_remaining_runtime_rejected_count.store(0, std::memory_order_release);
  }
  g_remaining_runtime_epoch_lock.clear(std::memory_order_release);
}

void remaining_reject_impl(const CombatCaptureContext *context, std::int32_t source_line) {
  if (context != nullptr) {
    remaining_refresh_epoch(*context);
  }
  LOGE("remaining-runtime capture rejected sourceLine=%d tick=%d "
       "generation=%llu stateEpoch=%llu",
       source_line, context == nullptr ? -1 : context->tick,
       static_cast<unsigned long long>(context == nullptr ? 0 : context->generation),
       static_cast<unsigned long long>(context == nullptr ? 0 : context->state_epoch));
  g_remaining_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
}

bool remaining_object_is_committed(const void *object, void *object_manager, bool *object_vector_valid) {
  if (object_vector_valid == nullptr) {
    return false;
  }
  *object_vector_valid = false;
  void **objects = nullptr;
  std::int32_t count = 0;
  if (!read_combat_object_vector(object_manager, &objects, &count)) {
    return false;
  }
  *object_vector_valid = true;
  for (std::int32_t index = 0; index < count; ++index) {
    if (objects[index] == object) {
      return true;
    }
  }
  return false;
}

RemainingRawRecord remaining_base_record(const CombatCaptureContext &context, RemainingRawKind kind,
                                         std::uintptr_t hook_offset, std::uintptr_t caller_offset = 0) {
  RemainingRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind = kind;
  record.hook_offset = hook_offset;
  record.caller_offset = caller_offset;
  return record;
}

std::uint64_t publish_remaining_record(RemainingRawRecord record) {
  CombatCaptureContext context;
  context.generation = record.generation;
  context.state_epoch = record.state_epoch;
  context.tick = record.tick;
  remaining_refresh_epoch(context);
  const std::uint64_t sequence = g_remaining_runtime_next_sequence.fetch_add(1, std::memory_order_acq_rel);
  record.sequence = sequence;
  const std::uint64_t first = g_remaining_runtime_epoch_first_sequence.load(std::memory_order_acquire);
  if (sequence >= first + kRemainingRuntimeRingCapacity) {
    g_remaining_runtime_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
  RemainingRingSlot &slot = g_remaining_runtime_ring[(sequence - 1) % kRemainingRuntimeRingCapacity];
  slot.published_sequence.store(0, std::memory_order_relaxed);
  slot.record = record;
  slot.published_sequence.store(sequence, std::memory_order_release);
  return sequence;
}

void remaining_publish_area_relations(const CombatCaptureContext &context, const void *area,
                                      const CombatEntityFact &area_fact, std::uintptr_t vtable_offset) {
  if (!process_range_is_readable(area, cr_remaining_runtime::kAreaEffectObjectRelatedTargetOffset + sizeof(void *))) {
    remaining_reject(&context);
    return;
  }
  struct RelationInput {
    std::size_t offset;
    RemainingRawKind kind;
  };
  const RelationInput relations[] = {
      {cr_remaining_runtime::kAreaEffectObjectParentOffset, RemainingRawKind::AreaRelationParent},
      {cr_remaining_runtime::kAreaEffectObjectFollowTargetOffset, RemainingRawKind::AreaRelationFollow},
      {cr_remaining_runtime::kAreaEffectObjectRelatedTargetOffset, RemainingRawKind::AreaRelationRelated},
  };
  for (const RelationInput &relation : relations) {
    const void *const pointer = read_object_field<const void *>(area, relation.offset);
    if (pointer == nullptr) {
      continue;
    }
    RemainingRawRecord record = remaining_base_record(context, relation.kind, cr_remaining_runtime::kCombatSpawnOffset);
    record.entity = area_fact;
    record.object_kind = area_fact.object_kind;
    record.runtime_vtable_offset = vtable_offset;
    capture_combat_entity(pointer, context.object_manager, &record.target);
    if (!record.target.validated || !record.target.present ||
        record.target.native_object_id == area_fact.native_object_id) {
      bool object_vector_valid = false;
      const bool committed = remaining_object_is_committed(pointer, context.object_manager, &object_vector_valid);
      // These offsets are relation-capable storage, not proof that every
      // non-null value is a committed LogicGameObject.  Some exact area
      // classes temporarily use one slot for an auxiliary/private
      // pointer.  A readable ObjectManager vector can prove that such a
      // pointer is outside the public relation stream.  If the pointer
      // is a committed object, capture failure or a self-relation stays
      // fail-closed.
      if (object_vector_valid && !committed) {
        continue;
      }
      remaining_reject(&context);
      continue;
    }
    record.complete_context = true;
    publish_remaining_record(record);
  }
}

bool remaining_same_lifetime(const RemainingRemovalCapture &left, const RemainingRemovalCapture &right) {
  return left.context.generation == right.context.generation && left.context.state_epoch == right.context.state_epoch &&
         left.object_pointer == right.object_pointer && left.entity.native_object_id == right.entity.native_object_id;
}

template <std::size_t Capacity>
bool remaining_store_lifetime(const RemainingRemovalCapture &capture, RemainingLifetimeEntry (&entries)[Capacity],
                              std::atomic_flag &lock) {
  remaining_lock(lock);
  RemainingLifetimeEntry *available = nullptr;
  for (auto &entry : entries) {
    if (entry.active && remaining_same_lifetime(entry.capture, capture)) {
      remaining_unlock(lock);
      return true;
    }
    if (!entry.active && available == nullptr) {
      available = &entry;
    }
  }
  if (available != nullptr) {
    available->active = true;
    available->capture = capture;
  }
  remaining_unlock(lock);
  return available != nullptr;
}

void remaining_track_committed_lifetime(const CombatCaptureContext &context, const void *object,
                                        const RemainingRawRecord &lifetime, bool exact_area_effect,
                                        std::int32_t area_remaining_life_ms) {
  RemainingRemovalCapture capture;
  capture.validated = true;
  capture.exact_area_effect = exact_area_effect;
  capture.object_pointer = object;
  capture.context = context;
  capture.entity = lifetime.entity;
  capture.runtime_vtable_offset = lifetime.runtime_vtable_offset;
  capture.data_global_id = lifetime.data_after_global_id;
  capture.area_remaining_life_ms = area_remaining_life_ms;
  if (!remaining_store_lifetime(capture, g_remaining_tracked_lifetimes, g_remaining_tracked_lifetime_lock)) {
    remaining_reject(&context);
  }
}

void remaining_runtime_note_committed_spawn(void *object_manager, void *object, std::int32_t commit_to_live) {
  if (commit_to_live == 0) {
    return;
  }
  // The exact shared spawn hook calls this immediately after committing the
  // object. Clear any prior proof for the address before it can represent a
  // recycled object with the same native ID.
  remaining_area_eligibility_cache_invalidate_pointer(object);
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  if (context.object_manager == nullptr) {
    context.object_manager = object_manager;
  }
  if (object_manager == nullptr || context.object_manager != object_manager) {
    remaining_reject(&context);
    return;
  }
  RemainingRawRecord lifetime =
      remaining_base_record(context, RemainingRawKind::LifetimeSpawn, cr_remaining_runtime::kCombatSpawnOffset);
  capture_combat_entity(object, object_manager, &lifetime.entity);
  if (!lifetime.entity.validated || !lifetime.entity.present) {
    remaining_reject(&context);
    return;
  }
  const void *const data = remaining_object_data(object);
  (void)remaining_data_global_id(data, &lifetime.data_after_global_id);
  lifetime.object_kind = lifetime.entity.object_kind;
  lifetime.runtime_vtable_offset = remaining_vtable_offset(object);
  lifetime.committed = true;
  lifetime.complete_context = true;
  publish_remaining_record(lifetime);

  std::uintptr_t area_vtable_offset = 0;
  const bool exact_area_effect = remaining_exact_area_effect(object, lifetime.entity, &area_vtable_offset);
  std::int32_t area_remaining_life_ms = kRemainingRuntimeUnset;
  if (exact_area_effect) {
    RemainingRawRecord area = lifetime;
    area.kind = RemainingRawKind::AreaCreate;
    area.runtime_vtable_offset = area_vtable_offset;
    if (process_range_is_readable(object,
                                  cr_remaining_runtime::kAreaEffectObjectRemainingLifeOffset + sizeof(std::int32_t))) {
      area_remaining_life_ms =
          read_object_field<std::int32_t>(object, cr_remaining_runtime::kAreaEffectObjectRemainingLifeOffset);
    }
    area.area_remaining_life_ms = area_remaining_life_ms;
    publish_remaining_record(area);
    remaining_publish_area_relations(context, object, lifetime.entity, area_vtable_offset);
  }
  remaining_track_committed_lifetime(context, object, lifetime, exact_area_effect, area_remaining_life_ms);

  remaining_lock(g_remaining_pending_spawn_attach_lock);
  for (RemainingPendingSpawnAttach &pending : g_remaining_pending_spawn_attaches) {
    if (!pending.active || pending.child_pointer != nullptr || pending.context.generation != context.generation ||
        pending.context.state_epoch != context.state_epoch ||
        pending.child_data_global_id != lifetime.data_after_global_id || context.tick < pending.context.tick ||
        context.tick - pending.context.tick > 2) {
      continue;
    }
    pending.child_pointer = object;
    pending.child_native_object_id = lifetime.entity.native_object_id;
    break;
  }
  remaining_unlock(g_remaining_pending_spawn_attach_lock);
}

bool remaining_runtime_apply_action_spawn_provenance(void *object_manager, void *object,
                                                     CombatRawRecord *combat_record) {
  if (object == nullptr || object_manager == nullptr) {
    return false;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return false;
  }
  RemainingPendingActionSpawn pending;
  remaining_lock(g_remaining_pending_action_spawn_lock);
  for (RemainingPendingActionSpawn &candidate : g_remaining_pending_action_spawns) {
    if (!candidate.active || candidate.child_pointer != object || candidate.capture.generation != context.generation ||
        candidate.capture.state_epoch != context.state_epoch) {
      continue;
    }
    pending = candidate;
    candidate = {};
    break;
  }
  remaining_unlock(g_remaining_pending_action_spawn_lock);
  if (!pending.active) {
    return false;
  }
  if (combat_record == nullptr || object == pending.source_pointer) {
    remaining_reject(&context);
    return true;
  }
  if (context.object_manager == nullptr) {
    context.object_manager = object_manager;
  }
  if (context.object_manager != object_manager || context.generation != pending.capture.generation ||
      context.state_epoch != pending.capture.state_epoch) {
    remaining_reject(&context);
    return true;
  }
  CombatEntityFact source;
  capture_combat_entity(pending.source_pointer, object_manager, &source);
  const bool live_source_identity_exact = source.validated && source.present &&
                                          source.native_object_id == pending.source_native_object_id &&
                                          source.native_object_id != combat_record->target.native_object_id;
  const bool live_source_absent = !source.validated || !source.present;
  const bool stored_source_exact =
      context.tick == pending.capture.tick && pending.source.validated && pending.source.present &&
      pending.source.native_object_id == pending.source_native_object_id &&
      pending.source.native_object_id != combat_record->target.native_object_id && pending.source_vtable_offset != 0;
  if (!stored_source_exact || (!live_source_absent && !live_source_identity_exact)) {
    LOGE("remaining action-spawn provenance source invalid tick=%d "
         "sourceValidated=%d sourcePresent=%d capturedSourceId=%u "
         "expectedSourceId=%u targetId=%u samePointer=%d "
         "captureTick=%d storedValidated=%d storedPresent=%d "
         "storedSourceId=%u storedVtableOffset=%llx",
         context.tick, source.validated ? 1 : 0, source.present ? 1 : 0, source.native_object_id,
         pending.source_native_object_id, combat_record->target.native_object_id,
         pending.source_pointer == object ? 1 : 0, pending.capture.tick, pending.source.validated ? 1 : 0,
         pending.source.present ? 1 : 0, pending.source.native_object_id,
         static_cast<unsigned long long>(pending.source_vtable_offset));
    remaining_reject(&context);
    return true;
  }
  const void *const child_data = remaining_object_data(object);
  std::int32_t source_data_global_id = pending.source_data_global_id;
  std::int32_t child_data_global_id = kRemainingRuntimeGlobalIdUnset;
  bool live_source_data_exact = false;
  if (live_source_identity_exact) {
    const void *const source_data = remaining_object_data(pending.source_pointer);
    std::int32_t live_source_data_global_id = kRemainingRuntimeGlobalIdUnset;
    live_source_data_exact = remaining_data_global_id(source_data, &live_source_data_global_id) &&
                             live_source_data_global_id == pending.source_data_global_id;
  }
  if (!live_source_data_exact) {
    // The exact ActionSpawnToLocation entry fact is authoritative for the
    // producer edge.  Some actions remove the source or transform its data
    // before their child commits later in the same native call/tick.  The
    // stored identity is admitted only by the same-tick contract above;
    // pointer recycling and every later-tick fallback remain rejected.
    source = pending.source;
  }
  if (!remaining_data_global_id(child_data, &child_data_global_id)) {
    remaining_reject(&context);
    return true;
  }

  combat_record->immediate_source = source;
  combat_record->source = source;
  combat_record->spawn_provenance.kind = CombatSpawnProvenanceKind::ActionSpawnToLocation;
  combat_record->spawn_provenance.hook_offset = cr_remaining_runtime::kActionSpawnToLocationExecuteOffset;
  combat_record->spawn_provenance.source_data_global_id = source_data_global_id;

  RemainingRawRecord record = remaining_base_record(context, RemainingRawKind::AreaActionSpawn,
                                                    cr_remaining_runtime::kActionSpawnToLocationExecuteOffset);
  record.entity = combat_record->target;
  record.source = source;
  record.target = combat_record->target;
  record.object_kind = source.object_kind;
  record.runtime_vtable_offset = pending.source_vtable_offset;
  record.data_after_global_id = child_data_global_id;
  record.configured_data_global_id = pending.action_data_global_id;
  record.committed = true;
  record.complete_context = true;
  publish_remaining_record(record);
  return true;
}

void remaining_runtime_apply_area_tick_spawn_provenance(void *object_manager, void *object,
                                                        CombatRawRecord *combat_record) {
  if (remaining_runtime_apply_action_spawn_provenance(object_manager, object, combat_record)) {
    return;
  }
  if (combat_record == nullptr || g_remaining_area_tick_spawn_depth == 0) {
    return;
  }
  const RemainingAreaTickSpawnContext &pending =
      g_remaining_area_tick_spawn_stack[g_remaining_area_tick_spawn_depth - 1];
  if (!pending.active || object == nullptr || object == pending.area_pointer || object_manager == nullptr) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  if (context.object_manager == nullptr) {
    context.object_manager = object_manager;
  }
  if (context.object_manager != object_manager || context.generation != pending.capture.generation ||
      context.state_epoch != pending.capture.state_epoch) {
    remaining_reject(&context);
    return;
  }
  CombatEntityFact source;
  capture_combat_entity(pending.area_pointer, object_manager, &source);
  if (!source.validated || !source.present || source.native_object_id != pending.area_native_object_id ||
      source.native_object_id == combat_record->target.native_object_id ||
      remaining_vtable_offset(pending.area_pointer) != pending.area_vtable_offset ||
      !cr_remaining_runtime::exact_area_effect_object(source.object_kind, pending.area_vtable_offset)) {
    remaining_reject(&context);
    return;
  }
  const void *const source_data = remaining_object_data(pending.area_pointer);
  const void *const child_data = remaining_object_data(object);
  std::int32_t source_data_global_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t child_data_global_id = kRemainingRuntimeGlobalIdUnset;
  if (!remaining_data_global_id(source_data, &source_data_global_id) ||
      !remaining_data_global_id(child_data, &child_data_global_id) ||
      source_data_global_id != pending.area_data_global_id) {
    remaining_reject(&context);
    return;
  }

  // The source is carried by an exact native dynamic scope, not derived
  // from deck slot, owner, card ID, coordinate, or elapsed time.
  combat_record->immediate_source = source;
  combat_record->source = source;
  combat_record->spawn_provenance.kind = CombatSpawnProvenanceKind::AreaTickNestedSpawn;
  combat_record->spawn_provenance.hook_offset = cr_remaining_runtime::kAreaEffectObjectTickOffset;
  combat_record->spawn_provenance.source_data_global_id = source_data_global_id;

  RemainingRawRecord record = remaining_base_record(context, RemainingRawKind::AreaActionSpawn,
                                                    cr_remaining_runtime::kAreaEffectObjectTickOffset);
  record.entity = combat_record->target;
  record.source = source;
  record.target = combat_record->target;
  record.object_kind = source.object_kind;
  record.runtime_vtable_offset = pending.area_vtable_offset;
  record.data_after_global_id = child_data_global_id;
  record.configured_data_global_id = source_data_global_id;
  record.committed = true;
  record.complete_context = true;
  publish_remaining_record(record);
}

bool remaining_runtime_capture_before_remove(void *object_manager, const void *object,
                                             RemainingRemovalCapture *result) {
  if (result == nullptr) {
    return false;
  }
  *result = {};
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return false;
  }
  if (context.object_manager == nullptr) {
    context.object_manager = object_manager;
  }
  if (object_manager == nullptr || context.object_manager != object_manager) {
    remaining_reject(&context);
    return false;
  }
  result->context = context;
  capture_combat_entity(object, object_manager, &result->entity);
  if (!result->entity.validated || !result->entity.present) {
    remaining_reject(&context);
    return false;
  }
  result->runtime_vtable_offset = remaining_vtable_offset(object);
  result->object_pointer = object;
  result->exact_area_effect =
      cr_remaining_runtime::exact_area_effect_object(result->entity.object_kind, result->runtime_vtable_offset);
  if (process_range_is_readable(object, cr_remaining_runtime::kObjectDataOffset + sizeof(void *))) {
    const void *const data = read_object_field<const void *>(object, cr_remaining_runtime::kObjectDataOffset);
    (void)remaining_data_global_id(data, &result->data_global_id);
  }
  if (result->exact_area_effect &&
      process_range_is_readable(object,
                                cr_remaining_runtime::kAreaEffectObjectRemainingLifeOffset + sizeof(std::int32_t))) {
    result->area_remaining_life_ms =
        read_object_field<std::int32_t>(object, cr_remaining_runtime::kAreaEffectObjectRemainingLifeOffset);
  }
  result->validated = true;
  return true;
}

template <std::size_t Capacity>
void remaining_clear_lifetime(const RemainingRemovalCapture &capture, RemainingLifetimeEntry (&entries)[Capacity],
                              std::atomic_flag &lock) {
  remaining_lock(lock);
  for (auto &entry : entries) {
    if (entry.active && remaining_same_lifetime(entry.capture, capture)) {
      entry = {};
    }
  }
  remaining_unlock(lock);
}

void remaining_runtime_note_confirmed_remove(const RemainingRemovalCapture &capture, bool absent_after) {
  if (!capture.validated) {
    return;
  }
  if (!absent_after) {
    if (!remaining_store_lifetime(capture, g_remaining_pending_removals, g_remaining_pending_removal_lock)) {
      remaining_reject(&capture.context);
    }
    return;
  }
  remaining_clear_lifetime(capture, g_remaining_tracked_lifetimes, g_remaining_tracked_lifetime_lock);
  remaining_clear_lifetime(capture, g_remaining_pending_removals, g_remaining_pending_removal_lock);
  // A cached proof is valid only while both native identities remain live in
  // this epoch. Invalidate every predicate involving the removed object
  // before its address or object ID can be reused.
  remaining_area_eligibility_cache_invalidate(capture.context, capture.entity.native_object_id);
  RemainingRawRecord lifetime = remaining_base_record(capture.context, RemainingRawKind::LifetimeDespawn,
                                                      cr_remaining_runtime::kCombatRemoveOffset);
  lifetime.entity = capture.entity;
  lifetime.object_kind = capture.entity.object_kind;
  lifetime.runtime_vtable_offset = capture.runtime_vtable_offset;
  lifetime.data_before_global_id = capture.data_global_id;
  lifetime.complete_context = true;
  publish_remaining_record(lifetime);
  if (!capture.exact_area_effect) {
    return;
  }
  RemainingRawRecord area = lifetime;
  area.kind = RemainingRawKind::AreaExpire;
  area.area_remaining_life_ms = capture.area_remaining_life_ms;
  publish_remaining_record(area);
}

// Both tables defer removal until the public object vector proves that the
// captured identity has disappeared. Batch at most 256 confirmations while
// holding the table lock; publishing after unlock preserves the lock order.
template <std::size_t Capacity>
void remaining_reconcile_lifetimes(const CombatCaptureContext &observation_context,
                                   RemainingLifetimeEntry (&entries)[Capacity], std::atomic_flag &lock) {
  if (observation_context.object_manager == nullptr) {
    return;
  }
  RemainingRemovalCapture confirmed[kRemainingPendingRemovalCapacity];
  std::size_t confirmed_count = 0;
  remaining_lock(lock);
  for (auto &entry : entries) {
    if (!entry.active) {
      continue;
    }
    const auto &capture = entry.capture;
    if (capture.context.generation != observation_context.generation ||
        capture.context.state_epoch != observation_context.state_epoch) {
      entry = {};
      continue;
    }
    bool is_member = false;
    if (!combat_try_pointer_membership(observation_context.object_manager, capture.object_pointer, &is_member)) {
      continue;
    }
    if (is_member) {
      CombatEntityFact current;
      capture_combat_entity(capture.object_pointer, observation_context.object_manager, &current);
      if (!current.validated || !current.present || current.native_object_id == capture.entity.native_object_id) {
        continue;
      }
    }
    confirmed[confirmed_count] = capture;
    confirmed[confirmed_count].context.tick = observation_context.tick;
    ++confirmed_count;
    entry = {};
    if (confirmed_count == kRemainingPendingRemovalCapacity) {
      break;
    }
  }
  remaining_unlock(lock);
  for (std::size_t index = 0; index < confirmed_count; ++index) {
    remaining_runtime_note_confirmed_remove(confirmed[index], true);
  }
}

void remaining_runtime_reconcile_pending_removals(const CombatCaptureContext &context) {
  remaining_reconcile_lifetimes(context, g_remaining_pending_removals, g_remaining_pending_removal_lock);
}

void remaining_runtime_reconcile_tracked_lifetimes(const CombatCaptureContext &context) {
  remaining_reconcile_lifetimes(context, g_remaining_tracked_lifetimes, g_remaining_tracked_lifetime_lock);
}

void remaining_runtime_reconcile_pending_spawn_attaches(const CombatCaptureContext &observation_context) {
  if (observation_context.object_manager == nullptr) {
    return;
  }
  RemainingRawRecord confirmed[kRemainingPendingSpawnAttachCapacity];
  std::size_t confirmed_count = 0;
  std::size_t rejected_count = 0;
  remaining_lock(g_remaining_pending_spawn_attach_lock);
  for (RemainingPendingSpawnAttach &pending : g_remaining_pending_spawn_attaches) {
    if (!pending.active) {
      continue;
    }
    if (pending.context.generation != observation_context.generation ||
        pending.context.state_epoch != observation_context.state_epoch) {
      pending = {};
      continue;
    }
    if (pending.child_pointer == nullptr) {
      if (observation_context.tick > pending.context.tick + 2) {
        ++rejected_count;
        pending = {};
      }
      continue;
    }
    bool source_member = false;
    bool child_member = false;
    if (!combat_try_pointer_membership(observation_context.object_manager, pending.source_pointer, &source_member) ||
        !combat_try_pointer_membership(observation_context.object_manager, pending.child_pointer, &child_member)) {
      continue;
    }
    if (!source_member || !child_member) {
      continue;
    }
    CombatEntityFact source;
    CombatEntityFact child;
    capture_combat_entity(pending.source_pointer, observation_context.object_manager, &source);
    capture_combat_entity(pending.child_pointer, observation_context.object_manager, &child);
    if (!source.validated || !source.present || !child.validated || !child.present) {
      continue;
    }
    if ((pending.source_native_object_id != 0 && source.native_object_id != pending.source_native_object_id) ||
        child.native_object_id != pending.child_native_object_id || source.native_object_id == child.native_object_id) {
      ++rejected_count;
      pending = {};
      continue;
    }
    const void *const child_data = remaining_object_data(pending.child_pointer);
    std::int32_t child_data_global_id = kRemainingRuntimeGlobalIdUnset;
    if (!remaining_data_global_id(child_data, &child_data_global_id) ||
        child_data_global_id != pending.child_data_global_id) {
      ++rejected_count;
      pending = {};
      continue;
    }
    RemainingRawRecord attachment = remaining_base_record(observation_context, RemainingRawKind::SpawnAttach,
                                                          cr_remaining_runtime::kCharacterSpawnHelperOffset,
                                                          cr_remaining_runtime::kSpawnAttachHelperReturnOffset);
    attachment.entity = child;
    attachment.source = source;
    attachment.target = child;
    attachment.configured_data_global_id = pending.child_data_global_id;
    attachment.data_after_global_id = child_data_global_id;
    attachment.committed = true;
    attachment.complete_context = true;
    confirmed[confirmed_count++] = attachment;
    pending = {};
  }
  remaining_unlock(g_remaining_pending_spawn_attach_lock);
  for (std::size_t index = 0; index < rejected_count; ++index) {
    remaining_reject(&observation_context);
  }
  for (std::size_t index = 0; index < confirmed_count; ++index) {
    publish_remaining_record(confirmed[index]);
  }
}

bool remaining_runtime_begin_remove(void *object_manager, void *object) {
  // Invalidate before the provider enters native removal. A concurrent fast
  // eligibility lookup either linearizes before this lock or observes no
  // entry; it never dereferences a pointer while removal is in progress.
  remaining_area_eligibility_cache_invalidate_pointer(object);
  if (g_remaining_remove_depth >= kRemainingRemoveDepth) {
    CombatCaptureContext context;
    if (current_combat_capture_context(&context)) {
      remaining_reject(&context);
    }
    return false;
  }
  RemainingRemovalCapture capture;
  if (!remaining_runtime_capture_before_remove(object_manager, object, &capture)) {
    return false;
  }
  g_remaining_remove_stack[g_remaining_remove_depth++] = capture;
  return true;
}

void remaining_runtime_end_remove(bool absent_after) {
  if (g_remaining_remove_depth == 0) {
    CombatCaptureContext context;
    if (current_combat_capture_context(&context)) {
      remaining_reject(&context);
    }
    return;
  }
  const RemainingRemovalCapture capture = g_remaining_remove_stack[--g_remaining_remove_depth];
  g_remaining_remove_stack[g_remaining_remove_depth] = {};
  remaining_runtime_note_confirmed_remove(capture, absent_after);
}

bool remaining_exact_tower_fact(const CombatEntityFact &tower) {
  if (!tower.validated || !tower.present || tower.card_id != -1 || (tower.owner != 0 && tower.owner != 1)) {
    return false;
  }
  constexpr std::int32_t kKingX = 9000;
  constexpr std::int32_t kLeftPrincessX = 3500;
  constexpr std::int32_t kRightPrincessX = 14500;
  constexpr std::int32_t kKingY[2] = {3000, 29000};
  constexpr std::int32_t kPrincessY[2] = {6500, 25500};
  return (tower.position_x == kKingX && tower.position_y == kKingY[tower.owner]) ||
         ((tower.position_x == kLeftPrincessX || tower.position_x == kRightPrincessX) &&
          tower.position_y == kPrincessY[tower.owner]);
}

void remaining_runtime_note_tower_aggro(const void *tower, const void *target_before, const void *target_after,
                                        bool attack_component_validated, bool tower_identity_validated) {
  if (target_before == target_after) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  if (!attack_component_validated || !tower_identity_validated) {
    remaining_reject(&context);
    return;
  }
  RemainingRawKind kind = RemainingRawKind::TowerAggroChange;
  if (target_before == nullptr) {
    kind = RemainingRawKind::TowerAggroAcquire;
  } else if (target_after == nullptr) {
    kind = RemainingRawKind::TowerAggroLose;
  }
  RemainingRawRecord record = remaining_base_record(context, kind, cr_remaining_runtime::kTowerTargetSetOffset);
  capture_combat_entity(tower, context.object_manager, &record.entity);
  capture_combat_entity(target_before, context.object_manager, &record.target_before);
  capture_combat_entity(target_after, context.object_manager, &record.target_after);
  if (!record.entity.validated || !record.entity.present || !record.target_before.validated ||
      !record.target_after.validated || (target_before != nullptr && !record.target_before.present) ||
      (target_after != nullptr && !record.target_after.present)) {
    remaining_reject(&context);
    return;
  }
  record.target = record.target_after;
  record.complete_context = true;
  publish_remaining_record(record);
}

void remaining_runtime_observe_tower_aggro(const void *tower, const void *target_before, const void *target_after,
                                           bool attack_component_validated) {
  if (target_before == target_after) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  CombatEntityFact fact;
  capture_combat_entity(tower, context.object_manager, &fact);
  if (!remaining_exact_tower_fact(fact)) {
    // The target-set provider is shared by every attacking entity.
    // Non-tower target transitions are outside the tower-aggro observable.
    return;
  }
  remaining_runtime_note_tower_aggro(tower, target_before, target_after, attack_component_validated, true);
}

void remaining_runtime_note_tower_activation(const void *tower, const void *target, const void *dispatched_action,
                                             const void *wait_static, const void *wait_runtime, bool condition_result,
                                             const void *caller_return_address, bool tower_identity_validated) {
  const std::uintptr_t caller_offset = remaining_caller_offset(caller_return_address);
  if (caller_offset != cr_remaining_runtime::kActionWaitDispatchReturnOffset) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  const bool readable =
      process_range_is_readable(wait_static,
                                cr_remaining_runtime::kActionWaitOnActivateActionOffset + sizeof(void *)) &&
      process_range_is_readable(wait_runtime, cr_remaining_runtime::kActionRuntimeOwnerOffset + sizeof(void *));
  const bool exact =
      readable && condition_result && tower_identity_validated && tower != nullptr && tower == target &&
      remaining_vtable_offset(wait_static) == cr_remaining_runtime::kActionWaitStaticVtableOffset &&
      remaining_vtable_offset(wait_runtime) == cr_remaining_runtime::kActionWaitRuntimeVtableOffset &&
      read_object_field<const void *>(wait_runtime, cr_remaining_runtime::kActionRuntimeStaticDataOffset) ==
          wait_static &&
      read_object_field<const void *>(wait_runtime, cr_remaining_runtime::kActionRuntimeOwnerOffset) == tower &&
      read_object_field<const void *>(wait_static, cr_remaining_runtime::kActionWaitOnActivateActionOffset) ==
          dispatched_action &&
      read_object_field<const void *>(wait_static, cr_remaining_runtime::kActionWaitConditionOffset) != nullptr;
  if (!exact) {
    remaining_reject(&context);
    return;
  }
  RemainingRawRecord record = remaining_base_record(context, RemainingRawKind::TowerActivate,
                                                    cr_remaining_runtime::kActionDispatchOffset, caller_offset);
  capture_combat_entity(tower, context.object_manager, &record.entity);
  if (!record.entity.validated || !record.entity.present) {
    remaining_reject(&context);
    return;
  }
  (void)remaining_data_global_id(dispatched_action, &record.configured_data_global_id);
  record.result = true;
  record.complete_context = true;
  publish_remaining_record(record);
}

void remaining_runtime_observe_action_dispatch(const void *owner, const void *target, const void *action,
                                               const void *caller_return_address) {
  if (remaining_caller_offset(caller_return_address) != cr_remaining_runtime::kActionWaitDispatchReturnOffset ||
      g_remaining_action_wait_depth == 0) {
    return;
  }
  const void *const wait_runtime = g_remaining_action_wait_stack[g_remaining_action_wait_depth - 1];
  const void *const wait_static =
      process_range_is_readable(wait_runtime, cr_remaining_runtime::kActionRuntimeStaticDataOffset + sizeof(void *))
          ? read_object_field<const void *>(wait_runtime, cr_remaining_runtime::kActionRuntimeStaticDataOffset)
          : nullptr;
  CombatCaptureContext context;
  CombatEntityFact owner_fact;
  bool tower_identity = false;
  if (current_combat_capture_context(&context)) {
    capture_combat_entity(owner, context.object_manager, &owner_fact);
    tower_identity = remaining_exact_tower_fact(owner_fact) && owner_fact.position_x == 9000;
  }
  // ActionWait is generic. Other cards legitimately reach the same dispatch
  // edge and are outside the King Tower activation observable.
  if (!tower_identity) {
    return;
  }
  remaining_runtime_note_tower_activation(owner, target, action, wait_static, wait_runtime, true, caller_return_address,
                                          tower_identity);
}

extern "C" void remaining_action_spawn_mode1_child_create_bridge(void *object_manager, void *child, std::int32_t option,
                                                                 std::int32_t commit_to_live,
                                                                 const void *caller_return_address,
                                                                 const void *ambient_source,
                                                                 const void *request_plus_0x48) {

  if (remaining_caller_offset(caller_return_address) !=
          cr_remaining_runtime::kActionSpawnMode1ChildCreateReturnOffset ||
      option != 1 || commit_to_live != 0) {
    return;
  }

  CombatCaptureContext current;
  const bool has_context = g_remaining_action_spawn_depth != 0 && current_combat_capture_context(&current);
  const RemainingActionSpawnContext *source =
      has_context ? &g_remaining_action_spawn_stack[g_remaining_action_spawn_depth - 1] : nullptr;
  const std::uintptr_t request_slot = reinterpret_cast<std::uintptr_t>(request_plus_0x48);
  const bool request_base_valid = request_slot >= cr_remaining_runtime::kActionSpawnMode1RequestBaseFromSlot;
  const void *const request =
      request_base_valid
          ? reinterpret_cast<const void *>(request_slot - cr_remaining_runtime::kActionSpawnMode1RequestBaseFromSlot)
          : nullptr;
  const bool readable =
      request != nullptr &&
      process_range_is_readable(request, cr_remaining_runtime::kActionSpawnMode1RequestSourceOffset + sizeof(void *)) &&
      child != nullptr && process_range_is_readable(child, cr_remaining_runtime::kObjectDataOffset + sizeof(void *));
  const void *const request_data =
      readable ? read_object_field<const void *>(request, cr_remaining_runtime::kActionSpawnMode1RequestDataOffset)
               : nullptr;
  const void *const request_source =
      readable ? read_object_field<const void *>(request, cr_remaining_runtime::kActionSpawnMode1RequestSourceOffset)
               : nullptr;
  const void *const child_data =
      readable ? read_object_field<const void *>(child, cr_remaining_runtime::kObjectDataOffset) : nullptr;

  const bool exact = source != nullptr && source->active && source->action_mode == 1 && object_manager != nullptr &&
                     object_manager == current.object_manager && source->capture.generation == current.generation &&
                     source->capture.state_epoch == current.state_epoch && child != source->source_pointer &&
                     ambient_source == source->source_pointer && request_source == source->source_pointer &&
                     request_data != nullptr && child_data == request_data;
  std::int32_t child_data_global_id = kRemainingRuntimeGlobalIdUnset;
  if (!exact || !remaining_data_global_id(child_data, &child_data_global_id) ||
      child_data_global_id != source->expected_child_data_global_id) {

    if (source != nullptr) {
      remaining_reject(&source->capture);
    }
    return;
  }

  RemainingPendingActionSpawn pending;
  pending.active = true;
  pending.capture = current;
  pending.source_pointer = source->source_pointer;
  pending.source_native_object_id = source->source_native_object_id;
  pending.source = source->source;
  pending.source_vtable_offset = source->source_vtable_offset;
  pending.source_data_global_id = source->source_data_global_id;
  pending.action_data_global_id = source->action_data_global_id;
  pending.child_pointer = child;
  remaining_lock(g_remaining_pending_action_spawn_lock);
  RemainingPendingActionSpawn *available = nullptr;
  for (RemainingPendingActionSpawn &candidate : g_remaining_pending_action_spawns) {
    if (candidate.active && candidate.capture.generation == current.generation &&
        candidate.capture.state_epoch == current.state_epoch && candidate.child_pointer == child) {
      available = &candidate;
      break;
    }
    if (!candidate.active && available == nullptr) {
      available = &candidate;
    }
  }
  if (available != nullptr) {
    *available = pending;
  }
  remaining_unlock(g_remaining_pending_action_spawn_lock);
  if (available == nullptr) {

    remaining_reject(&pending.capture);
    return;
  }
}

void remaining_action_spawn_to_location_execute_hook(void *action_data, void *action_context, void *target,
                                                     void *auxiliary) {
  bool pushed = false;
  CombatCaptureContext context;
  if (action_data != nullptr && action_context != nullptr &&
      g_remaining_action_spawn_depth < kRemainingActionSpawnDepth &&
      remaining_vtable_offset(action_data) == cr_remaining_runtime::kActionSpawnToLocationStaticVtableOffset &&
      process_range_is_readable(action_context, 0x10) && current_combat_capture_context(&context)) {
    const void *const source = read_object_field<const void *>(action_context, 0x08);
    CombatEntityFact source_fact;
    capture_combat_entity(source, context.object_manager, &source_fact);
    const void *const source_data = source != nullptr ? remaining_object_data(source) : nullptr;
    RemainingActionSpawnContext pending;
    pending.active = source_fact.validated && source_fact.present;
    pending.capture = context;
    pending.source_pointer = source;
    pending.source_native_object_id = source_fact.native_object_id;
    pending.source = source_fact;
    pending.source_vtable_offset = remaining_vtable_offset(source);
    pending.active = pending.active && remaining_data_global_id(source_data, &pending.source_data_global_id);
    pending.active = pending.active && remaining_data_global_id(action_data, &pending.action_data_global_id);
    pending.active = pending.active && pending.source_vtable_offset != 0;
    pending.action_mode =
        process_range_is_readable(action_data, 0xf0) ? read_object_field<std::int32_t>(action_data, 0xec) : -1;
    const void *const expected_child_data =
        process_range_is_readable(action_data, cr_remaining_runtime::kActionSpawnToLocationDataOffset + sizeof(void *))
            ? read_object_field<const void *>(action_data, cr_remaining_runtime::kActionSpawnToLocationDataOffset)
            : nullptr;
    (void)remaining_data_global_id(expected_child_data, &pending.expected_child_data_global_id);
    if (pending.active) {
      g_remaining_action_spawn_stack[g_remaining_action_spawn_depth++] = pending;
      pushed = true;
    }
  }
  g_original_remaining_action_spawn_to_location_execute(action_data, action_context, target, auxiliary);
  if (pushed) {
    --g_remaining_action_spawn_depth;
    g_remaining_action_spawn_stack[g_remaining_action_spawn_depth] = {};
  }
}

void remaining_pending_object_add_hook(void *object_manager, void *object) {

  if (object_manager != nullptr && object != nullptr && g_remaining_action_spawn_depth != 0) {

    const RemainingActionSpawnContext &context = g_remaining_action_spawn_stack[g_remaining_action_spawn_depth - 1];
    if (context.active) {
      RemainingPendingActionSpawn pending;
      pending.active = true;
      pending.capture = context.capture;
      pending.source_pointer = context.source_pointer;
      pending.source_native_object_id = context.source_native_object_id;
      pending.source = context.source;
      pending.source_vtable_offset = context.source_vtable_offset;
      pending.source_data_global_id = context.source_data_global_id;
      pending.action_data_global_id = context.action_data_global_id;
      pending.child_pointer = object;
      remaining_lock(g_remaining_pending_action_spawn_lock);
      RemainingPendingActionSpawn *available = nullptr;
      for (RemainingPendingActionSpawn &candidate : g_remaining_pending_action_spawns) {
        if (candidate.active && candidate.capture.generation == pending.capture.generation &&
            candidate.capture.state_epoch == pending.capture.state_epoch && candidate.child_pointer == object) {
          available = &candidate;
          break;
        }
        if (!candidate.active && available == nullptr) {
          available = &candidate;
        }
      }
      if (available != nullptr) {
        *available = pending;
      }
      remaining_unlock(g_remaining_pending_action_spawn_lock);
      if (available == nullptr) {
        remaining_reject(&pending.capture);
      }
    }
  }
  g_original_remaining_pending_object_add(object_manager, object);
}

void remaining_action_wait_update_hook(void *wait_runtime) {
  CombatCaptureContext context;
  const bool capture = current_combat_capture_context(&context);
  const bool readable = capture && process_range_is_readable(
                                       wait_runtime, cr_remaining_runtime::kActionRuntimeOwnerOffset + sizeof(void *));
  const void *const wait_static =
      readable ? read_object_field<const void *>(wait_runtime, cr_remaining_runtime::kActionRuntimeStaticDataOffset)
               : nullptr;
  const bool exact = readable &&
                     remaining_vtable_offset(wait_runtime) == cr_remaining_runtime::kActionWaitRuntimeVtableOffset &&
                     remaining_vtable_offset(wait_static) == cr_remaining_runtime::kActionWaitStaticVtableOffset;
  // This implementation edge is shared by non-ActionWait runtime variants.
  // A vtable mismatch is outside the exact tower-activation observable; only
  // an attested ActionWait that cannot be tracked fails closed below.
  bool pushed = false;
  if (exact) {
    if (g_remaining_action_wait_depth >= kRemainingActionWaitDepth) {
      remaining_reject(&context);
    } else {
      g_remaining_action_wait_stack[g_remaining_action_wait_depth++] = wait_runtime;
      pushed = true;
    }
  }
  g_original_remaining_action_wait_update(wait_runtime);
  if (pushed) {
    --g_remaining_action_wait_depth;
    g_remaining_action_wait_stack[g_remaining_action_wait_depth] = nullptr;
  }
}

bool remaining_validate_resource_source(const CombatCaptureContext &context, const void *source,
                                        cr_remaining_runtime::ResourceDeltaCause cause, std::int32_t amount_argument,
                                        CombatEntityFact *source_fact, std::int32_t *configured_amount,
                                        std::int32_t *resource_owner) {
  if (source_fact == nullptr || configured_amount == nullptr || resource_owner == nullptr || source == nullptr ||
      !process_range_is_readable(source, cr_remaining_runtime::kObjectDataOffset + sizeof(void *))) {
    return false;
  }
  capture_combat_entity(source, context.object_manager, source_fact);
  if (!source_fact->validated || !source_fact->present || (source_fact->owner != 0 && source_fact->owner != 1)) {
    return false;
  }
  const void *const data = read_object_field<const void *>(source, cr_remaining_runtime::kObjectDataOffset);
  if (!process_range_is_readable(data, cr_remaining_runtime::kManaOnDeathForOpponentOffset + sizeof(std::int32_t))) {
    return false;
  }
  switch (cause) {
  case cr_remaining_runtime::ResourceDeltaCause::Periodic: {
    const bool enabled = read_object_field<std::uint8_t>(data, cr_remaining_runtime::kManaGenerationEnabledOffset) != 0;
    const std::int32_t interval =
        read_object_field<std::int32_t>(data, cr_remaining_runtime::kManaGenerateTimeMsOffset);
    *configured_amount = read_object_field<std::int32_t>(data, cr_remaining_runtime::kManaCollectAmountOffset);
    *resource_owner = source_fact->owner;
    return enabled && interval > 0 && *configured_amount == amount_argument;
  }
  case cr_remaining_runtime::ResourceDeltaCause::DeathOwner:
    *configured_amount = read_object_field<std::int32_t>(data, cr_remaining_runtime::kManaOnDeathOffset);
    *resource_owner = source_fact->owner;
    return *configured_amount == amount_argument;
  case cr_remaining_runtime::ResourceDeltaCause::DeathOpponent:
    *configured_amount = read_object_field<std::int32_t>(data, cr_remaining_runtime::kManaOnDeathForOpponentOffset);
    *resource_owner = source_fact->owner ^ 1;
    return *configured_amount == amount_argument;
  case cr_remaining_runtime::ResourceDeltaCause::None:
    break;
  }
  return false;
}

void remaining_resource_delta_bridge(RemainingResourceDelta original, std::uintptr_t hook_offset, void *resource,
                                     std::int32_t amount_argument, std::int32_t accounting, const void *caller_address,
                                     const void *source) {
  const std::uintptr_t caller_offset = remaining_caller_offset(caller_address);
  const cr_remaining_runtime::ResourceDeltaCause cause =
      cr_remaining_runtime::classify_resource_delta(hook_offset, caller_offset);
  CombatCaptureContext context;
  const bool capture =
      cause != cr_remaining_runtime::ResourceDeltaCause::None && current_combat_capture_context(&context);
  std::int32_t pre = kRemainingRuntimeUnset;
  bool resource_readable = false;
  CombatEntityFact source_fact;
  std::int32_t configured_amount = kRemainingRuntimeUnset;
  std::int32_t resource_owner = kRemainingRuntimeUnset;
  bool source_valid = false;
  if (capture) {
    resource_readable =
        process_range_is_readable(resource, cr_remaining_runtime::kResourceFixedValueOffset + sizeof(std::int32_t));
    if (resource_readable) {
      pre = read_object_field<std::int32_t>(resource, cr_remaining_runtime::kResourceFixedValueOffset);
    }
    source_valid = remaining_validate_resource_source(context, source, cause, amount_argument, &source_fact,
                                                      &configured_amount, &resource_owner);
  }
  original(resource, amount_argument, accounting);
  if (!capture) {
    return;
  }
  const std::int32_t post =
      resource_readable ? read_object_field<std::int32_t>(resource, cr_remaining_runtime::kResourceFixedValueOffset)
                        : kRemainingRuntimeUnset;
  const std::int64_t requested = cr_remaining_runtime::requested_resource_delta_fixed(cause, amount_argument);
  const std::int64_t actual = static_cast<std::int64_t>(post) - static_cast<std::int64_t>(pre);
  if (!resource_readable || !source_valid || pre < 0 || post < pre || requested <= 0 || requested > 0x7fffffffLL ||
      actual < 0 || actual > requested || actual > 0x7fffffffLL) {
    remaining_reject(&context);
    return;
  }
  // A call made at the elixir cap is an attempted generation, not a delta.
  if (actual == 0) {
    return;
  }
  RemainingRawRecord record =
      remaining_base_record(context, RemainingRawKind::ResourceDelta, hook_offset, caller_offset);
  record.source = source_fact;
  record.resource_pre_fixed = pre;
  record.resource_post_fixed = post;
  record.resource_actual_delta_fixed = static_cast<std::int32_t>(actual);
  record.amount_argument = amount_argument;
  record.configured_amount_argument = configured_amount;
  record.resource_owner = resource_owner;
  record.resource_cause = cause;
  record.complete_context = true;
  publish_remaining_record(record);
}

extern "C" void remaining_resource_whole_bridge(void *resource, std::int32_t amount_argument, std::int32_t accounting,
                                                const void *caller_address, const void *source) {
  remaining_resource_delta_bridge(g_original_remaining_resource_whole, cr_remaining_runtime::kResourceWholeDeltaOffset,
                                  resource, amount_argument, accounting, caller_address, source);
}

extern "C" void remaining_resource_fixed_bridge(void *resource, std::int32_t amount_argument, std::int32_t accounting,
                                                const void *caller_address, const void *source) {
  remaining_resource_delta_bridge(g_original_remaining_resource_fixed, cr_remaining_runtime::kResourceFixedDeltaOffset,
                                  resource, amount_argument, accounting, caller_address, source);
}

extern "C" bool remaining_spawn_helper_begin_bridge(const void *source, const void *child_data, std::int32_t,
                                                    const void *caller_address) {
  if (remaining_caller_offset(caller_address) != cr_remaining_runtime::kSpawnAttachHelperReturnOffset) {
    return false;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return false;
  }
  std::int32_t child_data_global_id = kRemainingRuntimeGlobalIdUnset;
  const bool source_readable =
      process_range_is_readable(source, cr_remaining_runtime::kObjectDataOffset + sizeof(void *));
  const void *const source_data =
      source_readable ? read_object_field<const void *>(source, cr_remaining_runtime::kObjectDataOffset) : nullptr;
  const bool source_data_readable =
      process_range_is_readable(source_data, cr_remaining_runtime::kSpawnAttachOffset + sizeof(std::uint8_t));
  std::uint32_t source_native_object_id = 0;
  (void)remaining_fast_native_object_id(source, false, &source_native_object_id);
  const bool exact = source != nullptr && source_readable && child_data != nullptr &&
                     remaining_data_global_id(child_data, &child_data_global_id) && source_data_readable &&
                     read_object_field<std::uint8_t>(source_data, cr_remaining_runtime::kSpawnAttachOffset) != 0;
  if (!exact) {
    // The helper/caller pair is shared by ordinary character creation.
    // A source whose version-locked static SpawnAttach flag is clear is
    // outside this observable, not a failed capture. Only exact declared
    // producers enter the pending table and can later fail closed.
    return false;
  }
  remaining_lock(g_remaining_pending_spawn_attach_lock);
  RemainingPendingSpawnAttach *available = nullptr;
  for (RemainingPendingSpawnAttach &pending : g_remaining_pending_spawn_attaches) {
    if (pending.active && pending.context.generation == context.generation &&
        pending.context.state_epoch == context.state_epoch && pending.source_pointer == source &&
        pending.child_pointer == nullptr && pending.child_data_global_id == child_data_global_id) {
      remaining_unlock(g_remaining_pending_spawn_attach_lock);
      return false;
    }
    if (!pending.active && available == nullptr) {
      available = &pending;
    }
  }
  if (available != nullptr) {
    available->active = true;
    available->context = context;
    available->source_pointer = source;
    available->source_native_object_id = source_native_object_id;
    available->child_data_global_id = child_data_global_id;
  }
  remaining_unlock(g_remaining_pending_spawn_attach_lock);
  if (available == nullptr) {
    remaining_reject(&context);
  }
  // The helper queues the child and returns before the shared committed-spawn
  // hook observes it. The exact caller/data context above therefore survives
  // independently; tail-call the original helper without a TLS return frame.
  return false;
}

void remaining_transform_execute_hook(void *action_data, void *action_context, void *target, void *auxiliary) {
  CombatCaptureContext context;
  const bool capture = current_combat_capture_context(&context);
  const bool readable =
      capture &&
      process_range_is_readable(action_data, cr_remaining_runtime::kChangeResetTargetOffset + sizeof(std::uint8_t)) &&
      process_range_is_readable(action_context, 0x08 + sizeof(void *));
  const bool action_exact =
      readable && remaining_vtable_offset(action_data) == cr_remaining_runtime::kChangeGameObjectDataVtableOffset;
  const void *const subject = action_exact ? read_object_field<const void *>(action_context, 0x08) : nullptr;
  const void *const data_before = remaining_object_data(subject);
  const void *const expected_character =
      action_exact ? read_object_field<const void *>(action_data, cr_remaining_runtime::kChangeNewCharacterDataOffset)
                   : nullptr;
  const void *const expected_projectile =
      action_exact ? read_object_field<const void *>(action_data, cr_remaining_runtime::kChangeNewProjectileDataOffset)
                   : nullptr;
  CombatEntityFact before_fact;
  std::int32_t before_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t expected_character_id = kRemainingRuntimeGlobalIdUnset;
  std::int32_t expected_projectile_id = kRemainingRuntimeGlobalIdUnset;
  bool pre_valid = false;
  if (action_exact) {
    capture_combat_entity(subject, context.object_manager, &before_fact);
    pre_valid = before_fact.validated && before_fact.present && remaining_data_global_id(data_before, &before_id);
    if (expected_character != nullptr) {
      pre_valid &= remaining_data_global_id(expected_character, &expected_character_id);
    }
    if (expected_projectile != nullptr) {
      pre_valid &= remaining_data_global_id(expected_projectile, &expected_projectile_id);
    }
  }
  g_original_remaining_transform_execute(action_data, action_context, target, auxiliary);
  if (!capture) {
    return;
  }
  if (!action_exact || !pre_valid) {
    remaining_reject(&context);
    return;
  }
  const void *const data_after = remaining_object_data(subject);
  if (data_after == data_before) {
    return;
  }
  std::int32_t after_id = kRemainingRuntimeGlobalIdUnset;
  CombatEntityFact after_fact;
  capture_combat_entity(subject, context.object_manager, &after_fact);
  const bool exact_transition =
      after_fact.validated && after_fact.present && after_fact.native_object_id == before_fact.native_object_id &&
      remaining_data_global_id(data_after, &after_id) &&
      cr_remaining_runtime::exact_transform_transition(
          reinterpret_cast<std::uintptr_t>(data_before), reinterpret_cast<std::uintptr_t>(data_after),
          reinterpret_cast<std::uintptr_t>(expected_character), reinterpret_cast<std::uintptr_t>(expected_projectile));
  if (!exact_transition) {
    remaining_reject(&context);
    return;
  }
  RemainingRawRecord record = remaining_base_record(context, RemainingRawKind::Transform,
                                                    cr_remaining_runtime::kChangeGameObjectDataExecuteOffset);
  record.entity = after_fact;
  record.data_before_global_id = before_id;
  record.data_after_global_id = after_id;
  record.expected_character_data_global_id = expected_character_id;
  record.expected_projectile_data_global_id = expected_projectile_id;
  if (data_after == expected_character) {
    record.transform_kind = RemainingTransformKind::Character;
    record.configured_data_global_id = expected_character_id;
  } else {
    record.transform_kind = RemainingTransformKind::Projectile;
    record.configured_data_global_id = expected_projectile_id;
  }
  record.reset_target =
      read_object_field<std::uint8_t>(action_data, cr_remaining_runtime::kChangeResetTargetOffset) != 0;
  record.complete_context = true;
  publish_remaining_record(record);
}

void remaining_reject_current_context() {
  CombatCaptureContext current;
  if (current_combat_capture_context(&current)) {
    remaining_reject(&current);
  } else {
    remaining_reject(nullptr);
  }
}

bool remaining_area_damage_eligible_hook(void *subject, void *source_or_area, std::int32_t option) {
  const bool result = g_original_remaining_area_damage_eligible(subject, source_or_area, option);
  const std::uintptr_t subject_pointer = reinterpret_cast<std::uintptr_t>(subject);
  const std::uintptr_t source_pointer = reinterpret_cast<std::uintptr_t>(source_or_area);

  // The fast proof is available only while the exact hook and both attested
  // lifecycle providers are installed. It deliberately uses the combat
  // epoch atomics rather than materializing a full context (which reads the
  // manager graph and tick). Pointer lookup happens before either native-ID
  // load, so a remove observer can invalidate an address before native code
  // releases it.
  CombatCaptureContext fast_epoch;
  const bool fast_epoch_valid = remaining_area_eligibility_fast_epoch(&fast_epoch);
  remaining_lock(g_remaining_area_eligibility_cache_lock);
  if (fast_epoch_valid && remaining_area_eligibility_fast_epoch_matches(fast_epoch)) {
    remaining_area_eligibility_cache_refresh_locked(fast_epoch);
    RemainingAreaEligibilityCacheEntry *const fast_entry =
        remaining_area_eligibility_cache_find_locked(fast_epoch, subject_pointer, source_pointer, option);
    if (fast_entry != nullptr && fast_entry->result == result) {
      const std::uint32_t subject_native_object_id = remaining_attested_native_object_id(subject);
      const std::uint32_t source_native_object_id = remaining_attested_native_object_id(source_or_area);
      if (subject_native_object_id != 0 && (source_or_area == nullptr || source_native_object_id != 0) &&
          fast_entry->subject_native_object_id == subject_native_object_id &&
          fast_entry->source_native_object_id == source_native_object_id &&
          remaining_area_eligibility_fast_epoch_matches(fast_epoch)) {
        remaining_unlock(g_remaining_area_eligibility_cache_lock);
        return result;
      }
    }
  }

  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    return result;
  }
  remaining_area_eligibility_cache_refresh_locked(context);
  if (remaining_vtable_offset(subject) != cr_remaining_runtime::kCharacterVtableOffset ||
      remaining_vtable_offset(source_or_area) != cr_remaining_runtime::kAreaEffectObjectVtableOffset) {
    // The native eligibility provider is shared by non-area predicates.
    // Only an exact Character/AreaEffect pair belongs to this observable.
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    return result;
  }
  std::uint32_t subject_native_object_id = 0;
  std::uint32_t source_native_object_id = 0;
  const bool fast_identity_valid = remaining_fast_native_object_id(subject, false, &subject_native_object_id) &&
                                   remaining_fast_native_object_id(source_or_area, true, &source_native_object_id) &&
                                   remaining_vtable_offset(subject) == cr_remaining_runtime::kCharacterVtableOffset;
  if (!fast_identity_valid) {
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    remaining_reject(&context);
    return result;
  }

  if (!remaining_capture_context_still_current(context)) {
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    remaining_reject_current_context();
    return result;
  }
  RemainingAreaEligibilityCacheEntry *entry =
      remaining_area_eligibility_cache_find_locked(context, subject_pointer, source_pointer, option);
  if (entry != nullptr && entry->subject_native_object_id == subject_native_object_id &&
      entry->source_native_object_id == source_native_object_id && entry->subject_pointer == subject_pointer &&
      entry->source_pointer == source_pointer && entry->result == result) {
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    return result;
  }

  RemainingRawRecord record = remaining_base_record(context, RemainingRawKind::AreaDamageEligibility,
                                                    cr_remaining_runtime::kAreaDamageEligibleOffset);
  capture_combat_entity(subject, context.object_manager, &record.entity);
  capture_combat_entity(source_or_area, context.object_manager, &record.source);
  const bool source_valid = record.source.validated && (source_or_area == nullptr || record.source.present) &&
                            record.source.native_object_id == source_native_object_id;
  if (!record.entity.validated || !record.entity.present ||
      record.entity.native_object_id != subject_native_object_id ||
      remaining_vtable_offset(subject) != cr_remaining_runtime::kCharacterVtableOffset || !source_valid) {
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    bool subject_vector_valid = false;
    bool source_vector_valid = false;
    const bool subject_committed =
        remaining_object_is_committed(subject, context.object_manager, &subject_vector_valid);
    const bool source_committed =
        remaining_object_is_committed(source_or_area, context.object_manager, &source_vector_valid);
    // The eligibility provider can run while Drill children/areas are
    // between construction and ObjectManager commit. Fast native IDs are
    // valid in that interval, but no public entity relation exists yet.
    // A readable manager vector proves this is an out-of-stream transient;
    // committed capture failures remain fail-closed.
    if (subject_vector_valid && source_vector_valid && (!subject_committed || !source_committed)) {
      return result;
    }
    remaining_reject_current_context();
    return result;
  }
  if (!remaining_capture_context_still_current(context)) {
    remaining_unlock(g_remaining_area_eligibility_cache_lock);
    remaining_reject_current_context();
    return result;
  }
  record.option = option != 0;
  record.result = result;
  record.complete_context = true;
  publish_remaining_record(record);
  if (entry == nullptr) {
    entry = remaining_area_eligibility_cache_store_slot_locked(subject_pointer, source_pointer, option);
  }
  *entry = {};
  entry->valid = true;
  entry->generation = context.generation;
  entry->state_epoch = context.state_epoch;
  entry->subject_native_object_id = subject_native_object_id;
  entry->source_native_object_id = source_native_object_id;
  entry->option = option;
  entry->result = result;
  entry->subject_pointer = subject_pointer;
  entry->source_pointer = source_pointer;
  remaining_unlock(g_remaining_area_eligibility_cache_lock);
  return result;
}

bool append_remaining_record_json(char *response, std::size_t response_size, std::size_t *used,
                                  const RemainingRawRecord &record) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("sequence", record.sequence);
  json.field("tick", record.tick);
  json.field("kind", remaining_raw_kind_label(record.kind));
  json.field("hookOffset", record.hook_offset);
  json.nullable("callerOffset", record.caller_offset, record.caller_offset != 0);
  struct EntityField {
    const char *name;
    const CombatEntityFact *fact;
  };
  const EntityField entities[] = {
      {"entity", &record.entity},
      {"source", &record.source},
      {"target", &record.target},
      {"targetBefore", &record.target_before},
      {"targetAfter", &record.target_after},
  };
  for (const auto &entity : entities) {
    if (!json.raw(entity.name, "") || !append_combat_entity_fact(response, response_size, used, *entity.fact)) {
      return false;
    }
  }
  json.nullable("objectKind", record.object_kind, record.object_kind != kRemainingRuntimeUnset);
  json.nullable("runtimeVtableOffset", record.runtime_vtable_offset, record.runtime_vtable_offset != 0);
  struct IntegerField {
    const char *name;
    std::int32_t value;
  };
  const IntegerField global_ids[] = {
      {"dataBeforeGlobalId", record.data_before_global_id},
      {"dataAfterGlobalId", record.data_after_global_id},
      {"expectedCharacterDataGlobalId", record.expected_character_data_global_id},
      {"expectedProjectileDataGlobalId", record.expected_projectile_data_global_id},
      {"configuredDataGlobalId", record.configured_data_global_id},
  };
  for (const auto &field : global_ids) {
    json.nullable(field.name, static_cast<std::uint32_t>(field.value), field.value != kRemainingRuntimeGlobalIdUnset);
  }
  const IntegerField optional_values[] = {
      {"resourcePreFixed", record.resource_pre_fixed},
      {"resourcePostFixed", record.resource_post_fixed},
      {"resourceActualDeltaFixed", record.resource_actual_delta_fixed},
      {"amountArgument", record.amount_argument},
      {"configuredAmountArgument", record.configured_amount_argument},
      {"resourceOwner", record.resource_owner},
      {"areaRemainingLifeMs", record.area_remaining_life_ms},
  };
  for (const auto &field : optional_values) {
    json.nullable(field.name, field.value, field.value != kRemainingRuntimeUnset);
  }
  json.field("resourceCause", remaining_resource_cause_label(record.resource_cause));
  json.field("transformKind", remaining_transform_kind_label(record.transform_kind));
  json.field("result", record.result);
  json.field("option", record.option);
  json.field("committed", record.committed);
  json.field("resetTarget", record.reset_target);
  json.field("completeContext", record.complete_context);
  return json.end_object();
}

bool append_remaining_runtime_events_json(std::uint64_t generation, std::uint64_t state_epoch,
                                          std::int32_t observation_tick, char *response, std::size_t response_size,
                                          std::size_t *used) {
  if (response == nullptr || used == nullptr) {
    return false;
  }
  // An epoch with zero matching events is still a complete observed window.
  // Refresh from the enclosing rich-observation identity so capability does
  // not remain falsely unavailable until the first card mechanic fires.
  CombatCaptureContext observation_context;
  observation_context.generation = generation;
  observation_context.state_epoch = state_epoch;
  observation_context.tick = observation_tick;
  remaining_refresh_epoch(observation_context);
  CombatCaptureContext current_context;
  if (current_combat_capture_context(&current_context) && current_context.generation == generation &&
      current_context.state_epoch == state_epoch) {
    current_context.tick = observation_tick;
    remaining_runtime_reconcile_pending_removals(current_context);
    remaining_runtime_reconcile_tracked_lifetimes(current_context);
    remaining_runtime_reconcile_pending_spawn_attaches(current_context);
  }
  const bool installed = g_remaining_runtime_hooks_installed.load(std::memory_order_acquire);
  const bool identity = g_remaining_runtime_generation.load(std::memory_order_acquire) == generation &&
                        g_remaining_runtime_state_epoch.load(std::memory_order_acquire) == state_epoch;
  const std::uint64_t next = g_remaining_runtime_next_sequence.load(std::memory_order_acquire);
  const std::uint64_t epoch_first =
      identity ? g_remaining_runtime_epoch_first_sequence.load(std::memory_order_acquire) : next;
  const std::uint64_t retained_oldest =
      next > epoch_first + kRemainingRuntimeRingCapacity ? next - kRemainingRuntimeRingCapacity : epoch_first;
  const std::uint64_t transmit_from = retained_oldest;
  const std::uint64_t overflow = identity ? g_remaining_runtime_overflow_count.load(std::memory_order_acquire) : 0;
  const std::uint64_t rejected = identity ? g_remaining_runtime_rejected_count.load(std::memory_order_acquire) : 0;
  const bool complete = installed && identity && rejected == 0;
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("ok", true);
  json.field("schema", cr_remaining_runtime::kSchema);
  json.field("generation", generation);
  json.field("stateEpoch", state_epoch);
  json.field("observationTick", observation_tick);
  json.field("capacity", kRemainingRuntimeRingCapacity);
  json.field("hookSetAttested", g_remaining_runtime_hooks_attested.load(std::memory_order_acquire));
  json.field("hookSetInstalled", installed);
  json.raw("capability", "{");
  json.field("status", installed && identity ? "derived" : "unavailable");
  json.field("source", "exact-build-native-remaining-hook-ring");
  json.field("validation", "sha-build-id-16-byte-fingerprint-abi-identity");
  json.field("confidence", "high");
  json.field("failClosed", true);
  json.end_object();
  json.raw("ownerRelativeVisibility", "{");
  for (const char *name : {"publicByOwner", "targetableByOwner"}) {
    json.raw(name, "{");
    json.field("status", "unavailable");
    json.field("reason", "no-exact-owner-conditioned-native-producer");
    json.end_object();
  }
  json.end_object();
  json.field("epochFirstSequence", epoch_first);
  json.field("oldestRetainedSequence", retained_oldest);
  json.field("transmitFromSequence", transmit_from);
  json.field("nextSequence", next);
  json.field("overflowCount", overflow);
  json.field("rejectedCount", rejected);
  json.field("sequenceGapBeforeOldest", retained_oldest > epoch_first);
  json.field("complete", complete);
  if (!json.begin_array("events")) {
    return false;
  }
  bool first = true;
  if (identity) {
    for (std::uint64_t sequence = transmit_from; sequence < next; ++sequence) {
      RemainingRingSlot &slot = g_remaining_runtime_ring[(sequence - 1) % kRemainingRuntimeRingCapacity];
      if (slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        return false;
      }
      const RemainingRawRecord record = slot.record;
      if (record.sequence != sequence || record.generation != generation || record.state_epoch != state_epoch ||
          record.tick > observation_tick || slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        return false;
      }
      if ((!first && !append_json(response, response_size, used, ",")) ||
          !append_remaining_record_json(response, response_size, used, record)) {
        return false;
      }
      first = false;
    }
  }
  return append_json(response, response_size, used, "]}");
}

bool verify_remaining_runtime_shared_hook_dependencies() {
  const bool combat_attested = g_combat_hooks_attested.load(std::memory_order_acquire);
  const bool combat_installed = g_combat_hooks_installed.load(std::memory_order_acquire);
  const bool phase_attested = g_phase_runtime_hooks_attested.load(std::memory_order_acquire);
  const bool phase_installed = g_phase_runtime_hooks_installed.load(std::memory_order_acquire);
  struct SharedDependency {
    const char *label;
    const char *provider;
    bool attested;
    bool installed;
  };
  const SharedDependency values[] = {
      {"shared committed spawn", "combat-event", combat_attested, combat_installed},
      {"shared confirmed remove", "combat-event", combat_attested, combat_installed},
      {"shared action dispatch", "phase-runtime", phase_attested, phase_installed},
  };
  bool verified = true;
  for (const SharedDependency &value : values) {
    if (value.attested && value.installed) {
      continue;
    }
    LOGE("remaining-runtime shared hook prerequisite failed "
         "label=%s provider=%s attested=%d installed=%d",
         value.label, value.provider, value.attested ? 1 : 0, value.installed ? 1 : 0);
    verified = false;
  }
  return verified;
}

bool verify_remaining_runtime_hook_fingerprints(std::uintptr_t libg_base) {
  char sha256[65] = {};
  char build_id[41] = {};
  if (!sha256_file(g_libg_path, sha256) || !libg_build_id(build_id) ||
      std::strcmp(sha256, cr_remaining_runtime::kExactLibgSha256) != 0 ||
      std::strcmp(build_id, cr_remaining_runtime::kExactLibgBuildId) != 0) {
    LOGE("remaining-runtime attestation mismatch sha=%s "
         "buildId=%s",
         sha256, build_id);
    return false;
  }
  struct Fingerprint {
    std::uintptr_t offset;
    const std::uint8_t *bytes;
    const char *label;
  };
  // Combat owns spawn/remove and phase owns action-dispatch. Those exact
  // providers have already attested the original prologues and replaced
  // them before remaining-runtime is installed, so reading those three
  // addresses here would compare an inline jump against stale source bytes.
  // Keep every remaining-owned entry and every unmodified caller site under
  // direct fingerprint validation.
  const Fingerprint values[] = {
      {cr_remaining_runtime::kResourceWholeDeltaOffset, cr_remaining_runtime::kResourceWholeDeltaPrologue,
       "resource whole delta"},
      {cr_remaining_runtime::kResourceFixedDeltaOffset, cr_remaining_runtime::kResourceFixedDeltaPrologue,
       "resource fixed delta"},
      {cr_remaining_runtime::kPeriodicResourceCallOffset, cr_remaining_runtime::kPeriodicResourceCallFingerprint,
       "periodic resource caller"},
      {cr_remaining_runtime::kDeathOwnerResourceCallOffset, cr_remaining_runtime::kDeathOwnerResourceCallFingerprint,
       "death-owner resource caller"},
      {cr_remaining_runtime::kDeathOpponentResourceCallOffset,
       cr_remaining_runtime::kDeathOpponentResourceCallFingerprint, "death-opponent resource caller"},
      {cr_remaining_runtime::kCharacterSpawnHelperOffset, cr_remaining_runtime::kCharacterSpawnHelperPrologue,
       "SpawnAttach helper"},
      {cr_remaining_runtime::kSpawnAttachHelperCallOffset, cr_remaining_runtime::kSpawnAttachHelperCallFingerprint,
       "SpawnAttach exact caller"},
      {cr_remaining_runtime::kAreaEffectObjectTickOffset, cr_remaining_runtime::kAreaEffectObjectTickPrologue,
       "area-effect tick producer"},
      {cr_remaining_runtime::kAreaDamageEligibleOffset, cr_remaining_runtime::kAreaDamageEligiblePrologue,
       "contextual area eligibility"},
      {cr_remaining_runtime::kActionWaitUpdateOffset, cr_remaining_runtime::kActionWaitUpdatePrologue,
       "ActionWait update"},
      {cr_remaining_runtime::kActionSpawnToLocationExecuteOffset,
       cr_remaining_runtime::kActionSpawnToLocationExecutePrologue, "ActionSpawnToLocation execute"},
      {cr_remaining_runtime::kActionSpawnMode1ChildCreateOffset,
       cr_remaining_runtime::kActionSpawnMode1ChildCreatePrologue, "ActionSpawnToLocation mode-1 child create"},
      {cr_remaining_runtime::kPendingObjectAddOffset, cr_remaining_runtime::kPendingObjectAddPrologue,
       "pending object add"},
      {cr_remaining_runtime::kActionWaitDispatchCallOffset, cr_remaining_runtime::kActionWaitDispatchCallFingerprint,
       "ActionWait activation edge"},
      {cr_remaining_runtime::kChangeGameObjectDataExecuteOffset,
       cr_remaining_runtime::kChangeGameObjectDataExecutePrologue, "transform execute"},
      {cr_remaining_runtime::kChangeMutationCallOffset, cr_remaining_runtime::kChangeMutationCallFingerprint,
       "transform mutation call"},
  };
  bool verified = verify_remaining_runtime_shared_hook_dependencies();
  for (const Fingerprint &value : values) {
    if (std::memcmp(reinterpret_cast<const void *>(libg_base + value.offset), value.bytes, 16) != 0) {
      LOGE("remaining-runtime fingerprint mismatch "
           "label=%s offset=+0x%zx",
           value.label, static_cast<std::size_t>(value.offset));
      verified = false;
    }
  }
  return verified;
}

bool install_remaining_area_damage_eligible_hook(std::uintptr_t libg_base) {
#if !defined(__aarch64__)
  (void)libg_base;
  return false;
#else
  auto *const target = reinterpret_cast<std::uint8_t *>(libg_base + cr_remaining_runtime::kAreaDamageEligibleOffset);
  if (std::memcmp(target, cr_remaining_runtime::kAreaDamageEligiblePrologue, 16) != 0) {
    return false;
  }
  // f1cb90 starts:
  //   ldr x8,[x0,#0x180]
  //   cbz x8,f1cba0
  //   mov w0,wzr
  //   ret
  // The CBZ is PC-relative. Rebuild it as CBNZ-to-local-false and use an
  // absolute jump for the fall-through body.
  auto *const trampoline = static_cast<std::uint8_t *>(
      mmap(nullptr, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (trampoline == MAP_FAILED) {
    return false;
  }
  std::memcpy(trampoline, target, 4);
  const std::uint8_t cbnz_x8_to_false[4] = {
      0xa8,
      0x00,
      0x00,
      0xb5,
  };
  std::memcpy(trampoline + 4, cbnz_x8_to_false, 4);
  emit_absolute_jump(trampoline + 8, target + 16);
  std::memcpy(trampoline + 24, target + 8, 8);
  __builtin___clear_cache(reinterpret_cast<char *>(trampoline), reinterpret_cast<char *>(trampoline + 32));
  if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
    munmap(trampoline, 32);
    return false;
  }
  g_original_remaining_area_damage_eligible = reinterpret_cast<RemainingAreaDamageEligible>(trampoline);
  emit_absolute_jump(target, reinterpret_cast<const void *>(&remaining_area_damage_eligible_hook));
  __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target + 16));
  make_page_writable(target, PROT_READ | PROT_EXEC);
  return true;
#endif
}

void remaining_refresh_installed_state() {
  const bool installed = g_remaining_runtime_independent_hooks_installed.load(std::memory_order_acquire) &&
                         g_remaining_runtime_area_lifecycle_bound.load(std::memory_order_acquire) &&
                         g_remaining_runtime_tower_aggro_bound.load(std::memory_order_acquire) &&
                         g_remaining_runtime_tower_activation_bound.load(std::memory_order_acquire) &&
                         g_remaining_runtime_lifetime_bound.load(std::memory_order_acquire);
  g_remaining_runtime_hooks_installed.store(installed, std::memory_order_release);
}

void mark_remaining_runtime_shared_hooks_bound(bool area_lifecycle, bool tower_aggro, bool tower_activation,
                                               bool lifetime) {
  g_remaining_runtime_area_lifecycle_bound.store(area_lifecycle, std::memory_order_release);
  g_remaining_runtime_tower_aggro_bound.store(tower_aggro, std::memory_order_release);
  g_remaining_runtime_tower_activation_bound.store(tower_activation, std::memory_order_release);
  g_remaining_runtime_lifetime_bound.store(lifetime, std::memory_order_release);
  remaining_refresh_installed_state();
}

bool install_remaining_runtime_hooks(std::uintptr_t libg_base) {
  const bool attested = verify_remaining_runtime_hook_fingerprints(libg_base);
  g_remaining_runtime_hooks_attested.store(attested, std::memory_order_release);
  if (!attested) {
    return false;
  }
#if !defined(__aarch64__)
  (void)libg_base;
  g_remaining_runtime_independent_hooks_installed.store(false, std::memory_order_release);
  remaining_refresh_installed_state();
  return false;
#else
  bool installed = install_inline_hook(
      libg_base, cr_remaining_runtime::kResourceWholeDeltaOffset, cr_remaining_runtime::kResourceWholeDeltaPrologue,
      reinterpret_cast<const void *>(&remaining_resource_whole_entry),
      reinterpret_cast<void **>(&g_original_remaining_resource_whole), "remaining resource whole");
  installed &= install_inline_hook(
      libg_base, cr_remaining_runtime::kResourceFixedDeltaOffset, cr_remaining_runtime::kResourceFixedDeltaPrologue,
      reinterpret_cast<const void *>(&remaining_resource_fixed_entry),
      reinterpret_cast<void **>(&g_original_remaining_resource_fixed), "remaining resource fixed");
  installed &= install_inline_hook(libg_base, cr_remaining_runtime::kCharacterSpawnHelperOffset,
                                   cr_remaining_runtime::kCharacterSpawnHelperPrologue,
                                   reinterpret_cast<const void *>(&remaining_spawn_helper_entry),
                                   &g_original_remaining_spawn_helper, "remaining SpawnAttach helper");
  installed &= install_inline_hook(
      libg_base, cr_remaining_runtime::kAreaEffectObjectTickOffset, cr_remaining_runtime::kAreaEffectObjectTickPrologue,
      reinterpret_cast<const void *>(&remaining_area_effect_tick_hook),
      reinterpret_cast<void **>(&g_original_remaining_area_effect_tick), "remaining area-effect nested-spawn producer");
  installed &= install_inline_hook(libg_base, cr_remaining_runtime::kActionSpawnToLocationExecuteOffset,
                                   cr_remaining_runtime::kActionSpawnToLocationExecutePrologue,
                                   reinterpret_cast<const void *>(&remaining_action_spawn_to_location_execute_hook),
                                   reinterpret_cast<void **>(&g_original_remaining_action_spawn_to_location_execute),
                                   "remaining ActionSpawnToLocation producer");
  installed &= install_inline_hook(
      libg_base, cr_remaining_runtime::kPendingObjectAddOffset, cr_remaining_runtime::kPendingObjectAddPrologue,
      reinterpret_cast<const void *>(&remaining_pending_object_add_hook),
      reinterpret_cast<void **>(&g_original_remaining_pending_object_add), "remaining pending-object causal bridge");
  installed &= install_inline_hook(libg_base, cr_remaining_runtime::kActionSpawnMode1ChildCreateOffset,
                                   cr_remaining_runtime::kActionSpawnMode1ChildCreatePrologue,
                                   reinterpret_cast<const void *>(&remaining_action_spawn_mode1_child_create_entry),
                                   &g_original_remaining_action_spawn_mode1_child_create,
                                   "remaining action-spawn mode-1 child create");
  installed &= install_inline_hook(libg_base, cr_remaining_runtime::kChangeGameObjectDataExecuteOffset,
                                   cr_remaining_runtime::kChangeGameObjectDataExecutePrologue,
                                   reinterpret_cast<const void *>(&remaining_transform_execute_hook),
                                   reinterpret_cast<void **>(&g_original_remaining_transform_execute),
                                   "remaining transform execute");
  installed &= install_inline_hook(
      libg_base, cr_remaining_runtime::kActionWaitUpdateOffset, cr_remaining_runtime::kActionWaitUpdatePrologue,
      reinterpret_cast<const void *>(&remaining_action_wait_update_hook),
      reinterpret_cast<void **>(&g_original_remaining_action_wait_update), "remaining ActionWait activation context");
  installed &= install_remaining_area_damage_eligible_hook(libg_base);
  g_remaining_runtime_independent_hooks_installed.store(installed, std::memory_order_release);
  if (installed) {
    g_combat_remaining_spawn_observer = &remaining_runtime_note_committed_spawn;
    g_combat_remaining_spawn_provenance_observer = &remaining_runtime_apply_area_tick_spawn_provenance;
    g_combat_remaining_remove_begin_observer = &remaining_runtime_begin_remove;
    g_combat_remaining_remove_end_observer = &remaining_runtime_end_remove;
    g_phase_remaining_tower_aggro_observer = &remaining_runtime_observe_tower_aggro;
    g_phase_remaining_action_observer = &remaining_runtime_observe_action_dispatch;
  }
  mark_remaining_runtime_shared_hooks_bound(
      installed && g_combat_hooks_installed.load(std::memory_order_acquire),
      installed && g_phase_runtime_hooks_installed.load(std::memory_order_acquire),
      installed && g_phase_runtime_hooks_installed.load(std::memory_order_acquire),
      installed && g_combat_hooks_installed.load(std::memory_order_acquire));
  LOGI("exact-build remaining-runtime independent hooks "
       "installed=%d ring=%zu",
       installed ? 1 : 0, kRemainingRuntimeRingCapacity);
  return installed;
#endif
}
