// Exact-build phase hooks for Null's Royale 15.535.13.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc;
//   * cr_replay_probe.cpp supplies read_object_field, segment validation,
//     emit_absolute_jump, make_page_writable, install_inline_hook and logging;
//   * call install_phase_runtime_hooks(libg_base) after combat hooks;
//   * encode copy_phase_runtime_events/read_native_phase_snapshot wherever the
//     rich observation schema is joined.
//
// The hook layer is pointer-free at its public boundary.  Hot callbacks do not
// allocate and do not take the control/observation mutex.

#include "phase_runtime_layout.h"
#include "probe_json.h"
#include "probe_hook.h"

constexpr std::size_t kPhaseRuntimeRingCapacity = 4096;
constexpr std::size_t kPhaseRuntimeCacheCapacity = 512;
constexpr std::int32_t kPhaseRuntimeUnset = (-2147483647 - 1);

enum class PhaseRawKind : std::uint8_t {
  AttackStart = 1,
  AttackRelease = 2,
  EffectApply = 3,
  AttackScale = 4,
  MovementScale = 5,
  EffectiveMovementSpeed = 6,
  MovementDelta = 7,
  TargetReset = 8,
  ClassicChargeReady = 9,
  DeployScale = 10,
};

struct PhaseRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  PhaseRawKind kind = PhaseRawKind::AttackStart;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  CombatEntityFact entity;
  CombatEntityFact target_before;
  CombatEntityFact target_after;
  std::int32_t buff_global_id = kPhaseRuntimeUnset;
  std::int32_t buff_remaining_ms = kPhaseRuntimeUnset;
  std::int32_t speed_multiplier = kPhaseRuntimeUnset;
  std::int32_t hit_speed_multiplier = kPhaseRuntimeUnset;
  std::int32_t input_step = kPhaseRuntimeUnset;
  std::int32_t output_step = kPhaseRuntimeUnset;
  std::int32_t timeline_before = kPhaseRuntimeUnset;
  std::int32_t timeline_after = kPhaseRuntimeUnset;
  std::int32_t classic_charge_before = kPhaseRuntimeUnset;
  std::int32_t classic_charge_after = kPhaseRuntimeUnset;
  bool success = false;
};

struct PhaseRingSlot {
  std::atomic<std::uint64_t> published_sequence{0};
  PhaseRawRecord record;
};

struct PhaseRuntimeCachePayload {
  bool used = false;
  std::uintptr_t object = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t last_attack_scale_input = kPhaseRuntimeUnset;
  std::int32_t last_attack_scale_output = kPhaseRuntimeUnset;
  std::int32_t last_attack_scale_tick = -1;
  std::int32_t last_movement_scale_input = kPhaseRuntimeUnset;
  std::int32_t last_movement_scale_output = kPhaseRuntimeUnset;
  std::int32_t last_movement_scale_tick = -1;
  std::int32_t last_deploy_scale_input = kPhaseRuntimeUnset;
  std::int32_t last_deploy_scale_output = kPhaseRuntimeUnset;
  std::int32_t last_deploy_scale_tick = -1;
  std::int32_t last_effective_speed = kPhaseRuntimeUnset;
  std::int32_t last_effective_speed_tick = -1;
  std::int32_t last_movement_delta = kPhaseRuntimeUnset;
  std::int32_t last_movement_tick = -1;
  std::int32_t last_moving_state = -1;
};

struct PhaseRuntimeCache : PhaseRuntimeCachePayload {
  std::atomic_flag lock = ATOMIC_FLAG_INIT;
};

struct NativePhaseSnapshot {
  bool attack_validated = false;
  bool movement_validated = false;
  bool buffs_validated = false;
  std::int32_t attack_sequence_stage = kPhaseRuntimeUnset;
  std::int32_t attack_timeline_ms = kPhaseRuntimeUnset;
  std::int32_t load_remaining_ms = kPhaseRuntimeUnset;
  std::int32_t deploy_remaining_ms = kPhaseRuntimeUnset;
  std::int32_t deploy_previous_ms = kPhaseRuntimeUnset;
  std::int32_t configured_deploy_time_ms = kPhaseRuntimeUnset;
  std::int32_t hit_speed_ms = kPhaseRuntimeUnset;
  std::int32_t attack_dash_time_ms = kPhaseRuntimeUnset;
  std::int32_t base_movement_speed = kPhaseRuntimeUnset;
  std::int32_t charge_speed_multiplier = kPhaseRuntimeUnset;
  std::int32_t classic_charge_progress = kPhaseRuntimeUnset;
  std::int32_t speed_positive_percent = kPhaseRuntimeUnset;
  std::int32_t speed_negative_magnitude = kPhaseRuntimeUnset;
  std::int32_t hit_speed_positive_percent = kPhaseRuntimeUnset;
  std::int32_t hit_speed_negative_magnitude = kPhaseRuntimeUnset;
  std::int32_t attack_step_input = kPhaseRuntimeUnset;
  std::int32_t attack_step_output = kPhaseRuntimeUnset;
  std::int32_t attack_step_tick = -1;
  std::int32_t movement_step_input = kPhaseRuntimeUnset;
  std::int32_t movement_step_output = kPhaseRuntimeUnset;
  std::int32_t movement_step_tick = -1;
  std::int32_t deploy_step_input = kPhaseRuntimeUnset;
  std::int32_t deploy_step_output = kPhaseRuntimeUnset;
  std::int32_t deploy_step_tick = -1;
  std::int32_t effective_movement_speed = kPhaseRuntimeUnset;
  std::int32_t effective_movement_speed_tick = -1;
  std::int32_t movement_delta = kPhaseRuntimeUnset;
  std::int32_t movement_delta_tick = -1;
  std::int32_t attack_sequence_progress_raw = kPhaseRuntimeUnset;
  std::int32_t attack_sequence_progress_limit = kPhaseRuntimeUnset;
  std::int32_t attack_sequence_decay_remaining_ms = kPhaseRuntimeUnset;
  std::int32_t attack_sequence_decay_duration_ms = kPhaseRuntimeUnset;
};

PhaseRingSlot g_phase_runtime_ring[kPhaseRuntimeRingCapacity];
PhaseRuntimeCache g_phase_runtime_cache[kPhaseRuntimeCacheCapacity];
std::atomic<std::uint64_t> g_phase_runtime_next_sequence{1};
std::atomic<std::uint64_t> g_phase_runtime_epoch_first_sequence{1};
std::atomic<std::uint64_t> g_phase_runtime_generation{0};
std::atomic<std::uint64_t> g_phase_runtime_state_epoch{0};
std::atomic<std::uint64_t> g_phase_runtime_overflow_count{0};
std::atomic<std::uint64_t> g_phase_runtime_rejected_count{0};
std::atomic<bool> g_phase_runtime_hooks_attested{false};
std::atomic<bool> g_phase_runtime_hooks_installed{false};
std::atomic<bool> g_phase_buff_added_hook_installed{false};
std::atomic_flag g_phase_runtime_epoch_lock = ATOMIC_FLAG_INIT;
std::atomic<bool> g_phase_runtime_resetting{false};
std::atomic<std::uint32_t> g_phase_runtime_publishers{0};

using PhaseActionDispatch = void (*)(void *, void *, void *);
using PhaseAttackExecute = std::int32_t (*)(void *, void *, std::int32_t, void *, std::int32_t, std::int32_t,
                                            std::int32_t);
using PhaseTargetSet = void (*)(void *, void *, std::int32_t, std::int32_t, std::int32_t);
using PhaseBuffAdded = void (*)(void *, void *);
using PhaseScaleValue = std::int32_t (*)(void *, std::int32_t);
using PhaseEffectiveSpeed = std::int32_t (*)(void *);
using PhaseMovementDelta = void (*)(void *, std::int32_t, std::int32_t);
using PhaseRemainingActionObserver = void (*)(const void *, const void *, const void *, const void *);
using PhaseRemainingTowerAggroObserver = void (*)(const void *, const void *, const void *, bool);
using PhaseRemainingBuffAddedObserver = void (*)(const void *, const void *);

PhaseActionDispatch g_original_phase_action_dispatch = nullptr;
PhaseAttackExecute g_original_phase_attack_execute = nullptr;
PhaseTargetSet g_original_phase_target_set = nullptr;
PhaseBuffAdded g_original_phase_buff_added = nullptr;
PhaseScaleValue g_original_phase_movement_scale = nullptr;
PhaseScaleValue g_original_phase_attack_scale = nullptr;
PhaseEffectiveSpeed g_original_phase_effective_speed = nullptr;
PhaseMovementDelta g_original_phase_movement_delta = nullptr;
PhaseRemainingActionObserver g_phase_remaining_action_observer = nullptr;
PhaseRemainingTowerAggroObserver g_phase_remaining_tower_aggro_observer = nullptr;
PhaseRemainingBuffAddedObserver g_phase_remaining_buff_added_observer = nullptr;

const char *phase_raw_kind_label(PhaseRawKind kind) {
  switch (kind) {
  case PhaseRawKind::AttackStart:
    return "attack_start";
  case PhaseRawKind::AttackRelease:
    return "attack_release";
  case PhaseRawKind::EffectApply:
    return "effect_apply";
  case PhaseRawKind::AttackScale:
    return "attack_scale";
  case PhaseRawKind::MovementScale:
    return "movement_scale";
  case PhaseRawKind::EffectiveMovementSpeed:
    return "effective_movement_speed";
  case PhaseRawKind::MovementDelta:
    return "movement_delta";
  case PhaseRawKind::TargetReset:
    return "target_reset";
  case PhaseRawKind::ClassicChargeReady:
    return "classic_charge_ready";
  case PhaseRawKind::DeployScale:
    return "deploy_scale";
  }
  return "unknown";
}

std::uintptr_t phase_caller_offset(const void *address) {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
  return address == nullptr || raw < g_libg_base ? 0 : raw - g_libg_base;
}

void reset_phase_runtime_cache_payload(PhaseRuntimeCache *slot) {
  if (slot == nullptr) {
    return;
  }
  static_cast<PhaseRuntimeCachePayload &>(*slot) = {};
}

void commit_phase_runtime_epoch_locked(std::uint64_t generation, std::uint64_t state_epoch, bool clear_cache) {
  // Publish an invalid identity while first-sequence/counters/cache change,
  // so an observation can only see the complete old or complete new epoch.
  g_phase_runtime_generation.store(0, std::memory_order_release);
  g_phase_runtime_state_epoch.store(0, std::memory_order_release);
  const std::uint64_t first = g_phase_runtime_next_sequence.load(std::memory_order_acquire);
  g_phase_runtime_epoch_first_sequence.store(first, std::memory_order_release);
  g_phase_runtime_overflow_count.store(0, std::memory_order_release);
  g_phase_runtime_rejected_count.store(0, std::memory_order_release);
  if (clear_cache) {
    for (PhaseRuntimeCache &slot : g_phase_runtime_cache) {
      while (slot.lock.test_and_set(std::memory_order_acquire)) {
        // Lifecycle begin runs while combat capture is inactive.  A
        // pre-existing callback may finish, but no new callback can
        // acquire a cache entry until the combat epoch is re-armed.
      }
      reset_phase_runtime_cache_payload(&slot);
      slot.lock.clear(std::memory_order_release);
    }
  }
  // Ring storage is reset logically by epochFirstSequence; published slots
  // remain immutable until their globally monotonic sequence is reused.
  g_phase_runtime_state_epoch.store(state_epoch, std::memory_order_release);
  g_phase_runtime_generation.store(generation, std::memory_order_release);
}

void begin_phase_runtime_epoch(std::uint64_t generation, std::uint64_t state_epoch) {
  g_phase_runtime_resetting.store(true, std::memory_order_release);
  while (g_phase_runtime_publishers.load(std::memory_order_acquire) != 0) {
    // Drain records that passed the reset gate before combat capture was
    // deactivated.  A late stale callback is rejected by the identity
    // recheck in publish_phase_record.
  }
  while (g_phase_runtime_epoch_lock.test_and_set(std::memory_order_acquire)) {
    // This is a cold lifecycle boundary, not a hot hook callback.
  }
  commit_phase_runtime_epoch_locked(generation, state_epoch, true);
  g_phase_runtime_epoch_lock.clear(std::memory_order_release);
  g_phase_runtime_resetting.store(false, std::memory_order_release);
}

PhaseRawRecord phase_base_record(const CombatCaptureContext &context, PhaseRawKind kind, std::uintptr_t hook_offset,
                                 std::uintptr_t caller_offset) {
  PhaseRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind = kind;
  record.hook_offset = hook_offset;
  record.caller_offset = caller_offset;
  return record;
}

std::uint64_t publish_phase_record(PhaseRawRecord record) {
  if (g_phase_runtime_resetting.load(std::memory_order_acquire)) {
    return 0;
  }
  g_phase_runtime_publishers.fetch_add(1, std::memory_order_acq_rel);
  const bool accepted = !g_phase_runtime_resetting.load(std::memory_order_acquire) &&
                        g_phase_runtime_generation.load(std::memory_order_acquire) == record.generation &&
                        g_phase_runtime_state_epoch.load(std::memory_order_acquire) == record.state_epoch;
  if (!accepted) {
    g_phase_runtime_publishers.fetch_sub(1, std::memory_order_acq_rel);
    return 0;
  }
  const std::uint64_t sequence = g_phase_runtime_next_sequence.fetch_add(1, std::memory_order_acq_rel);
  record.sequence = sequence;
  const std::uint64_t first = g_phase_runtime_epoch_first_sequence.load(std::memory_order_acquire);
  if (sequence >= first + kPhaseRuntimeRingCapacity) {
    g_phase_runtime_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
  PhaseRingSlot &slot = g_phase_runtime_ring[(sequence - 1) % kPhaseRuntimeRingCapacity];
  slot.published_sequence.store(0, std::memory_order_relaxed);
  slot.record = record;
  slot.published_sequence.store(sequence, std::memory_order_release);
  g_phase_runtime_publishers.fetch_sub(1, std::memory_order_acq_rel);
  return sequence;
}

std::size_t phase_cache_hash(const void *object) {
  std::uintptr_t value = reinterpret_cast<std::uintptr_t>(object);
  value ^= value >> 17;
  value ^= value >> 31;
  return static_cast<std::size_t>(value) % kPhaseRuntimeCacheCapacity;
}

template <typename Callback>
bool update_phase_cache(const CombatCaptureContext &context, const void *object, Callback callback) {
  if (object == nullptr) {
    return false;
  }
  const std::size_t start = phase_cache_hash(object);
  for (std::size_t probe = 0; probe < kPhaseRuntimeCacheCapacity; ++probe) {
    PhaseRuntimeCache &slot = g_phase_runtime_cache[(start + probe) % kPhaseRuntimeCacheCapacity];
    if (slot.lock.test_and_set(std::memory_order_acquire)) {
      continue;
    }
    const bool match = slot.used && slot.object == reinterpret_cast<std::uintptr_t>(object) &&
                       slot.generation == context.generation && slot.state_epoch == context.state_epoch;
    const bool reusable =
        !slot.used || slot.generation != context.generation || slot.state_epoch != context.state_epoch;
    if (!match && !reusable) {
      slot.lock.clear(std::memory_order_release);
      continue;
    }
    if (!match) {
      reset_phase_runtime_cache_payload(&slot);
      slot.used = true;
      slot.object = reinterpret_cast<std::uintptr_t>(object);
      slot.generation = context.generation;
      slot.state_epoch = context.state_epoch;
    }
    callback(slot);
    slot.lock.clear(std::memory_order_release);
    return true;
  }
  g_phase_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
  return false;
}

// Phase-only events still reserve bounded owner membership. Their historical
// attack/reset payload is unused; capacity exhaustion must retain its old guard.
void reserve_phase_cache(const CombatCaptureContext &context, const void *object) {
  update_phase_cache(context, object, [](PhaseRuntimeCache &) {});
}

bool copy_phase_cache(std::uint64_t generation, std::uint64_t state_epoch, const void *object,
                      PhaseRuntimeCache *result) {
  if (object == nullptr || result == nullptr) {
    return false;
  }
  const std::size_t start = phase_cache_hash(object);
  for (std::size_t probe = 0; probe < kPhaseRuntimeCacheCapacity; ++probe) {
    PhaseRuntimeCache &slot = g_phase_runtime_cache[(start + probe) % kPhaseRuntimeCacheCapacity];
    if (slot.lock.test_and_set(std::memory_order_acquire)) {
      continue;
    }
    const bool match = slot.used && slot.object == reinterpret_cast<std::uintptr_t>(object) &&
                       slot.generation == generation && slot.state_epoch == state_epoch;
    if (!match) {
      slot.lock.clear(std::memory_order_release);
      continue;
    }
    static_cast<PhaseRuntimeCachePayload &>(*result) = slot;
    slot.lock.clear(std::memory_order_release);
    return true;
  }
  return false;
}

bool phase_component_at(const void *object, std::int32_t engine_type, void **result) {
  if (object == nullptr || result == nullptr || !process_range_is_readable(object, 0x28)) {
    return false;
  }
  void **const components = read_object_field<void **>(object, cr_phase_runtime::kObjectComponentsOffset);
  const std::int32_t capacity =
      read_object_field<std::int32_t>(object, cr_phase_runtime::kObjectComponentCapacityOffset);
  const std::int32_t count = read_object_field<std::int32_t>(object, cr_phase_runtime::kObjectComponentCountOffset);
  if (components == nullptr || engine_type < 0 || count <= engine_type || capacity < count || capacity > 64 ||
      !process_range_is_readable(components, static_cast<std::size_t>(count) * sizeof(void *))) {
    return false;
  }
  void *const component = components[engine_type];
  if (!process_range_is_readable(component, 0x28) || !combat_component_type_is(component, object, engine_type)) {
    return false;
  }
  *result = component;
  return true;
}

enum class AttackSequenceDecayReadStatus : std::uint8_t {
  NotApplicable = 0,
  Valid = 1,
  Invalid = 2,
};

AttackSequenceDecayReadStatus read_attack_sequence_decay_runtime(const void *object, std::int32_t attack_sequence_stage,
                                                                 NativePhaseSnapshot *result) {
  if (object == nullptr || result == nullptr ||
      !process_range_is_readable(object, cr_phase_runtime::kRuntimeVariableMapOffset +
                                             cr_phase_runtime::kRuntimeVariableCountOffset + sizeof(std::uint64_t))) {
    return AttackSequenceDecayReadStatus::Invalid;
  }
  const auto *const map = static_cast<const std::uint8_t *>(object) + cr_phase_runtime::kRuntimeVariableMapOffset;
  const void *const initial_head = read_object_field<const void *>(map, cr_phase_runtime::kRuntimeVariableHeadOffset);
  const std::uint64_t initial_count =
      read_object_field<std::uint64_t>(map, cr_phase_runtime::kRuntimeVariableCountOffset);
  if (initial_count > cr_phase_runtime::kRuntimeVariableMaximumCount ||
      ((initial_count == 0) != (initial_head == nullptr))) {
    return AttackSequenceDecayReadStatus::Invalid;
  }
  if (initial_count == 0) {
    return AttackSequenceDecayReadStatus::NotApplicable;
  }

  const void *seen_nodes[cr_phase_runtime::kRuntimeVariableMaximumCount] = {};
  std::int32_t seen_keys[cr_phase_runtime::kRuntimeVariableMaximumCount] = {};
  const void *node = initial_head;
  bool progress_known = false;
  bool decay_known = false;
  std::int32_t progress = 0;
  std::int32_t decay = 0;
  for (std::uint64_t index = 0; index < initial_count; ++index) {
    if (node == nullptr || !process_range_is_readable(node, cr_phase_runtime::kRuntimeVariableNodeSize)) {
      return AttackSequenceDecayReadStatus::Invalid;
    }
    for (std::uint64_t prior = 0; prior < index; ++prior) {
      if (seen_nodes[prior] == node) {
        return AttackSequenceDecayReadStatus::Invalid;
      }
    }
    const std::int64_t hash = read_object_field<std::int64_t>(node, cr_phase_runtime::kRuntimeVariableNodeHashOffset);
    const std::int32_t key = read_object_field<std::int32_t>(node, cr_phase_runtime::kRuntimeVariableNodeKeyOffset);
    const std::int32_t value = read_object_field<std::int32_t>(node, cr_phase_runtime::kRuntimeVariableNodeValueOffset);
    if (hash != static_cast<std::int64_t>(key)) {
      return AttackSequenceDecayReadStatus::Invalid;
    }
    for (std::uint64_t prior = 0; prior < index; ++prior) {
      if (seen_keys[prior] == key) {
        return AttackSequenceDecayReadStatus::Invalid;
      }
    }
    seen_nodes[index] = node;
    seen_keys[index] = key;
    if (key == cr_phase_runtime::kAttackSequenceProgressVariableKey) {
      progress_known = true;
      progress = value;
    } else if (key == cr_phase_runtime::kAttackSequenceDecayVariableKey) {
      decay_known = true;
      decay = value;
    }
    node = read_object_field<const void *>(node, cr_phase_runtime::kRuntimeVariableNodeNextOffset);
  }
  if (node != nullptr ||
      read_object_field<const void *>(map, cr_phase_runtime::kRuntimeVariableHeadOffset) != initial_head ||
      read_object_field<std::uint64_t>(map, cr_phase_runtime::kRuntimeVariableCountOffset) != initial_count) {
    return AttackSequenceDecayReadStatus::Invalid;
  }
  if (!progress_known && !decay_known) {
    return AttackSequenceDecayReadStatus::NotApplicable;
  }
  if (progress_known) {
    if (progress < 0 || progress > cr_phase_runtime::kAttackSequenceProgressLimit ||
        cr_phase_runtime::attack_sequence_stage_from_progress(progress) != attack_sequence_stage) {
      return AttackSequenceDecayReadStatus::Invalid;
    }
    result->attack_sequence_progress_raw = progress;
    result->attack_sequence_progress_limit = cr_phase_runtime::kAttackSequenceProgressLimit;
  }
  if (decay_known) {
    if (decay < 0 || decay > cr_phase_runtime::kAttackSequenceDecayDurationMs) {
      return AttackSequenceDecayReadStatus::Invalid;
    }
    result->attack_sequence_decay_remaining_ms = decay;
    result->attack_sequence_decay_duration_ms = cr_phase_runtime::kAttackSequenceDecayDurationMs;
  }
  return AttackSequenceDecayReadStatus::Valid;
}

bool read_native_phase_snapshot(const void *object, std::uint64_t generation, std::uint64_t state_epoch,
                                std::int32_t observation_tick, NativePhaseSnapshot *result) {
  if (result == nullptr) {
    return false;
  }
  *result = NativePhaseSnapshot{};
  void *attack = nullptr;
  if (!phase_component_at(object, 0, &attack)) {
    return false;
  }
  const void *const data = read_object_field<const void *>(object, cr_phase_runtime::kObjectDataOffset);
  if (data == nullptr || !process_range_is_readable(data, 0x6a4) || !process_range_is_readable(object, 0x164)) {
    return false;
  }
  const std::int32_t stage = read_object_field<std::int32_t>(attack, cr_phase_runtime::kAttackSequenceStageOffset);
  const std::int32_t timeline = read_object_field<std::int32_t>(attack, cr_phase_runtime::kAttackTimelineOffset);
  const std::int32_t load = read_object_field<std::int32_t>(attack, cr_phase_runtime::kAttackLoadRemainingOffset);
  const std::int32_t deploy = read_object_field<std::int32_t>(object, cr_phase_runtime::kObjectDeployRemainingOffset);
  const std::int32_t deploy_previous =
      read_object_field<std::int32_t>(object, cr_phase_runtime::kObjectDeployPreviousOffset);
  if (stage < -1 || stage > 255 || timeline < 0 || timeline > 3600000 || load < 0 || load > 3600000 || deploy < 0 ||
      deploy > 3600000 || deploy_previous < 0 || deploy_previous > 3600000) {
    return false;
  }
  result->attack_validated = true;
  result->attack_sequence_stage = stage;
  result->attack_timeline_ms = timeline;
  result->load_remaining_ms = load;
  result->deploy_remaining_ms = deploy;
  result->deploy_previous_ms = deploy_previous;
  result->configured_deploy_time_ms =
      read_object_field<std::int32_t>(data, cr_phase_runtime::kCharacterDeployTimeOffset);
  result->hit_speed_ms = read_object_field<std::int32_t>(data, cr_phase_runtime::kCharacterHitSpeedOffset);
  result->attack_dash_time_ms = read_object_field<std::int32_t>(data, cr_phase_runtime::kCharacterAttackDashTimeOffset);
  result->base_movement_speed = read_object_field<std::int32_t>(data, cr_phase_runtime::kCharacterSpeedOffset);
  result->charge_speed_multiplier =
      read_object_field<std::int32_t>(data, cr_phase_runtime::kCharacterChargeSpeedMultiplierOffset);
  if (read_attack_sequence_decay_runtime(object, stage, result) == AttackSequenceDecayReadStatus::Invalid) {
    return false;
  }

  void *movement = nullptr;
  if (phase_component_at(object, 1, &movement)) {
    const std::int32_t progress =
        read_object_field<std::int32_t>(movement, cr_phase_runtime::kClassicChargeProgressOffset);
    if (cr_phase_runtime::valid_classic_charge_progress(progress)) {
      result->movement_validated = true;
      result->classic_charge_progress = progress;
    }
  }

  void *buffs = nullptr;
  if (phase_component_at(object, 3, &buffs)) {
    void **const entries = read_object_field<void **>(buffs, cr_phase_runtime::kBuffEntriesOffset);
    const std::int32_t capacity = read_object_field<std::int32_t>(buffs, cr_phase_runtime::kBuffEntryCapacityOffset);
    const std::int32_t count = read_object_field<std::int32_t>(buffs, cr_phase_runtime::kBuffEntryCountOffset);
    if (count >= 0 && capacity >= count && capacity <= 64 &&
        (count == 0 || (entries != nullptr &&
                        process_range_is_readable(entries, static_cast<std::size_t>(count) * sizeof(void *))))) {
      cr_phase_runtime::EffectScale speed{};
      cr_phase_runtime::EffectScale hit_speed{};
      bool valid = true;
      for (std::int32_t index = 0; index < count; ++index) {
        const void *const entry = entries[index];
        if (entry == nullptr || !process_range_is_readable(entry, 0x48)) {
          valid = false;
          break;
        }
        const void *const asset = read_object_field<const void *>(entry, cr_phase_runtime::kBuffEntryAssetOffset);
        if (asset == nullptr || !process_range_is_readable(asset, 0xe8)) {
          valid = false;
          break;
        }
        const std::int32_t speed_value =
            read_object_field<std::int32_t>(asset, cr_phase_runtime::kBuffAssetSpeedMultiplierOffset);
        const std::int32_t hit_value =
            read_object_field<std::int32_t>(asset, cr_phase_runtime::kBuffAssetHitSpeedMultiplierOffset);
        cr_phase_runtime::accumulate_effect_multiplier(speed, speed_value);
        cr_phase_runtime::accumulate_effect_multiplier(hit_speed, hit_value);
      }
      if (valid) {
        result->buffs_validated = true;
        result->speed_positive_percent = speed.positive_percent;
        result->speed_negative_magnitude = speed.negative_magnitude;
        result->hit_speed_positive_percent = hit_speed.positive_percent;
        result->hit_speed_negative_magnitude = hit_speed.negative_magnitude;
      }
    }
  }

  PhaseRuntimeCache cached;
  if (copy_phase_cache(generation, state_epoch, object, &cached)) {
    result->attack_step_input = cached.last_attack_scale_input;
    result->attack_step_output = cached.last_attack_scale_output;
    result->attack_step_tick = cached.last_attack_scale_tick;
    result->movement_step_input = cached.last_movement_scale_input;
    result->movement_step_output = cached.last_movement_scale_output;
    result->movement_step_tick = cached.last_movement_scale_tick;
    result->deploy_step_input = cached.last_deploy_scale_input;
    result->deploy_step_output = cached.last_deploy_scale_output;
    result->deploy_step_tick = cached.last_deploy_scale_tick;
    result->effective_movement_speed = cached.last_effective_speed;
    result->effective_movement_speed_tick = cached.last_effective_speed_tick;
    result->movement_delta = cached.last_movement_delta;
    result->movement_delta_tick = cached.last_movement_tick;
  }
  if (result->movement_validated && g_original_phase_effective_speed != nullptr && observation_tick >= 0) {
    // The stock presentation controller may query effective speed even
    // when logic stepping does not.  Capture the authoritative getter at
    // observation time so headless and native-render do not depend on
    // those incidental presentation calls.
    const std::int32_t effective_speed = g_original_phase_effective_speed(const_cast<void *>(object));
    if (effective_speed >= 0 && effective_speed <= 100000) {
      result->effective_movement_speed = effective_speed;
      result->effective_movement_speed_tick = observation_tick;
    }
  }
  return true;
}

void phase_action_dispatch_hook(void *owner, void *target, void *action) {
  // The stock function's first instruction is a PC-relative cbz.  The
  // specialized installer relocates only the following 12-byte stack
  // prologue, so the null branch must be preserved here.
  if (action == nullptr) {
    return;
  }
  const void *const caller_address = __builtin_return_address(0);
  const std::uintptr_t caller = phase_caller_offset(caller_address);
  if (g_phase_remaining_action_observer != nullptr) {
    g_phase_remaining_action_observer(owner, target, action, caller_address);
  }
  g_original_phase_action_dispatch(owner, target, action);
  if (caller != cr_phase_runtime::kAttackStartReturnOffset &&
      caller != cr_phase_runtime::kClassicChargeActionReturnOffset) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  const PhaseRawKind kind = caller == cr_phase_runtime::kAttackStartReturnOffset ? PhaseRawKind::AttackStart
                                                                                 : PhaseRawKind::ClassicChargeReady;
  PhaseRawRecord record = phase_base_record(context, kind, cr_phase_runtime::kActionDispatchOffset, caller);
  capture_combat_entity(owner, context.object_manager, &record.entity);
  if (!record.entity.validated || !record.entity.present) {
    return;
  }
  reserve_phase_cache(context, owner);
  publish_phase_record(record);
}

std::int32_t phase_attack_execute_hook(void *component, void *target, std::int32_t target_index, void *auxiliary,
                                       std::int32_t flag_a, std::int32_t flag_b, std::int32_t final_target) {
  const std::uintptr_t caller = phase_caller_offset(__builtin_return_address(0));
  const std::int32_t result =
      g_original_phase_attack_execute(component, target, target_index, auxiliary, flag_a, flag_b, final_target);
  if (caller != cr_phase_runtime::kPrimaryAttackReturnOffset &&
      caller != cr_phase_runtime::kAdditionalAttackReturnOffset) {
    return result;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return result;
  }
  const void *const owner = component == nullptr
                                ? nullptr
                                : read_object_field<const void *>(component, cr_phase_runtime::kComponentOwnerOffset);
  PhaseRawRecord record =
      phase_base_record(context, PhaseRawKind::AttackRelease, cr_phase_runtime::kAttackExecuteOffset, caller);
  record.success = (result & 1) != 0;
  capture_combat_entity(owner, context.object_manager, &record.entity);
  capture_combat_entity(target, context.object_manager, &record.target_after);
  if (record.entity.validated && record.entity.present) {
    if (caller == cr_phase_runtime::kPrimaryAttackReturnOffset && record.success) {
      reserve_phase_cache(context, owner);
    }
    publish_phase_record(record);
  }
  return result;
}

void phase_buff_added_hook(void *component, void *entry) {
  const std::uintptr_t caller_offset = phase_caller_offset(__builtin_return_address(0));
  const bool visibility_enabled = g_visibility_runtime_transition_hooks_installed.load(std::memory_order_acquire);
  VisibilityMutationContext visibility_before;
  const bool visibility_prepared =
      visibility_enabled &&
      visibility_prepare_buff_transition(cr_visibility_runtime::kBuffApplyOffset, component, entry, &visibility_before);
  if (g_phase_remaining_buff_added_observer != nullptr) {
    g_phase_remaining_buff_added_observer(component, entry);
  }
  const void *const owner = component == nullptr
                                ? nullptr
                                : read_object_field<const void *>(component, cr_phase_runtime::kComponentOwnerOffset);
  const void *const asset =
      entry == nullptr ? nullptr : read_object_field<const void *>(entry, cr_phase_runtime::kBuffEntryAssetOffset);
  const bool readable = entry != nullptr && asset != nullptr && process_range_is_readable(entry, 0x48) &&
                        process_range_is_readable(asset, 0xe8);
  std::int32_t buff_id = kPhaseRuntimeUnset;
  std::int32_t remaining = kPhaseRuntimeUnset;
  std::int32_t speed = kPhaseRuntimeUnset;
  std::int32_t hit_speed = kPhaseRuntimeUnset;
  if (readable) {
    buff_id = read_object_field<std::int32_t>(asset, cr_phase_runtime::kBuffAssetGlobalIdOffset);
    remaining = read_object_field<std::int32_t>(entry, cr_phase_runtime::kBuffEntryRemainingOffset);
    speed = read_object_field<std::int32_t>(asset, cr_phase_runtime::kBuffAssetSpeedMultiplierOffset);
    hit_speed = read_object_field<std::int32_t>(asset, cr_phase_runtime::kBuffAssetHitSpeedMultiplierOffset);
  }
  g_original_phase_buff_added(component, entry);
  if (visibility_enabled) {
    visibility_commit_buff_transition(visibility_prepared, visibility_before, caller_offset);
  }
  CombatCaptureContext context;
  if (!readable || !current_combat_capture_context(&context)) {
    return;
  }
  PhaseRawRecord record =
      phase_base_record(context, PhaseRawKind::EffectApply, cr_phase_runtime::kBuffAddedOffset, caller_offset);
  record.buff_global_id = buff_id;
  record.buff_remaining_ms = remaining;
  record.speed_multiplier = speed;
  record.hit_speed_multiplier = hit_speed;
  capture_combat_entity(owner, context.object_manager, &record.entity);
  if (!record.entity.validated || !record.entity.present) {
    return;
  }
  reserve_phase_cache(context, owner);
  publish_phase_record(record);
}

std::int32_t phase_attack_scale_hook(void *component, std::int32_t input) {
  const std::uintptr_t caller = phase_caller_offset(__builtin_return_address(0));
  const std::int32_t result = g_original_phase_attack_scale(component, input);
  if (caller != cr_phase_runtime::kAttackScaleReturnOffset) {
    return result;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return result;
  }
  const void *const owner = component == nullptr
                                ? nullptr
                                : read_object_field<const void *>(component, cr_phase_runtime::kComponentOwnerOffset);
  bool changed = false;
  update_phase_cache(context, owner, [&](PhaseRuntimeCache &cache) {
    changed = cache.last_attack_scale_input != input || cache.last_attack_scale_output != result;
    cache.last_attack_scale_input = input;
    cache.last_attack_scale_output = result;
    cache.last_attack_scale_tick = context.tick;
  });
  if (changed) {
    PhaseRawRecord record =
        phase_base_record(context, PhaseRawKind::AttackScale, cr_phase_runtime::kAttackScaleOffset, caller);
    record.input_step = input;
    record.output_step = result;
    capture_combat_entity(owner, context.object_manager, &record.entity);
    if (record.entity.validated && record.entity.present) {
      publish_phase_record(record);
    }
  }
  return result;
}

std::int32_t phase_movement_scale_hook(void *component, std::int32_t input) {
  const std::uintptr_t caller = phase_caller_offset(__builtin_return_address(0));
  const std::int32_t result = g_original_phase_movement_scale(component, input);
  PhaseRawKind kind;
  if (caller == cr_phase_runtime::kDeployScaleReturnOffset) {
    kind = PhaseRawKind::DeployScale;
  } else if (caller == cr_phase_runtime::kMovementTickScaleReturnOffset ||
             caller == cr_phase_runtime::kEffectiveSpeedEffectReturnOffset) {
    kind = PhaseRawKind::MovementScale;
  } else {
    return result;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return result;
  }
  const void *const owner = component == nullptr
                                ? nullptr
                                : read_object_field<const void *>(component, cr_phase_runtime::kComponentOwnerOffset);
  bool changed = false;
  update_phase_cache(context, owner, [&](PhaseRuntimeCache &cache) {
    if (kind == PhaseRawKind::DeployScale) {
      changed = cache.last_deploy_scale_input != input || cache.last_deploy_scale_output != result;
      cache.last_deploy_scale_input = input;
      cache.last_deploy_scale_output = result;
      cache.last_deploy_scale_tick = context.tick;
    } else {
      changed = cache.last_movement_scale_input != input || cache.last_movement_scale_output != result;
      cache.last_movement_scale_input = input;
      cache.last_movement_scale_output = result;
      cache.last_movement_scale_tick = context.tick;
    }
  });
  if (changed) {
    PhaseRawRecord record = phase_base_record(context, kind, cr_phase_runtime::kMovementScaleOffset, caller);
    record.input_step = input;
    record.output_step = result;
    capture_combat_entity(owner, context.object_manager, &record.entity);
    if (record.entity.validated && record.entity.present) {
      publish_phase_record(record);
    }
  }
  return result;
}

std::int32_t phase_effective_speed_hook(void *object) {
  const std::int32_t result = g_original_phase_effective_speed(object);
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return result;
  }
  bool changed = false;
  update_phase_cache(context, object, [&](PhaseRuntimeCache &cache) {
    changed = cache.last_effective_speed != result;
    cache.last_effective_speed = result;
    cache.last_effective_speed_tick = context.tick;
  });
  if (changed) {
    PhaseRawRecord record = phase_base_record(context, PhaseRawKind::EffectiveMovementSpeed,
                                              cr_phase_runtime::kEffectiveMovementSpeedOffset,
                                              phase_caller_offset(__builtin_return_address(0)));
    record.output_step = result;
    capture_combat_entity(object, context.object_manager, &record.entity);
    if (record.entity.validated && record.entity.present) {
      publish_phase_record(record);
    }
  }
  return result;
}

void phase_movement_delta_hook(void *component, std::int32_t accepted_delta, std::int32_t flag) {
  const void *const owner = component == nullptr
                                ? nullptr
                                : read_object_field<const void *>(component, cr_phase_runtime::kComponentOwnerOffset);
  const std::int32_t before =
      component == nullptr ? kPhaseRuntimeUnset
                           : read_object_field<std::int32_t>(component, cr_phase_runtime::kClassicChargeProgressOffset);
  g_original_phase_movement_delta(component, accepted_delta, flag);
  const std::int32_t after =
      component == nullptr ? kPhaseRuntimeUnset
                           : read_object_field<std::int32_t>(component, cr_phase_runtime::kClassicChargeProgressOffset);
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  bool movement_changed = false;
  update_phase_cache(context, owner, [&](PhaseRuntimeCache &cache) {
    const std::int32_t moving = accepted_delta > 0 ? 1 : 0;
    movement_changed = cache.last_moving_state != moving;
    cache.last_moving_state = moving;
    cache.last_movement_delta = accepted_delta;
    cache.last_movement_tick = context.tick;
  });
  if (movement_changed) {
    PhaseRawRecord record =
        phase_base_record(context, PhaseRawKind::MovementDelta, cr_phase_runtime::kMovementDeltaOffset,
                          phase_caller_offset(__builtin_return_address(0)));
    record.input_step = accepted_delta;
    record.classic_charge_before = before;
    record.classic_charge_after = after;
    capture_combat_entity(owner, context.object_manager, &record.entity);
    if (record.entity.validated && record.entity.present) {
      publish_phase_record(record);
    }
  }
}

void phase_target_set_hook(void *component, void *new_target, std::int32_t flag_a, std::int32_t flag_b,
                           std::int32_t flag_c) {
  const void *const owner = component == nullptr
                                ? nullptr
                                : read_object_field<const void *>(component, cr_phase_runtime::kComponentOwnerOffset);
  const void *const old_target =
      component == nullptr ? nullptr
                           : read_object_field<const void *>(component, cr_phase_runtime::kAttackTargetOffset);
  const std::int32_t before = component == nullptr
                                  ? kPhaseRuntimeUnset
                                  : read_object_field<std::int32_t>(component, cr_phase_runtime::kAttackTimelineOffset);
  g_original_phase_target_set(component, new_target, flag_a, flag_b, flag_c);
  const void *const after_target =
      component == nullptr ? nullptr
                           : read_object_field<const void *>(component, cr_phase_runtime::kAttackTargetOffset);
  const std::int32_t after = component == nullptr
                                 ? kPhaseRuntimeUnset
                                 : read_object_field<std::int32_t>(component, cr_phase_runtime::kAttackTimelineOffset);
  if (old_target != after_target && g_phase_remaining_tower_aggro_observer != nullptr) {
    const bool component_validated =
        component != nullptr && owner != nullptr && combat_component_type_is(component, owner, 0);
    g_phase_remaining_tower_aggro_observer(owner, old_target, after_target, component_validated);
  }
  if (before <= 0 || after != 0) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  PhaseRawRecord record = phase_base_record(context, PhaseRawKind::TargetReset, cr_phase_runtime::kTargetSetOffset,
                                            phase_caller_offset(__builtin_return_address(0)));
  record.timeline_before = before;
  record.timeline_after = after;
  capture_combat_entity(owner, context.object_manager, &record.entity);
  capture_combat_entity(old_target, context.object_manager, &record.target_before);
  capture_combat_entity(after_target, context.object_manager, &record.target_after);
  if (!record.entity.validated || !record.entity.present) {
    return;
  }
  reserve_phase_cache(context, owner);
  publish_phase_record(record);
}

bool append_native_phase_snapshot_json(char *response, std::size_t response_size, std::size_t *used,
                                       const NativePhaseSnapshot &value) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("schema", "native-phase-object.v1");
  json.field("attackValidated", value.attack_validated);
  json.field("movementValidated", value.movement_validated);
  json.field("buffsValidated", value.buffs_validated);
  json.nullable_ints(
      {
          {"attackSequenceStage", value.attack_sequence_stage},
          {"attackTimelineMs", value.attack_timeline_ms},
          {"loadRemainingMs", value.load_remaining_ms},
          {"deployRemainingMs", value.deploy_remaining_ms},
          {"deployPreviousMs", value.deploy_previous_ms},
          {"configuredDeployTimeMs", value.configured_deploy_time_ms},
          {"hitSpeedMs", value.hit_speed_ms},
          {"attackDashTimeMs", value.attack_dash_time_ms},
          {"baseMovementSpeed", value.base_movement_speed},
          {"chargeSpeedMultiplier", value.charge_speed_multiplier},
          {"classicChargeProgress", value.classic_charge_progress},
          {"speedPositivePercent", value.speed_positive_percent},
          {"speedNegativeMagnitude", value.speed_negative_magnitude},
          {"hitSpeedPositivePercent", value.hit_speed_positive_percent},
          {"hitSpeedNegativeMagnitude", value.hit_speed_negative_magnitude},
          {"attackStepInput", value.attack_step_input},
          {"attackStepOutput", value.attack_step_output},
          {"attackStepTick", value.attack_step_tick},
          {"movementStepInput", value.movement_step_input},
          {"movementStepOutput", value.movement_step_output},
          {"movementStepTick", value.movement_step_tick},
          {"deployStepInput", value.deploy_step_input},
          {"deployStepOutput", value.deploy_step_output},
          {"deployStepTick", value.deploy_step_tick},
          {"effectiveMovementSpeed", value.effective_movement_speed},
          {"effectiveMovementSpeedTick", value.effective_movement_speed_tick},
          {"movementDelta", value.movement_delta},
          {"movementDeltaTick", value.movement_delta_tick},
          {"attackSequenceProgressRaw", value.attack_sequence_progress_raw},
          {"attackSequenceProgressLimit", value.attack_sequence_progress_limit},
          {"attackSequenceDecayRemainingMs", value.attack_sequence_decay_remaining_ms},
          {"attackSequenceDecayDurationMs", value.attack_sequence_decay_duration_ms},
      },
      kPhaseRuntimeUnset);
  return json.end_object();
}

bool append_phase_record(JsonWriter &json, const PhaseRawRecord &record) {
  json.begin_object();
  json.field("sequence", record.sequence);
  json.field("tick", record.tick);
  json.field("kind", phase_raw_kind_label(record.kind));
  json.field("hookOffset", record.hook_offset);
  json.nullable("callerOffset", record.caller_offset, record.caller_offset != 0);
  json.record("entity", append_combat_entity_fact, record.entity);
  json.record("targetBefore", append_combat_entity_fact, record.target_before);
  json.record("targetAfter", append_combat_entity_fact, record.target_after);
  json.nullable("buffGlobalId", static_cast<std::uint32_t>(record.buff_global_id),
                record.buff_global_id != kPhaseRuntimeUnset);
  json.nullable_ints(
      {
          {"buffRemainingMs", record.buff_remaining_ms},
          {"speedMultiplier", record.speed_multiplier},
          {"hitSpeedMultiplier", record.hit_speed_multiplier},
          {"inputStep", record.input_step},
          {"outputStep", record.output_step},
          {"timelineBefore", record.timeline_before},
          {"timelineAfter", record.timeline_after},
          {"classicChargeBefore", record.classic_charge_before},
          {"classicChargeAfter", record.classic_charge_after},
      },
      kPhaseRuntimeUnset);
  json.field("success", record.success);
  return json.end_object();
}

bool append_phase_runtime_events_json(std::uint64_t generation, std::uint64_t state_epoch,
                                      std::int32_t observation_tick, char *response, std::size_t response_size,
                                      std::size_t *used) {
  if (response == nullptr || used == nullptr) {
    return false;
  }
  const bool installed = g_phase_runtime_hooks_installed.load(std::memory_order_acquire);
  const bool identity = g_phase_runtime_generation.load(std::memory_order_acquire) == generation &&
                        g_phase_runtime_state_epoch.load(std::memory_order_acquire) == state_epoch;
  const std::uint64_t next = g_phase_runtime_next_sequence.load(std::memory_order_acquire);
  const std::uint64_t epoch_first =
      identity ? g_phase_runtime_epoch_first_sequence.load(std::memory_order_acquire) : next;
  const std::uint64_t retained_oldest =
      next > epoch_first + kPhaseRuntimeRingCapacity ? next - kPhaseRuntimeRingCapacity : epoch_first;
  const std::uint64_t transmit_from = retained_oldest;
  const std::uint64_t overflow = identity ? g_phase_runtime_overflow_count.load(std::memory_order_acquire) : 0;
  const std::uint64_t rejected = identity ? g_phase_runtime_rejected_count.load(std::memory_order_acquire) : 0;
  const bool complete = installed && identity && rejected == 0;
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("ok", true);
  json.field("schema", cr_phase_runtime::kSchema);
  json.field("generation", generation);
  json.field("stateEpoch", state_epoch);
  json.field("observationTick", observation_tick);
  json.field("capacity", kPhaseRuntimeRingCapacity);
  json.field("hookSetAttested", g_phase_runtime_hooks_attested.load(std::memory_order_acquire));
  json.field("hookSetInstalled", installed);
  json.begin_object("capability");
  json.field("status", installed && identity ? "derived" : "unavailable");
  json.field("source", "exact-build-native-phase-hook-ring");
  json.field("validation", "sha-build-id-prologue-abi-bounded-identity");
  json.field("confidence", "high");
  json.field("failClosed", true);
  json.end_object();
  json.field("epochFirstSequence", epoch_first);
  json.field("oldestRetainedSequence", retained_oldest);
  json.field("transmitFromSequence", transmit_from);
  json.field("nextSequence", next);
  json.field("overflowCount", overflow);
  json.field("rejectedCount", rejected);
  json.field("sequenceGapBeforeOldest", retained_oldest > epoch_first);
  json.field("complete", complete);
  json.begin_array("events");
  if (!json.good()) {
    return false;
  }
  bool first = true;
  if (identity) {
    for (std::uint64_t sequence = transmit_from; sequence < next; ++sequence) {
      PhaseRingSlot &slot = g_phase_runtime_ring[(sequence - 1) % kPhaseRuntimeRingCapacity];
      if (slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        return false;
      }
      const PhaseRawRecord record = slot.record;
      if (record.sequence != sequence || record.generation != generation || record.state_epoch != state_epoch ||
          record.tick > observation_tick || slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        return false;
      }
      if (!first) {
        json.append(",");
      }
      if (!append_phase_record(json, record)) {
        return false;
      }
      first = false;
    }
  }
  json.end_array();
  return json.end_object();
}

bool verify_phase_runtime_hook_fingerprints(std::uintptr_t libg_base, const NativeHookSpec *hooks, std::size_t count) {
  return verify_native_binary(g_libg_path, cr_phase_runtime::kExpectedLibgSha256,
                              cr_phase_runtime::kExpectedLibgBuildId) &&
         verify_native_hook_table(libg_base, hooks, count);
}

bool install_phase_action_dispatch_hook(std::uintptr_t libg_base) {
#if !defined(__aarch64__)
  (void)libg_base;
  return false;
#else
  auto *const target = reinterpret_cast<std::uint8_t *>(libg_base + cr_phase_runtime::kActionDispatchOffset);
  if (std::memcmp(target, cr_phase_runtime::kActionDispatchPrologue, 16) != 0) {
    return false;
  }
  // f23110 begins `cbz x2,f23178`; blindly copying that instruction would
  // branch relative to the trampoline.  Preserve the null branch in the
  // replacement and relocate only target+4..+16, then jump to target+16.
  void *const trampoline = mmap(nullptr, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (trampoline == MAP_FAILED) {
    return false;
  }
  std::memcpy(trampoline, target + 4, 12);
  emit_absolute_jump(static_cast<std::uint8_t *>(trampoline) + 12, target + 16);
  __builtin___clear_cache(static_cast<char *>(trampoline), static_cast<char *>(trampoline) + 32);
  if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
    munmap(trampoline, 32);
    return false;
  }
  g_original_phase_action_dispatch = reinterpret_cast<PhaseActionDispatch>(trampoline);
  emit_absolute_jump(target, reinterpret_cast<const void *>(&phase_action_dispatch_hook));
  __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target + 16));
  make_page_writable(target, PROT_READ | PROT_EXEC);
  return true;
#endif
}

bool install_phase_runtime_hooks(std::uintptr_t libg_base) {
  const NativeHookSpec hooks[] = {
      native_hook(cr_phase_runtime::kActionDispatchOffset, cr_phase_runtime::kActionDispatchPrologue,
                  &phase_action_dispatch_hook, &g_original_phase_action_dispatch, "phase action dispatch"),
      native_hook(cr_phase_runtime::kAttackExecuteOffset, cr_phase_runtime::kAttackExecutePrologue,
                  &phase_attack_execute_hook, &g_original_phase_attack_execute, "phase attack execute"),
      native_hook(cr_phase_runtime::kTargetSetOffset, cr_phase_runtime::kTargetSetPrologue, &phase_target_set_hook,
                  &g_original_phase_target_set, "phase target reset"),
      native_hook(cr_phase_runtime::kBuffAddedOffset, cr_phase_runtime::kBuffAddedPrologue, &phase_buff_added_hook,
                  &g_original_phase_buff_added, "phase Buff add", &g_phase_buff_added_hook_installed),
      native_hook(cr_phase_runtime::kMovementScaleOffset, cr_phase_runtime::kMovementScalePrologue,
                  &phase_movement_scale_hook, &g_original_phase_movement_scale, "phase movement scale"),
      native_hook(cr_phase_runtime::kAttackScaleOffset, cr_phase_runtime::kAttackScalePrologue,
                  &phase_attack_scale_hook, &g_original_phase_attack_scale, "phase attack scale"),
      native_hook(cr_phase_runtime::kEffectiveMovementSpeedOffset, cr_phase_runtime::kEffectiveMovementSpeedPrologue,
                  &phase_effective_speed_hook, &g_original_phase_effective_speed, "phase effective movement speed"),
      native_hook(cr_phase_runtime::kMovementDeltaOffset, cr_phase_runtime::kMovementDeltaPrologue,
                  &phase_movement_delta_hook, &g_original_phase_movement_delta, "phase movement delta"),
  };
  constexpr std::size_t count = sizeof(hooks) / sizeof(hooks[0]);
  const bool attested = verify_phase_runtime_hook_fingerprints(libg_base, hooks, count);
  g_phase_runtime_hooks_attested.store(attested, std::memory_order_release);
  if (!attested) {
    g_phase_buff_added_hook_installed.store(false, std::memory_order_release);
    return false;
  }
  bool installed = install_phase_action_dispatch_hook(libg_base);
  installed &= install_native_hook_table(libg_base, hooks + 1, count - 1);
  g_phase_runtime_hooks_installed.store(installed, std::memory_order_release);
  LOGI("exact-build phase hook set installed=%d ring=%zu cache=%zu", installed ? 1 : 0, kPhaseRuntimeRingCapacity,
       kPhaseRuntimeCacheCapacity);
  return installed;
}
