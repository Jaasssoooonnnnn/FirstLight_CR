// Exact-build public runtime snapshots for Dagger Duchess and Royal Chef.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc and remaining_runtime_telemetry.inc;
//   * cr_replay_probe.cpp supplies object-vector validation, JSON helpers,
//     binary attestation, inline-hook installation, and logging;
//   * call begin_tower_troop_runtime_epoch() at each state epoch and
//     install_tower_troop_runtime_hooks() during probe installation.
//
// The cache is pointer-free.  Native pointers are validated synchronously in
// the hook and reduced to stable native object IDs before publication.

#include "tower_troop_runtime_layout.h"
#include "probe_hook.h"
#include "probe_json.h"

constexpr std::size_t kTowerTroopRuntimeRingCapacity = 256;

enum class TowerTroopRuntimeKind : std::uint8_t {
  DaggerDuchess = 1,
  RoyalChef = 2,
};

struct TowerTroopRuntimeRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  TowerTroopRuntimeKind kind = TowerTroopRuntimeKind::DaggerDuchess;
  std::uint32_t source_native_object_id = 0;
  std::int32_t charge_count = 0;
  std::int32_t max_charge_count = 0;
  std::int32_t recharge_elapsed_ms = 0;
  std::int32_t recharge_duration_ms = 0;
  std::int32_t start_delay_remaining_ms = 0;
  std::int32_t start_delay_duration_ms = 0;
  std::int32_t cooking_contribution = 0;
  std::int32_t contribution_needed = 0;
  std::int32_t throw_delay_remaining_ms = cr_tower_troop_runtime::kChefNoThrowDelay;
  std::uint32_t target_native_object_id = 0;
};

struct TowerTroopRuntimeRingSlot {
  std::atomic<std::uint64_t> published_sequence{0};
  TowerTroopRuntimeRecord record;
};

TowerTroopRuntimeRingSlot g_tower_troop_runtime_ring[kTowerTroopRuntimeRingCapacity];
std::atomic<std::uint64_t> g_tower_troop_runtime_next_sequence{1};
std::atomic<std::uint64_t> g_tower_troop_runtime_generation{0};
std::atomic<std::uint64_t> g_tower_troop_runtime_state_epoch{0};
std::atomic<std::uint64_t> g_tower_troop_runtime_rejected_count{0};
std::atomic<bool> g_tower_troop_runtime_hooks_attested{false};
std::atomic<bool> g_tower_troop_runtime_hooks_installed{false};

using TowerTroopRuntimeFunction = void (*)(void *);
TowerTroopRuntimeFunction g_original_dagger_runtime_init = nullptr;
TowerTroopRuntimeFunction g_original_dagger_runtime_tick = nullptr;
TowerTroopRuntimeFunction g_original_chef_runtime_init = nullptr;
TowerTroopRuntimeFunction g_original_chef_runtime_tick = nullptr;

void begin_tower_troop_runtime_epoch(std::uint64_t generation, std::uint64_t state_epoch) {
  g_tower_troop_runtime_generation.store(generation, std::memory_order_release);
  g_tower_troop_runtime_state_epoch.store(state_epoch, std::memory_order_release);
  g_tower_troop_runtime_rejected_count.store(0, std::memory_order_release);
  for (TowerTroopRuntimeRingSlot &slot : g_tower_troop_runtime_ring) {
    slot.published_sequence.store(0, std::memory_order_release);
  }
}

void publish_tower_troop_runtime(TowerTroopRuntimeRecord record) {
  const std::uint64_t sequence = g_tower_troop_runtime_next_sequence.fetch_add(1, std::memory_order_acq_rel);
  record.sequence = sequence;
  TowerTroopRuntimeRingSlot &slot = g_tower_troop_runtime_ring[(sequence - 1) % kTowerTroopRuntimeRingCapacity];
  slot.published_sequence.store(0, std::memory_order_release);
  slot.record = record;
  slot.published_sequence.store(sequence, std::memory_order_release);
}

bool capture_tower_troop_runtime_common(void *runtime, std::uintptr_t expected_runtime_vtable,
                                        std::uintptr_t expected_static_vtable, std::size_t minimum_runtime_bytes,
                                        std::size_t minimum_static_bytes, CombatCaptureContext *context,
                                        const void **static_data, CombatEntityFact *source) {
  if (runtime == nullptr || context == nullptr || static_data == nullptr || source == nullptr ||
      !g_tower_troop_runtime_hooks_installed.load(std::memory_order_acquire) ||
      !current_combat_capture_context(context) || context->tick < 0 ||
      context->generation != g_tower_troop_runtime_generation.load(std::memory_order_acquire) ||
      context->state_epoch != g_tower_troop_runtime_state_epoch.load(std::memory_order_acquire) ||
      !process_range_is_readable(runtime, minimum_runtime_bytes) ||
      remaining_vtable_offset(runtime) != expected_runtime_vtable) {
    return false;
  }
  *static_data = read_object_field<const void *>(runtime, cr_tower_troop_runtime::kRuntimeStaticDataOffset);
  const void *const source_pointer =
      read_object_field<const void *>(runtime, cr_tower_troop_runtime::kRuntimeSourceObjectOffset);
  if (*static_data == nullptr || !process_range_is_readable(*static_data, minimum_static_bytes) ||
      remaining_vtable_offset(*static_data) != expected_static_vtable) {
    return false;
  }
  capture_combat_entity(source_pointer, context->object_manager, source);
  return source->validated && source->present && source->native_object_id != 0 && source->owner >= 0 &&
         source->owner < 2;
}

void capture_dagger_runtime(void *runtime) {
  if (runtime == nullptr || !process_range_is_readable(runtime, cr_tower_troop_runtime::kDaggerRechargeElapsedMsOffset +
                                                                    sizeof(std::int32_t))) {
    return;
  }
  const void *const preliminary_data =
      read_object_field<const void *>(runtime, cr_tower_troop_runtime::kRuntimeStaticDataOffset);
  if (preliminary_data == nullptr ||
      !process_range_is_readable(preliminary_data,
                                 cr_tower_troop_runtime::kDaggerRechargeIncrementOffset + sizeof(std::int32_t)) ||
      remaining_vtable_offset(preliminary_data) != cr_tower_troop_runtime::kDaggerStaticVtableOffset) {
    return;
  }
  const std::int32_t max_charge_count =
      read_object_field<std::int32_t>(preliminary_data, cr_tower_troop_runtime::kDaggerMaxChargeCountOffset);
  const std::int32_t recharge_duration_ms =
      read_object_field<std::int32_t>(preliminary_data, cr_tower_troop_runtime::kDaggerRechargeDurationMsOffset);
  const std::int32_t recharge_increment =
      read_object_field<std::int32_t>(preliminary_data, cr_tower_troop_runtime::kDaggerRechargeIncrementOffset);
  // This exact ActionBurstAttack implementation is also usable by other
  // content.  Admit only the current competitive Dagger Duchess contract.
  if (max_charge_count != 8 || recharge_duration_ms != 900 || recharge_increment != 1) {
    return;
  }

  CombatCaptureContext context;
  CombatEntityFact source;
  const void *static_data = nullptr;
  if (!capture_tower_troop_runtime_common(runtime, cr_tower_troop_runtime::kDaggerRuntimeVtableOffset,
                                          cr_tower_troop_runtime::kDaggerStaticVtableOffset,
                                          cr_tower_troop_runtime::kDaggerRechargeElapsedMsOffset + sizeof(std::int32_t),
                                          cr_tower_troop_runtime::kDaggerRechargeIncrementOffset + sizeof(std::int32_t),
                                          &context, &static_data, &source)) {
    return;
  }
  const std::int32_t charge_count =
      read_object_field<std::int32_t>(runtime, cr_tower_troop_runtime::kDaggerChargeCountOffset);
  const std::int32_t recharge_elapsed_ms =
      read_object_field<std::int32_t>(runtime, cr_tower_troop_runtime::kDaggerRechargeElapsedMsOffset);
  if (static_data != preliminary_data || charge_count < 0 || charge_count > max_charge_count ||
      recharge_elapsed_ms < 0 || recharge_elapsed_ms > recharge_duration_ms) {
    g_tower_troop_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  TowerTroopRuntimeRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind = TowerTroopRuntimeKind::DaggerDuchess;
  record.source_native_object_id = source.native_object_id;
  record.charge_count = charge_count;
  record.max_charge_count = max_charge_count;
  record.recharge_elapsed_ms = recharge_elapsed_ms;
  record.recharge_duration_ms = recharge_duration_ms;
  publish_tower_troop_runtime(record);
}

void capture_chef_runtime(void *runtime) {
  CombatCaptureContext context;
  CombatEntityFact source;
  const void *static_data = nullptr;
  if (!capture_tower_troop_runtime_common(
          runtime, cr_tower_troop_runtime::kChefRuntimeVtableOffset, cr_tower_troop_runtime::kChefStaticVtableOffset,
          cr_tower_troop_runtime::kChefTargetObjectOffset + sizeof(void *),
          cr_tower_troop_runtime::kChefStartDelayDurationMsOffset + sizeof(std::int32_t), &context, &static_data,
          &source)) {
    return;
  }
  const std::int32_t contribution_base =
      read_object_field<std::int32_t>(static_data, cr_tower_troop_runtime::kChefContributionNeededOffset);
  const std::int32_t start_delay_duration_ms =
      read_object_field<std::int32_t>(static_data, cr_tower_troop_runtime::kChefStartDelayDurationMsOffset);
  if (contribution_base != 23000 || start_delay_duration_ms != 7000) {
    LOGE("tower-troop runtime rejected reason=chef-static-contract "
         "contributionBase=%d startDelayDurationMs=%d",
         contribution_base, start_delay_duration_ms);
    g_tower_troop_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const std::int32_t start_delay_remaining_ms =
      read_object_field<std::int32_t>(runtime, cr_tower_troop_runtime::kChefStartDelayRemainingMsOffset);
  const std::int32_t cooking_contribution =
      read_object_field<std::int32_t>(runtime, cr_tower_troop_runtime::kChefCookingContributionOffset);
  const std::int32_t throw_delay_remaining_ms =
      read_object_field<std::int32_t>(runtime, cr_tower_troop_runtime::kChefThrowDelayRemainingMsOffset);
  const std::int32_t contribution_needed = contribution_base * cr_tower_troop_runtime::kChefContributionScale;
  const bool throw_delay_valid = throw_delay_remaining_ms == cr_tower_troop_runtime::kChefNoThrowDelay ||
                                 throw_delay_remaining_ms == cr_tower_troop_runtime::kChefTargetlessThrowDelay ||
                                 throw_delay_remaining_ms >= 0;
  if (start_delay_remaining_ms < 0 || start_delay_remaining_ms > start_delay_duration_ms || cooking_contribution < 0 ||
      contribution_needed <= 0 || !throw_delay_valid) {
    LOGE("tower-troop runtime rejected reason=chef-state-bounds "
         "sourceNativeObjectId=%u startDelayRemainingMs=%d "
         "startDelayDurationMs=%d cookingContribution=%d "
         "contributionNeeded=%d throwDelayRemainingMs=%d",
         source.native_object_id, start_delay_remaining_ms, start_delay_duration_ms, cooking_contribution,
         contribution_needed, throw_delay_remaining_ms);
    g_tower_troop_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  std::uint32_t target_native_object_id = 0;
  if (throw_delay_remaining_ms != cr_tower_troop_runtime::kChefNoThrowDelay) {
    const void *const target_pointer =
        read_object_field<const void *>(runtime, cr_tower_troop_runtime::kChefTargetObjectOffset);
    const std::int32_t target_handle =
        read_object_field<std::int32_t>(runtime, cr_tower_troop_runtime::kChefTargetHandleOffset);
    CombatEntityFact target;
    capture_combat_entity(target_pointer, context.object_manager, &target);
    if (!target.validated ||
        (target.present && (throw_delay_remaining_ms == cr_tower_troop_runtime::kChefTargetlessThrowDelay ||
                            target.owner != source.owner || target_handle <= 0))) {
      LOGE("tower-troop runtime rejected reason=chef-target "
           "sourceNativeObjectId=%u targetValidated=%d targetPresent=%d "
           "sourceOwner=%d targetOwner=%d targetHandle=%d "
           "capturedNativeObjectId=%u capturedObjectIndex=%d",
           source.native_object_id, target.validated ? 1 : 0, target.present ? 1 : 0, source.owner, target.owner,
           target_handle, target.native_object_id, target.object_index);
      g_tower_troop_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (target.present) {
      target_native_object_id = target.native_object_id;
    }
    // A null target is also authoritative. The selected recipient may die
    // before release, but ActionChefTower still spends the pancake and
    // completes the throw into empty space. Preserve the throw with ID 0.
  }

  TowerTroopRuntimeRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind = TowerTroopRuntimeKind::RoyalChef;
  record.source_native_object_id = source.native_object_id;
  record.start_delay_remaining_ms = start_delay_remaining_ms;
  record.start_delay_duration_ms = start_delay_duration_ms;
  record.cooking_contribution = cooking_contribution;
  record.contribution_needed = contribution_needed;
  record.throw_delay_remaining_ms = throw_delay_remaining_ms;
  record.target_native_object_id = target_native_object_id;
  publish_tower_troop_runtime(record);
}

extern "C" void tower_troop_dagger_init_hook(void *runtime) {
  g_original_dagger_runtime_init(runtime);
  capture_dagger_runtime(runtime);
}

extern "C" void tower_troop_dagger_tick_hook(void *runtime) {
  g_original_dagger_runtime_tick(runtime);
  capture_dagger_runtime(runtime);
}

extern "C" void tower_troop_chef_init_hook(void *runtime) {
  g_original_chef_runtime_init(runtime);
  capture_chef_runtime(runtime);
}

extern "C" void tower_troop_chef_tick_hook(void *runtime) {
  g_original_chef_runtime_tick(runtime);
  capture_chef_runtime(runtime);
}

bool latest_tower_troop_runtime(std::uint64_t generation, std::uint64_t state_epoch, std::int32_t observation_tick,
                                std::uint32_t source_native_object_id, TowerTroopRuntimeRecord *result) {
  if (result == nullptr || source_native_object_id == 0) {
    return false;
  }
  bool found = false;
  TowerTroopRuntimeRecord latest;
  for (TowerTroopRuntimeRingSlot &slot : g_tower_troop_runtime_ring) {
    const std::uint64_t sequence_before = slot.published_sequence.load(std::memory_order_acquire);
    if (sequence_before == 0) {
      continue;
    }
    const TowerTroopRuntimeRecord candidate = slot.record;
    const std::uint64_t sequence_after = slot.published_sequence.load(std::memory_order_acquire);
    if (sequence_before != sequence_after || candidate.sequence != sequence_before ||
        candidate.generation != generation || candidate.state_epoch != state_epoch ||
        candidate.tick > observation_tick || candidate.source_native_object_id != source_native_object_id) {
      continue;
    }
    if (!found || candidate.sequence > latest.sequence) {
      latest = candidate;
      found = true;
    }
  }
  if (found) {
    *result = latest;
  }
  return found;
}

std::size_t latest_tower_troop_runtimes(std::uint64_t generation, std::uint64_t state_epoch,
                                        std::int32_t observation_tick, TowerTroopRuntimeRecord *output,
                                        std::size_t capacity) {
  if (output == nullptr || capacity == 0 || observation_tick < 0) {
    return 0;
  }
  const std::size_t bounded_capacity =
      capacity < kMaxStoredTowerTroopRuntimes ? capacity : kMaxStoredTowerTroopRuntimes;
  std::uint64_t latest_sequences[kMaxStoredTowerTroopRuntimes] = {};
  std::size_t count = 0;
  for (TowerTroopRuntimeRingSlot &slot : g_tower_troop_runtime_ring) {
    const std::uint64_t sequence_before = slot.published_sequence.load(std::memory_order_acquire);
    if (sequence_before == 0) {
      continue;
    }
    const TowerTroopRuntimeRecord candidate = slot.record;
    const std::uint64_t sequence_after = slot.published_sequence.load(std::memory_order_acquire);
    if (sequence_before != sequence_after || candidate.sequence != sequence_before ||
        candidate.generation != generation || candidate.state_epoch != state_epoch ||
        candidate.tick > observation_tick || candidate.source_native_object_id == 0) {
      continue;
    }
    std::size_t index = count;
    for (std::size_t current = 0; current < count; ++current) {
      if (output[current].source_native_object_id == candidate.source_native_object_id) {
        index = current;
        break;
      }
    }
    if (index == count) {
      if (count >= bounded_capacity) {
        continue;
      }
      ++count;
    } else if (latest_sequences[index] >= candidate.sequence) {
      continue;
    }
    output[index] = candidate;
    latest_sequences[index] = candidate.sequence;
  }
  return count;
}

StoredTowerTroopRuntimeState stored_tower_troop_runtime_state(const TowerTroopRuntimeRecord &record) {
  StoredTowerTroopRuntimeState stored;
  stored.kind = static_cast<std::uint8_t>(record.kind);
  stored.source_native_object_id = record.source_native_object_id;
  stored.charge_count = record.charge_count;
  stored.max_charge_count = record.max_charge_count;
  stored.recharge_elapsed_ms = record.recharge_elapsed_ms;
  stored.recharge_duration_ms = record.recharge_duration_ms;
  stored.start_delay_remaining_ms = record.start_delay_remaining_ms;
  stored.start_delay_duration_ms = record.start_delay_duration_ms;
  stored.cooking_contribution = record.cooking_contribution;
  stored.contribution_needed = record.contribution_needed;
  stored.throw_delay_remaining_ms = record.throw_delay_remaining_ms;
  stored.target_native_object_id = record.target_native_object_id;
  return stored;
}

std::size_t copy_tower_troop_runtime_snapshot(std::uint64_t generation, std::uint64_t state_epoch, std::int32_t tick,
                                              StoredTowerTroopRuntimeState *output, std::size_t capacity) {
  if (output == nullptr)
    return 0;
  TowerTroopRuntimeRecord records[kMaxStoredTowerTroopRuntimes];
  const auto count = latest_tower_troop_runtimes(generation, state_epoch, tick, records, capacity);
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = stored_tower_troop_runtime_state(records[index]);
  }
  return count;
}

void restore_tower_troop_runtime_snapshot(std::uint64_t generation, std::uint64_t state_epoch, std::int32_t tick,
                                          const StoredTowerTroopRuntimeState *states, std::size_t count) {
  if (states == nullptr || count > kMaxStoredTowerTroopRuntimes || generation == 0 || state_epoch == 0 || tick < 0) {
    return;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const StoredTowerTroopRuntimeState &stored = states[index];
    if (stored.source_native_object_id == 0 ||
        (stored.kind != static_cast<std::uint8_t>(TowerTroopRuntimeKind::DaggerDuchess) &&
         stored.kind != static_cast<std::uint8_t>(TowerTroopRuntimeKind::RoyalChef))) {
      continue;
    }
    TowerTroopRuntimeRecord record;
    record.generation = generation;
    record.state_epoch = state_epoch;
    record.tick = tick;
    record.kind = static_cast<TowerTroopRuntimeKind>(stored.kind);
    record.source_native_object_id = stored.source_native_object_id;
    record.charge_count = stored.charge_count;
    record.max_charge_count = stored.max_charge_count;
    record.recharge_elapsed_ms = stored.recharge_elapsed_ms;
    record.recharge_duration_ms = stored.recharge_duration_ms;
    record.start_delay_remaining_ms = stored.start_delay_remaining_ms;
    record.start_delay_duration_ms = stored.start_delay_duration_ms;
    record.cooking_contribution = stored.cooking_contribution;
    record.contribution_needed = stored.contribution_needed;
    record.throw_delay_remaining_ms = stored.throw_delay_remaining_ms;
    record.target_native_object_id = stored.target_native_object_id;
    publish_tower_troop_runtime(record);
  }
}

bool append_tower_troop_runtime_object_json(char *response, std::size_t response_size, std::size_t *used,
                                            const TowerTroopRuntimeRecord &record) {
  JsonWriter json(response, response_size, used);
  const bool dagger = record.kind == TowerTroopRuntimeKind::DaggerDuchess;
  json.begin_object();
  json.field("schema", cr_tower_troop_runtime::kSchema);
  json.field("kind", dagger ? "dagger_duchess" : "royal_chef");
  json.field("observedTick", record.tick);
  if (dagger) {
    json.field("chargeCount", record.charge_count);
    json.field("maxChargeCount", record.max_charge_count);
    json.field("rechargeElapsedMs", record.recharge_elapsed_ms);
    json.field("rechargeDurationMs", record.recharge_duration_ms);
  } else {
    json.field("startDelayRemainingMs", record.start_delay_remaining_ms);
    json.field("startDelayDurationMs", record.start_delay_duration_ms);
    json.field("cookingContribution", record.cooking_contribution);
    json.field("contributionNeeded", record.contribution_needed);
    json.nullable("throwDelayRemainingMs", record.throw_delay_remaining_ms,
                  record.throw_delay_remaining_ms != cr_tower_troop_runtime::kChefNoThrowDelay);
    json.nullable("targetNativeObjectId", record.target_native_object_id, record.target_native_object_id != 0);
  }
  return json.end_object();
}

bool append_tower_troop_runtime_envelope_json(std::uint64_t generation, std::uint64_t state_epoch,
                                              std::int32_t observation_tick, char *response, std::size_t response_size,
                                              std::size_t *used) {
  const bool identity = g_tower_troop_runtime_generation.load(std::memory_order_acquire) == generation &&
                        g_tower_troop_runtime_state_epoch.load(std::memory_order_acquire) == state_epoch;
  const bool attested = g_tower_troop_runtime_hooks_attested.load(std::memory_order_acquire);
  const bool installed = g_tower_troop_runtime_hooks_installed.load(std::memory_order_acquire);
  const std::uint64_t rejected = identity ? g_tower_troop_runtime_rejected_count.load(std::memory_order_acquire) : 0;
  return append_json(response, response_size, used,
                     "{\"ok\":true,\"schema\":\"%s\",\"generation\":%llu,"
                     "\"stateEpoch\":%llu,\"observationTick\":%d,"
                     "\"hookSetAttested\":%s,\"hookSetInstalled\":%s,"
                     "\"rejectedCount\":%llu,\"complete\":%s}",
                     cr_tower_troop_runtime::kSchema, static_cast<unsigned long long>(generation),
                     static_cast<unsigned long long>(state_epoch), observation_tick, json_boolean(attested),
                     json_boolean(installed), static_cast<unsigned long long>(rejected),
                     json_boolean(installed && identity && rejected == 0));
}

const NativeHookSpec g_tower_troop_runtime_hook_specs[] = {
    native_hook(cr_tower_troop_runtime::kDaggerInitOffset, cr_tower_troop_runtime::kDaggerInitPrologue,
                &tower_troop_dagger_init_hook, &g_original_dagger_runtime_init, "Dagger Duchess runtime init"),
    native_hook(cr_tower_troop_runtime::kDaggerTickOffset, cr_tower_troop_runtime::kDaggerTickPrologue,
                &tower_troop_dagger_tick_hook, &g_original_dagger_runtime_tick, "Dagger Duchess runtime tick"),
    native_hook(cr_tower_troop_runtime::kChefInitOffset, cr_tower_troop_runtime::kChefInitPrologue,
                &tower_troop_chef_init_hook, &g_original_chef_runtime_init, "Royal Chef runtime init"),
    native_hook(cr_tower_troop_runtime::kChefTickOffset, cr_tower_troop_runtime::kChefTickPrologue,
                &tower_troop_chef_tick_hook, &g_original_chef_runtime_tick, "Royal Chef runtime tick"),
};

bool verify_tower_troop_runtime_hook_fingerprints(std::uintptr_t libg_base) {
  return verify_native_binary(g_libg_path, cr_tower_troop_runtime::kExactLibgSha256,
                              cr_tower_troop_runtime::kExactLibgBuildId) &&
         verify_native_hook_table(libg_base, g_tower_troop_runtime_hook_specs, 4);
}

bool install_tower_troop_runtime_hooks(std::uintptr_t libg_base) {
  const bool attested = verify_tower_troop_runtime_hook_fingerprints(libg_base);
  g_tower_troop_runtime_hooks_attested.store(attested, std::memory_order_release);
#if !defined(__aarch64__)
  g_tower_troop_runtime_hooks_installed.store(false, std::memory_order_release);
  return false;
#else
  const bool installed = attested && install_native_hook_table(libg_base, g_tower_troop_runtime_hook_specs, 4);
  g_tower_troop_runtime_hooks_installed.store(installed, std::memory_order_release);
  LOGI("tower_troop_runtime hooks installed=%d ring=%zu", installed ? 1 : 0, kTowerTroopRuntimeRingCapacity);
  return installed;
#endif
}
