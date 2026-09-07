// Exact-build, allowlisted action-owned movement telemetry.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc;
//   * cr_replay_probe.cpp supplies pointer validators, bounded libg-string
//     decoding, JSON helpers, the inline-hook installer, and attestation;
//   * begin_action_movement_runtime_epoch is called while combat capture is
//     inactive, before the new combat epoch is armed.
//
// The public boundary reports native action stages only.  It never creates an
// EffectState and never promotes these records to effect.stage.

#include "action_movement_runtime_layout.h"
#include "probe_hook.h"
#include "epoch_ring.h"
#include "event_ring_json.h"

constexpr std::size_t kActionMovementRingCapacity = 1024;
constexpr std::size_t kActionMovementNameCapacity = 129;

enum class ActionMovementRawKind : std::uint8_t {
  GoldenKnightHopLaunch = 1,
  BossBanditWarpPositionCommit = 2,
  EliteArcherWarpPositionCommit = 3,
};

struct ActionMovementIdentity {
  std::uint32_t action_data_global_id = 0;
  std::uintptr_t action_class_vtable_offset = 0;
  std::size_t action_name_size = 0;
  char action_name[kActionMovementNameCapacity] = {};
};

struct ActionMovementRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  ActionMovementRawKind kind = ActionMovementRawKind::GoldenKnightHopLaunch;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  std::uint32_t action_data_global_id = 0;
  std::uintptr_t action_class_vtable_offset = 0;
  std::uintptr_t runtime_class_vtable_offset = 0;
  std::size_t action_name_size = 0;
  char action_name[kActionMovementNameCapacity] = {};
  CombatEntityFact source_before;
  CombatEntityFact source_after;
  CombatEntityFact target;
  bool target_present = false;
  std::int32_t chain_index = -1;
  bool position_changed = false;
  bool complete_context = false;
};

EpochRing<ActionMovementRawRecord, kActionMovementRingCapacity> g_action_movement_ring;
std::atomic<bool> g_action_movement_hooks_attested{false};
std::atomic<bool> g_action_movement_hooks_installed{false};

using GoldenKnightHopCommit = void (*)(void *, void *);
using WarpPositionCommit = void (*)(void *, void *);

GoldenKnightHopCommit g_original_golden_knight_hop_commit = nullptr;
WarpPositionCommit g_original_warp_position_commit = nullptr;

std::uintptr_t action_movement_caller_offset(const void *address) {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
  return address == nullptr || raw < g_libg_base ? 0 : raw - g_libg_base;
}

void begin_action_movement_runtime_epoch(std::uint64_t generation, std::uint64_t state_epoch) {
  g_action_movement_ring.begin(generation, state_epoch);
}

void reject_action_movement_record(const CombatCaptureContext &context) {
  g_action_movement_ring.reject(context.generation, context.state_epoch);
}

std::uint64_t publish_action_movement_record(ActionMovementRawRecord record) {
  return g_action_movement_ring.publish(record);
}

bool same_action_movement_entity(const CombatEntityFact &before, const CombatEntityFact &after) {
  return before.validated && before.present && after.validated && after.present &&
         before.native_object_id == after.native_object_id && before.owner == after.owner &&
         before.object_index == after.object_index && before.secondary_index == after.secondary_index &&
         before.card_id == after.card_id && before.object_kind == after.object_kind;
}

bool action_movement_source_is(const CombatEntityFact &source, std::int32_t first_card_id,
                               std::int32_t second_card_id = 0) {
  return source.validated && source.present &&
         (source.card_id == first_card_id || (second_card_id != 0 && source.card_id == second_card_id)) &&
         source.native_object_id != 0 && source.owner >= 0 && source.owner <= 1 && source.object_kind >= 0;
}

bool read_action_movement_identity(const void *action_data, ActionMovementIdentity *result) {
  if (result == nullptr || action_data == nullptr ||
      !process_range_is_readable(action_data, cr_action_movement_runtime::kMinimumActionDataBytes)) {
    return false;
  }
  *result = {};
  const std::uintptr_t vtable = reinterpret_cast<std::uintptr_t>(read_object_field<const void *>(action_data, 0x00));
  if (vtable < g_libg_base ||
      !read_bounded_libg_string_utf8(static_cast<const std::uint8_t *>(action_data) +
                                         cr_action_movement_runtime::kLogicDataNameOffset,
                                     result->action_name, sizeof(result->action_name), &result->action_name_size)) {
    return false;
  }
  result->action_class_vtable_offset = vtable - g_libg_base;
  result->action_data_global_id =
      read_object_field<std::uint32_t>(action_data, cr_action_movement_runtime::kLogicDataGlobalIdOffset);
  return result->action_name_size != 0;
}

bool action_movement_name_is(const ActionMovementIdentity &identity, const char *expected) {
  return expected != nullptr && std::strcmp(identity.action_name, expected) == 0;
}

void copy_action_movement_identity(const ActionMovementIdentity &identity, ActionMovementRawRecord *record) {
  if (record == nullptr) {
    return;
  }
  record->action_data_global_id = identity.action_data_global_id;
  record->action_class_vtable_offset = identity.action_class_vtable_offset;
  record->action_name_size = identity.action_name_size;
  std::memcpy(record->action_name, identity.action_name, identity.action_name_size + 1);
}

void golden_knight_hop_commit_hook(void *action_runtime, void *target) {
  if (!g_action_movement_hooks_installed.load(std::memory_order_acquire)) {
    g_original_golden_knight_hop_commit(action_runtime, target);
    return;
  }

  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    g_original_golden_knight_hop_commit(action_runtime, target);
    return;
  }

  ActionMovementIdentity identity;
  void *action_data = nullptr;
  void *source = nullptr;
  std::uintptr_t runtime_vtable_offset = 0;
  std::int32_t chain_index = -1;
  bool runtime_readable =
      process_range_is_readable(action_runtime, cr_action_movement_runtime::kMinimumDashingAttackChainRuntimeBytes);
  if (runtime_readable) {
    const std::uintptr_t runtime_vtable =
        reinterpret_cast<std::uintptr_t>(read_object_field<const void *>(action_runtime, 0x00));
    runtime_vtable_offset = runtime_vtable >= g_libg_base ? runtime_vtable - g_libg_base : 0;
    action_data = read_object_field<void *>(action_runtime, cr_action_movement_runtime::kActionRuntimeDataOffset);
    source = read_object_field<void *>(action_runtime, cr_action_movement_runtime::kActionRuntimeOwnerOffset);
    chain_index =
        read_object_field<std::int32_t>(action_runtime, cr_action_movement_runtime::kDashingAttackChainIndexOffset);
  }
  const bool identity_read = read_action_movement_identity(action_data, &identity);
  if (!identity_read ||
      !action_movement_name_is(identity, cr_action_movement_runtime::kGoldenKnightExecuteActionName)) {
    g_original_golden_knight_hop_commit(action_runtime, target);
    return;
  }

  CombatEntityFact source_before;
  CombatEntityFact target_fact;
  capture_combat_entity(source, context.object_manager, &source_before);
  capture_combat_entity(target, context.object_manager, &target_fact);
  const std::uintptr_t caller = action_movement_caller_offset(__builtin_return_address(0));
  const bool exact =
      runtime_readable && runtime_vtable_offset == cr_action_movement_runtime::kDashingAttackChainRuntimeVtableOffset &&
      identity.action_class_vtable_offset == cr_action_movement_runtime::kDashingAttackChainDataVtableOffset &&
      identity.action_data_global_id != 0 && chain_index >= 0 && caller != 0 &&
      action_movement_source_is(source_before, cr_action_movement_runtime::kGoldenKnightCardId) &&
      target_fact.validated && target_fact.present && target_fact.native_object_id != source_before.native_object_id;

  g_original_golden_knight_hop_commit(action_runtime, target);

  if (!exact) {
    reject_action_movement_record(context);
    return;
  }

  CombatEntityFact source_after;
  capture_combat_entity(source, context.object_manager, &source_after);
  const bool runtime_still_readable =
      process_range_is_readable(action_runtime, cr_action_movement_runtime::kMinimumDashingAttackChainRuntimeBytes);
  const std::int32_t state_after =
      runtime_still_readable
          ? read_object_field<std::int32_t>(action_runtime, cr_action_movement_runtime::kDashingAttackChainStateOffset)
          : -1;
  const std::int32_t chain_index_after =
      runtime_still_readable
          ? read_object_field<std::int32_t>(action_runtime, cr_action_movement_runtime::kDashingAttackChainIndexOffset)
          : -1;
  if (!same_action_movement_entity(source_before, source_after) || state_after != 1 ||
      chain_index_after != chain_index) {
    reject_action_movement_record(context);
    return;
  }

  ActionMovementRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind = ActionMovementRawKind::GoldenKnightHopLaunch;
  record.hook_offset = cr_action_movement_runtime::kGoldenKnightHopOffset;
  record.caller_offset = caller;
  record.runtime_class_vtable_offset = runtime_vtable_offset;
  copy_action_movement_identity(identity, &record);
  record.source_before = source_before;
  record.source_after = source_after;
  record.target = target_fact;
  record.target_present = true;
  record.chain_index = chain_index;
  record.position_changed =
      source_before.position_x != source_after.position_x || source_before.position_y != source_after.position_y;
  record.complete_context = true;
  publish_action_movement_record(record);
}

void warp_position_commit_hook(void *action_data, void *action_context) {
  if (!g_action_movement_hooks_installed.load(std::memory_order_acquire)) {
    g_original_warp_position_commit(action_data, action_context);
    return;
  }

  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    g_original_warp_position_commit(action_data, action_context);
    return;
  }

  ActionMovementIdentity identity;
  const bool identity_read = read_action_movement_identity(action_data, &identity);
  const bool boss =
      identity_read && action_movement_name_is(identity, cr_action_movement_runtime::kBossBanditWarpActionName);
  const bool elite =
      identity_read && action_movement_name_is(identity, cr_action_movement_runtime::kEliteArcherWarpActionName);
  if (!boss && !elite) {
    g_original_warp_position_commit(action_data, action_context);
    return;
  }

  void *source = nullptr;
  const bool context_readable =
      process_range_is_readable(action_context, cr_action_movement_runtime::kMinimumActionExecutionContextBytes);
  if (context_readable) {
    source = read_object_field<void *>(action_context, cr_action_movement_runtime::kActionExecutionOwnerOffset);
  }
  CombatEntityFact source_before;
  capture_combat_entity(source, context.object_manager, &source_before);
  const std::uintptr_t caller = action_movement_caller_offset(__builtin_return_address(0));
  const bool source_exact =
      boss ? action_movement_source_is(source_before, cr_action_movement_runtime::kBossBanditCardId)
           : action_movement_source_is(source_before, cr_action_movement_runtime::kEliteArcherCardId,
                                       cr_action_movement_runtime::kEliteArcherHeroFormCardId);
  const bool exact =
      context_readable &&
      identity.action_class_vtable_offset == cr_action_movement_runtime::kWarpCharacterDataVtableOffset &&
      identity.action_data_global_id != 0 && caller != 0 && source_exact;

  g_original_warp_position_commit(action_data, action_context);

  if (!exact) {
    reject_action_movement_record(context);
    return;
  }

  CombatEntityFact source_after;
  capture_combat_entity(source, context.object_manager, &source_after);
  if (!same_action_movement_entity(source_before, source_after)) {
    reject_action_movement_record(context);
    return;
  }
  const bool position_changed =
      source_before.position_x != source_after.position_x || source_before.position_y != source_after.position_y;
  if (!position_changed) {
    // ActionWarpCharacter can return before its position setter when its
    // native guard rejects execution.  That is an attempt, not a movement
    // commit, and is intentionally absent from this envelope.
    return;
  }

  ActionMovementRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind =
      boss ? ActionMovementRawKind::BossBanditWarpPositionCommit : ActionMovementRawKind::EliteArcherWarpPositionCommit;
  record.hook_offset = cr_action_movement_runtime::kWarpPositionCommitOffset;
  record.caller_offset = caller;
  copy_action_movement_identity(identity, &record);
  record.source_before = source_before;
  record.source_after = source_after;
  record.target_present = false;
  record.position_changed = true;
  record.complete_context = true;
  publish_action_movement_record(record);
}

const char *action_movement_kind_label(ActionMovementRawKind kind) {
  switch (kind) {
  case ActionMovementRawKind::GoldenKnightHopLaunch:
    return "golden_knight_chain_hop_launch";
  case ActionMovementRawKind::BossBanditWarpPositionCommit:
    return "boss_bandit_warp_position_commit";
  case ActionMovementRawKind::EliteArcherWarpPositionCommit:
    return "elite_archer_warp_position_commit";
  }
  return nullptr;
}

const char *action_movement_mode_label(ActionMovementRawKind kind) {
  return kind == ActionMovementRawKind::GoldenKnightHopLaunch ? "dash_chain" : "warp";
}

const char *action_movement_stage_label(ActionMovementRawKind kind) {
  return kind == ActionMovementRawKind::GoldenKnightHopLaunch ? "hop_launch" : "position_commit";
}

bool append_action_movement_runtime_events_json(std::uint64_t generation, std::uint64_t state_epoch,
                                                std::int32_t observation_tick, char *response,
                                                std::size_t response_size, std::size_t *used) {
  return append_epoch_ring_events(
      g_action_movement_ring,
      {cr_action_movement_runtime::kSchema, "allowlisted-native-action-movement-hook-ring",
       "sha-build-id-prologue-abi-action-name-global-id-class-vtable-source-identity"},
      generation, state_epoch, observation_tick, g_action_movement_hooks_attested.load(std::memory_order_acquire),
      g_action_movement_hooks_installed.load(std::memory_order_acquire), response, response_size, used,
      [](JsonWriter &json, const ActionMovementRawRecord &record, bool first) {
        const char *kind = action_movement_kind_label(record.kind);
        if (!kind || record.action_name_size == 0)
          return false;
        json.append("%s{", first ? "" : ",");
        json.field("sequence", record.sequence);
        json.field("tick", record.tick);
        json.field("kind", kind);
        json.field("mode", action_movement_mode_label(record.kind));
        json.field("stage", action_movement_stage_label(record.kind));
        json.field("hookOffset", record.hook_offset);
        json.field("callerOffset", record.caller_offset);
        json.field("actionDataGlobalId", record.action_data_global_id);
        json.optional("actionName", true, append_json_utf8_string, record.action_name, record.action_name_size);
        json.field("actionClassVtableOffset", record.action_class_vtable_offset);
        json.nullable("runtimeClassVtableOffset", record.runtime_class_vtable_offset,
                      record.runtime_class_vtable_offset != 0);
        json.optional("sourceBefore", true, append_combat_entity_fact, record.source_before);
        json.optional("sourceAfter", true, append_combat_entity_fact, record.source_after);
        json.optional("target", record.target_present, append_combat_entity_fact, record.target);
        json.nullable("chainIndex", record.chain_index, record.chain_index >= 0);
        json.field("positionChanged", record.position_changed);
        json.field("completeContext", record.complete_context);
        return json.end_object();
      });
}

const NativeHookSpec g_action_movement_hook_specs[] = {
    native_hook(cr_action_movement_runtime::kGoldenKnightHopOffset,
                cr_action_movement_runtime::kGoldenKnightHopPrologue, &golden_knight_hop_commit_hook,
                &g_original_golden_knight_hop_commit, "action movement golden knight hop"),
    native_hook(cr_action_movement_runtime::kWarpPositionCommitOffset,
                cr_action_movement_runtime::kWarpPositionCommitPrologue, &warp_position_commit_hook,
                &g_original_warp_position_commit, "action movement warp position commit"),
};

bool verify_action_movement_hook_fingerprints(std::uintptr_t libg_base) {
  return verify_native_binary(g_libg_path, cr_action_movement_runtime::kExpectedLibgSha256,
                              cr_action_movement_runtime::kExpectedLibgBuildId) &&
         verify_native_hook_table(libg_base, g_action_movement_hook_specs, 2);
}

bool install_action_movement_runtime_hooks(std::uintptr_t libg_base) {
  const bool attested = verify_action_movement_hook_fingerprints(libg_base);
  g_action_movement_hooks_attested.store(attested, std::memory_order_release);
  const bool installed = attested && install_native_hook_table(libg_base, g_action_movement_hook_specs, 2, true);
  g_action_movement_hooks_installed.store(installed, std::memory_order_release);
  LOGI("action_movement hooks installed=%d ring=%zu", installed ? 1 : 0, kActionMovementRingCapacity);
  return installed;
}
