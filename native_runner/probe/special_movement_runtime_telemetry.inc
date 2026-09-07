// Exact-build ordinary special-movement hook for Null's Royale 15.535.13.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc;
//   * cr_replay_probe.cpp supplies the common pointer/identity validators,
//     hook installer, JSON encoder, binary attestation, and logging helpers;
//   * begin_special_movement_runtime_epoch is called while combat capture is
//     inactive, before the new combat epoch is armed.
//
// The public boundary is pointer-free.  A record names the exact native
// execution edge and preserves distinct pre/post entity facts; it does not
// synthesize an EffectState or infer a visual airborne/traversal stage.

#include "special_movement_runtime_layout.h"
#include "probe_hook.h"
#include "epoch_ring.h"
#include "event_ring_json.h"

constexpr std::size_t kSpecialMovementRingCapacity = 1024;

struct SpecialMovementRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  CombatEntityFact source_before;
  CombatEntityFact source_after;
  CombatEntityFact target;
  std::int32_t requested_x = 0;
  std::int32_t requested_y = 0;
  std::int32_t target_radius = 0;
  std::int32_t operation_flag = 0;
  std::int32_t cooldown_before_ms = 0;
  std::int32_t cooldown_after_ms = 0;
  bool position_changed = false;
  bool complete_context = false;
};

EpochRing<SpecialMovementRawRecord, kSpecialMovementRingCapacity> g_special_movement_ring;
std::atomic<bool> g_special_movement_hook_attested{false};
std::atomic<bool> g_special_movement_hook_installed{false};

using SpecialMovementExecute = void (*)(void *, std::int32_t, std::int32_t, std::int32_t, std::int32_t);

SpecialMovementExecute g_original_special_movement_execute = nullptr;

std::uintptr_t special_movement_caller_offset(const void *address) {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
  return address == nullptr || raw < g_libg_base ? 0 : raw - g_libg_base;
}

void begin_special_movement_runtime_epoch(std::uint64_t generation, std::uint64_t state_epoch) {
  g_special_movement_ring.begin(generation, state_epoch);
}

void reject_special_movement_record(const CombatCaptureContext &context) {
  g_special_movement_ring.reject(context.generation, context.state_epoch);
}

std::uint64_t publish_special_movement_record(SpecialMovementRawRecord record) {
  return g_special_movement_ring.publish(record);
}

bool same_special_movement_entity(const CombatEntityFact &before, const CombatEntityFact &after) {
  return before.validated && before.present && after.validated && after.present &&
         before.native_object_id == after.native_object_id && before.owner == after.owner &&
         before.object_index == after.object_index && before.secondary_index == after.secondary_index &&
         before.card_id == after.card_id && before.object_kind == after.object_kind;
}

void special_movement_execute_hook(void *component, std::int32_t requested_x, std::int32_t requested_y,
                                   std::int32_t target_radius, std::int32_t operation_flag) {
  const std::uintptr_t caller = special_movement_caller_offset(__builtin_return_address(0));
  if (caller != cr_special_movement_runtime::kExecuteCallerReturnOffset) {
    g_original_special_movement_execute(component, requested_x, requested_y, target_radius, operation_flag);
    return;
  }

  CombatCaptureContext context;
  const bool capture = current_combat_capture_context(&context);
  void *owner = nullptr;
  void *target = nullptr;
  std::int32_t cooldown_before = -1;
  CombatEntityFact source_before;
  CombatEntityFact target_fact;
  bool exact = false;
  if (capture && process_range_is_readable(component, cr_special_movement_runtime::kComponentDashCooldownOffset +
                                                          sizeof(std::int32_t))) {
    owner = read_object_field<void *>(component, cr_special_movement_runtime::kComponentOwnerOffset);
    target = read_object_field<void *>(component, cr_special_movement_runtime::kComponentTargetOffset);
    cooldown_before =
        read_object_field<std::int32_t>(component, cr_special_movement_runtime::kComponentDashCooldownOffset);
    capture_combat_entity(owner, context.object_manager, &source_before);
    capture_combat_entity(target, context.object_manager, &target_fact);
    exact = owner != nullptr && target != nullptr && combat_component_type_is(component, owner, 0) &&
            source_before.validated && source_before.present && target_fact.validated && target_fact.present &&
            target_radius >= 0 && operation_flag == cr_special_movement_runtime::kExpectedOperationFlag &&
            cooldown_before >= 0;
  }

  g_original_special_movement_execute(component, requested_x, requested_y, target_radius, operation_flag);

  if (!capture) {
    return;
  }
  if (!exact) {
    reject_special_movement_record(context);
    return;
  }

  CombatEntityFact source_after;
  capture_combat_entity(owner, context.object_manager, &source_after);
  const std::int32_t cooldown_after =
      process_range_is_readable(component,
                                cr_special_movement_runtime::kComponentDashCooldownOffset + sizeof(std::int32_t))
          ? read_object_field<std::int32_t>(component, cr_special_movement_runtime::kComponentDashCooldownOffset)
          : -1;
  if (!same_special_movement_entity(source_before, source_after) || cooldown_after < 0) {
    reject_special_movement_record(context);
    return;
  }

  SpecialMovementRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.hook_offset = cr_special_movement_runtime::kExecuteOffset;
  record.caller_offset = cr_special_movement_runtime::kExecuteCallerReturnOffset;
  record.source_before = source_before;
  record.source_after = source_after;
  record.target = target_fact;
  record.requested_x = requested_x;
  record.requested_y = requested_y;
  record.target_radius = target_radius;
  record.operation_flag = operation_flag;
  record.cooldown_before_ms = cooldown_before;
  record.cooldown_after_ms = cooldown_after;
  record.position_changed =
      source_before.position_x != source_after.position_x || source_before.position_y != source_after.position_y;
  record.complete_context = true;
  publish_special_movement_record(record);
}

bool append_special_movement_runtime_events_json(std::uint64_t generation, std::uint64_t state_epoch,
                                                 std::int32_t observation_tick, char *response,
                                                 std::size_t response_size, std::size_t *used) {
  return append_epoch_ring_events(
      g_special_movement_ring,
      {cr_special_movement_runtime::kSchema, "caller-filtered-native-special-movement-hook-ring",
       "sha-build-id-prologue-abi-type0-owner-target-identity"},
      generation, state_epoch, observation_tick, g_special_movement_hook_attested.load(std::memory_order_acquire),
      g_special_movement_hook_installed.load(std::memory_order_acquire), response, response_size, used,
      [](JsonWriter &json, const SpecialMovementRawRecord &record, bool first) {
        json.append("%s{", first ? "" : ",");
        json.field("sequence", record.sequence);
        json.field("tick", record.tick);
        json.field("kind", "ordinary_dash_execute");
        json.field("mode", "dash");
        json.field("stage", "execute");
        json.field("hookOffset", record.hook_offset);
        json.field("callerOffset", record.caller_offset);
        json.optional("sourceBefore", true, append_combat_entity_fact, record.source_before);
        json.optional("sourceAfter", true, append_combat_entity_fact, record.source_after);
        json.optional("target", true, append_combat_entity_fact, record.target);
        json.field("requestedX", record.requested_x);
        json.field("requestedY", record.requested_y);
        json.field("targetRadius", record.target_radius);
        json.field("operationFlag", record.operation_flag);
        json.field("cooldownBeforeMs", record.cooldown_before_ms);
        json.field("cooldownAfterMs", record.cooldown_after_ms);
        json.field("positionChanged", record.position_changed);
        json.field("completeContext", record.complete_context);
        return json.end_object();
      });
}

const NativeHookSpec g_special_movement_hook_specs[] = {
    native_hook(cr_special_movement_runtime::kExecuteOffset, cr_special_movement_runtime::kExecutePrologue,
                &special_movement_execute_hook, &g_original_special_movement_execute,
                "ordinary special movement execute"),
};

bool verify_special_movement_hook_fingerprint(std::uintptr_t libg_base) {
  return verify_native_binary(g_libg_path, cr_special_movement_runtime::kExpectedLibgSha256,
                              cr_special_movement_runtime::kExpectedLibgBuildId) &&
         verify_native_hook_table(libg_base, g_special_movement_hook_specs, 1);
}

bool install_special_movement_runtime_hook(std::uintptr_t libg_base) {
  const bool attested = verify_special_movement_hook_fingerprint(libg_base);
  g_special_movement_hook_attested.store(attested, std::memory_order_release);
  const bool installed = attested && install_native_hook_table(libg_base, g_special_movement_hook_specs, 1);
  g_special_movement_hook_installed.store(installed, std::memory_order_release);
  LOGI("special_movement hooks installed=%d ring=%zu", installed ? 1 : 0, kSpecialMovementRingCapacity);
  return installed;
}
