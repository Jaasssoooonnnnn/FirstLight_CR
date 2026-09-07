// Exact-build, allowlisted LogicCharacter state-transition telemetry.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc;
//   * cr_replay_probe.cpp supplies pointer/identity validators, the inline
//     hook installer, JSON helpers, binary attestation, and logging;
//   * begin_character_state_runtime_epoch is called while combat capture is
//     inactive, before the new combat epoch is armed.
//
// This envelope deliberately preserves raw native state integers and exact
// callers.  It does not name them as dash/jump/warp/effect stages until a
// card-scoped live golden proves the corresponding start/finish boundaries.

#include "character_state_runtime_layout.h"
#include "probe_hook.h"
#include "epoch_ring.h"
#include "event_ring_json.h"

constexpr std::size_t kCharacterStateRingCapacity = 2048;

struct CharacterStateRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  std::int32_t previous_state = -1;
  std::int32_t requested_state = -1;
  std::int32_t committed_state = -1;
  CombatEntityFact source_before;
  CombatEntityFact source_after;
  bool position_changed = false;
  bool complete_context = false;
};

EpochRing<CharacterStateRawRecord, kCharacterStateRingCapacity> g_character_state_ring;
std::atomic<bool> g_character_state_hook_attested{false};
std::atomic<bool> g_character_state_hook_installed{false};

using CharacterStateSetter = void (*)(void *, std::int32_t);

CharacterStateSetter g_original_character_state_setter = nullptr;

bool character_state_card_is_allowlisted(std::int32_t card_id) {
  switch (card_id) {
  case cr_character_state_runtime::kBanditCardId:
  case cr_character_state_runtime::kMegaKnightCardId:
  case cr_character_state_runtime::kEliteArcherCardId:
  case cr_character_state_runtime::kEliteArcherHeroFormCardId:
  case cr_character_state_runtime::kGoldenKnightCardId:
  case cr_character_state_runtime::kSuperHogCardId:
  case cr_character_state_runtime::kLittlePrinceCardId:
  case cr_character_state_runtime::kBossBanditCardId:
  case cr_character_state_runtime::kChampionGuardCardId:
    return true;
  default:
    return false;
  }
}

std::uintptr_t character_state_caller_offset(const void *address) {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
  return address == nullptr || raw < g_libg_base ? 0 : raw - g_libg_base;
}

void begin_character_state_runtime_epoch(std::uint64_t generation, std::uint64_t state_epoch) {
  g_character_state_ring.begin(generation, state_epoch);
}

void reject_character_state_record(const CombatCaptureContext &context) {
  g_character_state_ring.reject(context.generation, context.state_epoch);
}

std::uint64_t publish_character_state_record(CharacterStateRawRecord record) {
  return g_character_state_ring.publish(record);
}

bool same_character_state_entity(const CombatEntityFact &before, const CombatEntityFact &after) {
  return before.validated && before.present && after.validated && after.present &&
         before.native_object_id == after.native_object_id && before.owner == after.owner &&
         before.object_index == after.object_index && before.secondary_index == after.secondary_index &&
         before.card_id == after.card_id && before.object_kind == after.object_kind;
}

void character_state_setter_hook(void *character, std::int32_t requested_state) {
  if (!g_character_state_hook_installed.load(std::memory_order_acquire)) {
    g_original_character_state_setter(character, requested_state);
    return;
  }

  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    g_original_character_state_setter(character, requested_state);
    return;
  }

  const bool readable_before = process_range_is_readable(character, cr_character_state_runtime::kMinimumCharacterBytes);
  const std::int32_t raw_card_id = readable_before ? read_object_field<std::int32_t>(character, 0xac) : 0;
  if (!character_state_card_is_allowlisted(raw_card_id)) {
    // The setter is shared by all LogicCharacters.  Non-allowlisted
    // callbacks are outside this capability, not rejected evidence.
    g_original_character_state_setter(character, requested_state);
    return;
  }

  CombatEntityFact source_before;
  capture_combat_entity(character, context.object_manager, &source_before);
  const std::int32_t previous_state =
      readable_before ? read_object_field<std::int32_t>(character, cr_character_state_runtime::kCharacterStateOffset)
                      : -1;
  const void *caller_address = __builtin_return_address(0);
  const std::uintptr_t caller = character_state_caller_offset(caller_address);
  const bool exact = context.tick >= 0 && readable_before && source_before.validated && source_before.present &&
                     source_before.card_id == raw_card_id && source_before.object_kind == 5 &&
                     source_before.native_object_id != 0 && source_before.owner >= 0 && source_before.owner <= 1 &&
                     previous_state >= cr_character_state_runtime::kMinimumState &&
                     previous_state <= cr_character_state_runtime::kMaximumState &&
                     requested_state >= cr_character_state_runtime::kMinimumState &&
                     requested_state <= cr_character_state_runtime::kMaximumState && caller != 0 &&
                     libg_address_has_segment_flags(caller_address, sizeof(std::uint32_t), PF_X);

  g_original_character_state_setter(character, requested_state);

  if (!exact) {
    reject_character_state_record(context);
    return;
  }

  const bool readable_after = process_range_is_readable(character, cr_character_state_runtime::kMinimumCharacterBytes);
  CombatEntityFact source_after;
  capture_combat_entity(character, context.object_manager, &source_after);
  const std::int32_t committed_state =
      readable_after ? read_object_field<std::int32_t>(character, cr_character_state_runtime::kCharacterStateOffset)
                     : -1;
  if (!readable_after || !same_character_state_entity(source_before, source_after) ||
      committed_state != requested_state || committed_state < cr_character_state_runtime::kMinimumState ||
      committed_state > cr_character_state_runtime::kMaximumState) {
    reject_character_state_record(context);
    return;
  }
  if (previous_state == committed_state) {
    // f18654 returns immediately for a no-op request.  It is not a
    // transition and therefore does not consume a ring sequence.
    return;
  }

  CharacterStateRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.hook_offset = cr_character_state_runtime::kStateSetterOffset;
  record.caller_offset = caller;
  record.previous_state = previous_state;
  record.requested_state = requested_state;
  record.committed_state = committed_state;
  record.source_before = source_before;
  record.source_after = source_after;
  record.position_changed =
      source_before.position_x != source_after.position_x || source_before.position_y != source_after.position_y;
  record.complete_context = true;
  publish_character_state_record(record);
}

bool append_character_state_runtime_events_json(std::uint64_t generation, std::uint64_t state_epoch,
                                                std::int32_t observation_tick, char *response,
                                                std::size_t response_size, std::size_t *used) {
  return append_epoch_ring_events(
      g_character_state_ring,
      {cr_character_state_runtime::kSchema, "allowlisted-native-character-state-hook-ring",
       "sha-build-id-prologue-abi-kind5-card-identity-committed-state"},
      generation, state_epoch, observation_tick, g_character_state_hook_attested.load(std::memory_order_acquire),
      g_character_state_hook_installed.load(std::memory_order_acquire), response, response_size, used,
      [](JsonWriter &json, const CharacterStateRawRecord &record, bool first) {
        if (record.previous_state == record.committed_state || record.requested_state != record.committed_state ||
            !character_state_card_is_allowlisted(record.source_before.card_id)) {
          return false;
        }
        json.append("%s{", first ? "" : ",");
        json.field("sequence", record.sequence);
        json.field("tick", record.tick);
        json.field("kind", "native_character_state_transition");
        json.field("hookOffset", record.hook_offset);
        json.field("callerOffset", record.caller_offset);
        json.field("previousState", record.previous_state);
        json.field("requestedState", record.requested_state);
        json.field("committedState", record.committed_state);
        json.optional("sourceBefore", true, append_combat_entity_fact, record.source_before);
        json.optional("sourceAfter", true, append_combat_entity_fact, record.source_after);
        json.field("positionChanged", record.position_changed);
        json.field("completeContext", record.complete_context);
        return json.end_object();
      });
}

const NativeHookSpec g_character_state_hook_specs[] = {
    native_hook(cr_character_state_runtime::kStateSetterOffset, cr_character_state_runtime::kStateSetterPrologue,
                &character_state_setter_hook, &g_original_character_state_setter, "allowlisted character state setter"),
};

bool verify_character_state_hook_fingerprint(std::uintptr_t libg_base) {
  return verify_native_binary(g_libg_path, cr_character_state_runtime::kExpectedLibgSha256,
                              cr_character_state_runtime::kExpectedLibgBuildId) &&
         verify_native_hook_table(libg_base, g_character_state_hook_specs, 1);
}

bool install_character_state_runtime_hook(std::uintptr_t libg_base) {
  const bool attested = verify_character_state_hook_fingerprint(libg_base);
  g_character_state_hook_attested.store(attested, std::memory_order_release);
  const bool installed = attested && install_native_hook_table(libg_base, g_character_state_hook_specs, 1);
  g_character_state_hook_installed.store(installed, std::memory_order_release);
  LOGI("character_state hooks installed=%d ring=%zu", installed ? 1 : 0, kCharacterStateRingCapacity);
  return installed;
}
