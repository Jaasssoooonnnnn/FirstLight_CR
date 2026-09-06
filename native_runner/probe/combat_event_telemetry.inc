// Exact-build combat-event capture for Null's Royale 15.535.13.
//
// This file is included by cr_replay_probe.cpp after the native object/component
// readers.  Hot hooks never allocate and never take the control/observation
// mutex.  They publish pointer-free facts to a fixed-capacity sequence ring;
// identity resolution and JSON encoding happen only at observation time.

#include "combat_deployment_layout.h"
#include "combat_damage_capture.h"
#include "combat_impact_capture.h"
#include "probe_hook.h"
#include "probe_json.h"

constexpr const char *kCombatEventSchema = "native-combat-events.v1";
constexpr const char *kCombatExpectedLibgSha256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783";
constexpr const char *kCombatExpectedLibgBuildId = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";
constexpr std::size_t kCombatEventRingCapacity = 1024;
// Projectile terminal markers can outlive the observable event ring because the
// stock object manager does not necessarily make a projectile absent inside the
// remove hook.  A long, projectile-heavy match routinely crosses 256 terminal
// objects before the epoch ends, so that old bound rejected otherwise complete
// captures.  Keep this storage epoch-bounded and large enough for a full match;
// rejectedCaptureCount remains the fail-closed guard if the bound is ever hit.
constexpr std::size_t kCombatCausalMarkerCapacity = 4096;
constexpr std::int32_t kCombatMaxDeckSlots = 8;
constexpr std::int32_t kCombatUnsetInt = (-2147483647 - 1);

constexpr std::uintptr_t kCombatDamageOffset = 0x00f642c4;
constexpr std::uintptr_t kCombatHealOffset = 0x00f64d14;
constexpr std::uintptr_t kCombatConsumeCardOffset = 0x00f38a68;
constexpr std::uintptr_t kCombatSpawnOffset = 0x00f25b8c;
constexpr std::uintptr_t kCombatRemoveOffset = 0x00f25d90;
constexpr std::uintptr_t kCombatDeflectOffset = 0x00f1629c;
constexpr std::uintptr_t kCombatImpactFunctionOffset = 0x00f29514;
constexpr std::uintptr_t kCombatImpactBoundaryOffset = 0x00f2981c;
constexpr std::uintptr_t kCombatProjectileTerminalOffset = 0x00f28470;
constexpr std::uintptr_t kCombatExpireSourceLostCallOffset = 0x00f2a11c;
constexpr std::uintptr_t kCombatExpireSourceLostReturnOffset = 0x00f2a120;
constexpr std::uintptr_t kCombatExpireCancelCallOffset = 0x00f2a148;
constexpr std::uintptr_t kCombatExpireCancelReturnOffset = 0x00f2a14c;
constexpr std::uintptr_t kCombatHealEffectReturnOffset = 0x00f20df0;
constexpr std::uintptr_t kCombatHealProjectileReturnOffset = 0x00f2b374;
constexpr std::uintptr_t kCombatHealAreaReturnOffset = 0x00f319a4;

constexpr std::uint8_t kCombatDamagePrologue[16] = {
    0xff, 0x43, 0x03, 0xd1, 0xfd, 0x7b, 0x07, 0xa9, 0xfc, 0x6f, 0x08, 0xa9, 0xfa, 0x67, 0x09, 0xa9,
};
constexpr std::uint8_t kCombatHealPrologue[16] = {
    0xff, 0xc3, 0x01, 0xd1, 0xfd, 0x7b, 0x01, 0xa9, 0xfc, 0x6f, 0x02, 0xa9, 0xfa, 0x67, 0x03, 0xa9,
};
constexpr std::uint8_t kCombatConsumeCardPrologue[16] = {
    0xff, 0x83, 0x03, 0xd1, 0xfd, 0x7b, 0x08, 0xa9, 0xfc, 0x6f, 0x09, 0xa9, 0xfa, 0x67, 0x0a, 0xa9,
};
constexpr std::uint8_t kCombatSpawnPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf7, 0x0b, 0x00, 0xf9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
constexpr std::uint8_t kCombatRemovePrologue[16] = {
    0xfd, 0x7b, 0xbd, 0xa9, 0xf5, 0x0b, 0x00, 0xf9, 0xf4, 0x4f, 0x02, 0xa9, 0xfd, 0x03, 0x00, 0x91,
};
constexpr std::uint8_t kCombatDeflectPrologue[16] = {
    0xff, 0x83, 0x02, 0xd1, 0xfd, 0x7b, 0x04, 0xa9, 0xfc, 0x6f, 0x05, 0xa9, 0xfa, 0x67, 0x06, 0xa9,
};
constexpr std::uint8_t kCombatImpactPrologue[16] = {
    0xff, 0xc3, 0x02, 0xd1, 0xfd, 0x7b, 0x05, 0xa9, 0xfc, 0x6f, 0x06, 0xa9, 0xfa, 0x67, 0x07, 0xa9,
};
constexpr std::uint8_t kCombatTerminalPrologue[16] = {
    0xfd, 0x7b, 0xbe, 0xa9, 0xf3, 0x0b, 0x00, 0xf9, 0xfd, 0x03, 0x00, 0x91, 0x08, 0x24, 0x40, 0xf9,
};

enum class CombatRawKind : std::uint8_t {
  Damage = 1,
  Death = 2,
  Heal = 3,
  Spawn = 4,
  ProjectileSpawn = 5,
  Despawn = 6,
  ShieldDamage = 7,
  ShieldBreak = 8,
  ProjectileImpact = 9,
  ProjectileDeflect = 10,
  ProjectileExpire = 11,
  ProjectileTerminal = 12,
  CardPlay = 13,
};

enum class CombatPool : std::uint8_t {
  None = 0,
  Hitpoints = 1,
  BuiltInShield = 2,
  BuffShield = 3,
};

enum class CombatTerminalReason : std::uint8_t {
  None = 0,
  ObjectImpact = 1,
  SourceLost = 2,
  NonimpactCancel = 3,
  Unknown = 4,
};

// A direct producer scope is deliberately separate from cause_sequence.
// cause_sequence is reserved for same-subject combat transitions (damage ->
// death, terminal -> despawn), whereas a nested spawn has a different child
// identity. This records only an exact native dynamic extent; it never joins
// objects by owner, card, position, or tick.
enum class CombatSpawnProvenanceKind : std::uint8_t {
  None = 0,
  AreaTickNestedSpawn = 1,
  ActionSpawnToLocation = 2,
};

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

struct CombatDeploymentContext {
  bool validated = false;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::uint64_t deployment_sequence = 0;
  std::int32_t owner = -1;
  std::uint32_t played_card_global_id = 0;
  std::uint32_t effective_card_global_id = 0;
  std::uint32_t card_parameter = 0;
  std::int32_t deck_slot = -1;
  std::int32_t cost = -1;
  std::int32_t form_code = -1;
  std::int32_t consumed_tick = -1;
};

struct CombatSpawnProvenance {
  CombatSpawnProvenanceKind kind = CombatSpawnProvenanceKind::None;
  std::uintptr_t hook_offset = 0;
  std::int32_t source_data_global_id = 0;
};

struct CombatRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::uint64_t cause_sequence = 0;
  std::int32_t tick = -1;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  CombatRawKind kind = CombatRawKind::Damage;
  CombatPool pool = CombatPool::None;
  CombatTerminalReason terminal_reason = CombatTerminalReason::None;
  bool lethal = false;
  CombatDeploymentContext deployment;
  CombatSpawnProvenance spawn_provenance;
  CombatEntityFact target;
  CombatEntityFact immediate_source;
  CombatEntityFact source;
  CombatEntityFact projectile;
  CombatEntityFact related;
  CombatEntityFact route_source_before;
  CombatEntityFact route_target_before;
  CombatEntityFact route_source_after;
  CombatEntityFact route_target_after;
  std::int32_t requested_amount = kCombatUnsetInt;
  std::int32_t actual_amount = kCombatUnsetInt;
  std::int32_t pre_hp = kCombatUnsetInt;
  std::int32_t post_hp = kCombatUnsetInt;
  std::int32_t pre_builtin_shield = kCombatUnsetInt;
  std::int32_t post_builtin_shield = kCombatUnsetInt;
  std::int32_t pre_buff_shield = kCombatUnsetInt;
  std::int32_t post_buff_shield = kCombatUnsetInt;
  std::int32_t destination_before_x = kCombatUnsetInt;
  std::int32_t destination_before_y = kCombatUnsetInt;
  std::int32_t destination_after_x = kCombatUnsetInt;
  std::int32_t destination_after_y = kCombatUnsetInt;
};

struct CombatRingSlot {
  std::atomic<std::uint64_t> published_sequence{0};
  CombatRawRecord record;
};

struct CombatCausalMarker {
  bool used = false;
  std::uint64_t state_epoch = 0;
  std::uint64_t sequence = 0;
  std::uint32_t native_object_id = 0;
  CombatTerminalReason terminal_reason = CombatTerminalReason::None;
};

struct CombatCaptureContext {
  void *game_manager = nullptr;
  void *object_manager = nullptr;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
};

CombatRingSlot g_combat_event_ring[kCombatEventRingCapacity];
CombatCausalMarker g_combat_death_markers[kCombatCausalMarkerCapacity];
CombatCausalMarker g_combat_terminal_markers[kCombatCausalMarkerCapacity];
std::atomic<std::uint64_t> g_combat_next_sequence{1};
std::atomic<std::uint64_t> g_combat_next_deployment_sequence{1};
std::atomic<std::uint64_t> g_combat_epoch_first_sequence{1};
std::atomic<std::uint64_t> g_combat_overflow_count{0};
std::atomic<std::uint64_t> g_combat_rejected_capture_count{0};
std::atomic<void *> g_combat_game_manager{nullptr};
std::atomic<void *> g_combat_object_manager{nullptr};
std::atomic<std::uint64_t> g_combat_generation{0};
std::atomic<std::uint64_t> g_combat_state_epoch{0};
std::atomic<bool> g_combat_epoch_active{false};
std::atomic<bool> g_combat_hooks_attested{false};
std::atomic<bool> g_combat_hooks_installed{false};
std::atomic<bool> g_combat_hooks_enabled{false};
thread_local CombatDeploymentContext g_combat_active_deployment;
thread_local CombatDeploymentContext g_combat_pending_deployments[kPlayerCount];

using CombatDamageFunction = std::int32_t (*)(void *, std::int32_t, std::int32_t, std::int32_t, void *, std::int32_t,
                                              void *, void *, std::uint64_t, std::int32_t *, std::uint64_t,
                                              std::uint64_t);
using CombatHealFunction = void (*)(void *, std::int32_t, std::int32_t, void *, std::int32_t, std::int32_t,
                                    std::int32_t);
using CombatConsumeCardFunction = void (*)(void *, const NativeCardSelection *, std::int32_t, std::int32_t,
                                           std::int32_t, std::int32_t);
using CombatSpawnFunction = void (*)(void *, void *, std::int32_t, std::int32_t);
using CombatRemoveFunction = void (*)(void *, void *);
using CombatDeflectFunction = std::int32_t (*)(void *, void *);
using CombatImpactFunction = std::int32_t (*)(void *, void *, std::int32_t, std::int32_t, std::int32_t);
using CombatTerminalFunction = void (*)(void *);
using CombatRemainingSpawnObserver = void (*)(void *, void *, std::int32_t);
using CombatRemainingSpawnProvenanceObserver = void (*)(void *, void *, CombatRawRecord *);
using CombatRemainingRemoveBeginObserver = bool (*)(void *, void *);
using CombatRemainingRemoveEndObserver = void (*)(bool);

CombatDamageFunction g_original_combat_damage = nullptr;
CombatHealFunction g_original_combat_heal = nullptr;
CombatConsumeCardFunction g_original_combat_consume_card = nullptr;
CombatSpawnFunction g_original_combat_spawn = nullptr;
CombatRemoveFunction g_original_combat_remove = nullptr;
CombatDeflectFunction g_original_combat_deflect = nullptr;
CombatImpactFunction g_original_combat_impact = nullptr;
CombatTerminalFunction g_original_combat_terminal = nullptr;
CombatRemainingSpawnObserver g_combat_remaining_spawn_observer = nullptr;
CombatRemainingSpawnProvenanceObserver g_combat_remaining_spawn_provenance_observer = nullptr;
CombatRemainingRemoveBeginObserver g_combat_remaining_remove_begin_observer = nullptr;
CombatRemainingRemoveEndObserver g_combat_remaining_remove_end_observer = nullptr;

extern "C" void cr_combat_heal_entry();

const char *combat_raw_kind_label(CombatRawKind kind) {
  switch (kind) {
  case CombatRawKind::Damage:
    return "damage";
  case CombatRawKind::Death:
    return "death";
  case CombatRawKind::Heal:
    return "heal";
  case CombatRawKind::Spawn:
    return "spawn";
  case CombatRawKind::ProjectileSpawn:
    return "projectile_spawn";
  case CombatRawKind::Despawn:
    return "despawn";
  case CombatRawKind::ShieldDamage:
    return "shield_damage";
  case CombatRawKind::ShieldBreak:
    return "shield_break";
  case CombatRawKind::ProjectileImpact:
    return "projectile_impact";
  case CombatRawKind::ProjectileDeflect:
    return "projectile_deflect";
  case CombatRawKind::ProjectileExpire:
    return "projectile_expire";
  case CombatRawKind::ProjectileTerminal:
    return "projectile_terminal";
  case CombatRawKind::CardPlay:
    return "card_play";
  }
  return "unknown";
}

const char *combat_pool_label(CombatPool pool) {
  switch (pool) {
  case CombatPool::None:
    return "none";
  case CombatPool::Hitpoints:
    return "hitpoints";
  case CombatPool::BuiltInShield:
    return "built_in_shield";
  case CombatPool::BuffShield:
    return "buff_shield";
  }
  return "none";
}

const char *combat_terminal_reason_label(CombatTerminalReason reason) {
  switch (reason) {
  case CombatTerminalReason::None:
    return "none";
  case CombatTerminalReason::ObjectImpact:
    return "object_impact";
  case CombatTerminalReason::SourceLost:
    return "source_lost";
  case CombatTerminalReason::NonimpactCancel:
    return "nonimpact_cancel";
  case CombatTerminalReason::Unknown:
    return "unknown";
  }
  return "unknown";
}

const char *combat_spawn_provenance_label(CombatSpawnProvenanceKind kind) {
  switch (kind) {
  case CombatSpawnProvenanceKind::AreaTickNestedSpawn:
    return "area_tick_nested_spawn";
  case CombatSpawnProvenanceKind::ActionSpawnToLocation:
    return "action_spawn_to_location";
  case CombatSpawnProvenanceKind::None:
    break;
  }
  return "none";
}

const char *combat_card_form_label(std::int32_t form_code) {
  switch (form_code) {
  case 0:
    return "BasicForm";
  case 1:
    return "EvoForm";
  case 2:
    return "HeroForm";
  case 3:
    return "AutoChessForm";
  case 4:
    return "FormEnd";
  case 5:
    return "FlexSlot";
  }
  return nullptr;
}

bool read_combat_object_vector(void *object_manager, void ***objects_out, std::int32_t *count_out) {
  if (object_manager == nullptr || objects_out == nullptr || count_out == nullptr) {
    return false;
  }
  void **const objects = read_object_field<void **>(object_manager, 0x08);
  const std::int32_t capacity = read_object_field<std::int32_t>(object_manager, 0x10);
  const std::int32_t count = read_object_field<std::int32_t>(object_manager, 0x14);
  if (count < 0 || capacity < count || capacity > 100000 || (count != 0 && objects == nullptr)) {
    return false;
  }
  *objects_out = objects;
  *count_out = count;
  return true;
}

void *combat_object_manager_from_game_manager(void *game_manager) {
  if (game_manager == nullptr) {
    return nullptr;
  }
  void *const world = read_object_field<void *>(game_manager, 0xa8);
  void *const player_zero = world == nullptr ? nullptr : read_object_field<void *>(world, 0xe0);
  return player_zero == nullptr ? nullptr : read_object_field<void *>(player_zero, 0x10);
}

void reset_combat_causal_state() {
  for (CombatCausalMarker &marker : g_combat_death_markers) {
    marker = {};
  }
  for (CombatCausalMarker &marker : g_combat_terminal_markers) {
    marker = {};
  }
  g_combat_active_deployment = {};
  for (CombatDeploymentContext &pending : g_combat_pending_deployments) {
    pending = {};
  }
}

void begin_combat_event_epoch(void *game_manager, std::uint64_t generation, std::uint64_t state_epoch) {
  g_combat_epoch_active.store(false, std::memory_order_release);
  g_combat_game_manager.store(game_manager, std::memory_order_release);
  g_combat_object_manager.store(nullptr, std::memory_order_release);
  g_combat_generation.store(generation, std::memory_order_release);
  g_combat_state_epoch.store(state_epoch, std::memory_order_release);
  reset_combat_causal_state();
  for (CombatRingSlot &slot : g_combat_event_ring) {
    slot.published_sequence.store(0, std::memory_order_release);
  }
  const std::uint64_t first = g_combat_next_sequence.load(std::memory_order_acquire);
  g_combat_epoch_first_sequence.store(first, std::memory_order_release);
  g_combat_overflow_count.store(0, std::memory_order_release);
  g_combat_rejected_capture_count.store(0, std::memory_order_release);
  g_combat_epoch_active.store(game_manager != nullptr, std::memory_order_release);
}

void bind_combat_event_object_manager(void *game_manager) {
  if (game_manager == nullptr || g_combat_game_manager.load(std::memory_order_acquire) != game_manager) {
    return;
  }
  void *const object_manager = combat_object_manager_from_game_manager(game_manager);
  void **objects = nullptr;
  std::int32_t count = 0;
  if (read_combat_object_vector(object_manager, &objects, &count)) {
    (void)objects;
    (void)count;
    g_combat_object_manager.store(object_manager, std::memory_order_release);
  }
}

void end_combat_event_epoch(void *game_manager) {
  if (game_manager != nullptr && g_combat_game_manager.load(std::memory_order_acquire) != game_manager) {
    return;
  }
  g_combat_epoch_active.store(false, std::memory_order_release);
  g_combat_object_manager.store(nullptr, std::memory_order_release);
  g_combat_game_manager.store(nullptr, std::memory_order_release);
}

bool current_combat_capture_context(CombatCaptureContext *result) {
  if (result == nullptr || !g_combat_hooks_enabled.load(std::memory_order_acquire) ||
      !g_combat_epoch_active.load(std::memory_order_acquire)) {
    return false;
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    const std::uint64_t epoch_before = g_combat_state_epoch.load(std::memory_order_acquire);
    CombatCaptureContext value;
    value.game_manager = g_combat_game_manager.load(std::memory_order_acquire);
    value.object_manager = g_combat_object_manager.load(std::memory_order_acquire);
    value.generation = g_combat_generation.load(std::memory_order_acquire);
    value.state_epoch = epoch_before;
    if (value.object_manager == nullptr && value.game_manager != nullptr) {
      value.object_manager = combat_object_manager_from_game_manager(value.game_manager);
    }
    const std::uint64_t epoch_after = g_combat_state_epoch.load(std::memory_order_acquire);
    if (epoch_before != epoch_after || value.game_manager == nullptr || value.generation == 0 ||
        value.state_epoch == 0) {
      continue;
    }
    value.tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(value.game_manager);
    *result = value;
    return true;
  }
  return false;
}

bool resolve_combat_evolution_form(const void *base_card_data, std::int32_t form_code, std::uint32_t *global_id_out) {
  constexpr std::int32_t kMaxCombatFormCandidates = 16;
  if (global_id_out == nullptr || form_code != 1 || !process_range_is_readable(base_card_data, 0x230)) {
    return false;
  }
  NativeVectorView candidates;
  if (!read_native_vector(base_card_data, 0x220, kMaxCombatFormCandidates, &candidates) ||
      (candidates.count != 0 &&
       !process_range_is_readable(candidates.data, static_cast<std::size_t>(candidates.count) * sizeof(void *)))) {
    return false;
  }
  const auto *const items = static_cast<void *const *>(candidates.data);
  const void *selected = nullptr;
  for (std::int32_t index = 0; index < candidates.count; ++index) {
    const void *const candidate = items[index];
    if (!process_range_is_readable(candidate, 0x2b0)) {
      return false;
    }
    const void *const metadata = read_object_field<const void *>(candidate, 0x2a8);
    if (!process_range_is_readable(metadata, 0x68)) {
      return false;
    }
    const std::int32_t candidate_form = read_object_field<std::int32_t>(metadata, 0x64);
    if (candidate_form < 0 || candidate_form > 5) {
      return false;
    }
    if (candidate_form != form_code) {
      continue;
    }
    if (selected != nullptr) {
      return false;
    }
    selected = candidate;
  }
  if (selected == nullptr) {
    return false;
  }
  const std::uint32_t global_id = read_object_field<std::uint32_t>(selected, 0x40);
  if (global_id == 0) {
    return false;
  }
  *global_id_out = global_id;
  return true;
}

bool capture_combat_deployment(const CombatCaptureContext &capture, void *owner_root,
                               const NativeCardSelection *descriptor, CombatDeploymentContext *result) {
  if (result == nullptr) {
    return false;
  }
  *result = {};
  if (capture.game_manager == nullptr || capture.object_manager == nullptr || owner_root == nullptr ||
      !process_range_is_readable(descriptor, 20)) {
    LOGE("deployment validation rejected at capture-context");
    return false;
  }
  NativeCardSelection selection;
  std::memcpy(&selection, descriptor, 20);
  const std::int32_t form_code = static_cast<std::int32_t>(selection.packed & 0x0fU);
  const std::int32_t deck_slot = static_cast<std::int32_t>((selection.packed >> 22) & 0x3fU) - 1;
  const std::int32_t cost = static_cast<std::int32_t>(selection.packed >> 28);
  if (combat_card_form_label(form_code) == nullptr || deck_slot < 0 || deck_slot >= kCombatMaxDeckSlots ||
      selection.card_data == nullptr) {
    LOGE("deployment validation rejected at descriptor: form=%d slot=%d card=%d", form_code, deck_slot,
         selection.card_data != nullptr);
    return false;
  }

  void **objects = nullptr;
  std::int32_t object_count = 0;
  if (!read_combat_object_vector(capture.object_manager, &objects, &object_count)) {
    LOGE("deployment validation rejected at object-vector");
    return false;
  }
  const void *const world = read_object_field<const void *>(capture.game_manager, 0xa8);
  std::int32_t owner = -1;
  PlayerStateView player{};
  for (std::int32_t candidate_owner = 0; candidate_owner < kPlayerCount; ++candidate_owner) {
    PlayerStateView candidate{};
    if (!read_player_state(const_cast<void *>(world), candidate_owner, &candidate)) {
      LOGE("deployment validation rejected at world-player candidate=%d", candidate_owner);
      return false;
    }
    if (candidate.player != owner_root) {
      continue;
    }
    if (owner != -1) {
      LOGE("deployment validation rejected at duplicate-world-player");
      return false;
    }
    owner = candidate_owner;
    player = candidate;
  }
  if (owner < 0 || player.player != owner_root || player.deck_slots.count < 0 ||
      player.deck_slots.count > kCombatMaxDeckSlots) {
    LOGE("deployment validation rejected at player-state: owner=%d player=%d", owner, player.player == owner_root);
    return false;
  }
  CardStateView card{};
  if (!read_card_state(player, deck_slot, false, &card) || card.card_id <= 0 ||
      !process_range_is_readable(selection.card_data, 0x44)) {
    LOGE("deployment validation rejected at unreadable-card-state: "
         "slot=%d cardId=%d readable=%d",
         deck_slot, card.card_id, process_range_is_readable(selection.card_data, 0x44));
    return false;
  }
  const std::uint32_t selection_card_global_id = read_object_field<std::uint32_t>(selection.card_data, 0x40);
  bool exact_mirror_descriptor = false;
  if (card.card_data != selection.card_data && g_card_selection_build != nullptr && card.battle_deck_slot != nullptr) {
    NativeCardSelection rebuilt;
    cr_call_card_selection_build(card.battle_deck_slot, player.player, 0, &rebuilt, g_card_selection_build);
    const cr_combat_deployment::DescriptorIdentity consumed_identity{
        reinterpret_cast<std::uintptr_t>(selection.card_data),
        reinterpret_cast<std::uintptr_t>(selection.secondary_data),
        selection.packed,
    };
    const cr_combat_deployment::DescriptorIdentity rebuilt_identity{
        reinterpret_cast<std::uintptr_t>(rebuilt.card_data),
        reinterpret_cast<std::uintptr_t>(rebuilt.secondary_data),
        rebuilt.packed,
    };
    exact_mirror_descriptor = cr_combat_deployment::exact_mirror_descriptor_match(
        static_cast<std::uint32_t>(card.card_id), consumed_identity, rebuilt_identity);
  }
  if (card.card_data != selection.card_data && !exact_mirror_descriptor) {
    const std::int32_t selection_card_id = process_range_is_readable(selection.card_data, 0x44)
                                               ? read_object_field<std::int32_t>(selection.card_data, 0x40)
                                               : 0;
    const std::int32_t secondary_card_id = process_range_is_readable(selection.secondary_data, 0x44)
                                               ? read_object_field<std::int32_t>(selection.secondary_data, 0x40)
                                               : 0;
    LOGE("deployment validation rejected at card-state: slot=%d "
         "cardMatch=%d cardId=%d selectionCardId=%d "
         "secondaryCardId=%d secondaryRootMatch=%d readable=%d",
         deck_slot, card.card_data == selection.card_data, card.card_id, selection_card_id, secondary_card_id,
         selection.secondary_data == card.card_data, process_range_is_readable(selection.card_data, 0x44));
    return false;
  }
  const std::uint32_t played_card_global_id = static_cast<std::uint32_t>(card.card_id);
  if (played_card_global_id == 0 || (!exact_mirror_descriptor && selection_card_global_id != played_card_global_id)) {
    LOGE("deployment validation rejected at played-id: played=%u "
         "selection=%u card=%d mirror=%d",
         played_card_global_id, selection_card_global_id, card.card_id, exact_mirror_descriptor);
    return false;
  }

  std::uint32_t effective_card_global_id = 0;
  if (exact_mirror_descriptor) {
    effective_card_global_id = selection_card_global_id;
  } else if (form_code == cr_combat_deployment::kHeroFormCode) {
    if (g_card_selection_build == nullptr || card.battle_deck_slot == nullptr) {
      LOGE("HeroForm deployment validation precondition failed: builder=%d slot=%d", g_card_selection_build != nullptr,
           card.battle_deck_slot != nullptr);
      return false;
    }
    NativeCardSelection rebuilt;
    cr_call_card_selection_build(card.battle_deck_slot, player.player, 0, &rebuilt, g_card_selection_build);
    const cr_combat_deployment::DescriptorIdentity consumed_identity{
        reinterpret_cast<std::uintptr_t>(selection.card_data),
        reinterpret_cast<std::uintptr_t>(selection.secondary_data),
        selection.packed,
    };
    const cr_combat_deployment::DescriptorIdentity rebuilt_identity{
        reinterpret_cast<std::uintptr_t>(rebuilt.card_data),
        reinterpret_cast<std::uintptr_t>(rebuilt.secondary_data),
        rebuilt.packed,
    };
    if (!cr_combat_deployment::exact_hero_descriptor_match(consumed_identity, rebuilt_identity)) {
      LOGE("HeroForm descriptor rebuild mismatch: card=%d secondary=%d packed=%08x/%08x",
           selection.card_data == rebuilt.card_data, selection.secondary_data == rebuilt.secondary_data,
           selection.packed, rebuilt.packed);
      return false;
    }
    // HeroForm +0x08 is an exact associated/effective data identity, but
    // its concrete LogicData subtype is not established for this build.
    // Preserve it through the pointer equality proof above and expose no
    // guessed effectiveCardGlobalId.
  } else if (form_code == 0) {
    effective_card_global_id = played_card_global_id;
  } else if (form_code == 1 &&
             !resolve_combat_evolution_form(selection.card_data, form_code, &effective_card_global_id)) {
    return false;
  }
  if (!exact_mirror_descriptor && form_code != cr_combat_deployment::kHeroFormCode &&
      selection.secondary_data != nullptr) {
    if (!process_range_is_readable(selection.secondary_data, 0x44)) {
      return false;
    }
    const std::uint32_t secondary_global_id = read_object_field<std::uint32_t>(selection.secondary_data, 0x40);
    if (secondary_global_id == 0 ||
        (effective_card_global_id != 0 && secondary_global_id != effective_card_global_id)) {
      return false;
    }
    effective_card_global_id = secondary_global_id;
  }

  result->validated = true;
  result->generation = capture.generation;
  result->state_epoch = capture.state_epoch;
  result->deployment_sequence = g_combat_next_deployment_sequence.fetch_add(1, std::memory_order_acq_rel);
  result->owner = owner;
  result->played_card_global_id = played_card_global_id;
  result->effective_card_global_id = effective_card_global_id;
  result->card_parameter = selection.packed;
  result->deck_slot = deck_slot;
  result->cost = cost;
  result->form_code = form_code;
  result->consumed_tick = capture.tick;
  return result->deployment_sequence != 0;
}

bool combat_component_type_is(const void *component, const void *owner, std::int32_t expected_type) {
  if (component == nullptr || owner == nullptr || read_object_field<const void *>(component, 0x08) != owner) {
    return false;
  }
  const void *const vtable = read_object_field<const void *>(component, 0x00);
  if (!libg_address_has_segment_flags(vtable, 0x28, PF_R)) {
    return false;
  }
  const auto getter = read_object_field<NativeComponentTypeGetter>(vtable, 0x20);
  return libg_address_has_segment_flags(reinterpret_cast<const void *>(getter), sizeof(std::uint32_t), PF_X) &&
         getter(component) == expected_type;
}

void capture_combat_visibility(const void *object, CombatEntityFact *fact) {
  if (object == nullptr || fact == nullptr) {
    return;
  }
  void **const components = read_object_field<void **>(object, 0x18);
  const std::int32_t capacity = read_object_field<std::int32_t>(object, 0x20);
  const std::int32_t count = read_object_field<std::int32_t>(object, 0x24);
  if (count < 0 || capacity < count || capacity > kMaxObjectComponents || (count != 0 && components == nullptr)) {
    return;
  }
  if (count <= 3 || components[3] == nullptr) {
    fact->visibility_validated = true;
    fact->invisible_count = 0;
    return;
  }
  const void *const component = components[3];
  if (!combat_component_type_is(component, object, 3)) {
    return;
  }
  const std::int32_t effect_capacity = read_object_field<std::int32_t>(component, 0x20);
  const std::int32_t effect_count = read_object_field<std::int32_t>(component, 0x24);
  const std::int32_t invisible_count = read_object_field<std::int32_t>(component, 0x34);
  if (effect_count < 0 || effect_capacity < effect_count || effect_capacity > kMaxActiveEffects ||
      invisible_count < 0 || invisible_count > effect_count) {
    return;
  }
  fact->visibility_validated = true;
  fact->invisible_count = invisible_count;
}

std::int32_t combat_object_kind(const void *object) {
  if (object == nullptr) {
    return -1;
  }
  const void *const vtable = read_object_field<const void *>(object, 0x00);
  if (!libg_address_has_segment_flags(vtable, 0x18, PF_R)) {
    return -1;
  }
  const auto getter = read_object_field<LogicGameObjectKindGetter>(vtable, 0x10);
  if (!libg_address_has_segment_flags(reinterpret_cast<const void *>(getter), sizeof(std::uint32_t), PF_X)) {
    return -1;
  }
  const std::int32_t kind = getter(object);
  return kind >= 0 && kind < 32 ? kind : -1;
}

void capture_combat_entity(const void *pointer, void *object_manager, CombatEntityFact *result) {
  if (result == nullptr) {
    return;
  }
  *result = {};
  if (pointer == nullptr) {
    result->validated = true;
    return;
  }
  void **objects = nullptr;
  std::int32_t count = 0;
  if (!read_combat_object_vector(object_manager, &objects, &count)) {
    return;
  }
  bool found = false;
  for (std::int32_t index = 0; index < count; ++index) {
    if (objects[index] == pointer) {
      found = true;
      break;
    }
  }
  if (!found) {
    return;
  }
  const std::uint32_t native_object_id = read_object_field<std::uint32_t>(pointer, 0x08);
  if (native_object_id == 0 || !native_object_id_is_unique_in_vector(pointer, objects, count, native_object_id)) {
    return;
  }
  result->validated = true;
  result->present = true;
  result->native_object_id = native_object_id;
  result->owner = read_object_field<std::int32_t>(pointer, 0x78);
  result->object_index = read_object_field<std::int32_t>(pointer, 0x94);
  result->secondary_index = read_object_field<std::int32_t>(pointer, 0x98);
  result->card_id = read_object_field<std::int32_t>(pointer, 0xac);
  result->position_x = read_object_field<std::int32_t>(pointer, 0x7c);
  result->position_y = read_object_field<std::int32_t>(pointer, 0x80);
  result->object_kind = combat_object_kind(pointer);
  capture_combat_visibility(pointer, result);
}

bool exact_combat_projectile(const void *object) {
  if (object == nullptr || read_object_field<const void *>(object, 0x00) !=
                               reinterpret_cast<const void *>(g_libg_base + kProjectileVtableOffset)) {
    return false;
  }
  const void *const vtable = read_object_field<const void *>(object, 0x00);
  const auto kind_getter = read_object_field<LogicGameObjectKindGetter>(vtable, 0x10);
  if (kind_getter != reinterpret_cast<LogicGameObjectKindGetter>(g_libg_base + kProjectileKindGetterOffset) ||
      kind_getter(object) != 4) {
    return false;
  }
  const void *const data = read_object_field<const void *>(object, 0x48);
  const auto type_getter = reinterpret_cast<LogicDataTypeGetter>(g_libg_base + kLogicDataTypeGetterOffset);
  return data != nullptr && type_getter(data) == 0x0a && read_object_field<std::uint32_t>(data, 0x40) != 0;
}

void capture_projectile_routes(const void *projectile, void *object_manager, CombatEntityFact *source,
                               CombatEntityFact *target, std::int32_t *destination_x, std::int32_t *destination_y) {
  if (projectile == nullptr || !exact_combat_projectile(projectile)) {
    return;
  }
  capture_combat_entity(read_object_field<const void *>(projectile, 0x100), object_manager, source);
  capture_combat_entity(read_object_field<const void *>(projectile, 0x108), object_manager, target);
  if (destination_x != nullptr) {
    *destination_x = read_object_field<std::int32_t>(projectile, 0x120);
  }
  if (destination_y != nullptr) {
    *destination_y = read_object_field<std::int32_t>(projectile, 0x124);
  }
}

void select_combat_public_source(const void *immediate_source, const CombatCaptureContext &context,
                                 CombatRawRecord *record) {
  if (record == nullptr) {
    return;
  }
  capture_combat_entity(immediate_source, context.object_manager, &record->immediate_source);
  record->source = record->immediate_source;
  if (!record->immediate_source.validated || !record->immediate_source.present ||
      !exact_combat_projectile(immediate_source)) {
    return;
  }
  record->projectile = record->immediate_source;
  CombatEntityFact controller;
  capture_combat_entity(read_object_field<const void *>(immediate_source, 0x100), context.object_manager, &controller);
  if (controller.validated && controller.present) {
    record->source = controller;
  }
}

bool read_combat_type_two(const void *component, const void *object, std::int32_t *hp, std::int32_t *shield) {
  if (!combat_component_type_is(component, object, 2) || hp == nullptr || shield == nullptr) {
    return false;
  }
  *hp = read_object_field<std::int32_t>(component, 0x10);
  const std::int32_t maximum_hp = read_object_field<std::int32_t>(component, 0x14);
  *shield = read_object_field<std::int32_t>(component, 0x28);
  const std::int32_t maximum_shield = read_object_field<std::int32_t>(component, 0x2c);
  return cr_combat_damage::valid_hitpoint_state(*hp, maximum_hp) && maximum_shield >= 0 && *shield >= 0 &&
         *shield <= maximum_shield;
}

bool read_live_combat_type_two(const void *object, std::int32_t *hp, std::int32_t *shield, std::int32_t *maximum_hp,
                               std::int32_t *maximum_shield, bool *component_vector_valid, bool *type_two_present) {
  if (component_vector_valid == nullptr || type_two_present == nullptr || maximum_hp == nullptr ||
      maximum_shield == nullptr) {
    return false;
  }
  *component_vector_valid = false;
  *type_two_present = false;
  if (object == nullptr || hp == nullptr || shield == nullptr || !process_range_is_readable(object, 0x28)) {
    return false;
  }
  void **const components = read_object_field<void **>(object, 0x18);
  const std::int32_t capacity = read_object_field<std::int32_t>(object, 0x20);
  const std::int32_t count = read_object_field<std::int32_t>(object, 0x24);
  if (count < 0 || capacity < count || capacity > kMaxObjectComponents ||
      (count != 0 && (components == nullptr ||
                      !process_range_is_readable(components, static_cast<std::size_t>(count) * sizeof(void *))))) {
    return false;
  }
  *component_vector_valid = true;
  for (std::int32_t index = 0; index < count; ++index) {
    const void *const current = components[index];
    if (current == nullptr || !process_range_is_readable(current, 0x30) ||
        !combat_component_type_is(current, object, 2)) {
      continue;
    }
    *type_two_present = true;
    *hp = read_object_field<std::int32_t>(current, 0x10);
    *maximum_hp = read_object_field<std::int32_t>(current, 0x14);
    *shield = read_object_field<std::int32_t>(current, 0x28);
    *maximum_shield = read_object_field<std::int32_t>(current, 0x2c);
    return cr_combat_damage::valid_hitpoint_state(*hp, *maximum_hp) && *maximum_shield >= 0 && *shield >= 0 &&
           *shield <= *maximum_shield;
  }
  return false;
}

bool read_combat_buff_shield(const void *object, std::int32_t *current_shield) {
  if (object == nullptr || current_shield == nullptr) {
    return false;
  }
  void **const components = read_object_field<void **>(object, 0x18);
  const std::int32_t capacity = read_object_field<std::int32_t>(object, 0x20);
  const std::int32_t count = read_object_field<std::int32_t>(object, 0x24);
  if (count <= 3 || capacity < count || capacity > kMaxObjectComponents || components == nullptr ||
      components[3] == nullptr || !combat_component_type_is(components[3], object, 3)) {
    return false;
  }
  const void *const component = components[3];
  void **const entries = read_object_field<void **>(component, 0x18);
  const std::int32_t entry_capacity = read_object_field<std::int32_t>(component, 0x20);
  const std::int32_t entry_count = read_object_field<std::int32_t>(component, 0x24);
  if (entry_count < 0 || entry_capacity < entry_count || entry_capacity > kMaxActiveEffects ||
      (entry_count != 0 && entries == nullptr)) {
    return false;
  }
  for (std::int32_t index = 0; index < entry_count; ++index) {
    const void *const entry = entries[index];
    const void *const asset = entry == nullptr ? nullptr : read_object_field<const void *>(entry, 0x18);
    if (asset == nullptr ||
        read_object_field<const void *>(asset, 0x00) !=
            reinterpret_cast<const void *>(g_libg_base + kBuffAssetVtableOffset) ||
        read_object_field<std::int32_t>(asset, 0x140) <= 0) {
      continue;
    }
    const std::int32_t value = read_object_field<std::int32_t>(entry, 0x54);
    if (value < 0) {
      return false;
    }
    *current_shield = value;
    return true;
  }
  return false;
}

CombatRawRecord combat_base_record(const CombatCaptureContext &context, CombatRawKind kind,
                                   std::uintptr_t hook_offset) {
  CombatRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.kind = kind;
  record.hook_offset = hook_offset;
  return record;
}

std::uint64_t publish_combat_record(CombatRawRecord record) {
  const std::uint64_t sequence = g_combat_next_sequence.fetch_add(1, std::memory_order_acq_rel);
  record.sequence = sequence;
  CombatRingSlot &slot = g_combat_event_ring[(sequence - 1) % kCombatEventRingCapacity];
  const std::uint64_t replaced = slot.published_sequence.exchange(0, std::memory_order_acq_rel);
  if (replaced != 0) {
    g_combat_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
  slot.record = record;
  slot.published_sequence.store(sequence, std::memory_order_release);
  return sequence;
}

void store_combat_marker(CombatCausalMarker *markers, std::uint64_t state_epoch, std::uint32_t native_object_id,
                         std::uint64_t sequence, CombatTerminalReason reason = CombatTerminalReason::None) {
  if (markers == nullptr || native_object_id == 0 || sequence == 0) {
    return;
  }
  CombatCausalMarker *selected = nullptr;
  for (std::size_t index = 0; index < kCombatCausalMarkerCapacity; ++index) {
    CombatCausalMarker &marker = markers[index];
    if (marker.used && marker.state_epoch == state_epoch && marker.native_object_id == native_object_id) {
      selected = &marker;
      break;
    }
    if (!marker.used && selected == nullptr) {
      selected = &marker;
    }
  }
  if (selected == nullptr) {
    LOGE("combat capture rejected reason=causal-marker-capacity "
         "epoch=%llu nativeObjectId=%u sequence=%llu",
         static_cast<unsigned long long>(state_epoch), native_object_id, static_cast<unsigned long long>(sequence));
    g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  selected->used = true;
  selected->state_epoch = state_epoch;
  selected->native_object_id = native_object_id;
  selected->sequence = sequence;
  selected->terminal_reason = reason;
}

CombatCausalMarker consume_combat_marker(CombatCausalMarker *markers, std::uint64_t state_epoch,
                                         std::uint32_t native_object_id) {
  CombatCausalMarker result;
  if (markers == nullptr || native_object_id == 0) {
    return result;
  }
  for (std::size_t index = 0; index < kCombatCausalMarkerCapacity; ++index) {
    CombatCausalMarker &marker = markers[index];
    if (!marker.used || marker.state_epoch != state_epoch || marker.native_object_id != native_object_id) {
      continue;
    }
    result = marker;
    marker = {};
    return result;
  }
  return result;
}

bool combat_try_pointer_membership(void *object_manager, const void *object, bool *is_member);

std::int32_t combat_damage_hook(void *component, std::int32_t requested, std::int32_t metadata_two,
                                std::int32_t metadata_three, void *immediate_source, std::int32_t metadata_five,
                                void *auxiliary_six, void *auxiliary_seven, std::uint64_t option_eight,
                                std::int32_t *applied_amount_out, std::uint64_t option_ten,
                                std::uint64_t option_eleven) {
  CombatCaptureContext context;
  CombatRawRecord record;
  const void *target_object = component == nullptr ? nullptr : read_object_field<const void *>(component, 0x08);
  bool capture = current_combat_capture_context(&context);
  if (capture) {
    record = combat_base_record(context, CombatRawKind::Damage, kCombatDamageOffset);
    capture_combat_entity(target_object, context.object_manager, &record.target);
    select_combat_public_source(immediate_source, context, &record);
    capture = record.target.validated && record.target.present &&
              read_combat_type_two(component, target_object, &record.pre_hp, &record.pre_builtin_shield);
    std::int32_t buff_shield = 0;
    if (capture && read_combat_buff_shield(target_object, &buff_shield)) {
      record.pre_buff_shield = buff_shield;
    }
  }

  const std::int32_t lethal = g_original_combat_damage(component, requested, metadata_two, metadata_three,
                                                       immediate_source, metadata_five, auxiliary_six, auxiliary_seven,
                                                       option_eight, applied_amount_out, option_ten, option_eleven);
  if (!capture) {
    return lethal;
  }

  if (!read_combat_type_two(component, target_object, &record.post_hp, &record.post_builtin_shield)) {
    LOGE("combat capture rejected reason=damage-post-state tick=%d "
         "targetNativeObjectId=%u",
         context.tick, record.target.native_object_id);
    g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
    return lethal;
  }
  record.requested_amount = requested;
  record.lethal = (lethal & 1) != 0;
  std::int32_t expected_actual = 0;
  if (record.pre_builtin_shield > 0) {
    record.pool = CombatPool::BuiltInShield;
    expected_actual = record.pre_builtin_shield - record.post_builtin_shield;
  } else if (record.pre_buff_shield > 0) {
    record.pool = CombatPool::BuffShield;
    expected_actual = requested < record.pre_buff_shield ? requested : record.pre_buff_shield;
    if (expected_actual < 0) {
      expected_actual = 0;
    }
    record.post_buff_shield = record.pre_buff_shield - expected_actual;
  } else {
    record.pool = CombatPool::Hitpoints;
    expected_actual = record.pre_hp - record.post_hp;
  }
  const std::int32_t reported_native_actual = applied_amount_out == nullptr ? expected_actual : *applied_amount_out;
  const bool one_hitpoint_survival_clamp =
      record.pool == CombatPool::Hitpoints &&
      cr_combat_damage::is_one_hitpoint_survival_clamp(record.pre_hp, record.post_hp, requested, reported_native_actual,
                                                       record.lethal);
  // Berserker's exact native survival mechanic reports the pre-hit HP as
  // applied damage while retaining one observable HP. The combat event is
  // the state transition, so publish the proved HP delta for this one
  // fully identified clamp rather than poisoning the cumulative ring.
  const std::int32_t native_actual = one_hitpoint_survival_clamp ? expected_actual : reported_native_actual;
  const cr_combat_damage::CaptureDecision capture_decision =
      cr_combat_damage::classify_capture(expected_actual, native_actual, record.lethal);
  if (capture_decision == cr_combat_damage::CaptureDecision::IgnoreNoStateChange) {
    return lethal;
  }
  if (capture_decision == cr_combat_damage::CaptureDecision::RejectInconsistent) {
    LOGE("combat capture rejected reason=damage-amount-mismatch tick=%d "
         "targetNativeObjectId=%u requested=%d nativeActual=%d expectedActual=%d "
         "preHp=%d postHp=%d preBuiltInShield=%d postBuiltInShield=%d "
         "preBuffShield=%d postBuffShield=%d lethal=%d",
         context.tick, record.target.native_object_id, requested, native_actual, expected_actual, record.pre_hp,
         record.post_hp, record.pre_builtin_shield, record.post_builtin_shield, record.pre_buff_shield,
         record.post_buff_shield, record.lethal ? 1 : 0);
    g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
    return lethal;
  }
  record.actual_amount = native_actual;
  const std::uint64_t damage_sequence = publish_combat_record(record);

  if (record.pool == CombatPool::BuiltInShield || record.pool == CombatPool::BuffShield) {
    CombatRawRecord shield_damage = record;
    shield_damage.kind = CombatRawKind::ShieldDamage;
    shield_damage.cause_sequence = damage_sequence;
    shield_damage.lethal = false;
    const std::uint64_t shield_damage_sequence = publish_combat_record(shield_damage);
    const bool broke = record.pool == CombatPool::BuiltInShield
                           ? record.pre_builtin_shield > 0 && record.post_builtin_shield == 0
                           : record.pre_buff_shield > 0 && record.post_buff_shield == 0;
    if (broke) {
      CombatRawRecord shield_break = shield_damage;
      shield_break.kind = CombatRawKind::ShieldBreak;
      shield_break.cause_sequence = shield_damage_sequence;
      publish_combat_record(shield_break);
    }
  }
  if (record.lethal) {
    CombatRawRecord death = record;
    death.kind = CombatRawKind::Death;
    death.cause_sequence = damage_sequence;
    const std::uint64_t death_sequence = publish_combat_record(death);
    store_combat_marker(g_combat_death_markers, context.state_epoch, record.target.native_object_id, death_sequence);
  }
  return lethal;
}

extern "C" void cr_combat_heal_bridge(void *component, std::int32_t requested, std::int32_t modifier,
                                      void *context_pointer, std::int32_t metadata_four, std::int32_t metadata_five,
                                      std::int32_t metadata_six, const void *caller_x19, const void *caller_frame,
                                      const void *caller_return) {
  CombatCaptureContext context;
  CombatRawRecord record;
  const void *target_object = component == nullptr ? nullptr : read_object_field<const void *>(component, 0x08);
  bool capture = current_combat_capture_context(&context);
  if (capture) {
    record = combat_base_record(context, CombatRawKind::Heal, kCombatHealOffset);
    capture_combat_entity(target_object, context.object_manager, &record.target);
    capture = record.target.validated && record.target.present &&
              read_combat_type_two(component, target_object, &record.pre_hp, &record.pre_builtin_shield);
    record.caller_offset = caller_return == nullptr || reinterpret_cast<std::uintptr_t>(caller_return) < g_libg_base
                               ? 0
                               : reinterpret_cast<std::uintptr_t>(caller_return) - g_libg_base;
    const void *immediate_source = nullptr;
    if (record.caller_offset == kCombatHealEffectReturnOffset && caller_x19 != nullptr) {
      immediate_source = read_object_field<const void *>(caller_x19, 0x38);
    } else if (record.caller_offset == kCombatHealProjectileReturnOffset) {
      immediate_source = caller_x19;
    } else if (record.caller_offset == kCombatHealAreaReturnOffset && caller_frame != nullptr) {
      immediate_source = read_object_field<const void *>(static_cast<const std::uint8_t *>(caller_frame) - 0x20, 0);
    }
    if (record.caller_offset == kCombatHealEffectReturnOffset ||
        record.caller_offset == kCombatHealProjectileReturnOffset ||
        record.caller_offset == kCombatHealAreaReturnOffset) {
      select_combat_public_source(immediate_source, context, &record);
    }
  }

  g_original_combat_heal(component, requested, modifier, context_pointer, metadata_four, metadata_five, metadata_six);
  if (!capture) {
    return;
  }
  if (!read_combat_type_two(component, target_object, &record.post_hp, &record.post_builtin_shield)) {
    bool member_after = true;
    const bool membership_known = combat_try_pointer_membership(context.object_manager, target_object, &member_after);
    if (membership_known && !member_after) {
      // Some native heal-like actions replace or remove their subject
      // inside the call.  The pre-state was exact, and the bounded live
      // vector now proves there is no post-state to publish; this is an
      // absent transition, not a dropped heal record.
      return;
    }
    CombatEntityFact target_after;
    if (membership_known && member_after) {
      capture_combat_entity(target_object, context.object_manager, &target_after);
      if (target_after.validated && target_after.present &&
          target_after.native_object_id != record.target.native_object_id) {
        // The address was recycled during the call.  The old target's
        // post-state is absent; never attribute the new object's HP to
        // the historical heal subject.
        return;
      }
      bool component_vector_valid = false;
      bool type_two_present = false;
      std::int32_t live_maximum_hp = 0;
      std::int32_t live_maximum_shield = 0;
      if (target_after.validated && target_after.present &&
          read_live_combat_type_two(target_object, &record.post_hp, &record.post_builtin_shield, &live_maximum_hp,
                                    &live_maximum_shield, &component_vector_valid, &type_two_present)) {
        // The native call replaced its HP component in place.  The
        // current bounded component vector is authoritative for the
        // same live native identity.
      } else if (component_vector_valid && !type_two_present) {
        // The same live object no longer has an HP component, so this
        // heal-like call has no public post-state or positive delta.
        return;
      } else {
        LOGE("combat capture rejected reason=heal-post-state tick=%d "
             "targetNativeObjectId=%u membershipKnown=%d memberAfter=%d "
             "targetAfterValidated=%d targetAfterPresent=%d "
             "targetAfterNativeObjectId=%u componentVectorValid=%d "
             "typeTwoPresent=%d requested=%d modifier=%d "
             "preHp=%d postHp=%d maximumHp=%d preShield=%d "
             "postShield=%d maximumShield=%d",
             context.tick, record.target.native_object_id, membership_known ? 1 : 0, member_after ? 1 : 0,
             target_after.validated ? 1 : 0, target_after.present ? 1 : 0, target_after.native_object_id,
             component_vector_valid ? 1 : 0, type_two_present ? 1 : 0, requested, modifier, record.pre_hp,
             record.post_hp, live_maximum_hp, record.pre_builtin_shield, record.post_builtin_shield,
             live_maximum_shield);
        g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    } else {
      LOGE("combat capture rejected reason=heal-post-state tick=%d "
           "targetNativeObjectId=%u membershipKnown=%d memberAfter=%d",
           context.tick, record.target.native_object_id, membership_known ? 1 : 0, member_after ? 1 : 0);
      g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  record.requested_amount = requested;
  if (record.pre_builtin_shield > 0) {
    record.pool = CombatPool::BuiltInShield;
    record.actual_amount = record.post_builtin_shield - record.pre_builtin_shield;
  } else {
    record.pool = CombatPool::Hitpoints;
    record.actual_amount = record.post_hp - record.pre_hp;
  }
  if (record.actual_amount > 0) {
    publish_combat_record(record);
  }
}

void combat_consume_card_hook(void *owner_root, const NativeCardSelection *descriptor, std::int32_t argument_two,
                              std::int32_t argument_three, std::int32_t argument_four, std::int32_t argument_five) {
  const CombatDeploymentContext previous = g_combat_active_deployment;
  CombatDeploymentContext deployment;
  CombatCaptureContext capture;
  const bool capture_active = current_combat_capture_context(&capture);
  const bool deployment_valid =
      capture_active && capture_combat_deployment(capture, owner_root, descriptor, &deployment);
  if (capture_active && !deployment_valid) {
    g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
  }
  g_combat_active_deployment = deployment_valid ? deployment : CombatDeploymentContext{};
  if (deployment_valid && deployment.owner >= 0 && deployment.owner < kPlayerCount) {
    g_combat_pending_deployments[deployment.owner] = deployment;
  }
  g_original_combat_consume_card(owner_root, descriptor, argument_two, argument_three, argument_four, argument_five);
  if (deployment_valid) {
    CombatRawRecord record = combat_base_record(capture, CombatRawKind::CardPlay, kCombatConsumeCardOffset);
    record.deployment = deployment;
    publish_combat_record(record);
  }
  g_combat_active_deployment = previous;
}

void combat_spawn_hook(void *object_manager, void *object, std::int32_t option, std::int32_t commit_to_live) {
  const void *const caller_return = __builtin_return_address(0);
  g_original_combat_spawn(object_manager, object, option, commit_to_live);
  if (commit_to_live == 0) {
    return;
  }
  if (g_combat_remaining_spawn_observer != nullptr) {
    g_combat_remaining_spawn_observer(object_manager, object, commit_to_live);
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  if (context.object_manager == nullptr) {
    g_combat_object_manager.store(object_manager, std::memory_order_release);
    context.object_manager = object_manager;
  }
  if (context.object_manager != object_manager) {
    return;
  }
  CombatRawRecord record = combat_base_record(context, CombatRawKind::Spawn, kCombatSpawnOffset);
  const std::uintptr_t caller_raw = reinterpret_cast<std::uintptr_t>(caller_return);
  record.caller_offset = caller_return == nullptr || caller_raw < g_libg_base ? 0 : caller_raw - g_libg_base;
  capture_combat_entity(object, object_manager, &record.target);
  if (!record.target.validated || !record.target.present) {
    return;
  }
  // Remaining-runtime owns producer scopes such as an exact AEO tick. It
  // may fill source/immediateSource only after it has independently proved
  // the enclosing producer and both live identities. Generic spawns retain
  // their existing all-absent provenance.
  if (g_combat_remaining_spawn_provenance_observer != nullptr) {
    g_combat_remaining_spawn_provenance_observer(object_manager, object, &record);
  }
  if (g_combat_active_deployment.validated && g_combat_active_deployment.generation == context.generation &&
      g_combat_active_deployment.state_epoch == context.state_epoch) {
    record.deployment = g_combat_active_deployment;
    if (g_combat_active_deployment.owner >= 0 && g_combat_active_deployment.owner < kPlayerCount) {
      g_combat_pending_deployments[g_combat_active_deployment.owner] = {};
    }
  } else if (record.target.owner >= 0 && record.target.owner < kPlayerCount) {
    const CombatDeploymentContext &pending = g_combat_pending_deployments[record.target.owner];
    const bool card_identity_matches =
        record.target.card_id > 0 &&
        (static_cast<std::uint32_t>(record.target.card_id) == pending.played_card_global_id ||
         (pending.effective_card_global_id != 0 &&
          static_cast<std::uint32_t>(record.target.card_id) == pending.effective_card_global_id));
    if (pending.validated && pending.generation == context.generation && pending.state_epoch == context.state_epoch &&
        pending.consumed_tick == context.tick && card_identity_matches) {
      record.deployment = pending;
    }
  }
  if (exact_combat_projectile(object)) {
    record.kind = CombatRawKind::ProjectileSpawn;
    record.projectile = record.target;
    capture_projectile_routes(object, object_manager, &record.route_source_after, &record.route_target_after,
                              &record.destination_after_x, &record.destination_after_y);
    record.source = record.route_source_after;
  }
  publish_combat_record(record);
}

bool combat_try_pointer_membership(void *object_manager, const void *object, bool *is_member) {
  if (is_member == nullptr) {
    return false;
  }
  *is_member = false;
  void **objects = nullptr;
  std::int32_t count = 0;
  if (!read_combat_object_vector(object_manager, &objects, &count)) {
    return false;
  }
  for (std::int32_t index = 0; index < count; ++index) {
    if (objects[index] == object) {
      *is_member = true;
      break;
    }
  }
  return true;
}

void combat_remove_hook(void *object_manager, void *object) {
  const bool remaining_capture = g_combat_remaining_remove_begin_observer != nullptr &&
                                 g_combat_remaining_remove_begin_observer(object_manager, object);
  CombatCaptureContext context;
  CombatRawRecord record;
  bool capture = current_combat_capture_context(&context) && context.object_manager == object_manager;
  if (capture) {
    record = combat_base_record(context, CombatRawKind::Despawn, kCombatRemoveOffset);
    capture_combat_entity(object, object_manager, &record.target);
    capture = record.target.validated && record.target.present;
    if (capture && exact_combat_projectile(object)) {
      record.projectile = record.target;
      capture_projectile_routes(object, object_manager, &record.route_source_before, &record.route_target_before,
                                &record.destination_before_x, &record.destination_before_y);
      record.source = record.route_source_before;
    }
  }
  g_original_combat_remove(object_manager, object);
  bool member_after = false;
  const bool membership_known = combat_try_pointer_membership(object_manager, object, &member_after);
  const bool absent_after = membership_known && !member_after;
  if (remaining_capture && g_combat_remaining_remove_end_observer != nullptr) {
    g_combat_remaining_remove_end_observer(absent_after);
  }
  if (!capture || !absent_after) {
    return;
  }
  const CombatCausalMarker death =
      consume_combat_marker(g_combat_death_markers, context.state_epoch, record.target.native_object_id);
  const CombatCausalMarker terminal =
      consume_combat_marker(g_combat_terminal_markers, context.state_epoch, record.target.native_object_id);
  if (death.used) {
    record.cause_sequence = death.sequence;
  } else if (terminal.used) {
    record.cause_sequence = terminal.sequence;
    record.terminal_reason = terminal.terminal_reason;
  }
  publish_combat_record(record);
}

std::int32_t combat_deflect_hook(void *area_effect, void *projectile) {
  CombatCaptureContext context;
  CombatRawRecord record;
  bool capture = current_combat_capture_context(&context) && exact_combat_projectile(projectile);
  if (capture) {
    record = combat_base_record(context, CombatRawKind::ProjectileDeflect, kCombatDeflectOffset);
    capture_combat_entity(projectile, context.object_manager, &record.projectile);
    capture_combat_entity(area_effect, context.object_manager, &record.related);
    const void *const creator = area_effect == nullptr ? nullptr : read_object_field<const void *>(area_effect, 0x130);
    capture_combat_entity(creator, context.object_manager, &record.source);
    capture_projectile_routes(projectile, context.object_manager, &record.route_source_before,
                              &record.route_target_before, &record.destination_before_x, &record.destination_before_y);
    capture = record.projectile.validated && record.projectile.present;
  }
  const std::int32_t result = g_original_combat_deflect(area_effect, projectile);
  if (capture && (result & 1) != 0) {
    capture_projectile_routes(projectile, context.object_manager, &record.route_source_after,
                              &record.route_target_after, &record.destination_after_x, &record.destination_after_y);
    record.target = record.route_target_after;
    publish_combat_record(record);
  }
  return result;
}

std::int32_t combat_impact_hook(void *projectile, void *candidate, std::int32_t collision_x, std::int32_t collision_y,
                                std::int32_t collision_z) {
  (void)collision_z;
  CombatCaptureContext context;
  CombatRawRecord record;
  std::int32_t pre_count = -1;
  std::int32_t pre_capacity = -1;
  bool capture = current_combat_capture_context(&context) && exact_combat_projectile(projectile);
  if (capture) {
    record = combat_base_record(context, CombatRawKind::ProjectileImpact, kCombatImpactBoundaryOffset);
    capture_combat_entity(projectile, context.object_manager, &record.projectile);
    capture_combat_entity(candidate, context.object_manager, &record.target);
    capture_projectile_routes(projectile, context.object_manager, &record.route_source_before,
                              &record.route_target_before, &record.destination_before_x, &record.destination_before_y);
    record.source = record.route_source_before;
    record.destination_after_x = collision_x;
    record.destination_after_y = collision_y;
    pre_capacity = read_object_field<std::int32_t>(projectile, 0x68);
    pre_count = read_object_field<std::int32_t>(projectile, 0x6c);
    capture = record.projectile.validated && record.projectile.present && record.target.validated &&
              record.target.present && pre_count >= 0 && pre_capacity >= pre_count && pre_capacity <= 100000;
  }
  const std::int32_t result = g_original_combat_impact(projectile, candidate, collision_x, collision_y, collision_z);
  if (!capture || (result & 1) == 0) {
    return result;
  }
  const std::int32_t post_count = read_object_field<std::int32_t>(projectile, 0x6c);
  const std::int32_t post_capacity = read_object_field<std::int32_t>(projectile, 0x68);
  const std::int32_t *const processed = read_object_field<const std::int32_t *>(projectile, 0x60);
  const bool processed_readable =
      processed != nullptr && post_count > 0 && post_count <= 100000 &&
      process_range_is_readable(processed, static_cast<std::size_t>(post_count) * sizeof(*processed));
  const cr_combat_impact::ProcessedTargetDecision target_decision =
      processed_readable ? cr_combat_impact::classify_processed_target_append(
                               processed, pre_count, post_count, post_capacity, record.target.native_object_id)
                         : cr_combat_impact::ProcessedTargetDecision::RejectBounds;
  if (target_decision != cr_combat_impact::ProcessedTargetDecision::Publish) {
    const std::int32_t last_processed = processed_readable ? processed[post_count - 1] : -1;
    LOGE("combat capture rejected reason=impact-processed-target tick=%d "
         "projectileNativeObjectId=%u targetNativeObjectId=%u "
         "preCount=%d preCapacity=%d postCount=%d postCapacity=%d "
         "processedReadable=%d decision=%d lastProcessed=%d result=%d",
         context.tick, record.projectile.native_object_id, record.target.native_object_id, pre_count, pre_capacity,
         post_count, post_capacity, processed_readable ? 1 : 0, static_cast<std::int32_t>(target_decision),
         last_processed, result);
    g_combat_rejected_capture_count.fetch_add(1, std::memory_order_relaxed);
    return result;
  }
  const std::uint64_t sequence = publish_combat_record(record);
  if (read_object_field<std::uint8_t>(projectile, 0x170) != 0) {
    store_combat_marker(g_combat_terminal_markers, context.state_epoch, record.projectile.native_object_id, sequence,
                        CombatTerminalReason::ObjectImpact);
  }
  return result;
}

void combat_terminal_hook(void *projectile) {
  CombatCaptureContext context;
  CombatRawRecord record;
  const void *const caller_return = __builtin_return_address(0);
  const std::uintptr_t caller_offset =
      caller_return == nullptr || reinterpret_cast<std::uintptr_t>(caller_return) < g_libg_base
          ? 0
          : reinterpret_cast<std::uintptr_t>(caller_return) - g_libg_base;
  bool capture = current_combat_capture_context(&context) && exact_combat_projectile(projectile) &&
                 read_object_field<std::uint8_t>(projectile, 0x170) == 0;
  if (capture) {
    CombatRawKind kind = CombatRawKind::ProjectileTerminal;
    CombatTerminalReason reason = CombatTerminalReason::Unknown;
    std::uintptr_t hook_offset = kCombatProjectileTerminalOffset;
    if (caller_offset == kCombatExpireSourceLostReturnOffset) {
      kind = CombatRawKind::ProjectileExpire;
      reason = CombatTerminalReason::SourceLost;
      hook_offset = kCombatExpireSourceLostCallOffset;
    } else if (caller_offset == kCombatExpireCancelReturnOffset) {
      kind = CombatRawKind::ProjectileExpire;
      reason = CombatTerminalReason::NonimpactCancel;
      hook_offset = kCombatExpireCancelCallOffset;
    }
    record = combat_base_record(context, kind, hook_offset);
    record.caller_offset = caller_offset;
    record.terminal_reason = reason;
    capture_combat_entity(projectile, context.object_manager, &record.projectile);
    capture_projectile_routes(projectile, context.object_manager, &record.route_source_before,
                              &record.route_target_before, &record.destination_before_x, &record.destination_before_y);
    record.source = record.route_source_before;
    record.target = record.route_target_before;
    capture = record.projectile.validated && record.projectile.present;
  }
  g_original_combat_terminal(projectile);
  if (capture && read_object_field<std::uint8_t>(projectile, 0x170) != 0) {
    record.destination_after_x = read_object_field<std::int32_t>(projectile, 0x120);
    record.destination_after_y = read_object_field<std::int32_t>(projectile, 0x124);
    const std::uint64_t sequence = publish_combat_record(record);
    store_combat_marker(g_combat_terminal_markers, context.state_epoch, record.projectile.native_object_id, sequence,
                        record.terminal_reason);
  }
}

bool append_combat_deployment_context(char *response, std::size_t response_size, std::size_t *used,
                                      const CombatDeploymentContext &deployment) {
  JsonWriter json(response, response_size, used);
  if (!deployment.validated) {
    return json.append("null");
  }
  json.begin_object();
  json.field("deploymentSequence", deployment.deployment_sequence);
  json.field("owner", deployment.owner);
  json.field("playedCardGlobalId", deployment.played_card_global_id);
  json.nullable("effectiveCardGlobalId", deployment.effective_card_global_id, deployment.effective_card_global_id != 0);
  json.field("cardParameter", deployment.card_parameter);
  json.field("deckSlot", deployment.deck_slot);
  json.field("cost", deployment.cost);
  json.field("formCode", deployment.form_code);
  json.field("formName", combat_card_form_label(deployment.form_code));
  json.field("consumeHookOffset", kCombatConsumeCardOffset);
  return json.end_object();
}

bool append_combat_entity_fact(char *response, std::size_t response_size, std::size_t *used,
                               const CombatEntityFact &fact) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("validated", fact.validated);
  json.field("present", fact.present);
  json.nullable("nativeObjectId", fact.native_object_id, fact.present);
  char entity_key[96] = {};
  if (fact.present && !format_native_entity_key_json(entity_key, sizeof(entity_key), fact.native_object_id, fact.owner,
                                                     fact.object_index, fact.secondary_index)) {
    return false;
  }
  json.raw("entityKey", fact.present ? entity_key : "null");
  json.nullable("owner", fact.owner, fact.present);
  json.nullable("objectIndex", fact.object_index, fact.present);
  json.nullable("secondaryIndex", fact.secondary_index, fact.present);
  json.nullable("cardId", fact.card_id, fact.present);
  json.nullable("objectKind", fact.object_kind, fact.present);
  if (fact.present) {
    json.begin_array("position");
    json.element(fact.position_x);
    json.element(fact.position_y);
    json.end_array();
  } else {
    json.raw("position", "null");
  }
  json.field("visibilityValidated", fact.visibility_validated);
  json.nullable("invisibleCount", fact.invisible_count, fact.visibility_validated);
  return json.end_object();
}

bool append_combat_spawn_provenance(char *response, std::size_t response_size, std::size_t *used,
                                    const CombatSpawnProvenance &provenance) {
  JsonWriter json(response, response_size, used);
  if (provenance.kind == CombatSpawnProvenanceKind::None) {
    return json.append("null");
  }
  if (provenance.hook_offset == 0 || provenance.source_data_global_id == 0) {
    return false;
  }
  json.begin_object();
  json.field("kind", combat_spawn_provenance_label(provenance.kind));
  json.field("hookOffset", provenance.hook_offset);
  json.field("sourceDataGlobalId", static_cast<std::uint32_t>(provenance.source_data_global_id));
  return json.end_object();
}

void append_combat_destination(JsonWriter &json, const char *name, std::int32_t x, std::int32_t y) {
  if (x == kCombatUnsetInt || y == kCombatUnsetInt) {
    json.raw(name, "null");
  } else {
    json.begin_array(name);
    json.element(x);
    json.element(y);
    json.end_array();
  }
}

bool append_combat_record(JsonWriter &json, const CombatRawRecord &record) {
  json.begin_object();
  json.field("sequence", record.sequence);
  json.field("tick", record.tick);
  json.field("generation", record.generation);
  json.field("stateEpoch", record.state_epoch);
  json.field("kind", combat_raw_kind_label(record.kind));
  json.field("hookOffset", record.hook_offset);
  json.nullable("callerOffset", record.caller_offset, record.caller_offset != 0);
  json.nullable("causeSequence", record.cause_sequence, record.cause_sequence != 0);
  json.field("pool", combat_pool_label(record.pool));
  json.field("terminalReason", combat_terminal_reason_label(record.terminal_reason));
  json.field("lethal", record.lethal);
  json.record("deploymentContext", append_combat_deployment_context, record.deployment);
  json.record("target", append_combat_entity_fact, record.target);
  json.record("immediateSource", append_combat_entity_fact, record.immediate_source);
  json.record("source", append_combat_entity_fact, record.source);
  json.record("projectile", append_combat_entity_fact, record.projectile);
  json.record("related", append_combat_entity_fact, record.related);
  json.record("routeSourceBefore", append_combat_entity_fact, record.route_source_before);
  json.record("routeTargetBefore", append_combat_entity_fact, record.route_target_before);
  json.record("routeSourceAfter", append_combat_entity_fact, record.route_source_after);
  json.record("routeTargetAfter", append_combat_entity_fact, record.route_target_after);
  json.record("spawnProvenance", append_combat_spawn_provenance, record.spawn_provenance);
  json.nullable_ints(
      {
          {"requestedAmount", record.requested_amount},
          {"actualAmount", record.actual_amount},
          {"preHp", record.pre_hp},
          {"postHp", record.post_hp},
          {"preBuiltInShield", record.pre_builtin_shield},
          {"postBuiltInShield", record.post_builtin_shield},
          {"preBuffShield", record.pre_buff_shield},
          {"postBuffShield", record.post_buff_shield},
      },
      kCombatUnsetInt);
  append_combat_destination(json, "destinationBefore", record.destination_before_x, record.destination_before_y);
  append_combat_destination(json, "destinationAfter", record.destination_after_x, record.destination_after_y);
  return json.end_object();
}

bool append_combat_events_json(void *manager, std::uint64_t generation, std::uint64_t state_epoch,
                               std::int32_t observation_tick, char *response, std::size_t response_size,
                               std::size_t *used) {
  if (response == nullptr || used == nullptr) {
    return false;
  }
  const bool installed = g_combat_hooks_installed.load(std::memory_order_acquire);
  const bool identity_matches = installed && g_combat_epoch_active.load(std::memory_order_acquire) &&
                                g_combat_game_manager.load(std::memory_order_acquire) == manager &&
                                g_combat_generation.load(std::memory_order_acquire) == generation &&
                                g_combat_state_epoch.load(std::memory_order_acquire) == state_epoch;
  const std::uint64_t next_sequence = g_combat_next_sequence.load(std::memory_order_acquire);
  const std::uint64_t epoch_first =
      identity_matches ? g_combat_epoch_first_sequence.load(std::memory_order_acquire) : next_sequence;
  const std::uint64_t retained_oldest =
      next_sequence > epoch_first + kCombatEventRingCapacity ? next_sequence - kCombatEventRingCapacity : epoch_first;
  const std::uint64_t transmit_from = retained_oldest;
  const std::uint64_t overflow = identity_matches ? g_combat_overflow_count.load(std::memory_order_acquire) : 0;
  const std::uint64_t rejected = identity_matches ? g_combat_rejected_capture_count.load(std::memory_order_acquire) : 0;
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("ok", true);
  json.field("schema", kCombatEventSchema);
  json.field("generation", generation);
  json.field("stateEpoch", state_epoch);
  json.field("observationTick", observation_tick);
  json.field("capacity", kCombatEventRingCapacity);
  json.field("hookSetAttested", g_combat_hooks_attested.load(std::memory_order_acquire));
  json.field("hookSetInstalled", installed);
  json.begin_object("capability");
  json.field("status", identity_matches ? "derived" : "unavailable");
  json.field("source", "exact-build-native-hook-ring");
  json.field("validation", "libg-sha-build-id-offset-prologue-abi-bounded-identity");
  json.field("confidence", "high");
  json.field("failClosed", true);
  json.end_object();
  json.field("epochFirstSequence", epoch_first);
  json.field("oldestRetainedSequence", retained_oldest);
  json.field("transmitFromSequence", transmit_from);
  json.field("nextSequence", next_sequence);
  json.field("overflowCount", overflow);
  json.field("sequenceGapBeforeOldest", retained_oldest > epoch_first);
  json.field("rejectedCaptureCount", rejected);
  json.field("complete", !installed || (identity_matches && rejected == 0));
  json.begin_array("events");
  if (!json.good()) {
    return false;
  }
  if (identity_matches) {
    bool first = true;
    for (std::uint64_t sequence = transmit_from; sequence < next_sequence; ++sequence) {
      CombatRingSlot &slot = g_combat_event_ring[(sequence - 1) % kCombatEventRingCapacity];
      if (slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        LOGE("combat JSON ring mismatch sequence=%llu published=%llu", static_cast<unsigned long long>(sequence),
             static_cast<unsigned long long>(slot.published_sequence.load(std::memory_order_relaxed)));
        return false;
      }
      const CombatRawRecord record = slot.record;
      if (slot.published_sequence.load(std::memory_order_acquire) != sequence || record.sequence != sequence ||
          record.generation != generation || record.state_epoch != state_epoch || record.tick > observation_tick) {
        LOGE("combat JSON record mismatch sequence=%llu record=%llu tick=%d observation=%d",
             static_cast<unsigned long long>(sequence), static_cast<unsigned long long>(record.sequence), record.tick,
             observation_tick);
        return false;
      }
      if (!first) {
        json.append(",");
      }
      if (!append_combat_record(json, record)) {
        LOGE("combat JSON record failed sequence=%llu used=%zu capacity=%zu", static_cast<unsigned long long>(sequence),
             *used, response_size);
        return false;
      }
      first = false;
    }
  }
  json.end_array();
  return json.end_object();
}

bool verify_combat_hook_fingerprints(std::uintptr_t libg_base, const NativeHookSpec *hooks, std::size_t count) {
  return verify_native_binary(g_libg_path, kCombatExpectedLibgSha256, kCombatExpectedLibgBuildId) &&
         verify_native_hook_table(libg_base, hooks, count);
}

bool install_combat_event_hooks(std::uintptr_t libg_base) {
  const NativeHookSpec hooks[] = {
      native_hook(kCombatDamageOffset, kCombatDamagePrologue, &combat_damage_hook, &g_original_combat_damage,
                  "combat damage"),
      native_hook(kCombatHealOffset, kCombatHealPrologue, &cr_combat_heal_entry, &g_original_combat_heal,
                  "combat heal"),
      native_hook(kCombatConsumeCardOffset, kCombatConsumeCardPrologue, &combat_consume_card_hook,
                  &g_original_combat_consume_card, "combat consume-card deployment context"),
      native_hook(kCombatSpawnOffset, kCombatSpawnPrologue, &combat_spawn_hook, &g_original_combat_spawn,
                  "combat spawn"),
      native_hook(kCombatRemoveOffset, kCombatRemovePrologue, &combat_remove_hook, &g_original_combat_remove,
                  "combat remove"),
      native_hook(kCombatDeflectOffset, kCombatDeflectPrologue, &combat_deflect_hook, &g_original_combat_deflect,
                  "combat projectile deflect"),
      native_hook(kCombatImpactFunctionOffset, kCombatImpactPrologue, &combat_impact_hook, &g_original_combat_impact,
                  "combat projectile impact"),
      native_hook(kCombatProjectileTerminalOffset, kCombatTerminalPrologue, &combat_terminal_hook,
                  &g_original_combat_terminal, "combat projectile terminal"),
  };
  constexpr std::size_t count = sizeof(hooks) / sizeof(hooks[0]);
  const bool attested = verify_combat_hook_fingerprints(libg_base, hooks, count);
  g_combat_hooks_attested.store(attested, std::memory_order_release);
  if (!attested) {
    return false;
  }
  const bool installed = install_native_hook_table(libg_base, hooks, count);
  g_combat_hooks_installed.store(installed, std::memory_order_release);
  g_combat_hooks_enabled.store(installed, std::memory_order_release);
  LOGI("exact-build combat-event hook set installed=%d capacity=%zu", installed ? 1 : 0, kCombatEventRingCapacity);
  return installed;
}
