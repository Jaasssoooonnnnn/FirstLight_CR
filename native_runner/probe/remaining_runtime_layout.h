#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// Exact-build layout and call-site contract for runtime mechanics which are
// not snapshots of HP/position/effects.  These constants are valid only for
// Null's Royale 15.535.13 with the SHA-256/build-id below.
namespace cr_remaining_runtime {

inline constexpr char kSchema[] = "native-remaining-runtime.v1";
inline constexpr char kExactLibgSha256[] = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783";
inline constexpr char kExactLibgBuildId[] = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";

inline constexpr std::int32_t kResourceFixedPointScale = 10000;
inline constexpr std::size_t kLogicDataGlobalIdOffset = 0x40;

// Common LogicGameObject / LogicCharacter layout.
inline constexpr std::size_t kObjectNativeIdOffset = 0x08;
inline constexpr std::size_t kObjectDataOffset = 0x48;
inline constexpr std::size_t kCharacterSpawnUntargetableOffset = 0x174;

// LogicCharacterData resource/lifetime/visibility fields.
inline constexpr std::size_t kManaGenerationEnabledOffset = 0x3a4;
inline constexpr std::size_t kSpawnAttachOffset = 0x3a5;
inline constexpr std::size_t kAttachedCharacterDataOffset = 0x2f0;
inline constexpr std::size_t kManaGenerateLimitOffset = 0x3ac;
inline constexpr std::size_t kManaCollectAmountOffset = 0x3b0;
inline constexpr std::size_t kManaGenerateTimeMsOffset = 0x3b4;
inline constexpr std::size_t kManaOnDeathOffset = 0x3b8;
inline constexpr std::size_t kManaOnDeathForOpponentOffset = 0x3bc;
inline constexpr std::size_t kAllowAreaDamageWhenInvisibleOffset = 0x768;
inline constexpr std::size_t kUntargetableWhenSpawnedOffset = 0x76a;

// The resource object stores elixir in 1/10000 units.
inline constexpr std::size_t kResourceFixedValueOffset = 0x2f8;
inline constexpr std::uintptr_t kResourceWholeDeltaOffset = 0x00f3ba4c;
inline constexpr std::uintptr_t kResourceFixedDeltaOffset = 0x00f3b3d4;
inline constexpr std::uintptr_t kPeriodicResourceCallOffset = 0x00f1acd0;
inline constexpr std::uintptr_t kPeriodicResourceReturnOffset = 0x00f1acd4;
inline constexpr std::uintptr_t kDeathOwnerResourceCallOffset = 0x00f63714;
inline constexpr std::uintptr_t kDeathOwnerResourceReturnOffset = 0x00f63718;
inline constexpr std::uintptr_t kDeathOpponentResourceCallOffset = 0x00f63830;
inline constexpr std::uintptr_t kDeathOpponentResourceReturnOffset = 0x00f63834;

inline constexpr std::uint8_t kResourceWholeDeltaPrologue[16] = {
    0xfd, 0x7b, 0xbd, 0xa9, 0xf6, 0x57, 0x01, 0xa9, 0xf4, 0x4f, 0x02, 0xa9, 0xfd, 0x03, 0x00, 0x91,
};
inline constexpr std::uint8_t kResourceFixedDeltaPrologue[16] = {
    0xfd, 0x7b, 0xbd, 0xa9, 0xf5, 0x0b, 0x00, 0xf9, 0xf4, 0x4f, 0x02, 0xa9, 0xfd, 0x03, 0x00, 0x91,
};
inline constexpr std::uint8_t kPeriodicResourceCallFingerprint[16] = {
    0x5f, 0x83, 0x00, 0x94, 0xe0, 0x03, 0x13, 0xaa, 0x6f, 0x49, 0x00, 0x94, 0xc9, 0x57, 0xf7, 0x97,
};
inline constexpr std::uint8_t kDeathOwnerResourceCallFingerprint[16] = {
    0xce, 0x60, 0xff, 0x97, 0xe0, 0x03, 0x16, 0xaa, 0x39, 0x35, 0xf6, 0x97, 0xb8, 0x83, 0x1e, 0xf8,
};
inline constexpr std::uint8_t kDeathOpponentResourceCallFingerprint[16] = {
    0xe9, 0x5e, 0xff, 0x97, 0xe0, 0x03, 0x16, 0xaa, 0xf2, 0x34, 0xf6, 0x97, 0xf9, 0x03, 0x00, 0xaa,
};

enum class ResourceDeltaCause : std::uint8_t {
  None = 0,
  Periodic = 1,
  DeathOwner = 2,
  DeathOpponent = 3,
};

constexpr ResourceDeltaCause classify_resource_delta(std::uintptr_t hook_offset, std::uintptr_t caller_return_offset) {
  if (hook_offset == kResourceWholeDeltaOffset && caller_return_offset == kPeriodicResourceReturnOffset) {
    return ResourceDeltaCause::Periodic;
  }
  if (hook_offset == kResourceWholeDeltaOffset && caller_return_offset == kDeathOwnerResourceReturnOffset) {
    return ResourceDeltaCause::DeathOwner;
  }
  if (hook_offset == kResourceFixedDeltaOffset && caller_return_offset == kDeathOpponentResourceReturnOffset) {
    return ResourceDeltaCause::DeathOpponent;
  }
  return ResourceDeltaCause::None;
}

constexpr std::int64_t requested_resource_delta_fixed(ResourceDeltaCause cause, std::int32_t amount_argument) {
  if (amount_argument < 0 || cause == ResourceDeltaCause::None) {
    return -1;
  }
  if (cause == ResourceDeltaCause::DeathOpponent) {
    return amount_argument;
  }
  return static_cast<std::int64_t>(amount_argument) * kResourceFixedPointScale;
}

// LogicAreaEffectObject exact runtime identity and fields.
inline constexpr std::int32_t kAreaEffectObjectKind = 3;
inline constexpr std::uintptr_t kAreaEffectObjectVtableOffset = 0x0189c2e8;
inline constexpr std::uintptr_t kAreaEffectObjectFactoryOffset = 0x00f2384c;
inline constexpr std::uintptr_t kAreaEffectObjectTickOffset = 0x00f143a8;
inline constexpr std::size_t kAreaEffectObjectLevelOffset = 0xfc;
inline constexpr std::size_t kAreaEffectObjectRemainingLifeOffset = 0x100;
inline constexpr std::size_t kAreaEffectObjectParentOffset = 0x130;
inline constexpr std::size_t kAreaEffectObjectFollowTargetOffset = 0x138;
inline constexpr std::size_t kAreaEffectObjectRelatedTargetOffset = 0x140;

// LogicAreaEffectObjectData fields consumed by the runtime object.
inline constexpr std::size_t kAreaLifeDurationOffset = 0x170;
inline constexpr std::size_t kAreaRadiusOffset = 0x17c;
inline constexpr std::size_t kAreaAffectsHiddenOffset = 0x18d;
inline constexpr std::size_t kAreaCanBeTargetedOffset = 0x19b;
inline constexpr std::size_t kAreaStayAfterParentDiesOffset = 0x1b8;
inline constexpr std::size_t kAreaFollowBehaviourOffset = 0x1e8;

inline constexpr std::uintptr_t kCombatSpawnOffset = 0x00f25b8c;
inline constexpr std::uintptr_t kCombatRemoveOffset = 0x00f25d90;
inline constexpr std::uint8_t kCombatSpawnPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf7, 0x0b, 0x00, 0xf9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
inline constexpr std::uint8_t kCombatRemovePrologue[16] = {
    0xfd, 0x7b, 0xbd, 0xa9, 0xf5, 0x0b, 0x00, 0xf9, 0xf4, 0x4f, 0x02, 0xa9, 0xfd, 0x03, 0x00, 0x91,
};
inline constexpr std::uint8_t kAreaEffectObjectTickPrologue[16] = {
    0xff, 0x03, 0x06, 0xd1, 0xfd, 0x7b, 0x12, 0xa9, 0xfc, 0x6f, 0x13, 0xa9, 0xfa, 0x67, 0x14, 0xa9,
};

constexpr bool exact_area_effect_object(std::int32_t object_kind, std::uintptr_t runtime_vtable_offset) {
  return object_kind == kAreaEffectObjectKind && runtime_vtable_offset == kAreaEffectObjectVtableOffset;
}

// SpawnAttach=true reaches the generic character-spawn helper only through
// this exact call site.  A TLS context around the helper lets the existing
// committed-spawn hook attach the validated source to each validated child.
inline constexpr std::uintptr_t kCharacterSpawnHelperOffset = 0x00f19314;
inline constexpr std::uintptr_t kSpawnAttachHelperCallOffset = 0x00f18ff0;
inline constexpr std::uintptr_t kSpawnAttachHelperReturnOffset = 0x00f18ff4;
inline constexpr std::uint8_t kCharacterSpawnHelperPrologue[16] = {
    0xff, 0x43, 0x04, 0xd1, 0xfd, 0x7b, 0x0b, 0xa9, 0xfc, 0x6f, 0x0c, 0xa9, 0xfa, 0x67, 0x0d, 0xa9,
};
inline constexpr std::uint8_t kSpawnAttachHelperCallFingerprint[16] = {
    0xc9, 0x00, 0x00, 0x94, 0xa0, 0x43, 0x00, 0xd1, 0xc2, 0xc5, 0x11, 0x94, 0xbf, 0x2e, 0x00, 0x71,
};

constexpr bool exact_spawn_attach_path(std::uintptr_t helper_offset, std::uintptr_t caller_return_offset) {
  return helper_offset == kCharacterSpawnHelperOffset && caller_return_offset == kSpawnAttachHelperReturnOffset;
}

// Contextual native predicate: bool(subject, source, option).  Character
// vtable+0x148 points to f1cb90.  f1cb90 has a +0x180 early-false guard and
// falls through to the larger body at f1cba0; hooking only the body loses
// authoritative negative results.
inline constexpr std::uintptr_t kCharacterVtableOffset = 0x0189c4e8;
inline constexpr std::size_t kAreaDamageEligibleVtableSlot = 0x148;
inline constexpr std::uintptr_t kAreaDamageEligibleOffset = 0x00f1cb90;
inline constexpr std::uintptr_t kAreaDamageEligibleBodyOffset = 0x00f1cba0;
inline constexpr std::uint8_t kAreaDamageEligiblePrologue[16] = {
    0x08, 0xc0, 0x40, 0xf9, 0x68, 0x00, 0x00, 0xb4, 0xe0, 0x03, 0x00, 0x12, 0xc0, 0x03, 0x5f, 0xd6,
};

// King-tower start action:
// ActionWaitToActivate runtime checks its condition, then calls f23110.
inline constexpr std::uintptr_t kTowerTargetSetOffset = 0x00f5c894;
inline constexpr std::uintptr_t kActionDispatchOffset = 0x00f23110;
inline constexpr std::uintptr_t kActionSpawnToLocationExecuteOffset = 0x00e7b864;
inline constexpr std::uintptr_t kActionSpawnToLocationStaticVtableOffset = 0x01894588;
inline constexpr std::size_t kActionSpawnToLocationDataOffset = 0xf0;
// ActionSpawnToLocation mode 1 creates its child through the concrete object
// manager entry at f2ffb8.  The f24188 caller reaches this function only after
// the exact request has produced the child; its LR and caller-frame request
// slot therefore form a synchronous, pointer-exact causal bridge.  Mode 3
// uses a different path and is intentionally excluded by the caller guard.
inline constexpr std::uintptr_t kActionSpawnMode1ChildCreateOffset = 0x00f2ffb8;
inline constexpr std::uintptr_t kActionSpawnMode1ChildCreateReturnOffset = 0x00f24bb0;
inline constexpr std::size_t kActionSpawnMode1RequestSlotOffset = 0x68;
inline constexpr std::size_t kActionSpawnMode1RequestDataOffset = 0x00;
inline constexpr std::size_t kActionSpawnMode1RequestSourceOffset = 0x08;
inline constexpr std::size_t kActionSpawnMode1RequestBaseFromSlot = 0x48;
inline constexpr std::uintptr_t kPendingObjectAddOffset = 0x00f25928;
inline constexpr std::uintptr_t kActionWaitUpdateOffset = 0x00f587c4;
inline constexpr std::uintptr_t kActionWaitDispatchCallOffset = 0x00f5881c;
inline constexpr std::uintptr_t kActionWaitDispatchReturnOffset = 0x00f58820;
inline constexpr std::uintptr_t kActionWaitStaticVtableOffset = 0x01894fb0;
inline constexpr std::uintptr_t kActionWaitRuntimeVtableOffset = 0x018a1a40;
inline constexpr std::size_t kActionWaitConditionOffset = 0xf0;
inline constexpr std::size_t kActionWaitOnActivateActionOffset = 0xf8;
inline constexpr std::size_t kActionRuntimeStaticDataOffset = 0x18;
inline constexpr std::size_t kActionRuntimeOwnerOffset = 0x20;

inline constexpr std::uint8_t kActionDispatchPrologue[16] = {
    0x42, 0x03, 0x00, 0xb4, 0xfd, 0x7b, 0xbd, 0xa9, 0xf6, 0x57, 0x01, 0xa9, 0xf4, 0x4f, 0x02, 0xa9,
};
inline constexpr std::uint8_t kActionSpawnToLocationExecutePrologue[16] = {
    0xff, 0x83, 0x03, 0xd1, 0xfd, 0x7b, 0x08, 0xa9, 0xfc, 0x6f, 0x09, 0xa9, 0xfa, 0x67, 0x0a, 0xa9,
};
inline constexpr std::uint8_t kActionSpawnMode1ChildCreatePrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf7, 0x0b, 0x00, 0xf9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
inline constexpr std::uint8_t kPendingObjectAddPrologue[16] = {
    0x08, 0x00, 0x40, 0xf9, 0x02, 0xe0, 0x41, 0x39, 0xe3, 0x03, 0x1f, 0x2a, 0x04, 0x21, 0x40, 0xf9,
};
inline constexpr std::uint8_t kActionWaitUpdatePrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf7, 0x0b, 0x00, 0xf9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
inline constexpr std::uint8_t kActionWaitDispatchCallFingerprint[16] = {
    0x3d, 0x2a, 0xff, 0x97, 0xe0, 0x03, 0x13, 0xaa, 0xf4, 0x4f, 0x43, 0xa9, 0xf7, 0x0b, 0x40, 0xf9,
};

// ActionChangeGameObjectData exact static object and execution entry.
inline constexpr std::uintptr_t kChangeGameObjectDataVtableOffset = 0x0188cc98;
inline constexpr std::uintptr_t kChangeGameObjectDataExecuteOffset = 0x00e58638;
inline constexpr std::uintptr_t kChangeMutationCallOffset = 0x00e58678;
inline constexpr std::size_t kChangeNewCharacterDataOffset = 0xf0;
inline constexpr std::size_t kChangeNewProjectileDataOffset = 0xf8;
inline constexpr std::size_t kChangeResetTargetOffset = 0x100;
inline constexpr std::uint8_t kChangeGameObjectDataExecutePrologue[16] = {
    0xfd, 0x7b, 0xbd, 0xa9, 0xf5, 0x0b, 0x00, 0xf9, 0xf4, 0x4f, 0x02, 0xa9, 0xfd, 0x03, 0x00, 0x91,
};
inline constexpr std::uint8_t kChangeMutationCallFingerprint[16] = {
    0x00, 0x01, 0x3f, 0xd6, 0xd3, 0x02, 0x00, 0xb4, 0xe0, 0x03, 0x15, 0xaa, 0x3b, 0x55, 0x03, 0x94,
};

constexpr bool exact_transform_transition(std::uintptr_t old_data, std::uintptr_t new_data,
                                          std::uintptr_t expected_character_data,
                                          std::uintptr_t expected_projectile_data) {
  if (old_data == 0 || new_data == 0 || old_data == new_data) {
    return false;
  }
  const bool character_match = expected_character_data != 0 && new_data == expected_character_data;
  const bool projectile_match = expected_projectile_data != 0 && new_data == expected_projectile_data;
  return character_match != projectile_match;
}

static_assert(classify_resource_delta(kResourceWholeDeltaOffset, kPeriodicResourceReturnOffset) ==
              ResourceDeltaCause::Periodic);
static_assert(classify_resource_delta(kResourceWholeDeltaOffset, kDeathOwnerResourceReturnOffset) ==
              ResourceDeltaCause::DeathOwner);
static_assert(classify_resource_delta(kResourceFixedDeltaOffset, kDeathOpponentResourceReturnOffset) ==
              ResourceDeltaCause::DeathOpponent);
static_assert(requested_resource_delta_fixed(ResourceDeltaCause::Periodic, 1) == 10000);
static_assert(exact_area_effect_object(3, 0x0189c2e8));
static_assert(exact_spawn_attach_path(kCharacterSpawnHelperOffset, kSpawnAttachHelperReturnOffset));

} // namespace cr_remaining_runtime
