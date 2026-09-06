#pragma once

// Exact-build phase/timing layout for Null's Royale 15.535.13.
//
// This header is deliberately independent from Android and from the live
// probe.  It contains only the attested offsets, ABI fingerprints and pure
// integer helpers shared by the native hook layer and host-side golden tests.

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cr_phase_runtime {

constexpr const char *kSchema = "native-phase-runtime.v1";
constexpr const char *kExpectedLibgSha256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783";
constexpr const char *kExpectedLibgBuildId = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";

// Runtime component identities.
constexpr std::uintptr_t kAttackComponentVtableOffset = 0x018a1dd0;
constexpr std::uintptr_t kMovementComponentVtableOffset = 0x018a1f38;
constexpr std::uintptr_t kBuffComponentVtableOffset = 0x018a1d58;
constexpr std::uintptr_t kAttackComponentTypeGetterOffset = 0x00f5c878;
constexpr std::uintptr_t kMovementComponentTypeGetterOffset = 0x00f6939c;
constexpr std::uintptr_t kBuffComponentTypeGetterOffset = 0x00f59aac;

// Hookable exact-build functions.
constexpr std::uintptr_t kActionDispatchOffset = 0x00f23110;
constexpr std::uintptr_t kAttackExecuteOffset = 0x00f5f100;
constexpr std::uintptr_t kTargetSetOffset = 0x00f5c894;
constexpr std::uintptr_t kBuffAddedOffset = 0x00f5a2d4;
constexpr std::uintptr_t kMovementScaleOffset = 0x00f5b3c8;
constexpr std::uintptr_t kAttackScaleOffset = 0x00f5b4ac;
constexpr std::uintptr_t kEffectiveMovementSpeedOffset = 0x00f1d004;
constexpr std::uintptr_t kMovementDeltaOffset = 0x00f68808;

// Caller return addresses used to disambiguate generic functions.
constexpr std::uintptr_t kAttackStartReturnOffset = 0x00f61a7c;
constexpr std::uintptr_t kClassicChargeActionReturnOffset = 0x00f68958;
constexpr std::uintptr_t kPrimaryAttackReturnOffset = 0x00f61d10;
constexpr std::uintptr_t kAdditionalAttackReturnOffset = 0x00f61e34;
constexpr std::uintptr_t kAttackScaleReturnOffset = 0x00f62030;
constexpr std::uintptr_t kEffectiveSpeedEffectReturnOffset = 0x00f1d218;
constexpr std::uintptr_t kMovementTickScaleReturnOffset = 0x00f66290;
constexpr std::uintptr_t kDeployScaleReturnOffset = 0x00f1abc0;

constexpr std::uint8_t kActionDispatchPrologue[16] = {
    0x42, 0x03, 0x00, 0xb4, 0xfd, 0x7b, 0xbd, 0xa9, 0xf6, 0x57, 0x01, 0xa9, 0xf4, 0x4f, 0x02, 0xa9,
};
constexpr std::uint8_t kAttackExecutePrologue[16] = {
    0xff, 0x83, 0x05, 0xd1, 0xfd, 0x7b, 0x10, 0xa9, 0xfc, 0x6f, 0x11, 0xa9, 0xfa, 0x67, 0x12, 0xa9,
};
constexpr std::uint8_t kTargetSetPrologue[16] = {
    0xfd, 0x7b, 0xbb, 0xa9, 0xf9, 0x0b, 0x00, 0xf9, 0xf8, 0x5f, 0x02, 0xa9, 0xf6, 0x57, 0x03, 0xa9,
};
constexpr std::uint8_t kBuffAddedPrologue[16] = {
    0xff, 0xc3, 0x01, 0xd1, 0xfd, 0x7b, 0x01, 0xa9, 0xfc, 0x6f, 0x02, 0xa9, 0xfa, 0x67, 0x03, 0xa9,
};
constexpr std::uint8_t kMovementScalePrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf8, 0x5f, 0x01, 0xa9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
constexpr std::uint8_t kAttackScalePrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf8, 0x5f, 0x01, 0xa9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
constexpr std::uint8_t kEffectiveMovementSpeedPrologue[16] = {
    0xfd, 0x7b, 0xbd, 0xa9, 0xf5, 0x0b, 0x00, 0xf9, 0xf4, 0x4f, 0x02, 0xa9, 0xfd, 0x03, 0x00, 0x91,
};
constexpr std::uint8_t kMovementDeltaPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf7, 0x0b, 0x00, 0xf9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};

// LogicGameObject / LogicCharacter fields.
constexpr std::size_t kObjectComponentsOffset = 0x18;
constexpr std::size_t kObjectComponentCapacityOffset = 0x20;
constexpr std::size_t kObjectComponentCountOffset = 0x24;
constexpr std::size_t kObjectDataOffset = 0x48;
constexpr std::size_t kObjectPositionXOffset = 0x7c;
constexpr std::size_t kObjectPositionYOffset = 0x80;
constexpr std::size_t kObjectDeployRemainingOffset = 0x15c;
constexpr std::size_t kObjectDeployPreviousOffset = 0x160;

// LogicGameObject's inline runtime-variable hash map.  The bucket array uses
// predecessor links and is intentionally ignored; the authoritative iterable
// order is the bounded head->next chain.
constexpr std::size_t kRuntimeVariableMapOffset = 0xc8;
constexpr std::size_t kRuntimeVariableHeadOffset = 0x10;
constexpr std::size_t kRuntimeVariableCountOffset = 0x18;
constexpr std::size_t kRuntimeVariableNodeNextOffset = 0x00;
constexpr std::size_t kRuntimeVariableNodeHashOffset = 0x08;
constexpr std::size_t kRuntimeVariableNodeKeyOffset = 0x10;
constexpr std::size_t kRuntimeVariableNodeValueOffset = 0x14;
constexpr std::size_t kRuntimeVariableNodeSize = 0x18;
constexpr std::uint64_t kRuntimeVariableMaximumCount = 128;

// Exact typed variables used by the manual four-stage attack sequence.  The
// duration is the authoritative LogicVariableData fallback reached by the
// ResetDecayCounter RHS when the key is absent from the runtime map.
constexpr std::int32_t kAttackSequenceProgressVariableKey = 1846274699;
constexpr std::int32_t kAttackSequenceDecayVariableKey = 531698662;
constexpr std::int32_t kAttackSequenceProgressLimit = 50;
constexpr std::int32_t kAttackSequenceDecayDurationMs = 7000;

constexpr std::int32_t attack_sequence_stage_from_progress(std::int32_t progress) {
  return progress < 0 || progress > kAttackSequenceProgressLimit ? -1
         : progress < 4                                          ? 0
         : progress < 9                                          ? 1
         : progress < 49                                         ? 2
                                                                 : 3;
}

// Common / type-0 attack component fields.
constexpr std::size_t kComponentOwnerOffset = 0x08;
constexpr std::size_t kAttackTargetOffset = 0x10;
constexpr std::size_t kAttackSequenceStageOffset = 0x20;
constexpr std::size_t kAttackTimelineOffset = 0x24;
constexpr std::size_t kAttackLoadRemainingOffset = 0x28;

// Type-1 movement component fields.
constexpr std::size_t kClassicChargeProgressOffset = 0x1e0;
constexpr std::int32_t kClassicChargeUnavailable = -1;

// Type-3 active Buff container and entry fields.
constexpr std::size_t kBuffEntriesOffset = 0x18;
constexpr std::size_t kBuffEntryCapacityOffset = 0x20;
constexpr std::size_t kBuffEntryCountOffset = 0x24;
constexpr std::size_t kBuffEntryRemainingOffset = 0x08;
constexpr std::size_t kBuffEntryAssetOffset = 0x18;
constexpr std::size_t kBuffEntrySourceOffset = 0x38;
constexpr std::size_t kBuffEntrySourceSideOffset = 0x40;

// LogicCharacterData fields.
constexpr std::size_t kCharacterSpeedOffset = 0x410;
constexpr std::size_t kCharacterHitSpeedOffset = 0x418;
constexpr std::size_t kCharacterChargeRangeOffset = 0x438;
constexpr std::size_t kCharacterChargeSpeedMultiplierOffset = 0x43c;
constexpr std::size_t kCharacterLoadTimeOffset = 0x48c;
constexpr std::size_t kCharacterDeployTimeOffset = 0x490;
constexpr std::size_t kCharacterAttackDashTimeOffset = 0x6a0;

// LogicCharacterBuffData fields.
constexpr std::size_t kBuffAssetGlobalIdOffset = 0x40;
constexpr std::size_t kBuffAssetHitSpeedMultiplierOffset = 0xe0;
constexpr std::size_t kBuffAssetSpeedMultiplierOffset = 0xe4;

struct EffectScale {
  std::int32_t positive_percent = 100;
  std::int32_t negative_magnitude = 0;
};

constexpr void accumulate_effect_multiplier(EffectScale &scale, std::int32_t value) {
  if (value >= 1 && value > scale.positive_percent) {
    scale.positive_percent = value;
  } else if (value < 0) {
    const std::int64_t magnitude64 = -static_cast<std::int64_t>(value);
    const auto maximum = std::numeric_limits<std::int32_t>::max();
    const std::int32_t magnitude = magnitude64 > maximum ? maximum : static_cast<std::int32_t>(magnitude64);
    if (magnitude > scale.negative_magnitude) {
      scale.negative_magnitude = magnitude;
    }
  }
}

constexpr bool valid_classic_charge_progress(std::int32_t progress) { return progress >= kClassicChargeUnavailable; }

} // namespace cr_phase_runtime
