#pragma once

// Exact-build native LogicGameObject state-transition boundary for
// Null's Royale 15.535.13.
//
// This is a raw character-state producer.  It does not label a transition as
// an effect or movement stage until a card-scoped live golden joins the exact
// transition to its native movement/action boundary.

#include <cstddef>
#include <cstdint>

namespace cr_character_state_runtime {

constexpr const char *kSchema = "native-character-state-runtime.v1";
constexpr const char *kExpectedLibgSha256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783";
constexpr const char *kExpectedLibgBuildId = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";

constexpr std::uintptr_t kStateSetterOffset = 0x00f18654;
constexpr std::uint8_t kStateSetterPrologue[16] = {
    0xff, 0xc3, 0x03, 0xd1, 0xfd, 0x7b, 0x09, 0xa9, 0xfc, 0x6f, 0x0a, 0xa9, 0xfa, 0x67, 0x0b, 0xa9,
};

constexpr std::size_t kCharacterStateOffset = 0x11c;
constexpr std::size_t kMinimumCharacterBytes = kCharacterStateOffset + sizeof(std::int32_t);
constexpr std::int32_t kMinimumState = 0;
constexpr std::int32_t kMaximumState = 16;

constexpr std::int32_t kBanditCardId = 26000046;
constexpr std::int32_t kMegaKnightCardId = 26000055;
constexpr std::int32_t kEliteArcherCardId = 26000062;
constexpr std::int32_t kEliteArcherHeroFormCardId = 203000062;
constexpr std::int32_t kGoldenKnightCardId = 26000074;
constexpr std::int32_t kSuperHogCardId = 26000081;
constexpr std::int32_t kLittlePrinceCardId = 26000093;
constexpr std::int32_t kBossBanditCardId = 26000103;
// ChampionGuard is the row-115 LogicCharacter spawned by Little Prince's
// ActionSpawnGuard path.  It is intentionally retained as its own moving
// entity rather than being relabeled as the Little Prince.
constexpr std::int32_t kChampionGuardCardId = 26000115;

} // namespace cr_character_state_runtime
