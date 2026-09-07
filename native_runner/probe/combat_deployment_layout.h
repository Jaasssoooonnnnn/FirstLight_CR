#pragma once

// Exact-build, host-testable descriptor identity rules for deployment capture.
// Null's Royale 15.535.13 writes NativeCardSelection as base data at +0x00,
// associated/effective data at +0x08, and the packed descriptor at +0x10.

#include <cstdint>

namespace cr_combat_deployment {

constexpr std::int32_t kHeroFormCode = 2;
constexpr std::int32_t kBasicFormCode = 0;
constexpr std::uint32_t kMirrorCardGlobalId = 28000006U;

struct DescriptorIdentity {
  std::uintptr_t base_data = 0;
  std::uintptr_t associated_data = 0;
  std::uint32_t packed = 0;
};

constexpr std::int32_t form_code(std::uint32_t packed) { return static_cast<std::int32_t>(packed & 0x0fU); }

// HeroForm's +0x08 value is produced by the exact row builder and may be null
// (named live Monk deployment, 2026-07-24).  When non-null its concrete
// LogicData subtype is not proven, so it must not be dereferenced as
// LogicSpellData merely to manufacture a global card identity.  Rebuilding
// and comparing the complete descriptor, including an exact null, proves the
// association without weakening ordinary/Evo validation.
constexpr bool exact_hero_descriptor_match(const DescriptorIdentity &consumed, const DescriptorIdentity &rebuilt) {
  return form_code(consumed.packed) == kHeroFormCode && form_code(rebuilt.packed) == kHeroFormCode &&
         consumed.base_data != 0 && consumed.base_data == rebuilt.base_data &&
         consumed.associated_data == rebuilt.associated_data && consumed.packed == rebuilt.packed;
}

// Mirror is the one BasicForm selection whose network command carries the
// effective card as consumed.base_data. Rebuilding the live deck-slot
// selection yields root Mirror in rebuilt.base_data and that same effective
// card in rebuilt.associated_data. The exact packed word joins both views.
constexpr bool exact_mirror_descriptor_match(std::uint32_t root_card_global_id, const DescriptorIdentity &consumed,
                                             const DescriptorIdentity &rebuilt) {
  return root_card_global_id == kMirrorCardGlobalId && form_code(consumed.packed) == kBasicFormCode &&
         form_code(rebuilt.packed) == kBasicFormCode && consumed.base_data != 0 && consumed.associated_data == 0 &&
         rebuilt.base_data != 0 && rebuilt.base_data != consumed.base_data &&
         rebuilt.associated_data == consumed.base_data && consumed.packed == rebuilt.packed;
}

static_assert(form_code(0x51502802U) == kHeroFormCode);

} // namespace cr_combat_deployment
