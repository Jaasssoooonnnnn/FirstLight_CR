#pragma once

// Exact-build, host-testable descriptor identity rules for deployment capture.
// Null's Royale 15.535.13 writes NativeCardSelection as base data at +0x00,
// associated/effective data at +0x08, and the packed descriptor at +0x10.

#include <cstdint>
#include <array>

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

// A queued Mirror command retains its original native descriptor even if a
// subsequent card play changes the live Mirror selection before consumption.
struct QueuedMirrorProof {
  bool active = false;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t owner = -1;
  std::int32_t deck_slot = -1;
  std::int32_t registered_tick = -1;
  DescriptorIdentity descriptor{};
};

class QueuedMirrorProofStore {
 public:
  static constexpr std::size_t capacity = 64;
  constexpr void clear() { proofs_ = {}; }
  constexpr bool remember(QueuedMirrorProof proof) {
    for (auto &entry : proofs_) {
      if (!entry.active) {
        proof.active = true;
        entry = proof;
        return true;
      }
    }
    return false;
  }
  constexpr bool consume(std::uint32_t root_card_id, std::uintptr_t root_data,
                         const DescriptorIdentity &consumed, std::uint64_t generation,
                         std::uint64_t state_epoch, std::int32_t owner,
                         std::int32_t deck_slot, std::int32_t tick) {
    for (auto &entry : proofs_) {
      if (entry.active && owner >= 0 && owner < 2 && deck_slot >= 0 && deck_slot < 8 &&
          entry.generation == generation && entry.state_epoch == state_epoch &&
          entry.owner == owner && entry.deck_slot == deck_slot &&
          entry.registered_tick >= 0 && tick >= entry.registered_tick &&
          entry.descriptor.base_data == root_data &&
          exact_mirror_descriptor_match(root_card_id, consumed, entry.descriptor)) {
        entry.active = false;
        return true;
      }
    }
    return false;
  }
 private:
  std::array<QueuedMirrorProof, capacity> proofs_{};
};

} // namespace cr_combat_deployment
