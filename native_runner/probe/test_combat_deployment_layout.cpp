#include "combat_deployment_layout.h"
#include <cassert>
#include <cstdio>

static void test_queued_mirror_proof() {
  using namespace cr_combat_deployment;
  constexpr DescriptorIdentity queued{0x1000, 0x2000, 0x20802c00U};
  constexpr DescriptorIdentity later_live{0x1000, 0x3000, 0x20802c00U};
  constexpr DescriptorIdentity consumed{0x2000, 0, 0x20802c00U};
  static_assert(!exact_mirror_descriptor_match(kMirrorCardGlobalId, consumed, later_live));
  QueuedMirrorProof proof{false, 17, 23, 1, 1, 200, queued};
  QueuedMirrorProofStore store;
  assert(store.remember(proof));
  // Failed identity checks must not consume or weaken the original proof.
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 18, 23, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 24, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 0, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 2, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1001, consumed, 17, 23, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 1, 199));
  assert(!store.consume(26000030U, 0x1000, consumed, 17, 23, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, {0x3000, 0, consumed.packed}, 17, 23, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, {0x2000, 0x1000, consumed.packed}, 17, 23, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, {0x2000, 0, consumed.packed ^ 0x10000000U}, 17, 23, 1, 1, 220));
  assert(store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 1, 220));
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 1, 220));

  // Independent resident matches and epoch reset retain no cross-match proof.
  QueuedMirrorProofStore other_slot;
  assert(store.remember(proof));
  assert(!other_slot.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 1, 220));
  store.clear();
  assert(!store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 1, 220));
  for (std::size_t i = 0; i < QueuedMirrorProofStore::capacity; ++i) assert(store.remember(proof));
  assert(!store.remember(proof));
  assert(store.consume(kMirrorCardGlobalId, 0x1000, consumed, 17, 23, 1, 1, 220));
  assert(store.remember(proof));
  std::puts("queued Mirror identity, delayed consumption, isolation and capacity checks passed");
}

int main() {
  test_queued_mirror_proof();
    using cr_combat_deployment::DescriptorIdentity;
    using cr_combat_deployment::exact_hero_descriptor_match;
    using cr_combat_deployment::exact_mirror_descriptor_match;

    constexpr DescriptorIdentity hero{0x1000, 0x2000, 0x51502802U};
    static_assert(exact_hero_descriptor_match(hero, hero));
    static_assert(!exact_hero_descriptor_match(
        hero, DescriptorIdentity{0x1001, 0x2000, 0x51502802U}));
    static_assert(!exact_hero_descriptor_match(
        hero, DescriptorIdentity{0x1000, 0x2001, 0x51502802U}));
    static_assert(!exact_hero_descriptor_match(
        hero, DescriptorIdentity{0x1000, 0x2000, 0x51502812U}));
    static_assert(exact_hero_descriptor_match(
        DescriptorIdentity{0x1000, 0, 0x51502802U},
        DescriptorIdentity{0x1000, 0, 0x51502802U}));

    // The Hero-specific rule must never admit BasicForm or EvoForm.
    static_assert(!exact_hero_descriptor_match(
        DescriptorIdentity{0x1000, 0x2000, 0x51502800U},
        DescriptorIdentity{0x1000, 0x2000, 0x51502800U}));
    static_assert(!exact_hero_descriptor_match(
        DescriptorIdentity{0x1000, 0x2000, 0x51502801U},
        DescriptorIdentity{0x1000, 0x2000, 0x51502801U}));

    constexpr DescriptorIdentity mirror_consumed{
        0x2000, 0, 0x51401800U};
    constexpr DescriptorIdentity mirror_rebuilt{
        0x1000, 0x2000, 0x51401800U};
    static_assert(exact_mirror_descriptor_match(
        28000006U, mirror_consumed, mirror_rebuilt));
    static_assert(!exact_mirror_descriptor_match(
        28000005U, mirror_consumed, mirror_rebuilt));
    static_assert(!exact_mirror_descriptor_match(
        28000006U,
        DescriptorIdentity{0x2000, 0x3000, 0x51401800U},
        mirror_rebuilt));
    static_assert(!exact_mirror_descriptor_match(
        28000006U,
        DescriptorIdentity{0x2000, 0, 0x51401801U},
        DescriptorIdentity{0x1000, 0x2000, 0x51401801U}));
    static_assert(!exact_mirror_descriptor_match(
        28000006U,
        mirror_consumed,
        DescriptorIdentity{0x1000, 0x3000, 0x51401800U}));

    assert(exact_hero_descriptor_match(hero, hero));
    assert(exact_mirror_descriptor_match(
        28000006U, mirror_consumed, mirror_rebuilt));
    return 0;
}
