#include "combat_deployment_layout.h"
#include <cassert>
#include <cstdio>

int main() {
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
