// Exact-build visibility-transition telemetry for Null's Royale 15.535.13.
//
// The shared probe already owns f5a2d4 through phase_runtime_telemetry.inc;
// adding a second inline hook would be unsafe.  The owner of that hook calls
// visibility_prepare_buff_transition before the original and
// visibility_commit_buff_transition after it.  This producer is the sole
// owner of the independent f5a578 remove hook.  Ordinary acquisition remains
// unavailable: its count-gate instructions contain PC-relative control flow
// and must not be installed through the generic trampoline.
//
// Inclusion contract:
//   * include after combat_event_telemetry.inc and before
//     phase_runtime_telemetry.inc;
//   * g_libg_base is the attested libg base;
//   * read_object_field<T>, process_range_is_readable and
//     combat_component_type_is, CombatCaptureContext/CombatEntityFact,
//     append_json, append_combat_entity_fact, install_inline_hook and logging
//     are available;
//   * call install_visibility_runtime_hooks only after the phase hook owner
//     has published the exact f5a2d4 install result.

#include <atomic>
#include "visibility_runtime_layout.h"

enum class VisibilityMutationKind : std::uint8_t {
  Apply = 1,
  Remove = 2,
};

using VisibilityPhaseEdge = cr_visibility_runtime::VisibilityTransition;

struct VisibilityMutationContext {
  bool validated = false;
  VisibilityMutationKind kind = VisibilityMutationKind::Apply;
  std::uintptr_t hook_offset = 0;
  void *component = nullptr;
  void *subject = nullptr;
  void *entry = nullptr;
  void *buff_asset = nullptr;
  std::int32_t active_entry_count = -1;
  std::int32_t invisible_count_before = -1;
  std::uint32_t buff_global_id = 0;
  bool buff_invisible = false;
};

struct VisibilityTransitionFact {
  bool validated = false;
  VisibilityMutationKind kind = VisibilityMutationKind::Apply;
  VisibilityPhaseEdge edge = VisibilityPhaseEdge::None;
  std::uintptr_t hook_offset = 0;
  void *subject = nullptr;
  void *component = nullptr;
  void *entry = nullptr;
  void *buff_asset = nullptr;
  std::int32_t active_entry_count = -1;
  std::int32_t invisible_count_before = -1;
  std::int32_t invisible_count_after = -1;
  std::uint32_t buff_global_id = 0;
  bool buff_invisible = false;
};

bool visibility_runtime_vtable_is(const void *object, std::uintptr_t expected_offset) {
  if (object == nullptr || !process_range_is_readable(object, sizeof(void *))) {
    return false;
  }
  const void *const vtable = read_object_field<const void *>(object, 0);
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(vtable);
  return vtable != nullptr && raw >= g_libg_base && raw - g_libg_base == expected_offset;
}

bool visibility_prepare_buff_transition(std::uintptr_t hook_offset, void *component, void *entry,
                                        VisibilityMutationContext *result) {
  if (result == nullptr) {
    return false;
  }
  *result = VisibilityMutationContext{};
  VisibilityMutationKind kind;
  if (hook_offset == cr_visibility_runtime::kBuffApplyOffset) {
    kind = VisibilityMutationKind::Apply;
  } else if (hook_offset == cr_visibility_runtime::kBuffRemoveOffset) {
    kind = VisibilityMutationKind::Remove;
  } else {
    return false;
  }
  if (component == nullptr || entry == nullptr || !process_range_is_readable(component, 0x38) ||
      !process_range_is_readable(entry, 0x28) ||
      !visibility_runtime_vtable_is(component, cr_visibility_runtime::kBuffComponentVtableOffset)) {
    return false;
  }
  void *const subject = read_object_field<void *>(component, cr_visibility_runtime::kComponentOwnerOffset);
  void *const entry_component = read_object_field<void *>(entry, cr_visibility_runtime::kBuffEntryComponentOffset);
  void *const buff_asset = read_object_field<void *>(entry, cr_visibility_runtime::kBuffEntryAssetOffset);
  if (subject == nullptr || entry_component != component || buff_asset == nullptr ||
      !process_range_is_readable(subject, 0x50) || !process_range_is_readable(buff_asset, 0x254) ||
      !combat_component_type_is(component, subject, cr_visibility_runtime::kBuffComponentType) ||
      !visibility_runtime_vtable_is(buff_asset, cr_visibility_runtime::kBuffAssetVtableOffset)) {
    return false;
  }
  const std::int32_t capacity =
      read_object_field<std::int32_t>(component, cr_visibility_runtime::kBuffEntryCapacityOffset);
  const std::int32_t count = read_object_field<std::int32_t>(component, cr_visibility_runtime::kBuffEntryCountOffset);
  const std::int32_t invisible =
      read_object_field<std::int32_t>(component, cr_visibility_runtime::kBuffInvisibleCountOffset);
  const std::uint8_t invisible_flag =
      read_object_field<std::uint8_t>(buff_asset, cr_visibility_runtime::kBuffAssetInvisibleOffset);
  const std::uint32_t buff_global_id =
      read_object_field<std::uint32_t>(buff_asset, cr_visibility_runtime::kBuffAssetGlobalIdOffset);
  if (capacity < 0 || count < 0 || count > capacity || capacity > 1024 || invisible < 0 || invisible > count + 1 ||
      invisible_flag > 1 || buff_global_id == 0) {
    return false;
  }
  result->validated = true;
  result->kind = kind;
  result->hook_offset = hook_offset;
  result->component = component;
  result->subject = subject;
  result->entry = entry;
  result->buff_asset = buff_asset;
  result->active_entry_count = count;
  result->invisible_count_before = invisible;
  result->buff_global_id = buff_global_id;
  result->buff_invisible = invisible_flag != 0;
  return true;
}

bool visibility_finish_buff_transition(const VisibilityMutationContext &before, VisibilityTransitionFact *result) {
  if (result == nullptr) {
    return false;
  }
  *result = VisibilityTransitionFact{};
  if (!before.validated || before.component == nullptr || !process_range_is_readable(before.component, 0x38) ||
      !visibility_runtime_vtable_is(before.component, cr_visibility_runtime::kBuffComponentVtableOffset)) {
    return false;
  }
  const std::int32_t after =
      read_object_field<std::int32_t>(before.component, cr_visibility_runtime::kBuffInvisibleCountOffset);
  if (after < 0) {
    return false;
  }
  if (before.buff_invisible) {
    if (before.kind == VisibilityMutationKind::Apply) {
      if (after != before.invisible_count_before + 1 || after > before.active_entry_count) {
        return false;
      }
    } else if (before.invisible_count_before < 1 || after != before.invisible_count_before - 1 ||
               after > before.active_entry_count || before.invisible_count_before > before.active_entry_count + 1) {
      return false;
    }
  } else if (after != before.invisible_count_before) {
    return false;
  }

  const auto edge = before.buff_invisible
                        ? cr_visibility_runtime::classify_visibility_transition(before.invisible_count_before, after)
                        : VisibilityPhaseEdge::None;
  result->validated = true;
  result->kind = before.kind;
  result->edge = edge;
  result->hook_offset = before.hook_offset;
  result->subject = before.subject;
  result->component = before.component;
  result->entry = before.entry;
  result->buff_asset = before.buff_asset;
  result->active_entry_count = before.active_entry_count;
  result->invisible_count_before = before.invisible_count_before;
  result->invisible_count_after = after;
  result->buff_global_id = before.buff_global_id;
  result->buff_invisible = before.buff_invisible;
  return true;
}

constexpr std::size_t kVisibilityRuntimeRingCapacity = 4096;

struct VisibilityRuntimeRawRecord {
  std::uint64_t sequence = 0;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
  VisibilityPhaseEdge edge = VisibilityPhaseEdge::None;
  std::uintptr_t hook_offset = 0;
  std::uintptr_t caller_offset = 0;
  CombatEntityFact subject;
  std::uint32_t buff_global_id = 0;
  std::int32_t invisible_count_before = -1;
  std::int32_t invisible_count_after = -1;
  bool complete_context = false;
};

struct VisibilityRuntimeRingSlot {
  std::atomic<std::uint64_t> published_sequence{0};
  VisibilityRuntimeRawRecord record;
};

VisibilityRuntimeRingSlot g_visibility_runtime_ring[kVisibilityRuntimeRingCapacity];
std::atomic<std::uint64_t> g_visibility_runtime_next_sequence{1};
std::atomic<std::uint64_t> g_visibility_runtime_epoch_first_sequence{1};
std::atomic<std::uint64_t> g_visibility_runtime_generation{0};
std::atomic<std::uint64_t> g_visibility_runtime_state_epoch{0};
std::atomic<std::uint64_t> g_visibility_runtime_overflow_count{0};
std::atomic<std::uint64_t> g_visibility_runtime_rejected_count{0};
std::atomic<bool> g_visibility_runtime_hooks_attested{false};
std::atomic<bool> g_visibility_runtime_shared_add_hook_bound{false};
std::atomic<bool> g_visibility_runtime_remove_hook_installed{false};
std::atomic<bool> g_visibility_runtime_transition_hooks_installed{false};
std::atomic_flag g_visibility_runtime_epoch_lock = ATOMIC_FLAG_INIT;

using VisibilityBuffRemove = void (*)(void *, void *);
VisibilityBuffRemove g_original_visibility_buff_remove = nullptr;

const char *visibility_phase_edge_label(VisibilityPhaseEdge edge) {
  switch (edge) {
  case VisibilityPhaseEdge::BecameInvisible:
    return "became_invisible";
  case VisibilityPhaseEdge::BecameVisible:
    return "became_visible";
  case VisibilityPhaseEdge::None:
    break;
  }
  return "unknown";
}

std::uintptr_t visibility_caller_offset(const void *address) {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
  return address == nullptr || raw < g_libg_base ? 0 : raw - g_libg_base;
}

std::uintptr_t visibility_expected_caller_offset(VisibilityMutationKind kind) {
  return kind == VisibilityMutationKind::Apply ? cr_visibility_runtime::kBuffApplyOnlyCallOffset + 4
                                               : cr_visibility_runtime::kBuffRemoveOnlyCallOffset + 4;
}

void visibility_refresh_epoch(const CombatCaptureContext &context) {
  if (g_visibility_runtime_generation.load(std::memory_order_acquire) == context.generation &&
      g_visibility_runtime_state_epoch.load(std::memory_order_acquire) == context.state_epoch) {
    return;
  }
  if (g_visibility_runtime_epoch_lock.test_and_set(std::memory_order_acquire)) {
    g_visibility_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (g_visibility_runtime_generation.load(std::memory_order_relaxed) != context.generation ||
      g_visibility_runtime_state_epoch.load(std::memory_order_relaxed) != context.state_epoch) {
    const std::uint64_t first = g_visibility_runtime_next_sequence.load(std::memory_order_acquire);
    g_visibility_runtime_generation.store(context.generation, std::memory_order_release);
    g_visibility_runtime_state_epoch.store(context.state_epoch, std::memory_order_release);
    g_visibility_runtime_epoch_first_sequence.store(first, std::memory_order_release);
    g_visibility_runtime_overflow_count.store(0, std::memory_order_release);
    g_visibility_runtime_rejected_count.store(0, std::memory_order_release);
  }
  g_visibility_runtime_epoch_lock.clear(std::memory_order_release);
}

void visibility_reject_capture(const CombatCaptureContext &context) {
  visibility_refresh_epoch(context);
  g_visibility_runtime_rejected_count.fetch_add(1, std::memory_order_relaxed);
}

bool visibility_subject_is_committed(const void *subject, void *object_manager, bool *object_vector_valid) {
  if (object_vector_valid == nullptr) {
    return false;
  }
  *object_vector_valid = false;
  void **objects = nullptr;
  std::int32_t count = 0;
  if (!read_combat_object_vector(object_manager, &objects, &count)) {
    return false;
  }
  *object_vector_valid = true;
  for (std::int32_t index = 0; index < count; ++index) {
    if (objects[index] == subject) {
      return true;
    }
  }
  return false;
}

std::uint64_t publish_visibility_runtime_record(VisibilityRuntimeRawRecord record) {
  visibility_refresh_epoch(CombatCaptureContext{
      nullptr,
      nullptr,
      record.generation,
      record.state_epoch,
      record.tick,
  });
  const std::uint64_t sequence = g_visibility_runtime_next_sequence.fetch_add(1, std::memory_order_acq_rel);
  record.sequence = sequence;
  const std::uint64_t first = g_visibility_runtime_epoch_first_sequence.load(std::memory_order_acquire);
  if (sequence >= first + kVisibilityRuntimeRingCapacity) {
    g_visibility_runtime_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
  VisibilityRuntimeRingSlot &slot = g_visibility_runtime_ring[(sequence - 1) % kVisibilityRuntimeRingCapacity];
  slot.published_sequence.store(0, std::memory_order_relaxed);
  slot.record = record;
  slot.published_sequence.store(sequence, std::memory_order_release);
  return sequence;
}

void visibility_commit_buff_transition(bool prepared, const VisibilityMutationContext &before,
                                       std::uintptr_t caller_offset) {
  if (!g_visibility_runtime_transition_hooks_installed.load(std::memory_order_acquire)) {
    return;
  }
  CombatCaptureContext context;
  if (!current_combat_capture_context(&context)) {
    return;
  }
  if (!prepared || caller_offset != visibility_expected_caller_offset(before.kind)) {
    visibility_reject_capture(context);
    return;
  }
  VisibilityTransitionFact fact;
  if (!visibility_finish_buff_transition(before, &fact) || !fact.validated) {
    visibility_reject_capture(context);
    return;
  }
  if (fact.edge == VisibilityPhaseEdge::None) {
    return;
  }
  VisibilityRuntimeRawRecord record;
  record.generation = context.generation;
  record.state_epoch = context.state_epoch;
  record.tick = context.tick;
  record.edge = fact.edge;
  record.hook_offset = fact.hook_offset;
  record.caller_offset = caller_offset;
  record.buff_global_id = fact.buff_global_id;
  record.invisible_count_before = fact.invisible_count_before;
  record.invisible_count_after = fact.invisible_count_after;
  capture_combat_entity(fact.subject, context.object_manager, &record.subject);
  if (!record.subject.validated || !record.subject.present || !record.subject.visibility_validated ||
      record.subject.invisible_count != fact.invisible_count_after) {
    bool object_vector_valid = false;
    const bool committed = visibility_subject_is_committed(fact.subject, context.object_manager, &object_vector_valid);
    // Auto-invisibility can be applied before a freshly constructed
    // character is committed to ObjectManager, or removed after it has
    // left the manager.  Those transitions have no stable public entity
    // identity yet/anymore and are outside this committed-entity stream.
    // A readable vector that proves absence is therefore not a rejected
    // capture.  Once the exact subject is committed, every identity or
    // visibility mismatch remains fail-closed below.
    if (object_vector_valid && !committed) {
      return;
    }
    visibility_reject_capture(context);
    return;
  }
  record.complete_context = true;
  publish_visibility_runtime_record(record);
}

void visibility_buff_removed_hook(void *component, void *entry) {
  const std::uintptr_t caller_offset = visibility_caller_offset(__builtin_return_address(0));
  VisibilityMutationContext before;
  const bool prepared =
      visibility_prepare_buff_transition(cr_visibility_runtime::kBuffRemoveOffset, component, entry, &before);
  g_original_visibility_buff_remove(component, entry);
  visibility_commit_buff_transition(prepared, before, caller_offset);
}

bool append_visibility_runtime_events_json(std::uint64_t generation, std::uint64_t state_epoch,
                                           std::int32_t observation_tick, char *response, std::size_t response_size,
                                           std::size_t *used) {
  if (response == nullptr || used == nullptr) {
    return false;
  }
  CombatCaptureContext observation_context;
  observation_context.generation = generation;
  observation_context.state_epoch = state_epoch;
  observation_context.tick = observation_tick;
  visibility_refresh_epoch(observation_context);
  const bool transition_installed = g_visibility_runtime_transition_hooks_installed.load(std::memory_order_acquire);
  const bool identity = g_visibility_runtime_generation.load(std::memory_order_acquire) == generation &&
                        g_visibility_runtime_state_epoch.load(std::memory_order_acquire) == state_epoch;
  const std::uint64_t next = g_visibility_runtime_next_sequence.load(std::memory_order_acquire);
  const std::uint64_t epoch_first =
      identity ? g_visibility_runtime_epoch_first_sequence.load(std::memory_order_acquire) : next;
  const std::uint64_t retained_oldest =
      next > epoch_first + kVisibilityRuntimeRingCapacity ? next - kVisibilityRuntimeRingCapacity : epoch_first;
  const std::uint64_t transmit_from = retained_oldest;
  const std::uint64_t overflow = identity ? g_visibility_runtime_overflow_count.load(std::memory_order_acquire) : 0;
  const std::uint64_t rejected = identity ? g_visibility_runtime_rejected_count.load(std::memory_order_acquire) : 0;
  const bool complete = transition_installed && identity && rejected == 0;
  if (!append_json(response, response_size, used,
                   "{\"ok\":true,\"schema\":\"%s\","
                   "\"generation\":%llu,\"stateEpoch\":%llu,"
                   "\"observationTick\":%d,\"capacity\":%zu,"
                   "\"hookSetAttested\":%s,"
                   "\"transitionHookSetInstalled\":%s,"
                   "\"contextualGateHookSetInstalled\":false,"
                   "\"capability\":{\"status\":\"%s\","
                   "\"source\":\"exact-build-native-invisibility-transition-ring\","
                   "\"validation\":"
                   "\"sha-build-id-call-and-prologue-fingerprint-abi-count-edge\","
                   "\"confidence\":\"high\",\"failClosed\":true},"
                   "\"contextualGateCapability\":{\"status\":\"unavailable\","
                   "\"source\":\"no-safe-pc-relative-gate-bridge\","
                   "\"validation\":\"fail-closed-no-generic-trampoline\","
                   "\"confidence\":\"none\",\"failClosed\":true},"
                   "\"ownerRelativeVisibility\":{"
                   "\"publicByOwner\":{\"status\":\"unavailable\","
                   "\"reason\":"
                   "\"no-exact-owner-conditioned-native-producer\"},"
                   "\"targetableByOwner\":{\"status\":\"unavailable\","
                   "\"reason\":"
                   "\"no-exact-owner-conditioned-native-producer\"}},"
                   "\"epochFirstSequence\":%llu,"
                   "\"oldestRetainedSequence\":%llu,"
                   "\"transmitFromSequence\":%llu,"
                   "\"nextSequence\":%llu,\"overflowCount\":%llu,"
                   "\"rejectedCount\":%llu,"
                   "\"sequenceGapBeforeOldest\":%s,"
                   "\"complete\":%s,\"events\":[",
                   cr_visibility_runtime::kSchema, static_cast<unsigned long long>(generation),
                   static_cast<unsigned long long>(state_epoch), observation_tick, kVisibilityRuntimeRingCapacity,
                   json_boolean(g_visibility_runtime_hooks_attested.load(std::memory_order_acquire)),
                   json_boolean(transition_installed), transition_installed && identity ? "derived" : "unavailable",
                   static_cast<unsigned long long>(epoch_first), static_cast<unsigned long long>(retained_oldest),
                   static_cast<unsigned long long>(transmit_from), static_cast<unsigned long long>(next),
                   static_cast<unsigned long long>(overflow), static_cast<unsigned long long>(rejected),
                   json_boolean(retained_oldest > epoch_first), json_boolean(complete))) {
    return false;
  }
  bool first = true;
  if (identity) {
    for (std::uint64_t sequence = transmit_from; sequence < next; ++sequence) {
      VisibilityRuntimeRingSlot &slot = g_visibility_runtime_ring[(sequence - 1) % kVisibilityRuntimeRingCapacity];
      if (slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        return false;
      }
      const VisibilityRuntimeRawRecord record = slot.record;
      if (record.sequence != sequence || record.generation != generation || record.state_epoch != state_epoch ||
          record.tick > observation_tick || slot.published_sequence.load(std::memory_order_acquire) != sequence) {
        return false;
      }
      if (!append_json(response, response_size, used,
                       "%s{\"sequence\":%llu,\"tick\":%d,"
                       "\"kind\":\"%s\",\"hookOffset\":%llu,"
                       "\"callerOffset\":%llu,\"subject\":",
                       first ? "" : ",", static_cast<unsigned long long>(record.sequence), record.tick,
                       visibility_phase_edge_label(record.edge), static_cast<unsigned long long>(record.hook_offset),
                       static_cast<unsigned long long>(record.caller_offset)) ||
          !append_combat_entity_fact(response, response_size, used, record.subject) ||
          !append_json(response, response_size, used,
                       ",\"buffGlobalId\":%u,"
                       "\"invisibleCountBefore\":%d,"
                       "\"invisibleCountAfter\":%d,"
                       "\"scope\":\"native_invisibility_phase\","
                       "\"completeContext\":%s}",
                       record.buff_global_id, record.invisible_count_before, record.invisible_count_after,
                       json_boolean(record.complete_context))) {
        return false;
      }
      first = false;
    }
  }
  return append_json(response, response_size, used, "]}");
}

bool verify_visibility_runtime_transition_fingerprints(std::uintptr_t libg_base) {
  char sha256[65] = {};
  char build_id[41] = {};
  if (!sha256_file(g_libg_path, sha256) || !libg_build_id(build_id) ||
      std::strcmp(sha256, cr_visibility_runtime::kExpectedLibgSha256) != 0 ||
      std::strcmp(build_id, cr_visibility_runtime::kExpectedLibgBuildId) != 0) {
    LOGE("visibility-runtime attestation mismatch sha=%s "
         "buildId=%s",
         sha256, build_id);
    return false;
  }
  struct Fingerprint {
    std::uintptr_t offset;
    const std::uint8_t *bytes;
    const char *label;
  };
  // f5a2d4 is already patched by the phase owner when this verifier runs.
  // Its dedicated install result proves that install_inline_hook observed
  // the original prologue; re-reading the now-modified bytes would produce
  // a false attestation failure.  Its sole caller remains immutable and is
  // checked here together with the visibility-owned remove path.
  const Fingerprint values[] = {
      {cr_visibility_runtime::kBuffApplyOnlyCallOffset, cr_visibility_runtime::kBuffApplyCallFingerprint,
       "Buff apply caller"},
      {cr_visibility_runtime::kBuffRemoveOffset, cr_visibility_runtime::kBuffRemovePrologue, "Buff remove"},
      {cr_visibility_runtime::kBuffRemoveOnlyCallOffset, cr_visibility_runtime::kBuffRemoveCallFingerprint,
       "Buff remove caller"},
  };
  for (const Fingerprint &value : values) {
    if (std::memcmp(reinterpret_cast<const void *>(libg_base + value.offset), value.bytes, 16) != 0) {
      LOGE("visibility-runtime fingerprint mismatch "
           "label=%s offset=+0x%zx",
           value.label, static_cast<std::size_t>(value.offset));
      return false;
    }
  }
  return true;
}

void visibility_refresh_installed_state() {
  const bool installed = g_visibility_runtime_hooks_attested.load(std::memory_order_acquire) &&
                         g_visibility_runtime_shared_add_hook_bound.load(std::memory_order_acquire) &&
                         g_visibility_runtime_remove_hook_installed.load(std::memory_order_acquire);
  g_visibility_runtime_transition_hooks_installed.store(installed, std::memory_order_release);
}

bool install_visibility_runtime_hooks(std::uintptr_t libg_base, bool shared_add_hook_installed) {
  const bool attested = shared_add_hook_installed && verify_visibility_runtime_transition_fingerprints(libg_base);
  g_visibility_runtime_hooks_attested.store(attested, std::memory_order_release);
  g_visibility_runtime_shared_add_hook_bound.store(attested && shared_add_hook_installed, std::memory_order_release);
  if (!attested || !shared_add_hook_installed) {
    g_visibility_runtime_remove_hook_installed.store(false, std::memory_order_release);
    visibility_refresh_installed_state();
    return false;
  }
#if !defined(__aarch64__)
  (void)libg_base;
  g_visibility_runtime_remove_hook_installed.store(false, std::memory_order_release);
  visibility_refresh_installed_state();
  return false;
#else
  const bool remove_installed = install_inline_hook(
      libg_base, cr_visibility_runtime::kBuffRemoveOffset, cr_visibility_runtime::kBuffRemovePrologue,
      reinterpret_cast<const void *>(&visibility_buff_removed_hook),
      reinterpret_cast<void **>(&g_original_visibility_buff_remove), "visibility Buff remove");
  g_visibility_runtime_remove_hook_installed.store(remove_installed, std::memory_order_release);
  visibility_refresh_installed_state();
  const bool installed = g_visibility_runtime_transition_hooks_installed.load(std::memory_order_acquire);
  LOGI("exact-build visibility transition hooks "
       "installed=%d sharedAdd=%d remove=%d ring=%zu "
       "contextualGate=0",
       installed ? 1 : 0, shared_add_hook_installed ? 1 : 0, remove_installed ? 1 : 0, kVisibilityRuntimeRingCapacity);
  return installed;
#endif
}
