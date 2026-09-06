#pragma once

// Host-testable validation for the exact-build projectile-impact hook.
//
// The native impact boundary may append more than one processed target during
// one call.  The current candidate is proved by its exact native object ID in
// the newly appended range; accepting an older entry or a duplicate remains
// fail-closed.

#include <cstdint>

namespace cr_combat_impact {

enum class ProcessedTargetDecision : std::uint8_t {
  Publish = 0,
  RejectBounds = 1,
  RejectMissing = 2,
  RejectDuplicate = 3,
};

constexpr ProcessedTargetDecision classify_processed_target_append(const std::int32_t *processed,
                                                                   std::int32_t pre_count, std::int32_t post_count,
                                                                   std::int32_t post_capacity,
                                                                   std::uint32_t target_native_object_id) {
  if (processed == nullptr || pre_count < 0 || post_count <= pre_count || post_capacity < post_count ||
      post_capacity > 100000 || target_native_object_id == 0) {
    return ProcessedTargetDecision::RejectBounds;
  }
  std::int32_t matches = 0;
  for (std::int32_t index = pre_count; index < post_count; ++index) {
    if (processed[index] == static_cast<std::int32_t>(target_native_object_id)) {
      ++matches;
    }
  }
  if (matches == 0) {
    return ProcessedTargetDecision::RejectMissing;
  }
  if (matches != 1) {
    return ProcessedTargetDecision::RejectDuplicate;
  }
  return ProcessedTargetDecision::Publish;
}

} // namespace cr_combat_impact
