#pragma once

// Host-testable decision rule for the exact-build damage hook.
//
// Both proved callers initialise appliedAmountOut with the requested amount.
// Native early exits (for example, a hit against an already-zero-HP object
// which has not yet left the live vector) need not overwrite that cell.  A
// positive value in the cell is therefore authoritative only when the
// HP/shield state actually changed by the same amount.

#include <cstdint>

namespace cr_combat_damage {

enum class CaptureDecision : std::uint8_t {
  IgnoreNoStateChange = 0,
  Publish = 1,
  RejectInconsistent = 2,
};

constexpr bool valid_hitpoint_state(std::int32_t current_hitpoints, std::int32_t static_maximum_hitpoints) {
  // The maximum field is the character's static HP cap, not a universal
  // upper bound.  Evolution Witch's exact native heal path can retain
  // current HP above it.  Component identity and ownership are validated by
  // the caller; the raw HP state itself is valid when both values are
  // non-negative.
  return current_hitpoints >= 0 && static_maximum_hitpoints >= 0;
}

constexpr bool is_one_hitpoint_survival_clamp(std::int32_t pre_hitpoints, std::int32_t post_hitpoints,
                                              std::int32_t requested_amount, std::int32_t reported_applied_amount,
                                              bool lethal) {
  return !lethal && pre_hitpoints > 1 && post_hitpoints == 1 && requested_amount >= pre_hitpoints &&
         reported_applied_amount == pre_hitpoints;
}

constexpr CaptureDecision classify_capture(std::int32_t observed_state_delta, std::int32_t reported_applied_amount,
                                           bool lethal) {
  if (observed_state_delta == 0 && !lethal) {
    return CaptureDecision::IgnoreNoStateChange;
  }
  if (observed_state_delta <= 0 || reported_applied_amount <= 0 || reported_applied_amount != observed_state_delta) {
    return CaptureDecision::RejectInconsistent;
  }
  return CaptureDecision::Publish;
}

} // namespace cr_combat_damage
