#include "combat_damage_capture.h"
#include "combat_impact_capture.h"

#include <cassert>

int main() {
    using cr_combat_damage::CaptureDecision;
    using cr_combat_damage::classify_capture;
    using cr_combat_damage::is_one_hitpoint_survival_clamp;
    using cr_combat_damage::valid_hitpoint_state;
    using cr_combat_impact::ProcessedTargetDecision;
    using cr_combat_impact::classify_processed_target_append;

    // Exact live Evolution Witch reproduction: a legal heal leaves current
    // HP above the character's static maximum field.
    static_assert(valid_hitpoint_state(1586, 1341));
    static_assert(valid_hitpoint_state(0, 0));
    static_assert(!valid_hitpoint_state(-1, 1341));
    static_assert(!valid_hitpoint_state(1341, -1));

    // Phoenix live reproduction: a post-death call retains the caller's
    // requested amount in appliedAmountOut but changes no HP/shield state.
    static_assert(
        classify_capture(0, 50, false) ==
        CaptureDecision::IgnoreNoStateChange);

    static_assert(classify_capture(50, 50, false) == CaptureDecision::Publish);
    static_assert(classify_capture(121, 121, true) == CaptureDecision::Publish);

    // A lethal result without a state transition and every non-zero mismatch
    // remain fail-closed rather than being hidden by the zero-delta exception.
    static_assert(
        classify_capture(0, 50, true) ==
        CaptureDecision::RejectInconsistent);
    static_assert(
        classify_capture(50, 0, false) ==
        CaptureDecision::RejectInconsistent);
    static_assert(
        classify_capture(50, 49, false) ==
        CaptureDecision::RejectInconsistent);
    static_assert(
        classify_capture(-1, 50, false) ==
        CaptureDecision::RejectInconsistent);

    // Berserker-style survival retains exactly one observable HP even though
    // appliedAmountOut reports the full pre-hit HP. Callers normalize this
    // proved clamp to the observed delta before classifying the event.
    static_assert(is_one_hitpoint_survival_clamp(67, 1, 81, 67, false));
    static_assert(!is_one_hitpoint_survival_clamp(67, 0, 81, 67, false));
    static_assert(!is_one_hitpoint_survival_clamp(67, 1, 81, 66, false));
    static_assert(!is_one_hitpoint_survival_clamp(67, 1, 81, 67, true));
    static_assert(
        classify_capture(66, 66, false) == CaptureDecision::Publish);

    // Live 2026-08-23 reproduction: one exact-build impact call appended
    // three processed targets, and the current candidate was not last.
    constexpr std::int32_t appended_targets[] = {5000024, 5000026, 5000025};
    static_assert(
        classify_processed_target_append(
            appended_targets, 0, 3, 4, 5000026) ==
        ProcessedTargetDecision::Publish);
    // Only the newly appended range is evidence for this call.
    constexpr std::int32_t old_and_new_targets[] = {5000026, 5000024, 5000025};
    static_assert(
        classify_processed_target_append(
            old_and_new_targets, 1, 3, 4, 5000026) ==
        ProcessedTargetDecision::RejectMissing);
    constexpr std::int32_t duplicate_targets[] = {5000026, 5000026};
    static_assert(
        classify_processed_target_append(
            duplicate_targets, 0, 2, 2, 5000026) ==
        ProcessedTargetDecision::RejectDuplicate);
    static_assert(
        classify_processed_target_append(
            appended_targets, 1, 1, 4, 5000026) ==
        ProcessedTargetDecision::RejectBounds);

    assert(
        classify_capture(0, 50, false) ==
        CaptureDecision::IgnoreNoStateChange);
    return 0;
}
