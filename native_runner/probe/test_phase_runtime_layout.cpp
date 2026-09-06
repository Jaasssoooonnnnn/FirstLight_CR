#include "phase_runtime_layout.h"
#include "special_movement_runtime_layout.h"

#include <cassert>
#include <initializer_list>

int main() {
    using namespace cr_phase_runtime;

    EffectScale scale;
    for (const auto value : {0, 1, 99, 100, 130, -30, -100}) {
        accumulate_effect_multiplier(scale, value);
    }
    assert(scale.positive_percent == 130);
    assert(scale.negative_magnitude == 100);
    accumulate_effect_multiplier(scale, std::numeric_limits<std::int32_t>::min());
    assert(scale.negative_magnitude == std::numeric_limits<std::int32_t>::max());
    accumulate_effect_multiplier(scale, std::numeric_limits<std::int32_t>::max());
    assert(scale.positive_percent == std::numeric_limits<std::int32_t>::max());
    assert(!valid_classic_charge_progress(-2));
    assert(valid_classic_charge_progress(-1));
    assert(valid_classic_charge_progress(0));
    assert(valid_classic_charge_progress(10000));

    assert(kAttackSequenceStageOffset == 0x20);
    assert(kClassicChargeProgressOffset == 0x1e0);
    assert(kBuffAssetHitSpeedMultiplierOffset == 0xe0);
    assert(kBuffAssetSpeedMultiplierOffset == 0xe4);
    assert(
        cr_special_movement_runtime::kExecuteOffset == 0x00f609dc);
    assert(
        cr_special_movement_runtime::kExecuteCallerReturnOffset ==
        0x00f61bf0);
    assert(
        cr_special_movement_runtime::kComponentDashCooldownOffset == 0x40);
    return 0;
}
