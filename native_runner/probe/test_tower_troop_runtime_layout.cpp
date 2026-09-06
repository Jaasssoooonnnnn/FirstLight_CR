#include "tower_troop_runtime_layout.h"

#include <cassert>

int main() {
    using namespace cr_tower_troop_runtime;

    assert(kDaggerRuntimeVtableOffset == 0x0189e0c8);
    assert(kDaggerStaticVtableOffset == 0x0188c3f8);
    assert(kDaggerChargeCountOffset == 0x60);
    assert(kDaggerRechargeElapsedMsOffset == 0x64);
    assert(kDaggerMaxChargeCountOffset == 0xec);
    assert(kDaggerRechargeDurationMsOffset == 0xf0);
    assert(kDaggerRechargeIncrementOffset == 0xf4);

    assert(kChefRuntimeVtableOffset == 0x0189e738);
    assert(kChefStaticVtableOffset == 0x0188cf78);
    assert(kChefStartDelayRemainingMsOffset == 0x68);
    assert(kChefCookingContributionOffset == 0x6c);
    assert(kChefTargetHandleOffset == 0x70);
    assert(kChefThrowDelayRemainingMsOffset == 0x74);
    assert(kChefTargetObjectOffset == 0x78);
    assert(kChefContributionNeededOffset == 0xec);
    assert(kChefStartDelayDurationMsOffset == 0x184);
    assert(kChefContributionScale == 20);
    return 0;
}
