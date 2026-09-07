#include "action_movement_runtime_layout.h"

#include <cassert>
#include <cstring>

int main() {
    using namespace cr_action_movement_runtime;

    static_assert(kGoldenKnightHopOffset == 0x00f49d24);
    static_assert(kWarpPositionCommitOffset == 0x00e82a34);
    static_assert(
        kDashingAttackChainDataVtableOffset == 0x0188daf8);
    static_assert(
        kDashingAttackChainRuntimeVtableOffset == 0x0189eca0);
    static_assert(kWarpCharacterDataVtableOffset == 0x01895120);
    static_assert(kActionRuntimeDataOffset == 0x18);
    static_assert(kActionRuntimeOwnerOffset == 0x20);
    static_assert(kDashingAttackChainStateOffset == 0x60);
    static_assert(kDashingAttackChainIndexOffset == 0x64);
    static_assert(kActionExecutionOwnerOffset == 0x08);

    assert(
        std::strcmp(
            kGoldenKnightExecuteActionName,
            "GoldenKnight_Execute_Charge") == 0);
    assert(
        std::strcmp(
            kBossBanditWarpActionName,
            "BossBandit_ability_warp") == 0);
    assert(
        std::strcmp(
            kEliteArcherWarpActionName,
            "EliteArcherHero_Ability_Warp") == 0);
    return 0;
}
