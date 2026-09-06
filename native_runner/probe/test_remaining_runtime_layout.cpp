#include "remaining_runtime_layout.h"

#include <cassert>

int main() {
    using namespace cr_remaining_runtime;

    static_assert(
        classify_resource_delta(
            kResourceWholeDeltaOffset,
            kPeriodicResourceReturnOffset) ==
        ResourceDeltaCause::Periodic);
    static_assert(
        classify_resource_delta(
            kResourceWholeDeltaOffset,
            kDeathOwnerResourceReturnOffset) ==
        ResourceDeltaCause::DeathOwner);
    static_assert(
        classify_resource_delta(
            kResourceFixedDeltaOffset,
            kDeathOpponentResourceReturnOffset) ==
        ResourceDeltaCause::DeathOpponent);
    static_assert(
        classify_resource_delta(
            kResourceFixedDeltaOffset,
            kPeriodicResourceReturnOffset) ==
        ResourceDeltaCause::None);

    static_assert(
        requested_resource_delta_fixed(ResourceDeltaCause::Periodic, 1) ==
        10000);
    static_assert(
        requested_resource_delta_fixed(
            ResourceDeltaCause::DeathOwner, 2) == 20000);
    static_assert(
        requested_resource_delta_fixed(
            ResourceDeltaCause::DeathOpponent, 40000) == 40000);
    static_assert(
        requested_resource_delta_fixed(ResourceDeltaCause::None, 1) == -1);

    static_assert(exact_area_effect_object(3, 0x0189c2e8));
    static_assert(!exact_area_effect_object(3, 0x0189c2f0));
    static_assert(!exact_area_effect_object(4, 0x0189c2e8));
    static_assert(
        exact_spawn_attach_path(
            kCharacterSpawnHelperOffset,
            kSpawnAttachHelperReturnOffset));
    static_assert(
        !exact_spawn_attach_path(
            kCharacterSpawnHelperOffset,
            kSpawnAttachHelperReturnOffset + 4));

    static_assert(
        exact_transform_transition(
            0x1000, 0x2000, 0x2000, 0));
    static_assert(
        exact_transform_transition(
            0x1000, 0x3000, 0, 0x3000));
    static_assert(
        !exact_transform_transition(
            0x1000, 0x1000, 0x1000, 0));
    static_assert(
        !exact_transform_transition(
            0x1000, 0x2000, 0x2000, 0x2000));

    assert(kObjectDataOffset == 0x48);
    assert(kResourceFixedValueOffset == 0x2f8);
    assert(kAttachedCharacterDataOffset == 0x2f0);
    assert(kSpawnAttachOffset == 0x3a5);
    assert(kAreaEffectObjectRemainingLifeOffset == 0x100);
    assert(kActionWaitDispatchReturnOffset == 0x00f58820);
    assert(kChangeGameObjectDataExecuteOffset == 0x00e58638);
    return 0;
}
