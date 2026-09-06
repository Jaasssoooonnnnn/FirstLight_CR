#include "combat_deployment_layout.h"

#include <cassert>

int main() {
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
