#include "visibility_runtime_layout.h"

#include <cassert>

int main() {
    using namespace cr_visibility_runtime;

    static_assert(
        classify_visibility_transition(0, 1) ==
        VisibilityTransition::BecameInvisible);
    static_assert(
        classify_visibility_transition(1, 0) ==
        VisibilityTransition::BecameVisible);
    static_assert(
        classify_visibility_transition(1, 2) ==
        VisibilityTransition::None);
    static_assert(
        classify_visibility_transition(2, 1) ==
        VisibilityTransition::None);

    assert(kBuffInvisibleCountOffset == 0x34);
    return 0;
}
