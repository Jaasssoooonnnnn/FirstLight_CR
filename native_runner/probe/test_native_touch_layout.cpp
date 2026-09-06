#include "native_touch_layout.h"

#include <cassert>

int main() {
    using namespace cr_native_touch;
    constexpr Layout layout = kNativeRender1080x1920;

    for (std::int32_t slot = 0; slot < 4; ++slot) {
        const HandHit top = hand_at_pixel(
            layout,
            layout.hand_centers_x[slot],
            layout.top_hand_y);
        assert(top.hit);
        assert(top.owner == 0);
        assert(top.hand_index == 3 - slot);

        const HandHit bottom = hand_at_pixel(
            layout,
            layout.hand_centers_x[slot],
            layout.bottom_hand_y);
        assert(bottom.hit);
        assert(bottom.owner == 1);
        assert(bottom.hand_index == slot);
    }

    assert(!hand_at_pixel(layout, 282, layout.top_hand_y).hit);
    assert(!hand_at_pixel(layout, 900, layout.top_hand_y).hit);
    assert(!hand_at_pixel(layout, 211, 970).hit);
    assert(!hand_at_pixel(layout, -1, layout.top_hand_y).hit);

    assert(arena_contains_pixel(layout, 54, 282));
    assert(arena_contains_pixel(layout, 1026, 1666));
    assert(arena_contains_pixel(layout, 540, 970));
    assert(!arena_contains_pixel(layout, 53, 970));
    assert(!arena_contains_pixel(layout, 540, 1667));

    // Decorative padding clamps to the nearest logical outer cell.
    constexpr WorldPoint top_left = arena_pixel_to_world(layout, 54, 282);
    static_assert(top_left.x == 500 && top_left.y == 500);
    constexpr WorldPoint center = arena_pixel_to_world(layout, 551, 1070);
    static_assert(center.x == 9500 && center.y == 16500);
    constexpr WorldPoint bottom_right =
        arena_pixel_to_world(layout, 1026, 1666);
    static_assert(bottom_right.x == 17500 && bottom_right.y == 31500);

    // Measured visual centers of the first/last columns and king-tower back row.
    constexpr WorldPoint far_left = arena_pixel_to_world(layout, 135, 1350);
    static_assert(far_left.x == 500 && far_left.y == 23500);
    constexpr WorldPoint far_right = arena_pixel_to_world(layout, 943, 1350);
    static_assert(far_right.x == 17500 && far_right.y == 23500);
    constexpr WorldPoint king_back = arena_pixel_to_world(layout, 540, 1647);
    static_assert(king_back.x == 9500 && king_back.y == 31500);

    // Conversion is fail-closed at the outermost legal deployment centers.
    constexpr WorldPoint beyond = arena_pixel_to_world(layout, 2000, 3000);
    static_assert(beyond.x == 17500 && beyond.y == 31500);

    return 0;
}
