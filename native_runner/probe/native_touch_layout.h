#pragma once

#include <cstdint>

// Pure, platform-independent geometry for the stock v15 replay HUD used by
// native-render.  Keeping this separate from the Android hook makes every hit
// box and the pixel-to-world transform host-testable.
namespace cr_native_touch {

struct Layout {
  std::int32_t screen_width;
  std::int32_t screen_height;
  std::int32_t arena_left;
  std::int32_t arena_top;
  std::int32_t arena_right;
  std::int32_t arena_bottom;
  std::int32_t world_left;
  std::int32_t world_top;
  std::int32_t world_right;
  std::int32_t world_bottom;
  std::int32_t top_hand_y;
  std::int32_t bottom_hand_y;
  std::int32_t hand_half_width;
  std::int32_t hand_half_height;
  std::int32_t hand_centers_x[4];
};

// Measured from captures/native_render_offline_final.png (1080x1920).
// The Android Surface itself is 1080x1920 even when the MuMu host window is
// resized, so host-window scaling does not change these input coordinates.
inline constexpr Layout kNativeRender1080x1920 = {
    1080, 1920, 54, 282, 1026, 1666, 111, 436, 967, 1666, 171, 1749, 67, 86, {211, 354, 497, 640},
};

struct HandHit {
  bool hit = false;
  std::int32_t owner = -1;
  std::int32_t hand_index = -1;
};

struct WorldPoint {
  std::int32_t x = 0;
  std::int32_t y = 0;
};

constexpr std::int32_t absolute(std::int32_t value) { return value < 0 ? -value : value; }

constexpr bool screen_point_valid(const Layout &layout, std::int32_t x, std::int32_t y) {
  return x >= 0 && y >= 0 && x < layout.screen_width && y < layout.screen_height;
}

constexpr HandHit hand_at_pixel(const Layout &layout, std::int32_t x, std::int32_t y) {
  if (!screen_point_valid(layout, x, y)) {
    return {};
  }

  std::int32_t owner = -1;
  if (absolute(y - layout.top_hand_y) <= layout.hand_half_height) {
    owner = 0;
  } else if (absolute(y - layout.bottom_hand_y) <= layout.hand_half_height) {
    owner = 1;
  } else {
    return {};
  }

  for (std::int32_t index = 0; index < 4; ++index) {
    if (absolute(x - layout.hand_centers_x[index]) <= layout.hand_half_width) {
      // The spectator HUD mirrors owner 0's hand at the top but keeps
      // owner 1's bottom hand in native order.
      const std::int32_t hand_index = owner == 0 ? 3 - index : index;
      return {true, owner, hand_index};
    }
  }
  return {};
}

constexpr bool arena_contains_pixel(const Layout &layout, std::int32_t x, std::int32_t y) {
  return x >= layout.arena_left && x <= layout.arena_right && y >= layout.arena_top && y <= layout.arena_bottom;
}

constexpr std::int32_t clamp_deploy(std::int64_t value, std::int32_t minimum, std::int32_t maximum) {
  return value <= minimum ? minimum : value >= maximum ? maximum : static_cast<std::int32_t>(value);
}

// Inverse of the measured logical 18x32 replay-HUD grid.  The visible arena
// catchment includes decorative padding, which clamps to the nearest cell.
// Returning cell centers mirrors the stock command stream and keeps the outer
// columns and the final rows behind the king towers reachable.
constexpr WorldPoint arena_pixel_to_world(const Layout &layout, std::int32_t x, std::int32_t y) {
  const std::int64_t pixel_width = layout.world_right - layout.world_left;
  const std::int64_t pixel_height = layout.world_bottom - layout.world_top;
  if (pixel_width <= 0 || pixel_height <= 0) {
    return {};
  }

  const auto cell_center = [](std::int32_t pixel, std::int32_t minimum, std::int32_t maximum,
                              std::int32_t cell_count) constexpr -> std::int32_t {
    if (pixel <= minimum) {
      return 500;
    }
    if (pixel >= maximum) {
      return (cell_count - 1) * 1000 + 500;
    }
    const std::int64_t raw_world = static_cast<std::int64_t>(pixel - minimum) * cell_count * 1000 / (maximum - minimum);
    const std::int32_t cell = clamp_deploy(raw_world / 1000, 0, cell_count - 1);
    return cell * 1000 + 500;
  };

  // Native world Y grows in the same direction as the stock replay surface:
  // owner 0 is near Y=0 at the top and owner 1 near Y=32000 at the bottom.
  return {
      cell_center(x, layout.world_left, layout.world_right, 18),
      cell_center(y, layout.world_top, layout.world_bottom, 32),
  };
}

} // namespace cr_native_touch
