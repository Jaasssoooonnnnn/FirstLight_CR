#pragma once

#include <cstdint>

struct NativeTouchBindings {
  // Both callbacks run on Android's touch-delivery thread.  The ready check
  // must be lock-free or short, and submit_hand_action must only enqueue the
  // action; it must not wait for a future render/game frame.
  bool (*is_native_render_ready)(void *context) = nullptr;
  // Must change whenever configure-native/reset publishes a new manager.
  // Selection is scoped to this token so it cannot leak into a later match.
  std::uint64_t (*native_render_generation)(void *context) = nullptr;
  bool (*submit_hand_action)(void *context, std::int32_t owner, std::int32_t hand_index, std::int32_t world_x,
                             std::int32_t world_y) = nullptr;
  void *context = nullptr;
};

struct NativeTouchSelectionSnapshot {
  std::int32_t owner = -1;
  std::int32_t hand_index = -1;
  std::uint64_t revision = 0;
  std::uint64_t generation = 0;
  bool gesture_captured = false;
};

struct NativeStockAbilityReceipt {
  std::int32_t x = -1;
  std::int32_t y = -1;
  std::int32_t controller_slot = -1;
};

// Installs an inline hook at the v15.535.13 GameApp.nOnTouchEvent JNI entry.
// The caller must pass the already-validated base address of the exact libg
// build used by the native runner.
bool install_native_touch_interceptor(std::uintptr_t libg_base, const NativeTouchBindings &bindings);

NativeTouchSelectionSnapshot native_touch_selection_snapshot();
void native_touch_clear_selection();
