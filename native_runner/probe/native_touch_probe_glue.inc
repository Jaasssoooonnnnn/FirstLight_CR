// Integration shim for cr_replay_probe.cpp.
//
// Include native_touch_interceptor.h with the other local headers, then
// include this file immediately after native_replay_runner_hook() and before
// install_inline_hook().  At that location every GameState/observation helper
// used below has already been defined in the probe's anonymous namespace.

bool probe_native_touch_ready(void *) {
  if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::NativeRender)) {
    return false;
  }
  const std::uint64_t submitted = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const std::uint64_t loaded = g_native_render_loaded_sequence.load(std::memory_order_acquire);
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  if (submitted == 0 || loaded != submitted || manager == nullptr ||
      g_native_render_tick.load(std::memory_order_acquire) < 0) {
    return false;
  }

  // A finalized ReplayBattleController still owns a readable manager while
  // its result screen is visible.  Do not steal result-screen taps merely
  // because nativeRenderReady remains true at the transport level.
  BattleResultView result;
  const bool readable = read_battle_result(read_object_field<void *>(manager, 0xa8), &result);
  return !readable || !result.finalized;
}

std::uint64_t probe_native_touch_generation(void *) {
  return g_native_render_loaded_sequence.load(std::memory_order_acquire);
}

bool probe_submit_native_touch_hand_action(void *, std::int32_t owner, std::int32_t hand_index, std::int32_t world_x,
                                           std::int32_t world_y) {
  constexpr std::int32_t kTouchCommandDelayTicks = 2;
  char command_json[768] = {};

  pthread_mutex_lock(&g_control_mutex);
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  const std::uint64_t submitted = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const bool ready =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      submitted != 0 && g_native_render_loaded_sequence.load(std::memory_order_acquire) == submitted &&
      manager != nullptr;
  const std::int32_t current_tick = g_native_render_tick.load(std::memory_order_acquire);

  // Never wait on the Android touch thread.  If a host action currently
  // occupies the one-command mailbox, reject this click and leave its card
  // selected so the user can tap the arena again.
  const bool mailbox_available = g_pending_native_command_sequence == g_processed_native_command_sequence;
  DeployRequest request;
  const bool resolved = ready && mailbox_available && current_tick >= 0 &&
                        current_tick <= INT32_MAX - kTouchCommandDelayTicks &&
                        resolve_hand_deploy_request(manager, owner, hand_index, world_x, world_y, &request);
  const bool formatted = resolved && format_deploy_command(request, current_tick + kTouchCommandDelayTicks,
                                                           command_json, sizeof(command_json));

  if (formatted) {
    ++g_pending_native_command_sequence;
    std::memcpy(g_pending_native_command_json, command_json, std::strlen(command_json) + 1);
    g_last_native_injection_succeeded = false;
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
  return formatted;
}

bool install_probe_native_touch(std::uintptr_t libg_base) {
  NativeTouchBindings bindings;
  bindings.is_native_render_ready = &probe_native_touch_ready;
  bindings.native_render_generation = &probe_native_touch_generation;
  bindings.submit_hand_action = &probe_submit_native_touch_hand_action;
  return install_native_touch_interceptor(libg_base, bindings);
}

bool format_probe_native_touch_status(char *response, std::size_t response_size) {
  if (response == nullptr || response_size == 0) {
    return false;
  }
  const NativeTouchSelectionSnapshot selection = native_touch_selection_snapshot();
  const int written = std::snprintf(
      response, response_size,
      "{\"ok\":true,\"mode\":\"native-render-touch\","
      "\"selectedOwner\":%d,\"selectedHandIndex\":%d,"
      "\"selectionRevision\":%llu,\"generation\":%llu,"
      "\"gestureCaptured\":%s}",
      selection.owner, selection.hand_index, static_cast<unsigned long long>(selection.revision),
      static_cast<unsigned long long>(selection.generation), selection.gesture_captured ? "true" : "false");
  return written > 0 && static_cast<std::size_t>(written) < response_size;
}
