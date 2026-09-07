#include "probe_json.h"
#include <initializer_list>

// Resident multi-match execution for semantic headless training.
//
// The control server is a single persistent pthread.  After the legacy
// headless worker has exited, that thread becomes the sole libg execution
// lane.  Each slot owns a live GameStateManager and its replay root; switching
// slots only swaps probe bookkeeping and telemetry epochs, never serializes or
// reconstructs the native match.

constexpr std::size_t kResidentMaxSlots = 16;

struct ResidentMatchSlot {
  bool occupied = false;
  void *manager = nullptr;
  void *replay_root = nullptr;
  char replay_json[kMaxReplayConfigBytes] = {};
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::uint64_t config_revision = 0;
  std::uint64_t completed_steps = 0;
  std::int32_t tick = -1;
  std::int32_t last_loaded_queue_count = -1;
  std::uint32_t stagnant_steps = 0;
  bool ended = false;
  bool telemetry_ready = false;
  // Tower troop actions are not guaranteed to tick during every five-tick
  // resident slice.  Preserve their last authoritative state across slot
  // switches instead of losing it when the process-global hook epoch resets.
  std::size_t tower_troop_runtime_count = 0;
  StoredTowerTroopRuntimeState tower_troop_runtimes[kMaxStoredTowerTroopRuntimes];
};

ResidentMatchSlot g_resident_slots[kResidentMaxSlots];
std::int32_t g_resident_bound_slot = -1;
std::uint64_t g_resident_switch_count = 0;
std::uint64_t g_resident_total_steps = 0;
// Binding a resident slot restores that match's observation identity into the
// legacy globals.  New matches must not allocate from those restored values:
// doing so can reuse another live slot's generation/state-epoch pair.
std::uint64_t g_resident_generation_frontier = 0;
std::uint64_t g_resident_state_epoch_frontier = 0;

bool resident_slot_id_valid(std::int32_t slot_id) {
  return slot_id >= 0 && static_cast<std::size_t>(slot_id) < kResidentMaxSlots;
}

void resident_save_bound_slot() {
  if (!resident_slot_id_valid(g_resident_bound_slot)) {
    return;
  }
  ResidentMatchSlot &slot = g_resident_slots[g_resident_bound_slot];
  if (!slot.occupied || slot.manager != g_controlled_manager) {
    return;
  }
  slot.generation = g_control_generation;
  slot.state_epoch = g_state_epoch;
  slot.config_revision = g_config_processed_sequence;
  slot.completed_steps = g_completed_steps;
  slot.tick = g_last_controlled_tick;
  slot.last_loaded_queue_count = g_last_loaded_queue_count;
  slot.ended = g_control_ended;
  if (slot.telemetry_ready && slot.tick >= 0) {
    slot.tower_troop_runtime_count = copy_tower_troop_runtime_snapshot(
        slot.generation, slot.state_epoch, slot.tick, slot.tower_troop_runtimes, kMaxStoredTowerTroopRuntimes);
  }
}

void resident_clear_probe_context() {
  g_controlled_manager = nullptr;

  g_controlled_step_active = false;
  g_completed_steps = 0;
  g_step_budget = 0;
  g_last_controlled_tick = -1;
  g_control_gate_enabled = true;
  g_control_ended = false;
  g_last_loaded_queue_count = -1;
  g_pending_command_sequence = 0;
  g_processed_command_sequence = 0;
  g_pending_command_json[0] = '\0';
  g_last_injection_succeeded = false;
  g_snapshot_request_sequence = 0;
  g_snapshot_processed_sequence = 0;
  g_pending_snapshot_operation = SnapshotOperation::None;
  g_pending_snapshot_handle = 0;
  g_last_snapshot_operation_succeeded = false;
  g_last_snapshot_operation_error = SnapshotError::None;
  g_last_snapshot_operation_result = {};
  for (auto &snapshot : g_stored_snapshots) {
    snapshot = {};
  }
  g_next_snapshot_handle = 1;
}

bool resident_bind_slot(std::int32_t slot_id, bool reset_telemetry = true) {
  if (!resident_slot_id_valid(slot_id)) {
    return false;
  }
  ResidentMatchSlot &slot = g_resident_slots[slot_id];
  if (!slot.occupied || slot.manager == nullptr) {
    return false;
  }
  if (g_resident_bound_slot == slot_id && g_controlled_manager == slot.manager) {
    return true;
  }

  resident_save_bound_slot();
  resident_clear_probe_context();
  g_resident_bound_slot = slot_id;
  ++g_resident_switch_count;

  g_controlled_manager = slot.manager;
  g_control_generation = slot.generation;
  g_state_epoch = slot.state_epoch;
  g_config_processed_sequence = slot.config_revision;
  g_completed_steps = slot.completed_steps;
  g_last_controlled_tick = slot.tick;
  g_last_loaded_queue_count = slot.last_loaded_queue_count;
  g_control_ended = slot.ended;

  if (reset_telemetry) {
    begin_combat_event_epoch(slot.manager, slot.generation, slot.state_epoch);
    begin_runtime_telemetry_epoch(slot.manager, slot.generation, slot.state_epoch);
    bind_combat_event_object_manager(slot.manager);
    restore_tower_troop_runtime_snapshot(slot.generation, slot.state_epoch, slot.tick, slot.tower_troop_runtimes,
                                         slot.tower_troop_runtime_count);
    // Runtime hook rings are process-local.  They become attributable to
    // this slot only after this newly bound epoch executes a native step.
    slot.telemetry_ready = false;
  }
  return true;
}

void resident_remove_trace(void *manager) {
  pthread_mutex_lock(&g_trace_mutex);
  for (auto &trace : g_manager_traces) {
    if (trace.manager == manager) {
      trace = {};
      break;
    }
  }
  pthread_mutex_unlock(&g_trace_mutex);
}

void resident_destroy_slot(std::int32_t slot_id) {
  if (!resident_slot_id_valid(slot_id)) {
    return;
  }
  ResidentMatchSlot &slot = g_resident_slots[slot_id];
  if (!slot.occupied) {
    return;
  }
  if (g_resident_bound_slot != slot_id) {
    resident_bind_slot(slot_id);
  }
  pthread_mutex_lock(&g_control_mutex);
  clear_scheduled_replay_actions_for_manager_locked(slot.manager);
  pthread_mutex_unlock(&g_control_mutex);
  release_all_stored_snapshots_on_worker();
  end_combat_event_epoch(slot.manager);
  resident_remove_trace(slot.manager);
  if (slot.manager != nullptr) {
    LOGI("resident slot %d destroying manager=%p", slot_id, slot.manager);
    g_game_state_destroy(slot.manager);
    g_game_state_base_destroy(slot.manager);
    g_game_state_delete(slot.manager);
  }
  release_json_object(slot.replay_root);
  slot = {};
  resident_clear_probe_context();
  g_resident_bound_slot = -1;
}

bool resident_start_mode(char *response, std::size_t response_size) {
  pthread_mutex_lock(&g_control_mutex);
  if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::Headless)) {
    pthread_mutex_unlock(&g_control_mutex);
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"error\":\"resident mode requires semantic headless mode\"}");
    return false;
  }
  if (!g_resident_mode_active) {
    g_resident_mode_requested = true;
    if (g_native_worker_running) {
      ++g_reset_request_sequence;
      pthread_cond_broadcast(&g_control_cond);
      while (g_native_worker_running) {
        pthread_cond_wait(&g_control_cond, &g_control_mutex);
      }
    }
    if (g_control_generation > g_resident_generation_frontier) {
      g_resident_generation_frontier = g_control_generation;
    }
    if (g_state_epoch > g_resident_state_epoch_frontier) {
      g_resident_state_epoch_frontier = g_state_epoch;
    }
    resident_clear_probe_context();
    g_resident_mode_active = true;
    g_resident_bound_slot = -1;
    LOGI("resident multi-match mode activated");
  }
  pthread_mutex_unlock(&g_control_mutex);
  return true;
}

bool resident_parse_replay(const char *replay_json, void **replay_root_out, std::int32_t *location_id_out,
                           std::int32_t *stage_out) {
  if (replay_root_out == nullptr || location_id_out == nullptr || stage_out == nullptr) {
    return false;
  }
  *replay_root_out = nullptr;
  *location_id_out = 0;
  *stage_out = replay_json == nullptr || replay_json[0] == '\0' ? 0 : 1;
  if (replay_json == nullptr || replay_json[0] != '{' || !extract_location_id(replay_json, location_id_out) ||
      !validate_empty_match_streams(replay_json)) {
    return false;
  }
  void *root = parse_json_text(replay_json);
  if (root == nullptr) {
    return false;
  }
  *stage_out = 2;
  void *battle = lookup_json_member(root, "battle");
  if (battle == nullptr) {
    release_json_object(root);
    return false;
  }
  *stage_out = 3;
  void *commands = lookup_json_member(root, "cmd");
  void *events = lookup_json_member(root, "evt");
  std::int32_t command_count = -1;
  std::int32_t event_count = -1;
  if (!inspect_pointer_array(commands, &command_count) || !inspect_pointer_array(events, &event_count) ||
      command_count != 0 || event_count != 0) {
    release_json_object(root);
    return false;
  }
  *stage_out = 6;
  *replay_root_out = root;
  return true;
}

bool resident_construct_slot(std::int32_t slot_id, const char *replay_json, char *response, std::size_t response_size) {
  if (!resident_slot_id_valid(slot_id) || replay_json == nullptr) {
    return false;
  }
  const std::size_t replay_size = std::strlen(replay_json);
  if (replay_size < 2 || replay_size >= kMaxReplayConfigBytes) {
    std::snprintf(
        response, response_size,
        "{\"ok\":false,\"envId\":%d,\"error\":\"configuration must be one compact JSON object under 64 KiB\"}",
        slot_id);
    return false;
  }

  void *replay_root = nullptr;
  std::int32_t location_id = 0;
  std::int32_t stage = 0;
  if (!resident_parse_replay(replay_json, &replay_root, &location_id, &stage)) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"envId\":%d,\"stage\":%d,\"error\":\"resident configuration was rejected\"}", slot_id,
                  stage);
    return false;
  }

  if (g_resident_slots[slot_id].occupied) {
    resident_destroy_slot(slot_id);
  }
  resident_save_bound_slot();
  resident_clear_probe_context();
  g_resident_bound_slot = slot_id;

  void *const context_a = g_game_state_context_a();
  void *const context_b = g_game_state_context_b();
  if (!prewarm_location_tilemap(context_a, location_id)) {
    release_json_object(replay_root);
    g_resident_bound_slot = -1;
    std::snprintf(response, response_size, "{\"ok\":false,\"envId\":%d,\"error\":\"could not prepare native tilemap\"}",
                  slot_id);
    return false;
  }
  void *const manager = g_game_state_new(0x230);
  if (manager == nullptr) {
    release_json_object(replay_root);
    g_resident_bound_slot = -1;
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"envId\":%d,\"error\":\"could not allocate GameStateManager\"}", slot_id);
    return false;
  }

  LOGI("resident slot %d constructing manager=%p root=%p tid=%ld", slot_id, manager, replay_root,
       static_cast<long>(syscall(__NR_gettid)));
  g_game_state_constructor(manager, 1, context_a, context_b, 0);
  g_game_state_setup(manager, 1);
  // game_state_load_hook allocates the next identity by incrementing these
  // globals.  Seed it from the process-wide frontier, never from whichever
  // resident slot happened to be bound before this construction.
  g_control_generation = g_resident_generation_frontier;
  g_state_epoch = g_resident_state_epoch_frontier;
  g_expected_headless_load_manager.store(manager, std::memory_order_release);
  game_state_load_hook(manager, replay_root, nullptr, nullptr);
  g_expected_headless_load_manager.store(nullptr, std::memory_order_release);
  if (g_control_generation > g_resident_generation_frontier) {
    g_resident_generation_frontier = g_control_generation;
  }
  if (g_state_epoch > g_resident_state_epoch_frontier) {
    g_resident_state_epoch_frontier = g_state_epoch;
  }
  if (g_controlled_manager != manager || g_last_loaded_queue_count != 0) {
    end_combat_event_epoch(manager);
    g_game_state_destroy(manager);
    g_game_state_base_destroy(manager);
    g_game_state_delete(manager);
    release_json_object(replay_root);
    resident_clear_probe_context();
    g_resident_bound_slot = -1;
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"envId\":%d,\"error\":\"native manager did not load a fresh empty-stream match\"}",
                  slot_id);
    return false;
  }

  ResidentMatchSlot &slot = g_resident_slots[slot_id];
  slot.occupied = true;
  slot.manager = manager;
  slot.replay_root = replay_root;
  std::memcpy(slot.replay_json, replay_json, replay_size + 1);
  slot.generation = g_control_generation;
  slot.state_epoch = g_state_epoch;
  slot.config_revision = g_control_generation;
  slot.completed_steps = 0;
  slot.tick = g_last_controlled_tick;
  slot.last_loaded_queue_count = g_last_loaded_queue_count;
  slot.stagnant_steps = 0;
  slot.ended = false;
  slot.telemetry_ready = true;
  g_config_processed_sequence = slot.config_revision;
  resident_save_bound_slot();
  std::snprintf(response, response_size,
                "{\"ok\":true,\"mode\":\"resident-headless\",\"envId\":%d,"
                "\"generation\":%llu,\"stateEpoch\":%llu,\"tick\":%d,\"ready\":true}",
                slot_id, static_cast<unsigned long long>(slot.generation),
                static_cast<unsigned long long>(slot.state_epoch), slot.tick);
  return true;
}

bool resident_step_bound(ResidentMatchSlot &slot, std::uint64_t requested_steps) {
  if (slot.manager == nullptr || requested_steps == 0 || requested_steps > 1000000ULL) {
    return false;
  }

  for (std::uint64_t index = 0; index < requested_steps && !g_control_ended; ++index) {
    const std::int32_t before = g_game_state_tick == nullptr ? -1 : g_game_state_tick(slot.manager);
    process_scheduled_replay_actions(slot.manager);
    g_original_game_state_step(slot.manager);
    const std::int32_t after = g_game_state_tick == nullptr ? -1 : g_game_state_tick(slot.manager);
    ++g_completed_steps;
    ++g_resident_total_steps;
    g_last_controlled_tick = after;
    if (after == before) {
      ++slot.stagnant_steps;
    } else {
      slot.stagnant_steps = 0;
    }
    BattleResultView result;
    const bool readable = read_battle_result(read_object_field<void *>(slot.manager, 0xa8), &result);
    if ((readable && result.finalized) || slot.stagnant_steps >= 4) {
      g_control_ended = true;
    }
  }

  slot.telemetry_ready = true;
  resident_save_bound_slot();
  return true;
}

bool resident_send_observation(int socket_fd, std::int32_t slot_id, const char *kind) {
  if (!resident_bind_slot(slot_id)) {
    send_control_error(socket_fd, "resident env is not configured");
    return false;
  }
  if (std::strcmp(kind, "observe") != 0 && !g_resident_slots[slot_id].telemetry_ready) {
    send_control_error(socket_fd, "resident rich telemetry requires at least one step after this env is rebound");
    return false;
  }
  const std::size_t response_size =
      std::strcmp(kind, "observe") == 0 ? kOrdinaryObservationResponseBytes : kFullObservationResponseBytes;
  std::unique_ptr<char[]> response(new (std::nothrow) char[response_size]());
  if (response == nullptr) {
    send_control_error(socket_fd, "could not allocate resident observation response");
    return false;
  }
  bool succeeded = false;
  if (std::strcmp(kind, "observe") == 0) {
    succeeded = build_observation_json(g_controlled_manager, g_control_generation, response.get(), response_size);
  } else if (std::strcmp(kind, "observe-rich") == 0) {
    succeeded = build_rich_observation_json(g_controlled_manager, g_control_generation, response.get(), response_size);
  } else {
    succeeded = build_atomic_observation_json_locked(response.get(), response_size);
  }
  if (!succeeded && response[0] == '\0') {
    std::snprintf(response.get(), response_size,
                  "{\"ok\":false,\"envId\":%d,\"error\":\"failed to encode resident observation\"}", slot_id);
  }
  send_control_response(socket_fd, response.get());
  return succeeded;
}

bool resident_send_step_observation(int socket_fd, std::int32_t slot_id, std::uint64_t steps) {
  if (!resident_bind_slot(slot_id) || steps == 0 || steps > 1000000ULL) {
    send_control_error(socket_fd, "resident step requires a configured env and 1..1000000 ticks");
    return false;
  }
  ResidentMatchSlot &slot = g_resident_slots[slot_id];
  if (!resident_step_bound(slot, steps)) {
    send_control_error(socket_fd, "resident native step failed");
    return false;
  }
  return resident_send_observation(socket_fd, slot_id, "observe");
}

void resident_send_status(int socket_fd) {
  char response[kControlResponseBytes] = {};
  std::size_t used = 0;
  std::size_t occupied = 0;
  for (const auto &slot : g_resident_slots)
    occupied += slot.occupied;
  JsonWriter writer(response, sizeof(response), used);
  writer.begin_object();
  writer.field("ok", true);
  writer.field("mode", "resident-headless");
  writer.field("capacity", kResidentMaxSlots);
  writer.field("occupied", occupied);
  writer.field("boundEnvId", g_resident_bound_slot);
  writer.field("switches", g_resident_switch_count);
  writer.field("totalSteps", g_resident_total_steps);
  writer.begin_array("slots");
  for (std::size_t index = 0; index < kResidentMaxSlots; ++index) {
    const auto &slot = g_resident_slots[index];
    if (!slot.occupied)
      continue;
    writer.object_element();
    writer.field("envId", index);
    writer.append(",\"manager\":\"%p\"", slot.manager);
    writer.field("generation", slot.generation);
    writer.field("stateEpoch", slot.state_epoch);
    writer.field("steps", slot.completed_steps);
    writer.field("tick", slot.tick);
    writer.field("ended", slot.ended);
    writer.field("telemetryReady", slot.telemetry_ready);
    writer.end_object();
  }
  writer.end_array();
  writer.end_object();
  send_control_response(socket_fd,
                        writer.good() ? response : "{\"ok\":false,\"error\":\"resident status encoding failed\"}");
}

bool handle_resident_env_command(int socket_fd, std::int32_t slot_id, const char *command) {
  char response[kControlResponseBytes] = {};
  if (!resident_slot_id_valid(slot_id)) {
    send_control_error(socket_fd, "env id must be in 0..15");
    return true;
  }
  if (std::strncmp(command, "configure ", 10) == 0) {
    if (!resident_start_mode(response, sizeof(response))) {
      send_control_response(socket_fd, response);
      return true;
    }
    resident_construct_slot(slot_id, command + 10, response, sizeof(response));
    send_control_response(socket_fd, response);
    return true;
  }
  if (!g_resident_mode_active) {
    send_control_error(socket_fd, "resident mode is not active; configure an env first");
    return true;
  }
  if (std::strcmp(command, "status") == 0) {
    if (!resident_bind_slot(slot_id)) {
      send_control_error(socket_fd, "resident env is not configured");
      return true;
    }
    ResidentMatchSlot &slot = g_resident_slots[slot_id];
    std::snprintf(response, sizeof(response),
                  "{\"ok\":true,\"mode\":\"resident-headless\",\"envId\":%d,"
                  "\"ready\":true,\"coldReady\":true,\"generation\":%llu,"
                  "\"stateEpoch\":%llu,\"steps\":%llu,\"tick\":%d,\"ended\":%s}",
                  slot_id, static_cast<unsigned long long>(slot.generation),
                  static_cast<unsigned long long>(slot.state_epoch),
                  static_cast<unsigned long long>(slot.completed_steps), slot.tick, slot.ended ? "true" : "false");
    send_control_response(socket_fd, response);
    return true;
  }
  if (std::strcmp(command, "observe") == 0 || std::strcmp(command, "observe-rich") == 0 ||
      std::strcmp(command, "observe-atomic") == 0) {
    resident_send_observation(socket_fd, slot_id, command);
    return true;
  }
  if (std::strcmp(command, "reset") == 0) {
    if (!g_resident_slots[slot_id].occupied) {
      send_control_error(socket_fd, "resident env is not configured");
      return true;
    }
    char replay_json[kMaxReplayConfigBytes] = {};
    std::memcpy(replay_json, g_resident_slots[slot_id].replay_json, sizeof(replay_json));
    resident_construct_slot(slot_id, replay_json, response, sizeof(response));
    send_control_response(socket_fd, response);
    return true;
  }
  if (std::strcmp(command, "close") == 0) {
    const bool existed = g_resident_slots[slot_id].occupied;
    resident_destroy_slot(slot_id);
    std::snprintf(response, sizeof(response), "{\"ok\":true,\"envId\":%d,\"closed\":%s}", slot_id,
                  existed ? "true" : "false");
    send_control_response(socket_fd, response);
    return true;
  }

  unsigned long long requested_steps = 0;
  char trailing = '\0';
  if (std::sscanf(command, "step %llu %c", &requested_steps, &trailing) == 1) {
    resident_send_step_observation(socket_fd, slot_id, static_cast<std::uint64_t>(requested_steps));
    return true;
  }
  if (std::strncmp(command, "inject ", 7) == 0) {
    if (!resident_bind_slot(slot_id) || !inject_logic_command(g_resident_slots[slot_id].manager, command + 7)) {
      send_control_error(socket_fd, "resident command injection failed");
      return true;
    }
    std::snprintf(response, sizeof(response),
                  "{\"ok\":true,\"mode\":\"resident-headless\",\"envId\":%d,"
                  "\"tick\":%d,\"injected\":true}",
                  slot_id, g_last_controlled_tick);
    send_control_response(socket_fd, response);
    return true;
  }
  if (std::strncmp(command, "deploy ", 7) == 0) {
    unsigned long long player_id = 0;
    unsigned int card_id = 0;
    unsigned int card_parameter = 0;
    int position_x = 0;
    int position_y = 0;
    int delay = 2;
    const int fields = std::sscanf(command, "deploy %llu %u %u %d %d %d", &player_id, &card_id, &card_parameter,
                                   &position_x, &position_y, &delay);
    if ((fields != 5 && fields != 6) || delay < 1 || !resident_bind_slot(slot_id)) {
      send_control_error(socket_fd, "invalid resident deploy command");
      return true;
    }
    DeployRequest action = {
        static_cast<std::uint64_t>(player_id),
        static_cast<std::uint32_t>(card_id),
        static_cast<std::uint32_t>(card_parameter),
        position_x,
        position_y,
    };
    char command_json[768] = {};
    const bool succeeded =
        format_deploy_command(action, g_last_controlled_tick + delay, command_json, sizeof(command_json)) &&
        inject_logic_command(g_resident_slots[slot_id].manager, command_json);
    std::snprintf(response, sizeof(response), "{\"ok\":%s,\"mode\":\"resident-headless\",\"envId\":%d,\"injected\":%s}",
                  succeeded ? "true" : "false", slot_id, succeeded ? "true" : "false");
    send_control_response(socket_fd, response);
    return true;
  }
  if (std::strncmp(command, "activate-ability ", 17) == 0) {
    std::int32_t owner = -1;
    std::int32_t object_index = -1;
    std::int32_t secondary_index = -1;
    std::int32_t delay = 1;
    int consumed = 0;
    int fields = std::sscanf(command + 17, "%d %d %d %d%n", &owner, &object_index, &secondary_index, &delay, &consumed);
    if (fields == 3) {
      consumed = 0;
      fields = std::sscanf(command + 17, "%d %d %d%n", &owner, &object_index, &secondary_index, &consumed);
      delay = 1;
    }
    if ((fields != 3 && fields != 4) || consumed <= 0 || delay < 1 || delay > 65535 || !resident_bind_slot(slot_id)) {
      send_control_error(socket_fd, "invalid resident ability request");
      return true;
    }
    AbilityRequest request;
    char command_json[384] = {};
    const bool succeeded =
        resolve_ready_ability_request(g_resident_slots[slot_id].manager, owner, object_index, secondary_index,
                                      &request) &&
        format_ability_command(request, g_last_controlled_tick + delay - 2, command_json, sizeof(command_json)) &&
        inject_logic_command(g_resident_slots[slot_id].manager, command_json);
    std::snprintf(response, sizeof(response),
                  "{\"ok\":%s,\"mode\":\"resident-headless\",\"envId\":%d,"
                  "\"tick\":%d,\"executeTick\":%d,\"injected\":%s}",
                  succeeded ? "true" : "false", slot_id, g_last_controlled_tick, g_last_controlled_tick + delay,
                  succeeded ? "true" : "false");
    send_control_response(socket_fd, response);
    return true;
  }

  if (is_replay_schedule_command(command)) {
    if (!resident_bind_slot(slot_id)) {
      send_control_error(socket_fd, "resident env is not configured");
      return true;
    }
    handle_replay_schedule_command(socket_fd, command);
    return true;
  }

  send_control_error(socket_fd, "resident commands: configure, status, observe, observe-rich, observe-atomic, reset, "
                                "close, step, deploy, inject, activate-ability, replay-schedule-card, "
                                "replay-schedule-ability, replay-schedule-status, replay-schedule-clear");
  return true;
}

constexpr std::uint16_t kResidentBatchVersion = 1;
constexpr std::size_t kResidentBatchHandshakeBytes = 12;
constexpr std::size_t kResidentBatchHeaderBytes = 20;
constexpr std::size_t kResidentBatchRequestEntryBytes = 8;
constexpr std::size_t kResidentBatchResponseEntryBytes = 12;
constexpr std::size_t kResidentBatchActionBytes = 20;
constexpr std::size_t kResidentBatchReceiptBytes = 40;
constexpr char kResidentBatchHandshakeMagic[4] = {'C', 'R', 'B', 'H'};
constexpr char kResidentBatchRequestMagic[4] = {'C', 'R', 'B', 'Q'};
constexpr char kResidentBatchResponseMagic[4] = {'C', 'R', 'B', 'S'};
constexpr std::uint8_t kResidentBatchActionPlay = 1;
constexpr std::uint8_t kResidentBatchActionAbility = 2;

struct ResidentBatchAction {
  std::uint8_t kind = 0;
  std::uint8_t owner = 0;
  std::uint16_t execute_offset = 0;
  std::int32_t argument0 = 0;
  std::int32_t argument1 = 0;
  std::int32_t argument2 = 0;
};

struct ResidentBatchReceipt {
  std::uint8_t kind = 0;
  std::uint8_t owner = 0;
  std::uint16_t action_index = 0;
  std::uint16_t status = 0;
  std::int32_t execute_tick = -1;
  std::uint32_t card_id = 0;
  std::uint32_t card_parameter = 0;
  std::int32_t deck_slot = -1;
  std::int32_t cost = -1;
  std::int32_t form_code = -1;
  std::int32_t argument0 = 0;
  std::int32_t argument1 = 0;
};

template <typename T> void resident_write_le(std::uint8_t *destination, T value) {
  using Unsigned = typename std::make_unsigned<T>::type;
  const auto bits = static_cast<Unsigned>(value);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    destination[i] = static_cast<std::uint8_t>(bits >> (i * 8));
  }
}

template <typename T> T resident_read_le(const std::uint8_t *source) {
  using Unsigned = typename std::make_unsigned<T>::type;
  Unsigned bits = 0;
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    bits |= static_cast<Unsigned>(source[i]) << (i * 8);
  }
  return static_cast<T>(bits);
}

constexpr char kResidentTrainingCaptureMagic[4] = {'C', 'R', 'T', 'F'};
constexpr std::uint16_t kResidentTrainingCaptureVersion = 4;
// Chaotic late-game states can retain enough combat/phase ring entries to
// exceed 512 KiB even though ordinary training frames are only a few KiB.
// This is scratch capacity only: the batch channel still sends writer.used.
constexpr std::size_t kResidentTrainingCaptureBytes = 4 * 1024 * 1024;
constexpr std::int32_t kResidentTrainingNoneI32 = INT32_MIN;

struct ResidentTrainingWriter {
  std::uint8_t *data = nullptr;
  std::size_t capacity = 0;
  std::size_t used = 0;
  bool valid = true;

  bool reserve(std::size_t size) {
    if (!valid || used > capacity || size > capacity - used) {
      valid = false;
    }
    return valid;
  }

  bool bytes(const void *source, std::size_t size) {
    if (source == nullptr || !reserve(size)) {
      return valid = false;
    }
    std::memcpy(data + used, source, size);
    used += size;
    return true;
  }

  template <typename T> bool integer(T value) {
    static_assert(std::is_integral<T>::value, "wire fields must be integers");
    if (!reserve(sizeof(T)))
      return false;
    resident_write_le<T>(data + used, value);
    used += sizeof(T);
    return true;
  }

  template <typename T> bool values(std::initializer_list<T> fields) {
    for (const T value : fields)
      integer(value);
    return valid;
  }

  bool u8(std::uint8_t value) { return integer(value); }
  bool u16(std::uint16_t value) { return integer(value); }
  bool u32(std::uint32_t value) { return integer(value); }
  bool i32(std::int32_t value) { return integer(value); }
  bool u64(std::uint64_t value) { return integer(value); }
};

std::uint32_t resident_training_flags(std::initializer_list<bool> fields) {
  std::uint32_t flags = 0;
  std::uint32_t bit = 1;
  for (const bool set : fields) {
    if (set)
      flags |= bit;
    bit <<= 1;
  }
  return flags;
}

std::int32_t resident_training_reference_slot(const NativeEntityReferenceView &reference, std::int32_t object_limit) {
  if (!reference.validated || !reference.has_value) {
    return -1;
  }
  return reference.slot >= 0 && reference.slot < object_limit ? reference.slot : -2;
}

std::int32_t resident_training_native_id_slot(std::uint32_t native_object_id, void *const *objects,
                                              std::int32_t object_limit) {
  if (native_object_id == 0 || objects == nullptr || object_limit < 0) {
    return -1;
  }
  for (std::int32_t slot = 0; slot < object_limit; ++slot) {
    NativeEntityIdentityView identity;
    if (objects[slot] != nullptr && read_native_entity_identity(objects[slot], &identity) &&
        identity.native_object_id == native_object_id) {
      return slot;
    }
  }
  return -2;
}

bool resident_write_training_combat_fact(ResidentTrainingWriter *writer, const CombatEntityFact &fact) {
  if (writer == nullptr || (fact.present && (!fact.validated || fact.native_object_id == 0)) ||
      (!fact.present && fact.visibility_validated)) {
    return false;
  }
  const std::uint32_t flags =
      (fact.validated ? 1U : 0U) | (fact.present ? 2U : 0U) | (fact.visibility_validated ? 4U : 0U);
  return writer->u32(flags) && writer->u32(fact.native_object_id) && writer->i32(fact.owner) &&
         writer->i32(fact.object_index) && writer->i32(fact.secondary_index) && writer->i32(fact.card_id) &&
         writer->i32(fact.object_kind) && writer->i32(fact.position_x) && writer->i32(fact.position_y) &&
         writer->i32(fact.invisible_count);
}

bool build_resident_training_capture(std::uint8_t *response, std::size_t response_size, std::size_t *payload_size) {
  if (response == nullptr || payload_size == nullptr || response_size < 4096 || g_controlled_manager == nullptr) {
    return false;
  }
  *payload_size = 0;
  void *const manager = g_controlled_manager;
  void *const world = read_object_field<void *>(manager, 0xa8);
  void *const king_tower = world == nullptr ? nullptr : read_object_field<void *>(world, 0xe0);
  void *const object_manager = king_tower == nullptr ? nullptr : read_object_field<void *>(king_tower, 0x10);
  if (world == nullptr || king_tower == nullptr || object_manager == nullptr) {
    return false;
  }

  void **const objects = read_object_field<void **>(object_manager, 0x08);
  const std::int32_t capacity = read_object_field<std::int32_t>(object_manager, 0x10);
  const std::int32_t count = read_object_field<std::int32_t>(object_manager, 0x14);
  if (count < 0 || capacity < count || capacity > 100000 || (count != 0 && objects == nullptr)) {
    return false;
  }
  const std::int32_t object_limit = count < kMaxObservedObjects ? count : kMaxObservedObjects;
  const bool truncated = count > object_limit;
  const std::int32_t tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
  if (tick < 0 || object_limit < 0 || object_limit > UINT16_MAX) {
    return false;
  }

  BattleResultView battle_result;
  if (!read_battle_result(world, &battle_result)) {
    return false;
  }
  void *const command_queue = read_object_field<void *>(manager, 0x38);
  const std::int32_t queued_commands =
      command_queue == nullptr ? -1 : read_object_field<std::int32_t>(command_queue, 0x14);
  const bool ended = battle_result.finalized || g_control_ended;
  const auto capture_flags =
      static_cast<std::uint16_t>(resident_training_flags({ended, battle_result.finalized, truncated}));

  ResidentTrainingWriter writer{
      response,
      response_size,
      0,
      true,
  };

  writer.bytes(kResidentTrainingCaptureMagic, sizeof(kResidentTrainingCaptureMagic));
  writer.values<std::uint16_t>({
      kResidentTrainingCaptureVersion,
      capture_flags,
  });
  writer.values<std::uint64_t>({
      g_control_generation,
      g_state_epoch,
      g_completed_steps,
  });
  writer.values<std::int32_t>({
      tick,
      battle_result.world_result_raw,
      battle_result.crowns_raw[0],
      battle_result.crowns_raw[1],
      queued_commands,
  });
  writer.values<std::uint16_t>({
      kPlayerCount,
      static_cast<std::uint16_t>(object_limit),
  });

  TowerTroopRuntimeRecord tower_troop_runtimes[kMaxStoredTowerTroopRuntimes] = {};
  const std::size_t tower_troop_runtime_count = latest_tower_troop_runtimes(
      g_control_generation, g_state_epoch, tick, tower_troop_runtimes, kMaxStoredTowerTroopRuntimes);

  for (std::int32_t owner = 0; owner < kPlayerCount; ++owner) {
    PlayerStateView player;
    NativePlayerRuntimeView runtime;
    if (!read_player_state(world, owner, &player) ||
        !read_native_player_runtime(world, object_manager, objects, count, owner, &runtime)) {
      return false;
    }
    const std::int32_t raw_hand_count = player.hand.count;
    auto *const hand_slots = static_cast<const std::int32_t *>(player.hand.data);
    std::int32_t hand_count = 0;
    for (std::int32_t hand_index = 0; hand_index < raw_hand_count; ++hand_index) {
      if (hand_slots[hand_index] >= 0) {
        ++hand_count;
      }
    }
    const std::int32_t cycle_count = player.cycle.count;
    const std::int32_t deck_count = player.deck_slots.count;
    const std::int32_t ability_count = runtime.ability_runtime_valid ? runtime.ability_count : 0;
    const std::int32_t evolution_count = runtime.evolution_runtime_valid ? runtime.evolution_slot_count : 0;
    if (raw_hand_count < 0 || raw_hand_count > 4 || hand_count < 0 || hand_count > 4 || cycle_count < 0 ||
        cycle_count > 8 || deck_count < 0 || deck_count > 8 || ability_count < 0 || ability_count > 2 ||
        evolution_count < 0 || evolution_count > 8) {
      return false;
    }
    const std::int32_t owner_root_slot = resident_training_reference_slot(runtime.owner_root, object_limit);
    if (runtime.owner_root.has_value && owner_root_slot < 0)
      return false;
    const auto player_flags = static_cast<std::uint16_t>(resident_training_flags({
        runtime.owner_root_valid,
        runtime.owner_root.has_value,
        runtime.ability_runtime_valid,
        runtime.evolution_runtime_valid,
    }));

    writer.values<std::int32_t>({
        owner,
        player.elixir_raw,
        battle_result.crowns_raw[owner],
    });
    writer.values<std::uint32_t>({
        player.account_id_high,
        player.account_id_low,
    });
    writer.u16(player_flags);
    writer.values<std::uint8_t>({
        static_cast<std::uint8_t>(hand_count),
        static_cast<std::uint8_t>(cycle_count),
        static_cast<std::uint8_t>(deck_count),
        static_cast<std::uint8_t>(ability_count),
        static_cast<std::uint8_t>(evolution_count),
        0,
    });
    writer.i32(owner_root_slot);

    for (std::int32_t hand_index = 0; hand_index < raw_hand_count; ++hand_index) {
      const std::int32_t deck_slot = hand_slots[hand_index];
      if (deck_slot < 0) {
        continue;
      }
      CardStateView card;
      const bool selection_valid = read_card_state(player, deck_slot, true, &card);
      if (!selection_valid && !read_card_state(player, deck_slot, false, &card)) {
        return false;
      }

      writer.values<std::int32_t>({
          hand_index,
          card.deck_slot,
          card.card_id,
          selection_valid ? card.command_card_id : card.card_id,
      });
      writer.u32(selection_valid ? card.card_parameter : UINT32_MAX);
      writer.i32(selection_valid ? card.cost : -1);
    }

    auto *const cycle_slots = static_cast<const std::int32_t *>(player.cycle.data);
    for (std::int32_t cycle_index = 0; cycle_index < cycle_count; ++cycle_index) {
      CardStateView card;

      if (!read_card_state(player, cycle_slots[cycle_index], false, &card)) {
        return false;
      }
      writer.values<std::int32_t>({
          card.deck_slot,
          card.card_id,
      });
    }
    for (std::int32_t deck_slot = 0; deck_slot < deck_count; ++deck_slot) {
      CardStateView card;

      if (!read_card_state(player, deck_slot, false, &card)) {
        return false;
      }
      writer.values<std::int32_t>({
          card.deck_slot,
          card.card_id,
      });
    }

    for (std::int32_t index = 0; index < ability_count; ++index) {
      const NativeChampionControllerView &ability = runtime.abilities[index];

      if (ability.action_data_name_size == 0 || ability.action_data_name_size > 128 || ability.champion_count < 0 ||
          ability.champion_count > kMaxObservedObjects) {
        return false;
      }
      writer.values<std::int32_t>({
          ability.controller_slot,
          ability.action_data_global_id,
          ability.selected_character_global_id,
          ability.remaining_cooldown_ms,
          ability.configured_cooldown_ms,
          ability.remaining_charges_raw,
          ability.max_charges,
          ability.button_state,
      });
      writer.values<std::uint16_t>({
          static_cast<std::uint16_t>(ability.action_data_name_size),
          static_cast<std::uint16_t>(ability.champion_count),
      });
      writer.bytes(ability.action_data_name, ability.action_data_name_size);
      for (std::int32_t champion_index = 0; champion_index < ability.champion_count; ++champion_index) {
        const std::int32_t slot = resident_training_reference_slot(ability.champions[champion_index], object_limit);

        if (slot < 0) {
          return false;
        }
        writer.i32(slot);
      }
    }

    for (std::int32_t index = 0; index < evolution_count; ++index) {
      const NativeEvolutionSlotView &evolution = runtime.evolution_slots[index];

      writer.values<std::int32_t>({
          evolution.deck_slot,
          evolution.card_id,
          evolution.base_spell_global_id,
          evolution.evolvable ? 1 : 0,
          evolution.evolvable ? evolution.evolution_form_global_id : kResidentTrainingNoneI32,
          evolution.progress,
          evolution.evolvable ? evolution.cycle_required : kResidentTrainingNoneI32,
          evolution.evolvable ? evolution.cycle_remaining : kResidentTrainingNoneI32,
          evolution.evolvable ? (evolution.ready ? 1 : 0) : kResidentTrainingNoneI32,
      });
    }
  }

  for (std::int32_t slot = 0; slot < object_limit; ++slot) {
    const void *const object = objects[slot];
    if (object == nullptr) {
      writer.u32(1U << 0U);
      writer.i32(slot);
      writer.u32(0);
      for (std::size_t index = 0; index < 15; ++index) {
        writer.i32(kResidentTrainingNoneI32);
      }
      continue;
    }

    NativeEntityIdentityView identity;
    if (!read_native_entity_identity(object, &identity) ||
        !native_object_id_is_unique_in_vector(object, objects, count, identity.native_object_id)) {
      return false;
    }
    NativeComponentInventoryView inventory;
    read_native_component_inventory(object, &inventory);
    NativeTargetEntityView target;
    resolve_native_target_entity(object, inventory, objects, count, &target);
    if (target.has_target && (target.object_slot < 0 || target.object_slot >= object_limit)) {
      return false;
    }
    NativeProjectileStateView projectile;
    const NativeProjectileReadStatus projectile_status =
        read_native_projectile_state(object, objects, count, &projectile);
    if (projectile_status == NativeProjectileReadStatus::Invalid) {
      return false;
    }
    if (projectile_status == NativeProjectileReadStatus::Valid) {
      const NativeEntityReferenceView references[] = {
          projectile.source,
          projectile.target,
          projectile.homing_target,
      };
      for (const NativeEntityReferenceView &reference : references) {
        if (reference.has_value && resident_training_reference_slot(reference, object_limit) < 0) {
          return false;
        }
      }
    }

    const bool type_two_valid = inventory.valid && inventory.count > 2 && inventory.slots[2].present &&
                                inventory.slots[2].owner_backlink_valid && inventory.slots[2].type_valid &&
                                inventory.slots[2].engine_type == 2;
    std::int32_t hitpoints = 0;
    std::int32_t maximum_hitpoints = 0;
    const bool has_hitpoints = type_two_valid && read_hitpoints(object, &hitpoints, &maximum_hitpoints);
    std::int32_t shield = 0;
    std::int32_t maximum_shield = 0;
    const bool has_shield = type_two_valid && read_shield(object, &shield, &maximum_shield);
    std::int32_t attack_stage = 0;
    const bool has_attack_stage = read_attack_sequence_stage(object, inventory, &attack_stage);
    NativeTypeThreeStateView effects;
    read_native_type_three_state(object, inventory, objects, count, &effects);
    NativePhaseSnapshot phase;
    const bool has_phase = read_native_phase_snapshot(object, g_control_generation, g_state_epoch, tick, &phase);
    const void *const object_data = read_object_field<const void *>(object, cr_remaining_runtime::kObjectDataOffset);
    std::int32_t signed_data_global_id = 0;
    if (!remaining_data_global_id(object_data, &signed_data_global_id)) {
      return false;
    }
    const std::uint32_t data_global_id = static_cast<std::uint32_t>(signed_data_global_id);
    const TowerTroopRuntimeRecord *tower_troop_runtime = nullptr;
    for (std::size_t index = 0; index < tower_troop_runtime_count; ++index) {
      if (tower_troop_runtimes[index].source_native_object_id == identity.native_object_id) {
        tower_troop_runtime = &tower_troop_runtimes[index];
        break;
      }
    }

    const auto object_flags = resident_training_flags({
        false, // Bit zero is reserved for null object slots.
        has_hitpoints,
        has_shield,
        target.validated,
        target.has_target,
        projectile_status == NativeProjectileReadStatus::Valid,
        has_phase,
        effects.active_effects_valid,
        effects.header_valid,
        true, // The data global ID above passed native validation.
        tower_troop_runtime != nullptr,
    });

    writer.u32(object_flags);
    writer.i32(slot);
    writer.u32(identity.native_object_id);
    writer.values<std::int32_t>({
        identity.owner,
        read_object_field<std::int32_t>(object, 0xac),
        read_object_field<std::int32_t>(object, 0x7c),
        read_object_field<std::int32_t>(object, 0x80),
        read_object_field<std::int32_t>(object, 0x84),
        read_object_field<std::int32_t>(object, 0x88),
        identity.object_index,
        identity.secondary_index,
        has_hitpoints ? hitpoints : kResidentTrainingNoneI32,
        has_hitpoints ? maximum_hitpoints : kResidentTrainingNoneI32,
        has_shield ? shield : kResidentTrainingNoneI32,
        has_shield ? maximum_shield : kResidentTrainingNoneI32,
        has_attack_stage ? attack_stage : kResidentTrainingNoneI32,
        effects.header_valid ? effects.invisible_count : kResidentTrainingNoneI32,
        target.has_target ? target.object_slot : -1,
    });
    writer.u32(data_global_id);

    if (projectile_status == NativeProjectileReadStatus::Valid) {
      const std::int32_t source_slot = resident_training_reference_slot(projectile.source, object_limit);
      const std::int32_t target_slot = resident_training_reference_slot(projectile.target, object_limit);
      const std::int32_t homing_slot = resident_training_reference_slot(projectile.homing_target, object_limit);
      const std::int32_t validation_mask = (projectile.source.validated ? 1 : 0) |
                                           (projectile.target.validated ? 2 : 0) |
                                           (projectile.homing_target.validated ? 4 : 0);

      writer.u32(projectile.data_global_id);
      writer.values<std::int32_t>({
          source_slot,
          target_slot,
          homing_slot,
          projectile.destination_x,
          projectile.destination_y,
          validation_mask,
          projectile.terminal ? 1 : 0,
      });
    }

    if (has_phase) {
      const std::uint32_t phase_flags =
          (phase.attack_validated ? 1U : 0U) | (phase.movement_validated ? 2U : 0U) | (phase.buffs_validated ? 4U : 0U);
      const std::int32_t values[] = {
          phase.attack_sequence_stage,
          phase.attack_timeline_ms,
          phase.load_remaining_ms,
          phase.deploy_remaining_ms,
          phase.deploy_previous_ms,
          phase.configured_deploy_time_ms,
          phase.hit_speed_ms,
          phase.attack_dash_time_ms,
          phase.base_movement_speed,
          phase.charge_speed_multiplier,
          phase.classic_charge_progress,
          phase.speed_positive_percent,
          phase.speed_negative_magnitude,
          phase.hit_speed_positive_percent,
          phase.hit_speed_negative_magnitude,
          phase.attack_step_input,
          phase.attack_step_output,
          phase.attack_step_tick,
          phase.movement_step_input,
          phase.movement_step_output,
          phase.movement_step_tick,
          phase.deploy_step_input,
          phase.deploy_step_output,
          phase.deploy_step_tick,
          phase.effective_movement_speed,
          phase.effective_movement_speed_tick,
          phase.movement_delta,
          phase.movement_delta_tick,
      };

      writer.u32(phase_flags);
      for (const std::int32_t value : values) {
        writer.i32(value);
      }
    }

    if (tower_troop_runtime != nullptr) {
      const TowerTroopRuntimeRecord &runtime = *tower_troop_runtime;

      writer.values<std::uint8_t>({
          static_cast<std::uint8_t>(runtime.kind),
          0,
          0,
          0,
      });
      writer.values<std::int32_t>({
          runtime.tick,
          runtime.charge_count,
          runtime.max_charge_count,
          runtime.recharge_elapsed_ms,
          runtime.recharge_duration_ms,
          runtime.start_delay_remaining_ms,
          runtime.start_delay_duration_ms,
          runtime.cooking_contribution,
          runtime.contribution_needed,
          runtime.throw_delay_remaining_ms,
      });
      writer.u32(runtime.target_native_object_id);
    }

    const std::int32_t effect_count = effects.active_effects_valid ? effects.count : 0;

    if (effect_count < 0 || effect_count > kMaxActiveEffects) {
      return false;
    }
    writer.u16(static_cast<std::uint16_t>(effect_count));
    for (std::int32_t index = 0; index < effect_count; ++index) {
      const NativeActiveEffectView &effect = effects.effects[index];
      std::int32_t source_slot = -1;
      if (effect.source_entity_resolved) {
        source_slot = resident_training_native_id_slot(effect.source_native_object_id, objects, object_limit);
        if (source_slot < 0) {
          return false;
        }
      }

      if (effect.name_size == 0 || effect.name_size > 128) {
        return false;
      }
      writer.u32(effect.buff_global_id);
      writer.values<std::int32_t>({
          effect.remaining_ms,
          source_slot,
      });
      writer.u8(effect.source_entity_validated ? 1 : 0);
      writer.u16(static_cast<std::uint16_t>(effect.name_size));
      writer.bytes(effect.name, effect.name_size);
    }
  }

  const bool combat_attested = g_combat_hooks_attested.load(std::memory_order_acquire);
  const bool combat_installed = g_combat_hooks_installed.load(std::memory_order_acquire);
  const bool combat_identity = combat_installed && g_combat_epoch_active.load(std::memory_order_acquire) &&
                               g_combat_game_manager.load(std::memory_order_acquire) == manager &&
                               g_combat_generation.load(std::memory_order_acquire) == g_control_generation &&
                               g_combat_state_epoch.load(std::memory_order_acquire) == g_state_epoch;
  const std::uint64_t combat_next = g_combat_next_sequence.load(std::memory_order_acquire);
  const std::uint64_t combat_epoch_first =
      combat_identity ? g_combat_epoch_first_sequence.load(std::memory_order_acquire) : combat_next;
  if (combat_next < combat_epoch_first) {
    return false;
  }
  const std::uint64_t combat_oldest = combat_next > combat_epoch_first + kCombatEventRingCapacity
                                          ? combat_next - kCombatEventRingCapacity
                                          : combat_epoch_first;
  const std::uint64_t combat_overflow = combat_identity ? g_combat_overflow_count.load(std::memory_order_acquire) : 0;
  const std::uint64_t expected_combat_overflow = combat_next - combat_epoch_first > kCombatEventRingCapacity
                                                     ? combat_next - combat_epoch_first - kCombatEventRingCapacity
                                                     : 0;
  const std::uint64_t combat_rejected =
      combat_identity ? g_combat_rejected_capture_count.load(std::memory_order_acquire) : 0;
  const std::uint64_t combat_event_count = combat_next - combat_oldest;
  const bool combat_complete = combat_identity && combat_attested && combat_installed && combat_rejected == 0 &&
                               combat_overflow == expected_combat_overflow &&
                               combat_event_count <= kCombatEventRingCapacity;
  if (!combat_complete || combat_event_count > UINT16_MAX) {
    return false;
  }
  const std::uint32_t combat_flags =
      (combat_attested ? 1U : 0U) | (combat_installed ? 2U : 0U) | (combat_complete ? 4U : 0U);

  writer.u32(combat_flags);
  writer.values<std::uint64_t>({
      combat_epoch_first,
      combat_oldest,
      combat_next,
      combat_overflow,
      combat_rejected,
  });
  writer.u16(static_cast<std::uint16_t>(combat_event_count));
  for (std::uint64_t sequence = combat_oldest; sequence < combat_next; ++sequence) {
    CombatRingSlot &slot = g_combat_event_ring[(sequence - 1) % kCombatEventRingCapacity];
    if (slot.published_sequence.load(std::memory_order_acquire) != sequence) {
      return false;
    }
    const CombatRawRecord record = slot.record;
    if (slot.published_sequence.load(std::memory_order_acquire) != sequence || record.sequence != sequence ||
        record.generation != g_control_generation || record.state_epoch != g_state_epoch || record.tick > tick ||
        record.hook_offset == 0) {
      return false;
    }
    if (record.deployment.validated &&
        (record.deployment.generation != g_control_generation || record.deployment.state_epoch != g_state_epoch ||
         record.deployment.deployment_sequence == 0 || record.deployment.owner < 0 ||
         record.deployment.owner >= kPlayerCount || record.deployment.played_card_global_id == 0 ||
         record.deployment.deck_slot < 0 || record.deployment.deck_slot >= kCombatMaxDeckSlots ||
         record.deployment.cost < 0 || record.deployment.cost > 15 || record.deployment.form_code < 0 ||
         record.deployment.form_code > 5)) {
      return false;
    }
    const std::uint32_t event_flags =
        (record.lethal ? 1U : 0U) | (record.caller_offset != 0 ? 2U : 0U) | (record.cause_sequence != 0 ? 4U : 0U);
    if (!writer.u64(record.sequence) || !writer.i32(record.tick) ||
        !writer.u8(static_cast<std::uint8_t>(record.kind)) || !writer.u8(static_cast<std::uint8_t>(record.pool)) ||
        !writer.u8(static_cast<std::uint8_t>(record.terminal_reason)) ||
        !writer.u8(static_cast<std::uint8_t>(event_flags)) || !writer.u16(0) || !writer.u64(record.hook_offset) ||
        !writer.u64(record.caller_offset) || !writer.u64(record.cause_sequence) ||
        !writer.i32(record.requested_amount) || !writer.i32(record.actual_amount) || !writer.i32(record.pre_hp) ||
        !writer.i32(record.post_hp) || !writer.i32(record.pre_builtin_shield) ||
        !writer.i32(record.post_builtin_shield) || !writer.i32(record.pre_buff_shield) ||
        !writer.i32(record.post_buff_shield) || !writer.u8(record.deployment.validated ? 1 : 0) || !writer.u8(0) ||
        !writer.u8(0) || !writer.u8(0) || !writer.u64(record.deployment.deployment_sequence) ||
        !writer.i32(record.deployment.owner) || !writer.u32(record.deployment.played_card_global_id) ||
        !writer.u32(record.deployment.effective_card_global_id) || !writer.u32(record.deployment.card_parameter) ||
        !writer.i32(record.deployment.deck_slot) || !writer.i32(record.deployment.cost) ||
        !writer.i32(record.deployment.form_code) || !resident_write_training_combat_fact(&writer, record.target) ||
        !resident_write_training_combat_fact(&writer, record.immediate_source) ||
        !resident_write_training_combat_fact(&writer, record.source) ||
        !resident_write_training_combat_fact(&writer, record.projectile)) {
      return false;
    }
  }

  const bool phase_identity = g_phase_runtime_generation.load(std::memory_order_acquire) == g_control_generation &&
                              g_phase_runtime_state_epoch.load(std::memory_order_acquire) == g_state_epoch;
  const std::uint64_t phase_next = g_phase_runtime_next_sequence.load(std::memory_order_acquire);
  const std::uint64_t phase_rejected =
      phase_identity ? g_phase_runtime_rejected_count.load(std::memory_order_acquire) : 0;
  const bool phase_attested = g_phase_runtime_hooks_attested.load(std::memory_order_acquire);
  const bool phase_installed = g_phase_runtime_hooks_installed.load(std::memory_order_acquire);
  const bool phase_complete = phase_identity && phase_installed && phase_rejected == 0;
  if (!phase_identity || !phase_attested || !phase_installed || !phase_complete) {
    return false;
  }
  const std::uint32_t phase_flags =
      (phase_attested ? 1U : 0U) | (phase_installed ? 2U : 0U) | (phase_complete ? 4U : 0U);

  writer.u32(phase_flags);
  writer.values<std::uint64_t>({
      phase_next,
      phase_next,
      phase_next,
      0,
      phase_rejected,
  });
  writer.u16(0);
  const bool tower_identity =
      g_tower_troop_runtime_generation.load(std::memory_order_acquire) == g_control_generation &&
      g_tower_troop_runtime_state_epoch.load(std::memory_order_acquire) == g_state_epoch;
  const bool tower_attested = g_tower_troop_runtime_hooks_attested.load(std::memory_order_acquire);
  const bool tower_installed = g_tower_troop_runtime_hooks_installed.load(std::memory_order_acquire);
  const std::uint64_t tower_rejected =
      tower_identity ? g_tower_troop_runtime_rejected_count.load(std::memory_order_acquire) : 0;
  const bool tower_complete = tower_identity && tower_attested && tower_installed && tower_rejected == 0;

  if (!tower_complete) {
    return false;
  }
  writer.u32((tower_attested ? 1U : 0U) | (tower_installed ? 2U : 0U) | (tower_complete ? 4U : 0U));
  writer.u64(tower_rejected);
  if (!writer.valid) {
    return false;
  }
  *payload_size = writer.used;
  return true;
}

void resident_write_receipt(std::uint8_t *destination, const ResidentBatchReceipt &receipt) {
  destination[0] = receipt.kind;
  destination[1] = receipt.owner;
  resident_write_le<std::uint16_t>(destination + 2, receipt.action_index);
  resident_write_le<std::uint16_t>(destination + 4, receipt.status);
  resident_write_le<std::uint16_t>(destination + 6, 0);
  resident_write_le<std::uint32_t>(destination + 8, static_cast<std::uint32_t>(receipt.execute_tick));
  resident_write_le<std::uint32_t>(destination + 12, receipt.card_id);
  resident_write_le<std::uint32_t>(destination + 16, receipt.card_parameter);
  resident_write_le<std::uint32_t>(destination + 20, static_cast<std::uint32_t>(receipt.deck_slot));
  resident_write_le<std::uint32_t>(destination + 24, static_cast<std::uint32_t>(receipt.cost));
  resident_write_le<std::uint32_t>(destination + 28, static_cast<std::uint32_t>(receipt.form_code));
  resident_write_le<std::uint32_t>(destination + 32, static_cast<std::uint32_t>(receipt.argument0));
  resident_write_le<std::uint32_t>(destination + 36, static_cast<std::uint32_t>(receipt.argument1));
}

bool resident_receive_exact(int socket_fd, std::uint8_t *destination, std::size_t size) {
  std::size_t received = 0;
  while (received < size) {
    const ssize_t count = recv(socket_fd, destination + received, size - received, 0);
    if (count == 0) {
      return false;
    }
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    received += static_cast<std::size_t>(count);
  }
  return true;
}

bool resident_send_batch_handshake(int socket_fd, std::uint16_t status) {
  std::uint8_t response[kResidentBatchHandshakeBytes] = {};
  std::memcpy(response, kResidentBatchHandshakeMagic, 4);
  resident_write_le<std::uint16_t>(response + 4, kResidentBatchVersion);
  resident_write_le<std::uint16_t>(response + 6, status);
  resident_write_le<std::uint16_t>(response + 8, static_cast<std::uint16_t>(kResidentMaxSlots));
  return send_all(socket_fd, reinterpret_cast<const char *>(response), sizeof(response));
}

bool resident_send_batch_response_header(int socket_fd, std::uint16_t status, std::uint64_t sequence,
                                         std::uint32_t count) {
  std::uint8_t response[kResidentBatchHeaderBytes] = {};
  std::memcpy(response, kResidentBatchResponseMagic, 4);
  resident_write_le<std::uint16_t>(response + 4, kResidentBatchVersion);
  resident_write_le<std::uint16_t>(response + 6, status);
  resident_write_le<std::uint64_t>(response + 8, sequence);
  resident_write_le<std::uint32_t>(response + 16, count);
  return send_all(socket_fd, reinterpret_cast<const char *>(response), sizeof(response));
}

bool resident_send_batch_entry(int socket_fd, std::uint16_t slot_id, std::uint16_t status,
                               const ResidentBatchReceipt *receipts, std::uint16_t receipt_count, const char *payload,
                               std::size_t payload_size, std::uint16_t flags = 0) {
  if (payload == nullptr || payload_size > UINT32_MAX || receipt_count > kMaxTransitionActions ||
      (receipt_count != 0 && receipts == nullptr)) {
    return false;
  }
  std::uint8_t header[kResidentBatchResponseEntryBytes] = {};
  resident_write_le<std::uint16_t>(header, slot_id);
  resident_write_le<std::uint16_t>(header + 2, status);
  resident_write_le<std::uint16_t>(header + 4, receipt_count);
  resident_write_le<std::uint16_t>(header + 6, flags);
  resident_write_le<std::uint32_t>(header + 8, static_cast<std::uint32_t>(payload_size));
  std::uint8_t encoded_receipts[kMaxTransitionActions * kResidentBatchReceiptBytes] = {};
  for (std::uint16_t index = 0; index < receipt_count; ++index) {
    resident_write_receipt(encoded_receipts + index * kResidentBatchReceiptBytes, receipts[index]);
  }
  const std::size_t receipt_bytes = static_cast<std::size_t>(receipt_count) * kResidentBatchReceiptBytes;

  return send_all(socket_fd, reinterpret_cast<const char *>(header), sizeof(header)) &&
         send_all(socket_fd, reinterpret_cast<const char *>(encoded_receipts), receipt_bytes) &&
         send_all(socket_fd, payload, payload_size);
}

void handle_resident_batch_session(int socket_fd, bool training_capture = false, bool compress_json = false) {
  const std::uint16_t handshake_status = g_resident_mode_active ? 0 : 1;
  if (!resident_send_batch_handshake(socket_fd, handshake_status) || handshake_status != 0) {
    return;
  }

  // The batch loop is a long-lived training data plane.  Reusing one scratch
  // buffer avoids allocating and zero-filling 12 MiB for every five-tick
  // frame.  Compact captures normally use only a few KiB; the full JSON
  // buffer is allocated lazily only if the compact writer fails closed.
  const std::size_t primary_payload_capacity =
      training_capture ? kResidentTrainingCaptureBytes : kFullObservationResponseBytes;
  std::unique_ptr<char[]> primary_payload(new (std::nothrow) char[primary_payload_capacity]);
  if (primary_payload == nullptr) {
    return;
  }
  std::unique_ptr<char[]> json_fallback_payload;
  const std::size_t compressed_payload_capacity =
      compress_json ? static_cast<std::size_t>(compressBound(static_cast<uLong>(kFullObservationResponseBytes))) : 0;
  std::unique_ptr<std::uint8_t[]> compressed_payload(
      compress_json ? new (std::nothrow) std::uint8_t[compressed_payload_capacity] : nullptr);
  if (compress_json && compressed_payload == nullptr) {
    return;
  }

  for (;;) {
    std::uint8_t header[kResidentBatchHeaderBytes] = {};
    if (!resident_receive_exact(socket_fd, header, sizeof(header))) {
      return;
    }
    if (std::memcmp(header, kResidentBatchRequestMagic, 4) != 0 ||
        resident_read_le<std::uint16_t>(header + 4) != kResidentBatchVersion ||
        resident_read_le<std::uint16_t>(header + 6) != 0) {
      resident_send_batch_response_header(socket_fd, 1, 0, 0);
      return;
    }
    const std::uint64_t sequence = resident_read_le<std::uint64_t>(header + 8);
    const std::uint32_t count = resident_read_le<std::uint32_t>(header + 16);
    if (count == 0 || count > kResidentMaxSlots) {
      resident_send_batch_response_header(socket_fd, 2, sequence, 0);
      return;
    }

    std::uint16_t slot_ids[kResidentMaxSlots] = {};
    std::uint16_t action_counts[kResidentMaxSlots] = {};
    std::uint32_t ticks[kResidentMaxSlots] = {};
    ResidentBatchAction actions[kResidentMaxSlots][kMaxTransitionActions] = {};
    bool seen[kResidentMaxSlots] = {};
    bool valid = true;
    for (std::uint32_t index = 0; index < count; ++index) {
      std::uint8_t entry[kResidentBatchRequestEntryBytes] = {};
      if (!resident_receive_exact(socket_fd, entry, sizeof(entry))) {
        return;
      }
      slot_ids[index] = resident_read_le<std::uint16_t>(entry);
      action_counts[index] = resident_read_le<std::uint16_t>(entry + 2);
      ticks[index] = resident_read_le<std::uint32_t>(entry + 4);
      const bool slot_valid = resident_slot_id_valid(slot_ids[index]) && g_resident_slots[slot_ids[index]].occupied &&
                              !seen[slot_ids[index]];
      if (!slot_valid || action_counts[index] > kMaxTransitionActions || ticks[index] == 0 || ticks[index] > 1000000U) {
        valid = false;
        break;
      }
      seen[slot_ids[index]] = true;
      for (std::uint16_t action_index = 0; action_index < action_counts[index]; ++action_index) {
        std::uint8_t raw_action[kResidentBatchActionBytes] = {};
        if (!resident_receive_exact(socket_fd, raw_action, sizeof(raw_action))) {
          return;
        }
        ResidentBatchAction &action = actions[index][action_index];
        action.kind = raw_action[0];
        action.owner = raw_action[1];
        action.execute_offset = resident_read_le<std::uint16_t>(raw_action + 2);
        action.argument0 = resident_read_le<std::int32_t>(raw_action + 4);
        action.argument1 = resident_read_le<std::int32_t>(raw_action + 8);
        action.argument2 = resident_read_le<std::int32_t>(raw_action + 12);
        const std::uint32_t reserved = resident_read_le<std::uint32_t>(raw_action + 16);
        const bool play_valid = action.kind == kResidentBatchActionPlay && action.argument0 >= 0;
        const bool ability_valid = action.kind == kResidentBatchActionAbility && action.argument0 >= -2 &&
                                   action.argument1 >= 0 && action.argument2 == 0;
        if (action.owner >= kPlayerCount || action.execute_offset == 0 || reserved != 0 ||
            (!play_valid && !ability_valid)) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        break;
      }
    }
    if (!valid) {
      resident_send_batch_response_header(socket_fd, 3, sequence, 0);
      return;
    }

    if (!resident_send_batch_response_header(socket_fd, 0, sequence, count)) {
      return;
    }
    for (std::uint32_t index = 0; index < count; ++index) {
      const std::uint16_t slot_id = slot_ids[index];
      ResidentMatchSlot &slot = g_resident_slots[slot_id];
      ResidentBatchReceipt receipts[kMaxTransitionActions] = {};
      bool succeeded = resident_bind_slot(slot_id);
      const std::int32_t current_tick = succeeded ? g_last_controlled_tick : -1;
      if (succeeded && current_tick < 0) {
        succeeded = false;
      }
      for (std::uint16_t action_index = 0; succeeded && action_index < action_counts[index]; ++action_index) {
        const ResidentBatchAction &action = actions[index][action_index];
        ResidentBatchReceipt &receipt = receipts[action_index];
        receipt.kind = action.kind;
        receipt.owner = action.owner;
        receipt.action_index = action_index;
        receipt.argument0 = action.argument0;
        receipt.argument1 = action.argument1;
        receipt.execute_tick = current_tick + action.execute_offset;
        char command_json[768] = {};
        if (action.kind == kResidentBatchActionPlay) {
          DeployRequest request;
          succeeded = resolve_hand_deploy_request(slot.manager, action.owner, action.argument0, action.argument1,
                                                  action.argument2, &request);
          const std::int32_t command_age_boundary = receipt.execute_tick - 1;
          const std::int32_t command_tick = command_age_boundary - static_cast<std::int32_t>(kLiveCommandAgeTicks);
          succeeded = succeeded && format_deploy_command(request, command_tick, command_json, sizeof(command_json)) &&
                      inject_logic_command(slot.manager, command_json);
          receipt.card_id = static_cast<std::uint32_t>(request.root_card_id);
          receipt.card_parameter = request.card_parameter;
          receipt.deck_slot = request.deck_slot;
          receipt.cost = request.cost;
          receipt.form_code = request.form_code;
        } else {
          AbilityRequest request;
          succeeded =
              resolve_ready_ability_request(slot.manager, action.owner, action.argument0, action.argument1, &request) &&
              format_ability_command(request, receipt.execute_tick - 2, command_json, sizeof(command_json)) &&
              inject_logic_command(slot.manager, command_json);
        }
        receipt.status = succeeded ? 0 : 1;
      }
      if (succeeded) {
        succeeded = resident_step_bound(slot, ticks[index]);
      }
      char *payload = primary_payload.get();
      std::size_t payload_capacity = primary_payload_capacity;
      payload[0] = '\0';

      std::size_t payload_size = 0;
      if (succeeded) {
        if (training_capture) {
          succeeded = build_resident_training_capture(reinterpret_cast<std::uint8_t *>(payload),
                                                      kResidentTrainingCaptureBytes, &payload_size);
          if (!succeeded) {
            if (json_fallback_payload == nullptr) {
              json_fallback_payload.reset(new (std::nothrow) char[kFullObservationResponseBytes]);
            }
            if (json_fallback_payload == nullptr) {
              return;
            }
            payload = json_fallback_payload.get();
            payload_capacity = kFullObservationResponseBytes;
            payload[0] = '\0';
            succeeded = build_atomic_observation_json_locked(payload, payload_capacity);
            payload_size = std::strlen(payload);
          }
        } else {
          succeeded = build_atomic_observation_json_locked(payload, payload_capacity);
          payload_size = std::strlen(payload);
        }
      }
      if (!succeeded && (training_capture || payload[0] == '\0')) {
        std::snprintf(payload, payload_capacity,
                      "{\"ok\":false,\"envId\":%u,"
                      "\"error\":\"resident batch step or atomic capture failed\"}",
                      static_cast<unsigned int>(slot_id));
        payload_size = std::strlen(payload);
      }

      const char *response_payload = payload;
      std::size_t response_payload_size = payload_size;
      std::uint16_t response_flags = 0;
      if (compress_json && payload_size != 0) {
        uLongf compressed_size = static_cast<uLongf>(compressed_payload_capacity);
        const int compressed =
            compress2(reinterpret_cast<Bytef *>(compressed_payload.get()), &compressed_size,
                      reinterpret_cast<const Bytef *>(payload), static_cast<uLong>(payload_size), Z_BEST_SPEED);
        if (compressed == Z_OK && compressed_size < payload_size) {
          response_payload = reinterpret_cast<const char *>(compressed_payload.get());
          response_payload_size = static_cast<std::size_t>(compressed_size);
          response_flags = 1;
        }
      }

      if (!resident_send_batch_entry(socket_fd, slot_id, succeeded ? 0 : 1, receipts, action_counts[index],
                                     response_payload, response_payload_size, response_flags)) {
        return;
      }
      // A response header commits this frame to exactly `count`
      // entries.  Keep the wire frame complete when one slot fails so
      // the host can reject only that match and continue the surviving
      // slots on the same synchronized session.
    }
  }
}

bool parse_resident_env_prefix(const char *command, std::int32_t *slot_id_out, const char **nested_command_out) {
  if (command == nullptr || slot_id_out == nullptr || nested_command_out == nullptr ||
      std::strncmp(command, "env ", 4) != 0) {
    return false;
  }
  int slot_id = -1;
  int consumed = 0;
  if (std::sscanf(command + 4, "%d %n", &slot_id, &consumed) != 1 || consumed <= 0) {
    return false;
  }
  const char *nested = command + 4 + consumed;
  while (*nested == ' ' || *nested == '\t') {
    ++nested;
  }
  if (*nested == '\0') {
    return false;
  }
  *slot_id_out = slot_id;
  *nested_command_out = nested;
  return true;
}
