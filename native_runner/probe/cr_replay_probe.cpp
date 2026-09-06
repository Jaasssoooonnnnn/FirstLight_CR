#define _GNU_SOURCE 1

#include <android/log.h>
#include <arpa/inet.h>
#include <cerrno>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <jni.h>
#include <link.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

#include "native_touch_interceptor.h"
#include "probe_json.h"

namespace {

constexpr const char *kLogTag = "CRNativeProbe";
constexpr std::uint16_t kControlPort = 26789;
constexpr std::size_t kEngineInstanceMaxCount = 24;
constexpr std::size_t kEngineInstanceSlots = 8;
constexpr const char *kAttestationVersion = "runner-attestation.v1";
constexpr const char *kProtocolVersion = "cr-native-control.v1";
constexpr const char *kPackageName = "nullsroyale.rel.free";
constexpr const char *kAbi = "arm64-v8a";
char g_user_fingerprint_path[256] = {};
char g_user_assets_path[256] = {};
const char *g_fingerprint_paths[] = {
    g_user_fingerprint_path,
    "/data/data/nullsroyale.rel.free/update/fingerprint.json",
};
const char *g_assets_paths[] = {
    g_user_assets_path,
    "/data/data/nullsroyale.rel.free/update/assets.scdb",
};
constexpr std::uintptr_t kReplayJsonSetterOffset = 0x00c8de8c;
constexpr std::uintptr_t kReplayManagerGetterOffset = 0x00c8c5b8;
constexpr std::uintptr_t kLibgStringFromUtf8Offset = 0x0138a800;
constexpr std::uintptr_t kNativeReplayRunnerOffset = 0x00c87ecc;
constexpr std::uintptr_t kReplayBattleControllerLifecycleOffset = 0x00c87620;
constexpr std::uintptr_t kReplayBattleControllerFullUpdateOffset = 0x00c88bbc;
constexpr std::uintptr_t kReplayBattleControllerClockOffset = 0x00c890bc;
constexpr std::uintptr_t kReplayBattleControllerAlternateUpdateOffset = 0x00c899e0;
constexpr std::uintptr_t kReplayClockGlobalPauseGuardOffset = 0x00cb6694;
constexpr std::uintptr_t kGameAppGetterOffset = 0x00755818;
constexpr std::uintptr_t kGameAppPauseGuardOffset = 0x0075c628;
constexpr std::uintptr_t kGameAppPendingEventSetterOffset = 0x0075a6dc;
constexpr std::size_t kGameAppPendingEventFlagsOffset = 0x1b8;
constexpr std::size_t kGameAppPendingEventOffset = 0x1bc;
constexpr std::size_t kReplayBattleControllerStateOffset = 0x30;
constexpr std::size_t kReplayBattleControllerAccumulatorOffset = 0x44;
constexpr std::size_t kReplayBattleControllerSpeedOffset = 0x74;
constexpr std::size_t kReplayBattleControllerSuspendedOffset = 0x18c;
constexpr std::uintptr_t kGameStateLoadOffset = 0x01061b1c;
constexpr std::uintptr_t kGameStateStepOffset = 0x010620a4;
constexpr std::uintptr_t kGameStateTickOffset = 0x01062bc0;
constexpr std::uintptr_t kGameStateContextAOffset = 0x0080031c;
constexpr std::uintptr_t kGameStateContextBOffset = 0x00eaf1c8;
constexpr std::uintptr_t kGameStateNewOffset = 0x017e4b20;
constexpr std::uintptr_t kGameStateDeleteOffset = 0x017e4b50;
constexpr std::uintptr_t kGameStateConstructorOffset = 0x01060ba8;
constexpr std::uintptr_t kGameStateSetupOffset = 0x01060f84;
constexpr std::uintptr_t kGameStateDestroyOffset = 0x010610b0;
constexpr std::uintptr_t kGameStateBaseDestroyOffset = 0x00988844;
constexpr std::uintptr_t kJsonParseOffset = 0x0121a648;
constexpr std::uintptr_t kLogicCommandParseOffset = 0x00d2fe78;
constexpr std::uintptr_t kLogicCommandAppendOffset = 0x00d2eef0;
constexpr std::uintptr_t kLogicCommandQueueConstructorOffset = 0x00d2eddc;
constexpr std::uintptr_t kLogicCommandQueueDestroyOffset = 0x00d2ee40;
// The stock UI resolves the active ReplayBattleController through c85c28 and
// finishes every player command through c893d0.  That second function owns the
// validation, command tick stamping and ClientInput network-message enqueue.
// Calling it on the game thread preserves the stock outbound authority while
// bypassing only GameApp::nOnTouchEvent and its multi-tick gesture state.
constexpr std::uintptr_t kStateSnapshotOffset = 0x00ad9a50;
constexpr std::uintptr_t kStateSnapshotDeleteOffset = 0x0105e8bc;
constexpr std::uintptr_t kByteStreamResetReadOffset = 0x0121220c;
constexpr std::uintptr_t kByteStreamFlipReadOffset = 0x0121221c;
constexpr std::uintptr_t kByteStreamResetChecksumOffset = 0x012141bc;
constexpr std::uintptr_t kGameStateDeserializeHeaderOffset = 0x010634c0;
constexpr std::uintptr_t kGameStateDeserializeBodyOffset = 0x0106379c;
constexpr std::uintptr_t kCardSelectionBuildOffset = 0x00e2d1bc;
constexpr std::uintptr_t kJsonMemberFindOffset = 0x00d0e100;
constexpr std::uintptr_t kLogicDataByGlobalIdOffset = 0x00dd1340;
constexpr std::uintptr_t kLocationTileMapEnsureOffset = 0x00857608;
constexpr std::uintptr_t kLocationTileMapLookupOffset = 0x00df8508;
constexpr std::uintptr_t kEglSwapBuffersGotOffset = 0x018e9d88;
// libg calls its AArch64 SHA256 self-integrity initializer exactly once from
// this site. Google's x86_64 libndk_translation advertises the guest feature
// but aborts on SHA256H/SHA256H2/SHA256SU instructions. On translated Android
// only, skip that unused-result initializer so the stock battle engine can
// finish booting. The call remains untouched on real AArch64 devices.
constexpr std::uintptr_t kTranslatedSha256InitializerCallOffset = 0x00c8ece0;
constexpr std::uint32_t kTranslatedSha256InitializerCall = 0x97eba894u;
constexpr std::uint32_t kAarch64Nop = 0xd503201fu;
constexpr std::uintptr_t kGameStateContextRootSlotOffset = 0x019ad878;
constexpr std::uintptr_t kBuffAssetVtableOffset = 0x01882500;
constexpr std::uintptr_t kBuffAssetClassNameGetterOffset = 0x00d961f4;
constexpr std::uintptr_t kLogicDataTypeGetterOffset = 0x00dc2560;
constexpr std::uintptr_t kProjectileVtableOffset = 0x0189cc28;
constexpr std::uintptr_t kProjectileKindGetterOffset = 0x00f27a58;
constexpr std::uintptr_t kCharacterVtableOffset = 0x0189c4e8;
constexpr std::size_t kProjectileDragStageOffset = 0x189;
constexpr std::size_t kProjectileDragBackSpeedOffset = 0x1b0;
constexpr std::uintptr_t kChampionControllerVtableOffset = 0x0189e4e8;
constexpr std::uintptr_t kChampionActionDataVtableOffset = 0x0188cb28;
constexpr std::uintptr_t kCharacterAbilityDataVtableOffset = 0x018823f0;
constexpr std::size_t kCharacterAbilityExtraSpawnBaseOffset = 0x104;
constexpr std::size_t kCharacterAbilityExtraSpawnLimitOffset = 0x108;
constexpr std::size_t kCharacterExtraSpawnAccumulatorOffset = 0x1f0;
constexpr std::uintptr_t kCaptureCharacterActionDataVtableOffset = 0x0188c848;
constexpr std::uintptr_t kCaptureCharacterRuntimeVtableOffset = 0x0189e2d8;
constexpr std::size_t kCaptureRuntimeMaximumTargets = 160;
constexpr std::int32_t kCaptureRuntimeMaximumVectorCapacity = 1024;
constexpr std::int32_t kCaptureRuntimeMaximumConfiguredLimit = 1000;
constexpr std::uintptr_t kThresholdRelocationActionDataVtableOffset = 0x0188ef18;
constexpr std::uintptr_t kThresholdRelocationRuntimeVtableOffset = 0x0189f6f0;
constexpr std::int32_t kThresholdRelocationMaximumThresholds = 16;
constexpr std::int32_t kThresholdRelocationMaximumVectorCapacity = 64;
constexpr std::uintptr_t kPeriodicAttackModifierActionDataVtableOffset = 0x0188eac8;
constexpr std::uintptr_t kPeriodicAttackModifierRuntimeVtableOffset = 0x0189f3d8;
constexpr const char *kRichTelemetrySchema = "native-rich-telemetry.v3";
// Raw objectIndex/secondaryIndex remain authoritative engine fields, but they
// are not a globally unique identity: projectiles can copy their source
// character tuple and engine-owned objects expose -1/-1. Reserve a distinct
// wire namespace for the engine's non-zero, vector-unique LogicGameObject+0x08
// ID instead of falling back to either raw tuple or the compacting vector slot.
constexpr std::int32_t kNativeObjectIdEntityKeyTag = -2;
constexpr std::int32_t kUniqueReadyAbilityEntityKeyTag = -3;
constexpr std::size_t kContentPostProcessReadyOffset = 0x6e8;
constexpr std::uint8_t kReplayJsonSetterPrologue[16] = {
    0xfd, 0x7b, 0xbe, 0xa9, 0xf4, 0x4f, 0x01, 0xa9, 0xfd, 0x03, 0x00, 0x91, 0xf4, 0x03, 0x01, 0xaa,
};
constexpr std::uint8_t kNativeReplayRunnerPrologue[16] = {
    0xff, 0x03, 0x02, 0xd1, 0xfd, 0x7b, 0x02, 0xa9, 0xfb, 0x1b, 0x00, 0xf9, 0xfa, 0x67, 0x04, 0xa9,
};
constexpr std::uint8_t kReplayBattleControllerLifecyclePrologue[16] = {
    0xff, 0x83, 0x03, 0xd1, 0xfd, 0x7b, 0x0a, 0xa9, 0xf8, 0x5f, 0x0b, 0xa9, 0xf6, 0x57, 0x0c, 0xa9,
};
constexpr std::uint8_t kReplayBattleControllerFullUpdatePrologue[16] = {
    0xff, 0x03, 0x02, 0xd1, 0xeb, 0x2b, 0x03, 0x6d, 0xe9, 0x23, 0x04, 0x6d, 0xfd, 0x7b, 0x05, 0xa9,
};
constexpr std::uint8_t kReplayBattleControllerClockPrologue[16] = {
    0x08, 0x30, 0x46, 0x39, 0x48, 0x00, 0x00, 0x34, 0xc0, 0x03, 0x5f, 0xd6, 0xff, 0x83, 0x03, 0xd1,
};
constexpr std::uint8_t kReplayBattleControllerAlternateUpdatePrologue[16] = {
    0xe8, 0x0f, 0x1e, 0xfc, 0xfd, 0xfb, 0x00, 0xa9, 0xf3, 0x0f, 0x00, 0xf9, 0xfd, 0x23, 0x00, 0x91,
};
constexpr std::uint8_t kGameAppPendingEventSetterPrologue[16] = {
    0x08, 0xc0, 0x41, 0xb9, 0x09, 0xbc, 0x41, 0xb9, 0x1f, 0x01, 0x02, 0x6b, 0x08, 0xe0, 0x46, 0x39,
};
constexpr std::uint8_t kGameStateLoadPrologue[16] = {
    0xff, 0x83, 0x01, 0xd1, 0xfd, 0x7b, 0x01, 0xa9, 0xf9, 0x13, 0x00, 0xf9, 0xf8, 0x5f, 0x03, 0xa9,
};
constexpr std::uint8_t kGameStateStepPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf8, 0x5f, 0x01, 0xa9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};

using ReplayJsonSetter = void (*)(void *, void *);
using ReplayManagerGetter = void *(*)();
using LibgStringFromUtf8 = void (*)(void *, const char *, std::int32_t);
using NativeReplayRunner = void (*)(void *, void *, void *);
using ReplayBattleControllerLifecycle = void (*)(void *);
using ReplayBattleControllerUpdate = void (*)(void *, float);
using GameAppPendingEventSetter = void (*)(void *, std::int32_t, std::int32_t, void *, void *, void *);
using NoArgInt32 = std::int32_t (*)();
using NoArgPointer = void *(*)();
using ObjectInt32 = std::int32_t (*)(void *);
using ObjectPointer = void *(*)(void *);
using GameStateLoad = void (*)(void *, void *, void *, void *);
using GameStateStep = void (*)(void *);
using GameStateTick = std::int32_t (*)(void *);
using GameStateContextGetter = void *(*)();
using GameStateNew = void *(*)(std::size_t);
using GameStateDelete = void (*)(void *);
using GameStateConstructor = void (*)(void *, std::int32_t, void *, void *, std::int32_t);
using GameStateSetup = void (*)(void *, std::int32_t);
using GameStateDestroy = void (*)(void *);
using JsonParse = void *(*)(void *, void *, std::int32_t);
using JsonMemberFind = void *(*)(void *, void *, std::int32_t);
using LogicCommandParse = void (*)(void *, void *);
using LogicCommandAppend = void (*)(void *, void **);
using LogicCommandQueueConstructor = void (*)(void *, void *);
using LogicCommandQueueDestroy = void (*)(void *);
using StateSnapshotDelete = void (*)(void *);
using ByteStreamReset = void (*)(void *);
using ByteStreamReadValue = std::int32_t (*)(void *);
using GameStateDeserializeHeader = void (*)(void *, void *);
using GameStateDeserializeBody = void (*)(void *, void *, std::int32_t, void *);
using LogicDataByGlobalId = void *(*)(void *, std::int32_t);
using LocationTileMapEnsure = void (*)(void *);
using LocationTileMapLookup = void *(*)(void *);
ReplayJsonSetter g_original_replay_json_setter = nullptr;
ReplayManagerGetter g_replay_manager_getter = nullptr;
LibgStringFromUtf8 g_libg_string_from_utf8 = nullptr;
JavaVM *g_java_vm = nullptr;
jclass g_native_dialog_manager_class = nullptr;
jmethodID g_native_dialog_dismiss_all = nullptr;
std::atomic<bool> g_probe_runtime_started{false};
std::atomic<bool> g_deferred_probe_bootstrap_started{false};
std::atomic<std::int64_t> g_last_dialog_dismiss_ns{0};
// Counts real native-render logical steps.  They are owned by the stock
// ReplayBattleController update pipeline, never driven from EGL.
std::atomic<std::uint64_t> g_native_forced_steps{0};
std::atomic<std::uint64_t> g_native_stock_step_callbacks{0};
std::atomic<std::uint64_t> g_native_controller_full_updates{0};
std::atomic<std::uint64_t> g_native_controller_clock_updates{0};
std::atomic<std::uint64_t> g_native_controller_alternate_updates{0};
std::atomic<std::uint64_t> g_native_offline_connection_errors_suppressed{0};
// Becomes true only after this process accepts a local control configuration.
// It deliberately survives a later runner-mode transition: an isolated
// offline process must not surface the stock connection dialog while an
// owned headless or native-render session is still alive.
std::atomic<bool> g_offline_control_session_active{false};
std::atomic<std::int32_t> g_native_controller_last_state{-1};
NativeReplayRunner g_original_native_replay_runner = nullptr;
ReplayBattleControllerLifecycle g_original_replay_battle_controller_lifecycle = nullptr;
ReplayBattleControllerUpdate g_original_replay_battle_controller_full_update = nullptr;
ReplayBattleControllerUpdate g_original_replay_battle_controller_clock = nullptr;
ReplayBattleControllerUpdate g_original_replay_battle_controller_alternate_update = nullptr;
GameAppPendingEventSetter g_original_game_app_pending_event_setter = nullptr;
GameStateLoad g_original_game_state_load = nullptr;
GameStateStep g_original_game_state_step = nullptr;
GameStateTick g_game_state_tick = nullptr;
GameStateContextGetter g_game_state_context_a = nullptr;
GameStateContextGetter g_game_state_context_b = nullptr;
GameStateNew g_game_state_new = nullptr;
GameStateDelete g_game_state_delete = nullptr;
GameStateConstructor g_game_state_constructor = nullptr;
GameStateSetup g_game_state_setup = nullptr;
GameStateDestroy g_game_state_destroy = nullptr;
GameStateDestroy g_game_state_base_destroy = nullptr;
JsonParse g_json_parse = nullptr;
JsonMemberFind g_json_member_find = nullptr;
void *g_logic_command_parse = nullptr;
LogicCommandAppend g_logic_command_append = nullptr;
LogicCommandQueueConstructor g_logic_command_queue_constructor = nullptr;
LogicCommandQueueDestroy g_logic_command_queue_destroy = nullptr;
void *g_state_snapshot = nullptr;
StateSnapshotDelete g_state_snapshot_delete = nullptr;
ByteStreamReset g_byte_stream_reset_read = nullptr;
ByteStreamReset g_byte_stream_flip_read = nullptr;
ByteStreamReset g_byte_stream_reset_checksum = nullptr;
GameStateDeserializeHeader g_game_state_deserialize_header = nullptr;
GameStateDeserializeBody g_game_state_deserialize_body = nullptr;
void *g_card_selection_build = nullptr;
LogicDataByGlobalId g_logic_data_by_global_id = nullptr;
LocationTileMapEnsure g_location_tilemap_ensure = nullptr;
LocationTileMapLookup g_location_tilemap_lookup = nullptr;

extern "C" void cr_call_logic_command_parse(void *json_object, void *command_context, void **result, void *target);
extern "C" void cr_call_state_snapshot(void *manager, void **result, void *target);
extern "C" void cr_call_card_selection_build(void *battle_deck_slot, void *player, std::int32_t alternate, void *result,
                                             void *target);

struct ManagerTrace {
  void *manager = nullptr;
  std::uint32_t steps = 0;
};

constexpr std::size_t kMaxTracedManagers = 8;
ManagerTrace g_manager_traces[kMaxTracedManagers];
pthread_mutex_t g_trace_mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic<std::uint32_t> g_load_calls{0};

pthread_mutex_t g_control_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t g_control_cond = PTHREAD_COND_INITIALIZER;
std::int32_t g_engine_instance_id = 0;
std::int32_t g_engine_android_user_id = 0;
std::int32_t g_engine_instance_cpu = -1;
std::uint16_t g_engine_instance_port = kControlPort;
void *g_controlled_manager = nullptr;
std::uint64_t g_control_generation = 0;
std::uint64_t g_completed_steps = 0;
std::uint64_t g_step_budget = 0;
std::int32_t g_last_controlled_tick = -1;
bool g_control_gate_enabled = true;
bool g_control_ended = false;
bool g_controlled_step_active = false;
// Native-render timing remains owned by ReplayBattleController. Quarter-speed
// units keep every supported multiplier exact without persistent float state:
// 1=.25x, 2=.5x, 4=1x, 8=2x, 16=4x.
std::uint32_t g_native_render_speed_quarters = 4;
bool g_native_render_paused = false;
// True only while an unpaused stock controller update can advance the world.
// Control commands temporarily set paused before waiting, preventing a new
// advancing update from entering while a speed/pause transition is committed.
bool g_native_render_step_active = false;
// A synchronized observation consumer asks the stock controller to run an
// exact number of logic ticks, then pause.  The target remains armed until the
// outer controller update exits so additional accumulator steps in that same
// rendered frame are suppressed by game_state_step_hook.
std::int32_t g_native_render_advance_target_tick = -1;
bool g_native_render_advance_reached = false;
bool g_native_render_advance_ended = false;
constexpr std::size_t kMaxPendingCommandBytes = 2048;
char g_pending_command_json[kMaxPendingCommandBytes] = {};
std::uint64_t g_pending_command_sequence = 0;
std::uint64_t g_processed_command_sequence = 0;
bool g_last_injection_succeeded = false;
bool g_native_worker_running = false;
// Resident multi-match mode keeps multiple GameStateManager instances alive
// while one serialized control-server thread is the only lane allowed to
// enter libg.  The legacy worker exits cleanly before this mode is activated.
bool g_resident_mode_requested = false;
bool g_resident_mode_active = false;
std::uint64_t g_reset_request_sequence = 0;
std::uint64_t g_reset_processed_sequence = 0;
constexpr std::size_t kMaxReplayConfigBytes = 64 * 1024;
constexpr std::size_t kMaxControlRequestBytes = kMaxReplayConfigBytes + 1024;
char g_pending_replay_json[kMaxReplayConfigBytes] = {};
char g_active_replay_json[kMaxReplayConfigBytes] = {};
std::uint64_t g_config_request_sequence = 0;
std::uint64_t g_config_processed_sequence = 0;
bool g_last_config_succeeded = false;
std::int32_t g_last_config_stage = 0;
void *g_last_config_root = nullptr;
void *g_last_config_commands = nullptr;
std::uint64_t g_last_config_command_words[8] = {};
void *g_last_config_events = nullptr;
std::int32_t g_last_config_command_count = -1;
std::int32_t g_last_config_event_count = -1;
std::int32_t g_last_config_location_id = 0;
std::int32_t g_last_loaded_queue_count = -1;
enum class RunnerMode : std::int32_t {
  Headless = 0,
  NativeRender = 1,
};
std::atomic<std::int32_t> g_runner_mode{static_cast<std::int32_t>(RunnerMode::Headless)};
char g_pending_native_render_json[kMaxReplayConfigBytes] = {};
std::uint64_t g_native_render_request_sequence = 0;
std::uint64_t g_native_render_processed_sequence = 0;
std::atomic<std::uint64_t> g_native_render_submitted_sequence{0};
std::atomic<std::uint64_t> g_native_render_loaded_sequence{0};
bool g_last_native_render_request_succeeded = false;
std::atomic<void *> g_native_render_replay_manager{nullptr};
std::atomic<void *> g_native_render_controller{nullptr};
std::atomic<void *> g_native_render_manager{nullptr};
std::atomic<std::int32_t> g_native_render_tick{-1};
std::uint64_t g_native_render_state_epoch = 0;
char g_pending_native_command_json[kMaxPendingCommandBytes] = {};
std::uint64_t g_pending_native_command_sequence = 0;
std::uint64_t g_processed_native_command_sequence = 0;
bool g_last_native_injection_succeeded = false;
// Replay actions are registered while the controlled manager is paused at
// startup, but their live hand/Champion identity is resolved by the game-step
// hook at execute_tick - 1.  The same semantic queue serves the stock renderer
// and the detached headless manager so replay fidelity does not depend on the
// host-side driver.
constexpr std::size_t kMaxScheduledReplayActions = 4096;
constexpr std::size_t kMaxScheduledAbilityHints = 260;
enum class ScheduledReplayActionKind : std::int32_t {
  Card = 1,
  Ability = 2,
};
enum class ScheduledReplayActionState : std::int32_t {
  Pending = 0,
  Succeeded = 1,
  Failed = 2,
};
enum class ScheduledReplayActionError : std::int32_t {
  None = 0,
  MissedBoundary = 1,
  CardUnavailable = 2,
  AbilityUnavailable = 3,
  CommandEncoding = 4,
  InjectionFailed = 5,
  GenerationChanged = 6,
  AbilityAmbiguous = 7,
};
struct ScheduledReplayAction {
  std::uint64_t sequence = 0;
  RunnerMode mode = RunnerMode::Headless;
  void *manager = nullptr;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  ScheduledReplayActionKind kind = ScheduledReplayActionKind::Card;
  ScheduledReplayActionState state = ScheduledReplayActionState::Pending;
  ScheduledReplayActionError error = ScheduledReplayActionError::None;
  std::int32_t registered_tick = -1;
  std::int32_t target_execute_tick = -1;
  std::int32_t processed_tick = -1;
  std::int32_t owner = -1;
  std::uint32_t card_id = 0;
  std::int32_t position_x = 0;
  std::int32_t position_y = 0;
  std::uint32_t resolved_command_card_id = 0;
  std::uint32_t resolved_card_parameter = 0;
  char ability_hints[kMaxScheduledAbilityHints + 1] = {};
};
ScheduledReplayAction g_scheduled_replay_actions[kMaxScheduledReplayActions] = {};
std::size_t g_scheduled_replay_action_count = 0;
std::uint64_t g_next_scheduled_replay_action_sequence = 0;
const void *g_fixture_battle = nullptr;
const void *g_fixture_command_array = nullptr;
std::atomic<void *> g_expected_headless_load_manager{nullptr};
std::atomic<std::uint32_t> g_tilemap_ensure_calls{0};
std::atomic<void *> g_last_location_data{nullptr};
std::atomic<void *> g_last_tilemap{nullptr};
pthread_mutex_t g_tilemap_ensure_mutex = PTHREAD_MUTEX_INITIALIZER;
std::uintptr_t g_libg_base = 0;
const ElfW(Phdr) *g_libg_phdr = nullptr;
ElfW(Half) g_libg_phnum = 0;
char g_libg_path[1024] = {};
std::atomic<bool> g_hook_render_gate{false};
std::atomic<bool> g_hook_native_replay_runner{false};
std::atomic<bool> g_hook_replay_battle_controller_lifecycle{false};
std::atomic<bool> g_hook_replay_battle_controller_full_update{false};
std::atomic<bool> g_hook_replay_battle_controller_clock{false};
std::atomic<bool> g_hook_replay_battle_controller_alternate_update{false};
std::atomic<bool> g_hook_game_app_pending_event_setter{false};
std::atomic<bool> g_hook_replay_json_setter{false};
std::atomic<bool> g_hook_game_state_load{false};
std::atomic<bool> g_hook_game_state_step{false};
std::atomic<bool> g_hook_install_ok{false};

struct Sha256Context {
  std::uint8_t block[64] = {};
  std::uint32_t state[8] = {};
  std::uint64_t bit_count = 0;
  std::size_t block_size = 0;
};

constexpr std::uint32_t kSha256Constants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t count) {
  return (value >> count) | (value << (32u - count));
}

void sha256_transform(Sha256Context *context, const std::uint8_t *block) {
  std::uint32_t words[64] = {};
  for (std::size_t index = 0; index < 16; ++index) {
    words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24u) |
                   (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16u) |
                   (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8u) |
                   static_cast<std::uint32_t>(block[index * 4 + 3]);
  }
  for (std::size_t index = 16; index < 64; ++index) {
    const std::uint32_t s0 =
        rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3u);
    const std::uint32_t s1 =
        rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10u);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }

  std::uint32_t a = context->state[0];
  std::uint32_t b = context->state[1];
  std::uint32_t c = context->state[2];
  std::uint32_t d = context->state[3];
  std::uint32_t e = context->state[4];
  std::uint32_t f = context->state[5];
  std::uint32_t g = context->state[6];
  std::uint32_t h = context->state[7];
  for (std::size_t index = 0; index < 64; ++index) {
    const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    const std::uint32_t choice = (e & f) ^ ((~e) & g);
    const std::uint32_t temporary1 = h + sum1 + choice + kSha256Constants[index] + words[index];
    const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  context->state[0] += a;
  context->state[1] += b;
  context->state[2] += c;
  context->state[3] += d;
  context->state[4] += e;
  context->state[5] += f;
  context->state[6] += g;
  context->state[7] += h;
}

void sha256_init(Sha256Context *context) {
  context->state[0] = 0x6a09e667u;
  context->state[1] = 0xbb67ae85u;
  context->state[2] = 0x3c6ef372u;
  context->state[3] = 0xa54ff53au;
  context->state[4] = 0x510e527fu;
  context->state[5] = 0x9b05688cu;
  context->state[6] = 0x1f83d9abu;
  context->state[7] = 0x5be0cd19u;
  context->bit_count = 0;
  context->block_size = 0;
}

void sha256_update(Sha256Context *context, const std::uint8_t *data, std::size_t size) {
  for (std::size_t index = 0; index < size; ++index) {
    context->block[context->block_size++] = data[index];
    if (context->block_size == sizeof(context->block)) {
      sha256_transform(context, context->block);
      context->bit_count += 512;
      context->block_size = 0;
    }
  }
}

void sha256_finish(Sha256Context *context, std::uint8_t digest[32]) {
  std::size_t index = context->block_size;
  context->block[index++] = 0x80u;
  if (index > 56) {
    while (index < 64) {
      context->block[index++] = 0;
    }
    sha256_transform(context, context->block);
    index = 0;
  }
  while (index < 56) {
    context->block[index++] = 0;
  }
  context->bit_count += context->block_size * 8u;
  for (std::size_t byte = 0; byte < 8; ++byte) {
    context->block[63 - byte] = static_cast<std::uint8_t>(context->bit_count >> (byte * 8u));
  }
  sha256_transform(context, context->block);
  for (std::size_t word = 0; word < 8; ++word) {
    digest[word * 4] = static_cast<std::uint8_t>(context->state[word] >> 24u);
    digest[word * 4 + 1] = static_cast<std::uint8_t>(context->state[word] >> 16u);
    digest[word * 4 + 2] = static_cast<std::uint8_t>(context->state[word] >> 8u);
    digest[word * 4 + 3] = static_cast<std::uint8_t>(context->state[word]);
  }
}

void bytes_to_hex(const std::uint8_t *bytes, std::size_t size, char *output, std::size_t output_size) {
  constexpr char kHex[] = "0123456789abcdef";
  if (output == nullptr || output_size < size * 2 + 1) {
    return;
  }
  for (std::size_t index = 0; index < size; ++index) {
    output[index * 2] = kHex[bytes[index] >> 4u];
    output[index * 2 + 1] = kHex[bytes[index] & 0x0fu];
  }
  output[size * 2] = '\0';
}

bool sha256_bytes(const void *data, std::size_t size, char output[65]) {
  if ((data == nullptr && size != 0) || output == nullptr) {
    return false;
  }
  Sha256Context context;
  sha256_init(&context);
  sha256_update(&context, static_cast<const std::uint8_t *>(data), size);
  std::uint8_t digest[32] = {};
  sha256_finish(&context, digest);
  bytes_to_hex(digest, sizeof(digest), output, 65);
  return output[0] != '\0';
}

bool sha256_file(const char *path, char output[65]) {
  if (path == nullptr || path[0] == '\0' || output == nullptr) {
    return false;
  }
  output[0] = '\0';
  const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return false;
  }
  Sha256Context context;
  sha256_init(&context);
  std::uint8_t buffer[64 * 1024] = {};
  bool succeeded = true;
  while (true) {
    const ssize_t count = read(descriptor, buffer, sizeof(buffer));
    if (count > 0) {
      sha256_update(&context, buffer, static_cast<std::size_t>(count));
      continue;
    }
    if (count == 0) {
      break;
    }
    if (errno == EINTR) {
      continue;
    }
    succeeded = false;
    break;
  }
  close(descriptor);
  if (!succeeded) {
    return false;
  }
  std::uint8_t digest[32] = {};
  sha256_finish(&context, digest);
  bytes_to_hex(digest, sizeof(digest), output, 65);
  return true;
}
constexpr std::size_t kControlResponseBytes = 64 * 1024;
constexpr std::size_t kOrdinaryObservationResponseBytes = 256 * 1024;
// Full observation captures are heap-backed.  The ordinary schema permits 256
// object-vector slots, while one worst-case rich item is bounded by 24 KiB.
// Twelve MiB covers the complete 256-slot ordinary + rich envelope plus the
// combat, phase, visibility and remaining-runtime rings.  The host transport
// retains a separate 16 MiB hard bound; an incomplete encoding fails closed.
constexpr std::size_t kFullObservationResponseBytes = 12 * 1024 * 1024;
constexpr std::int32_t kMaxObservedObjects = 256;
constexpr std::int32_t kMaxRichObservedObjects = kMaxObservedObjects;
constexpr std::int32_t kMaxObjectComponents = 64;
constexpr std::int32_t kMaxActiveEffects = 64;
constexpr std::int32_t kMaxAttackSequenceStage = 255;
constexpr std::int32_t kPlayerCount = 2;
constexpr std::int32_t kMaxDeckCards = 32;
constexpr std::int32_t kMaxHandCards = 16;
constexpr std::int32_t kMaxCycleCards = 32;
constexpr std::uint32_t kLiveCommandAgeTicks = 20;
constexpr std::size_t kMaxTransitionActions = 8;
constexpr std::size_t kMaxStoredSnapshots = 64;
constexpr std::size_t kMaxStoredTowerTroopRuntimes = 6;

enum class SnapshotOperation : std::int32_t {
  None = 0,
  Create = 1,
  Restore = 2,
  Release = 3,
};

enum class SnapshotError : std::int32_t {
  None = 0,
  TableFull = 1,
  CaptureFailed = 2,
  UnknownHandle = 3,
  StaleHandle = 4,
  RestoreUnavailable = 5,
  RestoreVerificationFailed = 6,
  ManagerChanged = 7,
};

struct StoredTowerTroopRuntimeState {
  std::uint8_t kind = 0;
  std::uint32_t source_native_object_id = 0;
  std::int32_t charge_count = 0;
  std::int32_t max_charge_count = 0;
  std::int32_t recharge_elapsed_ms = 0;
  std::int32_t recharge_duration_ms = 0;
  std::int32_t start_delay_remaining_ms = 0;
  std::int32_t start_delay_duration_ms = 0;
  std::int32_t cooking_contribution = 0;
  std::int32_t contribution_needed = 0;
  std::int32_t throw_delay_remaining_ms = -1;
  std::uint32_t target_native_object_id = 0;
};

struct StoredSnapshot {
  void *wrapper = nullptr;
  std::uint64_t handle = 0;
  RunnerMode mode = RunnerMode::Headless;
  std::uint64_t generation = 0;
  std::uint64_t config_revision = 0;
  std::uint64_t state_epoch = 0;
  std::uint64_t completed_steps = 0;
  std::uint64_t digest = 0;
  std::int32_t tick = -1;
  std::uint32_t serialized_bytes = 0;
  std::size_t tower_troop_runtime_count = 0;
  StoredTowerTroopRuntimeState tower_troop_runtimes[kMaxStoredTowerTroopRuntimes];
};

StoredSnapshot g_stored_snapshots[kMaxStoredSnapshots];
std::uint64_t g_next_snapshot_handle = 1;
std::uint64_t g_state_epoch = 0;
std::uint64_t g_snapshot_request_sequence = 0;
std::uint64_t g_snapshot_processed_sequence = 0;
std::uint64_t g_snapshot_processing_sequence = 0;
SnapshotOperation g_pending_snapshot_operation = SnapshotOperation::None;
RunnerMode g_pending_snapshot_mode = RunnerMode::Headless;
void *g_pending_snapshot_manager = nullptr;
std::uint64_t g_pending_snapshot_generation = 0;
std::uint64_t g_pending_snapshot_handle = 0;
bool g_last_snapshot_operation_succeeded = false;
SnapshotError g_last_snapshot_operation_error = SnapshotError::None;
StoredSnapshot g_last_snapshot_operation_result;

using EglSwapBuffers = std::int32_t (*)(void *, void *);
void **g_egl_swap_buffers_got = nullptr;
EglSwapBuffers g_original_egl_swap_buffers = nullptr;
pthread_mutex_t g_render_gate_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t g_render_gate_cond = PTHREAD_COND_INITIALIZER;
bool g_render_suppressed = false;

bool start_native_controlled_worker_locked();
void process_native_render_request_on_frame_thread();
void dismiss_native_dialogs_if_needed();
bool suppress_offline_connection_error_if_needed();
void game_state_step_hook(void *manager);
void game_state_load_hook(void *manager, void *battle, void *command_array, void *auxiliary);
void begin_combat_event_epoch(void *game_manager, std::uint64_t generation, std::uint64_t state_epoch);
void begin_runtime_telemetry_epoch(void *game_manager, std::uint64_t generation, std::uint64_t state_epoch);
void bind_combat_event_object_manager(void *game_manager);
void end_combat_event_epoch(void *game_manager);
std::size_t copy_tower_troop_runtime_snapshot(std::uint64_t generation, std::uint64_t state_epoch, std::int32_t tick,
                                              StoredTowerTroopRuntimeState *output, std::size_t capacity);
void restore_tower_troop_runtime_snapshot(std::uint64_t generation, std::uint64_t state_epoch, std::int32_t tick,
                                          const StoredTowerTroopRuntimeState *states, std::size_t count);
bool install_inline_hook(std::uintptr_t libg_base, std::uintptr_t offset, const std::uint8_t (&expected_prologue)[16],
                         const void *replacement, void **original, const char *label);
bool process_pending_native_command(void *manager);
void process_scheduled_replay_actions(void *manager);
bool format_probe_native_touch_status(char *response, std::size_t response_size);
void run_control_server(std::uint16_t port);

const char *native_render_speed_json(std::uint32_t speed_quarters) {
  switch (speed_quarters) {
  case 1:
    return "0.25";
  case 2:
    return "0.5";
  case 4:
    return "1";
  case 8:
    return "2";
  case 16:
    return "4";
  default:
    return "1";
  }
}

bool parse_native_render_speed(const char *text, std::uint32_t *speed_quarters_out) {
  if (text == nullptr || speed_quarters_out == nullptr) {
    return false;
  }
  while (*text == ' ' || *text == '\t') {
    ++text;
  }
  struct SpeedChoice {
    const char *text;
    std::uint32_t quarters;
  };
  constexpr SpeedChoice kChoices[] = {
      {"0.25", 1}, {"0.5", 2}, {"1", 4}, {"1.0", 4}, {"2", 8}, {"2.0", 8}, {"4", 16}, {"4.0", 16},
  };
  for (const auto &choice : kChoices) {
    const std::size_t size = std::strlen(choice.text);
    if (std::strncmp(text, choice.text, size) != 0) {
      continue;
    }
    const char *tail = text + size;
    while (*tail == ' ' || *tail == '\t') {
      ++tail;
    }
    if (*tail == '\0') {
      *speed_quarters_out = choice.quarters;
      return true;
    }
  }
  return false;
}

bool inspect_content_runtime(void **root_out = nullptr, void **context_out = nullptr) {
  void *root = nullptr;
  void *context = nullptr;
  if (g_libg_base != 0) {
    std::memcpy(&root, reinterpret_cast<const void *>(g_libg_base + kGameStateContextRootSlotOffset), sizeof(root));
    if (root != nullptr) {
      std::memcpy(&context, static_cast<const std::uint8_t *>(root) + 0x20, sizeof(context));
    }
  }
  if (root_out != nullptr) {
    *root_out = root;
  }
  if (context_out != nullptr) {
    *context_out = context;
  }
  std::uint8_t post_process_ready = 0;
  if (context != nullptr) {
    std::memcpy(&post_process_ready, static_cast<const std::uint8_t *>(context) + kContentPostProcessReadyOffset,
                sizeof(post_process_ready));
  }
  return root != nullptr && context != nullptr && post_process_ready == 1;
}

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, kLogTag, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, kLogTag, __VA_ARGS__)

bool prewarm_location_tilemap(void *content_context, std::int32_t location_id) {
  const std::uint32_t call = g_tilemap_ensure_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  void *location_data = nullptr;
  void *tilemap = nullptr;

  pthread_mutex_lock(&g_tilemap_ensure_mutex);
  if (content_context != nullptr && location_id > 0 && g_logic_data_by_global_id != nullptr) {
    location_data = g_logic_data_by_global_id(content_context, location_id);
  }
  if (location_data != nullptr && g_location_tilemap_lookup != nullptr) {
    tilemap = g_location_tilemap_lookup(location_data);
  }
  if (location_data != nullptr && tilemap == nullptr && inspect_content_runtime() &&
      g_location_tilemap_ensure != nullptr) {
    g_location_tilemap_ensure(location_data);
    if (g_location_tilemap_lookup != nullptr) {
      tilemap = g_location_tilemap_lookup(location_data);
    }
  }
  pthread_mutex_unlock(&g_tilemap_ensure_mutex);

  g_last_location_data.store(location_data, std::memory_order_release);
  g_last_tilemap.store(tilemap, std::memory_order_release);
  LOGI("cold tilemap prewarm #%u context=%p locationId=%d location=%p tilemap=%p", call, content_context, location_id,
       location_data, tilemap);
  return location_data != nullptr && tilemap != nullptr;
}

struct LibgLookup {
  std::uintptr_t base = 0;
  const ElfW(Phdr) *phdr = nullptr;
  ElfW(Half) phnum = 0;
  const char *path = nullptr;
};

int find_libg_callback(dl_phdr_info *info, std::size_t, void *opaque) {
  if (info == nullptr || info->dlpi_name == nullptr) {
    return 0;
  }
  if (std::strstr(info->dlpi_name, "libg.so") == nullptr) {
    return 0;
  }
  LibgLookup *lookup = static_cast<LibgLookup *>(opaque);
  lookup->base = static_cast<std::uintptr_t>(info->dlpi_addr);
  lookup->phdr = info->dlpi_phdr;
  lookup->phnum = info->dlpi_phnum;
  lookup->path = info->dlpi_name;
  return 1;
}

std::uintptr_t find_libg_base() {
  LibgLookup lookup;
  dl_iterate_phdr(find_libg_callback, &lookup);
  g_libg_phdr = lookup.phdr;
  g_libg_phnum = lookup.phnum;
  if (lookup.path != nullptr) {
    std::snprintf(g_libg_path, sizeof(g_libg_path), "%s", lookup.path);
  }
  return lookup.base;
}

std::uintptr_t parse_libg_proc_maps_line(char *line) {
  char *cursor = line;
  char *end = nullptr;
  const unsigned long long mapping_start = std::strtoull(cursor, &end, 16);
  if (end == cursor || *end != '-') {
    return 0;
  }
  cursor = end + 1;
  std::strtoull(cursor, &end, 16);
  if (end == cursor) {
    return 0;
  }
  cursor = end;
  for (int field = 0; field < 1; ++field) {
    while (*cursor == ' ') {
      ++cursor;
    }
    while (*cursor != '\0' && *cursor != ' ') {
      ++cursor;
    }
  }
  while (*cursor == ' ') {
    ++cursor;
  }
  const unsigned long long file_offset = std::strtoull(cursor, &end, 16);
  if (end == cursor || mapping_start < file_offset) {
    return 0;
  }
  cursor = end;
  for (int field = 0; field < 2; ++field) {
    while (*cursor == ' ') {
      ++cursor;
    }
    while (*cursor != '\0' && *cursor != ' ') {
      ++cursor;
    }
  }
  while (*cursor == ' ') {
    ++cursor;
  }
  const char *basename = std::strrchr(cursor, '/');
  if (basename == nullptr || std::strcmp(basename + 1, "libg.so") != 0) {
    return 0;
  }
  const std::uintptr_t candidate = static_cast<std::uintptr_t>(mapping_start - file_offset);
  const auto *header = reinterpret_cast<const ElfW(Ehdr) *>(candidate);
  if (std::memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 || header->e_phentsize != sizeof(ElfW(Phdr)) ||
      header->e_phnum == 0) {
    return 0;
  }
  g_libg_phdr = reinterpret_cast<const ElfW(Phdr) *>(candidate + header->e_phoff);
  g_libg_phnum = header->e_phnum;
  std::snprintf(g_libg_path, sizeof(g_libg_path), "%s", cursor);
  return candidate;
}

std::uintptr_t find_libg_base_from_proc_maps() {
  const int descriptor = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return 0;
  }
  char input[4096] = {};
  char line[2048] = {};
  std::size_t line_size = 0;
  std::uintptr_t result = 0;
  while (result == 0) {
    const ssize_t count = read(descriptor, input, sizeof(input));
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      break;
    }
    for (ssize_t index = 0; index < count; ++index) {
      const char value = input[index];
      if (value == '\n') {
        line[line_size] = '\0';
        result = parse_libg_proc_maps_line(line);
        line_size = 0;
        if (result != 0) {
          break;
        }
      } else if (line_size + 1 < sizeof(line)) {
        line[line_size++] = value;
      } else {
        line_size = 0;
      }
    }
  }
  close(descriptor);
  return result;
}

const char *first_readable_path(const char *const *candidates, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    if (candidates[index] != nullptr && access(candidates[index], R_OK) == 0) {
      return candidates[index];
    }
  }
  return nullptr;
}

bool read_text_file(const char *path, char **data_out, std::size_t *size_out,
                    std::size_t maximum_size = 32u * 1024u * 1024u) {
  if (path == nullptr || data_out == nullptr || size_out == nullptr) {
    return false;
  }
  *data_out = nullptr;
  *size_out = 0;
  struct stat status {};
  if (stat(path, &status) != 0 || status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) > maximum_size) {
    return false;
  }
  const std::size_t size = static_cast<std::size_t>(status.st_size);
  char *data = static_cast<char *>(std::malloc(size + 1));
  if (data == nullptr) {
    return false;
  }
  const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    std::free(data);
    return false;
  }
  std::size_t offset = 0;
  bool succeeded = true;
  while (offset < size) {
    const ssize_t count = read(descriptor, data + offset, size - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      succeeded = false;
      break;
    }
  }
  close(descriptor);
  if (!succeeded || offset != size) {
    std::free(data);
    return false;
  }
  data[size] = '\0';
  *data_out = data;
  *size_out = size;
  return true;
}

bool extract_last_json_string(const char *json, const char *key, char *output, std::size_t output_size) {
  if (json == nullptr || key == nullptr || output == nullptr || output_size == 0) {
    return false;
  }
  output[0] = '\0';
  char needle[128] = {};
  const int needle_size = std::snprintf(needle, sizeof(needle), "\"%s\"", key);
  if (needle_size <= 0 || static_cast<std::size_t>(needle_size) >= sizeof(needle)) {
    return false;
  }
  const char *cursor = json;
  const char *last_value = nullptr;
  std::size_t last_size = 0;
  while ((cursor = std::strstr(cursor, needle)) != nullptr) {
    const char *value = cursor + needle_size;
    while (*value == ' ' || *value == '\t' || *value == '\r' || *value == '\n') {
      ++value;
    }
    if (*value++ != ':') {
      cursor += needle_size;
      continue;
    }
    while (*value == ' ' || *value == '\t' || *value == '\r' || *value == '\n') {
      ++value;
    }
    if (*value++ != '"') {
      cursor += needle_size;
      continue;
    }
    const char *end = value;
    while (*end != '\0' && *end != '"' && *end != '\\') {
      ++end;
    }
    if (*end == '"') {
      last_value = value;
      last_size = static_cast<std::size_t>(end - value);
    }
    cursor += needle_size;
  }
  if (last_value == nullptr || last_size == 0 || last_size >= output_size) {
    return false;
  }
  std::memcpy(output, last_value, last_size);
  output[last_size] = '\0';
  return true;
}

bool is_lower_hex_string(const char *value, std::size_t expected_size) {
  if (value == nullptr || std::strlen(value) != expected_size) {
    return false;
  }
  for (std::size_t index = 0; index < expected_size; ++index) {
    if (!((value[index] >= '0' && value[index] <= '9') || (value[index] >= 'a' && value[index] <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool is_json_token(const char *value) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  for (const char *cursor = value; *cursor != '\0'; ++cursor) {
    const char character = *cursor;
    if (!((character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9') || character == '.' || character == '_' || character == '-')) {
      return false;
    }
  }
  return true;
}

bool process_maps_contains(const char *marker) {
  if (marker == nullptr || marker[0] == '\0') {
    return false;
  }
  FILE *maps = std::fopen("/proc/self/maps", "r");
  if (maps == nullptr) {
    return false;
  }
  char line[4096] = {};
  bool found = false;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    if (std::strstr(line, marker) != nullptr) {
      found = true;
      break;
    }
  }
  std::fclose(maps);
  return found;
}

std::size_t align_note_size(std::size_t value) { return (value + 3u) & ~static_cast<std::size_t>(3u); }

bool libg_build_id(char output[41]) {
  if (output == nullptr || g_libg_base == 0 || g_libg_phdr == nullptr) {
    return false;
  }
  output[0] = '\0';
  for (ElfW(Half) index = 0; index < g_libg_phnum; ++index) {
    const ElfW(Phdr) &header = g_libg_phdr[index];
    if (header.p_type != PT_NOTE || header.p_memsz < sizeof(ElfW(Nhdr))) {
      continue;
    }
    const auto *cursor =
        reinterpret_cast<const std::uint8_t *>(g_libg_base + static_cast<std::uintptr_t>(header.p_vaddr));
    const auto *end = cursor + header.p_memsz;
    while (cursor + sizeof(ElfW(Nhdr)) <= end) {
      const auto *note = reinterpret_cast<const ElfW(Nhdr) *>(cursor);
      cursor += sizeof(ElfW(Nhdr));
      const std::size_t name_size = align_note_size(note->n_namesz);
      const std::size_t description_size = align_note_size(note->n_descsz);
      if (cursor + name_size > end || cursor + name_size + description_size > end) {
        break;
      }
      const std::uint8_t *name = cursor;
      const std::uint8_t *description = cursor + name_size;
      if (note->n_type == NT_GNU_BUILD_ID && note->n_namesz >= 3 && std::memcmp(name, "GNU", 3) == 0 &&
          note->n_descsz == 20) {
        bytes_to_hex(description, note->n_descsz, output, 41);
        return true;
      }
      cursor += name_size + description_size;
    }
  }
  return false;
}

struct RunnerAttestationSnapshot {
  char probe_sha256[65] = {};
  char libg_sha256[65] = {};
  char libg_build_id[41] = {};
  char content_fingerprint_sha256[65] = {};
  char content_assets_sha256[65] = {};
  char content_version[64] = {};
  char content_manifest_sha1[41] = {};
  char attestation_digest[65] = {};
  bool hook_install_ok = false;
  bool hook_game_state_load = false;
  bool hook_game_state_step = false;
  bool hook_native_replay_runner = false;
  bool hook_render_gate = false;
  bool hook_replay_json_setter = false;
  bool capability_headless = false;
  bool capability_native_render = false;
  bool capability_snapshot_restore = false;
  bool capability_structured_observation = false;
  bool capability_two_sided_actions = false;
  bool content_ready = false;
  bool content_assets_mapped = false;
  bool production_ready = false;
};

const char *json_boolean(bool value) { return value ? "true" : "false"; }

bool collect_runner_attestation(RunnerAttestationSnapshot *result) {
  if (result == nullptr) {
    return false;
  }
  *result = RunnerAttestationSnapshot{};

  Dl_info probe_info{};
  if (dladdr(reinterpret_cast<const void *>(&sha256_file), &probe_info) != 0 && probe_info.dli_fname != nullptr) {
    sha256_file(probe_info.dli_fname, result->probe_sha256);
  }
  sha256_file(g_libg_path, result->libg_sha256);
  libg_build_id(result->libg_build_id);

  const char *fingerprint_path =
      first_readable_path(g_fingerprint_paths, sizeof(g_fingerprint_paths) / sizeof(g_fingerprint_paths[0]));
  const char *assets_path = first_readable_path(g_assets_paths, sizeof(g_assets_paths) / sizeof(g_assets_paths[0]));
  if (fingerprint_path != nullptr) {
    sha256_file(fingerprint_path, result->content_fingerprint_sha256);
    char *fingerprint = nullptr;
    std::size_t fingerprint_size = 0;
    if (read_text_file(fingerprint_path, &fingerprint, &fingerprint_size)) {
      extract_last_json_string(fingerprint, "version", result->content_version, sizeof(result->content_version));
      extract_last_json_string(fingerprint, "sha", result->content_manifest_sha1,
                               sizeof(result->content_manifest_sha1));
      std::free(fingerprint);
    }
  }
  if (assets_path != nullptr) {
    sha256_file(assets_path, result->content_assets_sha256);
  }
  if (!is_json_token(result->content_version)) {
    result->content_version[0] = '\0';
  }
  if (!is_lower_hex_string(result->content_manifest_sha1, 40)) {
    result->content_manifest_sha1[0] = '\0';
  }

  result->hook_install_ok = g_hook_install_ok.load(std::memory_order_acquire);
  result->hook_game_state_load = g_hook_game_state_load.load(std::memory_order_acquire);
  result->hook_game_state_step = g_hook_game_state_step.load(std::memory_order_acquire);
  result->hook_native_replay_runner = g_hook_native_replay_runner.load(std::memory_order_acquire);
  result->hook_render_gate = g_hook_render_gate.load(std::memory_order_acquire);
  result->hook_replay_json_setter = g_hook_replay_json_setter.load(std::memory_order_acquire);

  result->capability_headless = result->hook_game_state_load && result->hook_game_state_step &&
                                g_game_state_new != nullptr && g_game_state_constructor != nullptr &&
                                g_game_state_setup != nullptr;
  result->capability_native_render = result->hook_native_replay_runner && result->hook_replay_json_setter &&
                                     result->hook_render_gate &&
                                     g_hook_game_app_pending_event_setter.load(std::memory_order_acquire) &&
                                     g_hook_replay_battle_controller_lifecycle.load(std::memory_order_acquire) &&
                                     g_hook_replay_battle_controller_full_update.load(std::memory_order_acquire) &&
                                     g_hook_replay_battle_controller_clock.load(std::memory_order_acquire) &&
                                     g_hook_replay_battle_controller_alternate_update.load(std::memory_order_acquire) &&
                                     g_replay_manager_getter != nullptr;
  result->capability_snapshot_restore = result->capability_headless && g_state_snapshot != nullptr &&
                                        g_game_state_deserialize_header != nullptr &&
                                        g_game_state_deserialize_body != nullptr;
  result->capability_structured_observation = result->capability_headless && g_game_state_tick != nullptr;
  result->capability_two_sided_actions = result->capability_headless && g_logic_command_parse != nullptr &&
                                         g_logic_command_append != nullptr &&
                                         g_logic_command_queue_constructor != nullptr;
  result->content_ready = inspect_content_runtime();
  result->content_assets_mapped = process_maps_contains("assets.scdb");

  const bool hashes_complete =
      is_lower_hex_string(result->probe_sha256, 64) && is_lower_hex_string(result->libg_sha256, 64) &&
      is_lower_hex_string(result->libg_build_id, 40) && is_lower_hex_string(result->content_fingerprint_sha256, 64) &&
      is_lower_hex_string(result->content_assets_sha256, 64) && result->content_version[0] != '\0' &&
      is_lower_hex_string(result->content_manifest_sha1, 40);
  result->production_ready =
      hashes_complete && result->hook_install_ok && result->content_ready && result->content_assets_mapped &&
      result->hook_game_state_load && result->hook_game_state_step && result->hook_native_replay_runner &&
      result->hook_render_gate && result->hook_replay_json_setter && result->capability_headless &&
      result->capability_native_render && result->capability_snapshot_restore &&
      result->capability_structured_observation && result->capability_two_sided_actions;

  char identity[4096] = {};
  const int identity_size = std::snprintf(
      identity, sizeof(identity),
      "{\"abi\":\"%s\",\"capabilities\":{"
      "\"headless\":%s,\"native_render\":%s,\"snapshot_restore\":%s,"
      "\"structured_observation\":%s,\"two_sided_actions\":%s},"
      "\"content_assets_sha256\":\"%s\","
      "\"content_fingerprint_sha256\":\"%s\","
      "\"content_manifest_sha1\":\"%s\",\"content_version\":\"%s\","
      "\"hook_install_ok\":%s,\"hooks\":{"
      "\"game_state_load\":%s,\"game_state_step\":%s,"
      "\"native_replay_runner\":%s,\"render_gate\":%s,"
      "\"replay_json_setter\":%s},\"libg_build_id\":\"%s\","
      "\"libg_sha256\":\"%s\",\"package_name\":\"%s\","
      "\"probe_sha256\":\"%s\",\"protocol_version\":\"%s\","
      "\"version\":\"%s\"}",
      kAbi, json_boolean(result->capability_headless), json_boolean(result->capability_native_render),
      json_boolean(result->capability_snapshot_restore), json_boolean(result->capability_structured_observation),
      json_boolean(result->capability_two_sided_actions), result->content_assets_sha256,
      result->content_fingerprint_sha256, result->content_manifest_sha1, result->content_version,
      json_boolean(result->hook_install_ok), json_boolean(result->hook_game_state_load),
      json_boolean(result->hook_game_state_step), json_boolean(result->hook_native_replay_runner),
      json_boolean(result->hook_render_gate), json_boolean(result->hook_replay_json_setter), result->libg_build_id,
      result->libg_sha256, kPackageName, result->probe_sha256, kProtocolVersion, kAttestationVersion);
  if (identity_size <= 0 || static_cast<std::size_t>(identity_size) >= sizeof(identity)) {
    return false;
  }
  return sha256_bytes(identity, static_cast<std::size_t>(identity_size), result->attestation_digest);
}

bool format_runner_attestation(char *response, std::size_t response_size) {
  RunnerAttestationSnapshot attestation;
  if (response == nullptr || response_size == 0 || !collect_runner_attestation(&attestation)) {
    return false;
  }
  const int written = std::snprintf(
      response, response_size,
      "{\"ok\":true,\"attestation\":{"
      "\"version\":\"%s\",\"protocol_version\":\"%s\","
      "\"package_name\":\"%s\",\"abi\":\"%s\","
      "\"probe_sha256\":\"%s\",\"libg_sha256\":\"%s\","
      "\"libg_build_id\":\"%s\","
      "\"content_fingerprint_sha256\":\"%s\","
      "\"content_assets_sha256\":\"%s\","
      "\"content_version\":\"%s\",\"content_manifest_sha1\":\"%s\","
      "\"hook_install_ok\":%s,\"hooks\":{"
      "\"game_state_load\":%s,\"game_state_step\":%s,"
      "\"native_replay_runner\":%s,\"render_gate\":%s,"
      "\"replay_json_setter\":%s},\"capabilities\":{"
      "\"headless\":%s,\"native_render\":%s,\"snapshot_restore\":%s,"
      "\"structured_observation\":%s,\"two_sided_actions\":%s},"
      "\"content_ready\":%s,\"content_assets_mapped\":%s,"
      "\"production_ready\":%s,\"attestation_digest\":\"%s\"}}",
      kAttestationVersion, kProtocolVersion, kPackageName, kAbi, attestation.probe_sha256, attestation.libg_sha256,
      attestation.libg_build_id, attestation.content_fingerprint_sha256, attestation.content_assets_sha256,
      attestation.content_version, attestation.content_manifest_sha1, json_boolean(attestation.hook_install_ok),
      json_boolean(attestation.hook_game_state_load), json_boolean(attestation.hook_game_state_step),
      json_boolean(attestation.hook_native_replay_runner), json_boolean(attestation.hook_render_gate),
      json_boolean(attestation.hook_replay_json_setter), json_boolean(attestation.capability_headless),
      json_boolean(attestation.capability_native_render), json_boolean(attestation.capability_snapshot_restore),
      json_boolean(attestation.capability_structured_observation),
      json_boolean(attestation.capability_two_sided_actions), json_boolean(attestation.content_ready),
      json_boolean(attestation.content_assets_mapped), json_boolean(attestation.production_ready),
      attestation.attestation_digest);
  return written > 0 && static_cast<std::size_t>(written) < response_size;
}

void emit_absolute_jump(void *destination, const void *target) {
  // ldr x17, #8; br x17; .quad target
  const std::uint32_t instructions[2] = {0x58000051u, 0xd61f0220u};
  std::memcpy(destination, instructions, sizeof(instructions));
  std::memcpy(static_cast<std::uint8_t *>(destination) + 8, &target, sizeof(target));
}

bool make_page_writable(void *address, int protection) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const auto raw = reinterpret_cast<std::uintptr_t>(address);
  const auto page = raw & ~static_cast<std::uintptr_t>(page_size - 1);
  return mprotect(reinterpret_cast<void *>(page), static_cast<std::size_t>(page_size), protection) == 0;
}

bool process_uses_ndk_translation() {
  FILE *maps = std::fopen("/proc/self/maps", "r");
  if (maps == nullptr) {
    return false;
  }
  bool found = false;
  char line[1024] = {};
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    if (std::strstr(line, "libndk_translation.so") != nullptr) {
      found = true;
      break;
    }
  }
  std::fclose(maps);
  return found;
}

bool install_translated_sha256_compatibility_shim(std::uintptr_t libg_base) {
  if (!process_uses_ndk_translation()) {
    LOGI("translated SHA256 compatibility shim not required");
    return true;
  }
  auto *target = reinterpret_cast<std::uint32_t *>(libg_base + kTranslatedSha256InitializerCallOffset);
  std::uint32_t current = 0;
  std::memcpy(&current, target, sizeof(current));
  if (current == kAarch64Nop) {
    LOGI("translated SHA256 compatibility shim already installed");
    return true;
  }
  if (current != kTranslatedSha256InitializerCall) {
    LOGE("translated SHA256 initializer call mismatch at +0x%zx: 0x%08x",
         static_cast<std::size_t>(kTranslatedSha256InitializerCallOffset), current);
    return false;
  }
  if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
    LOGE("translated SHA256 compatibility mprotect RWX failed");
    return false;
  }
  std::memcpy(target, &kAarch64Nop, sizeof(kAarch64Nop));
  __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target) + sizeof(kAarch64Nop));
  make_page_writable(target, PROT_READ | PROT_EXEC);
  LOGI("translated SHA256 compatibility shim installed at libg+0x%zx",
       static_cast<std::size_t>(kTranslatedSha256InitializerCallOffset));
  return true;
}

extern "C" std::int32_t cr_render_gate_swap_buffers(void *display, void *surface) {
  process_native_render_request_on_frame_thread();
  dismiss_native_dialogs_if_needed();
  pthread_mutex_lock(&g_render_gate_mutex);
  while (g_render_suppressed) {
    pthread_cond_wait(&g_render_gate_cond, &g_render_gate_mutex);
  }
  EglSwapBuffers original = g_original_egl_swap_buffers;
  pthread_mutex_unlock(&g_render_gate_mutex);
  return original == nullptr ? 1 : original(display, surface);
}

bool write_egl_swap_buffers_got(void *target) {
  if (g_egl_swap_buffers_got == nullptr || target == nullptr ||
      (reinterpret_cast<std::uintptr_t>(g_egl_swap_buffers_got) & 0x7U) != 0) {
    return false;
  }
  if (!make_page_writable(g_egl_swap_buffers_got, PROT_READ | PROT_WRITE)) {
    return false;
  }
  __atomic_store_n(g_egl_swap_buffers_got, target, __ATOMIC_SEQ_CST);
  make_page_writable(g_egl_swap_buffers_got, PROT_READ);
  return true;
}

bool set_render_suppressed(bool suppress) {
  pthread_mutex_lock(&g_render_gate_mutex);
  if (g_render_suppressed == suppress) {
    pthread_mutex_unlock(&g_render_gate_mutex);
    return true;
  }
  if (g_egl_swap_buffers_got == nullptr || g_original_egl_swap_buffers == nullptr) {
    pthread_mutex_unlock(&g_render_gate_mutex);
    return false;
  }
  g_render_suppressed = suppress;
  if (!suppress) {
    pthread_cond_broadcast(&g_render_gate_cond);
  }
  pthread_mutex_unlock(&g_render_gate_mutex);
  LOGI("render gate changed: suppressed=%d", suppress ? 1 : 0);
  return true;
}

struct alignas(8) LibgString {
  std::int32_t cached_unicode_length = -1;
  std::int32_t byte_length = 0;
  union {
    char inline_bytes[8];
    const char *pointer;
  } storage = {};
};

bool initialize_libg_string(const char *text, LibgString *result) {
  if (text == nullptr || result == nullptr) {
    return false;
  }
  const std::size_t size = std::strlen(text);
  if (size > static_cast<std::size_t>(INT32_MAX)) {
    return false;
  }
  *result = {};
  result->cached_unicode_length = -1;
  result->byte_length = static_cast<std::int32_t>(size);
  if (size < sizeof(result->storage.inline_bytes)) {
    std::memcpy(result->storage.inline_bytes, text, size);
    result->storage.inline_bytes[size] = '\0';
  } else {
    result->storage.pointer = text;
  }
  return true;
}

void *parse_json_text(const char *json_text) {
  if (g_json_parse == nullptr || json_text == nullptr) {
    return nullptr;
  }
  LibgString input;
  if (!initialize_libg_string(json_text, &input) || input.byte_length < 2) {
    return nullptr;
  }
  return g_json_parse(&input, nullptr, 100);
}

bool extract_location_id(const char *json_text, std::int32_t *location_id_out) {
  if (json_text == nullptr || location_id_out == nullptr) {
    return false;
  }
  constexpr const char *kLocationKey = "\"location\":";
  const char *value = std::strstr(json_text, kLocationKey);
  if (value == nullptr) {
    return false;
  }
  value += std::strlen(kLocationKey);
  errno = 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || parsed <= 0 || parsed > INT32_MAX) {
    return false;
  }
  *location_id_out = static_cast<std::int32_t>(parsed);
  return true;
}

void process_native_render_request_on_frame_thread() {
  if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::NativeRender)) {
    return;
  }

  suppress_offline_connection_error_if_needed();

  char replay_json[kMaxReplayConfigBytes] = {};
  std::uint64_t sequence = 0;
  pthread_mutex_lock(&g_control_mutex);
  if (g_native_render_request_sequence != g_native_render_processed_sequence) {
    sequence = g_native_render_request_sequence;
    std::memcpy(replay_json, g_pending_native_render_json, sizeof(replay_json));
    replay_json[sizeof(replay_json) - 1] = '\0';
  }
  pthread_mutex_unlock(&g_control_mutex);
  if (sequence == 0 || replay_json[0] == '\0') {
    return;
  }

  bool succeeded = false;
  void *replay_manager = nullptr;
  if (inspect_content_runtime() && g_replay_manager_getter != nullptr && g_libg_string_from_utf8 != nullptr &&
      g_game_state_new != nullptr && g_original_replay_json_setter != nullptr) {
    replay_manager = g_replay_manager_getter();
    const std::size_t json_size = std::strlen(replay_json);
    if (replay_manager != nullptr && json_size < static_cast<std::size_t>(INT32_MAX)) {
      void *native_string = g_game_state_new(0x10);
      if (native_string != nullptr) {
        g_libg_string_from_utf8(native_string, replay_json, static_cast<std::int32_t>(json_size));
        LOGI("submitting offline native-render config #%llu replayManager=%p bytes=%zu tid=%ld",
             static_cast<unsigned long long>(sequence), replay_manager, json_size,
             static_cast<long>(syscall(__NR_gettid)));
        void *prior_controller = nullptr;
        std::memcpy(&prior_controller, static_cast<std::uint8_t *>(replay_manager) + 0x20, sizeof(prior_controller));
        void *const known_native_controller = g_native_render_controller.load(std::memory_order_acquire);
        if (prior_controller != nullptr && prior_controller != known_native_controller) {
          // At the offline login/error screen this slot can contain
          // a non-replay state object. The stock setter assumes any
          // non-null value is a ReplayBattleController and reads its
          // manager at +0x90. Start the replay subsystem from its
          // normal idle/null state so it creates the real controller
          // instead of dereferencing that unrelated screen object.
          void *empty = nullptr;
          std::memcpy(static_cast<std::uint8_t *>(replay_manager) + 0x20, &empty, sizeof(empty));
          LOGI("cleared incompatible offline replay slot: replayManager=%p prior=%p", replay_manager, prior_controller);
        }
        g_native_render_submitted_sequence.store(sequence, std::memory_order_release);
        g_original_replay_json_setter(replay_manager, native_string);
        succeeded = true;
      }
    }
  }

  g_native_render_replay_manager.store(replay_manager, std::memory_order_release);
  pthread_mutex_lock(&g_control_mutex);
  if (g_native_render_processed_sequence < sequence) {
    g_last_native_render_request_succeeded = succeeded;
    g_native_render_processed_sequence = sequence;
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
  if (!succeeded) {
    LOGE("offline native-render config #%llu could not be submitted", static_cast<unsigned long long>(sequence));
  }
}

void dismiss_native_dialogs_if_needed() {
  if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::NativeRender) ||
      g_java_vm == nullptr || g_native_dialog_manager_class == nullptr || g_native_dialog_dismiss_all == nullptr) {
    return;
  }

  timespec now = {};
  clock_gettime(CLOCK_MONOTONIC, &now);
  const std::int64_t now_ns = static_cast<std::int64_t>(now.tv_sec) * 1000000000LL + now.tv_nsec;
  std::int64_t previous = g_last_dialog_dismiss_ns.load(std::memory_order_relaxed);
  if (now_ns - previous < 250000000LL || !g_last_dialog_dismiss_ns.compare_exchange_strong(
                                             previous, now_ns, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    return;
  }

  JNIEnv *env = nullptr;
  bool attached_here = false;
  const jint environment_status = g_java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
  if (environment_status == JNI_EDETACHED) {
    if (g_java_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
      return;
    }
    attached_here = true;
  } else if (environment_status != JNI_OK || env == nullptr) {
    return;
  }

  env->CallStaticVoidMethod(g_native_dialog_manager_class, g_native_dialog_dismiss_all);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    LOGE("NativeDialogManager.nativeDialogDismissAll raised an exception");
  }
  if (attached_here) {
    g_java_vm->DetachCurrentThread();
  }
}

void *lookup_json_member(void *json_object, const char *key) {
  if (json_object == nullptr || key == nullptr || g_json_member_find == nullptr) {
    return nullptr;
  }
  LibgString name;
  if (!initialize_libg_string(key, &name)) {
    return nullptr;
  }
  void *entry = g_json_member_find(static_cast<std::uint8_t *>(json_object) + 0x18, &name, 1);
  if (entry == nullptr) {
    return nullptr;
  }
  void *value = nullptr;
  std::memcpy(&value, static_cast<std::uint8_t *>(entry) + 0x20, sizeof(value));
  return value;
}

bool inspect_pointer_array(const void *array, std::int32_t *count_out = nullptr) {
  if (array == nullptr) {
    return false;
  }
  const auto *bytes = static_cast<const std::uint8_t *>(array);
  void *data = nullptr;
  std::int32_t capacity = 0;
  std::int32_t count = 0;
  std::memcpy(&data, bytes + 0x08, sizeof(data));
  std::memcpy(&capacity, bytes + 0x10, sizeof(capacity));
  std::memcpy(&count, bytes + 0x14, sizeof(count));
  const bool plausible =
      capacity >= 0 && count >= 0 && count <= capacity && capacity <= 100000 && (count == 0 || data != nullptr);
  if (plausible && count_out != nullptr) {
    *count_out = count;
  }
  return plausible;
}

void release_json_object(void *object) {
  if (object == nullptr) {
    return;
  }
  auto **vtable = *reinterpret_cast<void ***>(object);
  if (vtable == nullptr || vtable[1] == nullptr) {
    return;
  }
  reinterpret_cast<void (*)(void *)>(vtable[1])(object);
}

bool validate_empty_match_streams(const char *json_text) {
  void *root = parse_json_text(json_text);
  if (root == nullptr) {
    return false;
  }
  void *const battle = lookup_json_member(root, "battle");
  void *const commands = lookup_json_member(root, "cmd");
  void *const events = lookup_json_member(root, "evt");
  std::int32_t command_count = -1;
  std::int32_t event_count = -1;
  const bool valid = battle != nullptr && inspect_pointer_array(commands, &command_count) && command_count == 0 &&
                     inspect_pointer_array(events, &event_count) && event_count == 0;
  release_json_object(root);
  return valid;
}

bool parse_logic_command(void *manager, const char *command_json, void **logic_command_out) {
  if (manager == nullptr || command_json == nullptr || g_json_parse == nullptr || g_logic_command_parse == nullptr ||
      logic_command_out == nullptr) {
    return false;
  }
  *logic_command_out = nullptr;
  const std::size_t command_size = std::strlen(command_json);
  if (command_size < 2 || command_size > static_cast<std::size_t>(INT32_MAX)) {
    return false;
  }

  void *json_object = parse_json_text(command_json);
  if (json_object == nullptr) {
    LOGE("native JSON parser rejected injected command");
    return false;
  }

  void *command_context = nullptr;
  std::memcpy(&command_context, static_cast<std::uint8_t *>(manager) + 0x208, sizeof(command_context));
  void *logic_command = nullptr;
  if (command_context != nullptr) {
    cr_call_logic_command_parse(json_object, command_context, &logic_command, g_logic_command_parse);
  }
  release_json_object(json_object);
  *logic_command_out = logic_command;
  return logic_command != nullptr;
}

bool inject_logic_command(void *manager, const char *command_json) {
  if (manager == nullptr || g_logic_command_append == nullptr) {
    return false;
  }
  void *command_queue = nullptr;
  std::memcpy(&command_queue, static_cast<std::uint8_t *>(manager) + 0x38, sizeof(command_queue));
  void *logic_command = nullptr;
  const bool parsed = command_queue != nullptr && parse_logic_command(manager, command_json, &logic_command);
  const bool succeeded = parsed && logic_command != nullptr;
  if (succeeded) {
    // LogicCommandQueue::append consumes the command owner.
    g_logic_command_append(command_queue, &logic_command);
  }
  void *command_context = nullptr;
  std::memcpy(&command_context, static_cast<std::uint8_t *>(manager) + 0x208, sizeof(command_context));
  LOGI("injected command parse=%s manager=%p queue=%p context=%p", succeeded ? "success" : "failure", manager,
       command_queue, command_context);
  return succeeded;
}



bool create_state_snapshot(void *manager, void **wrapper_out, std::uint64_t *digest, std::uint32_t *serialized_bytes) {
  if (manager == nullptr || wrapper_out == nullptr || digest == nullptr || serialized_bytes == nullptr ||
      g_state_snapshot == nullptr || g_state_snapshot_delete == nullptr) {
    return false;
  }
  *wrapper_out = nullptr;
  void *snapshot = nullptr;
  cr_call_state_snapshot(manager, &snapshot, g_state_snapshot);
  if (snapshot == nullptr) {
    return false;
  }

  // The returned 0x90-byte object is a byte-stream wrapper. Hash its
  // serialized payload, not the wrapper itself: the wrapper contains heap
  // pointers and therefore changes after every reset.
  std::int32_t payload_size = 0;
  const std::uint8_t *payload = nullptr;
  std::memcpy(&payload_size, static_cast<const std::uint8_t *>(snapshot) + 0x1c, sizeof(payload_size));
  std::memcpy(&payload, static_cast<const std::uint8_t *>(snapshot) + 0x40, sizeof(payload));
  constexpr std::int32_t kMaxSnapshotBytes = 32 * 1024 * 1024;
  if (payload_size < 0 || payload_size > kMaxSnapshotBytes || (payload_size != 0 && payload == nullptr)) {
    g_state_snapshot_delete(snapshot);
    return false;
  }

  constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  std::uint64_t value = kFnvOffset;
  for (std::int32_t index = 0; index < payload_size; ++index) {
    value ^= payload[index];
    value *= kFnvPrime;
  }
  *digest = value;
  *serialized_bytes = static_cast<std::uint32_t>(payload_size);
  *wrapper_out = snapshot;
  return true;
}

bool capture_state_snapshot(void *manager, std::uint64_t *digest, std::uint32_t *serialized_bytes) {
  void *snapshot = nullptr;
  const bool succeeded = create_state_snapshot(manager, &snapshot, digest, serialized_bytes);
  if (snapshot != nullptr && g_state_snapshot_delete != nullptr) {
    g_state_snapshot_delete(snapshot);
  }
  return succeeded;
}

bool send_all(int socket_fd, const char *data, std::size_t size) {
  while (size != 0) {
    const ssize_t sent = send(socket_fd, data, size, MSG_NOSIGNAL);
    if (sent <= 0) {
      return false;
    }
    data += sent;
    size -= static_cast<std::size_t>(sent);
  }
  return true;
}

template <typename T> T read_object_field(const void *object, std::size_t offset) {
  T value{};
  std::memcpy(&value, static_cast<const std::uint8_t *>(object) + offset, sizeof(value));
  return value;
}

bool libg_address_has_segment_flags(const void *address, std::size_t size, ElfW(Word) required_flags) {
  if (address == nullptr || size == 0 || g_libg_base == 0 || g_libg_phdr == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(address);
  if (begin > UINTPTR_MAX - size) {
    return false;
  }
  const std::uintptr_t end = begin + size;
  for (ElfW(Half) index = 0; index < g_libg_phnum; ++index) {
    const ElfW(Phdr) &header = g_libg_phdr[index];
    if (header.p_type != PT_LOAD || (header.p_flags & required_flags) != required_flags || header.p_memsz == 0) {
      continue;
    }
    if (static_cast<std::uintptr_t>(header.p_vaddr) > UINTPTR_MAX - g_libg_base) {
      continue;
    }
    const std::uintptr_t segment_begin = g_libg_base + static_cast<std::uintptr_t>(header.p_vaddr);
    if (static_cast<std::uintptr_t>(header.p_memsz) > UINTPTR_MAX - segment_begin) {
      continue;
    }
    const std::uintptr_t segment_end = segment_begin + static_cast<std::uintptr_t>(header.p_memsz);
    if (begin >= segment_begin && end <= segment_end) {
      return true;
    }
  }
  return false;
}

struct ProcessReadableRange {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

// Rich telemetry asks whether many engine-owned pointers are safe to inspect.
// Reopening and reparsing /proc/self/maps for every query made that check the
// dominant per-tick cost once phase/remaining-runtime hooks were enabled.
//
// The map is process-global and changes rarely compared with native battle
// ticks.  Cache only positive readable mappings and refresh periodically.  A
// cache miss first uses mincore as a cheap mapped-page discriminator: genuinely
// unmapped pointers remain fail-closed without reparsing, while a newly mapped
// page forces an immediate authoritative /proc/self/maps refresh.  Hot valid
// pointers become an in-memory binary search.  Positive entries are bounded by
// the short refresh interval; the original implementation already had an
// unavoidable unmap race between closing /proc/self/maps and dereferencing the
// checked pointer.
constexpr std::size_t kProcessReadableRangeCapacity = 4096;
constexpr std::int64_t kProcessReadableRefreshIntervalNs = 250000000LL;
ProcessReadableRange g_process_readable_ranges[kProcessReadableRangeCapacity] = {};
std::size_t g_process_readable_range_count = 0;
bool g_process_readable_cache_complete = false;
pthread_rwlock_t g_process_readable_cache_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_mutex_t g_process_readable_refresh_mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic<std::int64_t> g_process_readable_refreshed_ns{0};
std::atomic<std::uint64_t> g_process_readable_cache_hits{0};
std::atomic<std::uint64_t> g_process_readable_cache_misses{0};
std::atomic<std::uint64_t> g_process_readable_cache_refreshes{0};

std::int64_t process_monotonic_now_ns() {
  timespec now{};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return 0;
  }
  return static_cast<std::int64_t>(now.tv_sec) * 1000000000LL + static_cast<std::int64_t>(now.tv_nsec);
}

bool process_readable_cache_contains(std::uintptr_t begin, std::uintptr_t end) {
  pthread_rwlock_rdlock(&g_process_readable_cache_lock);
  std::size_t low = 0;
  std::size_t high = g_process_readable_range_count;
  while (low < high) {
    const std::size_t middle = low + (high - low) / 2;
    if (g_process_readable_ranges[middle].end <= begin) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const bool readable = low < g_process_readable_range_count && begin >= g_process_readable_ranges[low].begin &&
                        end <= g_process_readable_ranges[low].end;
  pthread_rwlock_unlock(&g_process_readable_cache_lock);
  return readable;
}

bool refresh_process_readable_cache(std::uintptr_t requested_begin, std::uintptr_t requested_end) {
  pthread_mutex_lock(&g_process_readable_refresh_mutex);

  FILE *maps = std::fopen("/proc/self/maps", "r");
  if (maps == nullptr) {
    pthread_mutex_unlock(&g_process_readable_refresh_mutex);
    return false;
  }

  ProcessReadableRange parsed[kProcessReadableRangeCapacity] = {};
  std::size_t parsed_count = 0;
  bool complete = true;
  bool requested_readable = false;
  char line[4096] = {};
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long long raw_begin = 0;
    unsigned long long raw_end = 0;
    char permissions[5] = {};
    if (std::sscanf(line, "%llx-%llx %4s", &raw_begin, &raw_end, permissions) != 3 || permissions[0] != 'r' ||
        raw_begin >= raw_end) {
      continue;
    }
    const std::uintptr_t mapping_begin = static_cast<std::uintptr_t>(raw_begin);
    const std::uintptr_t mapping_end = static_cast<std::uintptr_t>(raw_end);
    if (requested_begin >= mapping_begin && requested_end <= mapping_end) {
      requested_readable = true;
    }
    if (parsed_count != 0 && parsed[parsed_count - 1].end >= mapping_begin) {
      if (mapping_end > parsed[parsed_count - 1].end) {
        parsed[parsed_count - 1].end = mapping_end;
      }
      continue;
    }
    if (parsed_count == kProcessReadableRangeCapacity) {
      complete = false;
      continue;
    }
    parsed[parsed_count++] = {mapping_begin, mapping_end};
  }
  std::fclose(maps);

  pthread_rwlock_wrlock(&g_process_readable_cache_lock);
  std::memcpy(g_process_readable_ranges, parsed, parsed_count * sizeof(parsed[0]));
  g_process_readable_range_count = parsed_count;
  g_process_readable_cache_complete = complete;
  pthread_rwlock_unlock(&g_process_readable_cache_lock);

  g_process_readable_refreshed_ns.store(process_monotonic_now_ns(), std::memory_order_release);
  g_process_readable_cache_refreshes.fetch_add(1, std::memory_order_relaxed);
  pthread_mutex_unlock(&g_process_readable_refresh_mutex);
  return requested_readable;
}

bool process_range_is_readable(const void *address, std::size_t size) {
  if (address == nullptr || size == 0) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(address);
  if (begin > UINTPTR_MAX - size) {
    return false;
  }
  const std::uintptr_t end = begin + size;
  const std::int64_t now_ns = process_monotonic_now_ns();
  const std::int64_t refreshed_ns = g_process_readable_refreshed_ns.load(std::memory_order_acquire);
  if (refreshed_ns != 0 && now_ns >= refreshed_ns && now_ns - refreshed_ns < kProcessReadableRefreshIntervalNs &&
      process_readable_cache_contains(begin, end)) {
    g_process_readable_cache_hits.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  if (refreshed_ns != 0 && now_ns >= refreshed_ns && now_ns - refreshed_ns < kProcessReadableRefreshIntervalNs) {
    // Avoid turning repeated invalid-pointer probes into repeated maps
    // scans. If the first page is currently mapped, refresh immediately so
    // a mapping created after the snapshot is not hidden until the timer.
    pthread_rwlock_rdlock(&g_process_readable_cache_lock);
    const bool complete = g_process_readable_cache_complete;
    pthread_rwlock_unlock(&g_process_readable_cache_lock);
    if (complete) {
      static const long page_size = sysconf(_SC_PAGESIZE);
      if (page_size <= 0) {
        g_process_readable_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      const std::uintptr_t page_mask = static_cast<std::uintptr_t>(page_size - 1);
      void *const page = reinterpret_cast<void *>(begin & ~page_mask);
      unsigned char residency = 0;
      if (mincore(page, static_cast<std::size_t>(page_size), &residency) != 0) {
        g_process_readable_cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }
  }

  const bool readable = refresh_process_readable_cache(begin, end);
  if (readable) {
    g_process_readable_cache_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_process_readable_cache_misses.fetch_add(1, std::memory_order_relaxed);
  }
  return readable;
}

struct NativeEntityIdentityView {
  std::uint32_t native_object_id = 0;
  std::int32_t owner = 0;
  std::int32_t object_index = 0;
  std::int32_t secondary_index = 0;
};

bool read_native_entity_identity(const void *object, NativeEntityIdentityView *result) {
  if (object == nullptr || result == nullptr || !process_range_is_readable(object, 0x9c)) {
    return false;
  }
  *result = {};
  result->native_object_id = read_object_field<std::uint32_t>(object, 0x08);
  if (result->native_object_id == 0) {
    return false;
  }
  result->owner = read_object_field<std::int32_t>(object, 0x78);
  result->object_index = read_object_field<std::int32_t>(object, 0x94);
  result->secondary_index = read_object_field<std::int32_t>(object, 0x98);
  return true;
}

bool native_object_id_is_unique_in_vector(const void *object, void *const *objects, std::int32_t object_count,
                                          std::uint32_t native_object_id) {
  if (object == nullptr || objects == nullptr || object_count < 0 || native_object_id == 0) {
    return false;
  }
  std::int32_t pointer_matches = 0;
  std::int32_t id_matches = 0;
  for (std::int32_t slot = 0; slot < object_count; ++slot) {
    const void *const candidate = objects[slot];
    if (candidate == nullptr) {
      continue;
    }
    if (!process_range_is_readable(candidate, 0x0c)) {
      return false;
    }
    const std::uint32_t candidate_id = read_object_field<std::uint32_t>(candidate, 0x08);
    if (candidate_id == 0) {
      return false;
    }
    pointer_matches += candidate == object ? 1 : 0;
    id_matches += candidate_id == native_object_id ? 1 : 0;
  }
  return pointer_matches == 1 && id_matches == 1;
}

bool derive_native_entity_key(std::uint32_t native_object_id, std::int32_t object_index, std::int32_t secondary_index,
                              std::int32_t *key_object_index, std::int64_t *key_secondary_index) {
  if (native_object_id == 0 || key_object_index == nullptr || key_secondary_index == nullptr) {
    return false;
  }
  (void)object_index;
  (void)secondary_index;
  *key_object_index = kNativeObjectIdEntityKeyTag;
  *key_secondary_index = native_object_id;
  return true;
}

bool format_native_entity_key_json(char *destination, std::size_t destination_size, std::uint32_t native_object_id,
                                   std::int32_t owner, std::int32_t object_index, std::int32_t secondary_index) {
  std::int32_t key_object_index = 0;
  std::int64_t key_secondary_index = 0;
  if (destination == nullptr || destination_size == 0 ||
      !derive_native_entity_key(native_object_id, object_index, secondary_index, &key_object_index,
                                &key_secondary_index)) {
    return false;
  }
  const int written = std::snprintf(destination, destination_size, "[%d,%d,%lld]", owner, key_object_index,
                                    static_cast<long long>(key_secondary_index));
  return written >= 0 && static_cast<std::size_t>(written) < destination_size;
}

bool suppress_offline_connection_error_if_needed() {
  if (!g_offline_control_session_active.load(std::memory_order_acquire) || g_libg_base == 0) {
    return false;
  }
  const auto game_app_getter = reinterpret_cast<NoArgPointer>(g_libg_base + kGameAppGetterOffset);
  void *const game_app = game_app_getter();
  if (game_app == nullptr) {
    return false;
  }

  const std::int32_t event = read_object_field<std::int32_t>(game_app, kGameAppPendingEventOffset);
  // Offline cold start deliberately has no server. Event 2 and 3 share the
  // stock TID_ERROR_POP_UP_CONNECTION_FAILED_TITLE path. Leaving either
  // pending freezes ReplayBattleController's own clock; dispatching it via
  // GameApp's generic handler can tear down the locally-owned battle scene.
  if (event != 2 && event != 3) {
    return false;
  }

  const std::uint16_t flags = read_object_field<std::uint16_t>(game_app, kGameAppPendingEventFlagsOffset);
  // Match the field widths and order used by GameApp's stock pending-event
  // cleanup, but intentionally do not dispatch the connection failure into
  // the application state machine. That dispatch marks +0x243 and causes a
  // full client reload on the next main update.
  const std::uint16_t cleared_flags = 0;
  const std::uint64_t cleared_event = 0;
  std::memcpy(static_cast<std::uint8_t *>(game_app) + kGameAppPendingEventOffset, &cleared_event,
              sizeof(cleared_event));
  std::memcpy(static_cast<std::uint8_t *>(game_app) + kGameAppPendingEventFlagsOffset, &cleared_flags,
              sizeof(cleared_flags));
  const std::uint64_t count = g_native_offline_connection_errors_suppressed.fetch_add(1, std::memory_order_relaxed) + 1;
  if (count <= 3) {
    LOGI("suppressed offline connection failure #%llu gameApp=%p event=%d flags=0x%x",
         static_cast<unsigned long long>(count), game_app, event, static_cast<unsigned int>(flags));
  }
  return true;
}

bool append_json(char *response, std::size_t response_size, std::size_t *used, const char *format, ...) {
  if (response == nullptr || used == nullptr || format == nullptr || *used >= response_size) {
    return false;
  }
  va_list arguments;
  va_start(arguments, format);
  const int written = std::vsnprintf(response + *used, response_size - *used, format, arguments);
  va_end(arguments);
  if (written < 0 || static_cast<std::size_t>(written) >= response_size - *used) {
    return false;
  }
  *used += static_cast<std::size_t>(written);
  return true;
}

bool append_native_entity_key_json(char *response, std::size_t response_size, std::size_t *used,
                                   std::uint32_t native_object_id, std::int32_t owner, std::int32_t object_index,
                                   std::int32_t secondary_index) {
  JsonWriter json(response, response_size, used);
  char entity_key[96] = {};
  return format_native_entity_key_json(entity_key, sizeof(entity_key), native_object_id, owner, object_index,
                                       secondary_index) &&
         json.append("%s", entity_key);
}

bool valid_utf8_bytes(const char *text, std::size_t size) {
  if (text == nullptr) {
    return false;
  }
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(text);
  std::size_t index = 0;
  while (index < size) {
    const std::uint8_t lead = bytes[index];
    if (lead <= 0x7f) {
      ++index;
      continue;
    }
    if (lead >= 0xc2 && lead <= 0xdf) {
      if (index + 1 >= size || (bytes[index + 1] & 0xc0) != 0x80) {
        return false;
      }
      index += 2;
      continue;
    }
    if (lead >= 0xe0 && lead <= 0xef) {
      if (index + 2 >= size || (bytes[index + 2] & 0xc0) != 0x80) {
        return false;
      }
      const std::uint8_t second = bytes[index + 1];
      if ((lead == 0xe0 && (second < 0xa0 || second > 0xbf)) || (lead == 0xed && (second < 0x80 || second > 0x9f)) ||
          (lead != 0xe0 && lead != 0xed && (second & 0xc0) != 0x80)) {
        return false;
      }
      index += 3;
      continue;
    }
    if (lead >= 0xf0 && lead <= 0xf4) {
      if (index + 3 >= size || (bytes[index + 2] & 0xc0) != 0x80 || (bytes[index + 3] & 0xc0) != 0x80) {
        return false;
      }
      const std::uint8_t second = bytes[index + 1];
      if ((lead == 0xf0 && (second < 0x90 || second > 0xbf)) || (lead == 0xf4 && (second < 0x80 || second > 0x8f)) ||
          (lead != 0xf0 && lead != 0xf4 && (second & 0xc0) != 0x80)) {
        return false;
      }
      index += 4;
      continue;
    }
    return false;
  }
  return true;
}

bool append_json_utf8_string(char *response, std::size_t response_size, std::size_t *used, const char *text,
                             std::size_t size) {
  JsonWriter json(response, response_size, used);
  if (response == nullptr || used == nullptr || text == nullptr || !valid_utf8_bytes(text, size) ||
      !json.append("\"")) {
    return false;
  }
  constexpr char kHexDigits[] = "0123456789abcdef";
  for (std::size_t index = 0; index < size; ++index) {
    const std::uint8_t byte = static_cast<std::uint8_t>(text[index]);
    if (byte == '"' || byte == '\\') {
      if (*used + 2 >= response_size) {
        return false;
      }
      response[(*used)++] = '\\';
      response[(*used)++] = static_cast<char>(byte);
      response[*used] = '\0';
    } else if (byte < 0x20) {
      if (*used + 6 >= response_size) {
        return false;
      }
      response[(*used)++] = '\\';
      response[(*used)++] = 'u';
      response[(*used)++] = '0';
      response[(*used)++] = '0';
      response[(*used)++] = kHexDigits[byte >> 4];
      response[(*used)++] = kHexDigits[byte & 0x0f];
      response[*used] = '\0';
    } else {
      if (*used + 1 >= response_size) {
        return false;
      }
      response[(*used)++] = static_cast<char>(byte);
      response[*used] = '\0';
    }
  }
  return json.append("\"");
}

bool read_bounded_libg_string_utf8(const void *string_object, char *output, std::size_t output_size,
                                   std::size_t *byte_length_out) {
  constexpr std::int32_t kMaximumBytes = 128;
  if (string_object == nullptr || output == nullptr || output_size < 2 || byte_length_out == nullptr ||
      !process_range_is_readable(string_object, sizeof(LibgString))) {
    return false;
  }
  const std::int32_t byte_length = read_object_field<std::int32_t>(string_object, 0x04);
  if (byte_length <= 0 || byte_length > kMaximumBytes || static_cast<std::size_t>(byte_length) >= output_size) {
    return false;
  }
  const char *data = nullptr;
  if (byte_length < 8) {
    data = static_cast<const char *>(string_object) + 0x08;
  } else {
    data = read_object_field<const char *>(string_object, 0x08);
  }
  const std::size_t size = static_cast<std::size_t>(byte_length);
  if (!process_range_is_readable(data, size) || !valid_utf8_bytes(data, size)) {
    return false;
  }
  std::memcpy(output, data, size);
  output[size] = '\0';
  *byte_length_out = size;
  return true;
}

struct NativeVectorView {
  void *data = nullptr;
  std::int32_t capacity = 0;
  std::int32_t count = 0;
};

bool read_native_vector(const void *object, std::size_t offset, std::int32_t maximum_capacity,
                        NativeVectorView *result) {
  if (object == nullptr || result == nullptr || maximum_capacity < 0) {
    return false;
  }
  result->data = read_object_field<void *>(object, offset);
  result->capacity = read_object_field<std::int32_t>(object, offset + 0x08);
  result->count = read_object_field<std::int32_t>(object, offset + 0x0c);
  return result->count >= 0 && result->capacity >= result->count && result->capacity <= maximum_capacity &&
         (result->count == 0 || result->data != nullptr);
}

struct NativeCardSelection {
  void *card_data = nullptr;
  void *secondary_data = nullptr;
  std::uint32_t packed = 0;
  std::uint32_t padding = 0;
};

static_assert(sizeof(NativeCardSelection) == 24);

struct BattleResultView {
  bool readable = false;
  bool finalized = false;
  std::int32_t world_result_raw = -1;
  std::int32_t crowns_raw[kPlayerCount] = {};
};

bool read_hitpoints(const void *object, std::int32_t *hitpoints, std::int32_t *maximum_hitpoints);

bool read_native_crowns(void *world, std::int32_t *crowns_raw) {
  if (world == nullptr || crowns_raw == nullptr) {
    return false;
  }

  void *player_zero = read_object_field<void *>(world, 0xe0);
  void *object_manager = player_zero == nullptr ? nullptr : read_object_field<void *>(player_zero, 0x10);
  if (object_manager == nullptr) {
    return false;
  }
  void **objects = read_object_field<void **>(object_manager, 0x08);
  const std::int32_t capacity = read_object_field<std::int32_t>(object_manager, 0x10);
  const std::int32_t count = read_object_field<std::int32_t>(object_manager, 0x14);
  if (count < 0 || capacity < count || capacity > 100000 || (count != 0 && objects == nullptr)) {
    return false;
  }

  // v15 does not keep the live crown score in player+0x33c (that word stays
  // zero after a princess tower is destroyed). Crown score is derived from
  // the native tower graph: each missing princess tower is one crown and a
  // missing king tower is three. These are the fixed native 1v1 tower cells.
  constexpr std::int32_t kKingX = 9000;
  constexpr std::int32_t kLeftPrincessX = 3500;
  constexpr std::int32_t kRightPrincessX = 14500;
  constexpr std::int32_t kKingY[kPlayerCount] = {3000, 29000};
  constexpr std::int32_t kPrincessY[kPlayerCount] = {6500, 25500};
  bool king_alive[kPlayerCount] = {};
  bool left_princess_alive[kPlayerCount] = {};
  bool right_princess_alive[kPlayerCount] = {};
  std::int32_t recognized_towers = 0;
  for (std::int32_t index = 0; index < count; ++index) {
    void *object = objects[index];
    if (object == nullptr || read_object_field<std::int32_t>(object, 0xac) != -1) {
      continue;
    }
    const std::int32_t owner = read_object_field<std::int32_t>(object, 0x78);
    if (owner < 0 || owner >= kPlayerCount) {
      continue;
    }
    const std::int32_t position_x = read_object_field<std::int32_t>(object, 0x7c);
    const std::int32_t position_y = read_object_field<std::int32_t>(object, 0x80);
    std::int32_t hitpoints = 0;
    std::int32_t maximum_hitpoints = 0;
    if (!read_hitpoints(object, &hitpoints, &maximum_hitpoints)) {
      continue;
    }
    bool *tower_alive = nullptr;
    if (position_x == kKingX && position_y == kKingY[owner]) {
      tower_alive = &king_alive[owner];
    } else if (position_x == kLeftPrincessX && position_y == kPrincessY[owner]) {
      tower_alive = &left_princess_alive[owner];
    } else if (position_x == kRightPrincessX && position_y == kPrincessY[owner]) {
      tower_alive = &right_princess_alive[owner];
    }
    if (tower_alive != nullptr) {
      ++recognized_towers;
      *tower_alive = hitpoints > 0;
    }
  }
  if (recognized_towers == 0) {
    return false;
  }
  for (std::int32_t owner = 0; owner < kPlayerCount; ++owner) {
    const std::int32_t opponent = 1 - owner;
    crowns_raw[owner] = !king_alive[opponent] ? 3
                                              : static_cast<std::int32_t>(!left_princess_alive[opponent]) +
                                                    static_cast<std::int32_t>(!right_princess_alive[opponent]);
  }
  return true;
}

bool read_battle_result(void *world, BattleResultView *result) {
  if (world == nullptr || result == nullptr) {
    return false;
  }
  BattleResultView value;
  value.finalized = read_object_field<std::uint8_t>(world, 0x1e0) != 0;
  value.world_result_raw = read_object_field<std::int32_t>(world, 0x1b8);
  for (std::int32_t owner = 0; owner < kPlayerCount; ++owner) {
    if (read_object_field<void *>(world, 0xe0 + owner * sizeof(void *)) == nullptr) {
      return false;
    }
  }
  if (!read_native_crowns(world, value.crowns_raw)) {
    return false;
  }
  value.readable = true;
  *result = value;
  return true;
}

bool consume_state_snapshot_prelude(void *stream) {
  if (stream == nullptr) {
    return false;
  }
  void *vtable = read_object_field<void *>(stream, 0x00);
  if (vtable == nullptr) {
    return false;
  }
  auto read_byte = reinterpret_cast<ByteStreamReadValue>(read_object_field<void *>(vtable, 0x150));
  auto read_boolean = reinterpret_cast<ByteStreamReadValue>(read_object_field<void *>(vtable, 0xe0));
  if (read_byte == nullptr || read_boolean == nullptr) {
    return false;
  }

  // StateSnapshot's wrapper protocol precedes the raw GameState stream:
  // compression byte, wrapper-version byte, and has-full-state boolean.
  // The public engine restore path at 0xccea2c consumes these exact values
  // before handing the stream to 0x10634c0/0x106379c.
  const std::int32_t compression = read_byte(stream) & 0xff;
  const std::int32_t wrapper_version = read_byte(stream) & 0xff;
  const bool has_full_state = (read_boolean(stream) & 1) != 0;
  return compression == 0 && wrapper_version == 0 && has_full_state;
}

std::size_t count_stored_snapshots_locked() {
  std::size_t count = 0;
  for (const auto &snapshot : g_stored_snapshots) {
    if (snapshot.handle != 0 && snapshot.wrapper != nullptr) {
      ++count;
    }
  }
  return count;
}

const char *snapshot_error_message(SnapshotError error) {
  switch (error) {
  case SnapshotError::None:
    return "";
  case SnapshotError::TableFull:
    return "native snapshot handle table is full";
  case SnapshotError::CaptureFailed:
    return "native state snapshot capture failed";
  case SnapshotError::UnknownHandle:
    return "native snapshot handle does not exist";
  case SnapshotError::StaleHandle:
    return "native snapshot handle belongs to a different manager generation";
  case SnapshotError::RestoreUnavailable:
    return "native snapshot restore functions are unavailable";
  case SnapshotError::RestoreVerificationFailed:
    return "native snapshot restore verification failed";
  case SnapshotError::ManagerChanged:
    return "native manager changed while processing snapshot request";
  }
  return "unknown native snapshot error";
}

struct SnapshotManagerIdentity {
  RunnerMode mode = RunnerMode::Headless;
  void *manager = nullptr;
  std::uint64_t generation = 0;
  std::uint64_t config_revision = 0;
  std::uint64_t state_epoch = 0;
  std::uint64_t completed_steps = 0;
};

bool current_snapshot_identity_locked(RunnerMode mode, SnapshotManagerIdentity *result) {
  if (result == nullptr || g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(mode)) {
    return false;
  }

  SnapshotManagerIdentity identity;
  identity.mode = mode;
  if (mode == RunnerMode::Headless) {
    if (!g_native_worker_running || g_controlled_manager == nullptr || g_control_generation == 0) {
      return false;
    }
    identity.manager = g_controlled_manager;
    identity.generation = g_control_generation;
    identity.config_revision = g_config_processed_sequence;
    identity.state_epoch = g_state_epoch;
    identity.completed_steps = g_completed_steps;
  } else if (mode == RunnerMode::NativeRender) {
    identity.manager = g_native_render_manager.load(std::memory_order_acquire);
    identity.generation = g_native_render_loaded_sequence.load(std::memory_order_acquire);
    const std::uint64_t submitted = g_native_render_submitted_sequence.load(std::memory_order_acquire);
    if (identity.manager == nullptr || identity.generation == 0 || identity.generation != submitted ||
        identity.generation != g_native_render_request_sequence ||
        g_native_render_processed_sequence != g_native_render_request_sequence) {
      return false;
    }
    identity.config_revision = identity.generation;
    identity.state_epoch = g_native_render_state_epoch;
    identity.completed_steps = g_native_forced_steps.load(std::memory_order_acquire);
  } else {
    return false;
  }
  *result = identity;
  return true;
}

bool snapshot_identity_matches_locked(const SnapshotManagerIdentity &expected) {
  SnapshotManagerIdentity current;
  return current_snapshot_identity_locked(expected.mode, &current) && current.manager == expected.manager &&
         current.generation == expected.generation && current.config_revision == expected.config_revision;
}

StoredSnapshot *find_stored_snapshot_locked(std::uint64_t handle) {
  for (auto &snapshot : g_stored_snapshots) {
    if (snapshot.handle == handle && snapshot.wrapper != nullptr) {
      return &snapshot;
    }
  }
  return nullptr;
}

StoredSnapshot *find_free_snapshot_slot_locked() {
  for (auto &snapshot : g_stored_snapshots) {
    if (snapshot.handle == 0 && snapshot.wrapper == nullptr) {
      return &snapshot;
    }
  }
  return nullptr;
}

void complete_snapshot_operation_locked(std::uint64_t sequence, bool succeeded, SnapshotError error,
                                        const StoredSnapshot &result = {}) {
  g_last_snapshot_operation_succeeded = succeeded;
  g_last_snapshot_operation_error = error;
  g_last_snapshot_operation_result = result;
  g_snapshot_processed_sequence = sequence;
  g_snapshot_processing_sequence = 0;
  g_pending_snapshot_operation = SnapshotOperation::None;
  g_pending_snapshot_mode = RunnerMode::Headless;
  g_pending_snapshot_manager = nullptr;
  g_pending_snapshot_generation = 0;
  g_pending_snapshot_handle = 0;
  pthread_cond_broadcast(&g_control_cond);
}

void process_snapshot_operation_on_worker(void *manager, RunnerMode mode, std::uint64_t sequence,
                                          SnapshotOperation operation, std::uint64_t requested_handle) {
  if (operation == SnapshotOperation::Create) {
    StoredSnapshot metadata;
    StoredSnapshot *slot = nullptr;
    pthread_mutex_lock(&g_control_mutex);
    SnapshotManagerIdentity identity;
    if (!current_snapshot_identity_locked(mode, &identity) || identity.manager != manager) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::ManagerChanged);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }
    slot = find_free_snapshot_slot_locked();
    if (slot == nullptr) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::TableFull);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }
    metadata.handle = g_next_snapshot_handle++;
    if (metadata.handle == 0) {
      metadata.handle = g_next_snapshot_handle++;
    }
    metadata.mode = mode;
    metadata.generation = identity.generation;
    metadata.config_revision = identity.config_revision;
    metadata.state_epoch = identity.state_epoch;
    metadata.completed_steps = identity.completed_steps;
    metadata.tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
    metadata.tower_troop_runtime_count =
        copy_tower_troop_runtime_snapshot(identity.generation, identity.state_epoch, metadata.tick,
                                          metadata.tower_troop_runtimes, kMaxStoredTowerTroopRuntimes);
    pthread_mutex_unlock(&g_control_mutex);

    void *wrapper = nullptr;
    const bool captured = create_state_snapshot(manager, &wrapper, &metadata.digest, &metadata.serialized_bytes);
    metadata.wrapper = wrapper;
    const bool prepared_for_read = captured && wrapper != nullptr && g_byte_stream_flip_read != nullptr;
    if (prepared_for_read) {
      // StateSnapshot serializes into a write-mode ByteStream, where the
      // byte count lives at stream+0x14. Flip it once into read mode so
      // the count is retained at stream+0x18; subsequent 0x121220c
      // calls can then rewind the read cursor without erasing payload.
      g_byte_stream_flip_read(static_cast<std::uint8_t *>(wrapper) + 0x08);
    }

    bool stored = false;
    pthread_mutex_lock(&g_control_mutex);
    SnapshotManagerIdentity current;
    if (prepared_for_read && current_snapshot_identity_locked(mode, &current) && current.manager == manager &&
        current.generation == metadata.generation && current.config_revision == metadata.config_revision &&
        slot->handle == 0 && slot->wrapper == nullptr) {
      *slot = metadata;
      stored = true;
      complete_snapshot_operation_locked(sequence, true, SnapshotError::None, metadata);
    } else {
      complete_snapshot_operation_locked(
          sequence, false,
          captured
              ? (g_byte_stream_flip_read == nullptr ? SnapshotError::RestoreUnavailable : SnapshotError::ManagerChanged)
              : SnapshotError::CaptureFailed);
    }
    pthread_mutex_unlock(&g_control_mutex);
    if (!stored && wrapper != nullptr && g_state_snapshot_delete != nullptr) {
      g_state_snapshot_delete(wrapper);
    }
    return;
  }

  if (operation == SnapshotOperation::Restore) {
    StoredSnapshot saved;
    pthread_mutex_lock(&g_control_mutex);
    StoredSnapshot *snapshot = find_stored_snapshot_locked(requested_handle);
    if (snapshot == nullptr) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::UnknownHandle);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }
    SnapshotManagerIdentity identity;
    if (!current_snapshot_identity_locked(mode, &identity) || identity.manager != manager || snapshot->mode != mode ||
        snapshot->generation != identity.generation || snapshot->config_revision != identity.config_revision) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::StaleHandle);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }
    saved = *snapshot;
    if (mode == RunnerMode::Headless) {
      g_step_budget = 0;
    }
    pthread_mutex_unlock(&g_control_mutex);

    if (g_byte_stream_reset_read == nullptr || g_byte_stream_reset_checksum == nullptr ||
        g_game_state_deserialize_header == nullptr || g_game_state_deserialize_body == nullptr) {
      pthread_mutex_lock(&g_control_mutex);
      complete_snapshot_operation_locked(sequence, false, SnapshotError::RestoreUnavailable);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }

    void *const stream = static_cast<std::uint8_t *>(saved.wrapper) + 0x08;
    g_byte_stream_reset_read(stream);
    const bool valid_prelude = consume_state_snapshot_prelude(stream);
    if (valid_prelude) {
      g_game_state_deserialize_header(manager, stream);
      g_game_state_deserialize_body(manager, stream, 1, nullptr);
    }

    const std::int32_t restored_tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
    std::uint64_t verification_digest = 0;
    std::uint32_t verification_bytes = 0;
    const bool verified_snapshot = capture_state_snapshot(manager, &verification_digest, &verification_bytes);
    void *restored_world = read_object_field<void *>(manager, 0xa8);
    BattleResultView battle_result;
    const bool result_readable = read_battle_result(restored_world, &battle_result);
    const bool verified = valid_prelude && verified_snapshot && result_readable && restored_tick == saved.tick &&
                          verification_digest == saved.digest && verification_bytes == saved.serialized_bytes;

    pthread_mutex_lock(&g_control_mutex);
    SnapshotManagerIdentity current;
    if (!current_snapshot_identity_locked(mode, &current) || current.manager != manager ||
        current.generation != saved.generation || current.config_revision != saved.config_revision) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::ManagerChanged);
    } else if (mode == RunnerMode::Headless) {
      ++g_state_epoch;
      begin_runtime_telemetry_epoch(manager, g_control_generation, g_state_epoch);
      bind_combat_event_object_manager(manager);
      restore_tower_troop_runtime_snapshot(g_control_generation, g_state_epoch, restored_tick,
                                           saved.tower_troop_runtimes, saved.tower_troop_runtime_count);
      g_completed_steps = saved.completed_steps;
      g_last_controlled_tick = restored_tick;
      g_step_budget = 0;
      g_control_ended = result_readable && battle_result.finalized;
      StoredSnapshot result = saved;
      result.state_epoch = g_state_epoch;
      complete_snapshot_operation_locked(
          sequence, verified, verified ? SnapshotError::None : SnapshotError::RestoreVerificationFailed, result);
    } else {
      ++g_native_render_state_epoch;
      if (g_native_render_state_epoch == 0) {
        ++g_native_render_state_epoch;
      }
      begin_runtime_telemetry_epoch(manager, saved.generation, g_native_render_state_epoch);
      bind_combat_event_object_manager(manager);
      restore_tower_troop_runtime_snapshot(saved.generation, g_native_render_state_epoch, restored_tick,
                                           saved.tower_troop_runtimes, saved.tower_troop_runtime_count);
      g_native_render_tick.store(restored_tick, std::memory_order_release);
      g_native_forced_steps.store(saved.completed_steps, std::memory_order_release);
      StoredSnapshot result = saved;
      result.state_epoch = g_native_render_state_epoch;
      complete_snapshot_operation_locked(
          sequence, verified, verified ? SnapshotError::None : SnapshotError::RestoreVerificationFailed, result);
    }
    pthread_mutex_unlock(&g_control_mutex);
    return;
  }

  if (operation == SnapshotOperation::Release) {
    StoredSnapshot released;
    pthread_mutex_lock(&g_control_mutex);
    StoredSnapshot *snapshot = find_stored_snapshot_locked(requested_handle);
    if (snapshot == nullptr) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::UnknownHandle);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }
    SnapshotManagerIdentity identity;
    if (!current_snapshot_identity_locked(mode, &identity) || identity.manager != manager || snapshot->mode != mode ||
        snapshot->generation != identity.generation || snapshot->config_revision != identity.config_revision) {
      complete_snapshot_operation_locked(sequence, false, SnapshotError::StaleHandle);
      pthread_mutex_unlock(&g_control_mutex);
      return;
    }
    released = *snapshot;
    *snapshot = {};
    pthread_mutex_unlock(&g_control_mutex);

    if (released.wrapper != nullptr && g_state_snapshot_delete != nullptr) {
      g_state_snapshot_delete(released.wrapper);
    }
    released.wrapper = nullptr;
    pthread_mutex_lock(&g_control_mutex);
    complete_snapshot_operation_locked(sequence, true, SnapshotError::None, released);
    pthread_mutex_unlock(&g_control_mutex);
    return;
  }

  pthread_mutex_lock(&g_control_mutex);
  complete_snapshot_operation_locked(sequence, false, SnapshotError::ManagerChanged);
  pthread_mutex_unlock(&g_control_mutex);
}

void process_pending_snapshot_operation_on_worker(void *manager, RunnerMode mode) {
  std::uint64_t sequence = 0;
  SnapshotOperation operation = SnapshotOperation::None;
  std::uint64_t handle = 0;

  pthread_mutex_lock(&g_control_mutex);
  if (g_snapshot_request_sequence != g_snapshot_processed_sequence && g_snapshot_processing_sequence == 0) {
    SnapshotManagerIdentity identity;
    if (g_pending_snapshot_mode != mode || !current_snapshot_identity_locked(mode, &identity) ||
        identity.manager != manager || g_pending_snapshot_manager != manager ||
        g_pending_snapshot_generation != identity.generation) {
      complete_snapshot_operation_locked(g_snapshot_request_sequence, false, SnapshotError::ManagerChanged);
    } else {
      sequence = g_snapshot_request_sequence;
      operation = g_pending_snapshot_operation;
      handle = g_pending_snapshot_handle;
      g_snapshot_processing_sequence = sequence;
    }
  }
  pthread_mutex_unlock(&g_control_mutex);

  if (sequence != 0) {
    process_snapshot_operation_on_worker(manager, mode, sequence, operation, handle);
  }
}

void release_all_stored_snapshots_on_worker() {
  void *wrappers[kMaxStoredSnapshots] = {};
  std::size_t wrapper_count = 0;
  pthread_mutex_lock(&g_control_mutex);
  for (auto &snapshot : g_stored_snapshots) {
    if (snapshot.wrapper != nullptr) {
      wrappers[wrapper_count++] = snapshot.wrapper;
    }
    snapshot = {};
  }
  if (g_snapshot_request_sequence != g_snapshot_processed_sequence) {
    complete_snapshot_operation_locked(g_snapshot_request_sequence, false, SnapshotError::ManagerChanged);
  }
  pthread_mutex_unlock(&g_control_mutex);
  if (g_state_snapshot_delete != nullptr) {
    for (std::size_t index = 0; index < wrapper_count; ++index) {
      g_state_snapshot_delete(wrappers[index]);
    }
  }
}

struct PlayerStateView {
  void *player = nullptr;
  void *identity = nullptr;
  std::uint32_t account_id_high = 0;
  std::uint32_t account_id_low = 0;
  std::int32_t elixir_raw = 0;
  std::int32_t crowns_raw = 0;
  void *deck = nullptr;
  NativeVectorView deck_slots;
  NativeVectorView hand;
  NativeVectorView cycle;
};

bool read_player_state(void *world, std::int32_t owner, PlayerStateView *result) {
  if (world == nullptr || result == nullptr || owner < 0 || owner >= kPlayerCount) {
    return false;
  }
  result->player = read_object_field<void *>(world, 0xe0 + owner * sizeof(void *));
  result->identity = read_object_field<void *>(world, 0x30 + owner * sizeof(void *));
  result->deck = read_object_field<void *>(world, 0x88 + owner * sizeof(void *));
  if (result->player == nullptr || result->identity == nullptr || result->deck == nullptr) {
    return false;
  }
  result->account_id_high = read_object_field<std::uint32_t>(result->identity, 0x00);
  result->account_id_low = read_object_field<std::uint32_t>(result->identity, 0x04);
  result->elixir_raw = read_object_field<std::int32_t>(result->player, 0x2f8);
  result->crowns_raw = 0;
  return read_native_vector(result->deck, 0x20, kMaxDeckCards, &result->deck_slots) &&
         read_native_vector(result->player, 0x220, kMaxHandCards, &result->hand) &&
         read_native_vector(result->player, 0x230, kMaxCycleCards, &result->cycle);
}

struct CardStateView {
  std::int32_t deck_slot = -1;
  void *battle_deck_slot = nullptr;
  void *card_data = nullptr;
  std::int32_t card_id = 0;
  std::int32_t command_card_id = 0;
  std::uint32_t card_parameter = 0;
  std::int32_t cost = -1;
  bool selection_valid = false;
};

bool read_card_state(const PlayerStateView &player, std::int32_t deck_slot, bool build_selection,
                     CardStateView *result) {
  if (result == nullptr || deck_slot < 0 || deck_slot >= player.deck_slots.count || player.deck_slots.data == nullptr) {
    return false;
  }
  auto **slots = static_cast<void **>(player.deck_slots.data);
  void *battle_deck_slot = slots[deck_slot];
  if (battle_deck_slot == nullptr) {
    return false;
  }
  void *card_data = read_object_field<void *>(battle_deck_slot, 0x10);
  if (card_data == nullptr) {
    return false;
  }

  result->deck_slot = deck_slot;
  result->battle_deck_slot = battle_deck_slot;
  result->card_data = card_data;
  result->card_id = read_object_field<std::int32_t>(card_data, 0x40);
  result->command_card_id = result->card_id;
  if (!build_selection) {
    return true;
  }
  if (g_card_selection_build == nullptr) {
    return false;
  }

  NativeCardSelection selection;
  cr_call_card_selection_build(battle_deck_slot, player.player, 0, &selection, g_card_selection_build);
  const std::int32_t encoded_slot = static_cast<std::int32_t>((selection.packed >> 22) & 0x3fU) - 1;
  if (selection.card_data != card_data || encoded_slot != deck_slot) {
    return false;
  }
  const std::int32_t form_code = static_cast<std::int32_t>(selection.packed & 0x0fU);
  if (form_code == 0 && selection.secondary_data != nullptr) {
    if (!process_range_is_readable(selection.secondary_data, 0x44)) {
      return false;
    }
    const std::int32_t secondary_card_id = read_object_field<std::int32_t>(selection.secondary_data, 0x40);
    if (secondary_card_id <= 0) {
      return false;
    }
    // Mirror keeps its root deck card in card_data while secondary_data is
    // the exact effective card expected in the command's `os` field.
    result->command_card_id = secondary_card_id;
  }
  result->card_parameter = selection.packed;
  result->cost = static_cast<std::int32_t>(selection.packed >> 28);
  result->selection_valid = true;
  return true;
}

struct NativeComponentSlotView {
  bool present = false;
  bool owner_backlink_valid = false;
  bool vtable_offset_valid = false;
  bool type_getter_offset_valid = false;
  bool type_valid = false;
  std::int32_t engine_type = -1;
  std::uintptr_t vtable_libg_offset = 0;
  std::uintptr_t type_getter_libg_offset = 0;
};

struct NativeComponentInventoryView {
  bool valid = false;
  std::int32_t capacity = 0;
  std::int32_t count = 0;
  NativeComponentSlotView slots[kMaxObjectComponents] = {};
};

using NativeComponentTypeGetter = std::int32_t (*)(const void *component);

bool read_native_component_inventory(const void *object, NativeComponentInventoryView *result) {
  if (object == nullptr || result == nullptr) {
    return false;
  }
  *result = {};
  void **components = read_object_field<void **>(object, 0x18);
  result->capacity = read_object_field<std::int32_t>(object, 0x20);
  result->count = read_object_field<std::int32_t>(object, 0x24);
  if (result->count < 0 || result->capacity < result->count || result->capacity > kMaxObjectComponents ||
      (result->count != 0 && components == nullptr)) {
    return false;
  }
  result->valid = true;

  // libg+0x113b194 computes LogicGameObject's component hash by iterating
  // this vector and invoking vtable+0x20. It asserts the returned engine
  // type is below 32 before setting the corresponding bit. Keep the call
  // fail-closed: the owner backlink and both libg segment permissions must
  // validate before executing the virtual getter.
  for (std::int32_t slot = 0; slot < result->count; ++slot) {
    NativeComponentSlotView &view = result->slots[slot];
    const void *component = components[slot];
    view.present = component != nullptr;
    if (component == nullptr) {
      continue;
    }
    view.owner_backlink_valid = read_object_field<const void *>(component, 0x08) == object;
    if (!view.owner_backlink_valid) {
      continue;
    }
    const void *vtable = read_object_field<const void *>(component, 0x00);
    if (!libg_address_has_segment_flags(vtable, 0x28, PF_R)) {
      continue;
    }
    view.vtable_offset_valid = true;
    view.vtable_libg_offset = reinterpret_cast<std::uintptr_t>(vtable) - g_libg_base;
    NativeComponentTypeGetter getter = read_object_field<NativeComponentTypeGetter>(vtable, 0x20);
    if (!libg_address_has_segment_flags(reinterpret_cast<const void *>(getter), sizeof(std::uint32_t), PF_X)) {
      continue;
    }
    view.type_getter_offset_valid = true;
    view.type_getter_libg_offset = reinterpret_cast<std::uintptr_t>(getter) - g_libg_base;
    const std::int32_t engine_type = getter(component);
    if (engine_type < 0 || engine_type >= 32) {
      continue;
    }
    view.type_valid = true;
    view.engine_type = engine_type;
  }
  return true;
}

struct NativeActiveEffectView {
  std::uint32_t buff_global_id = 0;
  std::int32_t remaining_ms = 0;
  std::size_t name_size = 0;
  char name[129] = {};
  bool source_entity_validated = false;
  bool source_entity_resolved = false;
  std::uint32_t source_native_object_id = 0;
  std::int32_t source_owner = 0;
  std::int32_t source_object_index = 0;
  std::int32_t source_secondary_index = 0;
};

struct NativeTypeThreeStateView {
  bool header_valid = false;
  bool active_effects_valid = false;
  std::int32_t capacity = 0;
  std::int32_t count = 0;
  std::int32_t invisible_count = 0;
  NativeActiveEffectView effects[kMaxActiveEffects] = {};
};

bool read_native_type_three_state(const void *object, const NativeComponentInventoryView &inventory,
                                  void *const *objects, std::int32_t object_count, NativeTypeThreeStateView *result) {
  if (object == nullptr || result == nullptr || object_count < 0 || (object_count != 0 && objects == nullptr)) {
    return false;
  }
  *result = {};
  if (!inventory.valid || inventory.count <= 3 || !inventory.slots[3].present ||
      !inventory.slots[3].owner_backlink_valid || !inventory.slots[3].type_valid ||
      inventory.slots[3].engine_type != 3) {
    return false;
  }
  void **components = read_object_field<void **>(object, 0x18);
  const void *component = components == nullptr ? nullptr : components[3];
  if (!process_range_is_readable(component, 0x38)) {
    return false;
  }

  void **entries = read_object_field<void **>(component, 0x18);
  result->capacity = read_object_field<std::int32_t>(component, 0x20);
  result->count = read_object_field<std::int32_t>(component, 0x24);
  result->invisible_count = read_object_field<std::int32_t>(component, 0x34);
  if (result->count < 0 || result->capacity < result->count || result->capacity > kMaxActiveEffects ||
      result->invisible_count < 0 || result->invisible_count > result->count ||
      (result->count != 0 && entries == nullptr) ||
      (result->count != 0 &&
       !process_range_is_readable(entries, static_cast<std::size_t>(result->count) * sizeof(void *)))) {
    return false;
  }
  result->header_valid = true;

  // libg+f595f0/f59910 encode/decode the type-3 T** vector. Each child is
  // 0x70 bytes (constructor/decode libg+f2023c/f205d8). Promote only the
  // exact Buff asset path: asset vtable libg+0x1882500 and its vtable+0x70
  // class-name method libg+d961f4 (static return "Buff"). Asset name and
  // global ID are the exact layouts used by libg+dc25ec/dc2570. Any child
  // failing these guards invalidates the whole activeEffects array rather
  // than silently omitting an engine entry.
  for (std::int32_t index = 0; index < result->count; ++index) {
    const void *entry = entries[index];
    if (!process_range_is_readable(entry, 0x70)) {
      return true;
    }
    NativeActiveEffectView &effect = result->effects[index];
    effect.remaining_ms = read_object_field<std::int32_t>(entry, 0x08);
    if (effect.remaining_ms < -1) {
      return true;
    }
    const void *asset = read_object_field<const void *>(entry, 0x18);
    if (!process_range_is_readable(asset, 0x44)) {
      return true;
    }
    const void *asset_vtable = read_object_field<const void *>(asset, 0x00);
    if (g_libg_base == 0 || asset_vtable != reinterpret_cast<const void *>(g_libg_base + kBuffAssetVtableOffset) ||
        !libg_address_has_segment_flags(asset_vtable, 0x78, PF_R)) {
      return true;
    }
    const void *class_name_getter = read_object_field<const void *>(asset_vtable, 0x70);
    if (class_name_getter != reinterpret_cast<const void *>(g_libg_base + kBuffAssetClassNameGetterOffset) ||
        !libg_address_has_segment_flags(class_name_getter, sizeof(std::uint32_t), PF_X)) {
      return true;
    }
    effect.buff_global_id = read_object_field<std::uint32_t>(asset, 0x40);
    if (effect.buff_global_id == 0 ||
        !read_bounded_libg_string_utf8(static_cast<const std::uint8_t *>(asset) + 0x28, effect.name,
                                       sizeof(effect.name), &effect.name_size)) {
      return true;
    }

    const void *source = read_object_field<const void *>(entry, 0x38);
    if (source == nullptr) {
      effect.source_entity_validated = true;
    } else {
      for (std::int32_t source_slot = 0; source_slot < object_count; ++source_slot) {
        if (objects[source_slot] != source || !process_range_is_readable(source, 0xa0)) {
          continue;
        }
        NativeEntityIdentityView identity;
        if (!read_native_entity_identity(source, &identity) ||
            !native_object_id_is_unique_in_vector(source, objects, object_count, identity.native_object_id)) {
          return true;
        }
        effect.source_entity_validated = true;
        effect.source_entity_resolved = true;
        effect.source_native_object_id = identity.native_object_id;
        effect.source_owner = identity.owner;
        effect.source_object_index = identity.object_index;
        effect.source_secondary_index = identity.secondary_index;
        break;
      }
    }
  }
  result->active_effects_valid = true;
  return true;
}

bool read_attack_sequence_stage(const void *object, const NativeComponentInventoryView &inventory,
                                std::int32_t *stage) {
  if (object == nullptr || stage == nullptr || !inventory.valid || inventory.count <= 0 ||
      !inventory.slots[0].present || !inventory.slots[0].owner_backlink_valid || !inventory.slots[0].type_valid ||
      inventory.slots[0].engine_type != 0) {
    return false;
  }
  void **components = read_object_field<void **>(object, 0x18);
  const void *component = components == nullptr ? nullptr : components[0];
  if (!process_range_is_readable(component, 0x24)) {
    return false;
  }
  *stage = read_object_field<std::int32_t>(component, 0x20);
  // Exact-build static readers constrain this attack-sequence field to the
  // configured AttackSequence list. Keep a separate conservative wire bound
  // instead of treating Inferno Dragon's three stages as a global maximum.
  // Inferno Dragon 26000037 generation/stateEpoch 18
  // supplied live transitions 0 -> 1 -> 2 at 2000/4000 ms, damage deltas
  // 29/99/349, a target-change reset to 0, then another 0 -> 1 -> 2 cycle.
  return *stage >= 0 && *stage <= kMaxAttackSequenceStage;
}

struct NativeTargetEntityView {
  bool validated = false;
  bool has_target = false;
  std::int32_t object_slot = -1;
  std::uint32_t native_object_id = 0;
  std::int32_t owner = 0;
  std::int32_t object_index = 0;
  std::int32_t secondary_index = 0;
};

bool resolve_native_target_entity(const void *object, const NativeComponentInventoryView &inventory,
                                  void *const *objects, std::int32_t object_count, NativeTargetEntityView *result) {
  if (object == nullptr || result == nullptr || object_count < 0 || (object_count != 0 && objects == nullptr)) {
    return false;
  }
  *result = {};
  if (!inventory.valid || inventory.count <= 0 || !inventory.slots[0].present ||
      !inventory.slots[0].owner_backlink_valid || !inventory.slots[0].type_valid ||
      inventory.slots[0].engine_type != 0) {
    return false;
  }
  void **components = read_object_field<void **>(object, 0x18);
  const void *movement_component = components == nullptr ? nullptr : components[0];
  if (!process_range_is_readable(movement_component, 0x18)) {
    return false;
  }

  // Named live golden (generation/stateEpoch 1): Giant 26000003 object slot
  // 6 had a null value at tick 131, then this field matched the owner-1
  // right-tower object pointer at tick 530 while its movement target was
  // (14500, 25500). Resolve only by identity in the current bounded object
  // vector and never expose the process pointer itself.
  const void *target = read_object_field<const void *>(movement_component, 0x10);
  if (target == nullptr) {
    result->validated = true;
    return true;
  }
  for (std::int32_t slot = 0; slot < object_count; ++slot) {
    if (objects[slot] != target) {
      continue;
    }
    NativeEntityIdentityView identity;
    if (!read_native_entity_identity(target, &identity) ||
        !native_object_id_is_unique_in_vector(target, objects, object_count, identity.native_object_id)) {
      return false;
    }
    result->validated = true;
    result->has_target = true;
    result->object_slot = slot;
    result->native_object_id = identity.native_object_id;
    result->owner = identity.owner;
    result->object_index = identity.object_index;
    result->secondary_index = identity.secondary_index;
    return true;
  }
  return false;
}

struct NativeEntityReferenceView {
  bool validated = false;
  bool has_value = false;
  std::int32_t slot = -1;
  std::uint32_t native_object_id = 0;
  std::int32_t owner = 0;
  std::int32_t object_index = 0;
  std::int32_t secondary_index = 0;
};

void resolve_bounded_entity_reference(const void *pointer, void *const *objects, std::int32_t object_count,
                                      NativeEntityReferenceView *result) {
  if (result == nullptr) {
    return;
  }
  *result = {};
  if (pointer == nullptr) {
    result->validated = true;
    return;
  }
  if (objects == nullptr || object_count < 0) {
    return;
  }
  for (std::int32_t slot = 0; slot < object_count; ++slot) {
    if (objects[slot] != pointer || !process_range_is_readable(pointer, 0xb0)) {
      continue;
    }
    NativeEntityIdentityView identity;
    if (!read_native_entity_identity(pointer, &identity) ||
        !native_object_id_is_unique_in_vector(pointer, objects, object_count, identity.native_object_id)) {
      return;
    }
    result->validated = true;
    result->has_value = true;
    result->slot = slot;
    result->native_object_id = identity.native_object_id;
    result->owner = identity.owner;
    result->object_index = identity.object_index;
    result->secondary_index = identity.secondary_index;
    return;
  }
}

using LogicGameObjectKindGetter = std::int32_t (*)(const void *);
using LogicDataTypeGetter = std::int32_t (*)(const void *);

bool validate_exact_game_object_kind(const void *object, std::size_t readable_size,
                                     std::uintptr_t expected_vtable_offset, std::uintptr_t expected_getter_offset,
                                     std::int32_t expected_kind) {
  if (object == nullptr || g_libg_base == 0 || !process_range_is_readable(object, readable_size)) {
    return false;
  }
  const void *const vtable = read_object_field<const void *>(object, 0x00);
  if (vtable != reinterpret_cast<const void *>(g_libg_base + expected_vtable_offset) ||
      !libg_address_has_segment_flags(vtable, 0x18, PF_R)) {
    return false;
  }
  const auto getter = read_object_field<LogicGameObjectKindGetter>(vtable, 0x10);
  if (getter != reinterpret_cast<LogicGameObjectKindGetter>(g_libg_base + expected_getter_offset) ||
      !libg_address_has_segment_flags(reinterpret_cast<const void *>(getter), sizeof(std::uint32_t), PF_X)) {
    return false;
  }
  return getter(object) == expected_kind;
}

enum class NativeProjectileReadStatus : std::int32_t {
  NotProjectile = 0,
  Valid = 1,
  Invalid = 2,
};

struct NativeProjectileStateView {
  std::uint32_t data_global_id = 0;
  NativeEntityReferenceView source;
  NativeEntityReferenceView target;
  NativeEntityReferenceView homing_target;
  std::int32_t destination_x = 0;
  std::int32_t destination_y = 0;
  bool terminal = false;
  // -1 means this projectile data has no native drag-back stage.  Exact
  // drag-capable projectiles expose the engine's persisted 0/1 latch.
  std::int32_t drag_stage = -1;
};

NativeProjectileReadStatus read_native_projectile_state(const void *object, void *const *objects,
                                                        std::int32_t object_count, NativeProjectileStateView *result) {
  if (object == nullptr || result == nullptr || g_libg_base == 0) {
    return NativeProjectileReadStatus::NotProjectile;
  }
  *result = {};
  if (!process_range_is_readable(object, 0x08)) {
    return NativeProjectileReadStatus::NotProjectile;
  }
  const void *const vtable = read_object_field<const void *>(object, 0x00);
  if (vtable != reinterpret_cast<const void *>(g_libg_base + kProjectileVtableOffset)) {
    return NativeProjectileReadStatus::NotProjectile;
  }
  if (!validate_exact_game_object_kind(object, kProjectileDragStageOffset + sizeof(std::uint8_t),
                                       kProjectileVtableOffset, kProjectileKindGetterOffset, 4)) {
    return NativeProjectileReadStatus::Invalid;
  }

  const void *const data = read_object_field<const void *>(object, 0x48);
  const auto type_getter = reinterpret_cast<LogicDataTypeGetter>(g_libg_base + kLogicDataTypeGetterOffset);
  if (!process_range_is_readable(data, kProjectileDragBackSpeedOffset + sizeof(std::int32_t)) ||
      !libg_address_has_segment_flags(reinterpret_cast<const void *>(type_getter), sizeof(std::uint32_t), PF_X) ||
      type_getter(data) != 0x0a) {
    return NativeProjectileReadStatus::Invalid;
  }
  result->data_global_id = read_object_field<std::uint32_t>(data, 0x40);
  if (result->data_global_id == 0) {
    return NativeProjectileReadStatus::Invalid;
  }

  resolve_bounded_entity_reference(read_object_field<const void *>(object, 0x100), objects, object_count,
                                   &result->source);
  resolve_bounded_entity_reference(read_object_field<const void *>(object, 0x108), objects, object_count,
                                   &result->target);
  resolve_bounded_entity_reference(read_object_field<const void *>(object, 0x110), objects, object_count,
                                   &result->homing_target);
  result->destination_x = read_object_field<std::int32_t>(object, 0x120);
  result->destination_y = read_object_field<std::int32_t>(object, 0x124);
  const std::uint8_t terminal = read_object_field<std::uint8_t>(object, 0x170);
  if (terminal > 1) {
    return NativeProjectileReadStatus::Invalid;
  }
  result->terminal = terminal != 0;
  const std::int32_t drag_back_speed = read_object_field<std::int32_t>(data, kProjectileDragBackSpeedOffset);
  if (drag_back_speed > 0) {
    const std::uint8_t drag_stage = read_object_field<std::uint8_t>(object, kProjectileDragStageOffset);
    if (drag_stage > 1) {
      return NativeProjectileReadStatus::Invalid;
    }
    result->drag_stage = static_cast<std::int32_t>(drag_stage);
  }
  return NativeProjectileReadStatus::Valid;
}

enum class NativeEntityResourceReadStatus : std::int32_t {
  NotApplicable = 0,
  Valid = 1,
  Invalid = 2,
};

struct NativeEntityResourceStateView {
  std::int32_t current_raw = 0;
  std::int32_t capacity_raw = 0;
  std::int32_t base_raw = 0;
  std::int32_t limit_raw = 0;
};

NativeEntityResourceReadStatus read_native_entity_resource_state(const void *object,
                                                                 NativeEntityResourceStateView *result) {
  if (object == nullptr || result == nullptr || g_libg_base == 0) {
    return NativeEntityResourceReadStatus::NotApplicable;
  }
  *result = {};
  if (!process_range_is_readable(object, sizeof(void *)) ||
      read_object_field<const void *>(object, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kCharacterVtableOffset)) {
    return NativeEntityResourceReadStatus::NotApplicable;
  }
  if (!process_range_is_readable(object, kCharacterExtraSpawnAccumulatorOffset + sizeof(std::int32_t))) {
    return NativeEntityResourceReadStatus::Invalid;
  }
  const void *const character_data = read_object_field<const void *>(object, 0x48);
  if (!process_range_is_readable(character_data, 0x718)) {
    return NativeEntityResourceReadStatus::Invalid;
  }
  const void *const ability_data = read_object_field<const void *>(character_data, 0x710);
  if (ability_data == nullptr || !process_range_is_readable(ability_data, sizeof(void *)) ||
      read_object_field<const void *>(ability_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kCharacterAbilityDataVtableOffset)) {
    return NativeEntityResourceReadStatus::NotApplicable;
  }
  if (!process_range_is_readable(ability_data, kCharacterAbilityExtraSpawnLimitOffset + sizeof(std::int32_t))) {
    return NativeEntityResourceReadStatus::Invalid;
  }
  const std::int32_t base = read_object_field<std::int32_t>(ability_data, kCharacterAbilityExtraSpawnBaseOffset);
  const std::int32_t limit = read_object_field<std::int32_t>(ability_data, kCharacterAbilityExtraSpawnLimitOffset);
  if (base <= 0 || limit <= base) {
    return NativeEntityResourceReadStatus::NotApplicable;
  }
  const std::int32_t current = read_object_field<std::int32_t>(object, kCharacterExtraSpawnAccumulatorOffset);
  const std::int32_t capacity = limit - base;
  if (current < 0 || current > capacity) {
    return NativeEntityResourceReadStatus::Invalid;
  }
  result->current_raw = current;
  result->capacity_raw = capacity;
  result->base_raw = base;
  result->limit_raw = limit;
  return NativeEntityResourceReadStatus::Valid;
}

enum class NativePeriodicAttackModifierReadStatus : std::int32_t {
  NotApplicable = 0,
  Valid = 1,
  Invalid = 2,
};

enum class NativePeriodicAttackModifierPhase : std::int32_t {
  SourceAlive = 0,
  SourceDeathLinger = 1,
};

struct NativePeriodicAttackModifierStateView {
  std::uint32_t action_data_global_id = 0;
  NativePeriodicAttackModifierPhase phase = NativePeriodicAttackModifierPhase::SourceAlive;
  std::int32_t period_attacks = 0;
  std::int32_t completed_attacks = 0;
  std::int32_t added_damage_raw = 0;
  std::int32_t linger_duration_ms = 0;
  std::int32_t linger_remaining_ms = 0;
  std::uint32_t source_native_object_id = 0;
  NativeEntityReferenceView source;
};

const char *native_periodic_attack_modifier_phase_label(NativePeriodicAttackModifierPhase phase) {
  return phase == NativePeriodicAttackModifierPhase::SourceAlive ? "source_alive" : "source_death_linger";
}

NativePeriodicAttackModifierReadStatus
read_native_periodic_attack_modifier_state(const void *object, void *const *objects, std::int32_t object_count,
                                           NativePeriodicAttackModifierStateView *result) {
  if (object == nullptr || objects == nullptr || object_count < 0 || result == nullptr || g_libg_base == 0 ||
      !process_range_is_readable(object, 0x30)) {
    return NativePeriodicAttackModifierReadStatus::NotApplicable;
  }
  *result = {};
  const void *const queue = read_object_field<const void *>(object, 0x28);
  if (queue == nullptr || !process_range_is_readable(queue, 0x60)) {
    return NativePeriodicAttackModifierReadStatus::NotApplicable;
  }
  void *const *const runtimes = read_object_field<void *const *>(queue, 0x50);
  const std::int32_t runtime_capacity = read_object_field<std::int32_t>(queue, 0x58);
  const std::int32_t runtime_count = read_object_field<std::int32_t>(queue, 0x5c);
  if (runtime_count < 0 || runtime_capacity < runtime_count || runtime_capacity > 256 ||
      (runtime_count != 0 &&
       (runtimes == nullptr ||
        !process_range_is_readable(runtimes, static_cast<std::size_t>(runtime_count) * sizeof(void *))))) {
    return NativePeriodicAttackModifierReadStatus::Invalid;
  }
  const void *modifier_runtime = nullptr;
  for (std::int32_t index = 0; index < runtime_count; ++index) {
    const void *const candidate = runtimes[index];
    if (candidate == nullptr || !process_range_is_readable(candidate, sizeof(void *)) ||
        read_object_field<const void *>(candidate, 0x00) !=
            reinterpret_cast<const void *>(g_libg_base + kPeriodicAttackModifierRuntimeVtableOffset)) {
      continue;
    }
    if (modifier_runtime != nullptr) {
      return NativePeriodicAttackModifierReadStatus::Invalid;
    }
    modifier_runtime = candidate;
  }
  if (modifier_runtime == nullptr) {
    return NativePeriodicAttackModifierReadStatus::NotApplicable;
  }
  if (!process_range_is_readable(modifier_runtime, 0x7c) ||
      read_object_field<const void *>(modifier_runtime, 0x20) != object) {
    return NativePeriodicAttackModifierReadStatus::Invalid;
  }
  const void *const action_data = read_object_field<const void *>(modifier_runtime, 0x18);
  if (!process_range_is_readable(action_data, 0x110) ||
      read_object_field<const void *>(action_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kPeriodicAttackModifierActionDataVtableOffset)) {
    return NativePeriodicAttackModifierReadStatus::Invalid;
  }

  const std::uint8_t finished = read_object_field<std::uint8_t>(modifier_runtime, 0x10);
  const std::int32_t lifecycle_state = read_object_field<std::int32_t>(modifier_runtime, 0x60);
  if (finished > 1 || lifecycle_state < 0 || lifecycle_state > 2) {
    return NativePeriodicAttackModifierReadStatus::Invalid;
  }
  if (finished != 0 || lifecycle_state == 0) {
    return NativePeriodicAttackModifierReadStatus::NotApplicable;
  }

  const std::uint32_t action_data_global_id = read_object_field<std::uint32_t>(action_data, 0x40);
  const std::int32_t period = read_object_field<std::int32_t>(action_data, 0xec);
  const std::int32_t added_damage = read_object_field<std::int32_t>(action_data, 0x100);
  const std::int32_t linger_duration = read_object_field<std::int32_t>(action_data, 0x108);
  const std::int32_t completed = read_object_field<std::int32_t>(modifier_runtime, 0x70);
  const std::int32_t linger_remaining = read_object_field<std::int32_t>(modifier_runtime, 0x74);
  const std::uint32_t source_native_object_id = read_object_field<std::uint32_t>(modifier_runtime, 0x78);
  if (action_data_global_id == 0 || period <= 0 || added_damage < 0 || linger_duration <= 0 || completed < 0 ||
      completed >= period || linger_remaining < 0 || linger_remaining > linger_duration ||
      source_native_object_id == 0 || (lifecycle_state == 1 && linger_remaining != 0) ||
      (lifecycle_state == 2 && linger_remaining == 0)) {
    return NativePeriodicAttackModifierReadStatus::Invalid;
  }

  const void *source_pointer = nullptr;
  for (std::int32_t slot = 0; slot < object_count; ++slot) {
    NativeEntityIdentityView identity;
    if (objects[slot] != nullptr && read_native_entity_identity(objects[slot], &identity) &&
        identity.native_object_id == source_native_object_id) {
      source_pointer = objects[slot];
      break;
    }
  }
  resolve_bounded_entity_reference(source_pointer, objects, object_count, &result->source);
  if (!result->source.validated) {
    return NativePeriodicAttackModifierReadStatus::Invalid;
  }

  result->action_data_global_id = action_data_global_id;
  // The source leaves the active object vector one update before this
  // runtime advances from state 1 to state 2.  That exact boundary is the
  // first 50 ms slice of FinishIfInstigatorDies: expose the full configured
  // duration now, then consume the native +0x74 countdown from state 2.
  const bool source_loss_boundary = lifecycle_state == 1 && !result->source.has_value;
  result->phase = lifecycle_state == 1 && !source_loss_boundary ? NativePeriodicAttackModifierPhase::SourceAlive
                                                                : NativePeriodicAttackModifierPhase::SourceDeathLinger;
  result->period_attacks = period;
  result->completed_attacks = completed;
  result->added_damage_raw = added_damage;
  result->linger_duration_ms = linger_duration;
  result->linger_remaining_ms = source_loss_boundary ? linger_duration : linger_remaining;
  result->source_native_object_id = source_native_object_id;
  return NativePeriodicAttackModifierReadStatus::Valid;
}

enum class NativeCaptureRuntimeReadStatus : std::int32_t {
  NotApplicable = 0,
  Valid = 1,
  Invalid = 2,
};

enum class NativeCaptureTargetPhase : std::int32_t {
  AcquiredDelay = 0,
  GrabPause = 1,
  Dragging = 2,
  Contained = 3,
  ReleasePending = 4,
};

struct NativeCaptureTargetStateView {
  std::uint32_t native_object_id = 0;
  NativeEntityReferenceView entity;
  std::int32_t elapsed_ms = 0;
  std::int32_t phase_budget_remaining_ms = 0;
  bool phase_budget_remaining_known = false;
  NativeCaptureTargetPhase phase = NativeCaptureTargetPhase::AcquiredDelay;
};

struct NativeCaptureRuntimeStateView {
  std::uint32_t action_data_global_id = 0;
  std::int32_t drag_delay_ms = 0;
  std::int32_t grab_pause_ms = 0;
  std::int32_t capture_drag_time_ms = 0;
  std::int32_t configured_cooldown_ms = 0;
  std::int32_t cooldown_remaining_ms = 0;
  std::int32_t hit_frequency_ms = 0;
  std::int32_t hit_accumulator_ms = 0;
  bool completion_result_current_update = false;
  bool first_capture_handled = false;
  std::int32_t target_count = 0;
  NativeCaptureTargetStateView targets[kCaptureRuntimeMaximumTargets];
};

const char *native_capture_target_phase_label(NativeCaptureTargetPhase phase) {
  switch (phase) {
  case NativeCaptureTargetPhase::AcquiredDelay:
    return "acquired_delay";
  case NativeCaptureTargetPhase::GrabPause:
    return "grab_pause";
  case NativeCaptureTargetPhase::Dragging:
    return "dragging";
  case NativeCaptureTargetPhase::Contained:
    return "contained";
  case NativeCaptureTargetPhase::ReleasePending:
    return "release_pending";
  }
  return "release_pending";
}

bool read_capture_i32_vector(const void *runtime, std::size_t data_offset, std::size_t capacity_offset,
                             std::size_t count_offset, std::int32_t maximum_capacity, const std::int32_t **data_out,
                             std::int32_t *count_out) {
  if (runtime == nullptr || data_out == nullptr || count_out == nullptr ||
      !process_range_is_readable(runtime, count_offset + sizeof(std::int32_t))) {
    return false;
  }
  const auto *const data = read_object_field<const std::int32_t *>(runtime, data_offset);
  const std::int32_t capacity = read_object_field<std::int32_t>(runtime, capacity_offset);
  const std::int32_t count = read_object_field<std::int32_t>(runtime, count_offset);
  if (count < 0 || capacity < count || capacity > maximum_capacity ||
      (count != 0 &&
       (data == nullptr || !process_range_is_readable(data, static_cast<std::size_t>(count) * sizeof(std::int32_t))))) {
    return false;
  }
  *data_out = data;
  *count_out = count;
  return true;
}

NativeCaptureRuntimeReadStatus read_native_capture_runtime_state(const void *object, void *const *objects,
                                                                 std::int32_t object_count,
                                                                 NativeCaptureRuntimeStateView *result) {
  if (object == nullptr || objects == nullptr || object_count < 0 || result == nullptr || g_libg_base == 0 ||
      !process_range_is_readable(object, 0x30)) {
    return NativeCaptureRuntimeReadStatus::NotApplicable;
  }
  *result = {};
  const void *const queue = read_object_field<const void *>(object, 0x28);
  if (queue == nullptr || !process_range_is_readable(queue, 0x60)) {
    return NativeCaptureRuntimeReadStatus::NotApplicable;
  }
  void *const *const runtimes = read_object_field<void *const *>(queue, 0x50);
  const std::int32_t runtime_capacity = read_object_field<std::int32_t>(queue, 0x58);
  const std::int32_t runtime_count = read_object_field<std::int32_t>(queue, 0x5c);
  if (runtime_count < 0 || runtime_capacity < runtime_count || runtime_capacity > 256 ||
      (runtime_count != 0 &&
       (runtimes == nullptr ||
        !process_range_is_readable(runtimes, static_cast<std::size_t>(runtime_count) * sizeof(void *))))) {
    return NativeCaptureRuntimeReadStatus::Invalid;
  }
  const void *capture_runtime = nullptr;
  for (std::int32_t index = 0; index < runtime_count; ++index) {
    const void *const candidate = runtimes[index];
    if (candidate == nullptr || !process_range_is_readable(candidate, sizeof(void *))) {
      continue;
    }
    if (read_object_field<const void *>(candidate, 0x00) !=
        reinterpret_cast<const void *>(g_libg_base + kCaptureCharacterRuntimeVtableOffset)) {
      continue;
    }
    if (capture_runtime != nullptr) {
      return NativeCaptureRuntimeReadStatus::Invalid;
    }
    capture_runtime = candidate;
  }
  if (capture_runtime == nullptr) {
    return NativeCaptureRuntimeReadStatus::NotApplicable;
  }
  if (!process_range_is_readable(capture_runtime, 0xca) ||
      read_object_field<const void *>(capture_runtime, 0x20) != object) {
    return NativeCaptureRuntimeReadStatus::Invalid;
  }
  const void *const action_data = read_object_field<const void *>(capture_runtime, 0x18);
  if (!process_range_is_readable(action_data, 0x234) ||
      read_object_field<const void *>(action_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kCaptureCharacterActionDataVtableOffset)) {
    return NativeCaptureRuntimeReadStatus::Invalid;
  }

  const std::uint32_t action_data_global_id = read_object_field<std::uint32_t>(action_data, 0x40);
  const std::int32_t capture_limit = read_object_field<std::int32_t>(action_data, 0xf8);
  const std::int32_t drag_delay = read_object_field<std::int32_t>(action_data, 0x108);
  const std::int32_t capture_drag_time = read_object_field<std::int32_t>(action_data, 0x10c);
  const std::int32_t configured_cooldown = read_object_field<std::int32_t>(action_data, 0x114);
  const std::int32_t grab_pause = read_object_field<std::int32_t>(action_data, 0x168);
  const std::int32_t hit_frequency = read_object_field<std::int32_t>(action_data, 0xf4);
  if (action_data_global_id == 0 || capture_limit <= 0 || capture_limit > kCaptureRuntimeMaximumConfiguredLimit ||
      drag_delay < 0 || capture_drag_time < 0 || grab_pause < 0 || configured_cooldown < 0 || hit_frequency <= 0) {
    return NativeCaptureRuntimeReadStatus::Invalid;
  }

  const std::int32_t *active_ids = nullptr;
  const std::int32_t *completed_ids = nullptr;
  const std::int32_t *elapsed_values = nullptr;
  std::int32_t active_count = 0;
  std::int32_t completed_count = 0;
  std::int32_t elapsed_count = 0;
  if (!read_capture_i32_vector(capture_runtime, 0x60, 0x68, 0x6c, kCaptureRuntimeMaximumVectorCapacity, &active_ids,
                               &active_count) ||
      !read_capture_i32_vector(capture_runtime, 0x70, 0x78, 0x7c, kCaptureRuntimeMaximumVectorCapacity, &completed_ids,
                               &completed_count) ||
      !read_capture_i32_vector(capture_runtime, 0x80, 0x88, 0x8c, kCaptureRuntimeMaximumVectorCapacity, &elapsed_values,
                               &elapsed_count) ||
      active_count != elapsed_count || completed_count > active_count || active_count > capture_limit ||
      active_count > static_cast<std::int32_t>(kCaptureRuntimeMaximumTargets)) {
    return NativeCaptureRuntimeReadStatus::Invalid;
  }
  for (std::int32_t index = 0; index < active_count; ++index) {
    if (active_ids[index] <= 0 || elapsed_values[index] < 0) {
      return NativeCaptureRuntimeReadStatus::Invalid;
    }
    for (std::int32_t other = 0; other < index; ++other) {
      if (active_ids[other] == active_ids[index]) {
        return NativeCaptureRuntimeReadStatus::Invalid;
      }
    }
  }
  for (std::int32_t index = 0; index < completed_count; ++index) {
    bool found = false;
    for (std::int32_t active = 0; active < active_count; ++active) {
      found = found || completed_ids[index] == active_ids[active];
    }
    for (std::int32_t other = 0; other < index; ++other) {
      if (completed_ids[other] == completed_ids[index]) {
        return NativeCaptureRuntimeReadStatus::Invalid;
      }
    }
    if (completed_ids[index] <= 0 || !found) {
      return NativeCaptureRuntimeReadStatus::Invalid;
    }
  }

  const std::int32_t cooldown_remaining = read_object_field<std::int32_t>(capture_runtime, 0xc0);
  const std::int32_t hit_accumulator = read_object_field<std::int32_t>(capture_runtime, 0xc4);
  const std::uint8_t completion_result = read_object_field<std::uint8_t>(capture_runtime, 0xc8);
  const std::uint8_t first_capture_handled = read_object_field<std::uint8_t>(capture_runtime, 0xc9);
  if (cooldown_remaining < 0 || cooldown_remaining > configured_cooldown || hit_accumulator < 0 ||
      completion_result > 1 || first_capture_handled > 1) {
    return NativeCaptureRuntimeReadStatus::Invalid;
  }

  result->action_data_global_id = action_data_global_id;
  result->drag_delay_ms = drag_delay;
  result->grab_pause_ms = grab_pause;
  result->capture_drag_time_ms = capture_drag_time;
  result->configured_cooldown_ms = configured_cooldown;
  result->cooldown_remaining_ms = cooldown_remaining;
  result->hit_frequency_ms = hit_frequency;
  result->hit_accumulator_ms = hit_accumulator;
  result->completion_result_current_update = completion_result != 0;
  result->first_capture_handled = first_capture_handled != 0;
  result->target_count = active_count;
  for (std::int32_t index = 0; index < active_count; ++index) {
    NativeCaptureTargetStateView &target = result->targets[index];
    target.native_object_id = static_cast<std::uint32_t>(active_ids[index]);
    target.elapsed_ms = elapsed_values[index];
    const void *target_pointer = nullptr;
    for (std::int32_t slot = 0; slot < object_count; ++slot) {
      const void *const candidate = objects[slot];
      NativeEntityIdentityView identity;
      if (candidate != nullptr && read_native_entity_identity(candidate, &identity) &&
          identity.native_object_id == target.native_object_id) {
        target_pointer = candidate;
        break;
      }
    }
    if (target_pointer != nullptr) {
      resolve_bounded_entity_reference(target_pointer, objects, object_count, &target.entity);
    }
    bool completed = false;
    for (std::int32_t completed_index = 0; completed_index < completed_count; ++completed_index) {
      completed = completed || completed_ids[completed_index] == active_ids[index];
    }
    if (!target.entity.validated || !target.entity.has_value) {
      target.phase = NativeCaptureTargetPhase::ReleasePending;
    } else if (completed) {
      target.phase = NativeCaptureTargetPhase::Contained;
    } else if (target.elapsed_ms < drag_delay) {
      target.phase = NativeCaptureTargetPhase::AcquiredDelay;
      target.phase_budget_remaining_ms = drag_delay - target.elapsed_ms;
      target.phase_budget_remaining_known = true;
    } else if (target.elapsed_ms < drag_delay + grab_pause) {
      target.phase = NativeCaptureTargetPhase::GrabPause;
      target.phase_budget_remaining_ms = drag_delay + grab_pause - target.elapsed_ms;
      target.phase_budget_remaining_known = true;
    } else {
      target.phase = NativeCaptureTargetPhase::Dragging;
      const std::int32_t remaining = drag_delay + grab_pause + capture_drag_time - target.elapsed_ms;
      target.phase_budget_remaining_ms = remaining > 0 ? remaining : 0;
      target.phase_budget_remaining_known = true;
    }
  }
  return NativeCaptureRuntimeReadStatus::Valid;
}

enum class NativeThresholdRelocationReadStatus : std::int32_t {
  NotApplicable = 0,
  Valid = 1,
  Invalid = 2,
};

enum class NativeThresholdRelocationPhase : std::int32_t {
  WaitingThreshold = 0,
  Relocating = 1,
  Exhausted = 2,
};

struct NativeThresholdRelocationStateView {
  std::uint32_t action_data_global_id = 0;
  NativeThresholdRelocationPhase phase = NativeThresholdRelocationPhase::WaitingThreshold;
  std::int32_t stage = 0;
  std::int32_t relocation_index = 0;
  std::int32_t threshold_count = 0;
  std::int32_t thresholds[kThresholdRelocationMaximumThresholds] = {};
  std::int32_t hide_duration_ms = 0;
  std::int32_t remaining_ms = 0;
  bool burrowed = false;
};

const char *native_threshold_relocation_phase_label(NativeThresholdRelocationPhase phase) {
  switch (phase) {
  case NativeThresholdRelocationPhase::WaitingThreshold:
    return "waiting_threshold";
  case NativeThresholdRelocationPhase::Relocating:
    return "relocating";
  case NativeThresholdRelocationPhase::Exhausted:
    return "exhausted";
  }
  return "exhausted";
}

NativeThresholdRelocationReadStatus read_native_threshold_relocation_state(const void *object,
                                                                           NativeThresholdRelocationStateView *result) {
  if (object == nullptr || result == nullptr || g_libg_base == 0 || !process_range_is_readable(object, 0x120)) {
    return NativeThresholdRelocationReadStatus::NotApplicable;
  }
  *result = {};
  const void *const queue = read_object_field<const void *>(object, 0x28);
  if (queue == nullptr || !process_range_is_readable(queue, 0x60)) {
    return NativeThresholdRelocationReadStatus::NotApplicable;
  }
  void *const *const runtimes = read_object_field<void *const *>(queue, 0x50);
  const std::int32_t runtime_capacity = read_object_field<std::int32_t>(queue, 0x58);
  const std::int32_t runtime_count = read_object_field<std::int32_t>(queue, 0x5c);
  if (runtime_count < 0 || runtime_capacity < runtime_count || runtime_capacity > 256 ||
      (runtime_count != 0 &&
       (runtimes == nullptr ||
        !process_range_is_readable(runtimes, static_cast<std::size_t>(runtime_count) * sizeof(void *))))) {
    return NativeThresholdRelocationReadStatus::Invalid;
  }
  const void *relocation_runtime = nullptr;
  for (std::int32_t index = 0; index < runtime_count; ++index) {
    const void *const candidate = runtimes[index];
    if (candidate == nullptr || !process_range_is_readable(candidate, sizeof(void *)) ||
        read_object_field<const void *>(candidate, 0x00) !=
            reinterpret_cast<const void *>(g_libg_base + kThresholdRelocationRuntimeVtableOffset)) {
      continue;
    }
    if (relocation_runtime != nullptr) {
      return NativeThresholdRelocationReadStatus::Invalid;
    }
    relocation_runtime = candidate;
  }
  if (relocation_runtime == nullptr) {
    return NativeThresholdRelocationReadStatus::NotApplicable;
  }
  if (!process_range_is_readable(relocation_runtime, 0x79) ||
      read_object_field<const void *>(relocation_runtime, 0x20) != object) {
    return NativeThresholdRelocationReadStatus::Invalid;
  }
  const void *const action_data = read_object_field<const void *>(relocation_runtime, 0x18);
  if (!process_range_is_readable(action_data, 0x158) ||
      read_object_field<const void *>(action_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kThresholdRelocationActionDataVtableOffset)) {
    return NativeThresholdRelocationReadStatus::Invalid;
  }

  const std::uint32_t action_data_global_id = read_object_field<std::uint32_t>(action_data, 0x40);
  const std::int32_t hide_duration = read_object_field<std::int32_t>(action_data, 0xf4);
  const std::int32_t *const thresholds = read_object_field<const std::int32_t *>(action_data, 0x148);
  const std::int32_t threshold_capacity = read_object_field<std::int32_t>(action_data, 0x150);
  const std::int32_t threshold_count = read_object_field<std::int32_t>(action_data, 0x154);
  if (action_data_global_id == 0 || hide_duration <= 0 || threshold_count <= 0 ||
      threshold_count > kThresholdRelocationMaximumThresholds || threshold_capacity < threshold_count ||
      threshold_capacity > kThresholdRelocationMaximumVectorCapacity || thresholds == nullptr ||
      !process_range_is_readable(thresholds, static_cast<std::size_t>(threshold_count) * sizeof(std::int32_t))) {
    return NativeThresholdRelocationReadStatus::Invalid;
  }
  for (std::int32_t index = 0; index < threshold_count; ++index) {
    if (thresholds[index] < 0 || thresholds[index] > 100 ||
        (index != 0 && thresholds[index] >= thresholds[index - 1])) {
      return NativeThresholdRelocationReadStatus::Invalid;
    }
  }

  const std::int32_t remaining = read_object_field<std::int32_t>(relocation_runtime, 0x60);
  const std::int32_t stage = read_object_field<std::int32_t>(relocation_runtime, 0x74);
  const std::uint8_t commit_pending = read_object_field<std::uint8_t>(relocation_runtime, 0x78);
  const std::int32_t owner_state = read_object_field<std::int32_t>(object, 0x11c);
  const std::int32_t terminal_stage = threshold_count * 2 + 1;
  if (remaining < 0 || remaining > hide_duration || stage < 1 || stage > terminal_stage || commit_pending > 1 ||
      owner_state < 0 || owner_state > 16) {
    return NativeThresholdRelocationReadStatus::Invalid;
  }

  result->action_data_global_id = action_data_global_id;
  result->stage = stage;
  result->relocation_index = (stage - 1) / 2;
  result->threshold_count = threshold_count;
  result->hide_duration_ms = hide_duration;
  result->remaining_ms = remaining;
  result->burrowed = stage % 2 == 0 && owner_state == 6 && remaining > 0;
  result->phase = stage == terminal_stage ? NativeThresholdRelocationPhase::Exhausted
                  : (stage % 2 == 0)      ? NativeThresholdRelocationPhase::Relocating
                                          : NativeThresholdRelocationPhase::WaitingThreshold;
  for (std::int32_t index = 0; index < threshold_count; ++index) {
    result->thresholds[index] = thresholds[index];
  }
  return NativeThresholdRelocationReadStatus::Valid;
}

const char *ability_button_state_label(std::int32_t value) {
  constexpr const char *kLabels[] = {
      "invalid/no match",       "ChampionAbsent",      "Ready",
      "ChampionDeploying",      "LimitedAvailability", "ERR_START",
      "AllChargesConsumed",     "ChampionPending",     "OnCooldown",
      "NotEnoughElixir",        "ChampionCasting",     "Disabled",
      "TemporarilyUnavailable", "NoYetAvailable",      "ERR_MAX",
  };
  return value >= 0 && value < static_cast<std::int32_t>(sizeof(kLabels) / sizeof(kLabels[0])) ? kLabels[value]
                                                                                               : nullptr;
}

bool ability_button_state_is_queueable(std::int32_t value) {
  // Ready is the ordinary queueable state. Some hero-form abilities use
  // ActionOverrideAbilityButtonState to expose LimitedAvailability while the
  // controller's one current carrier remains legally activatable. Keep every
  // other transient/terminal state fail-closed; cooldown, charges, controller
  // identity and carrier uniqueness are checked independently below.
  return value == 2 || value == 4;
}

struct NativeChampionControllerView {
  std::int32_t controller_slot = 0;
  std::int32_t action_data_global_id = 0;
  std::size_t action_data_name_size = 0;
  char action_data_name[129] = {};
  std::int32_t selected_character_global_id = 0;
  std::int32_t remaining_cooldown_ms = 0;
  std::int32_t configured_cooldown_ms = 0;
  std::int32_t remaining_charges_raw = 0;
  std::int32_t max_charges = 0;
  std::int32_t button_state = 0;
  std::int32_t champion_count = 0;
  NativeEntityReferenceView champions[kMaxObservedObjects] = {};
};

struct NativeEvolutionSlotView {
  std::int32_t deck_slot = -1;
  std::int32_t card_id = 0;
  std::int32_t base_spell_global_id = 0;
  bool evolvable = false;
  std::int32_t evolution_form_global_id = 0;
  std::int32_t progress = 0;
  std::int32_t cycle_required = 0;
  std::int32_t cycle_remaining = 0;
  bool ready = false;
};

struct NativePlayerRuntimeView {
  bool owner_root_valid = false;
  NativeEntityReferenceView owner_root;
  bool ability_runtime_valid = false;
  std::int32_t ability_count = 0;
  NativeChampionControllerView abilities[2] = {};
  bool evolution_runtime_valid = false;
  std::int32_t evolution_slot_count = 0;
  NativeEvolutionSlotView evolution_slots[8] = {};
};

bool champion_controller_is_exact_empty(const void *owner_root, const void *controller) {
  if (owner_root == nullptr || controller == nullptr || g_libg_base == 0 ||
      !process_range_is_readable(controller, 0xa8) ||
      read_object_field<const void *>(controller, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kChampionControllerVtableOffset) ||
      read_object_field<const void *>(controller, 0x20) != owner_root ||
      read_object_field<const void *>(controller, 0x90) != nullptr ||
      read_object_field<std::int32_t>(controller, 0x98) != 0) {
    return false;
  }
  const void *const champion_vector = read_object_field<const void *>(controller, 0xa0);
  NativeVectorView champions;
  return read_native_vector(champion_vector, 0x00, kMaxObservedObjects, &champions) && champions.count == 0;
}

bool read_champion_controller(const void *owner_root, std::int32_t owner, std::int32_t controller_slot,
                              const void *controller, void *const *objects, std::int32_t object_count,
                              NativeChampionControllerView *result) {
  if (owner_root == nullptr || controller == nullptr || result == nullptr || controller_slot < 1 ||
      controller_slot > 2 || g_libg_base == 0 || !process_range_is_readable(controller, 0xa8)) {
    return false;
  }
  *result = {};
  const void *const controller_vtable = read_object_field<const void *>(controller, 0x00);
  if (controller_vtable != reinterpret_cast<const void *>(g_libg_base + kChampionControllerVtableOffset) ||
      read_object_field<const void *>(controller, 0x20) != owner_root) {
    return false;
  }
  const void *const action_data = read_object_field<const void *>(controller, 0x18);
  char controller_action_name[129] = {};
  std::size_t controller_action_name_size = 0;
  if (!process_range_is_readable(action_data, 0x44) ||
      read_object_field<const void *>(action_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kChampionActionDataVtableOffset) ||
      !read_bounded_libg_string_utf8(static_cast<const std::uint8_t *>(action_data) + 0x28, controller_action_name,
                                     sizeof(controller_action_name), &controller_action_name_size)) {
    return false;
  }
  result->action_data_global_id = read_object_field<std::int32_t>(action_data, 0x40);
  if (result->action_data_global_id <= 0 || controller_action_name_size == 0) {
    return false;
  }

  const void *const selected_character = read_object_field<const void *>(controller, 0x90);
  if (!process_range_is_readable(selected_character, 0x718)) {
    return false;
  }
  result->selected_character_global_id = read_object_field<std::int32_t>(selected_character, 0x40);
  const void *const ability_data = read_object_field<const void *>(selected_character, 0x710);
  if (!process_range_is_readable(ability_data, 0xb8) ||
      read_object_field<const void *>(ability_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kCharacterAbilityDataVtableOffset) ||
      !read_bounded_libg_string_utf8(static_cast<const std::uint8_t *>(ability_data) + 0x28, result->action_data_name,
                                     sizeof(result->action_data_name), &result->action_data_name_size)) {
    return false;
  }
  result->controller_slot = controller_slot;
  result->remaining_cooldown_ms = read_object_field<std::int32_t>(controller, 0x78);
  result->configured_cooldown_ms = read_object_field<std::int32_t>(controller, 0x7c);
  result->remaining_charges_raw = read_object_field<std::int32_t>(controller, 0x80);
  result->button_state = read_object_field<std::int32_t>(controller, 0x98);
  result->max_charges = read_object_field<std::int32_t>(ability_data, 0xac);
  const std::int32_t configured_from_data = read_object_field<std::int32_t>(ability_data, 0xb4);
  if (result->configured_cooldown_ms < 0 || result->remaining_cooldown_ms < 0 ||
      result->remaining_cooldown_ms > result->configured_cooldown_ms ||
      configured_from_data != result->configured_cooldown_ms ||
      ability_button_state_label(result->button_state) == nullptr) {
    return false;
  }
  if ((result->max_charges > 0 &&
       (result->remaining_charges_raw < 0 || result->remaining_charges_raw > result->max_charges)) ||
      (result->max_charges <= 0 && result->remaining_charges_raw != -1)) {
    return false;
  }

  const void *const champion_vector = read_object_field<const void *>(controller, 0xa0);
  NativeVectorView champions;
  if (!read_native_vector(champion_vector, 0x00, kMaxObservedObjects, &champions) ||
      (champions.count != 0 &&
       !process_range_is_readable(champions.data, static_cast<std::size_t>(champions.count) * sizeof(void *)))) {
    return false;
  }
  auto **members = static_cast<void **>(champions.data);
  std::int32_t unique_live_card_id = 0;
  for (std::int32_t index = 0; index < champions.count; ++index) {
    const void *const member = members[index];
    NativeEntityReferenceView resolved;
    resolve_bounded_entity_reference(member, objects, object_count, &resolved);
    const std::int32_t member_card_id =
        process_range_is_readable(member, 0xb0) ? read_object_field<std::int32_t>(member, 0xac) : 0;
    if (!resolved.validated || !resolved.has_value ||
        read_object_field<const void *>(member, 0x48) != selected_character || resolved.owner != owner ||
        member_card_id <= 0) {
      return false;
    }
    if (champions.count == 1) {
      unique_live_card_id = member_card_id;
    }
    result->champions[index] = resolved;
  }
  result->champion_count = champions.count;
  if (result->selected_character_global_id <= 0) {
    if (result->champion_count != 1 || unique_live_card_id <= 0) {
      return false;
    }
    // Hero-form character data is a derived EXT record and does not carry
    // a normal positive LogicCharacterData global ID at +0x40.  Bind the
    // controller to its one exact live member instead of suppressing the
    // otherwise fully validated ability runtime.
    result->selected_character_global_id = unique_live_card_id;
  }
  return true;
}

bool champion_controller_is_exact_absent(const void *owner_root, const void *controller) {
  if (owner_root == nullptr || controller == nullptr || g_libg_base == 0 ||
      !process_range_is_readable(controller, 0xa8) ||
      read_object_field<const void *>(controller, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kChampionControllerVtableOffset) ||
      read_object_field<const void *>(controller, 0x20) != owner_root ||
      read_object_field<std::int32_t>(controller, 0x98) != 1) {
    return false;
  }

  const void *const action_data = read_object_field<const void *>(controller, 0x18);
  char action_name[129] = {};
  std::size_t action_name_size = 0;
  if (!process_range_is_readable(action_data, 0x44) ||
      read_object_field<const void *>(action_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kChampionActionDataVtableOffset) ||
      read_object_field<std::int32_t>(action_data, 0x40) <= 0 ||
      !read_bounded_libg_string_utf8(static_cast<const std::uint8_t *>(action_data) + 0x28, action_name,
                                     sizeof(action_name), &action_name_size) ||
      action_name_size == 0) {
    return false;
  }

  const void *const selected_character = read_object_field<const void *>(controller, 0x90);
  if (!process_range_is_readable(selected_character, 0x718) ||
      read_object_field<std::int32_t>(selected_character, 0x40) == 0) {
    return false;
  }
  const void *const ability_data = read_object_field<const void *>(selected_character, 0x710);
  char ability_name[129] = {};
  std::size_t ability_name_size = 0;
  if (!process_range_is_readable(ability_data, 0xb8) ||
      read_object_field<const void *>(ability_data, 0x00) !=
          reinterpret_cast<const void *>(g_libg_base + kCharacterAbilityDataVtableOffset) ||
      !read_bounded_libg_string_utf8(static_cast<const std::uint8_t *>(ability_data) + 0x28, ability_name,
                                     sizeof(ability_name), &ability_name_size) ||
      ability_name_size == 0) {
    return false;
  }

  const std::int32_t remaining_cooldown = read_object_field<std::int32_t>(controller, 0x78);
  const std::int32_t configured_cooldown = read_object_field<std::int32_t>(controller, 0x7c);
  const std::int32_t remaining_charges = read_object_field<std::int32_t>(controller, 0x80);
  const std::int32_t max_charges = read_object_field<std::int32_t>(ability_data, 0xac);
  const std::int32_t configured_from_data = read_object_field<std::int32_t>(ability_data, 0xb4);
  // An untouched absent controller uses -1 as its charge sentinel. After a
  // carrier has consumed its ability and left the arena, the stock runtime
  // preserves the last bounded charge count instead. Both are exact absent
  // states as long as there is no controller member below.
  if (configured_cooldown < 0 || remaining_cooldown < 0 || remaining_cooldown > configured_cooldown ||
      configured_from_data != configured_cooldown || max_charges <= 0 ||
      (remaining_charges != -1 && (remaining_charges < 0 || remaining_charges > max_charges))) {
    return false;
  }

  const void *const champion_vector = read_object_field<const void *>(controller, 0xa0);
  NativeVectorView champions;
  return read_native_vector(champion_vector, 0x00, kMaxObservedObjects, &champions) && champions.count == 0;
}

bool read_evolution_slot(const PlayerStateView &player, const std::int32_t *progress, std::int32_t deck_slot,
                         NativeEvolutionSlotView *result) {
  constexpr std::int32_t kMaxEvolutionCandidates = 16;
  if (progress == nullptr || result == nullptr || deck_slot < 0 || deck_slot >= player.deck_slots.count) {
    return false;
  }
  *result = {};
  CardStateView card;
  if (!read_card_state(player, deck_slot, false, &card) || !process_range_is_readable(card.card_data, 0x230)) {
    return false;
  }
  result->deck_slot = deck_slot;
  result->card_id = card.card_id;
  result->base_spell_global_id = read_object_field<std::int32_t>(card.card_data, 0x40);
  result->progress = progress[deck_slot];
  if (result->card_id <= 0 || result->base_spell_global_id != result->card_id || result->progress < 0) {
    return false;
  }

  NativeVectorView candidates;
  if (!read_native_vector(card.card_data, 0x220, kMaxEvolutionCandidates, &candidates) ||
      (candidates.count != 0 &&
       !process_range_is_readable(candidates.data, static_cast<std::size_t>(candidates.count) * sizeof(void *)))) {
    return false;
  }
  auto **candidate_items = static_cast<void **>(candidates.data);
  const void *evolution_form = nullptr;
  for (std::int32_t index = 0; index < candidates.count; ++index) {
    const void *const candidate = candidate_items[index];
    if (!process_range_is_readable(candidate, 0x2b0)) {
      return false;
    }
    const void *const form_metadata = read_object_field<const void *>(candidate, 0x2a8);
    if (!process_range_is_readable(form_metadata, 0x68)) {
      return false;
    }
    const std::int32_t form_code = read_object_field<std::int32_t>(form_metadata, 0x64);
    if (form_code != 1) {
      continue;
    }
    if (evolution_form != nullptr) {
      return false;
    }
    evolution_form = candidate;
  }
  if (evolution_form == nullptr) {
    return result->progress == 0;
  }

  result->evolvable = true;
  result->evolution_form_global_id = read_object_field<std::int32_t>(evolution_form, 0x40);
  result->cycle_required = read_object_field<std::int32_t>(evolution_form, 0x284);
  if (result->evolution_form_global_id <= 0 || result->cycle_required <= 0 ||
      result->progress > result->cycle_required) {
    return false;
  }
  result->cycle_remaining = result->cycle_required - result->progress;
  result->ready = result->progress >= result->cycle_required;
  return true;
}

bool read_native_player_runtime(void *world, void *object_manager, void *const *objects, std::int32_t object_count,
                                std::int32_t owner, NativePlayerRuntimeView *result) {
  if (result == nullptr) {
    return false;
  }
  *result = {};
  PlayerStateView player;
  if (!read_player_state(world, owner, &player)) {
    return true;
  }
  const void *const owner_root = player.player;
  PlayerStateView other_player;
  if (!read_player_state(world, 1 - owner, &other_player) || other_player.player == owner_root ||
      read_object_field<void *>(owner_root, 0x10) != object_manager) {
    return true;
  }

  // The world slot is the authoritative owner-root relation. Player roots are
  // not guaranteed to be members of the public battle-object vector, so a
  // missing public entity key is a validated null rather than a reason to
  // discard controller state. If the pointer is present in the vector, its
  // identity must still pass the exact bounded-identity checks.
  bool owner_root_in_object_vector = false;
  for (std::int32_t slot = 0; slot < object_count; ++slot) {
    if (objects != nullptr && objects[slot] == owner_root) {
      owner_root_in_object_vector = true;
      break;
    }
  }
  resolve_bounded_entity_reference(owner_root, objects, object_count, &result->owner_root);
  if (owner_root_in_object_vector && (!result->owner_root.validated || !result->owner_root.has_value)) {
    result->owner_root = {};
    return true;
  }
  if (!owner_root_in_object_vector) {
    result->owner_root = {};
    result->owner_root.validated = true;
  }
  result->owner_root_valid = true;

  result->ability_runtime_valid = true;
  for (std::int32_t index = 0; index < 2; ++index) {
    const void *const controller =
        read_object_field<const void *>(owner_root, 0x3a0 + static_cast<std::size_t>(index) * sizeof(void *));
    if (controller == nullptr) {
      continue;
    }
    if (read_object_field<const void *>(controller, 0x90) == nullptr) {
      if (!champion_controller_is_exact_empty(owner_root, controller)) {
        result->ability_runtime_valid = false;
        result->ability_count = 0;
        break;
      }
      continue;
    }
    NativeChampionControllerView &ability = result->abilities[result->ability_count];
    if (!read_champion_controller(owner_root, owner, index + 1, controller, objects, object_count, &ability)) {
      result->ability_runtime_valid = false;
      result->ability_count = 0;
      break;
    }
    ++result->ability_count;
  }

  NativeVectorView progress;
  if (player.deck_slots.count < 0 || player.deck_slots.count > 8 ||
      !read_native_vector(owner_root, 0x2e8, kMaxDeckCards, &progress) || progress.count != player.deck_slots.count ||
      (progress.count != 0 &&
       !process_range_is_readable(progress.data, static_cast<std::size_t>(progress.count) * sizeof(std::int32_t)))) {
    return true;
  }
  const auto *progress_values = static_cast<const std::int32_t *>(progress.data);
  for (std::int32_t slot = 0; slot < progress.count; ++slot) {
    if (!read_evolution_slot(player, progress_values, slot, &result->evolution_slots[slot])) {
      result->evolution_slot_count = 0;
      return true;
    }
  }
  result->evolution_slot_count = progress.count;
  result->evolution_runtime_valid = true;
  return true;
}

bool append_native_entity_reference_json(char *response, std::size_t response_size, std::size_t *used,
                                         const NativeEntityReferenceView &reference) {
  JsonWriter json(response, response_size, used);
  if (!reference.validated || !reference.has_value) {
    return json.append("null");
  }
  return json.write(append_native_entity_key_json, reference.native_object_id, reference.owner, reference.object_index,
                    reference.secondary_index);
}

bool append_native_capture_runtime_json(char *response, std::size_t response_size, std::size_t *used,
                                        const NativeCaptureRuntimeStateView &capture) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("schema", "native-capture-runtime.v1");
  json.field("actionDataGlobalId", capture.action_data_global_id);
  json.field("phase", capture.target_count > 0            ? "active"
                      : capture.cooldown_remaining_ms > 0 ? "release_cooldown"
                                                          : "idle");
  json.field("dragDelayMs", capture.drag_delay_ms);
  json.field("grabPauseMs", capture.grab_pause_ms);
  json.field("captureDragTimeMs", capture.capture_drag_time_ms);
  json.field("configuredCooldownMs", capture.configured_cooldown_ms);
  json.field("cooldownRemainingMs", capture.cooldown_remaining_ms);
  json.field("hitFrequencyMs", capture.hit_frequency_ms);
  json.field("hitAccumulatorMs", capture.hit_accumulator_ms);
  json.field("completionResultCurrentUpdate", capture.completion_result_current_update);
  json.field("firstCaptureHandled", capture.first_capture_handled);
  json.begin_array("targets");
  for (std::int32_t index = 0; json.good() && index < capture.target_count; ++index) {
    const NativeCaptureTargetStateView &target = capture.targets[index];
    json.object_element();
    json.field("targetNativeObjectId", target.native_object_id);
    json.record("targetEntityKey", append_native_entity_reference_json, target.entity);
    json.field("targetResolved", target.entity.validated && target.entity.has_value);
    json.field("phase", native_capture_target_phase_label(target.phase));
    json.field("elapsedMs", target.elapsed_ms);
    json.nullable("phaseBudgetRemainingMs", target.phase_budget_remaining_ms, target.phase_budget_remaining_known);
    json.end_object();
  }
  json.end_array();
  json.field("status", "authoritative");
  return json.end_object();
}

bool append_native_threshold_relocation_json(char *response, std::size_t response_size, std::size_t *used,
                                             const NativeThresholdRelocationStateView &relocation) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("schema", "native-threshold-relocation-runtime.v1");
  json.field("actionDataGlobalId", relocation.action_data_global_id);
  json.field("phase", native_threshold_relocation_phase_label(relocation.phase));
  json.field("stage", relocation.stage);
  json.field("relocationIndex", relocation.relocation_index);
  json.field("hideDurationMs", relocation.hide_duration_ms);
  json.field("remainingMs", relocation.remaining_ms);
  json.field("burrowed", relocation.burrowed);
  json.begin_array("thresholdsPercent");
  for (std::int32_t index = 0; json.good() && index < relocation.threshold_count; ++index) {
    json.element(relocation.thresholds[index]);
  }
  json.end_array();
  json.field("status", "authoritative");
  return json.end_object();
}

bool append_native_projectile_json(char *response, std::size_t response_size, std::size_t *used,
                                   const NativeProjectileStateView &projectile) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("projectileDataGlobalId", projectile.data_global_id);
  json.record("sourceEntityKey", append_native_entity_reference_json, projectile.source);
  json.field("sourceEntityValidated", projectile.source.validated);
  json.record("targetEntityKey", append_native_entity_reference_json, projectile.target);
  json.field("targetEntityValidated", projectile.target.validated);
  json.record("homingTargetEntityKey", append_native_entity_reference_json, projectile.homing_target);
  json.field("homingTargetEntityValidated", projectile.homing_target.validated);
  json.field("destinationX", projectile.destination_x);
  json.field("destinationY", projectile.destination_y);
  json.field("terminal", projectile.terminal);
  json.field("nativePhase", projectile.terminal ? "terminal_or_finished_processing" : "in_flight");
  json.nullable("dragStage", projectile.drag_stage == 0 ? "outbound" : "drag_back_active", projectile.drag_stage >= 0);
  return json.end_object();
}

bool append_native_entity_resource_json(char *response, std::size_t response_size, std::size_t *used,
                                        const NativeEntityResourceStateView &resource) {
  JsonWriter json(response, response_size, used);
  return json.append("{\"kind\":\"extra_spawn_accumulator\","
                     "\"currentRaw\":%d,\"capacityRaw\":%d,"
                     "\"baseRaw\":%d,\"limitRaw\":%d,"
                     "\"normalized\":%.9g,\"status\":\"authoritative\"}",
                     resource.current_raw, resource.capacity_raw, resource.base_raw, resource.limit_raw,
                     static_cast<double>(resource.current_raw) / static_cast<double>(resource.capacity_raw));
}

bool append_native_periodic_attack_modifier_json(char *response, std::size_t response_size, std::size_t *used,
                                                 const NativePeriodicAttackModifierStateView &modifier) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("schema", "native-periodic-attack-modifier-runtime.v1");
  json.field("actionDataGlobalId", modifier.action_data_global_id);
  json.field("phase", native_periodic_attack_modifier_phase_label(modifier.phase));
  json.field("periodAttacks", modifier.period_attacks);
  json.field("completedAttacks", modifier.completed_attacks);
  json.field("addedDamageRaw", modifier.added_damage_raw);
  json.field("lingerDurationMs", modifier.linger_duration_ms);
  json.field("lingerRemainingMs", modifier.linger_remaining_ms);
  json.field("sourceNativeObjectId", modifier.source_native_object_id);
  json.record("sourceEntityKey", append_native_entity_reference_json, modifier.source);
  json.field("sourceResolved", modifier.source.validated && modifier.source.has_value);
  json.field("status", "authoritative");
  return json.end_object();
}

bool append_native_player_runtime_json(char *response, std::size_t response_size, std::size_t *used, std::int32_t owner,
                                       const NativePlayerRuntimeView &player) {
  JsonWriter json(response, response_size, used);
  json.begin_object();
  json.field("owner", owner);
  json.field("ownerRootValidated", player.owner_root_valid);
  json.record("ownerEntityKey", append_native_entity_reference_json, player.owner_root);
  if (!player.owner_root_valid || !player.ability_runtime_valid) {
    json.raw("abilityRuntime", "null");
  } else {
    json.begin_array("abilityRuntime");
    for (std::int32_t index = 0; json.good() && index < player.ability_count; ++index) {
      const NativeChampionControllerView &ability = player.abilities[index];
      const char *label = ability_button_state_label(ability.button_state);
      if (!label)
        return false;
      json.object_element();
      json.field("controllerSlot", ability.controller_slot);
      json.field("actionDataGlobalId", ability.action_data_global_id);
      json.record("actionDataName", append_json_utf8_string, ability.action_data_name, ability.action_data_name_size);
      json.field("selectedCharacterDataGlobalId", ability.selected_character_global_id);
      json.field("remainingCooldownMs", ability.remaining_cooldown_ms);
      json.field("configuredCooldownMs", ability.configured_cooldown_ms);
      json.field("remainingChargesRaw", ability.remaining_charges_raw);
      json.field("maxCharges", ability.max_charges);
      json.field("buttonState", ability.button_state);
      json.field("buttonStateLabel", label);
      json.field("available", ability_button_state_is_queueable(ability.button_state));
      json.begin_array("championEntityKeys");
      for (std::int32_t champion = 0; json.good() && champion < ability.champion_count; ++champion) {
        json.append("%s", champion == 0 ? "" : ",");
        json.write(append_native_entity_reference_json, ability.champions[champion]);
      }
      json.end_array();
      json.end_object();
    }
    json.end_array();
  }
  if (!player.owner_root_valid || !player.evolution_runtime_valid) {
    json.raw("evolutionRuntime", "null");
  } else {
    json.begin_array("evolutionRuntime");
    for (std::int32_t index = 0; json.good() && index < player.evolution_slot_count; ++index) {
      const NativeEvolutionSlotView &evolution = player.evolution_slots[index];
      json.object_element();
      json.field("deckSlot", evolution.deck_slot);
      json.field("cardId", evolution.card_id);
      json.field("baseSpellGlobalId", evolution.base_spell_global_id);
      json.field("evolvable", evolution.evolvable);
      json.nullable("evolutionFormGlobalId", evolution.evolution_form_global_id, evolution.evolvable);
      json.field("progress", evolution.progress);
      json.nullable("cycleRequired", evolution.cycle_required, evolution.evolvable);
      json.nullable("cycleRemaining", evolution.cycle_remaining, evolution.evolvable);
      json.nullable("ready", evolution.ready, evolution.evolvable);
      json.end_object();
    }
    json.end_array();
  }
  return json.end_object();
}

bool read_hitpoints(const void *object, std::int32_t *hitpoints, std::int32_t *maximum_hitpoints) {
  if (object == nullptr || hitpoints == nullptr || maximum_hitpoints == nullptr) {
    return false;
  }

  void **components = read_object_field<void **>(object, 0x18);
  const std::int32_t capacity = read_object_field<std::int32_t>(object, 0x20);
  const std::int32_t count = read_object_field<std::int32_t>(object, 0x24);
  if (count < 0 || capacity < count || capacity > kMaxObjectComponents || (count != 0 && components == nullptr)) {
    return false;
  }

  // v15 LogicGameObject keeps its hitpoint component in slot 2. The first
  // word of a component is a MuMu-translated type address and changes after
  // a process restart, so validate the stable component->owner backlink
  // instead of hard-coding that translated address.
  if (count <= 2 || components[2] == nullptr || read_object_field<const void *>(components[2], 0x08) != object) {
    return false;
  }
  *hitpoints = read_object_field<std::int32_t>(components[2], 0x10);
  *maximum_hitpoints = read_object_field<std::int32_t>(components[2], 0x14);
  return *maximum_hitpoints >= 0;
}

bool read_shield(const void *object, std::int32_t *shield, std::int32_t *maximum_shield) {
  if (object == nullptr || shield == nullptr || maximum_shield == nullptr) {
    return false;
  }

  void **components = read_object_field<void **>(object, 0x18);
  const std::int32_t capacity = read_object_field<std::int32_t>(object, 0x20);
  const std::int32_t count = read_object_field<std::int32_t>(object, 0x24);
  if (count <= 2 || capacity < count || capacity > kMaxObjectComponents || components == nullptr ||
      components[2] == nullptr || read_object_field<const void *>(components[2], 0x08) != object ||
      !process_range_is_readable(components[2], 0x30)) {
    return false;
  }

  // Exact-build static paths load ShieldHitpoints into data+0x7ac
  // (libg+d9712c), expose it at libg+d98360, initialize component type 2
  // max shield at +0x2c (libg+f6261c), and copy it to current shield +0x28
  // in the constructor (libg+f624c0). Named live golden: Dark Prince
  // cardId 26000027, generation/stateEpoch 13, started at tick 131 with
  // 150/150. Enemy-tower hits produced 100/150 at tick 234, 50/150 at
  // tick 247, 0/150 at tick 261 while HP stayed 750, then HP became 700
  // at tick 277 while shield remained 0/150.
  *shield = read_object_field<std::int32_t>(components[2], 0x28);
  *maximum_shield = read_object_field<std::int32_t>(components[2], 0x2c);
  return *maximum_shield >= 0 && *shield >= 0 && *shield <= *maximum_shield;
}

#include "combat_event_telemetry.inc"
#include "visibility_runtime_telemetry.inc"
#include "phase_runtime_telemetry.inc"
#include "special_movement_runtime_telemetry.inc"
#include "action_movement_runtime_telemetry.inc"
#include "character_state_runtime_telemetry.inc"
#include "remaining_runtime_telemetry.inc"
#include "tower_troop_runtime_telemetry.inc"

void begin_runtime_telemetry_epoch(void *game_manager, std::uint64_t generation, std::uint64_t state_epoch) {
  // Close the combat capture gate first so no new phase callback can obtain
  // an old context while the phase ring/cache identity is reset.
  end_combat_event_epoch(nullptr);
  begin_phase_runtime_epoch(generation, state_epoch);
  begin_special_movement_runtime_epoch(generation, state_epoch);
  begin_action_movement_runtime_epoch(generation, state_epoch);
  begin_character_state_runtime_epoch(generation, state_epoch);
  begin_tower_troop_runtime_epoch(generation, state_epoch);
  begin_combat_event_epoch(game_manager, generation, state_epoch);
}

std::uint64_t runtime_state_epoch_for_manager_locked(void *manager, std::uint64_t generation) {
  if (manager == g_controlled_manager) {
    return g_state_epoch;
  }
  if (manager != nullptr &&
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      g_native_render_manager.load(std::memory_order_acquire) == manager &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation &&
      g_native_render_state_epoch != 0) {
    return g_native_render_state_epoch;
  }
  return generation;
}

bool build_observation_json(void *manager, std::uint64_t generation, char *response, std::size_t response_size,
                            bool *complete_out = nullptr) {
  if (complete_out != nullptr) {
    *complete_out = false;
  }
  if (manager == nullptr || response == nullptr || response_size < 256) {
    return false;
  }

  void *world = read_object_field<void *>(manager, 0xa8);
  void *king_tower = world == nullptr ? nullptr : read_object_field<void *>(world, 0xe0);
  void *object_manager = king_tower == nullptr ? nullptr : read_object_field<void *>(king_tower, 0x10);
  if (world == nullptr || king_tower == nullptr || object_manager == nullptr) {
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"active object graph is not ready\"}");
    return false;
  }

  void **objects = read_object_field<void **>(object_manager, 0x08);
  const std::int32_t capacity = read_object_field<std::int32_t>(object_manager, 0x10);
  const std::int32_t count = read_object_field<std::int32_t>(object_manager, 0x14);
  if (count < 0 || capacity < count || capacity > 100000 || (count != 0 && objects == nullptr)) {
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"active object vector is invalid\"}");
    return false;
  }

  void *command_queue = read_object_field<void *>(manager, 0x38);
  const std::int32_t queued_commands =
      command_queue == nullptr ? -1 : read_object_field<std::int32_t>(command_queue, 0x14);
  const std::int32_t tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
  BattleResultView battle_result;
  if (!read_battle_result(world, &battle_result)) {
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"native battle result is invalid\"}");
    return false;
  }
  const bool ended = battle_result.finalized || (manager == g_controlled_manager && g_control_ended);
  const bool headless_observation = manager == g_controlled_manager;
  const std::uint64_t observation_state_epoch = runtime_state_epoch_for_manager_locked(manager, generation);
  char world_result_json[32] = "null";
  char winner_json[32] = "null";
  if (battle_result.finalized) {
    std::snprintf(world_result_json, sizeof(world_result_json), "%d", battle_result.world_result_raw);
    if (battle_result.world_result_raw == 0 || battle_result.world_result_raw == 1) {
      std::snprintf(winner_json, sizeof(winner_json), "%d", battle_result.world_result_raw);
    }
  }
  std::size_t used = 0;
  JsonWriter json(response, response_size, used);
  if (!json.append("{\"ok\":true,\"generation\":%llu,\"stateEpoch\":%llu,"
                   "\"tick\":%d,\"steps\":%llu,\"ended\":%s,\"finalized\":%s,"
                   "\"worldResult\":%s,\"worldResultRaw\":%d,\"winner\":%s,"
                   "\"crownsRaw\":[%d,%d],\"snapshotHandles\":%zu,"
                   "\"count\":%d,\"queuedCommands\":%d,\"players\":[",
                   static_cast<unsigned long long>(generation),
                   static_cast<unsigned long long>(observation_state_epoch), tick,
                   static_cast<unsigned long long>(g_completed_steps), ended ? "true" : "false",
                   battle_result.finalized ? "true" : "false", world_result_json, battle_result.world_result_raw,
                   winner_json, battle_result.crowns_raw[0], battle_result.crowns_raw[1],
                   headless_observation ? count_stored_snapshots_locked() : 0, count, queued_commands)) {
    return false;
  }

  for (std::int32_t owner = 0; owner < kPlayerCount; ++owner) {
    PlayerStateView player;
    if (!read_player_state(world, owner, &player)) {
      std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"native player state is invalid\",\"owner\":%d}",
                    owner);
      return false;
    }
    player.crowns_raw = battle_result.crowns_raw[owner];
    const std::uint64_t account_id = (static_cast<std::uint64_t>(player.account_id_high) << 32) | player.account_id_low;
    if (!json.append("%s{\"owner\":%d,\"accountId\":%llu,\"accountIdHigh\":%u,"
                     "\"accountIdLow\":%u,\"elixirRaw\":%d,\"elixir\":%.4f,"
                     "\"crownsRaw\":%d,\"hand\":[",
                     owner == 0 ? "" : ",", owner, static_cast<unsigned long long>(account_id), player.account_id_high,
                     player.account_id_low, player.elixir_raw, static_cast<double>(player.elixir_raw) / 10000.0,
                     player.crowns_raw)) {
      return false;
    }

    auto *hand_slots = static_cast<const std::int32_t *>(player.hand.data);
    std::int32_t emitted_hand_count = 0;
    for (std::int32_t hand_index = 0; hand_index < player.hand.count; ++hand_index) {
      // Hero/champion cycling can expose a bounded -1 hand hole while
      // the replacement card is pending. It is not a corrupt deck
      // pointer and must not invalidate the rest of the observation.
      if (hand_slots[hand_index] < 0) {
        continue;
      }
      CardStateView card;
      const bool selection_valid = read_card_state(player, hand_slots[hand_index], true, &card);
      if (!selection_valid && !read_card_state(player, hand_slots[hand_index], false, &card)) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"error\":\"native hand card is invalid\","
                      "\"owner\":%d,\"handIndex\":%d,\"deckSlotRaw\":%d,"
                      "\"deckSlotCount\":%d}",
                      owner, hand_index, hand_slots[hand_index], player.deck_slots.count);
        return false;
      }
      const bool appended =
          selection_valid ? json.append("%s{\"handIndex\":%d,\"deckSlot\":%d,\"cardId\":%d,"
                                        "\"commandCardId\":%d,"
                                        "\"cardParameter\":%u,\"cost\":%d}",
                                        emitted_hand_count == 0 ? "" : ",", hand_index, card.deck_slot, card.card_id,
                                        card.command_card_id, card.card_parameter, card.cost)
                          : json.append("%s{\"handIndex\":%d,\"deckSlot\":%d,\"cardId\":%d,"
                                        "\"cardParameter\":null,\"cost\":null}",
                                        emitted_hand_count == 0 ? "" : ",", hand_index, card.deck_slot, card.card_id);
      if (!appended) {
        return false;
      }
      ++emitted_hand_count;
    }

    if (!json.append("],\"cycle\":[")) {
      return false;
    }
    auto *cycle_slots = static_cast<const std::int32_t *>(player.cycle.data);
    for (std::int32_t cycle_index = 0; cycle_index < player.cycle.count; ++cycle_index) {
      CardStateView card;
      if (!read_card_state(player, cycle_slots[cycle_index], false, &card) ||
          !json.append("%s{\"cycleIndex\":%d,\"deckSlot\":%d,\"cardId\":%d}", cycle_index == 0 ? "" : ",", cycle_index,
                       card.deck_slot, card.card_id)) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"error\":\"native cycle card is invalid\","
                      "\"owner\":%d,\"cycleIndex\":%d}",
                      owner, cycle_index);
        return false;
      }
    }

    if (!json.append("],\"nextCard\":")) {
      return false;
    }
    if (player.cycle.count == 0) {
      if (!json.append("null")) {
        return false;
      }
    } else {
      CardStateView next_card;
      if (!read_card_state(player, cycle_slots[0], false, &next_card) ||
          !json.append("{\"deckSlot\":%d,\"cardId\":%d}", next_card.deck_slot, next_card.card_id)) {
        return false;
      }
    }

    if (!json.append(",\"deck\":[")) {
      return false;
    }
    for (std::int32_t deck_slot = 0; deck_slot < player.deck_slots.count; ++deck_slot) {
      CardStateView card;
      if (!read_card_state(player, deck_slot, true, &card) ||
          !json.append("%s{\"deckSlot\":%d,\"cardId\":%d,"
                       "\"commandCardId\":%d,\"cardParameter\":%u,\"cost\":%d}",
                       deck_slot == 0 ? "" : ",", card.deck_slot, card.card_id, card.command_card_id,
                       card.card_parameter, card.cost)) {
        return false;
      }
    }
    if (!json.append("]}")) {
      return false;
    }
  }

  if (!json.append("],\"objects\":[")) {
    return false;
  }

  std::int32_t returned = 0;
  bool truncated = count > kMaxObservedObjects;
  const std::int32_t limit = count < kMaxObservedObjects ? count : kMaxObservedObjects;
  for (std::int32_t index = 0; index < limit; ++index) {
    void *object = objects[index];
    char item[320] = {};
    int written = 0;
    if (object == nullptr) {
      written = std::snprintf(item, sizeof(item), "%s{\"slot\":%d,\"null\":true}", returned == 0 ? "" : ",", index);
    } else {
      NativeEntityIdentityView identity;
      if (!read_native_entity_identity(object, &identity) ||
          !native_object_id_is_unique_in_vector(object, objects, count, identity.native_object_id)) {
        return false;
      }
      const std::uint32_t native_object_id = identity.native_object_id;
      const std::int32_t owner = identity.owner;
      const std::int32_t position_x = read_object_field<std::int32_t>(object, 0x7c);
      const std::int32_t position_y = read_object_field<std::int32_t>(object, 0x80);
      const std::int32_t target_x = read_object_field<std::int32_t>(object, 0x84);
      const std::int32_t target_y = read_object_field<std::int32_t>(object, 0x88);
      const std::int32_t object_index = identity.object_index;
      const std::int32_t secondary_index = identity.secondary_index;
      const std::int32_t card_id = read_object_field<std::int32_t>(object, 0xac);
      std::int32_t hitpoints = 0;
      std::int32_t maximum_hitpoints = 0;
      const bool has_hitpoints = read_hitpoints(object, &hitpoints, &maximum_hitpoints);
      if (has_hitpoints) {
        written =
            std::snprintf(item, sizeof(item),
                          "%s{\"slot\":%d,\"nativeObjectId\":%u,\"owner\":%d,\"cardId\":%d,\"x\":%d,\"y\":%d,"
                          "\"targetX\":%d,\"targetY\":%d,\"objectIndex\":%d,"
                          "\"secondaryIndex\":%d,\"hp\":%d,\"maxHp\":%d}",
                          returned == 0 ? "" : ",", index, native_object_id, owner, card_id, position_x, position_y,
                          target_x, target_y, object_index, secondary_index, hitpoints, maximum_hitpoints);
      } else {
        written = std::snprintf(item, sizeof(item),
                                "%s{\"slot\":%d,\"nativeObjectId\":%u,\"owner\":%d,\"cardId\":%d,\"x\":%d,\"y\":%d,"
                                "\"targetX\":%d,\"targetY\":%d,\"objectIndex\":%d,"
                                "\"secondaryIndex\":%d,\"hp\":null,\"maxHp\":null}",
                                returned == 0 ? "" : ",", index, native_object_id, owner, card_id, position_x,
                                position_y, target_x, target_y, object_index, secondary_index);
      }
    }
    if (written < 0 || static_cast<std::size_t>(written) >= sizeof(item) ||
        used + static_cast<std::size_t>(written) + 96 >= response_size) {
      truncated = true;
      break;
    }
    std::memcpy(response + used, item, static_cast<std::size_t>(written));
    used += static_cast<std::size_t>(written);
    response[used] = '\0';
    ++returned;
  }

  const int tail_written = std::snprintf(response + used, response_size - used, "],\"returned\":%d,\"truncated\":%s}",
                                         returned, truncated ? "true" : "false");
  const bool encoded = tail_written >= 0 && static_cast<std::size_t>(tail_written) < response_size - used;
  if (encoded && complete_out != nullptr) {
    *complete_out = !truncated;
  }
  return encoded;
}

bool build_rich_observation_json(void *manager, std::uint64_t generation, char *response, std::size_t response_size,
                                 bool *complete_out = nullptr) {
  if (complete_out != nullptr) {
    *complete_out = false;
  }
  if (manager == nullptr || response == nullptr || response_size < 4096) {
    return false;
  }

  void *world = read_object_field<void *>(manager, 0xa8);
  void *king_tower = world == nullptr ? nullptr : read_object_field<void *>(world, 0xe0);
  void *object_manager = king_tower == nullptr ? nullptr : read_object_field<void *>(king_tower, 0x10);
  if (world == nullptr || king_tower == nullptr || object_manager == nullptr) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"native-rich-telemetry.v3\","
                  "\"error\":\"active object graph is not ready\"}");
    return false;
  }

  void **objects = read_object_field<void **>(object_manager, 0x08);
  const std::int32_t capacity = read_object_field<std::int32_t>(object_manager, 0x10);
  const std::int32_t count = read_object_field<std::int32_t>(object_manager, 0x14);
  if (count < 0 || capacity < count || capacity > 100000 || (count != 0 && objects == nullptr)) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"native-rich-telemetry.v3\","
                  "\"error\":\"active object vector is invalid\"}");
    return false;
  }

  const std::int32_t tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
  const std::uint64_t state_epoch = runtime_state_epoch_for_manager_locked(manager, generation);
  std::size_t used = 0;
  JsonWriter json(response, response_size, used);
  if (!json.append(
          "{\"ok\":true,\"schema\":\"%s\","
          "\"statusEnum\":[\"authoritative\",\"derived\",\"pending\",\"unavailable\"],"
          "\"generation\":%llu,\"stateEpoch\":%llu,\"tick\":%d,\"provenance\":{"
          "\"slot\":{\"status\":\"derived\",\"source\":\"object-vector-index\","
          "\"validation\":\"bounded-object-vector\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"nativeObjectId\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x08\","
          "\"validation\":\"nonzero;unique-in-current-bounded-object-vector;named-live-fresh-towers-5000000-through-"
          "5000005-and-cross-projectile-damage-5000005\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"entityKey\":{\"status\":\"derived\",\"source\":\"tagged-nativeObjectId\","
          "\"validation\":\"owner-minus2-nativeObjectId;nativeObjectId-nonzero-and-current-vector-unique;raw-index-"
          "tuples-are-not-identities;slot-and-pointer-never-used\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"owner\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x78\","
          "\"validation\":\"exact-build-runtime-observation\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"cardId\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0xac\","
          "\"validation\":\"exact-build-runtime-observation\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"dataGlobalId\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x48->LogicData+0x10\","
          "\"validation\":\"positive-uint32-logic-data-identity\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"x\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x7c\","
          "\"validation\":\"exact-build-runtime-observation\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"y\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x80\","
          "\"validation\":\"exact-build-runtime-observation\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"targetX\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x84\","
          "\"validation\":\"coordinate-only;not-target-entity\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"targetY\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x88\","
          "\"validation\":\"coordinate-only;not-target-entity\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"targetEntityKey\":{\"status\":\"authoritative\","
          "\"source\":\"engine-component-type-0+0x10/current-object-vector-identity\","
          "\"validation\":\"owner-backlink;engine-type-0;readable-field;bounded-pointer-identity;named-live-golden-"
          "giant-26000003-generation-1-state-epoch-1-ticks-131-180-530\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"targetEntityValidated\":{\"status\":\"derived\",\"source\":\"targetEntityKey-validation\","
          "\"validation\":\"true-only-for-null-target-or-current-object-vector-match\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"objectIndex\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x94\","
          "\"validation\":\"exact-build-runtime-observation\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"secondaryIndex\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x98\","
          "\"validation\":\"exact-build-runtime-observation\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"hp\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-2+0x10\","
          "\"validation\":\"owner-backlink;engine-type;captured-damage-delta\",\"confidence\":\"high\",\"failClosed\":"
          "true},"
          "\"maxHp\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-2+0x14\","
          "\"validation\":\"owner-backlink;engine-type;nonnegative\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"shield.current\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-2+0x28\","
          "\"validation\":\"owner-backlink;engine-type-2;readable-through-0x2f;0<=current<=max;static-loader-d9712c-"
          "accessor-d98360-init-f6261c-constructor-f624c0;named-live-golden-dark-prince-26000027-generation-13-state-"
          "epoch-13-ticks-131-234-247-261-277\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"shield.max\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-2+0x2c\","
          "\"validation\":\"owner-backlink;engine-type-2;readable-through-0x2f;nonnegative;static-loader-d9712c-"
          "accessor-d98360-init-f6261c-constructor-f624c0;named-live-golden-dark-prince-26000027-generation-13-state-"
          "epoch-13-ticks-131-234-247-261-277\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"attackSequenceStage\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-0+0x20\","
          "\"validation\":\"owner-backlink;engine-type-0;readable-through-0x23;wire-bound-0-255;configured-sequence-"
          "count-is-card-specific;named-live-golden-inferno-dragon-26000037-generation-18-state-epoch-18-observed-"
          "stages-0-1-2-ticks-351-391-497-536-576\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"activeEffects\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-3+0x18/+0x20/+0x24\","
          "\"validation\":\"owner-backlink;engine-type-3;bounded-pointer-vector;child-size-0x70;all-entries-or-null\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"activeEffects.buffGlobalId\":{\"status\":\"authoritative\",\"source\":\"buff-asset+0x40\","
          "\"validation\":\"asset-vtable-libg+0x1882500;class-getter-libg+d961f4;static-layout-libg+dc2570;positive-"
          "global-id;ice-wizard-slow-down-golden-9000003\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"activeEffects.name\":{\"status\":\"authoritative\",\"source\":\"buff-asset+0x28-LibgString\","
          "\"validation\":\"static-layout-libg+dc25ec;readable;1-128-byte-valid-utf8;named-goldens-Rage-Freeze-"
          "IceWizardSlowDown-Invisibility\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"activeEffects.remainingMs\":{\"status\":\"authoritative\",\"source\":\"type-3-child+0x08\","
          "\"validation\":\"range-minus-1-or-nonnegative;minus-1-means-non-expiring;ice-wizard-slow-down-countdown-"
          "minus-50-per-tick-and-hit-renewal-to-about-2500ms\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"activeEffects.sourceEntityKey\":{\"status\":\"authoritative\",\"source\":\"type-3-child+0x38/"
          "current-object-vector-identity\","
          "\"validation\":\"exposed-only-for-bounded-pointer-identity;absolute-pointer-not-exposed;ice-wizard-slow-"
          "down-source-golden\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"activeEffects.sourceEntityValidated\":{\"status\":\"derived\",\"source\":\"type-3-child+0x38-pointer-"
          "validation\","
          "\"validation\":\"true-for-null-or-current-bounded-vector-member;false-for-nonmember\",\"confidence\":"
          "\"high\",\"failClosed\":true},"
          "\"invisibleCount\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-3+0x34\","
          "\"validation\":\"owner-backlink;engine-type-3;range-0-to-active-entry-count;named-live-golden-royal-ghost-"
          "26000050-generation-19-state-epoch-19-ticks-131-276-537-569\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"visibilityState\":{\"status\":\"derived\",\"source\":\"invisibleCount\","
          "\"validation\":\"invisible-iff-count-greater-than-zero;owner-relative-render-visibility-not-claimed\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.container\":{\"status\":\"authoritative\","
          "\"source\":\"LogicGameObject+0x18/+0x20/+0x24\","
          "\"validation\":\"count-capacity-pointer-bounds\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.present\":{\"status\":\"authoritative\",\"source\":\"component-vector-entry\","
          "\"validation\":\"bounded-component-vector\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.ownerBacklinkValid\":{\"status\":\"authoritative\","
          "\"source\":\"component+0x08\",\"validation\":\"equals-containing-object\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.vtableLibgOffset\":{\"status\":\"authoritative\","
          "\"source\":\"validated-vtable-minus-libg-base\","
          "\"validation\":\"inside-readable-libg-segment;named-live-golden-generation-1-tower-tick-0-and-giant-"
          "26000003-tick-131;absolute-pointer-not-exposed\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.typeGetterLibgOffset\":{\"status\":\"authoritative\","
          "\"source\":\"validated-vtable+0x20-minus-libg-base\","
          "\"validation\":\"inside-executable-libg-segment;named-live-golden-generation-1-tower-tick-0-and-giant-"
          "26000003-tick-131;absolute-pointer-not-exposed\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.engineType\":{\"status\":\"authoritative\","
          "\"source\":\"libg+0x113b194/vtable+0x20\","
          "\"validation\":\"owner-backlink;readable-vtable;executable-getter;range-0-31;named-live-golden-generation-1-"
          "tower-tick-0-and-giant-26000003-tick-131\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.typeValid\":{\"status\":\"derived\",\"source\":\"engineType-validation\","
          "\"validation\":\"all-guards-passed\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"components.slots.slotMatchesType\":{\"status\":\"derived\",\"source\":\"slot-equals-engineType\","
          "\"validation\":\"only-when-typeValid\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"projectile\":{\"status\":\"derived\",\"source\":\"exact-kind-4-LogicProjectile\","
          "\"validation\":\"exact-vtable-kind-getter-and-LogicProjectileData-type-0x0a;static-high-no-named-live-"
          "transition\",\"confidence\":\"medium\",\"failClosed\":true},"
          "\"projectile.projectileDataGlobalId\":{\"status\":\"derived\",\"source\":\"LogicProjectileData+0x40\","
          "\"validation\":\"positive-global-id-after-exact-data-type-check\",\"confidence\":\"medium\",\"failClosed\":"
          "true},"
          "\"projectile.sourceEntityKey\":{\"status\":\"derived\",\"source\":\"LogicProjectile+0x100\","
          "\"validation\":\"same-bounded-object-vector-identity-or-null\",\"confidence\":\"medium\",\"failClosed\":"
          "true},"
          "\"projectile.targetEntityKey\":{\"status\":\"derived\",\"source\":\"LogicProjectile+0x108\","
          "\"validation\":\"same-bounded-object-vector-identity-or-null\",\"confidence\":\"medium\",\"failClosed\":"
          "true},"
          "\"projectile.homingTargetEntityKey\":{\"status\":\"derived\",\"source\":\"LogicProjectile+0x110\","
          "\"validation\":\"same-bounded-object-vector-identity-or-null;destination-getters-prefer-this-target;cleared-"
          "by-object-removal-callback\",\"confidence\":\"medium\",\"failClosed\":true},"
          "\"projectile.destination\":{\"status\":\"derived\",\"source\":\"LogicProjectile+0x120/+0x124\","
          "\"validation\":\"current-flight-destination-coordinates-only\",\"confidence\":\"medium\",\"failClosed\":"
          "true},"
          "\"projectile.terminal\":{\"status\":\"derived\",\"source\":\"LogicProjectile+0x170\","
          "\"validation\":\"strict-bit;terminal-reason-not-claimed\",\"confidence\":\"medium\",\"failClosed\":true},"
          "\"projectile.nativePhase\":{\"status\":\"derived\",\"source\":\"projectile.terminal\","
          "\"validation\":\"zero-in-flight;one-terminal-or-finished-processing;never-impact-or-expiry\",\"confidence\":"
          "\"medium\",\"failClosed\":true},"
          "\"projectile.dragStage\":{\"status\":\"authoritative\","
          "\"source\":\"LogicProjectile+0x189 gated by LogicProjectileData+0x1b0 DragBackSpeed\","
          "\"validation\":\"exact-projectile-vtable-and-data-type;only-drag-capable-data;strict-0-or-1-persisted-"
          "latch\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"entityResourceRuntime\":{\"status\":\"authoritative\","
          "\"source\":\"LogicCharacter+0x1f0 gated by CharacterData+0x710 CharacterAbilityData+0x104/+0x108\","
          "\"validation\":\"exact-character-and-ability-data-vtables;base-positive;limit-greater-than-base;raw-bounded-"
          "by-limit-minus-base\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"periodicAttackModifierRuntime\":{\"status\":\"authoritative\","
          "\"source\":\"exact periodic-attack-modifier runtime/action-data vtables in carrier action queue\","
          "\"validation\":\"owner-backlink;period-bounded-counter;native-lifecycle-state;source-identity;linger-"
          "timer\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"captureRuntime\":{\"status\":\"authoritative\","
          "\"source\":\"exact ActionCaptureCharacter runtime/action-data vtables and owner action queue\","
          "\"validation\":\"owner-backlink;bounded-active-completed-elapsed-vectors;completed-subset;live-evo-positive-"
          "and-basic-negative\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"thresholdRelocationRuntime\":{\"status\":\"authoritative\","
          "\"source\":\"exact threshold-relocation runtime/action-data vtables and owner action queue\","
          "\"validation\":\"owner-backlink;bounded-threshold-vector;stage-derived-from-threshold-count;live-two-cycle-"
          "positive-and-basic-negative\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"players.ownerRoot\":{\"status\":\"derived\",\"source\":\"exact-kind-5-player-owner-root\","
          "\"validation\":\"unique-current-vector-member;exact-vtable-kind-manager-and-owner-selector\",\"confidence\":"
          "\"medium\",\"failClosed\":true},"
          "\"players.abilityRuntime\":{\"status\":\"derived\",\"source\":\"owner+0x3a0/+0x3a8-champion-controllers\","
          "\"validation\":\"exact-controller-action-data-selected-character-and-champion-vector-joins;static-high\","
          "\"confidence\":\"medium\",\"failClosed\":true},"
          "\"players.abilityRuntime.buttonState\":{\"status\":\"derived\",\"source\":\"champion-controller+0x98\","
          "\"validation\":\"exact-enum-0-through-14;available-only-for-Ready-value-2\",\"confidence\":\"medium\","
          "\"failClosed\":true},"
          "\"players.abilityRuntime.cooldown\":{\"status\":\"derived\",\"source\":\"champion-controller+0x78/+0x7c\","
          "\"validation\":\"zero-to-configured-and-configured-equals-character-ability-data+0xb4\",\"confidence\":"
          "\"medium\",\"failClosed\":true},"
          "\"players.abilityRuntime.charges\":{\"status\":\"derived\",\"source\":\"champion-controller+0x80\","
          "\"validation\":\"minus-one-unlimited-or-bounded-by-character-ability-data+0xac\",\"confidence\":\"medium\","
          "\"failClosed\":true},"
          "\"players.evolutionRuntime\":{\"status\":\"derived\",\"source\":\"owner+0x2e8-progress-vector-and-exact-"
          "deck-slot-join\","
          "\"validation\":\"bounded-slot-vector;exact-base-spell;unique-form-1-candidate;static-high\",\"confidence\":"
          "\"medium\",\"failClosed\":true},"
          "\"players.evolutionRuntime.cycle\":{\"status\":\"derived\",\"source\":\"progress-and-EvoForm-DarkElixirCost+"
          "0x284\","
          "\"validation\":\"zero-to-threshold;remaining-max-threshold-minus-progress-zero;ready-at-threshold\","
          "\"confidence\":\"medium\",\"failClosed\":true},"
          "\"players.evolutionRuntime.playedForm\":{\"status\":\"unavailable\",\"source\":\"consumed-runtime-card-"
          "descriptor-not-retained\","
          "\"validation\":\"no-state-snapshot-substitution;no-per-entity-evolved-claim\",\"confidence\":\"none\","
          "\"failClosed\":true},"
          "\"towerTroopRuntime\":{\"status\":\"derived\",\"source\":\"exact-ActionBurstAttack-and-ActionChefTower-"
          "runtime\","
          "\"validation\":\"sha-build-id-four-prologues-runtime-and-static-vtables-bounded-object-identities\","
          "\"confidence\":\"high\",\"failClosed\":true}},"
          "\"capabilities\":{"
          "\"entityCore\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject-runtime\","
          "\"validation\":\"exact-build-observe-path\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"targetCoordinates\":{\"status\":\"authoritative\",\"source\":\"LogicGameObject+0x84/+0x88\","
          "\"validation\":\"coordinates-only\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"hitpoints\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-2\","
          "\"validation\":\"captured-live-damage-delta\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"componentInventory\":{\"status\":\"authoritative\",\"source\":\"engine-component-hash-virtual-getter\","
          "\"validation\":\"static-rela-identity;named-live-golden-generation-1-state-epoch-1-tower-component-slots-0-"
          "2-3-tick-0-and-giant-26000003-object-slot-6-component-slots-0-1-2-3-tick-131\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"shield\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-2+0x28/+0x2c\","
          "\"validation\":\"static-producer-consumer-paths;named-live-golden-dark-prince-26000027-generation-13-state-"
          "epoch-13-150-to-0-before-hp-damage;no-shield-objects-are-0/0\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"attackSequenceStage\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-0+0x20\","
          "\"validation\":\"wire-bound-0-255;card-specific-sequence-count;inferno-dragon-observed-0-1-2-damage-ramp-"
          "and-target-reset-golden\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"buffs\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-3-activeEffects\","
          "\"validation\":\"exact-Buff-asset-identity;bounded-all-entry-decoding;named-live-goldens\",\"confidence\":"
          "\"high\",\"failClosed\":true},"
          "\"debuffs\":{\"status\":\"unavailable\",\"source\":\"no-validated-entity-traversal\","
          "\"validation\":\"fail-closed-null\",\"confidence\":\"none\",\"failClosed\":true},"
          "\"slow\":{\"status\":\"derived\",\"source\":\"phaseRuntime-Buff-Speed-and-HitSpeed-consumers\","
          "\"validation\":\"exact-build-f5b3c8-f5b4ac;catalog-identity;fail-closed-envelope\",\"confidence\":\"high\","
          "\"failClosed\":true},"
          "\"rage\":{\"status\":\"derived\",\"source\":\"phaseRuntime-Buff-Speed-and-HitSpeed-consumers\","
          "\"validation\":\"exact-build-f5b3c8-f5b4ac;catalog-identity;fail-closed-envelope\",\"confidence\":\"high\","
          "\"failClosed\":true},"
          "\"stun\":{\"status\":\"derived\",\"source\":\"phaseRuntime-full-stop-Buff-and-consumer-zero\","
          "\"validation\":\"pending-attack-plus-effect-apply-plus-f5b4ac-zero;reset-separate\",\"confidence\":"
          "\"medium\",\"failClosed\":true},"
          "\"freeze\":{\"status\":\"derived\",\"source\":\"phaseRuntime-full-stop-Buff-consumers\","
          "\"validation\":\"exact-Buff-ID-join-and-zero-native-step;fail-closed-envelope\",\"confidence\":\"medium\","
          "\"failClosed\":true},"
          "\"attackPhase\":{\"status\":\"derived\",\"source\":\"phaseRuntime-attested-hook-edges-and-type0-timers\","
          "\"validation\":\"start-release-load-gate-interrupt;ambiguous-snapshot-unknown\",\"confidence\":\"high\","
          "\"failClosed\":true},"
          "\"deployPhase\":{\"status\":\"derived\",\"source\":\"LogicCharacter+0x15c-and-phaseRuntime-deploy-step\","
          "\"validation\":\"live-remaining-budget;wall-time-only-with-known-consumer-step\",\"confidence\":\"high\","
          "\"failClosed\":true},"
          "\"chargeStage\":{\"status\":\"derived\",\"source\":\"type0-AttackSequence-plus-named-Inferno-static-join\","
          "\"validation\":\"Inferno-only;generic-stage-not-promoted;classic-charge-separate\",\"confidence\":\"high\","
          "\"failClosed\":true},"
          "\"targetEntity\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-0+0x10\","
          "\"validation\":\"bounded-current-object-pointer-identity;named-live-golden-giant-26000003-generation-1-"
          "state-epoch-1-null-tick-131-positive-ticks-180-and-530\","
          "\"confidence\":\"high\",\"failClosed\":true},"
          "\"sourceEntity\":{\"status\":\"unavailable\",\"source\":\"no-validated-runtime-layout\","
          "\"validation\":\"fail-closed-null\",\"confidence\":\"none\",\"failClosed\":true},"
          "\"projectile\":{\"status\":\"derived\",\"source\":\"exact-kind-4-LogicProjectile-runtime\","
          "\"validation\":\"exact-build-static-high-layout;bounded-relations;no-impact-inference\",\"confidence\":"
          "\"medium\",\"failClosed\":true},"
          "\"impact\":{\"status\":\"unavailable\",\"source\":\"no-validated-runtime-layout\","
          "\"validation\":\"fail-closed-null\",\"confidence\":\"none\",\"failClosed\":true},"
          "\"visibility\":{\"status\":\"unavailable\",\"source\":\"no-validated-runtime-layout\","
          "\"validation\":\"fail-closed-null\",\"confidence\":\"none\",\"failClosed\":true},"
          "\"invisibility\":{\"status\":\"authoritative\",\"source\":\"engine-component-type-3+0x34\","
          "\"validation\":\"royal-ghost-26000050-generation-19-state-epoch-19-entry-and-count-synchronous-"
          "transitions\",\"confidence\":\"high\",\"failClosed\":true},"
          "\"abilityRuntime\":{\"status\":\"derived\",\"source\":\"exact-player-owner-champion-controller-runtime\","
          "\"validation\":\"static-high-exact-layout-and-bounded-joins;catalog-ID-resolution-host-side\","
          "\"confidence\":\"medium\",\"failClosed\":true},"
          "\"evolutionRuntime\":{\"status\":\"derived\",\"source\":\"exact-player-owner-progress-vector-runtime\","
          "\"validation\":\"static-high-exact-deck-slot-and-EvoForm-threshold-join;played-form-unavailable\","
          "\"confidence\":\"medium\",\"failClosed\":true},"
          "\"towerTroopRuntime\":{\"status\":\"derived\",\"source\":\"exact-tower-troop-action-runtime-hooks\","
          "\"validation\":\"Dagger-charge-and-reload;Chef-delay-contribution-and-pending-target\",\"confidence\":"
          "\"high\",\"failClosed\":true}},"
          "\"count\":%d,\"objects\":[",
          kRichTelemetrySchema, static_cast<unsigned long long>(generation),
          static_cast<unsigned long long>(state_epoch), tick, count)) {
    return false;
  }

  std::int32_t returned = 0;
  bool truncated = count > kMaxRichObservedObjects;
  const std::int32_t limit = count < kMaxRichObservedObjects ? count : kMaxRichObservedObjects;
  for (std::int32_t index = 0; index < limit; ++index) {
    const void *object = objects[index];
    char item[48 * 1024] = {};
    std::size_t item_used = 0;
    JsonWriter item_json(item, sizeof(item), item_used);
    bool item_failed = false;
    if (object == nullptr) {
      if (!item_json.append("%s{\"slot\":%d,\"null\":true}", returned == 0 ? "" : ",", index)) {
        truncated = true;
        break;
      }
    } else {
      NativeEntityIdentityView identity;
      if (!read_native_entity_identity(object, &identity) ||
          !native_object_id_is_unique_in_vector(object, objects, count, identity.native_object_id)) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"native object identity failed validation\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      const std::uint32_t native_object_id = identity.native_object_id;
      const std::int32_t owner = identity.owner;
      const std::int32_t position_x = read_object_field<std::int32_t>(object, 0x7c);
      const std::int32_t position_y = read_object_field<std::int32_t>(object, 0x80);
      const std::int32_t target_x = read_object_field<std::int32_t>(object, 0x84);
      const std::int32_t target_y = read_object_field<std::int32_t>(object, 0x88);
      const std::int32_t object_index = identity.object_index;
      const std::int32_t secondary_index = identity.secondary_index;
      const std::int32_t card_id = read_object_field<std::int32_t>(object, 0xac);
      NativeComponentInventoryView inventory;
      read_native_component_inventory(object, &inventory);
      NativeTargetEntityView target_entity;
      resolve_native_target_entity(object, inventory, objects, count, &target_entity);
      NativeProjectileStateView projectile;
      const NativeProjectileReadStatus projectile_status =
          read_native_projectile_state(object, objects, count, &projectile);
      if (projectile_status == NativeProjectileReadStatus::Invalid) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"exact projectile candidate failed validation\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      NativeEntityResourceStateView entity_resource;
      const NativeEntityResourceReadStatus entity_resource_status =
          read_native_entity_resource_state(object, &entity_resource);
      if (entity_resource_status == NativeEntityResourceReadStatus::Invalid) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"exact entity resource candidate failed validation\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      NativePeriodicAttackModifierStateView periodic_attack_modifier;
      const NativePeriodicAttackModifierReadStatus periodic_attack_modifier_status =
          read_native_periodic_attack_modifier_state(object, objects, count, &periodic_attack_modifier);
      if (periodic_attack_modifier_status == NativePeriodicAttackModifierReadStatus::Invalid) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"exact periodic attack modifier failed validation\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      NativeCaptureRuntimeStateView capture_runtime;
      const NativeCaptureRuntimeReadStatus capture_runtime_status =
          read_native_capture_runtime_state(object, objects, count, &capture_runtime);
      if (capture_runtime_status == NativeCaptureRuntimeReadStatus::Invalid) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"exact capture runtime failed validation\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      NativeThresholdRelocationStateView threshold_relocation;
      const NativeThresholdRelocationReadStatus threshold_relocation_status =
          read_native_threshold_relocation_state(object, &threshold_relocation);
      if (threshold_relocation_status == NativeThresholdRelocationReadStatus::Invalid) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"exact threshold relocation runtime failed validation\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      char target_entity_key_json[96] = "null";
      if (target_entity.validated && target_entity.has_target) {
        if (!format_native_entity_key_json(target_entity_key_json, sizeof(target_entity_key_json),
                                           target_entity.native_object_id, target_entity.owner,
                                           target_entity.object_index, target_entity.secondary_index)) {
          return false;
        }
      }
      char entity_key_json[96] = {};
      if (!format_native_entity_key_json(entity_key_json, sizeof(entity_key_json), native_object_id, owner,
                                         object_index, secondary_index)) {
        return false;
      }

      std::int32_t hitpoints = 0;
      std::int32_t maximum_hitpoints = 0;
      const bool type_two_valid = inventory.valid && inventory.count > 2 && inventory.slots[2].present &&
                                  inventory.slots[2].owner_backlink_valid && inventory.slots[2].type_valid &&
                                  inventory.slots[2].engine_type == 2;
      const bool has_hitpoints = type_two_valid && read_hitpoints(object, &hitpoints, &maximum_hitpoints);
      char hp_json[32] = "null";
      char maximum_hp_json[32] = "null";
      if (has_hitpoints) {
        std::snprintf(hp_json, sizeof(hp_json), "%d", hitpoints);
        std::snprintf(maximum_hp_json, sizeof(maximum_hp_json), "%d", maximum_hitpoints);
      }
      std::int32_t shield = 0;
      std::int32_t maximum_shield = 0;
      const bool has_shield = type_two_valid && read_shield(object, &shield, &maximum_shield);
      char shield_json[128] = "null";
      if (has_shield) {
        std::snprintf(shield_json, sizeof(shield_json), "{\"current\":%d,\"max\":%d,\"status\":\"authoritative\"}",
                      shield, maximum_shield);
      }
      std::int32_t attack_sequence_stage = 0;
      const bool has_attack_sequence_stage = read_attack_sequence_stage(object, inventory, &attack_sequence_stage);
      char attack_sequence_stage_json[16] = "null";
      if (has_attack_sequence_stage) {
        std::snprintf(attack_sequence_stage_json, sizeof(attack_sequence_stage_json), "%d", attack_sequence_stage);
      }
      NativeTypeThreeStateView type_three_state;
      read_native_type_three_state(object, inventory, objects, count, &type_three_state);
      char invisible_count_json[16] = "null";
      const char *visibility_state_json = "null";
      if (type_three_state.header_valid) {
        std::snprintf(invisible_count_json, sizeof(invisible_count_json), "%d", type_three_state.invisible_count);
        visibility_state_json = type_three_state.invisible_count > 0 ? "\"invisible\"" : "\"visible\"";
      }
      NativePhaseSnapshot phase_snapshot;
      const bool has_phase_snapshot =
          read_native_phase_snapshot(object, generation, state_epoch, tick, &phase_snapshot);
      const void *const object_data = read_object_field<const void *>(object, cr_remaining_runtime::kObjectDataOffset);
      std::int32_t signed_data_global_id = 0;
      if (!remaining_data_global_id(object_data, &signed_data_global_id)) {
        std::snprintf(response, response_size,
                      "{\"ok\":false,\"schema\":\"%s\","
                      "\"error\":\"live object lacks exact LogicData identity\","
                      "\"slot\":%d}",
                      kRichTelemetrySchema, index);
        return false;
      }
      const std::uint32_t data_global_id = static_cast<std::uint32_t>(signed_data_global_id);

      if (!item_json.append("%s{\"slot\":%d,\"nativeObjectId\":%u,\"entityKey\":%s,\"owner\":%d,"
                            "\"cardId\":%d,\"dataGlobalId\":%u,\"x\":%d,\"y\":%d,\"targetX\":%d,\"targetY\":%d,"
                            "\"targetEntityKey\":%s,\"targetEntityValidated\":%s,"
                            "\"objectIndex\":%d,\"secondaryIndex\":%d,\"hp\":%s,\"maxHp\":%s,"
                            "\"shield\":%s,\"attackSequenceStage\":%s,"
                            "\"invisibleCount\":%s,\"visibilityState\":%s,\"projectile\":",
                            returned == 0 ? "" : ",", index, native_object_id, entity_key_json, owner, card_id,
                            data_global_id, position_x, position_y, target_x, target_y, target_entity_key_json,
                            target_entity.validated ? "true" : "false", object_index, secondary_index, hp_json,
                            maximum_hp_json, shield_json, attack_sequence_stage_json, invisible_count_json,
                            visibility_state_json)) {
        truncated = true;
        break;
      }

      if (projectile_status == NativeProjectileReadStatus::Valid) {
        item_json.write(append_native_projectile_json, projectile);
      } else {
        item_json.append("null");
      }
      item_json.optional("entityResourceRuntime", entity_resource_status == NativeEntityResourceReadStatus::Valid,
                         append_native_entity_resource_json, entity_resource);
      item_json.optional("periodicAttackModifierRuntime",
                         periodic_attack_modifier_status == NativePeriodicAttackModifierReadStatus::Valid,
                         append_native_periodic_attack_modifier_json, periodic_attack_modifier);
      item_json.optional("captureRuntime", capture_runtime_status == NativeCaptureRuntimeReadStatus::Valid,
                         append_native_capture_runtime_json, capture_runtime);
      item_json.optional("thresholdRelocationRuntime",
                         threshold_relocation_status == NativeThresholdRelocationReadStatus::Valid,
                         append_native_threshold_relocation_json, threshold_relocation);
      item_json.optional("phaseRuntime", has_phase_snapshot, append_native_phase_snapshot_json, phase_snapshot);
      if (!item_json.good()) {
        truncated = true;
        break;
      }
      TowerTroopRuntimeRecord tower_troop_runtime;
      const bool has_tower_troop_runtime =
          latest_tower_troop_runtime(generation, state_epoch, tick, native_object_id, &tower_troop_runtime);
      item_json.optional("towerTroopRuntime", has_tower_troop_runtime, append_tower_troop_runtime_object_json,
                         tower_troop_runtime);
      if (!item_json.append(",\"activeEffects\":")) {
        truncated = true;
        break;
      }

      if (type_three_state.active_effects_valid) {
        if (!item_json.append("[")) {
          item_failed = true;
        }
        for (std::int32_t effect_index = 0; !item_failed && effect_index < type_three_state.count; ++effect_index) {
          const NativeActiveEffectView &effect = type_three_state.effects[effect_index];
          char source_entity_key_json[96] = "null";
          if (effect.source_entity_resolved) {
            if (!format_native_entity_key_json(source_entity_key_json, sizeof(source_entity_key_json),
                                               effect.source_native_object_id, effect.source_owner,
                                               effect.source_object_index, effect.source_secondary_index)) {
              item_failed = true;
              break;
            }
          }
          if (!item_json.append("%s{\"buffGlobalId\":%u,\"name\":", effect_index == 0 ? "" : ",",
                                effect.buff_global_id) ||
              !item_json.write(append_json_utf8_string, effect.name, effect.name_size) ||
              !item_json.append(",\"remainingMs\":%d,\"sourceEntityKey\":%s,"
                                "\"sourceEntityValidated\":%s}",
                                effect.remaining_ms, source_entity_key_json,
                                effect.source_entity_validated ? "true" : "false")) {
            item_failed = true;
          }
        }
        if (!item_failed && !item_json.append("],")) {
          item_failed = true;
        }
      } else if (!item_json.append("null,")) {
        item_failed = true;
      }
      if (!item_failed && !item_json.append("\"components\":{\"valid\":%s,\"count\":%d,\"capacity\":%d,\"slots\":[",
                                            inventory.valid ? "true" : "false", inventory.count, inventory.capacity)) {
        item_failed = true;
      }

      if (!item_failed && inventory.valid) {
        for (std::int32_t slot = 0; slot < inventory.count; ++slot) {
          const NativeComponentSlotView &component = inventory.slots[slot];
          char type_json[16] = "null";
          char vtable_offset_json[32] = "null";
          char type_getter_offset_json[32] = "null";
          if (component.type_valid) {
            std::snprintf(type_json, sizeof(type_json), "%d", component.engine_type);
          }
          if (component.vtable_offset_valid) {
            std::snprintf(vtable_offset_json, sizeof(vtable_offset_json), "%llu",
                          static_cast<unsigned long long>(component.vtable_libg_offset));
          }
          if (component.type_getter_offset_valid) {
            std::snprintf(type_getter_offset_json, sizeof(type_getter_offset_json), "%llu",
                          static_cast<unsigned long long>(component.type_getter_libg_offset));
          }
          const char *owner_backlink_json =
              component.present ? (component.owner_backlink_valid ? "true" : "false") : "null";
          const char *slot_matches_type_json =
              component.type_valid ? (component.engine_type == slot ? "true" : "false") : "null";
          if (!item_json.append("%s{\"slot\":%d,\"present\":%s,\"ownerBacklinkValid\":%s,"
                                "\"vtableLibgOffset\":%s,\"typeGetterLibgOffset\":%s,"
                                "\"engineType\":%s,\"typeValid\":%s,\"slotMatchesType\":%s}",
                                slot == 0 ? "" : ",", slot, component.present ? "true" : "false", owner_backlink_json,
                                vtable_offset_json, type_getter_offset_json, type_json,
                                component.type_valid ? "true" : "false", slot_matches_type_json)) {
            item_failed = true;
            break;
          }
        }
      }
      if (item_failed || !item_json.append("]}}")) {
        truncated = true;
        break;
      }
    }
    if (used + item_used + 96 >= response_size) {
      truncated = true;
      break;
    }
    std::memcpy(response + used, item, item_used);
    used += item_used;
    response[used] = '\0';
    ++returned;
  }

  const char *encode_stage = "objects-close";
  bool encoded = json.append("],\"players\":[");
  for (std::int32_t owner = 0; encoded && owner < 2; ++owner) {
    encode_stage = owner == 0 ? "player-0-runtime" : "player-1-runtime";
    NativePlayerRuntimeView player_runtime;
    if (!read_native_player_runtime(world, object_manager, objects, count, owner, &player_runtime) ||
        !json.append("%s", owner == 0 ? "" : ",") ||
        !json.write(append_native_player_runtime_json, owner, player_runtime)) {
      encoded = false;
    }
  }
  if (encoded) {
    encode_stage = "combat-events-label";
    encoded = json.append("],\"combatEvents\":");
  }
  if (encoded) {
    encode_stage = "combat-events";
    encoded = append_combat_events_json(manager, generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "phase-runtime-label";
    encoded = json.append(",\"phaseRuntime\":");
  }
  if (encoded) {
    encode_stage = "phase-runtime";
    encoded = append_phase_runtime_events_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "special-movement-runtime-label";
    encoded = json.append(",\"specialMovementRuntime\":");
  }
  if (encoded) {
    encode_stage = "special-movement-runtime";
    encoded =
        append_special_movement_runtime_events_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "action-movement-runtime-label";
    encoded = json.append(",\"actionMovementRuntime\":");
  }
  if (encoded) {
    encode_stage = "action-movement-runtime";
    encoded = append_action_movement_runtime_events_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "character-state-runtime-label";
    encoded = json.append(",\"characterStateRuntime\":");
  }
  if (encoded) {
    encode_stage = "character-state-runtime";
    encoded = append_character_state_runtime_events_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "visibility-runtime-label";
    encoded = json.append(",\"visibilityRuntime\":");
  }
  if (encoded) {
    encode_stage = "visibility-runtime";
    encoded = append_visibility_runtime_events_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "remaining-runtime-label";
    encoded = json.append(",\"remainingRuntime\":");
  }
  if (encoded) {
    encode_stage = "remaining-runtime";
    encoded = append_remaining_runtime_events_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "tower-troop-runtime-label";
    encoded = json.append(",\"towerTroopRuntime\":");
  }
  if (encoded) {
    encode_stage = "tower-troop-runtime";
    encoded = append_tower_troop_runtime_envelope_json(generation, state_epoch, tick, response, response_size, &used);
  }
  if (encoded) {
    encode_stage = "footer";
    encoded = json.append(",\"returned\":%d,\"truncated\":%s}", returned, truncated ? "true" : "false");
  }
  if (!encoded) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"rich observation section failed to encode\","
                  "\"stage\":\"%s\",\"bytesUsed\":%llu,\"capacity\":%llu}",
                  kRichTelemetrySchema, encode_stage, static_cast<unsigned long long>(used),
                  static_cast<unsigned long long>(response_size));
  }
  if (encoded && complete_out != nullptr) {
    *complete_out = !truncated;
  }
  return encoded;
}

struct ObservationCaptureIdentity {
  RunnerMode mode = RunnerMode::Headless;
  void *manager = nullptr;
  std::uint64_t generation = 0;
  std::uint64_t state_epoch = 0;
  std::int32_t tick = -1;
};

// g_control_mutex must remain held while capturing and comparing this token.
// Native-render callers additionally wait for g_native_render_step_active to
// clear before the initial capture, so the stock controller cannot begin a new
// advancing update until both nested envelopes have been encoded.
ObservationCaptureIdentity observation_capture_identity_locked() {
  ObservationCaptureIdentity identity;
  identity.mode = static_cast<RunnerMode>(g_runner_mode.load(std::memory_order_acquire));
  if (identity.mode == RunnerMode::NativeRender) {
    identity.manager = g_native_render_manager.load(std::memory_order_acquire);
    identity.generation = g_native_render_request_sequence;
    identity.state_epoch = g_native_render_state_epoch;
  } else {
    identity.manager = g_controlled_manager;
    identity.generation = g_control_generation;
    identity.state_epoch = g_state_epoch;
  }
  identity.tick =
      identity.manager == nullptr || g_game_state_tick == nullptr ? -1 : g_game_state_tick(identity.manager);
  return identity;
}

bool observation_capture_identity_matches_locked(const ObservationCaptureIdentity &expected) {
  const ObservationCaptureIdentity current = observation_capture_identity_locked();
  return current.mode == expected.mode && current.manager == expected.manager &&
         current.generation == expected.generation && current.state_epoch == expected.state_epoch &&
         current.tick == expected.tick;
}

bool build_atomic_observation_json_locked(char *response, std::size_t response_size) {
  constexpr const char *kSchema = "native-observation-capture.v1";
  if (response == nullptr || response_size < kFullObservationResponseBytes) {
    return false;
  }
  response[0] = '\0';
  const ObservationCaptureIdentity identity = observation_capture_identity_locked();
  if (identity.manager == nullptr) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"active manager is not ready for atomic observation\"}",
                  kSchema);
    return false;
  }

  std::size_t used = 0;
  JsonWriter json(response, response_size, used);
  if (!json.append("{\"ok\":true,\"schema\":\"%s\",\"atomic\":true,"
                   "\"maxObjects\":%d,\"generation\":%llu,\"stateEpoch\":%llu,"
                   "\"tick\":%d,\"ordinary\":",
                   kSchema, kMaxObservedObjects, static_cast<unsigned long long>(identity.generation),
                   static_cast<unsigned long long>(identity.state_epoch), identity.tick)) {
    return false;
  }

  bool ordinary_complete = false;
  if (!build_observation_json(identity.manager, identity.generation, response + used, response_size - used,
                              &ordinary_complete)) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"failed to encode atomic ordinary observation\"}",
                  kSchema);
    return false;
  }
  if (!ordinary_complete) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"atomic ordinary observation exceeded the complete 256-object bound\"}",
                  kSchema);
    return false;
  }
  used += std::strlen(response + used);
  if (!observation_capture_identity_matches_locked(identity) || !json.append(",\"rich\":")) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"manager identity changed during atomic observation\"}",
                  kSchema);
    return false;
  }

  bool rich_complete = false;
  if (!build_rich_observation_json(identity.manager, identity.generation, response + used, response_size - used,
                                   &rich_complete)) {
    const char *const rich_error = response + used;
    const std::size_t rich_error_capacity = response_size - used;
    if (rich_error_capacity > 0 && std::strncmp(rich_error, "{\"ok\":false", 11) == 0) {
      const std::size_t rich_error_size = ::strnlen(rich_error, rich_error_capacity);
      if (rich_error_size < rich_error_capacity) {
        std::memmove(response, rich_error, rich_error_size + 1);
      } else {
        response[0] = '\0';
      }
    } else {
      response[0] = '\0';
    }
    if (response[0] == '\0') {
      std::snprintf(response, response_size,
                    "{\"ok\":false,\"schema\":\"%s\","
                    "\"error\":\"failed to encode atomic rich observation\"}",
                    kSchema);
    }
    return false;
  }
  if (!rich_complete) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"atomic rich observation exceeded the complete 256-object/capacity bound\"}",
                  kSchema);
    return false;
  }
  used += std::strlen(response + used);
  if (!observation_capture_identity_matches_locked(identity) || !json.append("}")) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\","
                  "\"error\":\"manager identity changed during atomic observation\"}",
                  kSchema);
    return false;
  }
  return true;
}

void send_control_response(int socket_fd, const char *response) {
  send_all(socket_fd, response, std::strlen(response));
  send_all(socket_fd, "\n", 1);
}

template <std::size_t N> void send_control_error(int socket_fd, const char (&message)[N]) {
  // Even a string made entirely of control bytes fits after JSON escaping.
  char response[6 * N + 32] = {};
  std::size_t used = 0;
  JsonWriter json(response, sizeof(response), used);
  json.begin_object();
  json.field("ok", false);
  json.field("error", message);
  json.end_object();
  send_control_response(socket_fd, response);
}



bool queue_command_and_wait(const char *command_json, char *response, std::size_t response_size) {
  const std::size_t command_size = std::strlen(command_json);
  if (command_size == 0 || command_size >= sizeof(g_pending_command_json)) {
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"injected JSON is too large\"}");
    return false;
  }

  pthread_mutex_lock(&g_control_mutex);
  if (g_controlled_manager == nullptr) {
    pthread_mutex_unlock(&g_control_mutex);
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"headless manager is not ready\"}");
    return false;
  }
  while (g_pending_command_sequence != g_processed_command_sequence) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  const std::uint64_t generation = g_control_generation;
  const std::uint64_t sequence = ++g_pending_command_sequence;
  std::memcpy(g_pending_command_json, command_json, command_size + 1);
  g_control_gate_enabled = true;
  g_last_injection_succeeded = false;
  pthread_cond_broadcast(&g_control_cond);
  while (g_controlled_manager != nullptr && g_control_generation == generation &&
         g_processed_command_sequence < sequence) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  const bool completed =
      g_controlled_manager != nullptr && g_control_generation == generation && g_processed_command_sequence >= sequence;
  const bool succeeded = completed && g_last_injection_succeeded;
  std::snprintf(response, response_size, "{\"ok\":%s,\"generation\":%llu,\"sequence\":%llu,\"tick\":%d}",
                succeeded ? "true" : "false", static_cast<unsigned long long>(g_control_generation),
                static_cast<unsigned long long>(sequence), g_last_controlled_tick);
  pthread_mutex_unlock(&g_control_mutex);
  return succeeded;
}

bool queue_native_command_and_wait(const char *command_json, char *response, std::size_t response_size) {
  const std::size_t command_size = std::strlen(command_json);
  if (command_size == 0 || command_size >= sizeof(g_pending_native_command_json)) {
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"stock-owned native command JSON is too large\"}");
    return false;
  }

  pthread_mutex_lock(&g_control_mutex);
  const RunnerMode mode = static_cast<RunnerMode>(g_runner_mode.load(std::memory_order_acquire));
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  const std::uint64_t generation = g_native_render_loaded_sequence.load(std::memory_order_acquire);
  const auto identity_is_current = [&]() {

    return mode == RunnerMode::NativeRender &&
           g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
           g_native_render_manager.load(std::memory_order_acquire) == manager &&
           g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation &&
           g_native_render_submitted_sequence.load(std::memory_order_acquire) == generation;
  };
  if (manager == nullptr || generation == 0 || !identity_is_current()) {
    pthread_mutex_unlock(&g_control_mutex);
    std::snprintf(response, response_size, "{\"ok\":false,\"error\":\"stock-owned native manager is not ready\"}");
    return false;
  }
  while (identity_is_current() && g_pending_native_command_sequence != g_processed_native_command_sequence) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  if (!identity_is_current()) {
    pthread_mutex_unlock(&g_control_mutex);
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"error\":\"stock-owned native manager changed while waiting\"}");
    return false;
  }
  const std::uint64_t sequence = ++g_pending_native_command_sequence;
  std::memcpy(g_pending_native_command_json, command_json, command_size + 1);
  g_last_native_injection_succeeded = false;
  pthread_cond_broadcast(&g_control_cond);
  while (identity_is_current() && g_processed_native_command_sequence < sequence) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  const bool completed = identity_is_current() && g_processed_native_command_sequence >= sequence;
  const bool succeeded = completed && g_last_native_injection_succeeded;
  std::snprintf(response, response_size,
                "{\"ok\":%s,\"mode\":\"%s\",\"generation\":%llu,"
                "\"sequence\":%llu,\"tick\":%d}",
                succeeded ? "true" : "false", "native-render",
                static_cast<unsigned long long>(generation), static_cast<unsigned long long>(sequence),
                g_native_render_tick.load(std::memory_order_acquire));
  pthread_mutex_unlock(&g_control_mutex);
  return succeeded;
}

struct DeployRequest {
  std::uint64_t player_id = 0;
  std::uint32_t card_id = 0;
  std::uint32_t card_parameter = 0;
  std::int32_t position_x = 0;
  std::int32_t position_y = 0;
  std::uint32_t player_id_high = 0;
  std::int32_t root_card_id = 0;
  std::int32_t deck_slot = -1;
  std::int32_t cost = -1;
  std::int32_t form_code = -1;
};

struct AbilityRequest {
  std::uint32_t player_id_high = 0;
  std::uint32_t player_id_low = 0;
  std::int32_t champion_game_id = -1;
  std::int32_t controller_slot = -1;
  std::int32_t action_data_global_id = 0;
  std::int32_t remaining_charges_raw = 0;
  std::int32_t button_state = 0;
};

bool resolve_hand_deploy_request(void *manager, std::int32_t owner, std::int32_t hand_index, std::int32_t position_x,
                                 std::int32_t position_y, DeployRequest *result) {
  if (manager == nullptr || result == nullptr) {
    return false;
  }
  void *world = read_object_field<void *>(manager, 0xa8);
  PlayerStateView player;
  if (!read_player_state(world, owner, &player) || hand_index < 0 || hand_index >= player.hand.count) {
    return false;
  }
  const auto *hand_slots = static_cast<const std::int32_t *>(player.hand.data);
  CardStateView card;
  if (!read_card_state(player, hand_slots[hand_index], true, &card) || !card.selection_valid) {
    return false;
  }
  result->player_id = player.account_id_low;
  result->card_id = static_cast<std::uint32_t>(card.command_card_id);
  result->card_parameter = card.card_parameter;
  result->position_x = position_x;
  result->position_y = position_y;
  result->player_id_high = player.account_id_high;
  result->root_card_id = card.card_id;
  result->deck_slot = card.deck_slot;
  result->cost = card.cost;
  result->form_code = static_cast<std::int32_t>(card.card_parameter & 0x0fU);
  return true;
}

bool resolve_card_deploy_request(void *manager, std::int32_t owner, std::uint32_t expected_root_card_id,
                                 std::int32_t position_x, std::int32_t position_y, DeployRequest *result) {
  if (manager == nullptr || result == nullptr || owner < 0 || owner >= kPlayerCount || expected_root_card_id == 0) {
    return false;
  }
  void *const world = read_object_field<void *>(manager, 0xa8);
  PlayerStateView player;
  if (!read_player_state(world, owner, &player) || player.hand.count < 0 ||
      (player.hand.count != 0 && player.hand.data == nullptr)) {
    return false;
  }

  const auto *const hand_slots = static_cast<const std::int32_t *>(player.hand.data);
  DeployRequest match;
  std::int32_t matches = 0;
  for (std::int32_t hand_index = 0; hand_index < player.hand.count; ++hand_index) {
    CardStateView card;
    if (!read_card_state(player, hand_slots[hand_index], true, &card) || !card.selection_valid ||
        static_cast<std::uint32_t>(card.card_id) != expected_root_card_id) {
      continue;
    }
    match.player_id = player.account_id_low;
    match.card_id = static_cast<std::uint32_t>(card.command_card_id);
    match.card_parameter = card.card_parameter;
    match.position_x = position_x;
    match.position_y = position_y;
    match.player_id_high = player.account_id_high;
    match.root_card_id = card.card_id;
    match.deck_slot = card.deck_slot;
    match.cost = card.cost;
    match.form_code = static_cast<std::int32_t>(card.card_parameter & 0x0fU);
    ++matches;
  }
  if (matches != 1) {
    return false;
  }
  *result = match;
  return true;
}

bool ascii_case_insensitive_contains(const char *value, std::size_t value_size, const char *token) {
  if (value == nullptr || token == nullptr || *token == '\0') {
    return false;
  }
  const std::size_t token_size = std::strlen(token);
  if (token_size > value_size) {
    return false;
  }
  const auto ascii_lower = [](char character) {
    return character >= 'A' && character <= 'Z' ? static_cast<char>(character + ('a' - 'A')) : character;
  };
  for (std::size_t start = 0; start + token_size <= value_size; ++start) {
    bool matches = true;
    for (std::size_t index = 0; index < token_size; ++index) {
      if (ascii_lower(value[start + index]) != ascii_lower(token[index])) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return true;
    }
  }
  return false;
}

enum class AbilityResolutionStatus : std::int32_t {
  Resolved = 0,
  Unavailable = 1,
  Ambiguous = 2,
};

bool resolve_ready_ability_request(void *manager, std::int32_t owner, std::int32_t object_index,
                                   std::int32_t secondary_index, AbilityRequest *result,
                                   const char *expected_name_hint = nullptr,
                                   AbilityResolutionStatus *resolution_status = nullptr) {
  if (resolution_status != nullptr) {
    *resolution_status = AbilityResolutionStatus::Unavailable;
  }
  if (manager == nullptr || result == nullptr || owner < 0 || owner >= kPlayerCount) {
    return false;
  }
  const bool select_unique_ready = object_index == kUniqueReadyAbilityEntityKeyTag && secondary_index == 0;
  *result = {};
  void *const world = read_object_field<void *>(manager, 0xa8);
  PlayerStateView player;
  if (!read_player_state(world, owner, &player)) {
    return false;
  }
  void *const object_manager = read_object_field<void *>(player.player, 0x10);
  NativeVectorView object_vector;
  if (!read_native_vector(object_manager, 0x08, 100000, &object_vector) ||
      (object_vector.count != 0 &&
       !process_range_is_readable(object_vector.data,
                                  static_cast<std::size_t>(object_vector.count) * sizeof(void *)))) {
    return false;
  }
  auto **const objects = static_cast<void **>(object_vector.data);

  const void *const owner_root = player.player;
  PlayerStateView other_player;
  if (!read_player_state(world, 1 - owner, &other_player) || other_player.player == owner_root ||
      read_object_field<void *>(owner_root, 0x10) != object_manager) {
    return false;
  }

  const void *champion = nullptr;
  std::uint32_t selected_native_object_id = 0;
  std::int32_t selected_controller_slot = -1;
  std::int32_t selected_action_data_global_id = 0;
  std::int32_t selected_remaining_charges_raw = 0;
  std::int32_t selected_button_state = 0;
  bool latest_candidate_ambiguous = false;
  std::int32_t matches = 0;
  for (std::int32_t controller_index = 0; controller_index < 2; ++controller_index) {
    const void *const controller = read_object_field<const void *>(
        owner_root, 0x3a0 + static_cast<std::size_t>(controller_index) * sizeof(void *));
    if (controller == nullptr) {
      continue;
    }
    NativeChampionControllerView ability;
    if (!read_champion_controller(owner_root, owner, controller_index + 1, controller, objects, object_vector.count,
                                  &ability)) {
      // A second Hero/Champion slot may be configured but have no live
      // carrier.  Prove that it is either structurally empty or exactly
      // ChampionAbsent before ignoring it; every other malformed or
      // potentially queueable controller remains a hard failure.
      if (champion_controller_is_exact_empty(owner_root, controller) ||
          champion_controller_is_exact_absent(owner_root, controller)) {
        continue;
      }
      return false;
    }
    if (select_unique_ready && expected_name_hint != nullptr &&
        !ascii_case_insensitive_contains(ability.action_data_name, ability.action_data_name_size, expected_name_hint)) {
      continue;
    }
    for (std::int32_t champion_index = 0; champion_index < ability.champion_count; ++champion_index) {
      const NativeEntityReferenceView &candidate = ability.champions[champion_index];
      const bool exact_public_identity =
          (object_index == kNativeObjectIdEntityKeyTag && secondary_index > 0 &&
           static_cast<std::uint64_t>(secondary_index) == candidate.native_object_id) ||
          (candidate.object_index == object_index && candidate.secondary_index == secondary_index);
      if (!candidate.validated || !candidate.has_value || candidate.owner != owner ||
          (!select_unique_ready && !exact_public_identity) || candidate.slot < 0 ||
          candidate.slot >= object_vector.count) {
        continue;
      }
      // The public source tuple is accepted only when it belongs to one
      // exact current controller member and that controller is queueable.
      // NotEnoughElixir is deliberately non-queueable, so semantic replay
      // selection can fall back from a newer unaffordable carrier to the
      // next newest eligible one. Native legality still rechecks every
      // transient character gate when the command is consumed.
      if (!ability_button_state_is_queueable(ability.button_state) || ability.remaining_cooldown_ms != 0 ||
          ability.remaining_charges_raw == 0) {
        if (select_unique_ready) {
          continue;
        }
        return false;
      }
      ++matches;
      const void *const candidate_champion = objects[candidate.slot];
      if (select_unique_ready) {
        // Collected replay ability events do not identify their source.
        // Native object ids increase with creation order, so prefer the
        // latest currently eligible carrier. An equal latest id cannot
        // be ordered safely and remains an explicit ambiguity failure.
        if (candidate.native_object_id > selected_native_object_id) {
          champion = candidate_champion;
          selected_native_object_id = candidate.native_object_id;
          selected_controller_slot = ability.controller_slot;
          selected_action_data_global_id = ability.action_data_global_id;
          selected_remaining_charges_raw = ability.remaining_charges_raw;
          selected_button_state = ability.button_state;
          latest_candidate_ambiguous = false;
        } else if (candidate.native_object_id == selected_native_object_id) {
          latest_candidate_ambiguous = true;
        }
        continue;
      }
      champion = candidate_champion;
      selected_controller_slot = ability.controller_slot;
      selected_action_data_global_id = ability.action_data_global_id;
      selected_remaining_charges_raw = ability.remaining_charges_raw;
      selected_button_state = ability.button_state;
    }
  }
  if (matches == 0) {
    return false;
  }
  if ((!select_unique_ready && matches > 1) || (select_unique_ready && latest_candidate_ambiguous)) {
    if (resolution_status != nullptr) {
      *resolution_status = AbilityResolutionStatus::Ambiguous;
    }
    return false;
  }
  if (champion == nullptr || !process_range_is_readable(champion, 0x0c)) {
    return false;
  }
  const std::int32_t champion_game_id = read_object_field<std::int32_t>(champion, 0x08);
  if (champion_game_id < 0) {
    return false;
  }
  result->player_id_high = player.account_id_high;
  result->player_id_low = player.account_id_low;
  result->champion_game_id = champion_game_id;
  result->controller_slot = selected_controller_slot;
  result->action_data_global_id = selected_action_data_global_id;
  result->remaining_charges_raw = selected_remaining_charges_raw;
  result->button_state = selected_button_state;
  if (resolution_status != nullptr) {
    *resolution_status = AbilityResolutionStatus::Resolved;
  }
  return true;
}





















bool format_deploy_command(const DeployRequest &request, std::int32_t command_tick, char *command_json,
                           std::size_t command_json_size) {
  if (command_json == nullptr || command_json_size == 0 ||
      command_tick > INT32_MAX - static_cast<std::int32_t>(kLiveCommandAgeTicks)) {
    return false;
  }
  const int written = std::snprintf(command_json, command_json_size,
                                    "{\"ct\":86,\"c\":{\"t\":%d,\"t2\":%d,\"idHi\":%u,\"idLo\":%llu,"
                                    "\"px\":%d,\"py\":%d,\"sid\":-1,\"sel\":{\"os\":%u,\"pd\":%u}}}",
                                    command_tick, command_tick + static_cast<std::int32_t>(kLiveCommandAgeTicks),
                                    request.player_id_high, static_cast<unsigned long long>(request.player_id),
                                    request.position_x, request.position_y, request.card_id, request.card_parameter);
  return written > 0 && static_cast<std::size_t>(written) < command_json_size;
}

bool format_ability_command(const AbilityRequest &request, std::int32_t command_tick, char *command_json,
                            std::size_t command_json_size) {
  if (command_json == nullptr || command_json_size == 0 || command_tick < 0 || command_tick == INT32_MAX ||
      request.champion_game_id < 0) {
    return false;
  }
  const int written = std::snprintf(command_json, command_json_size,
                                    "{\"ct\":2,\"c\":{\"t\":%d,\"t2\":%d,\"idHi\":%u,"
                                    "\"idLo\":%u,\"cgid\":%d}}",
                                    command_tick, command_tick + 1, request.player_id_high, request.player_id_low,
                                    request.champion_game_id);
  return written > 0 && static_cast<std::size_t>(written) < command_json_size;
}

const char *scheduled_replay_action_kind_name(ScheduledReplayActionKind kind) {
  return kind == ScheduledReplayActionKind::Ability ? "ability" : "card";
}

const char *scheduled_replay_action_state_name(ScheduledReplayActionState state) {
  switch (state) {
  case ScheduledReplayActionState::Pending:
    return "pending";
  case ScheduledReplayActionState::Succeeded:
    return "succeeded";
  case ScheduledReplayActionState::Failed:
    return "failed";
  }
  return "failed";
}

const char *scheduled_replay_action_error_name(ScheduledReplayActionError error) {
  switch (error) {
  case ScheduledReplayActionError::None:
    return "none";
  case ScheduledReplayActionError::MissedBoundary:
    return "missed-boundary";
  case ScheduledReplayActionError::CardUnavailable:
    return "card-unavailable";
  case ScheduledReplayActionError::AbilityUnavailable:
    return "ability-unavailable";
  case ScheduledReplayActionError::CommandEncoding:
    return "command-encoding";
  case ScheduledReplayActionError::InjectionFailed:
    return "injection-failed";
  case ScheduledReplayActionError::GenerationChanged:
    return "generation-changed";
  case ScheduledReplayActionError::AbilityAmbiguous:
    return "ability-ambiguous";
  }
  return "unknown";
}

void clear_scheduled_replay_actions_locked() {
  std::memset(g_scheduled_replay_actions, 0, sizeof(g_scheduled_replay_actions));
  g_scheduled_replay_action_count = 0;
}

void clear_scheduled_replay_actions_for_manager_locked(void *manager) {
  if (manager == nullptr) {
    return;
  }
  std::size_t retained = 0;
  for (std::size_t index = 0; index < g_scheduled_replay_action_count; ++index) {
    if (g_scheduled_replay_actions[index].manager == manager) {
      continue;
    }
    if (retained != index) {
      g_scheduled_replay_actions[retained] = g_scheduled_replay_actions[index];
    }
    ++retained;
  }
  for (std::size_t index = retained; index < g_scheduled_replay_action_count; ++index) {
    g_scheduled_replay_actions[index] = {};
  }
  g_scheduled_replay_action_count = retained;
}

AbilityResolutionStatus resolve_scheduled_replay_ability(void *manager, const ScheduledReplayAction &scheduled,
                                                         AbilityRequest *result) {
  if (std::strcmp(scheduled.ability_hints, "-") == 0) {
    AbilityResolutionStatus status = AbilityResolutionStatus::Unavailable;
    resolve_ready_ability_request(manager, scheduled.owner, kUniqueReadyAbilityEntityKeyTag, 0, result, nullptr,
                                  &status);
    return status;
  }

  char hints[kMaxScheduledAbilityHints + 1] = {};
  std::memcpy(hints, scheduled.ability_hints, sizeof(hints));
  hints[sizeof(hints) - 1] = '\0';
  char *save_pointer = nullptr;
  AbilityResolutionStatus fallback = AbilityResolutionStatus::Unavailable;
  for (char *hint = ::strtok_r(hints, ",", &save_pointer); hint != nullptr;
       hint = ::strtok_r(nullptr, ",", &save_pointer)) {
    AbilityResolutionStatus status = AbilityResolutionStatus::Unavailable;
    resolve_ready_ability_request(manager, scheduled.owner, kUniqueReadyAbilityEntityKeyTag, 0, result, hint, &status);
    if (status == AbilityResolutionStatus::Resolved) {
      return status;
    }
    if (status == AbilityResolutionStatus::Ambiguous) {
      fallback = status;
    }
  }
  return fallback;
}

void process_scheduled_replay_actions(void *manager) {
  if (manager == nullptr || g_game_state_tick == nullptr) {
    return;
  }
  pthread_mutex_lock(&g_control_mutex);
  const ObservationCaptureIdentity identity = observation_capture_identity_locked();
  const std::int32_t current_tick = identity.tick;
  const bool current = (identity.mode == RunnerMode::Headless || identity.mode == RunnerMode::NativeRender) &&
                       identity.manager == manager && identity.generation != 0 && identity.state_epoch != 0 &&
                       current_tick >= 0;
  if (!current) {
    pthread_mutex_unlock(&g_control_mutex);
    return;
  }

  bool changed = false;
  for (std::size_t index = 0; index < g_scheduled_replay_action_count; ++index) {
    ScheduledReplayAction &scheduled = g_scheduled_replay_actions[index];
    if (scheduled.state != ScheduledReplayActionState::Pending) {
      continue;
    }
    // Resident slots share one process-global queue.  A step on one live
    // manager must leave every other slot's pending actions untouched.
    if (scheduled.manager != identity.manager) {
      continue;
    }
    if (scheduled.mode != identity.mode || scheduled.generation != identity.generation ||
        scheduled.state_epoch != identity.state_epoch) {
      scheduled.state = ScheduledReplayActionState::Failed;
      scheduled.error = ScheduledReplayActionError::GenerationChanged;
      scheduled.processed_tick = current_tick;
      changed = true;
      continue;
    }
    const std::int32_t boundary_tick = scheduled.target_execute_tick - 1;
    if (current_tick < boundary_tick) {
      continue;
    }
    scheduled.processed_tick = current_tick;
    if (current_tick > boundary_tick) {
      scheduled.state = ScheduledReplayActionState::Failed;
      scheduled.error = ScheduledReplayActionError::MissedBoundary;
      changed = true;
      continue;
    }

    char command_json[768] = {};
    bool resolved = false;
    AbilityResolutionStatus ability_resolution = AbilityResolutionStatus::Unavailable;
    bool formatted = false;
    if (scheduled.kind == ScheduledReplayActionKind::Card) {
      DeployRequest request;
      resolved = resolve_card_deploy_request(manager, scheduled.owner, scheduled.card_id, scheduled.position_x,
                                             scheduled.position_y, &request);
      if (resolved) {
        scheduled.resolved_command_card_id = request.card_id;
        scheduled.resolved_card_parameter = request.card_parameter;
        formatted = format_deploy_command(request, boundary_tick - static_cast<std::int32_t>(kLiveCommandAgeTicks),
                                          command_json, sizeof(command_json));
      }
    } else {
      AbilityRequest request;
      ability_resolution = resolve_scheduled_replay_ability(manager, scheduled, &request);
      resolved = ability_resolution == AbilityResolutionStatus::Resolved;
      if (resolved) {
        formatted =
            format_ability_command(request, scheduled.target_execute_tick - 2, command_json, sizeof(command_json));
      }
    }

    if (!resolved) {
      scheduled.state = ScheduledReplayActionState::Failed;
      scheduled.error = scheduled.kind == ScheduledReplayActionKind::Ability
                            ? ability_resolution == AbilityResolutionStatus::Ambiguous
                                  ? ScheduledReplayActionError::AbilityAmbiguous
                                  : ScheduledReplayActionError::AbilityUnavailable
                            : ScheduledReplayActionError::CardUnavailable;
    } else if (!formatted) {
      scheduled.state = ScheduledReplayActionState::Failed;
      scheduled.error = ScheduledReplayActionError::CommandEncoding;
    } else if (!inject_logic_command(manager, command_json)) {
      scheduled.state = ScheduledReplayActionState::Failed;
      scheduled.error = ScheduledReplayActionError::InjectionFailed;
    } else {
      scheduled.state = ScheduledReplayActionState::Succeeded;
      scheduled.error = ScheduledReplayActionError::None;
    }
    changed = true;
  }
  if (changed) {
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
}

struct StepResult {
  bool completed = false;
  bool ended = false;
  std::uint64_t generation = 0;
  std::uint64_t start_steps = 0;
  std::uint64_t completed_steps = 0;
  std::int32_t tick = -1;
};

StepResult advance_controlled_manager(std::uint64_t requested_steps) {
  StepResult result;
  pthread_mutex_lock(&g_control_mutex);
  if (g_controlled_manager == nullptr || requested_steps == 0) {
    pthread_mutex_unlock(&g_control_mutex);
    return result;
  }

  result.generation = g_control_generation;
  result.start_steps = g_completed_steps;
  const std::uint64_t target = g_completed_steps + requested_steps;
  g_control_gate_enabled = true;
  if (!g_control_ended) {
    g_step_budget += requested_steps;
    pthread_cond_broadcast(&g_control_cond);
  }
  while (g_controlled_manager != nullptr && g_control_generation == result.generation && g_completed_steps < target &&
         !g_control_ended) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  result.completed = g_controlled_manager != nullptr && g_control_generation == result.generation &&
                     (g_completed_steps >= target || g_control_ended);
  result.ended = g_control_ended;
  result.completed_steps = g_completed_steps;
  result.tick = g_last_controlled_tick;
  pthread_mutex_unlock(&g_control_mutex);
  return result;
}

void execute_transition(int socket_fd, unsigned int delay, unsigned int requested_advance, const DeployRequest *actions,
                        unsigned int action_count) {
  char response[kControlResponseBytes] = {};
  const bool native_render_mode =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
  pthread_mutex_lock(&g_control_mutex);
  void *const manager =
      native_render_mode ? g_native_render_manager.load(std::memory_order_acquire) : g_controlled_manager;
  const bool ready = manager != nullptr;
  const std::int32_t current_tick =
      native_render_mode ? g_native_render_tick.load(std::memory_order_acquire) : g_last_controlled_tick;
  pthread_mutex_unlock(&g_control_mutex);
  if (!ready || current_tick < 0 || delay > static_cast<unsigned int>(INT32_MAX - current_tick)) {
    send_control_error(socket_fd, "active native manager is not ready for transition");
    return;
  }

  const std::int32_t command_tick = current_tick + static_cast<std::int32_t>(delay);
  for (unsigned int index = 0; index < action_count; ++index) {
    char command_json[768] = {};
    const bool formatted = format_deploy_command(actions[index], command_tick, command_json, sizeof(command_json));
    const bool queued =
        formatted && (native_render_mode ? queue_native_command_and_wait(command_json, response, sizeof(response))
                                         : queue_command_and_wait(command_json, response, sizeof(response)));
    if (!queued) {
      if (response[0] == '\0') {
        std::snprintf(response, sizeof(response), "{\"ok\":false,\"error\":\"failed to queue transition action\"}");
      }
      send_control_response(socket_fd, response);
      return;
    }
  }

  if (native_render_mode) {
    pthread_mutex_lock(&g_control_mutex);
    const bool observed = g_native_render_manager.load(std::memory_order_acquire) == manager &&
                          build_observation_json(manager, g_native_render_request_sequence, response, sizeof(response));
    pthread_mutex_unlock(&g_control_mutex);
    if (!observed && response[0] == '\0') {
      std::snprintf(response, sizeof(response),
                    "{\"ok\":false,\"error\":\"failed to encode native-render observation\"}");
    }
    send_control_response(socket_fd, response);
    return;
  }

  std::uint64_t advance = requested_advance;
  if (advance == 0) {
    advance = action_count == 0 ? 1 : static_cast<std::uint64_t>(delay) + kLiveCommandAgeTicks + 1;
  }
  if (advance == 0 || advance > 1000000ULL) {
    send_control_error(socket_fd, "transition advance must resolve to 1..1000000 ticks");
    return;
  }
  const StepResult step_result = advance_controlled_manager(advance);
  if (!step_result.completed) {
    send_control_error(socket_fd, "native transition did not complete");
    return;
  }

  pthread_mutex_lock(&g_control_mutex);
  const bool observed = build_observation_json(g_controlled_manager, g_control_generation, response, sizeof(response));
  pthread_mutex_unlock(&g_control_mutex);
  if (!observed && response[0] == '\0') {
    std::snprintf(response, sizeof(response), "{\"ok\":false,\"error\":\"failed to encode transition observation\"}");
  }
  send_control_response(socket_fd, response);
}

struct SnapshotOperationReceipt {
  bool completed = false;
  bool succeeded = false;
  SnapshotError error = SnapshotError::ManagerChanged;
  StoredSnapshot result;
};

SnapshotOperationReceipt request_snapshot_operation(SnapshotOperation operation, std::uint64_t handle) {
  SnapshotOperationReceipt receipt;
  pthread_mutex_lock(&g_control_mutex);
  RunnerMode mode = static_cast<RunnerMode>(g_runner_mode.load(std::memory_order_acquire));
  SnapshotManagerIdentity identity;
  if (!current_snapshot_identity_locked(mode, &identity)) {
    pthread_mutex_unlock(&g_control_mutex);
    return receipt;
  }
  while (g_snapshot_request_sequence != g_snapshot_processed_sequence && snapshot_identity_matches_locked(identity)) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  if (g_snapshot_request_sequence != g_snapshot_processed_sequence ||
      !current_snapshot_identity_locked(mode, &identity)) {
    pthread_mutex_unlock(&g_control_mutex);
    return receipt;
  }

  const bool restore_native = operation == SnapshotOperation::Restore && mode == RunnerMode::NativeRender;
  const bool prior_native_pause = restore_native ? g_native_render_paused : false;
  if (restore_native) {
    g_native_render_paused = true;
    while (g_native_render_step_active && snapshot_identity_matches_locked(identity)) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    if (!snapshot_identity_matches_locked(identity)) {
      pthread_mutex_unlock(&g_control_mutex);
      return receipt;
    }
  }

  const std::uint64_t sequence = ++g_snapshot_request_sequence;
  g_pending_snapshot_operation = operation;
  g_pending_snapshot_mode = mode;
  g_pending_snapshot_manager = identity.manager;
  g_pending_snapshot_generation = identity.generation;
  g_pending_snapshot_handle = handle;
  g_last_snapshot_operation_succeeded = false;
  g_last_snapshot_operation_error = SnapshotError::None;
  g_last_snapshot_operation_result = {};
  pthread_cond_broadcast(&g_control_cond);
  while (g_snapshot_processed_sequence < sequence && snapshot_identity_matches_locked(identity)) {
    pthread_cond_wait(&g_control_cond, &g_control_mutex);
  }
  if (g_snapshot_processed_sequence < sequence && g_snapshot_processing_sequence != sequence) {
    complete_snapshot_operation_locked(sequence, false, SnapshotError::ManagerChanged);
  }
  receipt.completed = g_snapshot_processed_sequence >= sequence;
  if (receipt.completed) {
    receipt.succeeded = g_last_snapshot_operation_succeeded;
    receipt.error = g_last_snapshot_operation_error;
    receipt.result = g_last_snapshot_operation_result;
  }
  if (restore_native && snapshot_identity_matches_locked(identity)) {
    g_native_render_paused = prior_native_pause;
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
  return receipt;
}

bool parse_positive_uint64(const char *text, std::uint64_t *result) {
  if (text == nullptr || result == nullptr || text[0] == '\0') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || end == nullptr || end[0] != '\0' || parsed == 0) {
    return false;
  }
  *result = static_cast<std::uint64_t>(parsed);
  return true;
}

bool parse_snapshot_handle(const char *text, std::uint64_t *handle) { return parse_positive_uint64(text, handle); }

bool is_replay_schedule_command(const char *command) {
  return command != nullptr && (std::strcmp(command, "replay-schedule-clear") == 0 ||
                                std::strncmp(command, "replay-schedule-status ", 23) == 0 ||
                                std::strncmp(command, "replay-schedule-card ", 21) == 0 ||
                                std::strncmp(command, "replay-schedule-ability ", 24) == 0);
}

bool handle_replay_schedule_command(int socket_fd, const char *command) {
  if (!is_replay_schedule_command(command)) {
    return false;
  }
  char response[kControlResponseBytes] = {};
  if (std::strcmp(command, "replay-schedule-clear") == 0) {
    pthread_mutex_lock(&g_control_mutex);
    const ObservationCaptureIdentity identity = observation_capture_identity_locked();
    clear_scheduled_replay_actions_for_manager_locked(identity.manager);
    const std::int32_t tick = identity.tick;
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);
    std::snprintf(response, sizeof(response), "{\"ok\":true,\"tick\":%d}", tick);
    send_control_response(socket_fd, response);
    return true;
  }

  if (std::strncmp(command, "replay-schedule-status ", 23) == 0) {
    unsigned long long requested_sequence = 0;
    int consumed = 0;
    const int fields = std::sscanf(command + 23, "%llu%n", &requested_sequence, &consumed);
    const char *trailing = fields == 1 ? command + 23 + consumed : "x";
    while (*trailing == ' ' || *trailing == '\t') {
      ++trailing;
    }
    if (fields != 1 || requested_sequence == 0 || *trailing != '\0') {
      send_control_error(socket_fd, "replay-schedule-status expects SEQUENCE");
      return true;
    }

    pthread_mutex_lock(&g_control_mutex);
    const ObservationCaptureIdentity identity = observation_capture_identity_locked();
    const ScheduledReplayAction *found = nullptr;
    for (std::size_t index = 0; index < g_scheduled_replay_action_count; ++index) {
      const ScheduledReplayAction &candidate = g_scheduled_replay_actions[index];
      if (candidate.sequence == static_cast<std::uint64_t>(requested_sequence) &&
          candidate.manager == identity.manager) {
        found = &candidate;
        break;
      }
    }
    if (found == nullptr) {
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "scheduled replay action is unknown for this manager");
      return true;
    }
    std::snprintf(response, sizeof(response),
                  "{\"ok\":true,\"sequence\":%llu,\"mode\":\"%s\","
                  "\"generation\":%llu,\"stateEpoch\":%llu,"
                  "\"kind\":\"%s\",\"state\":\"%s\",\"error\":\"%s\","
                  "\"owner\":%d,\"cardId\":%u,\"registeredAtTick\":%d,"
                  "\"queuedAtTick\":%d,\"executeTick\":%d,"
                  "\"resolvedCommandCardId\":%u,\"resolvedCardParameter\":%u}",
                  static_cast<unsigned long long>(found->sequence),
                  found->mode == RunnerMode::NativeRender ? "native-render" : "headless",
                  static_cast<unsigned long long>(found->generation),
                  static_cast<unsigned long long>(found->state_epoch), scheduled_replay_action_kind_name(found->kind),
                  scheduled_replay_action_state_name(found->state), scheduled_replay_action_error_name(found->error),
                  found->owner, found->card_id, found->registered_tick, found->processed_tick,
                  found->target_execute_tick, found->resolved_command_card_id, found->resolved_card_parameter);
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return true;
  }

  const bool schedule_card = std::strncmp(command, "replay-schedule-card ", 21) == 0;
  std::int32_t owner = -1;
  unsigned int card_id = 0;
  std::int32_t position_x = 0;
  std::int32_t position_y = 0;
  std::int32_t target_execute_tick = -1;
  char ability_hints[kMaxScheduledAbilityHints + 1] = {};
  int consumed = 0;
  const int fields = schedule_card ? std::sscanf(command + 21, "%d %u %d %d %d%n", &owner, &card_id, &position_x,
                                                 &position_y, &target_execute_tick, &consumed)
                                   : std::sscanf(command + 24, "%d %260s %d%n", &owner, ability_hints,
                                                 &target_execute_tick, &consumed);
  const int expected_fields = schedule_card ? 5 : 3;
  const char *trailing = fields == expected_fields ? command + (schedule_card ? 21 : 24) + consumed : "x";
  while (*trailing == ' ' || *trailing == '\t') {
    ++trailing;
  }
  bool hints_valid = schedule_card || std::strcmp(ability_hints, "-") == 0;
  if (!schedule_card && !hints_valid) {
    hints_valid = ability_hints[0] != '\0';
    bool previous_comma = true;
    for (std::size_t index = 0; hints_valid && ability_hints[index] != '\0'; ++index) {
      const char character = ability_hints[index];
      const bool alphanumeric = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
                                (character >= '0' && character <= '9');
      if (character == ',') {
        hints_valid = !previous_comma && ability_hints[index + 1] != '\0';
        previous_comma = true;
      } else {
        hints_valid = alphanumeric;
        previous_comma = false;
      }
    }
  }
  const bool parsed = fields == expected_fields && consumed > 0 && *trailing == '\0' && owner >= 0 &&
                      owner < kPlayerCount && target_execute_tick >= 1 && hints_valid &&
                      (!schedule_card || (card_id != 0 && position_x >= 0 && position_x < 18000 && position_y >= 0 &&
                                          position_y < 32000));
  if (!parsed) {
    send_control_response(
        socket_fd, schedule_card
                       ? "{\"ok\":false,\"error\":\"replay-schedule-card expects OWNER CARD_ID X Y EXECUTE_TICK\"}"
                       : "{\"ok\":false,\"error\":\"replay-schedule-ability expects OWNER NAME_HINTS EXECUTE_TICK\"}");
    return true;
  }

  pthread_mutex_lock(&g_control_mutex);
  const ObservationCaptureIdentity identity = observation_capture_identity_locked();
  const bool native_render_ready =
      identity.mode == RunnerMode::NativeRender && identity.manager != nullptr && identity.generation != 0 &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) == identity.generation &&
      g_native_render_submitted_sequence.load(std::memory_order_acquire) == identity.generation;
  const bool headless_ready = identity.mode == RunnerMode::Headless && identity.manager != nullptr &&
                              identity.generation != 0 && (g_native_worker_running || g_resident_mode_active);
  const std::int32_t current_tick = identity.tick;
  if ((!native_render_ready && !headless_ready) || current_tick < 0) {
    pthread_mutex_unlock(&g_control_mutex);
    send_control_error(socket_fd, "active replay manager is not ready for scheduling");
    return true;
  }
  if (target_execute_tick <= current_tick) {
    pthread_mutex_unlock(&g_control_mutex);
    send_control_error(socket_fd, "scheduled replay action execute tick is not in the future");
    return true;
  }
  if (g_scheduled_replay_action_count >= kMaxScheduledReplayActions) {
    pthread_mutex_unlock(&g_control_mutex);
    send_control_error(socket_fd, "scheduled replay action capacity exceeded");
    return true;
  }

  ScheduledReplayAction &scheduled = g_scheduled_replay_actions[g_scheduled_replay_action_count++];
  scheduled = {};
  scheduled.sequence = ++g_next_scheduled_replay_action_sequence;
  scheduled.mode = identity.mode;
  scheduled.manager = identity.manager;
  scheduled.generation = identity.generation;
  scheduled.state_epoch = identity.state_epoch;
  scheduled.kind = schedule_card ? ScheduledReplayActionKind::Card : ScheduledReplayActionKind::Ability;
  scheduled.registered_tick = current_tick;
  scheduled.target_execute_tick = target_execute_tick;
  scheduled.owner = owner;
  scheduled.card_id = static_cast<std::uint32_t>(card_id);
  scheduled.position_x = position_x;
  scheduled.position_y = position_y;
  if (!schedule_card) {
    std::memcpy(scheduled.ability_hints, ability_hints, sizeof(scheduled.ability_hints));
    scheduled.ability_hints[sizeof(scheduled.ability_hints) - 1] = '\0';
  }
  std::snprintf(
      response, sizeof(response),
      "{\"ok\":true,\"sequence\":%llu,\"mode\":\"%s\","
      "\"generation\":%llu,\"stateEpoch\":%llu,"
      "\"kind\":\"%s\",\"registeredAtTick\":%d,"
      "\"executeTick\":%d}",
      static_cast<unsigned long long>(scheduled.sequence),
      scheduled.mode == RunnerMode::NativeRender ? "native-render" : "headless",
      static_cast<unsigned long long>(scheduled.generation), static_cast<unsigned long long>(scheduled.state_epoch),
      scheduled_replay_action_kind_name(scheduled.kind), scheduled.registered_tick, scheduled.target_execute_tick);
  pthread_mutex_unlock(&g_control_mutex);
  send_control_response(socket_fd, response);
  return true;
}

#include "resident_multimatch.inc"
#include "engine_instance.inc"

void handle_control_command(int socket_fd, const char *command) {
  char response[kControlResponseBytes] = {};
  if (handle_engine_instance_command(socket_fd, command)) {
    return;
  }
  std::int32_t resident_slot_id = -1;
  const char *resident_command = nullptr;
  if (parse_resident_env_prefix(command, &resident_slot_id, &resident_command)) {
    handle_resident_env_command(socket_fd, resident_slot_id, resident_command);
    return;
  }
  if (std::strcmp(command, "multi-status") == 0) {
    if (!g_resident_mode_active) {
      send_control_response(socket_fd, "{\"ok\":true,\"mode\":\"legacy-headless\",\"capacity\":16,"
                                       "\"occupied\":0,\"active\":false,\"slots\":[]}");
    } else {
      resident_save_bound_slot();
      resident_send_status(socket_fd);
    }
    return;
  }
  if (std::strcmp(command, "multi-stop") == 0) {
    if (g_resident_mode_active) {
      for (std::size_t index = 0; index < kResidentMaxSlots; ++index) {
        resident_destroy_slot(static_cast<std::int32_t>(index));
      }
      g_resident_mode_active = false;
      g_resident_mode_requested = false;
      g_resident_bound_slot = -1;
      // Resident slots use per-slot configuration revisions.  Do not
      // leak those values back into the legacy singleton mailbox.
      g_config_request_sequence = 0;
      g_config_processed_sequence = 0;
      g_active_replay_json[0] = '\0';
      g_pending_replay_json[0] = '\0';
    }
    send_control_response(socket_fd, "{\"ok\":true,\"mode\":\"legacy-headless\",\"active\":false}");
    return;
  }
  if (g_resident_mode_active &&
      (std::strncmp(command, "configure ", 10) == 0 || std::strcmp(command, "reset") == 0 ||
       std::strncmp(command, "step ", 5) == 0 || std::strncmp(command, "play ", 5) == 0 ||
       std::strncmp(command, "transition ", 11) == 0 || std::strncmp(command, "inject ", 7) == 0 ||
       std::strcmp(command, "observe") == 0 || std::strcmp(command, "observe-rich") == 0 ||
       std::strcmp(command, "observe-atomic") == 0 || is_replay_schedule_command(command))) {
    send_control_error(socket_fd, "resident mode requires env ID command prefixes");
    return;
  }








  if (std::strncmp(command, "configure-native ", 17) == 0) {
    const char *replay_json = command + 17;
    const std::size_t replay_size = std::strlen(replay_json);
    std::int32_t location_id = 0;
    if (replay_size < 2 || replay_size >= sizeof(g_pending_native_render_json) || replay_json[0] != '{' ||
        !extract_location_id(replay_json, &location_id) || !validate_empty_match_streams(replay_json)) {
      send_control_error(socket_fd,
                         "configure-native expects one compact fresh-match JSON object with empty cmd/evt arrays");
      return;
    }
    if (!inspect_content_runtime()) {
      send_control_error(socket_fd, "native content runtime is not initialized yet");
      return;
    }

    // The stock replay manager owns the native battle scene. Queue the JSON
    // here; the permanent frame shim submits it before the next swap.
    set_render_suppressed(false);
    pthread_mutex_lock(&g_control_mutex);
    const std::uint64_t sequence = ++g_native_render_request_sequence;
    std::memcpy(g_pending_native_render_json, replay_json, replay_size + 1);
    g_last_native_render_request_succeeded = false;
    g_native_render_speed_quarters = 4;
    g_native_render_paused = false;
    g_native_render_advance_target_tick = -1;
    g_native_render_advance_reached = false;
    g_native_render_advance_ended = false;
    g_offline_control_session_active.store(true, std::memory_order_release);
    g_runner_mode.store(static_cast<std::int32_t>(RunnerMode::NativeRender), std::memory_order_release);
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);
    std::snprintf(response, sizeof(response),
                  "{\"ok\":true,\"accepted\":true,\"mode\":\"native-render\","
                  "\"sequence\":%llu,\"location\":%d}",
                  static_cast<unsigned long long>(sequence), location_id);
    send_control_response(socket_fd, response);
    return;
  }
  if (std::strncmp(command, "configure ", 10) == 0) {
    const char *replay_json = command + 10;
    const std::size_t replay_size = std::strlen(replay_json);
    if (replay_size < 2 || replay_size >= sizeof(g_pending_replay_json) || replay_json[0] != '{') {
      send_control_error(socket_fd, "configure expects one compact replay JSON object under 64 KiB");
      return;
    }

    g_offline_control_session_active.store(true, std::memory_order_release);
    g_runner_mode.store(static_cast<std::int32_t>(RunnerMode::Headless), std::memory_order_release);
    pthread_mutex_lock(&g_control_mutex);
    while (g_native_worker_running && g_config_request_sequence != g_config_processed_sequence) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    const std::uint64_t previous_generation = g_control_generation;
    const std::uint64_t sequence = ++g_config_request_sequence;
    std::memcpy(g_pending_replay_json, replay_json, replay_size + 1);
    g_last_config_succeeded = false;
    if (!inspect_content_runtime()) {
      g_config_processed_sequence = sequence;
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "native content runtime is not initialized yet");
      return;
    } else if (g_native_worker_running) {
      ++g_reset_request_sequence;
      g_control_gate_enabled = true;
      g_step_budget = 0;
    } else if (!start_native_controlled_worker_locked()) {
      g_config_processed_sequence = sequence;
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "could not start native match factory worker");
      return;
    }
    pthread_cond_broadcast(&g_control_cond);
    while (g_native_worker_running &&
           (g_config_processed_sequence < sequence || g_control_generation == previous_generation)) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    const bool completed = g_native_worker_running && g_config_processed_sequence >= sequence &&
                           g_last_config_succeeded && g_control_generation > previous_generation &&
                           g_controlled_manager != nullptr;
    std::snprintf(response, sizeof(response),
                  "{\"ok\":%s,\"configured\":%s,\"configRevision\":%llu,"
                  "\"generation\":%llu,\"tick\":%d%s}",
                  completed ? "true" : "false", g_active_replay_json[0] != '\0' ? "true" : "false",
                  static_cast<unsigned long long>(g_config_processed_sequence),
                  static_cast<unsigned long long>(g_control_generation), g_last_controlled_tick,
                  completed ? "" : ",\"error\":\"native replay configuration was rejected\"");
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strcmp(command, "render off") == 0 || std::strcmp(command, "render on") == 0) {
    const bool suppress = std::strcmp(command, "render off") == 0;
    const bool succeeded = set_render_suppressed(suppress);
    std::snprintf(response, sizeof(response), "{\"ok\":%s,\"renderSuppressed\":%s%s}", succeeded ? "true" : "false",
                  g_render_suppressed ? "true" : "false",
                  succeeded ? "" : ",\"error\":\"could not update native EGL render gate\"");
    send_control_response(socket_fd, response);
    return;
  }
  if (std::strcmp(command, "render status") == 0) {
    pthread_mutex_lock(&g_render_gate_mutex);
    const bool suppressed = g_render_suppressed;
    pthread_mutex_unlock(&g_render_gate_mutex);
    std::snprintf(response, sizeof(response), "{\"ok\":true,\"renderSuppressed\":%s}", suppressed ? "true" : "false");
    send_control_response(socket_fd, response);
    return;
  }
  if (std::strcmp(command, "touch status") == 0) {
    format_probe_native_touch_status(response, sizeof(response));
    send_control_response(socket_fd, response);
    return;
  }
  if (std::strcmp(command, "attest") == 0) {
    if (!format_runner_attestation(response, sizeof(response))) {
      send_control_error(socket_fd, "could not collect runner attestation");
      return;
    }
    send_control_response(socket_fd, response);
    return;
  }
  if (std::strcmp(command, "status") == 0) {
    void *content_root = nullptr;
    void *content_context = nullptr;
    const bool cold_ready = inspect_content_runtime(&content_root, &content_context);
    const bool content_published = content_root != nullptr && content_context != nullptr;
    const std::uint32_t tilemap_ensure_calls = g_tilemap_ensure_calls.load(std::memory_order_acquire);
    void *const last_location_data = g_last_location_data.load(std::memory_order_acquire);
    void *const last_tilemap = g_last_tilemap.load(std::memory_order_acquire);
    const bool native_render_mode =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
    void *const native_replay_manager = g_native_render_replay_manager.load(std::memory_order_acquire);
    void *const native_controller = g_native_render_controller.load(std::memory_order_acquire);
    void *const native_manager = g_native_render_manager.load(std::memory_order_acquire);
    const std::int32_t native_tick = g_native_render_tick.load(std::memory_order_acquire);
    const std::uint64_t native_loaded_sequence = g_native_render_loaded_sequence.load(std::memory_order_acquire);
    const std::uint64_t native_logical_steps = g_native_forced_steps.load(std::memory_order_acquire);
    const std::uint64_t native_stock_step_callbacks = g_native_stock_step_callbacks.load(std::memory_order_acquire);
    const bool offline_control_session = g_offline_control_session_active.load(std::memory_order_acquire);
    const std::uint64_t offline_connection_errors_suppressed =
        g_native_offline_connection_errors_suppressed.load(std::memory_order_acquire);
    pthread_mutex_lock(&g_render_gate_mutex);
    const bool render_suppressed = g_render_suppressed;
    pthread_mutex_unlock(&g_render_gate_mutex);
    pthread_mutex_lock(&g_control_mutex);
    const bool status_paused = native_render_mode ? g_native_render_paused
                               : g_control_gate_enabled;
    const char *const status_speed =
        native_render_mode ? native_render_speed_json(g_native_render_speed_quarters) : "null";
    const bool native_render_ready = native_render_mode
                                         ? native_manager != nullptr &&
                                               g_native_render_processed_sequence == g_native_render_request_sequence &&
                                               native_loaded_sequence == g_native_render_request_sequence
                                         : false;
    const bool ready = native_render_mode ? native_render_ready
                                            : g_controlled_manager != nullptr;
    void *const status_manager = native_render_mode ? native_manager
                                 : g_controlled_manager;
    BattleResultView status_battle_result;
    const bool terminal_readable =
        ready && status_manager != nullptr &&
        read_battle_result(read_object_field<void *>(status_manager, 0xa8), &status_battle_result);
    const bool status_ended = (terminal_readable && status_battle_result.finalized) ||
                              (!native_render_mode && g_control_ended);
    char status_world_result[32] = "null";
    char status_world_result_raw[32] = "null";
    char status_winner[32] = "null";
    char status_crowns[64] = "[null,null]";
    if (terminal_readable) {
      std::snprintf(status_world_result_raw, sizeof(status_world_result_raw), "%d",
                    status_battle_result.world_result_raw);
      std::snprintf(status_crowns, sizeof(status_crowns), "[%d,%d]", status_battle_result.crowns_raw[0],
                    status_battle_result.crowns_raw[1]);
      if (status_battle_result.finalized) {
        std::snprintf(status_world_result, sizeof(status_world_result), "%d", status_battle_result.world_result_raw);
        if (status_battle_result.world_result_raw == 0 || status_battle_result.world_result_raw == 1) {
          std::snprintf(status_winner, sizeof(status_winner), "%d", status_battle_result.world_result_raw);
        }
      }
    }
    std::snprintf(response, sizeof(response),
                  "{\"ok\":true,\"coldReady\":%s,\"contentReady\":%s,\"contentPublished\":%s,"
                  "\"contentRoot\":\"%p\",\"contentContext\":\"%p\","
                  "\"mode\":\"%s\",\"nativeRenderReady\":%s,"
                  "\"nativeRenderSequence\":%llu,\"nativeRenderProcessed\":%llu,\"nativeRenderLoaded\":%llu,"
                  "\"nativeReplayManager\":\"%p\",\"nativeController\":\"%p\","
                  "\"nativeManager\":\"%p\",\"nativeTick\":%d,"
                  "\"ready\":%s,\"matchReady\":%s,"
                  ""
                  "\"bootstrap\":\"cold-config\",\"replayBootstrap\":false,"
                  "\"armed\":%s,\"gate\":%s,\"paused\":%s,\"speed\":%s,\"ended\":%s,"
                  "\"terminalReadable\":%s,\"finalized\":%s,"
                  "\"worldResult\":%s,\"worldResultRaw\":%s,\"winner\":%s,"
                  "\"crownsRaw\":%s,\"stateEpoch\":%llu,\"snapshotHandles\":%zu,"
                  "\"renderSuppressed\":%s,\"offlineControlSession\":%s,"
                  "\"offlineConnectionErrorsSuppressed\":%llu,"
                  "\"configured\":%s,\"configRevision\":%llu,"
                  "\"generation\":%llu,\"resets\":%llu,\"steps\":%llu,"
                  "\"nativeStockStepCallbacks\":%llu,"
                  "\"tick\":%d,"
                  "\"manager\":\"%p\",\"fixtureBattle\":\"%p\","
                  "\"fixtureCommands\":\"%p\",\"configStage\":%d,"
                  "\"configRoot\":\"%p\",\"configCommands\":\"%p\",\"configEvents\":\"%p\","
                  "\"configLocationId\":%d,\"configCommandCount\":%d,\"configEventCount\":%d,\"loadedQueueCount\":%d,"
                  "\"tilemapEnsureCalls\":%u,\"locationData\":\"%p\",\"tilemap\":\"%p\","
                  "\"configCommandWords\":[\"%016llx\",\"%016llx\",\"%016llx\","
                  "\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"]}",
                  cold_ready ? "true" : "false", cold_ready ? "true" : "false", content_published ? "true" : "false",
                  content_root, content_context,
                  native_render_mode ? "native-render"
                  : "headless",
                  native_render_ready ? "true" : "false",
                  static_cast<unsigned long long>(g_native_render_request_sequence),
                  static_cast<unsigned long long>(g_native_render_processed_sequence),
                  static_cast<unsigned long long>(native_loaded_sequence), native_replay_manager, native_controller,
                  native_manager, native_tick, ready ? "true" : "false", ready ? "true" : "false",
                  "true",
                  status_paused ? "true" : "false", status_paused ? "true" : "false", status_speed,
                  status_ended ? "true" : "false", terminal_readable ? "true" : "false",
                  terminal_readable && status_battle_result.finalized ? "true" : "false", status_world_result,
                  status_world_result_raw, status_winner, status_crowns,
                  static_cast<unsigned long long>(native_render_mode ? g_native_render_state_epoch
                                                  : g_state_epoch),
                  count_stored_snapshots_locked(), render_suppressed ? "true" : "false",
                  offline_control_session ? "true" : "false",
                  static_cast<unsigned long long>(offline_connection_errors_suppressed),
                  (native_render_mode ? ready && g_last_native_render_request_succeeded
                   : g_active_replay_json[0] != '\0')
                      ? "true"
                      : "false",
                  static_cast<unsigned long long>(native_render_mode ? g_native_render_processed_sequence
                                                  : g_config_processed_sequence),
                  static_cast<unsigned long long>(native_render_mode ? g_native_render_request_sequence
                                                  : g_control_generation),
                  static_cast<unsigned long long>(g_reset_processed_sequence),
                  static_cast<unsigned long long>(native_render_mode ? native_logical_steps
                                                  : g_completed_steps),
                  static_cast<unsigned long long>(native_stock_step_callbacks),
                  native_render_mode ? native_tick
                  : g_last_controlled_tick,
                  native_render_mode ? native_manager
                  : g_controlled_manager,
                  g_fixture_battle, g_fixture_command_array, g_last_config_stage, g_last_config_root,
                  g_last_config_commands, g_last_config_events, g_last_config_location_id, g_last_config_command_count,
                  g_last_config_event_count, g_last_loaded_queue_count, tilemap_ensure_calls, last_location_data,
                  last_tilemap, static_cast<unsigned long long>(g_last_config_command_words[0]),
                  static_cast<unsigned long long>(g_last_config_command_words[1]),
                  static_cast<unsigned long long>(g_last_config_command_words[2]),
                  static_cast<unsigned long long>(g_last_config_command_words[3]),
                  static_cast<unsigned long long>(g_last_config_command_words[4]),
                  static_cast<unsigned long long>(g_last_config_command_words[5]),
                  static_cast<unsigned long long>(g_last_config_command_words[6]),
                  static_cast<unsigned long long>(g_last_config_command_words[7]));
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strcmp(command, "observe") == 0) {
    std::unique_ptr<char[]> observation_response(new (std::nothrow) char[kOrdinaryObservationResponseBytes]());
    if (observation_response == nullptr) {
      send_control_error(socket_fd, "could not allocate ordinary observation response");
      return;
    }
    pthread_mutex_lock(&g_control_mutex);
    void *const observed_manager =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender)
            ? g_native_render_manager.load(std::memory_order_acquire)
            : g_controlled_manager;
    const bool succeeded = build_observation_json(observed_manager,
                                                  g_runner_mode.load(std::memory_order_relaxed) ==
                                                          static_cast<std::int32_t>(RunnerMode::NativeRender)
                                                      ? g_native_render_request_sequence
                                                      : g_control_generation,
                                                  observation_response.get(), kOrdinaryObservationResponseBytes);
    pthread_mutex_unlock(&g_control_mutex);
    if (!succeeded && observation_response[0] == '\0') {
      std::snprintf(observation_response.get(), kOrdinaryObservationResponseBytes,
                    "{\"ok\":false,\"error\":\"failed to encode observation\"}");
    }
    send_control_response(socket_fd, observation_response.get());
    return;
  }

  if (std::strcmp(command, "observe-rich") == 0) {
    std::unique_ptr<char[]> rich_response(new (std::nothrow) char[kFullObservationResponseBytes]());
    if (rich_response == nullptr) {
      send_control_response(socket_fd, "{\"ok\":false,\"schema\":\"native-rich-telemetry.v3\","
                                       "\"error\":\"could not allocate rich observation response\"}");
      return;
    }
    pthread_mutex_lock(&g_control_mutex);
    void *const observed_manager =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender)
            ? g_native_render_manager.load(std::memory_order_acquire)
            : g_controlled_manager;
    const bool succeeded = build_rich_observation_json(observed_manager,
                                                       g_runner_mode.load(std::memory_order_relaxed) ==
                                                               static_cast<std::int32_t>(RunnerMode::NativeRender)
                                                           ? g_native_render_request_sequence
                                                           : g_control_generation,
                                                       rich_response.get(), kFullObservationResponseBytes);
    pthread_mutex_unlock(&g_control_mutex);
    if (!succeeded && rich_response[0] == '\0') {
      std::snprintf(rich_response.get(), kFullObservationResponseBytes,
                    "{\"ok\":false,\"schema\":\"native-rich-telemetry.v3\","
                    "\"error\":\"failed to encode rich observation\"}");
    }
    send_control_response(socket_fd, rich_response.get());
    return;
  }

  if (std::strcmp(command, "observe-atomic") == 0) {
    static std::unique_ptr<char[]> atomic_response(new (std::nothrow) char[kFullObservationResponseBytes]());
    if (atomic_response == nullptr) {
      send_control_error(socket_fd, "atomic capture workspace unavailable");
      return;
    }
    atomic_response[0] = '\0';
    pthread_mutex_lock(&g_control_mutex);
    while (g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
           g_native_render_step_active) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    const bool succeeded = build_atomic_observation_json_locked(atomic_response.get(), kFullObservationResponseBytes);
    pthread_mutex_unlock(&g_control_mutex);
    if (!succeeded && atomic_response[0] == '\0') {
      std::snprintf(atomic_response.get(), kFullObservationResponseBytes,
                    "{\"ok\":false,\"schema\":\"native-observation-capture.v1\","
                    "\"error\":\"failed to encode atomic observation\"}");
    }
    send_control_response(socket_fd, atomic_response.get());
    return;
  }

  if (std::strcmp(command, "snapshot-create") == 0 || std::strcmp(command, "create-snapshot") == 0) {
    const SnapshotOperationReceipt receipt = request_snapshot_operation(SnapshotOperation::Create, 0);
    if (!receipt.completed || !receipt.succeeded) {
      const SnapshotError error = receipt.completed ? receipt.error : SnapshotError::ManagerChanged;
      std::snprintf(response, sizeof(response), "{\"ok\":false,\"error\":\"%s\"}", snapshot_error_message(error));
    } else {
      std::snprintf(response, sizeof(response),
                    "{\"ok\":true,\"handle\":%llu,\"scope\":\"process\","
                    "\"generation\":%llu,\"configRevision\":%llu,\"stateEpoch\":%llu,"
                    "\"tick\":%d,\"steps\":%llu,\"digest\":\"%016llx\",\"bytes\":%u}",
                    static_cast<unsigned long long>(receipt.result.handle),
                    static_cast<unsigned long long>(receipt.result.generation),
                    static_cast<unsigned long long>(receipt.result.config_revision),
                    static_cast<unsigned long long>(receipt.result.state_epoch), receipt.result.tick,
                    static_cast<unsigned long long>(receipt.result.completed_steps),
                    static_cast<unsigned long long>(receipt.result.digest), receipt.result.serialized_bytes);
    }
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strncmp(command, "restore ", 8) == 0) {
    std::uint64_t handle = 0;
    if (!parse_snapshot_handle(command + 8, &handle)) {
      send_control_error(socket_fd, "restore expects one positive snapshot handle");
      return;
    }
    const SnapshotOperationReceipt receipt = request_snapshot_operation(SnapshotOperation::Restore, handle);
    if (!receipt.completed || !receipt.succeeded) {
      const SnapshotError error = receipt.completed ? receipt.error : SnapshotError::ManagerChanged;
      std::snprintf(response, sizeof(response), "{\"ok\":false,\"handle\":%llu,\"error\":\"%s\"}",
                    static_cast<unsigned long long>(handle), snapshot_error_message(error));
    } else {
      std::snprintf(response, sizeof(response),
                    "{\"ok\":true,\"restored\":true,\"handle\":%llu,"
                    "\"generation\":%llu,\"configRevision\":%llu,\"stateEpoch\":%llu,"
                    "\"tick\":%d,\"steps\":%llu,\"digest\":\"%016llx\",\"bytes\":%u}",
                    static_cast<unsigned long long>(receipt.result.handle),
                    static_cast<unsigned long long>(receipt.result.generation),
                    static_cast<unsigned long long>(receipt.result.config_revision),
                    static_cast<unsigned long long>(receipt.result.state_epoch), receipt.result.tick,
                    static_cast<unsigned long long>(receipt.result.completed_steps),
                    static_cast<unsigned long long>(receipt.result.digest), receipt.result.serialized_bytes);
    }
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strncmp(command, "release-snapshot ", 17) == 0) {
    std::uint64_t handle = 0;
    if (!parse_snapshot_handle(command + 17, &handle)) {
      send_control_error(socket_fd, "release-snapshot expects one positive snapshot handle");
      return;
    }
    const SnapshotOperationReceipt receipt = request_snapshot_operation(SnapshotOperation::Release, handle);
    if (!receipt.completed || !receipt.succeeded) {
      const SnapshotError error = receipt.completed ? receipt.error : SnapshotError::ManagerChanged;
      std::snprintf(response, sizeof(response), "{\"ok\":false,\"handle\":%llu,\"error\":\"%s\"}",
                    static_cast<unsigned long long>(handle), snapshot_error_message(error));
    } else {
      std::snprintf(response, sizeof(response),
                    "{\"ok\":true,\"released\":true,\"handle\":%llu,"
                    "\"generation\":%llu}",
                    static_cast<unsigned long long>(receipt.result.handle),
                    static_cast<unsigned long long>(receipt.result.generation));
    }
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strcmp(command, "reset") == 0) {
    if (g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender)) {
      pthread_mutex_lock(&g_control_mutex);
      if (g_pending_native_render_json[0] == '\0') {
        pthread_mutex_unlock(&g_control_mutex);
        send_control_error(socket_fd, "native-render configuration is not available");
        return;
      }
      const std::uint64_t sequence = ++g_native_render_request_sequence;
      g_last_native_render_request_succeeded = false;
      g_native_render_speed_quarters = 4;
      g_native_render_paused = false;
      g_native_render_advance_target_tick = -1;
      g_native_render_advance_reached = false;
      g_native_render_advance_ended = false;
      pthread_cond_broadcast(&g_control_cond);
      while (g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
             g_native_render_loaded_sequence.load(std::memory_order_acquire) < sequence) {
        pthread_cond_wait(&g_control_cond, &g_control_mutex);
      }
      const bool completed = g_native_render_loaded_sequence.load(std::memory_order_acquire) >= sequence &&
                             g_native_render_manager.load(std::memory_order_acquire) != nullptr;
      std::snprintf(response, sizeof(response),
                    "{\"ok\":%s,\"mode\":\"native-render\",\"generation\":%llu,\"tick\":%d}",
                    completed ? "true" : "false", static_cast<unsigned long long>(sequence),
                    g_native_render_tick.load(std::memory_order_acquire));
      pthread_mutex_unlock(&g_control_mutex);
      send_control_response(socket_fd, response);
      return;
    }
    pthread_mutex_lock(&g_control_mutex);
    if (g_controlled_manager == nullptr || !g_native_worker_running) {
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "native worker is not ready");
      return;
    }
    const std::uint64_t previous_generation = g_control_generation;
    const std::uint64_t sequence = ++g_reset_request_sequence;
    g_control_gate_enabled = true;
    g_step_budget = 0;
    pthread_cond_broadcast(&g_control_cond);
    while (g_native_worker_running && g_reset_processed_sequence < sequence) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    const bool completed = g_native_worker_running && g_reset_processed_sequence >= sequence &&
                           g_control_generation > previous_generation && g_controlled_manager != nullptr;
    std::snprintf(response, sizeof(response),
                  "{\"ok\":%s,\"generation\":%llu,\"resets\":%llu,\"steps\":%llu,\"tick\":%d}",
                  completed ? "true" : "false", static_cast<unsigned long long>(g_control_generation),
                  static_cast<unsigned long long>(g_reset_processed_sequence),
                  static_cast<unsigned long long>(g_completed_steps), g_last_controlled_tick);
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strncmp(command, "play ", 5) == 0) {
    const char *cursor = command + 5;
    unsigned int delay = 0;
    unsigned int requested_advance = 0;
    unsigned int action_count = 0;
    int consumed = 0;
    if (std::sscanf(cursor, "%u %u %u%n", &delay, &requested_advance, &action_count, &consumed) != 3 || delay == 0 ||
        action_count > kMaxTransitionActions) {
      send_control_error(socket_fd, "play expects DELAY ADVANCE ACTION_COUNT and at most 8 actions");
      return;
    }
    cursor += consumed;

    struct HandActionRequest {
      std::int32_t owner = -1;
      std::int32_t hand_index = -1;
      std::int32_t position_x = 0;
      std::int32_t position_y = 0;
    } hand_actions[kMaxTransitionActions];
    for (unsigned int index = 0; index < action_count; ++index) {
      consumed = 0;
      if (std::sscanf(cursor, " %d %d %d %d%n", &hand_actions[index].owner, &hand_actions[index].hand_index,
                      &hand_actions[index].position_x, &hand_actions[index].position_y, &consumed) != 4 ||
          consumed <= 0 || hand_actions[index].owner < 0 || hand_actions[index].owner >= kPlayerCount) {
        send_control_error(socket_fd, "invalid play action tuple");
        return;
      }
      cursor += consumed;
    }
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor != '\0') {
      send_control_error(socket_fd, "unexpected trailing play fields");
      return;
    }

    DeployRequest actions[kMaxTransitionActions] = {};
    pthread_mutex_lock(&g_control_mutex);
    void *manager = g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender)
                        ? g_native_render_manager.load(std::memory_order_acquire)
                        : g_controlled_manager;
    bool resolved = manager != nullptr;
    unsigned int failed_index = 0;
    for (unsigned int index = 0; resolved && index < action_count; ++index) {
      if (!resolve_hand_deploy_request(manager, hand_actions[index].owner, hand_actions[index].hand_index,
                                       hand_actions[index].position_x, hand_actions[index].position_y,
                                       &actions[index])) {
        resolved = false;
        failed_index = index;
      }
    }
    pthread_mutex_unlock(&g_control_mutex);
    if (!resolved) {
      std::snprintf(response, sizeof(response),
                    "{\"ok\":false,\"error\":\"could not resolve native hand action\","
                    "\"actionIndex\":%u}",
                    failed_index);
      send_control_response(socket_fd, response);
      return;
    }
    execute_transition(socket_fd, delay, requested_advance, actions, action_count);
    return;
  }

  if (std::strncmp(command, "transition ", 11) == 0) {
    const char *cursor = command + 11;
    unsigned int delay = 0;
    unsigned int requested_advance = 0;
    unsigned int action_count = 0;
    int consumed = 0;
    if (std::sscanf(cursor, "%u %u %u%n", &delay, &requested_advance, &action_count, &consumed) != 3 || delay == 0 ||
        action_count > kMaxTransitionActions) {
      send_control_error(socket_fd, "transition expects DELAY ADVANCE ACTION_COUNT and at most 8 actions");
      return;
    }
    cursor += consumed;

    DeployRequest actions[kMaxTransitionActions] = {};
    for (unsigned int index = 0; index < action_count; ++index) {
      unsigned long long player_id = 0;
      unsigned int card_id = 0;
      unsigned int card_parameter = 0;
      int position_x = 0;
      int position_y = 0;
      consumed = 0;
      if (std::sscanf(cursor, " %llu %u %u %d %d%n", &player_id, &card_id, &card_parameter, &position_x, &position_y,
                      &consumed) != 5 ||
          consumed <= 0) {
        send_control_error(socket_fd, "invalid transition action tuple");
        return;
      }
      actions[index] = {
          static_cast<std::uint64_t>(player_id),      static_cast<std::uint32_t>(card_id),
          static_cast<std::uint32_t>(card_parameter), static_cast<std::int32_t>(position_x),
          static_cast<std::int32_t>(position_y),
      };
      cursor += consumed;
    }
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor != '\0') {
      send_control_error(socket_fd, "unexpected trailing transition fields");
      return;
    }
    execute_transition(socket_fd, delay, requested_advance, actions, action_count);
    return;
  }

  unsigned long long requested_steps = 0;
  if (std::sscanf(command, "step %llu", &requested_steps) == 1) {
    if (requested_steps == 0 || requested_steps > 1000000ULL) {
      send_control_error(socket_fd, "step count must be 1..1000000");
      return;
    }
    const StepResult step_result = advance_controlled_manager(requested_steps);
    if (step_result.generation == 0) {
      send_control_error(socket_fd, "headless manager is not ready");
      return;
    }
    std::snprintf(response, sizeof(response),
                  "{\"ok\":%s,\"generation\":%llu,\"steps\":%llu,\"advanced\":%llu,"
                  "\"tick\":%d,\"ended\":%s}",
                  step_result.completed ? "true" : "false", static_cast<unsigned long long>(step_result.generation),
                  static_cast<unsigned long long>(step_result.completed_steps),
                  static_cast<unsigned long long>(step_result.completed_steps - step_result.start_steps),
                  step_result.tick, step_result.ended ? "true" : "false");
    send_control_response(socket_fd, response);
    return;
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
    if ((fields != 3 && fields != 4) || consumed <= 0 || delay < 1 || delay > 65535 || owner < 0 ||
        owner >= kPlayerCount || object_index < -2 || secondary_index < 0) {
      send_control_error(socket_fd, "activate-ability expects OWNER OBJECT_INDEX SECONDARY_INDEX [DELAY]");
      return;
    }
    const char *trailing = command + 17 + consumed;
    while (*trailing == ' ' || *trailing == '\t') {
      ++trailing;
    }
    if (*trailing != '\0') {
      send_control_error(socket_fd, "unexpected trailing activate-ability fields");
      return;
    }

    const RunnerMode ability_mode = static_cast<RunnerMode>(g_runner_mode.load(std::memory_order_acquire));
    const bool stock_owned_mode = ability_mode == RunnerMode::NativeRender;
    AbilityRequest request;
    std::int32_t command_tick = -1;
    pthread_mutex_lock(&g_control_mutex);
    void *const manager = ability_mode == RunnerMode::NativeRender
                              ? g_native_render_manager.load(std::memory_order_acquire)
                              : g_controlled_manager;
    command_tick = ability_mode == RunnerMode::NativeRender ? g_native_render_tick.load(std::memory_order_acquire)
                                                            : g_last_controlled_tick;
    AbilityResolutionStatus ability_resolution = AbilityResolutionStatus::Unavailable;
    if (command_tick >= 0) {
      resolve_ready_ability_request(manager, owner, object_index, secondary_index, &request, nullptr,
                                    &ability_resolution);
    }
    const bool resolved = ability_resolution == AbilityResolutionStatus::Resolved;
    pthread_mutex_unlock(&g_control_mutex);
    if (!resolved) {
      send_control_response(
          socket_fd, ability_resolution == AbilityResolutionStatus::Ambiguous
                         ? "{\"ok\":false,\"error\":\"native champion ability source is ambiguous\"}"
                         : "{\"ok\":false,\"error\":\"could not resolve one ready native champion ability source\"}");
      return;
    }

    char command_json[384] = {};
    // Ability command ``t2`` is a command-age boundary just like a card
    // command's ``t2``: the first state containing its effects is the
    // following tick. ``format_ability_command`` writes t2=t+1, so an
    // observable execute tick E must be encoded with t=E-2.
    if (!format_ability_command(request, command_tick + delay - 2, command_json, sizeof(command_json))) {
      send_control_error(socket_fd, "ability command tick is out of range");
      return;
    }
    if (stock_owned_mode) {
      queue_native_command_and_wait(command_json, response, sizeof(response));
    } else {
      queue_command_and_wait(command_json, response, sizeof(response));
    }
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strncmp(command, "inject ", 7) == 0) {
    const RunnerMode inject_mode = static_cast<RunnerMode>(g_runner_mode.load(std::memory_order_acquire));
    if (inject_mode == RunnerMode::NativeRender) {
      queue_native_command_and_wait(command + 7, response, sizeof(response));
    } else {
      queue_command_and_wait(command + 7, response, sizeof(response));
    }
    send_control_response(socket_fd, response);
    return;
  }

  unsigned long long player_id = 0;
  unsigned int card_id = 0;
  unsigned int card_parameter = 0;
  int position_x = 0;
  int position_y = 0;
  int delay = 2;
  const int deploy_fields = std::sscanf(command, "deploy %llu %u %u %d %d %d", &player_id, &card_id, &card_parameter,
                                        &position_x, &position_y, &delay);
  if (deploy_fields == 5 || deploy_fields == 6) {
    const bool native_render_mode =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
    pthread_mutex_lock(&g_control_mutex);
    const std::int32_t current_tick =
        native_render_mode ? g_native_render_tick.load(std::memory_order_acquire) : g_last_controlled_tick;
    pthread_mutex_unlock(&g_control_mutex);
    if (delay < 1) {
      delay = 1;
    }
    const std::int32_t command_tick = current_tick + delay;
    const DeployRequest request = {
        static_cast<std::uint64_t>(player_id),      static_cast<std::uint32_t>(card_id),
        static_cast<std::uint32_t>(card_parameter), static_cast<std::int32_t>(position_x),
        static_cast<std::int32_t>(position_y),
    };
    char command_json[768] = {};
    if (!format_deploy_command(request, command_tick, command_json, sizeof(command_json))) {
      send_control_error(socket_fd, "deploy command tick is out of range");
      return;
    }
    if (native_render_mode) {
      queue_native_command_and_wait(command_json, response, sizeof(response));
    } else {
      queue_command_and_wait(command_json, response, sizeof(response));
    }
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strncmp(command, "advance-native ", 15) == 0) {
    unsigned long long requested_steps = 0;
    char trailing = '\0';
    if (std::sscanf(command + 15, "%llu%c", &requested_steps, &trailing) != 1 || requested_steps == 0 ||
        requested_steps > 1000000ULL) {
      send_control_error(socket_fd, "advance-native expects 1..1000000 ticks");
      return;
    }
    pthread_mutex_lock(&g_control_mutex);
    const bool native_render_mode =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
    void *const manager = g_native_render_manager.load(std::memory_order_acquire);
    const std::uint64_t generation = g_native_render_loaded_sequence.load(std::memory_order_acquire);
    const std::int32_t current_tick = g_native_render_tick.load(std::memory_order_acquire);
    if (!native_render_mode || manager == nullptr || generation == 0 || current_tick < 0) {
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "native renderer is not ready for synchronized advance");
      return;
    }
    if (!g_native_render_paused || g_native_render_step_active || g_native_render_advance_target_tick >= 0) {
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "native renderer must be idle and paused before synchronized advance");
      return;
    }
    if (requested_steps > static_cast<unsigned long long>(INT32_MAX - current_tick)) {
      pthread_mutex_unlock(&g_control_mutex);
      send_control_error(socket_fd, "advance-native target tick is out of range");
      return;
    }
    const std::int32_t target_tick = current_tick + static_cast<std::int32_t>(requested_steps);
    g_native_render_advance_target_tick = target_tick;
    g_native_render_advance_reached = false;
    g_native_render_advance_ended = false;
    g_native_render_paused = false;
    pthread_cond_broadcast(&g_control_cond);
    while (g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
           g_native_render_manager.load(std::memory_order_acquire) == manager &&
           g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation &&
           g_native_render_advance_target_tick == target_tick &&
           (!g_native_render_advance_reached || g_native_render_step_active)) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    const std::int32_t final_tick = g_native_render_tick.load(std::memory_order_acquire);
    const bool ended = g_native_render_advance_ended;
    const bool completed = g_native_render_manager.load(std::memory_order_acquire) == manager &&
                           g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation &&
                           g_native_render_advance_reached && g_native_render_paused &&
                           ((ended && final_tick <= target_tick) || final_tick == target_tick);
    g_native_render_advance_target_tick = -1;
    g_native_render_advance_reached = false;
    g_native_render_advance_ended = false;
    std::snprintf(response, sizeof(response),
                  "{\"ok\":%s,\"mode\":\"native-render\",\"requestedTicks\":%llu,"
                  "\"targetTick\":%d,\"tick\":%d,\"ended\":%s,\"paused\":%s}",
                  completed ? "true" : "false", requested_steps, target_tick, final_tick, ended ? "true" : "false",
                  g_native_render_paused ? "true" : "false");
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }
  if (handle_replay_schedule_command(socket_fd, command)) {
    return;
  }

  if (std::strncmp(command, "speed ", 6) == 0) {
    std::uint32_t speed_quarters = 0;
    if (!parse_native_render_speed(command + 6, &speed_quarters)) {
      send_control_error(socket_fd, "speed must be one of 0.25, 0.5, 1, 2, 4");
      return;
    }
    if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::NativeRender)) {
      send_control_error(socket_fd, "speed is only available in native-render mode");
      return;
    }
    pthread_mutex_lock(&g_control_mutex);
    const bool paused = g_native_render_paused;
    // Briefly close the native timing gate so an in-flight stock controller
    // update cannot straddle two different multipliers.
    g_native_render_paused = true;
    while (g_native_render_step_active) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    g_native_render_speed_quarters = speed_quarters;
    g_native_render_paused = paused;
    const std::int32_t tick = g_native_render_tick.load(std::memory_order_acquire);
    std::snprintf(response, sizeof(response),
                  "{\"ok\":true,\"mode\":\"native-render\",\"speed\":%s,"
                  "\"paused\":%s,\"tick\":%d}",
                  native_render_speed_json(speed_quarters), paused ? "true" : "false", tick);
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strcmp(command, "pause") == 0) {
    pthread_mutex_lock(&g_control_mutex);
    const bool native_render_mode =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
    if (native_render_mode) {
      g_native_render_paused = true;
      while (g_native_render_step_active) {
        pthread_cond_wait(&g_control_cond, &g_control_mutex);
      }
      g_native_render_advance_target_tick = -1;
      g_native_render_advance_reached = false;
      g_native_render_advance_ended = false;
      pthread_cond_broadcast(&g_control_cond);
      std::snprintf(response, sizeof(response),
                    "{\"ok\":true,\"mode\":\"native-render\",\"gate\":true,"
                    "\"paused\":true,\"speed\":%s,\"tick\":%d}",
                    native_render_speed_json(g_native_render_speed_quarters),
                    g_native_render_tick.load(std::memory_order_acquire));
    } else {
      g_control_gate_enabled = true;
      g_step_budget = 0;
      std::snprintf(response, sizeof(response), "{\"ok\":true,\"gate\":true}");
    }
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }

  if (std::strcmp(command, "resume") == 0) {
    pthread_mutex_lock(&g_control_mutex);
    const bool native_render_mode =
        g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
    if (native_render_mode) {
      g_native_render_paused = false;
      std::snprintf(response, sizeof(response),
                    "{\"ok\":true,\"mode\":\"native-render\",\"gate\":false,"
                    "\"paused\":false,\"speed\":%s,\"tick\":%d}",
                    native_render_speed_json(g_native_render_speed_quarters),
                    g_native_render_tick.load(std::memory_order_acquire));
    } else {
      g_control_gate_enabled = false;
      pthread_cond_broadcast(&g_control_cond);
      std::snprintf(response, sizeof(response), "{\"ok\":true,\"gate\":false}");
    }
    pthread_mutex_unlock(&g_control_mutex);
    send_control_response(socket_fd, response);
    return;
  }

  send_control_error(
      socket_fd,
      "commands: configure REPLAY_JSON, configure-native REPLAY_JSON, "
      "status, attest, touch status, "
      "render on|off|status, observe, observe-rich, observe-atomic, session-v1, reset, step N, advance-native N, play DELAY ADVANCE "
      "COUNT [OWNER HAND_INDEX X Y]..., transition DELAY ADVANCE COUNT [PLAYER CARD PD X Y]..., deploy PLAYER CARD PD "
      "X Y [DELAY], activate-ability OWNER OBJECT_INDEX SECONDARY_INDEX [DELAY], replay-schedule-card OWNER CARD_ID X "
      "Y EXECUTE_TICK, replay-schedule-ability OWNER NAME_HINTS EXECUTE_TICK, replay-schedule-status SEQUENCE, "
      "replay-schedule-clear, inject JSON, speed 0.25|0.5|1|2|4, pause, resume");
}

bool receive_live_control_command(int socket_fd, char *command, std::size_t capacity) {
  if (command == nullptr || capacity < 2) {
    return false;
  }
  std::size_t used = 0;
  command[0] = '\0';
  while (used + 1 < capacity) {
    const ssize_t received = recv(socket_fd, command + used, capacity - used - 1, 0);
    if (received <= 0) {
      return false;
    }
    used += static_cast<std::size_t>(received);
    command[used] = '\0';
    char *const newline = std::strpbrk(command, "\r\n");
    if (newline != nullptr) {
      *newline = '\0';
      return true;
    }
  }
  return false;
}

void handle_persistent_control_session(int socket_fd) {
  constexpr std::size_t kLiveSessionCommandBytes = 4096;
  send_control_response(socket_fd, "{\"ok\":true,\"session\":\"control-session.v1\"}");
  for (;;) {
    char command[kLiveSessionCommandBytes] = {};
    if (!receive_live_control_command(socket_fd, command, sizeof(command))) {
      return;
    }
    if (std::strcmp(command, "session-close") == 0) {
      send_control_response(socket_fd, "{\"ok\":true,\"sessionClosed\":true}");
      return;
    }
    handle_control_command(socket_fd, command);
  }
}

void run_control_server(std::uint16_t port) {
  if (!engine_instance_pin_current_lane()) {
    return;
  }
  const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    LOGE("control socket() failed: %d", errno);
    return;
  }
  int reuse = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port);
  if (bind(server_fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0) {
    LOGE("control bind(127.0.0.1:%u) failed: %d", port, errno);
    close(server_fd);
    return;
  }
  if (listen(server_fd, 4) != 0) {
    LOGE("control listen() failed: %d", errno);
    close(server_fd);
    return;
  }
  LOGI("control server listening on 127.0.0.1:%u engine=%d pid=%d", port, g_engine_instance_id,
       static_cast<int>(getpid()));

  for (;;) {
    const int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      LOGE("control accept() failed: %d", errno);
      break;
    }
    const int no_delay = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
    char command[kMaxControlRequestBytes] = {};
    std::size_t used = 0;
    while (used + 1 < sizeof(command)) {
      const ssize_t received = recv(client_fd, command + used, sizeof(command) - used - 1, 0);
      if (received <= 0) {
        break;
      }
      used += static_cast<std::size_t>(received);
      command[used] = '\0';
      char *newline = std::strpbrk(command, "\r\n");
      if (newline != nullptr) {
        *newline = '\0';
        break;
      }
    }
    if (std::strcmp(command, "session-v1") == 0) {
      handle_persistent_control_session(client_fd);
    } else if (std::strcmp(command, "batch-v1") == 0) {
      handle_resident_batch_session(client_fd, false, false);
    } else if (std::strcmp(command, "batch-zlib-v1") == 0) {
      handle_resident_batch_session(client_fd, false, true);
    } else if (std::strcmp(command, "batch-fast-v1") == 0) {
      handle_resident_batch_session(client_fd, true, false);
    } else if (std::strcmp(command, "batch-fast-zlib-v1") == 0) {
      handle_resident_batch_session(client_fd, true, true);
    } else {
      handle_control_command(client_fd, command);
    }
    shutdown(client_fd, SHUT_RDWR);
    close(client_fd);
  }
  close(server_fd);
}

void *control_server_main(void *raw_port) {
  const std::uintptr_t port_value = reinterpret_cast<std::uintptr_t>(raw_port);
  run_control_server(static_cast<std::uint16_t>(port_value));
  return nullptr;
}

void start_control_server(std::uint16_t port = kControlPort) {
  pthread_t thread = {};
  const int error = pthread_create(&thread, nullptr, &control_server_main,
                                   reinterpret_cast<void *>(static_cast<std::uintptr_t>(port)));
  if (error != 0) {
    LOGE("pthread_create(control server) failed: %d", error);
    return;
  }
  pthread_detach(thread);
}

void replay_json_setter_hook(void *self, void *replay_json) {
  LOGI("replay setter called: self=%p payload=%p", self, replay_json);
  g_original_replay_json_setter(self, replay_json);
}

bool process_pending_native_command(void *manager) {
  if (manager == nullptr) {
    return false;
  }

  char command_json[kMaxPendingCommandBytes] = {};
  std::uint64_t command_sequence = 0;
  std::uint64_t generation = 0;
  pthread_mutex_lock(&g_control_mutex);
  generation = g_native_render_loaded_sequence.load(std::memory_order_acquire);
  const bool current_native_render =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      g_native_render_manager.load(std::memory_order_acquire) == manager && generation != 0 &&
      g_native_render_submitted_sequence.load(std::memory_order_acquire) == generation;
  if (current_native_render && g_pending_native_command_sequence != g_processed_native_command_sequence) {
    command_sequence = g_pending_native_command_sequence;
    std::memcpy(command_json, g_pending_native_command_json, sizeof(command_json));
    command_json[sizeof(command_json) - 1] = '\0';
  }
  pthread_mutex_unlock(&g_control_mutex);

  if (command_sequence == 0 || command_json[0] == '\0') {
    return false;
  }

  const bool succeeded = inject_logic_command(manager, command_json);
  pthread_mutex_lock(&g_control_mutex);
  const bool still_current =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      g_native_render_manager.load(std::memory_order_acquire) == manager &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation &&
      g_native_render_submitted_sequence.load(std::memory_order_acquire) == generation;
  if (still_current && g_processed_native_command_sequence < command_sequence) {
    g_last_native_injection_succeeded = succeeded;
    g_processed_native_command_sequence = command_sequence;
    g_pending_native_command_json[0] = '\0';
    pthread_cond_broadcast(&g_control_cond);
  }
  pthread_mutex_unlock(&g_control_mutex);
  return succeeded;
}





void game_app_pending_event_setter_hook(void *game_app, std::int32_t event, std::int32_t argument, void *text_a,
                                        void *text_b, void *text_c) {
  GameAppPendingEventSetter original = g_original_game_app_pending_event_setter;
  if (original == nullptr) {
    return;
  }
  if (g_offline_control_session_active.load(std::memory_order_acquire) && (event == 2 || event == 3)) {
    const std::uint64_t count =
        g_native_offline_connection_errors_suppressed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 3) {
      LOGI("blocked offline connection failure enqueue #%llu gameApp=%p event=%d argument=%d",
           static_cast<unsigned long long>(count), game_app, event, argument);
    }
    return;
  }
  original(game_app, event, argument, text_a, text_b, text_c);
}

bool is_current_native_controller(void *controller, std::uint64_t *generation_out) {
  const std::uint64_t submitted_generation = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const bool current =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      controller != nullptr && submitted_generation != 0 &&
      g_native_render_controller.load(std::memory_order_acquire) == controller &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) == submitted_generation;
  if (generation_out != nullptr) {
    *generation_out = submitted_generation;
  }
  return current;
}

void replay_battle_controller_lifecycle_hook(void *controller) {
  ReplayBattleControllerLifecycle original = g_original_replay_battle_controller_lifecycle;
  if (original == nullptr) {
    return;
  }
  original(controller);

  if (g_runner_mode.load(std::memory_order_acquire) != static_cast<std::int32_t>(RunnerMode::NativeRender) ||
      controller == nullptr || g_native_render_controller.load(std::memory_order_acquire) != controller) {
    return;
  }

  const std::int32_t state = read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset);
  const std::int32_t prior_state = g_native_controller_last_state.exchange(state, std::memory_order_acq_rel);
  if (state != prior_state) {
    LOGI("native ReplayBattleController lifecycle controller=%p state=%d prior=%d", controller, state, prior_state);
  }
  if (state != 8) {
    return;
  }

  const std::uint64_t submitted_generation = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  pthread_mutex_lock(&g_control_mutex);
  if (submitted_generation != 0 &&
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      g_native_render_controller.load(std::memory_order_acquire) == controller &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) != submitted_generation) {
    g_native_render_loaded_sequence.store(submitted_generation, std::memory_order_release);
    pthread_cond_broadcast(&g_control_cond);
    LOGI("native ReplayBattleController active: generation=%llu controller=%p manager=%p",
         static_cast<unsigned long long>(submitted_generation), controller,
         g_native_render_manager.load(std::memory_order_acquire));
  }
  pthread_mutex_unlock(&g_control_mutex);
}

void replay_battle_controller_full_update_hook(void *controller, float delta_seconds) {
  ReplayBattleControllerUpdate original = g_original_replay_battle_controller_full_update;
  if (original == nullptr) {
    return;
  }
  std::uint64_t generation = 0;
  if (!is_current_native_controller(controller, &generation)) {
    original(controller, delta_seconds);
    return;
  }

  suppress_offline_connection_error_if_needed();
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  if (manager != nullptr) {
    process_pending_snapshot_operation_on_worker(manager, RunnerMode::NativeRender);

    process_pending_native_command(manager);
  }
  pthread_mutex_lock(&g_control_mutex);
  const bool still_current = g_native_render_controller.load(std::memory_order_acquire) == controller &&
                             g_native_render_submitted_sequence.load(std::memory_order_acquire) == generation &&
                             g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation;
  if (!still_current) {
    pthread_mutex_unlock(&g_control_mutex);
    original(controller, delta_seconds);
    return;
  }
  const bool paused = g_native_render_paused;
  const float stock_speed = paused ? 0.0f : static_cast<float>(g_native_render_speed_quarters) / 4.0f;
  if (!paused) {
    g_native_render_step_active = true;
  }
  pthread_mutex_unlock(&g_control_mutex);

  // This is the controller's own playback multiplier, consumed immediately
  // before its 20 Hz clock and presentation pipeline.
  std::memcpy(static_cast<std::uint8_t *>(controller) + kReplayBattleControllerSpeedOffset, &stock_speed,
              sizeof(stock_speed));
  const std::uint64_t call = g_native_controller_full_updates.fetch_add(1, std::memory_order_relaxed) + 1;
  if (call <= 3) {
    LOGI("native controller full update #%llu controller=%p state=%d dt=%.6f speed=%.2f paused=%d",
         static_cast<unsigned long long>(call), controller,
         read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset),
         static_cast<double>(delta_seconds), static_cast<double>(stock_speed), paused ? 1 : 0);
  }
  original(controller, delta_seconds);

  if (!paused) {
    pthread_mutex_lock(&g_control_mutex);
    g_native_render_step_active = false;
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);
  }
}

void replay_battle_controller_clock_hook(void *controller, float delta_seconds) {
  ReplayBattleControllerUpdate original = g_original_replay_battle_controller_clock;
  if (original == nullptr) {
    return;
  }
  if (g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      controller != nullptr && g_native_render_controller.load(std::memory_order_acquire) == controller) {
    const std::uint64_t call = g_native_controller_clock_updates.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call <= 3) {
      void *const manager = read_object_field<void *>(controller, 0x90);
      void *const world = manager == nullptr ? nullptr : read_object_field<void *>(manager, 0xa8);
      const auto global_pause_guard = reinterpret_cast<NoArgInt32>(g_libg_base + kReplayClockGlobalPauseGuardOffset);
      const auto game_app_getter = reinterpret_cast<NoArgPointer>(g_libg_base + kGameAppGetterOffset);
      const auto game_app_pause_guard = reinterpret_cast<ObjectInt32>(g_libg_base + kGameAppPauseGuardOffset);
      const std::int32_t global_pause = global_pause_guard();
      void *const game_app = game_app_getter();
      const std::int32_t game_app_pause = game_app == nullptr ? -1 : game_app_pause_guard(game_app);
      const std::int32_t game_app_state = game_app == nullptr ? -1 : read_object_field<std::int32_t>(game_app, 0x1bc);
      LOGI("native controller clock #%llu controller=%p state=%d suspended=%u dt=%.6f accumulator=%.6f speed=%.2f "
           "manager=%p world=%p worldState=%d globalPause=%d gameApp=%p gameAppPause=%d gameAppState=%d",
           static_cast<unsigned long long>(call), controller,
           read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset),
           static_cast<unsigned int>(
               read_object_field<std::uint8_t>(controller, kReplayBattleControllerSuspendedOffset)),
           static_cast<double>(delta_seconds),
           static_cast<double>(read_object_field<float>(controller, kReplayBattleControllerAccumulatorOffset)),
           static_cast<double>(read_object_field<float>(controller, kReplayBattleControllerSpeedOffset)), manager,
           world, world == nullptr ? -1 : read_object_field<std::int32_t>(world, 0x18), global_pause, game_app,
           game_app_pause, game_app_state);
    }
  }
  original(controller, delta_seconds);
}

void replay_battle_controller_alternate_update_hook(void *controller, float delta_seconds) {
  ReplayBattleControllerUpdate original = g_original_replay_battle_controller_alternate_update;
  if (original == nullptr) {
    return;
  }
  std::uint64_t generation = 0;
  if (!is_current_native_controller(controller, &generation)) {
    original(controller, delta_seconds);
    return;
  }

  suppress_offline_connection_error_if_needed();
  void *const manager = g_native_render_manager.load(std::memory_order_acquire);
  if (manager != nullptr) {
    process_pending_snapshot_operation_on_worker(manager, RunnerMode::NativeRender);

    process_pending_native_command(manager);
  }
  pthread_mutex_lock(&g_control_mutex);
  const bool still_current = g_native_render_controller.load(std::memory_order_acquire) == controller &&
                             g_native_render_submitted_sequence.load(std::memory_order_acquire) == generation &&
                             g_native_render_loaded_sequence.load(std::memory_order_acquire) == generation;
  if (!still_current) {
    pthread_mutex_unlock(&g_control_mutex);
    original(controller, delta_seconds);
    return;
  }
  const bool paused = g_native_render_paused;
  const float scaled_delta =
      paused ? 0.0f : delta_seconds * (static_cast<float>(g_native_render_speed_quarters) / 4.0f);
  if (!paused) {
    g_native_render_step_active = true;
  }
  pthread_mutex_unlock(&g_control_mutex);

  const std::uint64_t call = g_native_controller_alternate_updates.fetch_add(1, std::memory_order_relaxed) + 1;
  if (call <= 3) {
    LOGI("native controller alternate update #%llu controller=%p dt=%.6f scaled=%.6f paused=%d",
         static_cast<unsigned long long>(call), controller, static_cast<double>(delta_seconds),
         static_cast<double>(scaled_delta), paused ? 1 : 0);
  }
  original(controller, scaled_delta);

  if (!paused) {
    pthread_mutex_lock(&g_control_mutex);
    g_native_render_step_active = false;
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);
  }
}

void game_state_load_hook(void *manager, void *battle, void *command_array, void *auxiliary) {
  const std::uint32_t call = g_load_calls.fetch_add(1, std::memory_order_relaxed) + 1;
  const bool headless_candidate = auxiliary == nullptr && battle != nullptr &&
                                  g_expected_headless_load_manager.load(std::memory_order_acquire) == manager;
  const bool native_render_candidate =
      auxiliary != nullptr && battle != nullptr &&
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender);
  LOGI("GameStateManager::loadReplay #%u tid=%ld manager=%p root=%p sidecar=%p aux=%p headless=%d", call,
       static_cast<long>(syscall(__NR_gettid)), manager, battle, command_array, auxiliary, headless_candidate ? 1 : 0);

  if (headless_candidate) {
    // Publish the predicted generation before loadReplay so committed
    // initial insertions (including towers) are not lost.  The values are
    // committed immediately below when this exact manager returns.
    begin_runtime_telemetry_epoch(manager, g_control_generation + 1, g_state_epoch + 1);
  } else if (native_render_candidate) {
    const std::uint64_t submitted_generation = g_native_render_submitted_sequence.load(std::memory_order_acquire);
    begin_runtime_telemetry_epoch(manager, submitted_generation, submitted_generation);
  }

  // The third argument is an optional battle-log JSON sidecar, not a
  // compiled command container.  The manager constructor owns the real
  // LogicCommandQueue at manager+0x38, while loadReplay compiles replay.cmd
  // directly from the root JSON.  Cold matches therefore pass a null
  // sidecar and require cmd/evt to be empty before entering this function.
  g_original_game_state_load(manager, battle, command_array, auxiliary);

  void *command_queue = nullptr;
  std::int32_t loaded_queue_count = -1;
  std::memcpy(&command_queue, static_cast<std::uint8_t *>(manager) + 0x38, sizeof(command_queue));
  inspect_pointer_array(command_queue, &loaded_queue_count);
  const std::int32_t tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
  if (headless_candidate || native_render_candidate) {
    bind_combat_event_object_manager(manager);
  }
  LOGI("GameStateManager::loadReplay #%u returned at tick=%d queue=%p queue_count=%d", call, tick, command_queue,
       loaded_queue_count);

  if (native_render_candidate) {
    void *const controller = static_cast<std::uint8_t *>(auxiliary) - 0x10;
    const std::uint64_t submitted_generation = g_native_render_submitted_sequence.load(std::memory_order_acquire);
    pthread_mutex_lock(&g_control_mutex);
    // Finish any mailbox item belonging to the prior native manager before
    // publishing the new generation. Waiting API callers will observe the
    // generation change and report failure instead of attaching to a reused
    // manager address.
    if (g_pending_native_command_sequence != g_processed_native_command_sequence) {
      g_last_native_injection_succeeded = false;
      g_processed_native_command_sequence = g_pending_native_command_sequence;
      g_pending_native_command_json[0] = '\0';
    }
    clear_scheduled_replay_actions_locked();
    g_native_render_controller.store(controller, std::memory_order_release);
    g_native_render_manager.store(manager, std::memory_order_release);
    g_native_render_tick.store(tick, std::memory_order_release);
    g_native_render_state_epoch = submitted_generation;
    g_native_render_advance_target_tick = -1;
    g_native_render_advance_reached = false;
    g_native_render_advance_ended = false;
    g_native_forced_steps.store(0, std::memory_order_release);
    g_native_stock_step_callbacks.store(0, std::memory_order_release);
    g_native_controller_full_updates.store(0, std::memory_order_release);
    g_native_controller_clock_updates.store(0, std::memory_order_release);
    g_native_controller_alternate_updates.store(0, std::memory_order_release);
    g_native_controller_last_state.store(-1, std::memory_order_release);
    // loadReplay returns during lifecycle state 3. Do not advertise the
    // native scene as ready until the controller lifecycle hook observes
    // the stock active state 8.
    g_native_render_loaded_sequence.store(0, std::memory_order_release);
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);
    LOGI("stock native-render manager captured pending active state: generation=%llu controller=%p manager=%p tick=%d "
         "queue=%d",
         static_cast<unsigned long long>(submitted_generation), controller, manager, tick, loaded_queue_count);
  }

  if (headless_candidate) {
    pthread_mutex_lock(&g_control_mutex);
    if (g_resident_mode_active) {
      clear_scheduled_replay_actions_for_manager_locked(manager);
    } else {
      clear_scheduled_replay_actions_locked();
    }
    g_last_loaded_queue_count = loaded_queue_count;
    g_controlled_manager = manager;
    ++g_control_generation;
    ++g_state_epoch;
    g_controlled_step_active = false;
    g_completed_steps = 0;
    g_step_budget = 0;
    g_last_controlled_tick = tick;
    g_pending_command_sequence = 0;
    g_processed_command_sequence = 0;
    g_pending_command_json[0] = '\0';
    g_last_injection_succeeded = false;
    g_control_gate_enabled = true;
    g_control_ended = false;
    pthread_cond_broadcast(&g_control_cond);
    const std::uint64_t generation = g_control_generation;
    pthread_mutex_unlock(&g_control_mutex);
    LOGI("cold headless manager armed: generation=%llu manager=%p sidecar=null loaded_queue=%d tick=%d",
         static_cast<unsigned long long>(generation), manager, loaded_queue_count, tick);
  }
}





void game_state_step_hook(void *manager) {
  const std::uint64_t submitted_generation = g_native_render_submitted_sequence.load(std::memory_order_acquire);
  const bool native_render_manager =
      g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender) &&
      submitted_generation != 0 &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) == submitted_generation &&
      g_native_render_manager.load(std::memory_order_acquire) == manager;
  if (native_render_manager) {
    pthread_mutex_lock(&g_control_mutex);
    const std::int32_t current_tick = g_native_render_tick.load(std::memory_order_acquire);
    const bool hold_at_synchronized_boundary =
        g_native_render_advance_target_tick >= 0 &&
        (g_native_render_advance_reached || current_tick >= g_native_render_advance_target_tick);
    if (hold_at_synchronized_boundary) {
      g_native_render_paused = true;
      g_native_render_advance_reached = true;
      pthread_cond_broadcast(&g_control_cond);
    }
    pthread_mutex_unlock(&g_control_mutex);
    if (hold_at_synchronized_boundary) {
      return;
    }
    g_native_stock_step_callbacks.fetch_add(1, std::memory_order_relaxed);
    // Usually consumed by the outer controller hook before this callback;
    // retain this fallback for stock paths that step the manager directly.

    // Resolve semantic replay actions on the exact pre-step world.  This
    // hook runs once for every native logic tick even when the renderer is
    // presenting at 4x, so the UI never has to brake to catch a boundary.
    process_scheduled_replay_actions(manager);
    process_pending_native_command(manager);
  }
  bool controlled = false;
  bool skip_step_for_reset = false;
  for (;;) {
    char pending_json[kMaxPendingCommandBytes] = {};
    std::uint64_t pending_sequence = 0;
    std::uint64_t snapshot_sequence = 0;
    std::uint64_t snapshot_handle = 0;
    SnapshotOperation snapshot_operation = SnapshotOperation::None;
    bool process_injection = false;
    bool process_snapshot_operation = false;

    pthread_mutex_lock(&g_control_mutex);
    while (g_controlled_manager == manager && g_control_gate_enabled && g_step_budget == 0 &&
           g_reset_request_sequence == g_reset_processed_sequence &&
           g_snapshot_request_sequence == g_snapshot_processed_sequence &&
           g_pending_command_sequence == g_processed_command_sequence) {
      pthread_cond_wait(&g_control_cond, &g_control_mutex);
    }
    if (g_controlled_manager == manager) {
      controlled = true;
      if (g_reset_request_sequence != g_reset_processed_sequence) {
        skip_step_for_reset = true;
      } else if (g_snapshot_request_sequence != g_snapshot_processed_sequence) {
        snapshot_sequence = g_snapshot_request_sequence;
        snapshot_handle = g_pending_snapshot_handle;
        snapshot_operation = g_pending_snapshot_operation;
        process_snapshot_operation = true;

      } else if (g_pending_command_sequence != g_processed_command_sequence) {
        pending_sequence = g_pending_command_sequence;
        std::memcpy(pending_json, g_pending_command_json, sizeof(pending_json));
        pending_json[sizeof(pending_json) - 1] = '\0';
        process_injection = true;
      } else if (g_control_gate_enabled && g_step_budget != 0) {
        --g_step_budget;
      }
      if (!skip_step_for_reset && !process_snapshot_operation && !process_injection) {
        g_controlled_step_active = true;
      }
    }
    pthread_mutex_unlock(&g_control_mutex);

    if (process_snapshot_operation) {
      process_snapshot_operation_on_worker(manager, RunnerMode::Headless, snapshot_sequence, snapshot_operation,
                                           snapshot_handle);
      continue;
    }

    if (!process_injection) {
      break;
    }

    const bool injection_succeeded = inject_logic_command(manager, pending_json);
    pthread_mutex_lock(&g_control_mutex);
    if (g_controlled_manager == manager && g_processed_command_sequence < pending_sequence) {
      g_last_injection_succeeded = injection_succeeded;
      g_processed_command_sequence = pending_sequence;
      pthread_cond_broadcast(&g_control_cond);
    }
    pthread_mutex_unlock(&g_control_mutex);
  }

  if (skip_step_for_reset) {
    return;
  }

  if (controlled) {
    // Resolve the same semantic replay schedule used by native-render on
    // the exact pre-step headless world.  No host observe/inject round trip
    // is needed at individual action boundaries.
    process_scheduled_replay_actions(manager);
  }
  constexpr std::uint32_t performed_step_calls = 1;
  BattleResultView stepped_battle_result;
  g_original_game_state_step(manager);
  const bool stepped_result_readable =
      read_battle_result(read_object_field<void *>(manager, 0xa8), &stepped_battle_result);

  const std::int32_t tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
  if (native_render_manager && g_native_render_manager.load(std::memory_order_acquire) == manager &&
      g_native_render_loaded_sequence.load(std::memory_order_acquire) == submitted_generation &&
      g_native_render_submitted_sequence.load(std::memory_order_acquire) == submitted_generation) {
    pthread_mutex_lock(&g_control_mutex);
    g_native_render_tick.store(tick, std::memory_order_release);
    g_native_forced_steps.fetch_add(1, std::memory_order_relaxed);
    if (g_native_render_advance_target_tick >= 0 &&
        (tick >= g_native_render_advance_target_tick || (stepped_result_readable && stepped_battle_result.finalized))) {
      g_native_render_paused = true;
      g_native_render_advance_reached = true;
      g_native_render_advance_ended = stepped_result_readable && stepped_battle_result.finalized;
      pthread_cond_broadcast(&g_control_cond);
    }
    pthread_mutex_unlock(&g_control_mutex);
  }
  if (controlled) {
    pthread_mutex_lock(&g_control_mutex);
    if (g_controlled_manager == manager) {
      g_controlled_step_active = false;
      g_completed_steps += performed_step_calls;
      g_last_controlled_tick = tick;
      if (stepped_result_readable && stepped_battle_result.finalized) {
        g_control_ended = true;
        g_step_budget = 0;
      }
      pthread_cond_broadcast(&g_control_cond);
    }
    pthread_mutex_unlock(&g_control_mutex);
  }
  std::uint32_t manager_steps = 0;
  pthread_mutex_lock(&g_trace_mutex);
  ManagerTrace *free_slot = nullptr;
  ManagerTrace *trace = nullptr;
  for (auto &candidate : g_manager_traces) {
    if (candidate.manager == manager) {
      trace = &candidate;
      break;
    }
    if (candidate.manager == nullptr && free_slot == nullptr) {
      free_slot = &candidate;
    }
  }
  if (trace == nullptr) {
    trace = free_slot;
    if (trace != nullptr) {
      trace->manager = manager;
      trace->steps = 0;
    }
  }
  if (trace != nullptr) {
    trace->steps += performed_step_calls;
    manager_steps = trace->steps;
  }
  pthread_mutex_unlock(&g_trace_mutex);

  if (manager_steps <= 3 || manager_steps == 10 || manager_steps == 100 || manager_steps == 1000) {
    LOGI("GameStateManager::step tid=%ld manager=%p call=%u tick=%d", static_cast<long>(syscall(__NR_gettid)), manager,
         manager_steps, tick);
  }
}

void *native_controlled_worker_main(void *opaque) {
  (void)opaque;

  if (g_game_state_context_a == nullptr || g_game_state_context_b == nullptr || g_game_state_new == nullptr ||
      g_game_state_delete == nullptr || g_game_state_constructor == nullptr || g_game_state_setup == nullptr ||
      g_game_state_destroy == nullptr || g_game_state_base_destroy == nullptr || g_json_parse == nullptr ||
      g_json_member_find == nullptr) {
    LOGE("native controlled worker is missing required GameState or JSON functions");
    pthread_mutex_lock(&g_control_mutex);
    g_native_worker_running = false;
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);
    return nullptr;
  }

  for (;;) {
    char replay_json[kMaxReplayConfigBytes] = {};
    bool pending_configuration = false;
    std::uint64_t configuration_sequence = 0;
    pthread_mutex_lock(&g_control_mutex);
    pending_configuration = g_config_request_sequence != g_config_processed_sequence;
    configuration_sequence = g_config_request_sequence;
    const char *configuration_source = pending_configuration ? g_pending_replay_json : g_active_replay_json;
    std::memcpy(replay_json, configuration_source, sizeof(replay_json));
    replay_json[sizeof(replay_json) - 1] = '\0';
    pthread_mutex_unlock(&g_control_mutex);

    void *configured_battle = nullptr;
    void *configured_battle_object = nullptr;
    void *configured_commands = nullptr;
    void *configured_events = nullptr;
    void *configuration_root_address = nullptr;
    void *configuration_commands_address = nullptr;
    void *configuration_events_address = nullptr;
    std::int32_t configuration_stage = replay_json[0] == '\0' ? 0 : 1;
    std::int32_t configuration_command_count = -1;
    std::int32_t configuration_event_count = -1;
    std::int32_t configuration_location_id = 0;
    std::uint64_t configuration_command_words[8] = {};
    if (replay_json[0] != '\0') {
      const bool has_location = extract_location_id(replay_json, &configuration_location_id);
      configured_battle = parse_json_text(replay_json);
      configuration_root_address = configured_battle;
      if (configured_battle != nullptr) {
        configuration_stage = 2;
      }
      configured_battle_object = lookup_json_member(configured_battle, "battle");
      if (configured_battle_object != nullptr) {
        configuration_stage = 3;
      }
      configured_commands = lookup_json_member(configured_battle, "cmd");
      configuration_commands_address = configured_commands;
      if (inspect_pointer_array(configured_commands, &configuration_command_count)) {
        configuration_stage = 4;
        std::memcpy(configuration_command_words, configured_commands, sizeof(configuration_command_words));
      }
      configured_events = lookup_json_member(configured_battle, "evt");
      configuration_events_address = configured_events;
      if (inspect_pointer_array(configured_events, &configuration_event_count)) {
        configuration_stage = 5;
      }
      if (configured_battle_object == nullptr || !has_location ||
          !inspect_pointer_array(configured_commands, &configuration_command_count) ||
          !inspect_pointer_array(configured_events, &configuration_event_count) || configuration_command_count != 0 ||
          configuration_event_count != 0) {
        LOGE("cold match config rejected: root=%p battle=%p location=%d cmd=%p/%d evt=%p/%d", configured_battle,
             configured_battle_object, configuration_location_id, configured_commands, configuration_command_count,
             configured_events, configuration_event_count);
        release_json_object(configured_battle);
        configured_battle = nullptr;
      } else {
        configuration_stage = 6;
      }
    }

    if (pending_configuration) {
      pthread_mutex_lock(&g_control_mutex);
      if (configuration_sequence == g_config_request_sequence && g_config_processed_sequence < configuration_sequence) {
        g_last_config_stage = configuration_stage;
        g_last_config_root = configuration_root_address;
        g_last_config_commands = configuration_commands_address;
        g_last_config_events = configuration_events_address;
        g_last_config_command_count = configuration_command_count;
        g_last_config_event_count = configuration_event_count;
        g_last_config_location_id = configuration_location_id;
        std::memcpy(g_last_config_command_words, configuration_command_words, sizeof(g_last_config_command_words));
        g_last_config_succeeded = false;
        if (configured_battle == nullptr) {
          g_config_processed_sequence = configuration_sequence;
          pthread_cond_broadcast(&g_control_cond);
        }
      }
      pthread_mutex_unlock(&g_control_mutex);

      if (configured_battle == nullptr) {
        pthread_mutex_lock(&g_control_mutex);
        std::memcpy(replay_json, g_active_replay_json, sizeof(replay_json));
        replay_json[sizeof(replay_json) - 1] = '\0';
        pthread_mutex_unlock(&g_control_mutex);
        if (replay_json[0] != '\0') {
          const bool has_location = extract_location_id(replay_json, &configuration_location_id);
          configured_battle = parse_json_text(replay_json);
          configured_battle_object = lookup_json_member(configured_battle, "battle");
          configured_commands = lookup_json_member(configured_battle, "cmd");
          configured_events = lookup_json_member(configured_battle, "evt");
          if (configured_battle_object == nullptr || !has_location ||
              !inspect_pointer_array(configured_commands, &configuration_command_count) ||
              !inspect_pointer_array(configured_events, &configuration_event_count) ||
              configuration_command_count != 0 || configuration_event_count != 0) {
            release_json_object(configured_battle);
            configured_battle = nullptr;
          }
        }
      }
    }

    const void *battle = configured_battle;
    if (battle == nullptr) {
      LOGE("native controlled worker has no valid cold match configuration");
      release_json_object(configured_battle);
      pthread_mutex_lock(&g_control_mutex);
      g_native_worker_running = false;
      pthread_cond_broadcast(&g_control_cond);
      pthread_mutex_unlock(&g_control_mutex);
      return nullptr;
    }

    void *const context_a = g_game_state_context_a();
    void *const context_b = g_game_state_context_b();
    if (!prewarm_location_tilemap(context_a, configuration_location_id)) {
      LOGE("native controlled worker could not prepare local tilemap for location %d", configuration_location_id);
      if (pending_configuration) {
        pthread_mutex_lock(&g_control_mutex);
        if (configuration_sequence == g_config_request_sequence &&
            g_config_processed_sequence < configuration_sequence) {
          g_last_config_succeeded = false;
          g_config_processed_sequence = configuration_sequence;
          pthread_cond_broadcast(&g_control_cond);
        }
        pthread_mutex_unlock(&g_control_mutex);
      }
      release_json_object(configured_battle);
      pthread_mutex_lock(&g_control_mutex);
      g_native_worker_running = false;
      pthread_cond_broadcast(&g_control_cond);
      pthread_mutex_unlock(&g_control_mutex);
      return nullptr;
    }
    void *const manager = g_game_state_new(0x230);
    if (manager == nullptr) {
      LOGE("native controlled worker could not allocate GameStateManager");
      pthread_mutex_lock(&g_control_mutex);
      g_native_worker_running = false;
      pthread_cond_broadcast(&g_control_cond);
      pthread_mutex_unlock(&g_control_mutex);
      release_json_object(configured_battle);
      return nullptr;
    }

    LOGI("native controlled worker constructing manager=%p root=%p sidecar=null contextA=%p contextB=%p tid=%ld",
         manager, battle, context_a, context_b, static_cast<long>(syscall(__NR_gettid)));
    g_game_state_constructor(manager, 1, context_a, context_b, 0);
    g_game_state_setup(manager, 1);
    g_expected_headless_load_manager.store(manager, std::memory_order_release);
    game_state_load_hook(manager, const_cast<void *>(battle), nullptr, nullptr);
    g_expected_headless_load_manager.store(nullptr, std::memory_order_release);

    if (pending_configuration) {
      pthread_mutex_lock(&g_control_mutex);
      if (configuration_sequence == g_config_request_sequence && g_config_processed_sequence < configuration_sequence) {
        const bool loaded = g_controlled_manager == manager && g_last_loaded_queue_count == 0;
        g_last_config_succeeded = loaded;
        if (loaded) {
          std::memcpy(g_active_replay_json, g_pending_replay_json, sizeof(g_active_replay_json));
          g_active_replay_json[sizeof(g_active_replay_json) - 1] = '\0';
        }
        g_config_processed_sequence = configuration_sequence;
        pthread_cond_broadcast(&g_control_cond);
      }
      pthread_mutex_unlock(&g_control_mutex);
    }

    pthread_mutex_lock(&g_control_mutex);
    g_reset_processed_sequence = g_reset_request_sequence;
    pthread_cond_broadcast(&g_control_cond);
    pthread_mutex_unlock(&g_control_mutex);

    std::int32_t previous_tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
    std::uint32_t stagnant_steps = 0;
    for (;;) {
      game_state_step_hook(manager);
      pthread_mutex_lock(&g_control_mutex);
      const bool reset_requested = g_reset_request_sequence != g_reset_processed_sequence;
      pthread_mutex_unlock(&g_control_mutex);
      if (reset_requested) {
        break;
      }

      const std::int32_t tick = g_game_state_tick == nullptr ? -1 : g_game_state_tick(manager);
      if (tick == previous_tick) {
        ++stagnant_steps;
      } else {
        stagnant_steps = 0;
        previous_tick = tick;
      }
      if (stagnant_steps >= 4) {
        pthread_mutex_lock(&g_control_mutex);
        if (g_controlled_manager == manager) {
          g_control_gate_enabled = true;
          g_step_budget = 0;
          g_control_ended = true;
          pthread_cond_broadcast(&g_control_cond);
        }
        pthread_mutex_unlock(&g_control_mutex);
      }
    }

    release_all_stored_snapshots_on_worker();
    end_combat_event_epoch(manager);

    pthread_mutex_lock(&g_control_mutex);
    if (g_controlled_manager == manager) {
      g_controlled_step_active = false;
      g_controlled_manager = nullptr;
      g_last_controlled_tick = -1;
      g_step_budget = 0;
      pthread_cond_broadcast(&g_control_cond);
    }
    pthread_mutex_unlock(&g_control_mutex);

    pthread_mutex_lock(&g_trace_mutex);
    for (auto &trace : g_manager_traces) {
      if (trace.manager == manager) {
        trace = {};
        break;
      }
    }
    pthread_mutex_unlock(&g_trace_mutex);

    LOGI("native controlled worker destroying manager=%p for reset", manager);
    g_game_state_destroy(manager);
    g_game_state_base_destroy(manager);
    g_game_state_delete(manager);
    release_json_object(configured_battle);

    pthread_mutex_lock(&g_control_mutex);
    if (g_resident_mode_requested) {
      g_native_worker_running = false;
      pthread_cond_broadcast(&g_control_cond);
      pthread_mutex_unlock(&g_control_mutex);
      LOGI("legacy native worker exited for resident multi-match mode");
      return nullptr;
    }
    pthread_mutex_unlock(&g_control_mutex);
  }
}

bool start_native_controlled_worker_locked() {
  if (g_native_worker_running) {
    return true;
  }
  g_native_worker_running = true;

  pthread_t thread{};
  const int error = pthread_create(&thread, nullptr, &native_controlled_worker_main, nullptr);
  if (error != 0) {
    g_native_worker_running = false;
    LOGE("pthread_create(native controlled worker) failed: %d", error);
    return false;
  }
  pthread_detach(thread);
  return true;
}

void native_replay_runner_hook(void *owner, void *battle, void *command_array) {
  pthread_mutex_lock(&g_control_mutex);
  const RunnerMode mode = static_cast<RunnerMode>(g_runner_mode.load(std::memory_order_acquire));
  const bool stock_owned = mode == RunnerMode::NativeRender;
  if (stock_owned) {
    pthread_mutex_unlock(&g_control_mutex);
    g_original_native_replay_runner(owner, battle, command_array);
    return;
  }

  pthread_mutex_unlock(&g_control_mutex);
  LOGI("stock replay pre-run bypassed; headless matches use cold configure (owner=%p root=%p sidecar=%p)", owner,
       battle, command_array);
}

#include "native_touch_probe_glue.inc"

bool install_inline_hook(std::uintptr_t libg_base, std::uintptr_t offset, const std::uint8_t (&expected_prologue)[16],
                         const void *replacement, void **original, const char *label) {
#if !defined(__aarch64__)
  (void)libg_base;
  (void)offset;
  (void)expected_prologue;
  (void)replacement;
  (void)original;
  (void)label;
  LOGE("probe must run as AArch64 code");
  return false;
#else
  auto *target = reinterpret_cast<std::uint8_t *>(libg_base + offset);
  if (std::memcmp(target, expected_prologue, sizeof(expected_prologue)) != 0) {
    LOGE("%s prologue mismatch at +0x%zx", label, static_cast<std::size_t>(offset));
    return false;
  }

  void *trampoline = mmap(nullptr, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (trampoline == MAP_FAILED) {
    LOGE("mmap trampoline failed");
    return false;
  }
  std::memcpy(trampoline, target, sizeof(expected_prologue));
  emit_absolute_jump(static_cast<std::uint8_t *>(trampoline) + 16, target + 16);
  __builtin___clear_cache(static_cast<char *>(trampoline), static_cast<char *>(trampoline) + 32);

  if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
    LOGE("mprotect RWX failed");
    return false;
  }
  *original = trampoline;
  emit_absolute_jump(target, replacement);
  __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target + 16));
  make_page_writable(target, PROT_READ | PROT_EXEC);

  LOGI("%s hook installed: libg=%p target=%p trampoline=%p", label, reinterpret_cast<void *>(libg_base), target,
       trampoline);
  return true;
#endif
}

bool install_hooks(std::uintptr_t libg_base) {
  g_libg_base = libg_base;
  const bool translated_sha256_compatible = install_translated_sha256_compatibility_shim(libg_base);
  g_egl_swap_buffers_got = reinterpret_cast<void **>(libg_base + kEglSwapBuffersGotOffset);
  void *egl_swap_buffers = nullptr;
  std::memcpy(&egl_swap_buffers, g_egl_swap_buffers_got, sizeof(egl_swap_buffers));
  g_original_egl_swap_buffers = reinterpret_cast<EglSwapBuffers>(egl_swap_buffers);
  LOGI("render gate GOT=%p original eglSwapBuffers=%p", g_egl_swap_buffers_got, egl_swap_buffers);
  const bool render_gate_installed = write_egl_swap_buffers_got(reinterpret_cast<void *>(&cr_render_gate_swap_buffers));
  g_hook_render_gate.store(render_gate_installed, std::memory_order_release);
  bool success = render_gate_installed && translated_sha256_compatible;
  LOGI("permanent frame/render shim installed=%d", render_gate_installed ? 1 : 0);
  g_replay_manager_getter = reinterpret_cast<ReplayManagerGetter>(libg_base + kReplayManagerGetterOffset);
  g_libg_string_from_utf8 = reinterpret_cast<LibgStringFromUtf8>(libg_base + kLibgStringFromUtf8Offset);
  g_game_state_tick = reinterpret_cast<GameStateTick>(libg_base + kGameStateTickOffset);
  g_game_state_context_a = reinterpret_cast<GameStateContextGetter>(libg_base + kGameStateContextAOffset);
  g_game_state_context_b = reinterpret_cast<GameStateContextGetter>(libg_base + kGameStateContextBOffset);
  g_game_state_new = reinterpret_cast<GameStateNew>(libg_base + kGameStateNewOffset);
  g_game_state_delete = reinterpret_cast<GameStateDelete>(libg_base + kGameStateDeleteOffset);
  g_game_state_constructor = reinterpret_cast<GameStateConstructor>(libg_base + kGameStateConstructorOffset);
  g_game_state_setup = reinterpret_cast<GameStateSetup>(libg_base + kGameStateSetupOffset);
  g_game_state_destroy = reinterpret_cast<GameStateDestroy>(libg_base + kGameStateDestroyOffset);
  g_game_state_base_destroy = reinterpret_cast<GameStateDestroy>(libg_base + kGameStateBaseDestroyOffset);
  g_json_parse = reinterpret_cast<JsonParse>(libg_base + kJsonParseOffset);
  g_json_member_find = reinterpret_cast<JsonMemberFind>(libg_base + kJsonMemberFindOffset);
  g_logic_command_parse = reinterpret_cast<void *>(libg_base + kLogicCommandParseOffset);
  g_logic_command_append = reinterpret_cast<LogicCommandAppend>(libg_base + kLogicCommandAppendOffset);
  g_logic_command_queue_constructor =
      reinterpret_cast<LogicCommandQueueConstructor>(libg_base + kLogicCommandQueueConstructorOffset);
  g_logic_command_queue_destroy =
      reinterpret_cast<LogicCommandQueueDestroy>(libg_base + kLogicCommandQueueDestroyOffset);
  g_state_snapshot = reinterpret_cast<void *>(libg_base + kStateSnapshotOffset);
  g_state_snapshot_delete = reinterpret_cast<StateSnapshotDelete>(libg_base + kStateSnapshotDeleteOffset);
  g_byte_stream_reset_read = reinterpret_cast<ByteStreamReset>(libg_base + kByteStreamResetReadOffset);
  g_byte_stream_flip_read = reinterpret_cast<ByteStreamReset>(libg_base + kByteStreamFlipReadOffset);
  g_byte_stream_reset_checksum = reinterpret_cast<ByteStreamReset>(libg_base + kByteStreamResetChecksumOffset);
  g_game_state_deserialize_header =
      reinterpret_cast<GameStateDeserializeHeader>(libg_base + kGameStateDeserializeHeaderOffset);
  g_game_state_deserialize_body =
      reinterpret_cast<GameStateDeserializeBody>(libg_base + kGameStateDeserializeBodyOffset);
  g_card_selection_build = reinterpret_cast<void *>(libg_base + kCardSelectionBuildOffset);
  g_logic_data_by_global_id = reinterpret_cast<LogicDataByGlobalId>(libg_base + kLogicDataByGlobalIdOffset);
  g_location_tilemap_ensure = reinterpret_cast<LocationTileMapEnsure>(libg_base + kLocationTileMapEnsureOffset);
  g_location_tilemap_lookup = reinterpret_cast<LocationTileMapLookup>(libg_base + kLocationTileMapLookupOffset);
  const NativeHookSpec runner_hooks[] = {
      native_hook(kNativeReplayRunnerOffset, kNativeReplayRunnerPrologue, &native_replay_runner_hook,
                  &g_original_native_replay_runner, "native replay pre-run", &g_hook_native_replay_runner),
      native_hook(kReplayJsonSetterOffset, kReplayJsonSetterPrologue, &replay_json_setter_hook,
                  &g_original_replay_json_setter, "replay JSON setter", &g_hook_replay_json_setter),
      native_hook(kGameAppPendingEventSetterOffset, kGameAppPendingEventSetterPrologue,
                  &game_app_pending_event_setter_hook, &g_original_game_app_pending_event_setter,
                  "GameApp::pendingEvent", &g_hook_game_app_pending_event_setter),
      native_hook(kReplayBattleControllerLifecycleOffset, kReplayBattleControllerLifecyclePrologue,
                  &replay_battle_controller_lifecycle_hook, &g_original_replay_battle_controller_lifecycle,
                  "ReplayBattleController::lifecycle", &g_hook_replay_battle_controller_lifecycle),
      native_hook(kReplayBattleControllerFullUpdateOffset, kReplayBattleControllerFullUpdatePrologue,
                  &replay_battle_controller_full_update_hook, &g_original_replay_battle_controller_full_update,
                  "ReplayBattleController::fullUpdate", &g_hook_replay_battle_controller_full_update),
      native_hook(kReplayBattleControllerClockOffset, kReplayBattleControllerClockPrologue,
                  &replay_battle_controller_clock_hook, &g_original_replay_battle_controller_clock,
                  "ReplayBattleController::clock", &g_hook_replay_battle_controller_clock),
      native_hook(kReplayBattleControllerAlternateUpdateOffset, kReplayBattleControllerAlternateUpdatePrologue,
                  &replay_battle_controller_alternate_update_hook,
                  &g_original_replay_battle_controller_alternate_update, "ReplayBattleController::alternateUpdate",
                  &g_hook_replay_battle_controller_alternate_update),
      native_hook(kGameStateLoadOffset, kGameStateLoadPrologue, &game_state_load_hook, &g_original_game_state_load,
                  "GameStateManager::loadReplay", &g_hook_game_state_load),
      native_hook(kGameStateStepOffset, kGameStateStepPrologue, &game_state_step_hook, &g_original_game_state_step,
                  "GameStateManager::step", &g_hook_game_state_step),
  };
  success &= install_native_hook_table(libg_base, runner_hooks, sizeof(runner_hooks) / sizeof(runner_hooks[0]));
  const bool touch_installed = install_probe_native_touch(libg_base);
  success &= touch_installed;
  // Combat events are an independently attested, fail-closed capability.
  // A mismatch must not disable the already-established runner lifecycle;
  // the nested combat envelope reports unavailable and emits no records.
  install_combat_event_hooks(libg_base);
  // Phase/timing hooks follow the same independent exact-build boundary.
  // Their envelope remains unavailable when attestation or any ABI hook
  // fails; runner lifecycle success is intentionally unaffected.
  install_phase_runtime_hooks(libg_base);
  // Ordinary dash execution is a distinct native domain from active Buff
  // effects.  Its caller-filtered envelope remains independently
  // fail-closed and never synthesizes EffectState.stage.
  install_special_movement_runtime_hook(libg_base);
  // Action-owned chain hops and atomic warps have a separate, exact-build
  // allowlist.  This envelope reports native action stages only and never
  // synthesizes EffectState.stage.
  install_action_movement_runtime_hooks(libg_base);
  // The shared LogicCharacter state setter is filtered to the exact
  // movement-card allowlist.  Raw state integers and callers remain a
  // separate evidentiary domain until named live goldens map them.
  install_character_state_runtime_hook(libg_base);
  // Visibility reuses the phase-owned f5a2d4 hook and uniquely owns
  // f5a578.  Contextual acquisition gates remain unavailable until a
  // PC-relative-safe AArch64 bridge exists.
  install_visibility_runtime_hooks(libg_base, g_phase_buff_added_hook_installed.load(std::memory_order_acquire));
  // Remaining non-snapshot mechanics join exact independent hooks to the
  // already-attested combat/phase lifecycle. A partial set stays
  // unavailable and records its rejected-count fail-closed.
  install_remaining_runtime_hooks(libg_base);
  // Dagger Duchess inventory and Royal Chef cooking live in their starting
  // Action runtimes, outside ordinary object/component snapshots.
  install_tower_troop_runtime_hooks(libg_base);
  g_hook_install_ok.store(success, std::memory_order_release);
  return success;
}

void initialize_probe_runtime_once(std::uintptr_t libg_base) {
  bool expected = false;
  if (!g_probe_runtime_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
    return;
  }
  const bool hooks_installed = install_hooks(libg_base);
  LOGI("runner hook installation complete=%d deferred=%d", hooks_installed ? 1 : 0,
       g_deferred_probe_bootstrap_started.load(std::memory_order_acquire) ? 1 : 0);
  start_control_server(g_engine_instance_port);
}

void *deferred_probe_bootstrap_main(void *) {
  constexpr int kPollAttempts = 1200;
  constexpr useconds_t kPollIntervalUs = 50u * 1000u;
  constexpr useconds_t kLibgSettleUs = 2u * 1000u * 1000u;
  constexpr useconds_t kExplicitBootstrapGraceUs = 30u * 1000u * 1000u;
  pthread_setname_np(pthread_self(), "CRProbeBoot");
  LOGI("deferred libg bootstrap worker running");
  // GameApp invokes nativeBootstrapProbe immediately after libg finishes
  // loading.  Scanning /proc maps from an ARM-translated pthread while that
  // loader is active can crash the whole process before the explicit JNI
  // callback runs.  Keep this worker only as a delayed fallback.
  usleep(kExplicitBootstrapGraceUs);
  if (g_probe_runtime_started.load(std::memory_order_acquire)) {
    LOGI("explicit post-libg bootstrap completed during fallback grace");
    return nullptr;
  }
  for (int attempt = 0; attempt < kPollAttempts; ++attempt) {
    if (g_probe_runtime_started.load(std::memory_order_acquire)) {
      LOGI("explicit post-libg bootstrap completed before fallback scan");
      return nullptr;
    }
    std::uintptr_t libg_base = find_libg_base_from_proc_maps();
    if (libg_base != 0) {
      // Loading libg through a native bridge can publish its mappings
      // before all constructors have completed.  Give that startup a
      // short head start before replacing any instructions or GOT slots.
      usleep(kLibgSettleUs);
      libg_base = find_libg_base_from_proc_maps();
      if (libg_base != 0) {
        LOGI("deferred libg bootstrap found base=%p attempt=%d", reinterpret_cast<void *>(libg_base), attempt + 1);
        initialize_probe_runtime_once(libg_base);
        return nullptr;
      }
    }
    if (attempt == 19) {
      LOGI("deferred libg bootstrap still waiting after one second");
    }
    usleep(kPollIntervalUs);
  }
  LOGE("deferred libg bootstrap timed out after 60 seconds");
  return nullptr;
}

void start_deferred_probe_bootstrap() {
  bool expected = false;
  if (!g_deferred_probe_bootstrap_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                                  std::memory_order_acquire)) {
    return;
  }
  pthread_t thread{};
  const int result = pthread_create(&thread, nullptr, deferred_probe_bootstrap_main, nullptr);
  if (result != 0) {
    g_deferred_probe_bootstrap_started.store(false, std::memory_order_release);
    LOGE("deferred libg bootstrap pthread_create failed errno=%d", result);
    return;
  }
  pthread_detach(thread);
  LOGI("deferred libg bootstrap thread started");
}

} // namespace

extern "C" JNIEXPORT void JNICALL Java_com_supercell_titan_GameApp_nativeBootstrapProbe(JNIEnv *, jclass) {
  std::uintptr_t libg_base = find_libg_base();
  if (libg_base == 0) {
    libg_base = find_libg_base_from_proc_maps();
  }
  if (libg_base == 0) {
    LOGE("explicit post-libg bootstrap could not find libg.so");
    return;
  }
  LOGI("explicit post-libg bootstrap found base=%p", reinterpret_cast<void *>(libg_base));
  initialize_probe_runtime_once(libg_base);
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  initialize_engine_instance();
  g_java_vm = vm;
  JNIEnv *env = nullptr;
  if (vm != nullptr && vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK && env != nullptr) {
    jclass local_dialog_manager = env->FindClass("com/supercell/titan/NativeDialogManager");
    if (local_dialog_manager != nullptr) {
      g_native_dialog_manager_class = static_cast<jclass>(env->NewGlobalRef(local_dialog_manager));
      env->DeleteLocalRef(local_dialog_manager);
      if (g_native_dialog_manager_class != nullptr) {
        g_native_dialog_dismiss_all =
            env->GetStaticMethodID(g_native_dialog_manager_class, "nativeDialogDismissAll", "()V");
      }
    }
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
  }
  const std::uintptr_t libg_base = find_libg_base();
  if (libg_base == 0) {
    // Native-bridge builds may lazily resolve guest libc calls.  Resolve
    // every syscall/parser used by the worker while JNI_OnLoad still owns
    // the loader recursively; otherwise libg's long-running initializer
    // can leave the first worker scan waiting on that loader lock.
    static_cast<void>(find_libg_base_from_proc_maps());
    usleep(0);
    LOGI("libg.so was not present during JNI_OnLoad; deferring bootstrap");
    start_deferred_probe_bootstrap();
    return JNI_VERSION_1_6;
  }
  initialize_probe_runtime_once(libg_base);
  return JNI_VERSION_1_6;
}
