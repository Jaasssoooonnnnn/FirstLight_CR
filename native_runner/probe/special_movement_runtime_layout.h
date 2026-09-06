#pragma once

// Exact-build ordinary special-movement layout for Null's Royale 15.535.13.
//
// This boundary is intentionally narrower than generic position telemetry:
// libg+f609dc is retained only when called from the type-0 attack-component
// path at return address f61bf0.  That caller reloads the configured dash
// cooldown, resolves the current target coordinates, and then executes the
// native position/state mutation.

#include <cstddef>
#include <cstdint>

namespace cr_special_movement_runtime {

constexpr const char *kSchema = "native-special-movement-runtime.v1";
constexpr const char *kExpectedLibgSha256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783";
constexpr const char *kExpectedLibgBuildId = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";

constexpr std::uintptr_t kExecuteOffset = 0x00f609dc;
constexpr std::uintptr_t kExecuteCallerReturnOffset = 0x00f61bf0;

constexpr std::uint8_t kExecutePrologue[16] = {
    0xff, 0xc3, 0x01, 0xd1, 0xfd, 0x7b, 0x01, 0xa9, 0xfc, 0x6f, 0x02, 0xa9, 0xfa, 0x67, 0x03, 0xa9,
};

constexpr std::size_t kComponentOwnerOffset = 0x08;
constexpr std::size_t kComponentTargetOffset = 0x10;
constexpr std::size_t kComponentDashCooldownOffset = 0x40;
constexpr std::size_t kObjectPositionXOffset = 0x7c;
constexpr std::size_t kObjectPositionYOffset = 0x80;
constexpr std::int32_t kExpectedOperationFlag = 1;

} // namespace cr_special_movement_runtime
