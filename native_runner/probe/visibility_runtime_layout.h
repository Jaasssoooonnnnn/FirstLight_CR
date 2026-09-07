#pragma once

// Exact-build invisibility transitions only. The ordinary target-acquisition
// gate has no safe native hook and remains unavailable in the public schema.
#include <cstddef>
#include <cstdint>

namespace cr_visibility_runtime {

inline constexpr const char *kSchema = "native-visibility-runtime.v1";
inline constexpr const char *kExpectedLibgSha256 = "110aa2b5cac391c498645e072b0d88729428c2c2845e7e2737ca8ee979059783";
inline constexpr const char *kExpectedLibgBuildId = "90e6f351f0dec4a28b494c3812d2629fd45d5a83";
inline constexpr std::uintptr_t kBuffComponentVtableOffset = 0x018a1d58;
inline constexpr std::uintptr_t kBuffAssetVtableOffset = 0x01882500;
inline constexpr std::int32_t kBuffComponentType = 3;
inline constexpr std::size_t kComponentOwnerOffset = 0x08;
inline constexpr std::size_t kBuffEntryCapacityOffset = 0x20;
inline constexpr std::size_t kBuffEntryCountOffset = 0x24;
inline constexpr std::size_t kBuffInvisibleCountOffset = 0x34;
inline constexpr std::size_t kBuffEntryAssetOffset = 0x18;
inline constexpr std::size_t kBuffEntryComponentOffset = 0x20;
inline constexpr std::size_t kBuffAssetGlobalIdOffset = 0x40;
inline constexpr std::size_t kBuffAssetInvisibleOffset = 0x108;
inline constexpr std::uintptr_t kBuffApplyOffset = 0x00f5a2d4;
inline constexpr std::uintptr_t kBuffApplyOnlyCallOffset = 0x00f59ea8;
inline constexpr std::uintptr_t kBuffRemoveOffset = 0x00f5a578;
inline constexpr std::uintptr_t kBuffRemoveOnlyCallOffset = 0x00f59304;
inline constexpr std::uint8_t kBuffRemovePrologue[16] = {
    0xff, 0x43, 0x02, 0xd1, 0xfd, 0x7b, 0x03, 0xa9, 0xfc, 0x6f, 0x04, 0xa9, 0xfa, 0x67, 0x05, 0xa9,
};
inline constexpr std::uint8_t kBuffApplyCallFingerprint[16] = {
    0x0b, 0x01, 0x00, 0x94, 0xdc, 0x00, 0x00, 0xb4, 0x9f, 0x03, 0x18, 0xeb, 0x80, 0x00, 0x00, 0x54,
};
inline constexpr std::uint8_t kBuffRemoveCallFingerprint[16] = {
    0x9d, 0x04, 0x00, 0x94, 0x68, 0x32, 0x41, 0x39, 0x88, 0x02, 0x00, 0x34, 0x69, 0x22, 0x4b, 0x29,
};

enum class VisibilityTransition : std::uint8_t {
  None = 0,
  BecameInvisible = 1,
  BecameVisible = 2,
};

constexpr VisibilityTransition classify_visibility_transition(std::int32_t before, std::int32_t after) {
  if (before < 0 || after < 0) {
    return VisibilityTransition::None;
  }
  if (before == 0 && after > 0) {
    return VisibilityTransition::BecameInvisible;
  }
  if (before > 0 && after == 0) {
    return VisibilityTransition::BecameVisible;
  }
  return VisibilityTransition::None;
}

} // namespace cr_visibility_runtime
