#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

inline bool verify_native_binary(const char *path, const char *expected_sha256, const char *expected_build_id) {
  char sha256[65] = {}, build_id[41] = {};
  if (sha256_file(path, sha256) && libg_build_id(build_id) && std::strcmp(sha256, expected_sha256) == 0 &&
      std::strcmp(build_id, expected_build_id) == 0) {
    return true;
  }
  LOGE("native binary attestation mismatch sha=%s buildId=%s", sha256, build_id);
  return false;
}

// Include after install_inline_hook is declared. Fingerprints and installations
// share one ordered table, while custom instruction relocation remains explicit.
struct NativeHookSpec {
  std::uintptr_t offset;
  const std::uint8_t (&prologue)[16];
  const void *replacement;
  void **original;
  const char *label;
  std::atomic<bool> *installed;
};

template <typename Replacement, typename Original>
NativeHookSpec native_hook(std::uintptr_t offset, const std::uint8_t (&prologue)[16], Replacement replacement,
                           Original *original, const char *label, std::atomic<bool> *installed = nullptr) {
  return {offset, prologue, reinterpret_cast<const void *>(replacement), reinterpret_cast<void **>(original),
          label,  installed};
}

inline bool verify_native_hook_table(std::uintptr_t libg_base, const NativeHookSpec *hooks, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    const auto &hook = hooks[index];
    if (std::memcmp(reinterpret_cast<const void *>(libg_base + hook.offset), hook.prologue, sizeof(hook.prologue)) !=
        0) {
      LOGE("hook fingerprint mismatch label=%s offset=+0x%zx", hook.label, static_cast<std::size_t>(hook.offset));
      return false;
    }
  }
  return true;
}

inline bool install_native_hook_table(std::uintptr_t libg_base, const NativeHookSpec *hooks, std::size_t count,
                                      bool stop_on_failure = false) {
  bool complete = true;
  for (std::size_t index = 0; index < count; ++index) {
    const auto &hook = hooks[index];
    const bool installed =
        install_inline_hook(libg_base, hook.offset, hook.prologue, hook.replacement, hook.original, hook.label);
    if (hook.installed != nullptr) {
      hook.installed->store(installed, std::memory_order_release);
    }
    complete &= installed;
    if (!installed && stop_on_failure)
      return false;
  }
  return complete;
}
