#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <type_traits>

// A bounded writer for the control protocol. Failed writes are sticky, so a
// caller can serialize a complete record and check good() once at its boundary.
// Raw fragments retain the protocol's explicit precision and field ordering.
class JsonWriter {
public:
  struct NamedInt {
    const char *name;
    std::int32_t value;
  };
  JsonWriter(char *buffer, std::size_t capacity, std::size_t &offset)
      : buffer_(buffer), capacity_(capacity), offset_(&offset) {}
  JsonWriter(char *buffer, std::size_t capacity, std::size_t *offset)
      : buffer_(buffer), capacity_(capacity), offset_(offset) {}

  bool good() const { return good_; }
  explicit operator bool() const { return good(); }

  bool append(const char *format, ...) {
    if (!good_ || !buffer_ || !offset_ || !format || *offset_ >= capacity_) {
      return good_ = false;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = std::vsnprintf(buffer_ + *offset_, capacity_ - *offset_, format, arguments);
    va_end(arguments);
    if (written < 0 || static_cast<std::size_t>(written) >= capacity_ - *offset_) {
      return good_ = false;
    }
    *offset_ += static_cast<std::size_t>(written);
    return true;
  }

  bool begin_object() { return bytes("{", 1); }
  bool begin_object(const char *name) { return key(name) && begin_object(); }
  bool object_element() { return separator() && begin_object(); }
  bool end_object() { return bytes("}", 1); }
  bool begin_array(const char *name) { return key(name) && bytes("[", 1); }
  bool end_array() { return bytes("]", 1); }

  template <typename T> bool field(const char *name, T value) { return key(name) && scalar(value); }

  template <typename T> bool nullable(const char *name, T value, bool valid) {
    return key(name) && (valid ? scalar(value) : append("null"));
  }

  bool nullable_ints(std::initializer_list<NamedInt> fields, std::int32_t unset) {
    for (const auto &field : fields) {
      if (!nullable(field.name, field.value, field.value != unset))
        return false;
    }
    return good();
  }

  bool raw(const char *name, const char *encoded) { return key(name) && append("%s", encoded); }

  // Existing record encoders share the same bounded destination and offset.
  template <typename Encoder, typename... Args> bool write(Encoder encoder, const Args &...args) {
    return good_ = good_ && encoder(buffer_, capacity_, offset_, args...);
  }

  template <typename Encoder, typename... Args>
  bool optional(const char *name, bool valid, Encoder encoder, const Args &...args) {
    return key(name) && (valid ? write(encoder, args...) : append("null"));
  }

  template <typename Encoder, typename... Args> bool record(const char *name, Encoder encoder, const Args &...args) {
    return key(name) && write(encoder, args...);
  }

  template <typename T> bool element(T value) { return separator() && scalar(value); }

  // Strings passed here have already crossed the runtime's UTF-8 validator.
  bool string(const char *value) {
    if (!value)
      return bytes("null", 4);
    if (!bytes("\"", 1))
      return false;
    const char *span = value;
    for (const unsigned char *cursor = reinterpret_cast<const unsigned char *>(value); *cursor; ++cursor) {
      if (*cursor == '"' || *cursor == '\\' || *cursor < 0x20) {
        const char *current = reinterpret_cast<const char *>(cursor);
        if (!bytes(span, static_cast<std::size_t>(current - span)))
          return false;
        if (*cursor < 0x20) {
          append("\\u%04x", static_cast<unsigned>(*cursor));
        } else {
          const char escaped[] = {'\\', static_cast<char>(*cursor)};
          bytes(escaped, sizeof(escaped));
        }
        span = current + 1;
      }
    }
    return bytes(span, std::strlen(span)) && bytes("\"", 1);
  }

private:
  bool bytes(const char *value, std::size_t size) {
    if (!good_ || !buffer_ || !offset_ || *offset_ >= capacity_ || size >= capacity_ - *offset_)
      return good_ = false;
    std::memcpy(buffer_ + *offset_, value, size);
    *offset_ += size;
    buffer_[*offset_] = '\0';
    return true;
  }
  bool separator() {
    if (!good_ || !buffer_ || !offset_)
      return good_ = false;
    const char previous = *offset_ ? buffer_[*offset_ - 1] : '\0';
    return previous == '{' || previous == '[' || previous == ',' || previous == '\0' || bytes(",", 1);
  }
  bool key(const char *name) { return separator() && string(name) && bytes(":", 1); }
  bool scalar(bool value) { return append("%s", value ? "true" : "false"); }
  bool scalar(const char *value) { return string(value); }
  template <typename T> bool scalar(T value) {
    static_assert(std::is_integral<T>::value, "explicit precision required for floats");
    if constexpr (std::is_signed<T>::value) {
      return append("%lld", static_cast<long long>(value));
    } else {
      return append("%llu", static_cast<unsigned long long>(value));
    }
  }

  char *buffer_;
  std::size_t capacity_;
  std::size_t *offset_;
  bool good_ = true;
};
