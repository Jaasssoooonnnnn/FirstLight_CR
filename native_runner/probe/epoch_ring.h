#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Fixed storage for hooks that publish pointer-free records. A lifecycle reset
// drains publishers, then changes identity and the retained sequence boundary.
// The native worker serializes writes; readers verify publication on both sides
// of the record copy, including generation/epoch and observation tick.
template <typename Record, std::size_t Capacity> class EpochRing {
public:
  static_assert(Capacity > 0, "an event ring requires storage");
  struct Slot {
    std::atomic<std::uint64_t> published_sequence{0};
    Record record;
  };
  struct Snapshot {
    bool identity;
    std::uint64_t next, first, oldest, overflow, rejected;
  };

  std::atomic<std::uint64_t> next_sequence{1};

  void begin(std::uint64_t generation, std::uint64_t state_epoch) {
    resetting_.store(true, std::memory_order_release);
    while (publishers_.load(std::memory_order_acquire) != 0) {
    }
    while (epoch_lock_.test_and_set(std::memory_order_acquire)) {
    }
    generation_.store(0, std::memory_order_release);
    state_epoch_.store(0, std::memory_order_release);
    first_.store(next_sequence.load(std::memory_order_acquire), std::memory_order_release);
    overflow_.store(0, std::memory_order_release);
    rejected_.store(0, std::memory_order_release);
    state_epoch_.store(state_epoch, std::memory_order_release);
    generation_.store(generation, std::memory_order_release);
    epoch_lock_.clear(std::memory_order_release);
    resetting_.store(false, std::memory_order_release);
  }

  void reject(std::uint64_t generation, std::uint64_t state_epoch) {
    if (enter(generation, state_epoch)) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      leave();
    }
  }

  std::uint64_t publish(Record record) {
    if (!enter(record.generation, record.state_epoch)) {
      return 0;
    }
    const auto sequence = next_sequence.fetch_add(1, std::memory_order_acq_rel);
    record.sequence = sequence;
    if (sequence >= first_.load(std::memory_order_acquire) + Capacity) {
      overflow_.fetch_add(1, std::memory_order_relaxed);
    }
    Slot &slot = slots_[(sequence - 1) % Capacity];
    slot.published_sequence.store(0, std::memory_order_relaxed);
    slot.record = record;
    slot.published_sequence.store(sequence, std::memory_order_release);
    leave();
    return sequence;
  }

  Snapshot snapshot(std::uint64_t generation, std::uint64_t state_epoch) const {
    const bool identity = matches(generation, state_epoch);
    const auto next = next_sequence.load(std::memory_order_acquire);
    const auto first = identity ? first_.load(std::memory_order_acquire) : next;
    return {identity,
            next,
            first,
            next > first + Capacity ? next - Capacity : first,
            identity ? overflow_.load(std::memory_order_acquire) : 0,
            identity ? rejected_.load(std::memory_order_acquire) : 0};
  }

  bool read(std::uint64_t sequence, std::uint64_t generation, std::uint64_t state_epoch, std::int32_t tick,
            Record &record) const {
    if (sequence == 0)
      return false;
    const Slot &slot = slots_[(sequence - 1) % Capacity];
    if (slot.published_sequence.load(std::memory_order_acquire) != sequence) {
      return false;
    }
    record = slot.record;
    return record.sequence == sequence && record.generation == generation && record.state_epoch == state_epoch &&
           record.tick <= tick && slot.published_sequence.load(std::memory_order_acquire) == sequence;
  }

private:
  bool matches(std::uint64_t generation, std::uint64_t state_epoch) const {
    return generation_.load(std::memory_order_acquire) == generation &&
           state_epoch_.load(std::memory_order_acquire) == state_epoch;
  }

  bool enter(std::uint64_t generation, std::uint64_t state_epoch) {
    if (resetting_.load(std::memory_order_acquire)) {
      return false;
    }
    publishers_.fetch_add(1, std::memory_order_acq_rel);
    if (!resetting_.load(std::memory_order_acquire) && matches(generation, state_epoch)) {
      return true;
    }
    leave();
    return false;
  }

  void leave() { publishers_.fetch_sub(1, std::memory_order_acq_rel); }

  Slot slots_[Capacity];
  std::atomic<std::uint64_t> generation_{0}, state_epoch_{0};
  std::atomic<std::uint64_t> first_{1}, overflow_{0}, rejected_{0};
  std::atomic<bool> resetting_{false};
  std::atomic<std::uint32_t> publishers_{0};
  std::atomic_flag epoch_lock_ = ATOMIC_FLAG_INIT;
};
