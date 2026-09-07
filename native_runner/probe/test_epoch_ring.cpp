#include "epoch_ring.h"
#include <cassert>

struct Record {
    std::uint64_t sequence = 0, generation = 0, state_epoch = 0;
    std::int32_t tick = -1;
    int payload = 0;
};

int main() {
    EpochRing<Record, 3> ring;
    Record record, read;
    assert(!ring.read(0, 0, 0, 0, read));
    ring.begin(4, 5);
    assert(ring.publish(record) == 0);
    record.generation = 4;
    record.state_epoch = 5;
    record.tick = 9;
    for (int index = 1; index <= 5; ++index) {
        record.payload = index * 13;
        assert(ring.publish(record) == static_cast<std::uint64_t>(index));
    }
    auto state = ring.snapshot(4, 5);
    assert(state.identity && state.first == 1 && state.oldest == 3);
    assert(state.next == 6 && state.overflow == 2 && state.rejected == 0);
    assert(!ring.read(2, 4, 5, 9, read));
    assert(!ring.read(3, 4, 5, 8, read));
    assert(!ring.read(3, 4, 6, 9, read));
    assert(ring.read(3, 4, 5, 9, read) && read.payload == 39);
    ring.reject(4, 5);
    ring.reject(4, 6);
    assert(ring.snapshot(4, 5).rejected == 1);
    assert(!ring.snapshot(4, 6).identity && ring.snapshot(4, 6).rejected == 0);
    ring.begin(4, 6);
    assert(ring.publish(record) == 0);
    state = ring.snapshot(4, 6);
    assert(state.first == 6 && state.oldest == 6 && state.next == 6);
    assert(state.overflow == 0 && state.rejected == 0);
    record.state_epoch = 6;
    assert(ring.publish(record) == 6);
    assert(ring.read(6, 4, 6, 9, read) && read.payload == 65);
}
