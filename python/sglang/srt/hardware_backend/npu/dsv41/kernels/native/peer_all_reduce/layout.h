#pragma once
#include <cstdint>

namespace peer_all_reduce {
// Every worker participates even for short messages. Changing the message
// length must never leave a channel's device epoch behind the other channels.
constexpr uint32_t kWorkers = 16;
constexpr uint32_t kSignalBytes = 512;
constexpr uint32_t kControlBytes = (5 * kWorkers + 3) * kSignalBytes;
constexpr uint32_t kMaxElements = 128 * 5120;
constexpr uint32_t kAllocationBytes = 8 * 1024 * 1024;
constexpr uint32_t kTileElements = 2048;
constexpr uint32_t kMoeTile = 5120; // One complete token per DMA/vector packet.
constexpr uint32_t kMoeReturnOffset = kControlBytes + 4 * kMaxElements * 2;
constexpr uint32_t kMoeAllocationBytes = 8 * 1024 * 1024;
static_assert(kControlBytes + kMaxElements * 2 <= kAllocationBytes);
static_assert(kMoeReturnOffset + kMaxElements <= kMoeAllocationBytes);
} // namespace peer_all_reduce
