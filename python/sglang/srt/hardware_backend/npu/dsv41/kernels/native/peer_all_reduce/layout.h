#pragma once
#include <cstdint>

namespace peer_all_reduce {
// Every worker participates even for short messages. Changing the message
// length must never leave a channel's device epoch behind the other channels.
constexpr uint32_t kWorkers = 16;
constexpr uint32_t kMoeReturnWorkers = 40;
constexpr uint32_t kMaxWorkers = kMoeReturnWorkers;
constexpr uint32_t kSignalBytes = 512;
constexpr uint32_t kControlBytes = (5 * kMaxWorkers + 3) * kSignalBytes;
constexpr uint32_t kMaxElements = 128 * 5120;
constexpr uint32_t kAllocationBytes = 8 * 1024 * 1024;
constexpr uint32_t kTileElements = 2048;
constexpr uint32_t kMoeTile = 5120; // One complete token per DMA/vector packet.
constexpr uint32_t kMoeMaxGlobalRows = 1024; // 128 * attention_tp local rows.
constexpr uint32_t kMoeReturnOffset = kControlBytes + kMoeMaxGlobalRows * kMoeTile * 2;
constexpr uint32_t kMoeAllocationBytes = 16 * 1024 * 1024;
static_assert(kControlBytes + kMaxElements * 2 <= kAllocationBytes);
static_assert(kMoeReturnOffset + 256 * kMoeTile * 2 <= kMoeAllocationBytes);
} // namespace peer_all_reduce
