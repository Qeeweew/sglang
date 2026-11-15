#pragma once
#include <cstdint>

// Device-visible expert-major route table, shared by routing and grouped GEMM.
// Header is followed by active_count Entries. It carries no synchronization;
// the publishing operator/stream owns visibility and lifetime.
namespace grouped_metadata {
struct Header {
    int32_t active_count, total_rows, status, version;
};
struct Entry {
    int32_t expert, begin, rows, reserved;
};
inline constexpr uint32_t kWords = sizeof(Entry) / sizeof(int32_t);
inline constexpr int32_t kVersion = 1;
enum HeaderWord : uint32_t { kActiveCount, kTotalRows, kStatus, kVersionWord };
enum EntryWord : uint32_t { kExpert, kBegin, kRowCount, kReserved };
static_assert(sizeof(Header) == sizeof(Entry));
static_assert(kWords == 4);
} // namespace grouped_metadata
