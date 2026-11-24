#pragma once
#include <qli/state.h>

namespace qli {
inline constexpr bool kExplicitPositions = QLI_EXPLICIT_POSITIONS;

__aicore__ inline int64_t visible_at(int32_t position, int64_t length) {
    return max(int64_t(0), min(length, (int64_t(position) + 1) / kRatio));
}

struct Task {
    uint32_t batch, row, queries, split, key_begin, key_end, request_rows, query_offset;
    int64_t original_keys;
    __aicore__ int valid_keys(uint32_t local, uintptr_t positions) const {
        if constexpr (kExplicitPositions)
            return visible_at(gm_pointer<int32_t>(positions)[row + local], original_keys / kRatio);
        const int64_t visible = (original_keys - request_rows + query_offset + local + 1) / kRatio;
        return max(int64_t(0), min(original_keys / kRatio, visible));
    }
};
class TaskCursor {
    const Sources &args_;
    uint32_t batch_ = 0, begin_ = 0, end_ = 0;
    uint64_t offset_ = 0;

  public:
    __aicore__ explicit TaskCursor(const Sources &args) : args_(args) {}
    __aicore__ bool locate(uint64_t ordinal, Task &task) {
        auto cu = gm_pointer<int32_t>(args_.cu);
        auto lengths = gm_pointer<int32_t>(args_.lengths);
        for (; batch_ < args_.batch; ++batch_) {
            begin_ = cu[batch_];
            end_ = cu[batch_ + 1];
            const uint64_t groups = (end_ - begin_ + kQueries - 1) / kQueries;
            const uint64_t tasks = groups * kKeySplits;
            if (ordinal >= offset_ + tasks) {
                offset_ += tasks;
                continue;
            }
            const uint64_t relative = ordinal - offset_;
            const uint32_t group = relative / kKeySplits;
            const uint32_t split = relative % kKeySplits;
            const uint32_t first = group * kQueries;
            const uint32_t queries = min(uint32_t(kQueries), end_ - begin_ - first);
            const int64_t length = lengths[batch_];
            const int64_t original = length * kRatio + gm_pointer<int32_t>(args_.residual)[batch_];
            int64_t visible;
            if constexpr (kExplicitPositions) {
                const auto positions = gm_pointer<int32_t>(args_.positions) + begin_ + first;
                visible = 0;
                // Valid positions are increasing within a request. Ignore trailing padding.
                for (uint32_t i = queries; i > 0; --i) {
                    const int32_t position = positions[i - 1];
                    if (position >= 0) {
                        visible = visible_at(position, length);
                        break;
                    }
                }
            } else {
                visible = max(
                    int64_t(0), min(length, (original - (end_ - begin_) + first + queries) / kRatio));
            }
            const uint64_t chunks = (visible + kKeyTileCapacity - 1) / kKeyTileCapacity;
            task = {batch_,
                    begin_ + first,
                    queries,
                    split,
                    uint32_t(chunks * split / kKeySplits * kKeyTileCapacity),
                    uint32_t(min(uint64_t(visible),
                                 chunks * (split + 1) / kKeySplits * kKeyTileCapacity)),
                    end_ - begin_,
                    first,
                    original};
            return true;
        }
        return false;
    }
};
} // namespace qli
