#pragma once
#include <cstdint>
namespace page_metadata {
// One arch2201 32-byte INT32 output line belongs to a request group.
inline constexpr uint32_t kRequestsPerGroup = 32 / sizeof(int32_t);
}

extern "C" void dsv41_draft_page_metadata_launch(
    void *stream, uint32_t blocks, uint8_t *req_to_token, uint8_t *full_to_swa, uint8_t *requests,
    uint8_t *prefix, uint8_t *full_slots, uint8_t *cu, uint8_t *lengths, uint8_t *table,
    uint8_t *slots, uint8_t *routed_rows, int64_t batch, int64_t block, int64_t request_rows,
    int64_t context, int64_t columns, int64_t mapping_size, int64_t page_rows, int64_t window_rows);
