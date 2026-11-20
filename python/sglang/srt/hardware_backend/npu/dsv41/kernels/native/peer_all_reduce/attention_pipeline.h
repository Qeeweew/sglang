#pragma once
#include <common/device.h>
#include <peer_all_reduce/protocol.h>

namespace peer_attention {
namespace p = peer_all_reduce;
namespace s = peer_protocol;
__aicore__ inline uint64_t prepare(GM_ADDR input, GM_ADDR local, GM_ADDR table, uint32_t rank,
                                   uint32_t rows) {
    const uint64_t epoch = s::begin(local, table, 2);
    auto raw = reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(0));
    auto x = reinterpret_cast<__gm__ bfloat16_t *>(input);
    auto stage = reinterpret_cast<__gm__ bfloat16_t *>(local + p::kControlBytes);
    for (uint32_t offset = AscendC::GetBlockIdx() * 5120; offset < rows * 5120;
         offset += p::kWorkers * 5120) {
        const uint32_t n = 5120;
        asc_copy_gm2ub(raw, x + offset, n * 2);
        native_device::pipe_fence<PIPE_MTE2, PIPE_MTE3, EVENT_ID0>();
        asc_copy_ub2gm(stage + offset, raw, n * 2);
        s::stored();
    }
    native_device::pipe_fence<PIPE_MTE2, PIPE_S, EVENT_ID0>();
    s::publish(s::signal(local, 1, AscendC::GetBlockIdx()), epoch);
    auto peer = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t *>(table)[1 - rank]);
    s::await(s::signal(peer, 1, AscendC::GetBlockIdx()), epoch);
    native_device::pipe_fence<PIPE_S, PIPE_MTE2, EVENT_ID0>();
    return epoch;
}
} // namespace peer_attention
