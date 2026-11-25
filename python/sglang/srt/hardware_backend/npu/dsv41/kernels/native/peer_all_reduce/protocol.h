#pragma once
#include <common/device.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <peer_all_reduce/layout.h>

namespace peer_protocol {
namespace p = peer_all_reduce;
__aicore__ inline __gm__ uint64_t *signal(GM_ADDR base, uint32_t kind, uint32_t channel) {
    return reinterpret_cast<__gm__ uint64_t *>(base +
                                               (kind * p::kMaxWorkers + channel) * p::kSignalBytes);
}
__aicore__ inline __gm__ uint64_t *global_signal(GM_ADDR base, uint32_t kind) {
    return reinterpret_cast<__gm__ uint64_t *>(base + (5 * p::kMaxWorkers + kind) * p::kSignalBytes);
}
__aicore__ inline void publish(__gm__ uint64_t *address, uint64_t epoch) {
    *reinterpret_cast<volatile __gm__ uint64_t *>(address) = epoch;
    __asc_aicore::asc_dcci_single(address);
}
__aicore__ inline uint64_t observe(__gm__ uint64_t *address) {
    __asc_aicore::asc_dcci_single(address);
    return *reinterpret_cast<volatile __gm__ uint64_t *>(address);
}
__aicore__ inline void await(__gm__ uint64_t *address, uint64_t epoch) {
    const int64_t start = AscendC::GetSystemCycle();
    while (observe(address) < epoch) {
        if (AscendC::GetSystemCycle() - start > 500000000LL)
            AscendC::Trap();
        const int64_t retry = AscendC::GetSystemCycle() + 8;
        while (AscendC::GetSystemCycle() < retry) {
        }
    }
}
__aicore__ inline void aggregate(GM_ADDR local, uint32_t kind, uint64_t epoch,
                                 uint32_t workers = p::kWorkers) {
    if (AscendC::GetBlockIdx() == 0) {
        for (uint32_t c = 0; c < workers; ++c)
            await(signal(local, kind, c), epoch);
        publish(global_signal(local, kind == 2 ? 0 : kind == 3 ? 2 : 1), epoch);
    }
}
__aicore__ inline uint64_t begin(GM_ADDR local, GM_ADDR table, uint32_t world,
                                GM_ADDR peer = nullptr, uint32_t workers = p::kWorkers) {
    // Read the completed global epoch so all collectives on this handle share one epoch.
    const uint64_t epoch = observe(global_signal(local, 0)) + 1;
    auto peers = reinterpret_cast<__gm__ uint64_t *>(table);
    // Only the leader polls remote free credits. Letting every AIV spin on
    // the same remote cache line floods HCCS and can starve a slower peer's
    // payload DMA. The remaining workers poll one local start credit.
    if (AscendC::GetBlockIdx() == 0) {
        if (peer) {
            await(global_signal(local, 0), epoch - 1);
            await(global_signal(peer, 0), epoch - 1);
        } else {
            for (uint32_t rank = 0; rank < world; ++rank)
                await(global_signal(reinterpret_cast<GM_ADDR>(peers[rank]), 0), epoch - 1);
        }
        for (uint32_t c = 0; c < workers; ++c)
            publish(signal(local, 4, c), epoch);
    }
    await(signal(local, 4, AscendC::GetBlockIdx()), epoch);
    native_device::pipe_fence<PIPE_S, PIPE_MTE2, EVENT_ID0>();
    return epoch;
}
__aicore__ inline void finish(GM_ADDR local, uint64_t epoch, uint32_t workers = p::kWorkers) {
    native_device::pipe_fence<PIPE_MTE2, PIPE_S, EVENT_ID0>();
    native_device::pipe_fence<PIPE_MTE3, PIPE_S, EVENT_ID0>();
    native_device::pipe_fence<PIPE_V, PIPE_S, EVENT_ID0>();
    publish(signal(local, 2, AscendC::GetBlockIdx()), epoch);
    aggregate(local, 2, epoch, workers);
}
__aicore__ inline void loaded() {
    native_device::pipe_fence<PIPE_MTE2, PIPE_V, EVENT_ID0>();
}
__aicore__ inline void stored() {
    native_device::pipe_fence<PIPE_MTE3, PIPE_MTE2, EVENT_ID0>();
}
__aicore__ inline void vector_store() {
    native_device::pipe_fence<PIPE_V, PIPE_MTE3, EVENT_ID0>();
}
} // namespace peer_protocol
