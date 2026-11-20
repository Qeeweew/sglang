#include "launch.h"
#include "layout.h"
#include "jit.h"
#include "actual_entry.h"
#include "torch.h"

namespace {
template <class... Args>
void launch_peer(native_jit::Kernel &kernel, void *stream, const Args &...args) {
    kernel.launch_persistent<native_jit::BlockSchedule::Batch>(stream, peer_all_reduce::kWorkers,
                                                                args...);
}

native_jit::Kernel &entry() {
    static thread_local native_jit::Kernel kernel(
        "peer_all_reduce_sum", "peer_all_reduce/peer_all_reduce.asc", {"--cce-auto-sync=off"},
        peer_all_reduce::kWorkers);
    return kernel;
}
} // namespace
void prepare_peer_all_reduce() {
    entry().prepare();
}
void launch_peer_all_reduce(void *stream, void *input, void *output, void *local, void *peer,
                            void *table, uint32_t elements) {
    launch_peer(entry(), stream, input, output, local, peer, table, elements);
}

namespace {
native_jit::Kernel &moe_entry(bool gather, bool finalize, uint32_t attention_tp, bool bf16 = false) {
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get({gather, finalize, attention_tp, bf16}, [&] {
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        if (bf16) flags.emplace_back("-DPEER_ROUTED_BF16=1");
        flags.emplace_back("-DPEER_ATTN_TP=" + std::to_string(attention_tp));
        if (gather)
            flags.emplace_back("-DPEER_GATHER=1");
        flags.emplace_back(std::string("-DPEER_FINALIZE=") + (finalize ? "1" : "0"));
        return std::make_unique<native_jit::Kernel>(gather     ? "peer_moe_gather"
                                                    : finalize ? "peer_moe_finalize_return"
                                                               : "peer_moe_return",
                                                    "peer_all_reduce/moe.asc", flags,
                                                    peer_all_reduce::kWorkers);
    });
}
native_jit::Kernel &mhc_entry(uint32_t world, bool normalize, uint32_t attention_tp, bool bf16 = false) {
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get({world, normalize, attention_tp, bf16}, [&] {
        std::vector<std::string> flags{"--cce-auto-sync=off",
                                       "-DMHC_BRANCHES=4",
                                       "-DMHC_WIDTH=5120",
                                       "-DMHC_WORKERS=16",
                                       "-DMHC_COLUMN_TILE=2560",
                                       "-DMHC_OUTPUT_COLUMNS=1",
                                       world == 2 ? "-DMHC_PEER_ATTN=1" : "-DMHC_PEER_MOE=1"};
        flags.emplace_back("-DPEER_ATTN_TP=" + std::to_string(attention_tp));
        if (bf16) flags.emplace_back("-DPEER_ROUTED_BF16=1");
        if (normalize)
            flags.emplace_back("-DMHC_POST_COMBINE=1");
        return std::make_unique<native_jit::Kernel>(
            world == 2 ? "peer_attention_mhc" : "peer_moe_mhc", "mhc/state.asc", flags,
            peer_all_reduce::kWorkers);
    });
}
} // namespace
void prepare_peer_moe(uint32_t attention_tp) {
    moe_entry(true, false, attention_tp).prepare();
    moe_entry(false, false, attention_tp).prepare();
    moe_entry(false, true, attention_tp).prepare();
    moe_entry(false, true, attention_tp, true).prepare();
}
void prepare_peer_mhc(uint32_t world, uint32_t attention_tp) {
    mhc_entry(world, true, attention_tp).prepare();
    if (world == 8) {
        mhc_entry(world, false, attention_tp).prepare();
        mhc_entry(world, false, attention_tp, true).prepare();
        mhc_entry(world, true, attention_tp, true).prepare();
    }
}
void launch_peer_moe(void *stream, void *input, void *inverse, void *ids, void *weights,
                     void *shared, void *output, void *local, void *table, uint32_t rank,
                     uint32_t rows, uint32_t topk, bool gather, bool finalize,
                     uint32_t attention_tp, bool routed_bf16) {
    launch_peer(moe_entry(gather, finalize, attention_tp, routed_bf16), stream, input, inverse,
                ids, weights, shared, output, local, table, rank, rows, topk);
}
void launch_peer_mhc(void *stream, void *input, void *residual, void *pre, void *post, void *comb,
                     void *weight, void *updated, void *output, void *local, void *table,
                     void *inverse, void *ids, void *weights, void *shared, uint32_t rank,
                     uint32_t rows, uint32_t topk, uint32_t world, bool normalize,
                     uint32_t attention_tp, bool routed_bf16) {
    auto &kernel = mhc_entry(world, normalize, attention_tp, routed_bf16);
    if (normalize)
        launch_peer(kernel, stream, input, residual, pre, post, comb, weight, updated, output,
                    uint64_t(rows), local, table, inverse, ids, weights, shared, rank, topk);
    else
        launch_peer(kernel, stream, input, residual, post, comb, updated, uint64_t(rows), local,
                    table, inverse, ids, weights, shared, rank, topk);
}

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_peer_all_reduce() -> ()", [] {
        prepare_peer_all_reduce();
        prepare_peer_mhc(2, 2);
        for (uint32_t tp : {1, 2}) {
            prepare_peer_moe(tp);
            prepare_peer_mhc(8, tp);
        }
    });
}
