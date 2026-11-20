#include "torch.h"
#include "jit.h"
#include "launch.h"
#include "workspace.h"
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

namespace native_host {
namespace {
void quantize_weight(const at::Tensor &weight, const at::Tensor &out, const at::Tensor &scales) {
    TORCH_CHECK_VALUE(weight.dim() == 2, "weight quantization requires [N,K]");
    auto rows = weight.size(0), k = weight.size(1);
    TORCH_CHECK_VALUE(rows > 0 && k > 0 && k <= INT32_MAX && k % w8a16::kGroup == 0,
                      "weight quantization requires positive N and K256 alignment");
    auto device = weight.device();
    check(weight, at::kBFloat16, device);
    check(out, at::kChar, device, {rows, k});
    check(scales, at::kHalf, device, {k / w8a16::kGroup, rows});
    disjoint({out, scales}, {weight});
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    for (const auto &tensor : {weight, out, scales}) record(tensor, stream);
    w8a16::QuantizeSources sources{uintptr_t(ptr(weight)), uintptr_t(ptr(out)),
                                   uintptr_t(ptr(scales)), rows};
    w8a16_quantize_launch(stream.stream(), sources, int(k), core_count(device.index()));
}

struct Problem {
    int64_t rows, k, n, batches;
    bool bf16;
};

struct TilePlan {
    int64_t bm, bn, bk;
};

Problem check_inputs(const at::Tensor &x, const at::Tensor &w, const at::Tensor &s) {
    TORCH_CHECK_VALUE(x.dim() == 2 || x.dim() == 3, "W8A16 input requires [M,K] or [M,B,K]");
    const auto m = x.size(0), k = x.size(-1), batches = x.dim() == 3 ? x.size(1) : 1;
    TORCH_CHECK_VALUE(w.dim() == 3 && w.size(2) == w8a16::kC0 && batches > 0 &&
                          w.size(0) % batches == 0,
                      "W8A16 repacked weights require [B*N/16,K,16]");
    const auto n = w.size(0) / batches * w8a16::kC0;
    TORCH_CHECK_VALUE(k > 0 && k <= UINT16_MAX && k % w8a16::kGroup == 0 &&
                          n > 0 && n <= INT32_MAX && n % 128 == 0 &&
                          batches * k <= UINT16_MAX && uint64_t(batches) * n <= UINT32_MAX / 2,
                      "W8A16 requires G256 K, N128, uint16 ND2NZ input stride and uint32 output byte stride");
    auto device = x.device();
    TORCH_CHECK_VALUE(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                      "W8A16 requires BF16/FP16 activation and output");
    check(x, x.scalar_type(), device);
    check(w, at::kChar, device, {batches * n / w8a16::kC0, k, w8a16::kC0});
    check(s, at::kHalf, device, {batches * n / w8a16::kC0, k / w8a16::kGroup, w8a16::kC0});
    return {m, k, n, batches, x.scalar_type() == at::kBFloat16};
}

TilePlan select_plan(const Problem &problem, int64_t workers, std::optional<int64_t> block_m,
                     std::optional<int64_t> block_n, std::optional<int64_t> block_k) {
    const auto [rows, k, n, batches, bf16] = problem;
    const int64_t default_m = rows <= 16 ? 16 : rows <= 64 ? 64 : 256;
    if (!block_m && !block_n && !block_k && rows > 0 && rows <= 256) {
        const auto bm = rows > 64 && rows <= 128 ? 128 : default_m;
        constexpr int64_t panel_issue_cost = 128 * 256;
        TilePlan best{bm, 64, 256};
        int64_t best_cost = INT64_MAX;
        for (int64_t bn : {256, 128, 64}) {
            for (int64_t bk : {512, 256}) {
                if (n % bn || k % bk || !w8a16::Storage(bm, bn, bk, bf16).fits())
                    continue;
                const auto jobs = batches * (n / bn);
                const auto waves = (jobs + workers - 1) / workers;
                const auto cost = waves * (k / bk) * (bn * bk + panel_issue_cost);
                if (cost < best_cost) {
                    best = {bm, bn, bk};
                    best_cost = cost;
                }
            }
        }
        return best;
    }

    const auto bm = block_m.value_or(default_m), bk = block_k.value_or(256);
    if (block_n)
        return {bm, *block_n, bk};
    TORCH_CHECK_VALUE(bm > 0, "invalid W8A16 tile plan");
    const auto row_tiles = std::max<int64_t>(1, rows / bm + (rows % bm != 0));
    for (int64_t bn : {256, 128}) {
        if ((bn < 256 || (bm <= 128 && bk <= 256)) && n % bn == 0 &&
            row_tiles * batches * (n / bn) >= workers)
            return {bm, bn, bk};
    }
    return {bm, 64, bk};
}

void check_plan(const Problem &problem, const TilePlan &plan) {
    const auto [bm, bn, bk] = plan;
    TORCH_CHECK_VALUE((bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256) &&
                          (bn == 64 || bn == 128 || bn == 256) && problem.n % bn == 0 &&
                          (bk == 128 || bk == 256 || bk == 512) && problem.k % bk == 0,
                      "invalid W8A16 tile plan");
    TORCH_CHECK_VALUE(w8a16::Storage(bm, bn, bk, problem.bf16).fits(),
                      "W8A16 tile exceeds arch2201 resource budget");
}

int64_t worker_count(const at::Device &device) {
    const auto workers = core_count(device.index(), true);
    TORCH_CHECK_VALUE(core_count(device.index()) == w8a16::kVectorActors * workers,
                      "W8A16 requires two AIV per AIC");
    return workers;
}

void launch_gemm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &s,
                 const at::Tensor &out, const std::optional<at::Tensor> &workspace,
                 const Problem &problem,
                 const TilePlan &plan, int64_t workers) {
    const auto [m, k, n, batches, bf16] = problem;
    const auto [bm, bn, bk] = plan;
    const auto device = x.device();
    TORCH_CHECK_VALUE(out.scalar_type() == x.scalar_type() || out.scalar_type() == at::kBFloat16,
                      "W8A16 output must match the activation or be BF16");
    check(out, out.scalar_type(), device);
    TORCH_CHECK_VALUE(out.sizes() == (x.dim() == 3 ? at::IntArrayRef({m, batches, n})
                                                : at::IntArrayRef({m, n})),
                      "W8A16 output shape differs");
    for (const auto &tensor : {x, w, s, out})
        TORCH_CHECK_VALUE(uintptr_t(ptr(tensor)) % (w8a16::kC0 * sizeof(at::Half)) == 0,
                          "W8A16 GM pointers require 32-byte alignment");
    const auto active = std::min(workers, w8a16::jobs(m, n, batches, bm, bn));
    const auto count = active * w8a16::kSlots * bn * bk;
    if (workspace) {
        check(*workspace, x.scalar_type(), device);
        TORCH_CHECK_VALUE(workspace->numel() >= count, "W8A16 workspace is too small");
        TORCH_CHECK_VALUE(uintptr_t(ptr(workspace)) % (w8a16::kC0 * sizeof(at::Half)) == 0,
                          "W8A16 workspace requires 32-byte alignment");
    }
    disjoint({out, workspace}, {x, w, s});
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    const auto acl_stream = stream.stream();
    const LaunchWorkspace scratch(workspace, count * x.element_size(), acl_stream);
    for (const auto &t : {x, w, s, out}) record(t, stream);
    track(workspace, stream);
    w8a16::Arguments a{uintptr_t(ptr(x)), uintptr_t(ptr(w)), uintptr_t(ptr(s)),
                       uintptr_t(ptr(out)), uintptr_t(scratch.data()), m,
                       int(k), int(n), int(batches), int(bm), int(bn), int(bk), int(workers), bf16,
                       out.scalar_type() == at::kBFloat16};
    w8a16_launch(acl_stream, a);
}

void gemm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &s,
          const at::Tensor &out, const std::optional<at::Tensor> &workspace,
          int64_t bm, int64_t bn, int64_t bk) {
    const auto problem = check_inputs(x, w, s);
    const TilePlan plan{bm, bn, bk};
    check_plan(problem, plan);
    c10_npu::NPUGuard guard(x.device());
    launch_gemm(x, w, s, out, workspace, problem, plan, worker_count(x.device()));
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t>
prepare_gemm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &s,
             const std::optional<at::Tensor> &out, const std::optional<at::Tensor> &workspace,
             std::optional<int64_t> block_m, std::optional<int64_t> block_n,
             std::optional<int64_t> block_k) {
    const auto problem = check_inputs(x, w, s);
    c10_npu::NPUGuard guard(x.device());
    TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None,
                "bind W8A16 and warm up before Graph capture");
    const auto workers = worker_count(x.device());
    const auto plan = select_plan(problem, workers, block_m, block_n, block_k);
    check_plan(problem, plan);
    auto output_shape = x.sizes().vec();
    output_shape.back() = problem.n;
    auto output = out ? *out : at::empty(output_shape, x.options());
    native_jit::PrepareOnlyScope prepare;
    launch_gemm(x, w, s, output, workspace, problem, plan, workers);
    return {output, plan.bm, plan.bn, plan.bk};
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("quantize_g256_weight(Tensor weight, Tensor(a!) out, Tensor(b!) scales) -> ()",
          &quantize_weight);
    m.def("w8a16(Tensor x, Tensor weight, Tensor scales, Tensor(a!) out, Tensor(b!)? workspace, "
          "int bm, int bn, int bk) -> ()", &gemm);
    m.def("prepare_w8a16(Tensor x, Tensor weight, Tensor scales, Tensor(a)? out=None, "
          "Tensor(b)? workspace=None, int? bm=None, int? bn=None, int? bk=None) "
          "-> (Tensor(a), int, int, int)", &prepare_gemm);
}
} // namespace native_host
