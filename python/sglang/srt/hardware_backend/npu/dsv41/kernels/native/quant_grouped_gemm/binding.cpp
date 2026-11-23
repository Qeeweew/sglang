#include "torch.h"
#include "tiling.h"
#include <common/grouped_metadata.h>
#include <common/l2_bypass.h>
#include "launch.h"
#include "jit.h"
#include "workspace.h"

namespace native_host {
namespace {
using namespace quant_gmm;
Geometry checked_geometry(int64_t k, int64_t n, int64_t bm) {
    TORCH_CHECK_VALUE(k > 0 && k <= UINT16_MAX && n > 0 && n <= INT32_MAX,
                      "positive K <= UINT16_MAX (ND2NZ source stride) and INT32 N required");
    const auto g = geometry(k, n, bm);
    TORCH_CHECK_VALUE(g[K], "K group32 / N block64 geometry and BM 16/128/256 required");
    return g;
}
std::vector<Geometry> geometries(int64_t k, int64_t n, int64_t bm = 0) {
    std::vector<Geometry> result;
    for (int v :
         {Storage::kDecodeTileRows, Storage::kPrefillTileRows, Storage::kWidePrefillTileRows}) {
        if (bm && bm != v)
            continue;
        result.push_back(checked_geometry(k, n, v));
    }
    TORCH_CHECK_VALUE(!result.empty(), "BM must be 0 (automatic), 16, 128 or 256");
    return result;
}
std::vector<int64_t> values(const Geometry &g) {
    return {g.values, g.values + Fields};
}
std::vector<int64_t> geometry_config(int64_t k, int64_t n, int64_t bm) {
    return values(geometries(k, n, bm)[0]);
}
int64_t select(const std::vector<Geometry> &gs, int64_t rows, double mean, int64_t bound,
               int64_t cores) {
    TORCH_CHECK_VALUE(std::isfinite(mean) && mean >= 0 && bound >= 0 && bound <= rows && cores > 0,
                      "valid host-known expert row bounds required");
    if (gs.size() == 1)
        return 0;
    auto index = [&](int bm) {
        for (size_t i = 0; i < gs.size(); ++i)
            if (gs[i][BM] == bm)
                return int64_t(i);
        TORCH_CHECK(false, "complete native grouped GEMM tiling required");
    };
    mean = std::min(mean, double(bound));
    if (mean < Storage::kPrefillTileRows) {
        const auto &compact = gs[index(Storage::kPrefillTileRows)];
        constexpr int kFullAccumulatorElements = arch2201::kL0CBytes / sizeof(float);
        constexpr int kMinimumCompactMeanRows = 2 * Storage::kFractalRows;
        if (bound > Storage::kDecodeTileRows &&
            (compact[BM] * compact[BN] == kFullAccumulatorElements ||
             mean >= kMinimumCompactMeanRows))
            return index(Storage::kPrefillTileRows);
        return index(Storage::kDecodeTileRows);
    }
    const auto compact_idx = index(Storage::kPrefillTileRows);
    const auto wide_idx = index(Storage::kWidePrefillTileRows);
    const auto &compact = gs[compact_idx];
    const auto &wide = gs[wide_idx];
    // Prefer BM=128 when it already fills the same L0C footprint as BM=256:
    // BM=256/BN=128 doubles ND2NZ activation DMA and saturates AIC MTE2.
    if (compact[BM] * compact[BN] >= wide[BM] * wide[BN])
        return compact_idx;
    return wide_idx;
}
int64_t select_config(int64_t k, int64_t n, int64_t rows, double mean, int64_t bound,
                      int64_t cores) {
    const auto gs = geometries(k, n);
    return gs[select(gs, rows, mean, bound, cores)][BM];
}
int64_t gemm_cores(int64_t device) {
    thread_local std::unordered_map<int64_t, int64_t> validated_devices;
    const auto found = validated_devices.find(device);
    if (found != validated_devices.end())
        return found->second;
    c10_npu::NPUGuard guard(at::Device(c10::DeviceType::PrivateUse1, device));
    const auto cores = core_count(device, true);
    const char *soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::string(soc).find("Ascend910B") == 0 &&
                    core_count(device) == kVectorLanesPerCube * cores,
                "native quantized GEMM requires Ascend 910B mixed cores");
    return validated_devices.emplace(device, cores).first->second;
}
void run(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &scales,
         const at::Tensor &routing, const at::Tensor &out,
         const std::optional<at::Tensor> &workspace, int64_t bm) {
    TORCH_CHECK_VALUE(x.dim() == 2 && out.dim() == 2 && packed.dim() == 2 && packed.size(0) > 0 &&
                          packed.size(0) <= INT32_MAX,
                      "matrix inputs and positive expert count required");
    const auto g = checked_geometry(x.size(1), out.size(1), bm ? bm : Storage::kDecodeTileRows);
    const auto rows = x.size(0), experts = packed.size(0);
    const auto device = x.device();
    check(x, at::kHalf, device, {rows, g[K]});
    check(packed, at::kInt, device, {experts, g[K] * g[N] / Storage::kCodesPerInt32});
    check(scales, at::kByte, device, {experts, g[K] * g[N] / Storage::kScaleGroup});
    check(out, at::kHalf, device, {rows, g[N]});
    check(routing, at::kInt, device);
    TORCH_CHECK_VALUE(routing.dim() == 1 &&
                          routing.numel() >= grouped_metadata::kWords * (1 + experts),
                      "compact metadata capacity must cover all experts");
    c10_npu::NPUGuard guard(device);
    const auto cores = gemm_cores(device.index());
    const int64_t count = cores * Storage::kSlots * g[STEP_K] * g[BN];
    if (workspace) {
        check(*workspace, at::kHalf, device);
        TORCH_CHECK_VALUE(workspace->numel() >= count, "workspace needs ", count, " FP16 elements");
    }
    disjoint({out, workspace}, {x, packed, scales, routing});
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    const auto acl_stream = stream.stream();
    const LaunchWorkspace scratch(workspace, count * sizeof(at::Half), acl_stream);
    for (const auto &t : {x, packed, scales, routing, out})
        record(t, stream);
    track(workspace, stream);
    const auto alias = l2_bypass_alias(device.index());
    const auto output_alias = g[BM] >= Storage::kPrefillTileRows ? alias : 0;
    Arguments args{};
    args.stream = acl_stream;
    args.cores = cores;
    args.experts = experts;
    args.activation = ptr(x);
    args.packed = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(packed.data_ptr()) + alias);
    args.scales = ptr(scales);
    args.routing = ptr(routing);
    args.output =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(out.data_ptr()) + output_alias);
    args.workspace = scratch.data();
    args.config = {int(g[K]), int(g[N])};
    args.tile_rows = int(g[BM]);
    args.tile_columns = int(g[BN]);
    dsv41_quant_gmm_launch(args);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("quant_gmm_geometry(int k, int n, int bm) -> int[]", &geometry_config);
    m.def("quant_gmm_select(int k, int n, int rows, float mean, int bound, int cores) -> int",
          &select_config);
    m.def("quant_gmm_cores(int device) -> int", &gemm_cores);
    m.def("prepare_quant_gmm(Tensor x, Tensor packed, Tensor scales, Tensor routing, "
          "Tensor(a!) out, Tensor(b!)? workspace, int bm) -> ()",
          [](const at::Tensor &x, const at::Tensor &packed, const at::Tensor &scales,
             const at::Tensor &routing, const at::Tensor &out,
             const std::optional<at::Tensor> &workspace,
             int64_t bm) {
              native_jit::PrepareOnlyScope prepare;
              run(x, packed, scales, routing, out, workspace, bm);
          });
    m.def("quant_gmm(Tensor x, Tensor packed, Tensor scales, Tensor routing, "
          "Tensor(a!) out, Tensor(b!)? workspace, int bm) -> ()",
          &run);
}
} // namespace native_host
