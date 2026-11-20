#include "launch.h"
#include "actual_entry.h"
#include <array>
#include <stdexcept>
#include "torch.h"
#include <memory>
namespace {
// Actual BM/BN resource plans only; native NZ has no stripe-capacity table.
using Key = std::array<int, 6>; // BM, BN, K, N, experts, workers

native_jit::Kernel &entry(const quant_gmm::Arguments &args) {
    static thread_local native_jit::KernelCache<Key> entries;
    const auto &shape = args.config;
    const Key key{args.tile_rows, args.tile_columns, shape.k, shape.n,
                  args.experts, args.cores};
    return entries.get(key, [&] {
        const auto tag = "quant_gmm_m" + std::to_string(args.tile_rows) + "_n" +
                         std::to_string(args.tile_columns) + "_k" + std::to_string(shape.k) + "_columns" +
                         std::to_string(shape.n) + "_experts" +
                         std::to_string(args.experts) + "_w" + std::to_string(args.cores);
        return std::make_unique<native_jit::Kernel>(
            tag.c_str(), "quant_grouped_gemm/grouped_gemm.asc",
            std::vector<std::string>{"--cce-auto-sync=off",
                                     "-DQUANT_GMM_BM=" + std::to_string(args.tile_rows),
                                     "-DQUANT_GMM_BN=" + std::to_string(args.tile_columns),
                                     "-DQUANT_GMM_K=" + std::to_string(shape.k),
                                     "-DQUANT_GMM_N=" + std::to_string(shape.n),
                                     "-DQUANT_GMM_EXPERTS=" + std::to_string(args.experts),
                                     "-DQUANT_GMM_WORKERS=" + std::to_string(args.cores)},
            args.cores);
    });
}
} // namespace
extern "C" void dsv41_quant_gmm_launch(const quant_gmm::Arguments &args) {
    const bool supported = (args.tile_rows == 16 || args.tile_rows == 128 || args.tile_rows == 256) &&
        args.tile_columns >= 64 && args.tile_columns <= 256 &&
        args.tile_columns % (quant_gmm::kVectorLanesPerCube * quant_gmm::Storage::kC0Elements) == 0 &&
        args.tile_rows * args.tile_columns * sizeof(float) <= arch2201::kL0CBytes;
    if (!supported)
        throw std::invalid_argument("unsupported grouped GEMM storage plan");
    // Routing metadata is device-resident; its job count cannot be inferred
    // from activation capacity without excluding possible expert/tile owners.
    entry(args).launch_persistent(args.stream, args.cores, args.activation, args.packed, args.scales,
                                  args.routing, args.output, args.workspace);
}

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_quant_grouped_gemm(int k, int n, int experts, int block_m, "
          "int workers=0) -> ()",
          [](int64_t k, int64_t n, int64_t experts, int64_t block_m,
             int64_t workers) {
              using namespace quant_gmm;
              TORCH_CHECK_VALUE(k > 0 && k <= INT32_MAX && n > 0 && n <= INT32_MAX && experts > 0 &&
                                    experts <= INT32_MAX,
                                "invalid grouped GEMM weight geometry");
              const auto plan = geometry(int(k), int(n), int(block_m));
              TORCH_CHECK_VALUE(plan[K] > 0, "invalid grouped GEMM storage plan");
              const auto capacity = native_host::core_count(c10_npu::current_device(), true);
              if (!workers)
                  workers = capacity;
              TORCH_CHECK_VALUE(workers > 0 && workers <= capacity,
                                "invalid grouped GEMM worker plan");
              Arguments args{};
              args.cores = int(workers);
              args.experts = int(experts);
              args.tile_rows = int(plan[BM]);
              args.tile_columns = int(plan[BN]);
              args.config = {int(k), int(n)};
              entry(args).prepare();
          });
}
