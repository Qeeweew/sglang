#include "plan.h"
#include "torch.h"
#include "jit.h"
#include "actual_entry.h"
#include <common/grouped_metadata.h>
#include <common/l2_bypass.h>

namespace mxfp4_a8 {
using namespace native_host;
template <bool g32>
void run(const at::Tensor &activation, const at::Tensor &codes, const at::Tensor &factors,
         const at::Tensor &channel, const at::Tensor &token, const at::Tensor &routing,
         const at::Tensor &output, const at::Tensor &scratch, bool prepare) {
    const auto device = activation.device();
    TORCH_CHECK_VALUE(activation.dim() == 2 && channel.dim() == 2, "expected matrices");
    const auto count = activation.size(0), depth = activation.size(1);
    const auto experts = channel.size(0), width = channel.size(1);
    TORCH_CHECK_VALUE(depth >= 256 && depth <= 8192 && depth % group == 0 && width > 0 &&
                          width <= UINT16_MAX && width % 64 == 0 && experts > 0 &&
                          experts <= INT32_MAX / grouped_metadata::kWords - 1 &&
                          count <= INT32_MAX - tile_rows(depth, width, g32),
                      "unsupported decode geometry");
    check(activation, at::kChar, device, {count, depth});
    check(channel, at::kFloat, device);
    check(codes, at::kByte, device, {experts, width * depth / 2});
    check(factors, at::kByte, device, {experts, width * depth / 64});
    check(token, at::kFloat, device, {count});
    const bool bf16 = g32 && output.scalar_type() == at::kBFloat16;
    check(output, bf16 ? at::kBFloat16 : at::kHalf, device, {count, width});
    check(routing, at::kInt, device);
    TORCH_CHECK_VALUE(routing.dim() == 1 &&
                          routing.numel() >= grouped_metadata::kWords * (1 + experts),
                      "compact routing capacity too small");
    c10_npu::NPUGuard guard(device);
    const int workers = core_count(device.index(), true);
    static thread_local int last_device = -1;
    if (last_device != device.index()) {
        const char *soc = aclrtGetSocName();
        TORCH_CHECK_VALUE(core_count(device.index()) == 2 * workers && soc &&
                              std::string(soc).find("Ascend910B") == 0,
                          "requires Ascend 910B mixed cores");
        last_device = device.index();
    }
    const int columns = tile_columns(depth, width, g32), panel = panel_depth(depth, columns, g32);
    const int local = columns / 2 * panel, dot = dot_depth(columns);
    const int64_t tasks = count * (width / columns);
    const int64_t serials = ((tasks + workers - 1) / workers) * (depth / panel) *
                            std::max(local / quantum_size(local), (panel + dot - 1) / dot);
    TORCH_CHECK_VALUE(tasks <= INT32_MAX - workers && serials <= INT32_MAX - slots,
                      "decode schedule exceeds INT32 range");
    check(scratch, at::kByte, device);
    TORCH_CHECK_VALUE(scratch.numel() >= scratch_bytes(depth, width, workers, g32),
                      "insufficient panel workspace");
    disjoint({output, scratch}, {activation, codes, factors, channel, token, routing});
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    for (const auto &tensor :
         {activation, codes, factors, channel, token, routing, output, scratch}) {
        TORCH_CHECK_VALUE(tensor.is_same(token) || uintptr_t(ptr(tensor)) % 32 == 0, "unaligned tensor base");
        record(tensor, stream);
    }
    const auto alias = l2_bypass_alias(device.index());
    const auto streaming_codes = uintptr_t(ptr(codes)) + alias;
    const Inputs inputs{
        uintptr_t(ptr(activation)), streaming_codes,         uintptr_t(ptr(factors)),
        uintptr_t(ptr(channel)),    uintptr_t(ptr(token)),   uintptr_t(ptr(routing)),
        uintptr_t(ptr(output)), uintptr_t(ptr(scratch)), int(experts)};
    if (prepare) {
        native_jit::PrepareOnlyScope scope;
        launch(stream.stream(), depth, width, workers, inputs, g32, bf16);
    } else {
        launch(stream.stream(), depth, width, workers, inputs, g32, bf16);
    }
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("h32_quant(Tensor input, Tensor indices, Tensor(a!) codes, Tensor(b!) scales, bool interleaved, Tensor? count=None, int shared_rows=0) -> ()",
          [](const at::Tensor &input, const at::Tensor &indices, const at::Tensor &codes,
             const at::Tensor &scales, bool interleaved, const std::optional<at::Tensor> &count, int64_t shared_rows) {
              const auto device = input.device();
              TORCH_CHECK_VALUE(input.dim() == 2 && (input.scalar_type() == at::kHalf ||
                                input.scalar_type() == at::kBFloat16), "expected FP16/BF16 matrix");
              const auto rows = input.size(0), width = input.size(1);
              TORCH_CHECK_VALUE(width >= 256 && width <= 8192 && width % 256 == 0, "invalid H32 width");
              check(input, input.scalar_type(), device); check(indices, at::kInt, device, {2, width});
              check(codes, at::kChar, device, {rows, width}); check(scales, at::kFloat, device, {rows});
              if (count) check(*count, at::kInt, device, {1});
              TORCH_CHECK_VALUE(shared_rows >= 0 && shared_rows <= rows && (!shared_rows || count), "invalid shared prefix");
              disjoint({codes, scales}, {input, indices, count});
              if (!rows) return;
              c10_npu::NPUGuard guard(device);
              auto stream = c10_npu::getCurrentNPUStream(device.index());
              for (const auto &tensor : {input, indices, codes, scales}) record(tensor, stream);
              track(count, stream);
              const uint64_t workers = core_count(device.index());
              const bool bf16 = input.scalar_type() == at::kBFloat16;
              static thread_local native_jit::ActualEntries<4> entries;
              entries.get({uint64_t(width), bf16, interleaved, workers}, [&] {
                  return std::make_unique<native_jit::Kernel>("h32_quant", "mxfp4_a8/h32.asc",
                      std::vector<std::string>{"--cce-auto-sync=off", "-DH32_WIDTH=" + std::to_string(width),
                      "-DH32_BF16=" + std::to_string(bf16), "-DH32_INTERLEAVED=" + std::to_string(interleaved)}, workers);
              }).launch_persistent(stream.stream(), rows, ptr(input), ptr(indices), ptr(codes), ptr(scales), rows, ptr(count), shared_rows);
          });
    m.def("mxfp4_a8(Tensor activation, Tensor codes, Tensor factors, Tensor channel, Tensor token, "
          "Tensor routing, Tensor(a!) output, Tensor(b!) scratch, bool prepare=False) -> ()",
          &run<false>);
    m.def("int4_g32_a8(Tensor activation, Tensor codes, Tensor factors, Tensor channel, Tensor token, "
          "Tensor routing, Tensor(a!) output, Tensor(b!) scratch, bool prepare=False) -> ()", &run<true>);
    m.def("mxfp4_a8_scratch(int depth, int width, int device) -> int",
          [](int64_t depth, int64_t width, int64_t device) {
              TORCH_CHECK_VALUE(depth >= 256 && depth <= 8192 && depth % group == 0 && width > 0 &&
                                    width <= UINT16_MAX && width % 64 == 0,
                                "unsupported geometry");
              const int workers = core_count(device, true);
              return std::max(scratch_bytes(depth, width, workers, true), scratch_bytes(depth, width, workers, false));
          });
}
} // namespace mxfp4_a8
