#include "torch.h"
#include "launch.h"
#include "format.h"
#include "jit.h"
namespace native_host {
namespace {
using namespace mxfp4_weight;
void pack(const at::Tensor &codes, const at::Tensor &scales, const at::Tensor &tables,
          const at::Tensor &packed, const at::Tensor &s2, int64_t pk) {
    const auto device = codes.device();
    check(codes, at::kByte, device);
    TORCH_CHECK_VALUE(codes.dim() == 2 && codes.size(0) > 0 && codes.size(1) > 0 &&
                          pk >= kScaleGroup && pk <= kMaxPanelRows && pk % kScaleGroup == 0 &&
                          codes.size(1) * kCodesPerByte % pk == 0 &&
                          codes.size(0) % 64 == 0,
                      "packing requires group32 panels and complete N blocks");
    check(scales, at::kByte, device, {codes.size(0), codes.size(1) / kPackedBytesPerScale});
    check(packed, at::kInt, device, {codes.numel() / int64_t(sizeof(int32_t))});
    TORCH_CHECK_VALUE(s2.scalar_type() == at::kHalf || s2.scalar_type() == at::kByte,
                      "packed scale output must be FP16 or preprocessed uint8");
    check(s2, s2.scalar_type(), device, {codes.numel() / kPackedBytesPerScale});
    check(tables, at::kByte, device);
    TORCH_CHECK_VALUE(tables.dim() == 1 && uint64_t(tables.numel()) == table_bytes(pk),
                      "invalid packing lookup tables");
    disjoint({packed, s2}, {codes, scales, tables});

    const int64_t blocks = core_count(codes.get_device());

    c10_npu::NPUGuard guard(codes.device());
    auto stream = c10_npu::getCurrentNPUStream(codes.get_device());
    for (const auto &t : {codes, scales, tables, packed, s2})
        record(t, stream);
    dsv41_mxfp4_pack(stream.stream(), blocks, codes.data_ptr(), scales.data_ptr(),
                     tables.data_ptr(), packed.data_ptr(), s2.data_ptr(),
                     codes.size(1) * kCodesPerByte, codes.size(0), pk,
                     s2.scalar_type() == at::kByte);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("prepare_mxfp4_pack(Tensor codes, Tensor scales, Tensor tables, Tensor packed, "
          "Tensor s2, int pk) -> ()",
          [](const at::Tensor &codes, const at::Tensor &scales, const at::Tensor &tables,
             const at::Tensor &packed, const at::Tensor &s2, int64_t pk) {
              native_jit::PrepareOnlyScope prepare;
              pack(codes, scales, tables, packed, s2, pk);
          });
    m.def("mxfp4_pack(Tensor codes, Tensor scales, Tensor tables, Tensor(a!) packed, Tensor(b!) "
          "s2, int pk) -> ()",
          &pack);
}
} // namespace native_host
