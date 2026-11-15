#include "torch.h"
#include "launch.h"
#include "layout.h"
#include "jit.h"
namespace native_host {
namespace {
void rms_norm(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &out, double eps) {
    auto device = x.device();
    check(x, at::kBFloat16, device);
    TORCH_CHECK_VALUE(x.dim() == 2 && x.size(1) > 0 &&
                          x.size(1) <= 2 * elementwise::kReductionPanelElements &&
                          x.size(1) % elementwise::kFp32VectorElements == 0 &&
                          (x.size(1) <= elementwise::kReductionPanelElements ||
                           x.size(1) % (2 * elementwise::kFp32VectorElements) == 0) &&
                          elementwise::RmsLayout(x.size(1)).bytes <= arch2201::kUbBytes &&
                          std::isfinite(eps) && eps > 0,
                      "invalid RMSNorm geometry/epsilon");
    check(weight, at::kBFloat16, device, {x.size(1)});
    check(out, at::kBFloat16, device, x.sizes());
    disjoint({out}, {x, weight});
    if (!x.size(0))
        return;
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(x.get_device());
    track(x, stream);
    track(weight, stream);
    track(out, stream);
    dsv41_rms_norm_launch(stream.stream(), u32(core_count(x.get_device())),
                          ptr(x), ptr(weight), ptr(out), x.size(0), u32(x.size(1)), f32(eps));
}
void swiglu(const at::Tensor &gate, const at::Tensor &up, const std::optional<at::Tensor> &count,
            const at::Tensor &out, double limit, bool packed) {
    auto device = gate.device();
    auto type = gate.scalar_type();
    check(gate, type, device);
    TORCH_CHECK_VALUE(type == at::kBFloat16 || type == at::kHalf, "SwiGLU requires BF16/FP16");
    TORCH_CHECK_VALUE(gate.dim() == 2 && gate.size(1) % (packed ? 64 : 32) == 0 &&
                          gate.size(1) > 0 && std::isfinite(limit) && limit > 0,
                      "invalid SwiGLU geometry/clamp");
    int64_t d = gate.size(1) / (packed ? 2 : 1);
    TORCH_CHECK_VALUE(
        d <= int64_t((arch2201::kUbBytes - elementwise::kFp32VectorElements * sizeof(float)) /
                     (3 * elementwise::kSlots * sizeof(uint16_t) + 4 * sizeof(float))) &&
            elementwise::SwiGluLayout(elementwise::align_elements(u32(d))).bytes <=
                arch2201::kUbBytes,
        "SwiGLU row exceeds arch22 UB capacity");
    check(up, type, device, gate.sizes());
    check(out, type, device, {gate.size(0), d});
    if (count)
        check(*count, at::kInt, device, {1});
    disjoint({out}, {gate, up, count});
    if (!gate.size(0))
        return;
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(gate.get_device());
    track(gate, stream);
    track(up, stream);
    track(count, stream);
    track(out, stream);
    dsv41_swiglu_launch(stream.stream(), u32(core_count(gate.get_device())), ptr(gate), ptr(up),
                        ptr(count), ptr(out), gate.size(0), u32(d), f32(limit), packed,
                        type == at::kHalf);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("rms_norm(Tensor x, Tensor weight, Tensor(a!) out, float epsilon) -> ()", &rms_norm);
    m.def("swiglu(Tensor gate, Tensor up, Tensor? count, Tensor(a!) out, float limit, bool packed) "
          "-> ()",
          &swiglu);
    m.def("prepare_rms_norm(Tensor x, Tensor weight, Tensor out, float epsilon) -> ()",
          [](const at::Tensor &x, const at::Tensor &weight, const at::Tensor &out, double eps) {
              native_jit::PrepareOnlyScope prepare;
              rms_norm(x, weight, out, eps);
          });
    m.def("prepare_swiglu(Tensor gate, Tensor up, Tensor? count, Tensor out, float limit, bool "
          "packed) -> ()",
          [](const at::Tensor &gate, const at::Tensor &up, const std::optional<at::Tensor> &count,
             const at::Tensor &out, double limit, bool packed) {
              native_jit::PrepareOnlyScope prepare;
              swiglu(gate, up, count, out, limit, packed);
          });
}
} // namespace native_host
