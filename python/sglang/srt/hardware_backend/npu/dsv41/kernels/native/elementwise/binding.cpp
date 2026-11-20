#include "torch.h"
#include "launch.h"
#include "layout.h"
#include "jit.h"
namespace native_host {
namespace {
std::tuple<at::Tensor, at::Tensor> split_rms_norm(const at::Tensor &input, const at::Tensor &first,
                                                  const at::Tensor &second, double first_eps,
                                                  double second_eps) {
    const auto device = input.device();
    TORCH_CHECK_VALUE(device.type() == c10::DeviceType::PrivateUse1 && input.dim() == 2 &&
                      input.scalar_type() == at::kBFloat16 && input.stride(1) == 1 &&
                      input.stride(0) >= input.size(1), "expected row-strided BF16 NPU matrix");
    const auto format = at_npu::native::get_npu_format(input);
    TORCH_CHECK_VALUE(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW, "expected ND input");
    for (const auto &weight : {first, second}) {
        check(weight, at::kBFloat16, device);
        TORCH_CHECK_VALUE(weight.dim() == 1 && weight.numel() > 0 &&
                          weight.numel() <= elementwise::kReductionPanelElements &&
                          weight.numel() % elementwise::kFp32VectorElements == 0,
                          "expected RMSNorm weight covering complete vector repeats");
    }
    TORCH_CHECK_VALUE(first.numel() + second.numel() == input.size(1) &&
                      std::isfinite(first_eps) && first_eps > 0 &&
                      std::isfinite(second_eps) && second_eps > 0, "invalid split/epsilons");
    c10_npu::NPUGuard guard(device);
    const auto rows = input.size(0);
    auto first_out = at::empty({rows, first.numel()}, input.options());
    auto second_out = at::empty({rows, second.numel()}, input.options());
    if (rows) {
        auto stream = c10_npu::getCurrentNPUStream(input.get_device());
        for (const auto &tensor : {input, first, second, first_out, second_out}) record(tensor, stream);
        split_rms_norm_launch(stream.stream(), u32(core_count(input.get_device())), ptr(input),
                              ptr(first), ptr(second), ptr(first_out), ptr(second_out),
                              u32(first.numel()), u32(second.numel()), f32(first_eps),
                              f32(second_eps), rows, input.stride(0));
    }
    return {first_out, second_out};
}
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
            const at::Tensor &out, double limit, bool packed,
            const std::optional<at::Tensor> &shared_out = std::nullopt) {
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
    int64_t shared_rows = 0;
    if (shared_out) {
        check(up, type, device);
        TORCH_CHECK_VALUE(up.dim() == 2 && up.size(1) == gate.size(1),
                          "joint SwiGLU expects two packed inputs with the same width");
        shared_rows = up.size(0);
        check(*shared_out, type, device, {shared_rows, d});
    } else {
        check(up, type, device, gate.sizes());
    }
    const auto rows = gate.size(0) + shared_rows;
    check(out, type, device, {gate.size(0), d});
    if (count)
        check(*count, at::kInt, device, {1});
    disjoint({out, shared_out}, {gate, up, count});
    if (!rows)
        return;
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(gate.get_device());
    track(gate, stream);
    track(up, stream);
    track(count, stream);
    track(out, stream);
    track(shared_out, stream);
    dsv41_swiglu_launch(stream.stream(), u32(core_count(gate.get_device())), ptr(gate), ptr(up),
                        ptr(count), ptr(out), rows, u32(d), f32(limit), packed,
                        type == at::kHalf, shared_out.has_value(), shared_rows, ptr(shared_out));
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("split_rms_norm(Tensor input, Tensor first_weight, Tensor second_weight, float first_eps, "
          "float second_eps) -> (Tensor, Tensor)", &split_rms_norm);
    m.def("rms_norm(Tensor x, Tensor weight, Tensor(a!) out, float epsilon) -> ()", &rms_norm);
    m.def("swiglu(Tensor gate, Tensor up, Tensor? count, Tensor(a!) out, float limit, bool packed) "
          "-> ()",
          [](const at::Tensor &gate, const at::Tensor &up, const std::optional<at::Tensor> &count,
             const at::Tensor &out, double limit, bool packed) { swiglu(gate, up, count, out, limit, packed); });
    m.def("joint_swiglu(Tensor shared, Tensor routed, Tensor count, Tensor(a!) shared_out, "
          "Tensor(b!) routed_out, float limit) -> ()",
          [](const at::Tensor &shared, const at::Tensor &routed, const at::Tensor &count,
             const at::Tensor &shared_out, const at::Tensor &routed_out, double limit) {
              swiglu(routed, shared, count, routed_out, limit, true, shared_out);
          });
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
