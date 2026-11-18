#include "torch.h"
#include "launch.h"
#include "jit.h"
#include "workspace.h"

namespace native_host {
namespace {
using dspark_schedule::kRequestGroup;
at::Tensor schedule(const at::Tensor &confidence, int64_t budget, int64_t minimum, int64_t maximum,
                    double eps, const std::optional<at::Tensor> &workspace,
                    const std::optional<at::Tensor> &output) {
    const auto device = confidence.device();
    check(confidence, at::kFloat, device);
    TORCH_CHECK_VALUE(confidence.dim() == 2 && confidence.size(1) >= 1,
                      "confidence requires FP32 [requests,gamma]");
    const auto rows = confidence.size(0), gamma = confidence.size(1);
    if (!maximum)
        maximum = gamma + 1;
    TORCH_CHECK_VALUE(minimum >= 0 && minimum <= maximum && maximum <= gamma + 1 &&
                          maximum < INT32_MAX,
                      "require 0<=min<=max<=gamma+1 and INT32 output lengths");
    TORCH_CHECK_VALUE(!(eps < 0), "survival_eps must be nonnegative");
    const auto columns = std::min(maximum, gamma),
               stride = (rows + kRequestGroup - 1) / kRequestGroup * kRequestGroup;
    auto out = output.value_or(at::Tensor());
    c10_npu::NPUGuard guard(device);
    if (!out.defined())
        out = at::empty({rows}, confidence.options().dtype(at::kInt));
    if (workspace)
        check(*workspace, at::kFloat, device, {columns, stride});
    check(out, at::kInt, device, {rows});
    disjoint({workspace, out}, {confidence});
    if (!rows)
        return out;
    auto stream = c10_npu::getCurrentNPUStream(confidence.get_device());
    const auto acl_stream = stream.stream();
    const LaunchWorkspace scratch(workspace, budget > 0 ? columns * stride * sizeof(float) : 0,
                                  acl_stream);
    for (const auto &t : {confidence, out})
        record(t, stream);
    track(workspace, stream);
    const auto blocks = core_count(confidence.get_device());
    if (budget > 0)
        dsv41_dspark_schedule_prepare_launch(acl_stream, u32(blocks), ptr(confidence),
                                             scratch.data(), rows, gamma, columns, stride, f32(eps));
    dsv41_dspark_schedule_finish_launch(acl_stream, u32(blocks), scratch.data(), ptr(out), rows,
                                        columns, stride, budget, minimum, maximum);
    return out;
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("prepare_dspark_schedule(Tensor confidence, int budget, int minimum, int maximum, float "
          "eps, "
          "Tensor? survival, Tensor out) -> ()",
          [](const at::Tensor &confidence, int64_t budget, int64_t minimum, int64_t maximum,
             double eps, const std::optional<at::Tensor> &survival, const at::Tensor &out) {
              native_jit::PrepareOnlyScope prepare;
              schedule(confidence, budget, minimum, maximum, eps, survival, out);
          });
    m.def("dspark_schedule(Tensor confidence, int budget, int minimum, int maximum, float eps, "
          "Tensor(a!)? survival=None, Tensor(b!)? out=None) -> Tensor(b!)",
          &schedule);
}
} // namespace native_host
