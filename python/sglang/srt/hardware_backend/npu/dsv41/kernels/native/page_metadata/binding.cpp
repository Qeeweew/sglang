#include "torch.h"
#include "launch.h"
#include "jit.h"

namespace native_host {
namespace {
void draft_page_metadata(const at::Tensor &req_to_token, const at::Tensor &full_to_swa,
                         const at::Tensor &requests, const at::Tensor &prefix,
                         const at::Tensor &full_slots, at::Tensor cu, at::Tensor lengths,
                         at::Tensor table, at::Tensor slots, at::Tensor routed_rows, int64_t block,
                         int64_t page_rows, int64_t window_rows) {
    const auto device = requests.device();
    check(requests, at::kLong, device);
    TORCH_CHECK_VALUE(requests.dim() == 1, "requests requires INT64 [batch]");
    const auto batch = requests.numel();
    check(req_to_token, at::kInt, device);
    check(full_to_swa, at::kLong, device);
    TORCH_CHECK_VALUE(req_to_token.dim() == 2 && req_to_token.size(0) >= 1 &&
                          req_to_token.size(1) >= 1 && full_to_swa.dim() == 1 &&
                          full_to_swa.numel() >= 1,
                      "expected nonempty request token table and full-to-SWA mapping");
    TORCH_CHECK_VALUE(block > 0 && batch <= INT32_MAX / block,
                      "positive block and INT32 query boundaries required");
    TORCH_CHECK_VALUE(page_rows > 0 && page_rows <= INT32_MAX && window_rows >= 0 &&
                          window_rows <= INT32_MAX,
                      "positive page_rows and nonnegative window_rows required");
    const auto context = req_to_token.size(1);
    const auto columns = (context + page_rows - 1) / page_rows;
    check(prefix, at::kLong, device, {batch});
    check(full_slots, at::kLong, device, {batch * block});
    check(cu, at::kInt, device, {batch + 1});
    check(lengths, at::kInt, device, {batch});
    check(table, at::kInt, device, {batch, columns});
    check(slots, at::kLong, device, {batch * block});
    check(routed_rows, at::kInt, device, {1});
    for (const auto &t : {cu, lengths, table, slots, routed_rows})
        TORCH_CHECK_VALUE(!t.numel() || reinterpret_cast<uintptr_t>(t.data_ptr()) % 32 == 0,
                          "draft metadata outputs require 32B-aligned data pointers");
    disjoint({cu, lengths, table, slots, routed_rows},
             {req_to_token, full_to_swa, requests, prefix, full_slots});
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(requests.get_device());
    for (const auto &t : {req_to_token, full_to_swa, requests, prefix, full_slots, cu, lengths,
                          table, slots, routed_rows})
        record(t, stream);
    // Eight-request groups keep shared CU/length output cache lines on one core.
    const auto blocks = core_count(requests.get_device());
    dsv41_draft_page_metadata_launch(stream.stream(), u32(blocks), ptr(req_to_token),
                                     ptr(full_to_swa), ptr(requests), ptr(prefix), ptr(full_slots),
                                     ptr(cu), ptr(lengths), ptr(table), ptr(slots),
                                     ptr(routed_rows), batch, block, req_to_token.size(0), context,
                                     columns, full_to_swa.numel(), page_rows, window_rows);
}
void prepare_draft_page_metadata(const at::Tensor &req_to_token, const at::Tensor &full_to_swa,
                                 const at::Tensor &requests, const at::Tensor &prefix,
                                 const at::Tensor &full_slots, at::Tensor cu, at::Tensor lengths,
                                 at::Tensor table, at::Tensor slots, at::Tensor routed_rows,
                                 int64_t block, int64_t page_rows, int64_t window_rows) {
    native_jit::PrepareOnlyScope scope;
    draft_page_metadata(req_to_token, full_to_swa, requests, prefix, full_slots, cu, lengths, table,
                        slots, routed_rows, block, page_rows, window_rows);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("draft_page_metadata(Tensor req_to_token, Tensor full_to_swa, Tensor requests, "
          "Tensor prefix, Tensor full_slots, Tensor(a!) cu, Tensor(b!) lengths, "
          "Tensor(c!) table, Tensor(d!) slots, Tensor(e!) routed_rows, int block, int "
          "page_rows=128, int window_rows=128) -> ()",
          &draft_page_metadata);
    m.def("prepare_draft_page_metadata(Tensor req_to_token, Tensor full_to_swa, Tensor requests, "
          "Tensor prefix, Tensor full_slots, Tensor(a!) cu, Tensor(b!) lengths, "
          "Tensor(c!) table, Tensor(d!) slots, Tensor(e!) routed_rows, int block, int "
          "page_rows=128, int window_rows=128) -> ()",
          &prepare_draft_page_metadata);
}
} // namespace native_host
