#pragma once
#include "jit.h"
#include "torch.h"

namespace native_host {
class LaunchWorkspace {
    struct Release {
        void operator()(uint8_t *address) const {
            c10_npu::NPUCachingAllocator::raw_delete(address);
        }
    };
    std::unique_ptr<uint8_t, Release> owned_;
    uint8_t *address_;

  public:
    LaunchWorkspace(const std::optional<at::Tensor> &provided, size_t bytes, aclrtStream stream)
        : owned_(provided || !bytes || native_jit::preparing_only()
                     ? nullptr
                     : static_cast<uint8_t *>(
                           c10_npu::NPUCachingAllocator::raw_alloc_with_stream(bytes, stream))),
          address_(provided ? ptr(*provided) : owned_.get()) {}

    uint8_t *data() const { return address_; }
};
} // namespace native_host
