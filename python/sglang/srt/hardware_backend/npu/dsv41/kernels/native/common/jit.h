#pragma once

// Host-only DeepJIT integration. Device pipelines must not include this header.
#include <cstdint>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <c10/util/Exception.h>
#include <deep_jit/backend/ascend/kernel.hpp>

namespace native_jit {

void configure(const std::string &source_root, const std::string &signature,
               const std::string &cache_root);

// Reuses an operator's checked host launch path to prepare kernels without
// executing device work. Used by bind() and explicit precompilation only.
bool preparing_only();
class PreparationBatch;
class PrepareOnlyScope {
    bool previous_;
    int exceptions_;
    std::unique_ptr<PreparationBatch> batch_;

  public:
    PrepareOnlyScope();
    ~PrepareOnlyScope() noexcept(false);
    PrepareOnlyScope(const PrepareOnlyScope &) = delete;
    PrepareOnlyScope &operator=(const PrepareOnlyScope &) = delete;
};

// Python-owned outer scope; operator PrepareOnlyScope instances join it.
void begin_prepare_batch();
void end_prepare_batch();

// Declare operator entries static thread_local. A warm launch performs one
// device-slot access, with no runtime lock, source hashing or disk-cache lookup.
// Flags describe dataflow/encoding and bind-stable geometry. Compile only the
// configurations actually used; live lengths and routing stay dynamic.
// Persistent entries bind a queried maximum worker plan. Launch occupancy is
// runtime-only; it does not change code generation or workspace capacity.
class Kernel {
    friend class PreparationBatch;
    struct DeviceSlot {
        std::shared_ptr<deep_jit::ascend::Kernel> handle;
    };
    std::string tag_;
    std::string source_;
    std::vector<std::string> flags_;
    std::vector<std::shared_ptr<DeviceSlot>> devices_;
    uint32_t workers_;

  public:
    Kernel(const char *tag, const char *relative_source, std::vector<std::string> flags = {},
           uint32_t workers = 0);
    const std::shared_ptr<deep_jit::ascend::Kernel> &prepare();

    // task_blocks covers every potentially active worker index. Cyclic jobs use
    // the job count, contiguous partitions use their nonempty partition count.
    // Keep constexpr scheduling strides: once cyclic jobs fill the budget the
    // entire grid must launch. Empty-output metadata stages explicitly pass 1.
    template <class... Args>
    void launch_persistent(void *stream, uint64_t task_blocks, const Args &...args) {
        TORCH_CHECK(workers_ > 0, "bind a worker plan before persistent launch");
        const auto &handle = prepare();
        if (preparing_only() || task_blocks == 0)
            return;
        deep_jit::ascend::LaunchOptions options;
        options.stream = stream;
        options.num_blocks = static_cast<int>(std::min<uint64_t>(workers_, task_blocks));
        options.num_ubuf_bytes = 0;
        options.num_launch_timeout_secs = 0;
        handle->launch(options, args...);
    }

};

} // namespace native_jit
