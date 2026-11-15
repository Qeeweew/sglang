#include "jit.h"
#include <ATen/core/ivalue.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>
#include <filesystem>
#include <cstdlib>
#include <mutex>
#include <exception>
#include <unordered_map>
#include <deep_jit/backend/ascend/backend.hpp>
#include <deep_jit/cache/compile_lock.hpp>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

namespace native_jit {
namespace {
using Runtime = deep_jit::Runtime<deep_jit::Ascend>;
struct Configuration {
    std::filesystem::path root;
    std::filesystem::path toolkit_root;
    std::string signature;
    std::string cache;
};
Configuration config;
std::mutex prepare_mutex;
std::unordered_map<int, std::unique_ptr<Runtime>> runtimes;
thread_local bool prepare_only = false;
thread_local PreparationBatch *active_batch = nullptr;
thread_local std::unique_ptr<PreparationBatch> requested_batch;
thread_local bool previous_prepare_only = false;

std::filesystem::path selected_toolkit_root() {
    const auto toolkit = deep_jit::Ascend::find_ascend_toolkit();
    return std::filesystem::canonical(toolkit.bisheng.parent_path().parent_path());
}

Runtime &runtime(int device) {
    auto &value = runtimes[device];
    if (!value) {
        TORCH_CHECK(!config.root.empty(), "configure DeepJIT before preparing native operators");
        TORCH_CHECK(selected_toolkit_root() == config.toolkit_root,
                    "Ascend toolkit changed after native configuration; restart and rebuild ",
                    "with the selected SDK");
        std::vector<std::string> prefixes;
        for (const auto &directory : std::filesystem::directory_iterator(config.root / "native")) {
            if (directory.is_directory())
                prefixes.push_back(directory.path().filename().string() + "/");
        }
        value = std::make_unique<Runtime>(deep_jit::Config(
            config.root, "DSV41", config.signature,
            {config.root / "native"}, prefixes));
    }
    return *value;
}
} // namespace

// The producer alone touches Runtime's parser/memory cache and device handles.
// Workers use its const compile-without-load entry and the shared DiskCache.
class PreparationBatch {
    struct Pending {
        std::shared_ptr<Kernel::DeviceSlot> slot;
        int device;
        Runtime *jit;
        std::string tag;
        std::string source;
        std::string digest;
        deep_jit::ascend::CompilerOptions options;
    };
    std::unique_ptr<deep_jit::Precompiler> compiler_;
    std::vector<std::shared_ptr<Pending>> pending_;

  public:
    void enqueue(std::shared_ptr<Kernel::DeviceSlot> slot, int device, Runtime &jit, std::string tag,
                 std::string source, std::string digest,
                 deep_jit::ascend::CompilerOptions options) {
        if (!compiler_)
            compiler_ = std::make_unique<deep_jit::Precompiler>(
                jit.env.get<int>("JIT_COMPILE_THREADS", 8), jit.env.get<bool>("JIT_DEBUG", false));
        const auto name = tag + "." + digest;
        auto pending = std::make_shared<Pending>(Pending{std::move(slot), device, &jit,
            std::move(tag), std::move(source), std::move(digest), std::move(options)});
        pending_.push_back(pending);
        compiler_->enqueue(name, [pending] {
            auto &jit = *pending->jit;
            size_t root;
            {
                auto slot = jit.disk_cache.entry(pending->tag, pending->digest);
                if (slot.hit) return;
                root = slot.root_index;
            }
            deep_jit::CompileLock lock(jit.disk_cache.paths[root], pending->digest);
            // No parser, memory-cache access or NPU context on this thread.
            jit.compile(pending->tag, pending->source, pending->digest, pending->options);
        });
    }

    void finish() {
        if (!compiler_) return;
        compiler_->finish(); // Join all tasks before propagating any error.
        compiler_.reset();
        // Compilation is complete. Load on the caller with the right device
        // context, retaining the ordinary per-entry warm handle and cache.
        for (const auto &pending : pending_) {
            if (pending->slot->handle) continue;
            const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, pending->device));
            const std::lock_guard lock(prepare_mutex);
            pending->slot->handle = pending->jit->compile(pending->tag, pending->source, pending->options);
            TORCH_CHECK(pending->slot->handle, "native runtime must not use DeepJIT dry-run mode");
        }
        pending_.clear();
    }
};

void configure(const std::string &source_root, const std::string &signature,
               const std::string &cache_root) {
    const std::lock_guard lock(prepare_mutex);
    const auto root = std::filesystem::canonical(source_root);
    const auto toolkit_root = selected_toolkit_root();
    TORCH_CHECK(std::filesystem::is_directory(root / "native"), "missing native source snapshot");
    TORCH_CHECK(config.root.empty() ||
                    (config.root == root && config.toolkit_root == toolkit_root &&
                     config.signature == signature && config.cache == cache_root),
                "native JIT source snapshot is already configured; restart to change it");
    config = {root, toolkit_root, signature, cache_root};
    if (!cache_root.empty()) {
        TORCH_CHECK(::setenv("DSV41_JIT_CACHE_DIR", cache_root.c_str(), 1) == 0,
                    "could not configure native JIT cache");
    }
}

bool preparing_only() {
    return prepare_only;
}
PrepareOnlyScope::PrepareOnlyScope() : previous_(prepare_only), exceptions_(std::uncaught_exceptions()) {
    if (!active_batch) {
        batch_ = std::make_unique<PreparationBatch>();
        active_batch = batch_.get();
    }
    prepare_only = true;
}
PrepareOnlyScope::~PrepareOnlyScope() noexcept(false) {
    prepare_only = previous_;
    if (!batch_) return;
    active_batch = nullptr;
    try {
        batch_->finish();
    } catch (...) {
        if (std::uncaught_exceptions() <= exceptions_) throw;
        // Preserve the original host validation error after draining workers.
    }
}

void begin_prepare_batch() {
    TORCH_CHECK(!active_batch, "native prepare batches cannot be nested");
    TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None,
                "begin native prepare batch before Graph capture");
    requested_batch = std::make_unique<PreparationBatch>();
    previous_prepare_only = prepare_only;
    prepare_only = true;
    active_batch = requested_batch.get();
}

void end_prepare_batch() {
    TORCH_CHECK(requested_batch, "no native prepare batch to finish");
    auto batch = std::move(requested_batch);
    active_batch = nullptr;
    prepare_only = previous_prepare_only;
    batch->finish();
}

Kernel::Kernel(const char *tag, const char *relative_source, std::vector<std::string> flags,
               uint32_t workers)
    : tag_(tag), source_(relative_source), flags_(std::move(flags)), workers_(workers) {}

const std::shared_ptr<deep_jit::ascend::Kernel> &Kernel::prepare() {
    const int device = c10_npu::current_device();
    TORCH_CHECK(device >= 0, "native JIT requires a current NPU device");
    if (static_cast<size_t>(device) < devices_.size() && devices_[device] && devices_[device]->handle)
        return devices_[device]->handle;
    TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None,
                "cold native kernel ", tag_, " during Graph capture; bind/prewarm it first");
    const std::lock_guard lock(prepare_mutex);
    auto &jit = runtime(device);
    const auto path = config.root / "native" / source_;
    TORCH_CHECK(std::filesystem::is_regular_file(path),
                "missing native kernel source: ", path.string());
    deep_jit::ascend::CompilerOptions options;
    options.extra_bisheng_flags = flags_;
    const auto source = deep_jit::read(path);
    const auto digest = jit.cache_key(source, jit.default_compiler_options.override_with(options));
    if (static_cast<size_t>(device) >= devices_.size())
        devices_.resize(device + 1);
    if (!devices_[device])
        devices_[device] = std::make_shared<DeviceSlot>();
    bool hit;
    size_t root;
    {
        // Let DeepJIT select the cache root, including its existing IO fallback.
        auto entry = jit.disk_cache.entry(tag_, digest);
        hit = entry.hit;
        root = entry.root_index;
    }
    if (active_batch && !hit) {
        active_batch->enqueue(devices_[device], device, jit, tag_, source, digest,
                              jit.default_compiler_options.override_with(options));
        return devices_[device]->handle;
    }
    std::shared_ptr<deep_jit::ascend::Kernel> kernel;
    if (hit) {
        kernel = jit.compile(tag_, source, options);
    } else {
        // Cold preparation only: wait per complete source/options digest, then
        // DeepJIT rechecks its disk cache before deciding whether to compile.
        deep_jit::CompileLock compile_lock(jit.disk_cache.paths[root], digest);
        kernel = jit.compile(tag_, source, options);
    }
    TORCH_CHECK(kernel, "native runtime must not use DeepJIT dry-run mode");
    devices_[device]->handle = std::move(kernel);
    return devices_[device]->handle;
}

} // namespace native_jit
