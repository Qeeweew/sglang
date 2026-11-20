#pragma once
#include <c10/util/Exception.h>
#include <cstdint>
#include <unordered_map>

extern "C" int rtGetL2CacheOffset(uint32_t device, uint64_t *offset);

namespace native_host {
inline uint64_t l2_bypass_alias(int device) {
    thread_local std::unordered_map<int, uint64_t> aliases;
    auto found = aliases.find(device);
    if (found == aliases.end()) {
        uint64_t alias = 0;
        const auto status = rtGetL2CacheOffset(device, &alias);
        TORCH_CHECK(status == 0 && alias, "910B L2 bypass alias unavailable: ", status);
        found = aliases.emplace(device, alias).first;
    }
    return found->second;
}
} // namespace native_host
