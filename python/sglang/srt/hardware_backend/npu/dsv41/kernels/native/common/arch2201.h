#pragma once
#include <cstdint>

// Host/device allocation contract for the project's sole target, Ascend 910B.
// CANN 9.1 asc/include/utils/base/sys_constants.h, __NPU_ARCH__ == 2201.
// The SDK reserves 256 bytes from UB/L1; these are usable allocation budgets.
// Shared numeric limits used by host planning and device code.
namespace arch2201 {
// CANN 9.1 adv_api/detail/matmul/stage/compute/mmad_compute.h, Compute().
inline constexpr int kMmadFractal = 16;
inline constexpr int kUnfencedMmadFractals = 10;

inline constexpr uint64_t kKiB = 1024;
inline constexpr uint64_t kSdkReservedBytes = 256;
inline constexpr uint64_t kUbBytes = 192 * kKiB - kSdkReservedBytes;
inline constexpr uint64_t kL1Bytes = 512 * kKiB - kSdkReservedBytes;
inline constexpr uint64_t kL0ABytes = 64 * kKiB;
inline constexpr uint64_t kL0BBytes = 64 * kKiB;
inline constexpr uint64_t kL0CBytes = 128 * kKiB;
} // namespace arch2201

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#include <utils/base/sys_constants.h>
static_assert(arch2201::kUbBytes == ASC_UB_SIZE);
static_assert(arch2201::kL1Bytes == ASC_L1_SIZE);
static_assert(arch2201::kL0ABytes == ASC_L0A_SIZE);
static_assert(arch2201::kL0BBytes == ASC_L0B_SIZE);
static_assert(arch2201::kL0CBytes == ASC_L0C_SIZE);
#endif
