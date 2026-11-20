#pragma once
#include <kernel_operator.h>

namespace native_device {
template <class T, class Offset> __aicore__ inline __ubuf__ T *ub_ptr(Offset byte_offset) {
    return reinterpret_cast<__ubuf__ T *>(uint64_t(byte_offset));
}

// Keep fixed event IDs in the instructions' immediate fields on 910B.
template <pipe_t From, pipe_t To, uint32_t Event> __aicore__ inline void pipe_fence() {
    set_flag(From, To, ::event_t(Event));
    wait_flag(From, To, ::event_t(Event));
}

// Runtime ping-pong slots use the register form.
template <pipe_t From, pipe_t To> __aicore__ inline void pipe_fence(uint32_t event) {
    set_flag(From, To, ::event_t(event));
    wait_flag(From, To, ::event_t(event));
}
} // namespace native_device
