#include "torch.h"
#include "jit.h"

TORCH_LIBRARY(dsv41, m) {
    m.def("abi() -> int", []() -> int64_t { return 25; });
    m.def("configure_jit(str source_root, str signature, str cache_root) -> ()",
          &native_jit::configure);
    m.def("begin_prepare_batch() -> ()", &native_jit::begin_prepare_batch);
    m.def("end_prepare_batch() -> ()", &native_jit::end_prepare_batch);
}
