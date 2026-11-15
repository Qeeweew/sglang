cmake_minimum_required(VERSION 3.22)

set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN installation")
if(NOT CANN_ROOT)
  set(CANN_ROOT /usr/local/Ascend/cann-9.1.0)
endif()
set(ENV{ASCEND_HOME_PATH} "${CANN_ROOT}")
project(dsv41_kernels LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  message(FATAL_ERROR "DeepJIT requires GCC 13 or newer; configure with g++-13")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
  "import torch, torch_npu; from pathlib import Path; print(torch.utils.cmake_prefix_path); print(Path(torch_npu.__file__).parent); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE torch_config OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" torch_config "${torch_config}")
list(GET torch_config 0 torch_cmake)
list(GET torch_config 1 torch_npu_root)
list(GET torch_config 2 torch_abi)
list(APPEND CMAKE_PREFIX_PATH "${torch_cmake}")
find_package(Torch REQUIRED)
find_library(NUMA_LIBRARY numa REQUIRED)
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${torch_npu_root}/lib" REQUIRED NO_DEFAULT_PATH)

set(DSV41_DEEPJIT_DIR "" CACHE PATH "Vendored DeepJIT root")
if(NOT EXISTS "${DSV41_DEEPJIT_DIR}/SOURCE.json")
  message(FATAL_ERROR "Set DSV41_DEEPJIT_DIR to SGLang's 3rdparty/deepjit")
endif()
include(cmake/NativeComponents.cmake)
set(DSV41_COMPONENTS "${DSV41_ALL_COMPONENTS}" CACHE STRING "Native operators to link")
add_library(dsv41_kernels SHARED native/common/library.cpp native/common/jit.cpp native/host_memory/binding.cpp)
target_compile_options(dsv41_kernels PRIVATE -O2 -Wall -Wextra)
target_compile_definitions(dsv41_kernels PRIVATE _GLIBCXX_USE_CXX11_ABI=${torch_abi} DJ_DISABLE_GIL=1)
# Prefer the selected CANN headers over the older ACL snapshot in Torch-NPU wheels.
target_include_directories(dsv41_kernels PRIVATE native native/common "${CANN_ROOT}/include")
target_include_directories(dsv41_kernels SYSTEM PRIVATE ${DSV41_DEEPJIT_DIR}/include "${torch_npu_root}/include" ${TORCH_INCLUDE_DIRS})
target_link_directories(dsv41_kernels PRIVATE "${CANN_ROOT}/lib64")
target_link_libraries(dsv41_kernels PRIVATE ${TORCH_LIBRARIES}
  ${TORCH_NPU_LIBRARY} ${NUMA_LIBRARY} ascendcl runtime dl)
target_link_options(dsv41_kernels PRIVATE "LINKER:-z,defs")
set_target_properties(dsv41_kernels PROPERTIES
  BUILD_RPATH "${torch_npu_root}/lib;${TORCH_INSTALL_PREFIX}/lib;${CANN_ROOT}/lib64")

# All host translation units share these contracts. Parse Torch/DeepJIT once;
# AICore JIT and the separate AICPU targets do not use the host PCH.
target_precompile_headers(dsv41_kernels PRIVATE native/common/torch.h native/common/jit.h)

foreach(op IN LISTS DSV41_COMPONENTS)
  dsv41_add_component("${op}")
endforeach()

install(TARGETS dsv41_kernels LIBRARY DESTINATION lib)
