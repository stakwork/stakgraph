cmake_minimum_required(VERSION 3.10)
project(stakgraph_cuda_test_suite LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CUDA_INCLUDE_DIRS}
)

# Common utilities
set(COMMON_SOURCES
    common/helper_cuda.h
    common/helper_math.h
)

# Kernel sources
set(KERNEL_SOURCES
    kernels/vector_add.cu
    kernels/matrix_multiply.cu
    kernels/reduction.cu
    kernels/convolution.cu
    kernels/tensor_ops.cu
)

# Memory sources
set(MEMORY_SOURCES
    memory/unified_memory.cu
    memory/pinned_memory.cu
    memory/texture_memory.cu
    memory/shared_memory.cu
)

# Multi-GPU sources
set(MULTI_GPU_SOURCES
    multi_gpu/p2p_access.cu
    multi_gpu/multi_device.cu
    multi_gpu/ipc_memory.cu
)

# Streams sources
set(STREAMS_SOURCES
    streams/async_ops.cu
    streams/stream_priorities.cu
    streams/concurrent_kernels.cu
)

# Advanced sources
set(ADVANCED_SOURCES
    advanced/cooperative_groups.cu
    advanced/dynamic_parallelism.cu
    advanced/tensor_cores.cu
    advanced/cuda_graphs.cu
)

# Host sources
set(HOST_SOURCES
    host/main.cu
    host/device_query.cpp
    host/error_handling.cuh
)

# Combine all sources
set(ALL_CUDA_SOURCES
    ${COMMON_SOURCES}
    ${KERNEL_SOURCES}
    ${MEMORY_SOURCES}
    ${MULTI_GPU_SOURCES}
    ${STREAMS_SOURCES}
    ${ADVANCED_SOURCES}
    ${HOST_SOURCES}
)

# Create CUDA test executable
add_executable(cuda_test_suite ${ALL_CUDA_SOURCES})

# Link CUDA libraries
target_link_libraries(cuda_test_suite
    ${CUDA_LIBRARIES}
    pthread
)

# Enable RDC (Relocatable Device Code) for dynamic parallelism
set_target_properties(cuda_test_suite PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
