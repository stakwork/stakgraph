#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Kernel using unified memory for direct access
__global__ void unifiedMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Direct access to unified memory - automatically migrated by GPU
        data[idx] = data[idx] * 2.0f;
    }
}

/// @brief Query unified memory page migration
__global__ void queryUnifiedMemoryUsage(float *data, int n, int *access_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Access unified memory - GPU automatically pages data
        atomicAdd(access_count, 1);
        data[idx] = data[idx] + 1.0f;
    }
}

/// @brief Prefetch unified memory to device
__host__ void prefetchUnifiedMemoryToDevice(float *um_data, int n, int device) {
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(um_data, n * sizeof(float), device));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

/// @brief Host launcher for unified memory operations
__host__ void hostUnifiedMemoryOps(float *um_data, int n) {
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Prefetch to device for better performance
    prefetchUnifiedMemoryToDevice(um_data, n, device);
    
    unifiedMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(um_data, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Prefetch back to host for CPU access
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(um_data, n * sizeof(float), cudaCpuDeviceId));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

/// @brief Allocate unified memory
float* allocateUnifiedMemory(size_t bytes) {
    float *ptr = NULL;
    CHECK_CUDA_ERROR(cudaMallocManaged((void **)&ptr, bytes));
    return ptr;
}

/// @brief Free unified memory
void freeUnifiedMemory(float *ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}
