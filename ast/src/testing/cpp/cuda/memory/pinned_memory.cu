#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Kernel using pinned host memory for fast PCIe transfers
__global__ void pinnedMemoryKernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = input[idx] * 1.5f;
    }
}

/// @brief Host launcher for pinned memory operations
__host__ void hostPinnedMemoryOps(float *h_pinned_input, float *h_pinned_output,
                                   float *d_temp, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Device can DMA from pinned host memory directly
    CHECK_CUDA_ERROR(cudaMemcpy(d_temp, h_pinned_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    pinnedMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_temp, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Device can DMA to pinned host memory directly  
    CHECK_CUDA_ERROR(cudaMemcpy(h_pinned_output, d_temp, n * sizeof(float), cudaMemcpyDeviceToHost));
}

/// @brief Allocate pinned (page-locked) host memory
float* allocatePinnedMemory(size_t bytes) {
    float *ptr = NULL;
    CHECK_CUDA_ERROR(cudaHostAlloc((void **)&ptr, bytes, cudaHostAllocDefault));
    return ptr;
}

/// @brief Free pinned host memory
void freePinnedMemory(float *ptr) {
    CHECK_CUDA_ERROR(cudaFreeHost(ptr));
}

/// @brief Measure PCIe bandwidth with pinned memory
__host__ float measurePCIeBandwidth(float *h_pinned_data, float *d_data, int size, int iterations) {
    cudaEvent_t start, stop;
    float msec = 0.0f;
    
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_pinned_data, size, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&msec, start, stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    float bandwidth = (size * iterations * 1e-6f) / (msec * 1e-3f); // GB/s
    return bandwidth;
}
