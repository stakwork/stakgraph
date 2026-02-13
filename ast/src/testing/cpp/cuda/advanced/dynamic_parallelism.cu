#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Kernel using dynamic parallelism
/// Recursively spawns child kernels
__global__ void dynamicParallelismKernel(float *data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] + (float)depth;
        
        // Spawn child kernel at reduced depth if conditions met
        if (idx == 0 && depth > 0 && n > 1024) {
            dynamicParallelismKernel<<<1, 256>>>(data, n/2, depth - 1);
        }
    }
}

/// @brief Helper function to determine optimal grid/block configuration
__device__ void getOptimalLaunchConfig(int n, dim3& blockDim, dim3& gridDim) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    blockDim = dim3(threadsPerBlock, 1, 1);
    gridDim = dim3(blocksPerGrid, 1, 1);
}

/// @brief Grid-level kernel launched dynamically
__global__ void dynamicGridKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 1.5f;
    }
}

/// @brief Parent kernel that launches child kernels dynamically
__global__ void parentKernel(float *data, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Single thread launches child kernel
        dynamicGridKernel<<<(n + 255) / 256, 256>>>(data, n);
        
        // Wait for children
        cudaDeviceSynchronize();
    }
}

/// @brief Host interface for dynamic parallelism
__host__ void hostDynamicParallelism(float *h_data, int n) {
    float *d_data = NULL;
    size_t size = n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    // Root kernel spawns children dynamically
    parentKernel<<<1, 256>>>(d_data, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

/// @brief Get device properties related to dynamic parallelism
__host__ void queryDynamicParallelismSupport() {
    int device;
    cudaDeviceProp props;
    
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, device));
    
    printf("Dynamic Parallelism Support: %s\n", 
           props.deviceOverlap ? "Yes" : "No");
}
