#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Kernel demonstrating shared memory usage
/// @brief Block-level cooperative reduction using shared memory
__global__ void sharedMemoryReduction(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduce shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/// @brief Shared memory with bank conflict avoidance
__global__ void sharedMemoryNoBankConflicts(float *output, int n) {
    __shared__ float sdata[256 + 1]; // +1 to avoid bank conflicts
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        sdata[tid + 1] = (float)idx; // Offset by 1 to avoid conflicts
        __syncthreads();
        
        if (tid > 0) {
            output[idx] = sdata[tid] + sdata[tid + 1];
        }
    }
}

/// @brief Constant memory usage for small data
__constant__ float constData[256];

/// @brief Kernel using constant memory
__global__ void constantMemoryKernel(const float *input, float *output, int n, int constSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < constSize; ++i) {
            sum += input[idx] * constData[i];
        }
        output[idx] = sum;
    }
}

/// @brief Copy data to constant memory
__host__ void copyToConstantMemory(const float *h_data, int size) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(constData, h_data, size * sizeof(float)));
}

/// @brief Host launcher for shared memory operations
__host__ void hostSharedMemoryOps(const float *h_input, float *h_output, int n) {
    float *d_input = NULL, *d_output = NULL;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int smemSize = threadsPerBlock * sizeof(float);
    
    // Launch reduction kernel with dynamic shared memory
    sharedMemoryReduction<<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_input, d_output, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

/// @brief Shared memory declaration for inter-thread communication
__shared__ int sharedCounter;
