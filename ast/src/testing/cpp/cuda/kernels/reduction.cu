#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Parallel reduction kernel for sum
/// @param g_idata input data
/// @param g_odata output data
/// @param n number of elements
__global__ void reduce_sum(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/// @brief Parallel reduction kernel for max value
__global__ void reduce_max(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : -FLT_MAX;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/// @brief Device function for reduction operation
__device__ float deviceReduceElement(float a, float b) {
    return a + b;
}

/// @brief Host launcher for reduction operation
__host__ float hostReduceSum(float *h_data, int n) {
    float *d_data = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int smemSize = threadsPerBlock * sizeof(float);
    
    float *d_result = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_result, blocksPerGrid * sizeof(float)));
    
    reduce_sum<<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_data, d_result, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    float *h_result = (float*)malloc(blocksPerGrid * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        sum += h_result[i];
    }
    
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    free(h_result);
    
    return sum;
}
