#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Basic vector addition kernel
/// @param A input vector
/// @param B input vector  
/// @param C output vector
/// @param numElements number of elements
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

/// @brief Device-side helper function for vector operations
__device__ float deviceVectorElementAdd(float a, float b) {
    return a + b;
}

/// @brief Host-side launcher for vector addition
__host__ void hostLaunchVectorAdd(const float *h_A, const float *h_B, float *h_C, int n) {
    size_t size = n * sizeof(float);
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}
