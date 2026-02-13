#include <cuda_runtime.h>
#include "../common/helper_cuda.h"
#include "../common/helper_math.h"

#define TILE_WIDTH 16

/// @brief Tiled matrix multiplication kernel with shared memory optimization
/// @param A device pointer to matrix A (M x K)
/// @param B device pointer to matrix B (K x N)
/// @param C device pointer to result matrix C (M x N)
/// @param M rows of A and C
/// @param N columns of B and C  
/// @param K columns of A and rows of B
__global__ void matrixMulTiled(const float *A, const float *B, float *C, 
                                int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float Cvalue = 0.0f;
    
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_WIDTH + ty < K) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

/// @brief Simple matrix multiplication without optimization
__global__ void matrixMulSimple(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/// @brief Device function for element calculation
__device__ float deviceMatrixElement(float a, float b) {
    return a * b;
}

/// @brief Host launcher for tiled matrix multiplication
__host__ void hostLaunchMatrixMul(const float *h_A, const float *h_B, float *h_C,
                                   int M, int N, int K) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, sizeC));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}
