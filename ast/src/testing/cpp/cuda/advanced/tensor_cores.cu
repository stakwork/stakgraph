#include <cuda_runtime.h>
#include <mma.h>
#include "../common/helper_cuda.h"

using namespace nvcuda::wmma;

#define M 16
#define N 16
#define K 16

/// @brief Tensor Core matrix multiplication using WMMA
/// Computes D = A * B + C
__global__ void tensorCoreMatmul(half *A, half *B, float *C,
                                 int M_dim, int N_dim, int K_dim) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM >= (M_dim + M - 1) / M || warpN >= (N_dim + N - 1) / N) return;
    
    // Declare the wmma matrices
    fragment<matrix_a, M, N, K, half, row_major> a_frag;
    fragment<matrix_b, M, N, K, half, col_major> b_frag;
    fragment<accumulator, M, N, K, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    // Load and compute
    for (int k = 0; k < K_dim; k += K) {
        int aRow = warpM * M;
        int bCol = warpN * N;
        
        if (aRow < M_dim && bCol < N_dim) {
            // Load fragments from global memory
            load_matrix_sync(a_frag, A + aRow * K_dim + k, K_dim);
            load_matrix_sync(b_frag, B + k * N_dim + bCol, N_dim);
            
            // Matrix multiply-accumulate
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result
    int cRow = warpM * M;
    int cCol = warpN * N;
    
    if (cRow < M_dim && cCol < N_dim) {
        store_matrix_sync(C + cRow * N_dim + cCol, c_frag, N_dim, mem_row_major);
    }
}

/// @brief Tensor Core with TF32 precision
__global__ void tensorCoreTF32(float *A, float *B, float *C,
                               int M_dim, int N_dim, int K_dim) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    fragment<matrix_a, M, N, K, float, row_major> a_frag;
    fragment<matrix_b, M, N, K, float, col_major> b_frag;
    fragment<accumulator, M, N, K, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K_dim; k += K) {
        load_matrix_sync(a_frag, A, K_dim);
        load_matrix_sync(b_frag, B, N_dim);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
}

/// @brief Host launcher for tensor core operations  
__host__ void hostTensorCoreMatmul(half *h_A, half *h_B, float *h_C,
                                    int M_dim, int N_dim, int K_dim) {
    half *d_A = NULL, *d_B = NULL;
    float *d_C = NULL;
    
    size_t sizeA = M_dim * K_dim * sizeof(half);
    size_t sizeB = K_dim * N_dim * sizeof(half);
    size_t sizeC = M_dim * N_dim * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, sizeC));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    
    dim3 blockDim(128, 1);
    dim3 gridDim((M_dim + M - 1) / M, (N_dim + N - 1) / N);
    
    tensorCoreMatmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_dim, N_dim, K_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

/// @brief Query tensor core support
__host__ void queryTensorCoreSupport() {
    int device;
    cudaDeviceProp props;
    
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, device));
    
    printf("Tensor Cores: %d\n", props.multiProcessorCount * 8);
}
