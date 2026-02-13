#include <cuda_runtime.h>
#include "../common/helper_cuda.h"
#include "../common/helper_math.h"

/// @brief Tensor transpose kernel (for 3D tensors)
/// @param input input tensor
/// @param output output tensor
/// @param dims dimensions [d0, d1, d2]
/// @param perm permutation [p0, p1, p2]
__global__ void tensorTranspose(const float *input, float *output,
                                 int d0, int d1, int d2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d0 * d1 * d2;
    
    if (idx >= n) return;
    
    int i0 = idx / (d1 * d2);
    int i1 = (idx / d2) % d1;
    int i2 = idx % d2;
    
    int out_idx = i2 * (d0 * d1) + i1 * d0 + i0;
    output[out_idx] = input[idx];
}

/// @brief Tensor broadcast kernel
/// @param input input tensor
/// @param output output tensor
/// @param elem_count total elements after broadcast
__global__ void tensorBroadcast(const float *input, float *output,
                                 int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    output[idx] = input[idx % input_size];
}

/// @brief Tensor element-wise multiplication
__global__ void tensorElementMul(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    C[idx] = A[idx] * B[idx];
}

/// @brief Tensor summation along axis
__global__ void tensorSumAxis(const float *input, float *output,
                               int dim0, int dim1, int dim2, int axis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (axis == 0) {
        if (idx >= dim1 * dim2) return;
        float sum = 0.0f;
        for (int i = 0; i < dim0; ++i) {
            sum += input[i * dim1 * dim2 + idx];
        }
        output[idx] = sum;
    } else if (axis == 1) {
        if (idx >= dim0 * dim2) return;
        float sum = 0.0f;
        for (int j = 0; j < dim1; ++j) {
            sum += input[(idx / dim2) * dim1 * dim2 + j * dim2 + (idx % dim2)];
        }
        output[idx] = sum;
    }
}

/// @brief Device function for tensor operation
__device__ float deviceTensorOp(float a, float b) {
    return a + b;
}

/// @brief Host launcher for tensor transpose
__host__ void hostTensorTranspose(const float *h_input, float *h_output,
                                   int d0, int d1, int d2) {
    size_t size = d0 * d1 * d2 * sizeof(float);
    float *d_input = NULL, *d_output = NULL;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (d0 * d1 * d2 + threadsPerBlock - 1) / threadsPerBlock;
    
    tensorTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d0, d1, d2);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}
