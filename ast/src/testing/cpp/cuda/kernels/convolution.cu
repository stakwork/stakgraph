#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

#define KERNEL_RADIUS 1
#define TILE_WIDTH 16

/// @brief 2D convolution kernel
/// @param input input image
/// @param output output image
/// @param width image width
/// @param height image height
/// @param kernel convolution kernel
/// @param kernelSize size of kernel (3x3, 5x5, etc)
__global__ void convolve2D(const float *input, float *output, 
                            int width, int height, 
                            const float *kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int radius = kernelSize / 2;
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int py = y + ky;
            int px = x + kx;
            
            if (px >= 0 && px < width && py >= 0 && py < height) {
                sum += input[py * width + px] * kernel[(ky + radius) * kernelSize + (kx + radius)];
            }
        }
    }
    
    output[y * width + x] = sum;
}

/// @brief Separable convolution kernel (horizontal)
__global__ void convolve1DH(const float *input, float *output,
                             int width, int height,
                             const float *kernel, int kernelSize) {
    __shared__ float sdata[TILE_WIDTH + 2 * KERNEL_RADIUS];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    
    if (y >= height) return;
    
    // Load shared memory with halo
    if (tx < KERNEL_RADIUS && x >= KERNEL_RADIUS) {
        sdata[tx] = input[y * width + (x - KERNEL_RADIUS)];
    }
    sdata[tx + KERNEL_RADIUS] = (x < width) ? input[y * width + x] : 0.0f;
    if (tx < KERNEL_RADIUS && x + TILE_WIDTH < width) {
        sdata[tx + TILE_WIDTH + KERNEL_RADIUS] = input[y * width + (x + TILE_WIDTH)];
    }
    __syncthreads();
    
    if (x < width) {
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; ++k) {
            sum += sdata[tx + k] * kernel[k];
        }
        output[y * width + x] = sum;
    }
}

/// @brief Device function for convolution element
__device__ float deviceConvolveElement(float input, float kernel) {
    return input * kernel;
}

/// @brief Host launcher for 2D convolution
__host__ void hostConvolve2D(const float *h_input, float *h_output,
                              int width, int height,
                              const float *h_kernel, int kernelSize) {
    size_t input_size = width * height * sizeof(float);
    size_t kernel_size = kernelSize * kernelSize * sizeof(float);
    size_t output_size = width * height * sizeof(float);
    
    float *d_input = NULL, *d_output = NULL, *d_kernel = NULL;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, output_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_kernel, kernel_size));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                  (height + blockDim.y - 1) / blockDim.y);
    
    convolve2D<<<gridDim, blockDim>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
}
