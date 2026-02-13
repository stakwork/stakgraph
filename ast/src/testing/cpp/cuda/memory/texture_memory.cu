#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

// Declare texture
texture<float, 1> textureRef;

/// @brief Kernel using texture memory for read-only cached access
__global__ void textureMemoryKernel(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Texture fetch - read-only cache optimized
        float val = tex1Dfetch(textureRef, idx);
        output[idx] = val * 2.0f;
    }
}

/// @brief Host launcher for texture memory operations
__host__ void hostTextureMemoryOps(const float *h_input, float *h_output,
                                    float *d_input, float *d_output, int n) {
    size_t size = n * sizeof(float);
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Bind texture to device memory
    CHECK_CUDA_ERROR(cudaBindTexture(NULL, textureRef, d_input, size));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    textureMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Unbind texture
    CHECK_CUDA_ERROR(cudaUnbindTexture(textureRef));
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
}

/// @brief Query texture cache properties
__host__ void queryTextureCacheProperties() {
    cudaDeviceProp props;
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, device));
    
    printf("Texture cache properties:\n");
    printf("  L2 cache size: %d KB\n", props.l2CacheSize / 1024);
}

/// @brief Surface operations for write access (alternative to texture)
__global__ void surfaceMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Surface memory allows both read and write with some caching
        data[idx] = data[idx] + 1.0f;
    }
}
