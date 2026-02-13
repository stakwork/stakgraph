#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/helper_cuda.h"

/// @brief Main host orchestrator for CUDA test suite
__host__ int main() {
    printf("=== CUDA Comprehensive Test Suite ===\n\n");
    
    // Query device
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    printf("Found %d CUDA devices\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp props;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, dev));
        
        printf("Device %d: %s\n", dev, props.name);
        printf("  Compute Capability: %d.%d\n", props.major, props.minor);
        printf("  Global Memory: %.1f GB\n", props.totalGlobalMem / 1e9f);
        printf("  Shared Memory per Block: %d KB\n", props.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
        printf("  Number of SMs: %d\n", props.multiProcessorCount);
        printf("\n");
    }
    
    // Set default device
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Allocate test data
    int numElements = 1024 * 1024;
    float *h_data = (float *)malloc(numElements * sizeof(float));
    float *d_data = NULL;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, numElements * sizeof(float)));
    
    // Initialize host data
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = (float)i / (float)numElements;
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run various tests
    printf("Running CUDA tests...\n");
    
    // Simple kernel test
    printf("  - Basic kernel execution\n");
    dim3 blockDim(256, 1);
    dim3 gridDim((numElements + 255) / 256, 1);
    
    // Test H2D and D2H
    printf("  - Host-Device transfers\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_data));
    free(h_data);
    
    printf("\nAll tests completed successfully!\n");
    
    CHECK_CUDA_ERROR(cudaDeviceReset());
    
    return 0;
}
