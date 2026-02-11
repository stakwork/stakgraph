#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/helper_cuda.h"

/// @brief Query and print all CUDA device properties
int queryAllDevices() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed\n");
        return -1;
    }
    
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, i);
        
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d\n", i);
            continue;
        }
        
        printf("=== Device %d: %s ===\n", i, props.name);
        printf("Driver Version: %.2f\n", props.driverVersion / 1000.0f);
        printf("Compute Capability: %d.%d\n", props.major, props.minor);
        printf("Total Global Memory: %.2f GB\n", (double)props.totalGlobalMem / 1e9);
        printf("Total Constant Memory: %d KB\n", props.totalConstMem / 1024);
        printf("Shared Memory per Block: %d KB\n", props.sharedMemPerBlock / 1024);
        printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
        printf("Block Dimensions: (%d, %d, %d)\n", props.maxThreadsDim[0], 
               props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("Grid Dimensions: (%d, %d, %d)\n", props.maxGridSize[0],
               props.maxGridSize[1], props.maxGridSize[2]);
        printf("Number of SMs: %d\n", props.multiProcessorCount);
        printf("Warp Size: %d\n", props.warpSize);
        printf("Memory Clock Rate: %.2f GHz\n", props.memoryClockRate / 1e6);
        printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
        printf("L2 Cache Size: %d KB\n", props.l2CacheSize / 1024);
        printf("ECC Enabled: %s\n", props.ECCEnabled ? "Yes" : "No");
        printf("Unified Addressing: %s\n", props.unifiedAddressing ? "Yes" : "No");
        printf("Concurrent Kernels: %s\n", props.concurrentKernels ? "Yes" : "No");
        printf("Async Engine Count: %d\n", props.asyncEngineCount);
        printf("\n");
    }
    
    return deviceCount;
}

/// @brief Get compute capability version
__host__ void getComputeCapabilityVersion(int deviceId) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, deviceId) == cudaSuccess) {
        printf("Device %d Compute Capability: %d.%d\n", deviceId, props.major, props.minor);
    }
}

/// @brief Measure memory bandwidth
__host__ float measureMemoryBandwidth() {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    int size = 100 * 1024 * 1024; // 100 MB
    float *d_data = NULL;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, size));
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemset(d_data, 0, size));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float bandwidth = (size / 1e9) / (milliseconds / 1000); // GB/s
    
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return bandwidth;
}
