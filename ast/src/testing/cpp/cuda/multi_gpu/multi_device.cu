#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/helper_cuda.h"

/// @brief Multi-device kernel launcher
__global__ void multiDeviceKernel(float *data, int n, int deviceId) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Device-specific computation
        data[idx] = data[idx] + (float)deviceId;
    }
}

/// @brief Distribute work across multiple GPUs
__host__ void distributeWorkMultiGPU(float *h_data, int n, int numDevices) {
    int elementsPerDevice = n / numDevices;
    float **d_data = (float **)malloc(numDevices * sizeof(float *));
    
    // Allocate and copy data to each device
    for (int dev = 0; dev < numDevices; ++dev) {
        CHECK_CUDA_ERROR(cudaSetDevice(dev));
        
        int start = dev * elementsPerDevice;
        int count = (dev == numDevices - 1) ? (n - start) : elementsPerDevice;
        size_t size = count * sizeof(float);
        
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data[dev], size));
        CHECK_CUDA_ERROR(cudaMemcpy(d_data[dev], h_data + start, size, cudaMemcpyHostToDevice));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
        
        multiDeviceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data[dev], count, dev);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Copy result back
        CHECK_CUDA_ERROR(cudaMemcpy(h_data + start, d_data[dev], size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaFree(d_data[dev]));
    }
    
    free(d_data);
}

/// @brief Get number of available GPUs
__host__ int getNumAvailableGPUs() {
    int count = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&count));
    return count;
}

/// @brief Synchronize all devices
__host__ void synchronizeAllDevices(int numDevices) {
    for (int dev = 0; dev < numDevices; ++dev) {
        CHECK_CUDA_ERROR(cudaSetDevice(dev));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

/// @brief Query device topology
__host__ void queryDeviceTopology(int numDevices) {
    for (int dev = 0; dev < numDevices; ++dev) {
        printf("Device %d:\n", dev);
        
        // Check connectivity to other devices
        for (int other = 0; other < numDevices; ++other) {
            if (dev != other) {
                int can_access = 0;
                CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&can_access, dev, other));
                printf("  P2P to device %d: %s\n", other, can_access ? "Yes" : "No");
            }
        }
    }
}
