#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Kernel with priority-sensitive operations
__global__ void priorityKernel(float *data, int n, int priority_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simulate work based on priority
        float result = 0.0f;
        for (int i = 0; i < (100 + priority_level * 10); ++i) {
            result += sinf((float)i);
        }
        data[idx] = data[idx] + result;
    }
}

/// @brief Create high and low priority streams
__host__ void hostStreamPriorities(float *d_data, int n, int numStreams) {
    // Get device stream priority range
    int priorityHigh = 0, priorityLow = 0;
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
    
    cudaStream_t *highPriorityStreams = (cudaStream_t *)malloc((numStreams/2) * sizeof(cudaStream_t));
    cudaStream_t *lowPriorityStreams = (cudaStream_t *)malloc((numStreams/2) * sizeof(cudaStream_t));
    
    // Create high priority streams
    for (int i = 0; i < numStreams/2; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&highPriorityStreams[i], 
                                                      cudaStreamNonBlocking, priorityHigh));
    }
    
    // Create low priority streams
    for (int i = 0; i < numStreams/2; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&lowPriorityStreams[i],
                                                      cudaStreamNonBlocking, priorityLow));
    }
    
    int elementsPerStream = n / numStreams;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (elementsPerStream + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernels on high priority streams
    for (int i = 0; i < numStreams/2; ++i) {
        int offset = i * elementsPerStream;
        priorityKernel<<<blocksPerGrid, threadsPerBlock, 0, highPriorityStreams[i]>>>(
            d_data + offset, elementsPerStream, 1);
    }
    
    // Launch kernels on low priority streams
    for (int i = 0; i < numStreams/2; ++i) {
        int offset = (numStreams/2 + i) * elementsPerStream;
        priorityKernel<<<blocksPerGrid, threadsPerBlock, 0, lowPriorityStreams[i]>>>(
            d_data + offset, elementsPerStream, 0);
    }
    
    // Synchronize high priority streams first
    for (int i = 0; i < numStreams/2; ++i) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(highPriorityStreams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(highPriorityStreams[i]));
    }
    
    for (int i = 0; i < numStreams/2; ++i) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(lowPriorityStreams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(lowPriorityStreams[i]));
    }
    
    free(highPriorityStreams);
    free(lowPriorityStreams);
}

/// @brief Query stream priority
__host__ int getStreamPriority(cudaStream_t stream) {
    int priority = 0;
    CHECK_CUDA_ERROR(cudaStreamGetPriority(stream, &priority));
    return priority;
}
