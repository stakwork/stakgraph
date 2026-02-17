#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Simple async kernel
__global__ void asyncKernel(float *data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] + (float)value;
    }
}

/// @brief Async memory operations using streams
__host__ void hostAsyncOperations(float *h_data, float *d_data, int n, int numStreams) {
    cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));
    
    // Create streams
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    int elementsPerStream = n / numStreams;
    
    // Async operations on each stream
    for (int s = 0; s < numStreams; ++s) {
        int offset = s * elementsPerStream;
        int count = (s == numStreams - 1) ? (n - offset) : elementsPerStream;
        size_t size = count * sizeof(float);
        
        // Async H2D copy
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data + offset, h_data + offset, size, 
                                         cudaMemcpyHostToDevice, streams[s]));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
        
        // Kernel on stream
        asyncKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(d_data + offset, count, s);
        
        // Async D2H copy
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data + offset, d_data + offset, size,
                                         cudaMemcpyDeviceToHost, streams[s]));
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
    
    free(streams);
}

/// @brief Record events in stream for synchronization
__host__ void hostStreamEvents(float *d_data, int n) {
    cudaStream_t stream;
    cudaEvent_t event1, event2;
    
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaEventCreate(&event1));
    CHECK_CUDA_ERROR(cudaEventCreate(&event2));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Record event
    CHECK_CUDA_ERROR(cudaEventRecord(event1, stream));
    
    // Kernel
    asyncKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n, 1);
    
    // Record another event
    CHECK_CUDA_ERROR(cudaEventRecord(event2, stream));
    
    // Wait for event
    CHECK_CUDA_ERROR(cudaEventSynchronize(event2));
    
    // Query elapsed time between events
    float elapsed = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed, event1, event2));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(event1));
    CHECK_CUDA_ERROR(cudaEventDestroy(event2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

/// @brief Stream query for non-blocking checks
__host__ bool isStreamReady(cudaStream_t stream) {
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess) {
        return true;
    } else if (err == cudaErrorNotReady) {
        return false;
    } else {
        CHECK_CUDA_ERROR(err);
        return false;
    }
}
