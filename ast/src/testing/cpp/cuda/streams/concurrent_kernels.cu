#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Kernel for concurrent execution
__global__ void concurrentKernel(float *data, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * multiplier;
    }
}

/// @brief Make multiple kernels run concurrently on the same stream
__host__ void hostConcurrentKernels(float *d_data, int n) {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Multiple kernels on same stream - execute sequentially but with overlap potential
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n, 2.0f);
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n, 3.0f);
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n, 4.0f);
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

/// @brief Multiple independent streams for true concurrency
__host__ void hostIndependentStreams(float *d_dataA, float *d_dataB, float *d_dataC, int n) {
    cudaStream_t streamA, streamB, streamC;
    CHECK_CUDA_ERROR(cudaStreamCreate(&streamA));
    CHECK_CUDA_ERROR(cudaStreamCreate(&streamB));
    CHECK_CUDA_ERROR(cudaStreamCreate(&streamC));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch independent kernels on different streams - can execute concurrently
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(d_dataA, n, 2.0f);
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(d_dataB, n, 3.0f);
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, streamC>>>(d_dataC, n, 4.0f);
    
    // Wait for all streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(streamA));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(streamB));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(streamC));
    
    CHECK_CUDA_ERROR(cudaStreamDestroy(streamA));
    CHECK_CUDA_ERROR(cudaStreamDestroy(streamB));
    CHECK_CUDA_ERROR(cudaStreamDestroy(streamC));
}

/// @brief Stream dependencies using events
__host__ void hostStreamDependencies(float *d_data, int n) {
    cudaStream_t streamA, streamB;
    cudaEvent_t eventA;
    
    CHECK_CUDA_ERROR(cudaStreamCreate(&streamA));
    CHECK_CUDA_ERROR(cudaStreamCreate(&streamB));
    CHECK_CUDA_ERROR(cudaEventCreate(&eventA));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel on stream A
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(d_data, n, 2.0f);
    
    // Record event in stream A
    CHECK_CUDA_ERROR(cudaEventRecord(eventA, streamA));
    
    // Make stream B wait for stream A
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(streamB, eventA, 0));
    
    // Launch kernel on stream B after stream A completes
    concurrentKernel<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(d_data, n, 3.0f);
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(streamB));
    CHECK_CUDA_ERROR(cudaEventDestroy(eventA));
    CHECK_CUDA_ERROR(cudaStreamDestroy(streamA));
    CHECK_CUDA_ERROR(cudaStreamDestroy(streamB));
}
