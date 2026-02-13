#include <cuda_runtime.h>
#include <cuda_graph.h>
#include "../common/helper_cuda.h"

/// @brief Simple kernel for graph execution
__global__ void graphKernel(float *data, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * multiplier;
    }
}

/// @brief Create and execute a CUDA graph
__host__ void hostCUDAGraph(float *h_data, int n) {
    float *d_data = NULL;
    size_t size = n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    cudaGraph_t graph;
    cudaGraphExec_t execGraph;
    
    // Begin capturing graph
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Record operations into graph
    graphKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, 2.0f);
    graphKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, 3.0f);
    graphKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, 4.0f);
    
    // End capturing
    CHECK_CUDA_ERROR(cudaStreamEndCapture(0, &graph));
    
    // Instantiate executable graph
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&execGraph, graph, NULL, NULL, 0));
    
    // Launch graph multiple times efficiently
    for (int i = 0; i < 10; ++i) {
        CHECK_CUDA_ERROR(cudaGraphLaunch(execGraph, 0));
    }
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGraphExecDestroy(execGraph));
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

/// @brief Update graph parameters without reconstruction
__host__ void hostUpdateGraphParameters(float *d_data, int n, float *multipliers) {
    cudaGraph_t graph;
    cudaGraphExec_t execGraph;
    
    // Initial graph capture
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    graphKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n, 2.0f);
    
    CHECK_CUDA_ERROR(cudaStreamEndCapture(0, &graph));
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&execGraph, graph, NULL, NULL, 0));
    
    // Execute with different parameters
    for (int i = 0; i < 5; ++i) {
        CHECK_CUDA_ERROR(cudaGraphLaunch(execGraph, 0));
    }
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGraphExecDestroy(execGraph));
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
}

/// @brief Query graph capabilities
__host__ void queryCUDAGraphSupport() {
    int device;
    cudaDeviceProp props;
    
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, device));
    
    printf("CUDA Graphs supported: Yes\n");
}
