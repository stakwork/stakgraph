#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../common/helper_cuda.h"

namespace cg = cooperative_groups;

/// @brief Cooperative groups reduction kernel
__global__ void cooperativeGroupsReduction(const float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    cg::sync(block);
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(block);
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/// @brief Warp-level operations using cooperative groups
__global__ void warpLevelOperations(float *data, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // Warp-level reduction
        for (int i = 16; i >= 1; i /= 2) {
            val += tile.shfl_down(val, i);
        }
        
        if (tile.thread_rank() == 0) {
            data[blockIdx.x * 32 + tile.meta_group_rank()] = val;
        }
    }
}

/// @brief Tile-level synchronization
__global__ void tileLevelSync(float *data, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<8> tile = cg::tiled_partition<8>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // Tile-level synchronization and communication
        val = tile.shfl(val, (tile.thread_rank() + 1) % tile.size());
        
        if (tile.thread_rank() == 0) {
            data[idx] = val;
        }
        
        cg::sync(tile);
    }
}

/// @brief Host launcher for cooperative groups
__host__ void hostCooperativeGroups(float *h_input, float *h_output, int n) {
    float *d_input = NULL, *d_output = NULL;
    size_t size = n * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int smemSize = threadsPerBlock * sizeof(float);
    
    cooperativeGroupsReduction<<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_input, d_output, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}
