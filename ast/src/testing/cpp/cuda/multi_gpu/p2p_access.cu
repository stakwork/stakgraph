#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief Check if GPU supports peer-to-peer access
__host__ bool canAccessPeer(int device0, int device1) {
    int can_access = 0;
    CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&can_access, device0, device1));
    return can_access != 0;
}

/// @brief Enable peer-to-peer access between two GPUs
__host__ void enablePeerAccess(int device0, int device1) {
    if (canAccessPeer(device0, device1)) {
        CHECK_CUDA_ERROR(cudaSetDevice(device0));
        CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(device1, 0));
    }
}

/// @brief P2P transfer kernel - direct GPU to GPU memory access
__global__ void p2pTransferKernel(const float *src, float *dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Direct peer-to-peer access - no host involvement
        dst[idx] = src[idx] * 2.0f;
    }
}

/// @brief Host launcher for peer-to-peer transfers
__host__ void hostP2PTransfer(float *h_data, int n, int srcDevice, int dstDevice) {
    float *d_src = NULL, *d_dst = NULL;
    size_t size = n * sizeof(float);
    
    // Allocate on source device
    CHECK_CUDA_ERROR(cudaSetDevice(srcDevice));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_src, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_data, size, cudaMemcpyHostToDevice));
    
    // Allocate on destination device
    CHECK_CUDA_ERROR(cudaSetDevice(dstDevice));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dst, size));
    
    // Enable peer access
    enablePeerAccess(dstDevice, srcDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Run kernel on destination device accessing source device memory
    p2pTransferKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    float *h_result = (float *)malloc(size);
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_dst, size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaSetDevice(srcDevice));
    CHECK_CUDA_ERROR(cudaFree(d_src));
    
    CHECK_CUDA_ERROR(cudaSetDevice(dstDevice));
    CHECK_CUDA_ERROR(cudaFree(d_dst));
    
    free(h_result);
}

/// @brief Query maximum theoretical P2P bandwidth
__host__ int getMaxP2PBandwidth(int srcDevice, int dstDevice) {
    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, srcDevice));
    
    // Theoretical bandwidth depends on PCIe or NVLink
    return 0; // Implementation depends on hardware
}
