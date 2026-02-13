#include <cuda_runtime.h>
#include "../common/helper_cuda.h"

/// @brief IPC memory handle structure
typedef struct {
    cudaIpcMemHandle_t handle;
    int device;
    size_t size;
} IPCMemoryInfo;

/// @brief Create IPC memory handle for sharing across processes
__host__ IPCMemoryInfo createIPCMemoryHandle(size_t size, int device) {
    CHECK_CUDA_ERROR(cudaSetDevice(device));
    
    float *d_ptr = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_ptr, size));
    
    cudaIpcMemHandle_t handle;
    CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&handle, d_ptr));
    
    IPCMemoryInfo info;
    info.handle = handle;
    info.device = device;
    info.size = size;
    
    return info;
}

/// @brief Open IPC memory handle in another process
__host__ float* openIPCMemoryHandle(const IPCMemoryInfo& info) {
    CHECK_CUDA_ERROR(cudaSetDevice(info.device));
    
    float *d_ptr = NULL;
    CHECK_CUDA_ERROR(cudaIpcOpenMemHandle((void **)&d_ptr, info.handle, cudaIpcMemLazyEnablePeerAccess));
    
    return d_ptr;
}

/// @brief Close IPC memory handle
__host__ void closeIPCMemoryHandle(float *d_ptr) {
    CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(d_ptr));
}

/// @brief Kernel for IPC memory operations
__global__ void ipcMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Access memory shared via IPC
        data[idx] = data[idx] * 2.0f;
    }
}

/// @brief Create IPC event handle for synchronization
__host__ cudaIpcEventHandle_t createIPCEventHandle() {
    cudaEvent_t event;
    CHECK_CUDA_ERROR(cudaEventCreate(&event));
    
    cudaIpcEventHandle_t handle;
    CHECK_CUDA_ERROR(cudaIpcGetEventHandle(&handle, event));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(event));
    
    return handle;
}

/// @brief Open IPC event handle
__host__ cudaEvent_t openIPCEventHandle(const cudaIpcEventHandle_t& handle) {
    cudaEvent_t event;
    CHECK_CUDA_ERROR(cudaIpcOpenEventHandle(&event, handle));
    return event;
}
