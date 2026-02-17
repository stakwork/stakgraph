#ifndef ERROR_HANDLING_CUH
#define ERROR_HANDLING_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/// @brief Safe CUDA memory allocation with error checking
__host__ inline void* safeCudaMalloc(size_t bytes) {
    void *ptr = NULL;
    cudaError_t err = cudaCalloc(&ptr, 1, bytes);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    
    return ptr;
}

/// @brief Safe CUDA memory free with error checking
__host__ inline cudaError_t safeCudaFree(void *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "Warning: Attempting to free NULL pointer\n");
        return cudaSuccess;
    }
    
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA free failed: %s\n", cudaGetErrorString(err));
    }
    
    return err;
}

/// @brief Safe synchronization with error reporting
__host__ inline cudaError_t safeCudaDeviceSynchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

/// @brief Safe stream synchronization
__host__ inline cudaError_t safeCudaStreamSynchronize(cudaStream_t stream) {
    cudaError_t err = cudaStreamSynchronize(stream);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Stream synchronization failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

/// @brief Get last error and reset
__host__ inline cudaError_t getAndResetLastError(const char *context) {
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", context, cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

/// @brief Kernel error wrapper
__host__ inline cudaError_t checkKernelLaunch(const char *kernel_name) {
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel '%s' launch failed: %s\n", kernel_name, cudaGetErrorString(err));
        return err;
    }
    
    // Optionally synchronize to catch errors quickly
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel '%s' execution failed: %s\n", kernel_name, cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

/// @brief Safe Memcpy with validation
__host__ inline cudaError_t safeMemcpy(void *dst, const void *src, size_t count,
                                       enum cudaMemcpyKind kind) {
    if (dst == NULL || src == NULL) {
        fprintf(stderr, "Memcpy error: NULL pointer detected\n");
        return cudaErrorInvalidValue;
    }
    
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess) {
        const char *kind_str = "";
        switch(kind) {
            case cudaMemcpyHostToDevice: kind_str = "H2D"; break;
            case cudaMemcpyDeviceToHost: kind_str = "D2H"; break;
            case cudaMemcpyDeviceToDevice: kind_str = "D2D"; break;
            default: kind_str = "Unknown";
        }
        
        fprintf(stderr, "Memcpy (%s) failed: %s\n", kind_str, cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

/// @brief Assert utility for CUDA code
#define CUDA_ASSERT(expr, msg) \
    if (!(expr)) { \
        fprintf(stderr, "Assertion failed: %s (%s)\n", #expr, msg); \
        exit(1); \
    }

#endif // ERROR_HANDLING_CUH
