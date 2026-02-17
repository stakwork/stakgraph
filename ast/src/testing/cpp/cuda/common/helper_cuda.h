#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d: %s\n", \
                #expr, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/// @brief Print basic CUDA device properties
void printDeviceProperties();

/// @brief Allocate pinned (page-locked) host memory
void* allocatePinnedMemory(size_t size);

/// @brief Free pinned memory
void freePinnedMemory(void* ptr);

/// @brief Query device compute capability
void getComputeCapability(int deviceId, int& major, int& minor);

/// @brief Synchronize device and check for errors
void deviceSyncAndCheck(const char* context);

#endif // HELPER_CUDA_H
