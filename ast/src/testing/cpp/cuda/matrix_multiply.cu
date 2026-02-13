#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int m = 0; m < (N / TILE_WIDTH); ++m) {
        As[ty][tx] = A[row * N + (m * TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

__device__ void deviceMatrixElement(float *result, const float *A, const float *B, int row, int col, int N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    *result = sum;
}
