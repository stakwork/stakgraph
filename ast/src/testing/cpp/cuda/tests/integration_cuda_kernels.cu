#include <cassert>
#include <cuda_runtime.h>

#define assert_eq(actual, expected) assert((actual) == (expected))

extern __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);
extern __global__ void matrixMul(const float *A, const float *B, float *C, int N);

void integration_test_vector_add_kernel() {
    const int n = 1;
    float h_A[n] = {2.0f};
    float h_B[n] = {3.0f};
    float h_C[n] = {0.0f};

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc((void **)&d_A, sizeof(float));
    cudaMalloc((void **)&d_B, sizeof(float));
    cudaMalloc((void **)&d_C, sizeof(float));

    cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float), cudaMemcpyHostToDevice);

    vectorAdd<<<1, 1>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    assert_eq(h_C[0], 5.0f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void integration_test_matrix_mul_kernel() {
    const int n = 1;
    float h_A[n] = {4.0f};
    float h_B[n] = {5.0f};
    float h_C[n] = {0.0f};

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc((void **)&d_A, sizeof(float));
    cudaMalloc((void **)&d_B, sizeof(float));
    cudaMalloc((void **)&d_C, sizeof(float));

    cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(1, 1);
    dim3 block(1, 1);
    matrixMul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    assert_eq(h_C[0], 20.0f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    integration_test_vector_add_kernel();
    integration_test_matrix_mul_kernel();
    return 0;
}
