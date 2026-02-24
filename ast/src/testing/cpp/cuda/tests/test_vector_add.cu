#include <cassert>
#include <cuda_runtime.h>

#define assert_eq(actual, expected) assert((actual) == (expected))

extern __host__ void hostLauncher(const float *h_A, const float *h_B, float *h_C, int n);

void test_host_launcher_vector_add() {
    const int n = 4;
    float h_A[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[n] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C[n] = {0.0f, 0.0f, 0.0f, 0.0f};

    hostLauncher(h_A, h_B, h_C, n);

    assert_eq(h_C[0], 6.0f);
    assert_eq(h_C[1], 8.0f);
    assert_eq(h_C[2], 10.0f);
    assert_eq(h_C[3], 12.0f);
}

void test_host_launcher_zeroed_output() {
    const int n = 2;
    float h_A[n] = {0.0f, 0.0f};
    float h_B[n] = {0.0f, 0.0f};
    float h_C[n] = {1.0f, 1.0f};

    hostLauncher(h_A, h_B, h_C, n);

    assert_eq(h_C[0], 0.0f);
    assert_eq(h_C[1], 0.0f);
}

int main() {
    test_host_launcher_vector_add();
    test_host_launcher_zeroed_output();
    return 0;
}
