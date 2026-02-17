#ifndef HELPER_MATH_H
#define HELPER_MATH_H

/// @brief Vector operations for CUDA
namespace cuda_math {

// Float2 operations
__device__ __host__ inline float2 make_float2(float x, float y) {
    return {x, y};
}

// Float3 operations
__device__ __host__ inline float3 make_float3(float x, float y, float z) {
    return {x, y, z};
}

// Float4 operations
__device__ __host__ inline float4 make_float4(float x, float y, float z, float w) {
    return {x, y, z, w};
}

// Magnitude calculation
__device__ __host__ inline float magnitude(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Dot product
__device__ __host__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product
__device__ __host__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Matrix transpose (4x4)
__device__ __host__ inline void transpose4x4(float* m) {
    float temp;
    temp = m[1]; m[1] = m[4]; m[4] = temp;
    temp = m[2]; m[2] = m[8]; m[8] = temp;
    temp = m[3]; m[3] = m[12]; m[12] = temp;
    temp = m[6]; m[6] = m[9]; m[9] = temp;
    temp = m[7]; m[7] = m[13]; m[13] = temp;
    temp = m[11]; m[11] = m[14]; m[14] = temp;
}

// MAtrix multiply (3x3)
__device__ __host__ inline void matmul3x3(float* A, float* B, float* C) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i*3+j] = 0;
            for (int k = 0; k < 3; k++) {
                C[i*3+j] += A[i*3+k] * B[k*3+j];
            }
        }
    }
}

} // namespace cuda_math

#endif // HELPER_MATH_H
