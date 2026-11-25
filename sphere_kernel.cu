// =============================================================================
// CUDA C++ Reference Source for sphere_kernel.ptx
// Generates and animates >= 1,000,000 points strictly on the outer layer of a
// 3D sphere inside a mapped OpenGL Vertex Buffer Object (zero-copy interop).
//
// If nvcc is installed, you can recompile this file to PTX with:
//   nvcc -ptx -arch=sm_86 -O3 sphere_kernel.cu -o sphere_kernel.ptx
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct alignas(16) PointVertex {
    float4 pos_wave; // x, y, z, wave_norm
    float4 color;    // r, g, b, alpha
};

__device__ __forceinline__ float pcg_uniform_f32(uint32_t seed) {
    uint32_t state = seed * 747796405u + 2891336453u;
    uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint32_t h     = (word >> 22u) ^ word;
    return static_cast<float>(h >> 8u) * (1.0f / 16777216.0f);
}

extern "C" __global__ void update_sphere_points(
    PointVertex* __restrict__ out_ptr,
    uint32_t count,
    float time_sec,
    uint32_t mode,
    uint32_t palette
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // 1. Exact uniform area sampling on the unit outer sphere S^2 (no moire)
    const float u1    = pcg_uniform_f32(idx * 2u + 1u);
    const float u2    = pcg_uniform_f32(idx * 2u + 2u);
    const float u3    = pcg_uniform_f32(idx * 2u + 3u);

    const float ny     = 2.0f * u1 - 1.0f;
    const float theta  = 6.28318530718f * u2;
    const float r_perp = sqrtf(fmaxf(0.0f, 1.0f - ny * ny));
    const float nx     = cosf(theta) * r_perp;
    const float nz     = sinf(theta) * r_perp;

    // 2. Outer layer radial profile
    float radius    = 1.0f;
    float wave_norm = 0.5f;

    if (mode == 0u) {
        // Pure unit outer sphere (exact R = 1.0)
        radius    = 1.0f;
        wave_norm = 0.5f + 0.5f * sinf(5.0f * ny + 3.0f * nx - 2.0f * time_sec);
    } else if (mode == 1u) {
        // Harmonic surface ripples across the outer spherical shell
        float a1  = 6.0f * nx + 5.0f * ny - 2.2f * time_sec;
        float a2  = 7.0f * nz - 4.0f * ny + 1.7f * time_sec;
        float a3  = 11.0f * ny + 3.0f * time_sec;
        float w   = 0.5f * sinf(a1) * cosf(a2) + 0.25f * sinf(a3);
        radius    = 1.0f + 0.030f * w;
        wave_norm = fminf(fmaxf(0.5f + 0.65f * w, 0.0f), 1.0f);
    } else {
        // Thin multi-micro-layer outer shell crust
        float shim = sinf(8.0f * nx + 2.0f * time_sec) * u3;
        radius     = 0.975f + 0.030f * u3 + 0.012f * shim;
        wave_norm  = u3;
    }

    // 3. Per-point color
    const float lat = 0.5f * ny + 0.5f;
    float r = 1.0f, g = 1.0f, b = 1.0f;

    if (palette == 0u) {
        r = 0.08f + 0.55f * (1.0f - lat) + 0.35f * wave_norm;
        g = 0.12f + 0.65f * lat + 0.35f * wave_norm;
        b = 0.75f + 0.25f * wave_norm;
    } else if (palette == 1u) {
        r = 0.08f + 0.90f * wave_norm;
        g = 0.15f + 0.65f * wave_norm + 0.20f * lat;
        b = 0.20f + 0.55f * (1.0f - wave_norm);
    } else if (palette == 2u) {
        r = 0.15f + 0.45f * wave_norm;
        g = 0.65f + 0.35f * wave_norm;
        b = 0.85f + 0.15f * lat;
    } else {
        r = 0.5f * nx + 0.5f;
        g = 0.5f * ny + 0.5f;
        b = 0.5f * nz + 0.5f;
    }

    PointVertex v;
    v.pos_wave = make_float4(nx * radius, ny * radius, nz * radius, wave_norm);
    v.color    = make_float4(fminf(r, 1.0f), fminf(g, 1.0f), fminf(b, 1.0f), 1.0f);
    out_ptr[idx] = v;
}
