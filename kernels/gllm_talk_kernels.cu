// =============================================================================
// Readable CUDA C++ Reference Mirror for kernels/gllm_talk_kernels.ptx
// Parallel 128^3 Voxel Fluid Splitting, Recombination & Sponge Carving Kernels
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct alignas(16) TalkPointVertex {
    float4 pos_val;
    float4 color;
};

struct alignas(16) CarveSeg {
    float4 p0_r;     // x0, y0, z0, radius
    float4 p1_d;     // x1, y1, z1, delta_porosity
    float4 dir_role; // dx, dy, dz, role_sign (-1 = question, +1 = answer)
};

struct alignas(16) FluidDrop {
    float4   pos_mass;   // x, y, z, fluid_mass
    uint32_t source_bit; // bitmask of injecting question content word
    uint32_t pad0;
    float    pad1, pad2;
};

extern "C" __global__ void step_fluid_split_recombine(
    const float4* __restrict__ sponge,   // {porosity, fx, fy, fz}
    const float*  __restrict__ fluid_in,
    const uint32_t* __restrict__ fmask_in,
    float*    __restrict__ fluid_out,
    uint32_t* __restrict__ fmask_out,
    uint32_t* __restrict__ stats,        // [0]=active, [1]=forks, [2]=collisions, [3]=peak*1000
    uint32_t grid
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t g2 = grid * grid;
    if (idx >= g2 * grid) return;

    float vol = fluid_in[idx];
    if (vol < 0.0015f) return;

    uint32_t mask = fmask_in[idx];
    atomicAdd(&stats[0], 1u);
    if (__popc(mask) >= 2) atomicAdd(&stats[2], 1u);
    atomicMax(&stats[3], static_cast<uint32_t>(rintf(vol * 1000.0f)));

    float4 sp = sponge[idx];
    float inv_f = rsqrtf(sp.y * sp.y + sp.z * sp.z + sp.w * sp.w + 1e-6f);
    float fx = sp.y * inv_f, fy = sp.z * inv_f, fz = sp.w * inv_f;

    uint32_t iz = idx / g2, rem = idx % g2, iy = rem / grid, ix = rem % grid;
    float w[6] = {};
    uint32_t nb[6] = {};
    uint32_t deg = 0;

    auto eval_nb = [&](int slot, bool ok, uint32_t nidx, float dir_dot) {
        if (!ok) return;
        float por = sponge[nidx].x;
        if (por > 0.02f) {
            w[slot] = por * (0.35f + 0.65f * fmaxf(dir_dot, 0.0f));
            nb[slot] = nidx;
            deg++;
        }
    };
    eval_nb(0, ix + 1 < grid, idx + 1,  fx);
    eval_nb(1, ix > 0,        idx - 1, -fx);
    eval_nb(2, iy + 1 < grid, idx + grid,  fy);
    eval_nb(3, iy > 0,        idx - grid, -fy);
    eval_nb(4, iz + 1 < grid, idx + g2,  fz);
    eval_nb(5, iz > 0,        idx - g2, -fz);

    if (deg >= 2) atomicAdd(&stats[1], 1u);
    float total_w = w[0] + w[1] + w[2] + w[3] + w[4] + w[5];

    atomicAdd(&fluid_out[idx], vol * (total_w > 1e-4f ? 0.36f : 0.88f));
    atomicOr(&fmask_out[idx], mask);
    if (total_w <= 1e-4f) return;

    float scale = (0.62f * vol) / total_w;
    for (int s = 0; s < 6; ++s) {
        if (w[s] > 0.0f) {
            atomicAdd(&fluid_out[nb[s]], w[s] * scale);
            atomicOr(&fmask_out[nb[s]], mask);
        }
    }
}
