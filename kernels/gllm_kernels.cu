// =============================================================================
// CUDA C++ Reference Source for kernels/gllm_kernels.ptx
// Generalized Geometry-Based Large Language Model (GLLM) CUDA Kernels:
//   1. init_outer_shell_dynamic : 1.2M outer sphere (assigned points visible)
//   2. train_accumulate_shadows : Atomic shadow accumulation for any WxH dataset
//   3. eval_score_shadows       : Cosine-similarity evaluation on GPU
//   4. emit_allocated_regions   : O(Allocated) renderer scaling to 1B+ lattice
// =============================================================================

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct alignas(16) PointVertex {
    float4 pos_val;
    float4 color;
};

struct alignas(16) GpuRegionDesc {
    uint32_t ox, oy, oz, width;
    uint32_t height, outer_idx, shadow_offset, class_id;
};

extern "C" __global__ void train_accumulate_shadows(
    const uint8_t* __restrict__ images,
    const uint32_t* __restrict__ class_ids,
    float* __restrict__ shadow_sum,
    uint32_t* __restrict__ class_cnt,
    uint32_t pixels_per_img,
    uint32_t num_classes,
    uint32_t start_idx,
    uint32_t batch_cnt
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_cnt * pixels_per_img) return;

    uint32_t img_idx  = start_idx + tid / pixels_per_img;
    uint32_t pix      = tid % pixels_per_img;
    uint32_t class_id = class_ids[img_idx];
    if (class_id >= num_classes) return;

    if (pix == 0u) atomicAdd(&class_cnt[class_id], 1u);

    uint8_t px = images[static_cast<size_t>(img_idx) * pixels_per_img + pix];
    if (px > 0u) {
        atomicAdd(&shadow_sum[static_cast<size_t>(class_id) * pixels_per_img + pix],
                  static_cast<float>(px) * (1.0f / 255.0f));
    }
}

extern "C" __global__ void eval_score_shadows(
    const uint8_t* __restrict__ images,
    const float* __restrict__ templates,
    float* __restrict__ scores,
    uint32_t pixels_per_img,
    uint32_t num_classes,
    uint32_t num_imgs
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_imgs * num_classes) return;

    uint32_t img_idx = tid / num_classes;
    uint32_t cls_idx = tid % num_classes;
    const uint8_t* img  = images + static_cast<size_t>(img_idx) * pixels_per_img;
    const float*   tmpl = templates + static_cast<size_t>(cls_idx) * pixels_per_img;

    float dot_val = 0.0f, norm2 = 0.0f;
    for (uint32_t p = 0; p < pixels_per_img; ++p) {
        float x = static_cast<float>(img[p]) * (1.0f / 255.0f);
        dot_val += x * tmpl[p];
        norm2   += x * x;
    }
    scores[tid] = dot_val / sqrtf(norm2 + 1e-6f);
}
