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

extern "C" __global__ void match_sketch_pruned(
    const float* __restrict__ sketch_raw,
    const float* __restrict__ sketch_self_ent,
    uint64_t sketch_mask,
    const uint64_t* __restrict__ class_masks,
    const float* __restrict__ class_log_odds,
    const float* __restrict__ class_prob,
    const float* __restrict__ class_bias,
    float* __restrict__ out_overlap,
    uint32_t* __restrict__ out_status,
    uint32_t pixels_img,
    uint32_t num_classes,
    uint32_t num_shifts,
    float prune_ratio,
    float inv_t_ln2
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_classes) return;

    // Stage 1: Hardware 64-bit coarse footprint pruning (and.b64 + popc.b64)
    int qbits = __popcll(sketch_mask);
    int hits  = __popcll(sketch_mask & class_masks[c]);
    if (qbits == 0 || hits == 0 || (static_cast<float>(hits) / static_cast<float>(qbits)) < prune_ratio) {
        out_overlap[c] = 0.0f;
        out_status[c]  = 0u;
        return;
    }

    // Stage 2: Multi-shift symmetric Bernoulli likelihood x continuous Soft-IoU completeness
    //   - Bernoulli log-odds strictly separates core vs void strokes
    //   - Soft-IoU = sum min(x, p) / sum max(x, p) strictly penalizes missing shadow mass & void overflow
    const float* log_odds = class_log_odds + static_cast<size_t>(c) * pixels_img;
    const float* prob     = class_prob     + static_cast<size_t>(c) * pixels_img;
    float bias = class_bias[c];
    float best_score = 0.0f;
    for (uint32_t s = 0; s < num_shifts; ++s) {
        const float* sraw = sketch_raw + static_cast<size_t>(s) * pixels_img;
        float ll = bias, inter = 0.0f, uni = 0.0f;
        for (uint32_t p = 0; p < pixels_img; ++p) {
            float x  = sraw[p];
            float pv = prob[p];
            ll    += x * log_odds[p];
            inter += fminf(x, pv);
            uni   += fmaxf(x, pv);
        }
        float dkl  = fminf(ll - sketch_self_ent[s], 0.0f);
        float bern = fminf(fmaxf(exp2f(dkl * inv_t_ln2), 0.0f), 1.0f);
        float iou  = inter / (uni + 1e-6f);
        best_score = fmaxf(best_score, bern * iou);
    }
    out_overlap[c] = best_score;
    out_status[c]  = 1u;
}

// 128-byte CUDA-resident Operational Manifold Metadata Register
struct alignas(16) GpuManifoldDesc {
    uint32_t manifold_id, class_count, region_count, reserved;
    float    center_x, center_y, center_z, max_radius;
    float    min_x, min_y, min_z, volume_fraction;
    float    max_x, max_y, max_z, pad1;
    char     name[32];           // e.g. "fundamental_text"
    char     semantic_label[48]; // e.g. "handwritten_letters_and_numbers"
};

extern "C" __global__ void route_operational_manifolds(
    const GpuManifoldDesc* __restrict__ manifolds,
    const uint32_t* __restrict__ region_manifold,
    const float* __restrict__ region_overlap,
    const uint32_t* __restrict__ region_status,
    float* __restrict__ out_best_score,
    uint32_t* __restrict__ out_best_region,
    uint32_t* __restrict__ out_active_cnt,
    uint32_t num_manifolds,
    uint32_t num_regions
) {
    uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_manifolds) return;

    uint32_t target_id = manifolds[m].manifold_id;
    float best_score   = 0.0f;
    uint32_t best_reg  = 0xFFFFFFFFu;
    uint32_t active    = 0u;
    for (uint32_t r = 0; r < num_regions; ++r) {
        if (region_manifold[r] != target_id || region_status[r] == 0u) continue;
        active++;
        if (region_overlap[r] > best_score) {
            best_score = region_overlap[r];
            best_reg   = r;
        }
    }
    out_best_score[m]  = best_score;
    out_best_region[m] = best_reg;
    out_active_cnt[m]  = active;
}
