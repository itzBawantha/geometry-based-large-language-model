#pragma once
// =============================================================================
// grad_domain.hpp — Gradient-Domain Sparse Edge Map & Directional Integral Engine
//
// Inspired directly by CMU 15-463 Lecture 9 ("Gradient-domain image processing"):
//   1. Log-luminance transform (p.108): multiplicative contrast invariance across
//      dark fur, light fur, shadows, and bright backgrounds.
//   2. Edge-preserving 3x3 bilateral smooth: suppresses sensor noise & fine fur
//      speckle while preserving crisp anatomical silhouettes (ears, eyes, nose).
//   3. Forward-difference gradient vector field (p.28-29):
//        gx = L(x+1, y) - L(x, y),   gy = L(x, y+1) - L(x, y)
//   4. Adaptive gradient-domain sparsification (p.135-136):
//        Flat walls, sky, beds, floors collapse to 0.0 (pure black), leaving
//        ONLY sharp structural edges glowing in [0..1].
//   5. 4-Orientation directional sparse edge integral tables (0, 45, 90, 135 deg)
//      for O(1) blank-region skipping and O(196) 28x28 directional edge-patch
//      extraction during inverted child-first bottom-up detection!
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gllm {

struct GradEdgeImage {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> edge_vis; // w * h sparse scalar edge map [0..255] for visualization
    std::vector<double>  sat_mag;  // (w+1)*(h+1) summed-area table of total sparse edge energy
    std::array<std::vector<double>, 4> sat_dir; // 4 orientation planes (0, 45, 90, 135 deg)
};

inline void build_grad_edge_image(const uint8_t* rgb, uint32_t W, uint32_t H, GradEdgeImage& out) {
    out.w = W;
    out.h = H;
    const size_t N = static_cast<size_t>(W) * H;
    out.edge_vis.assign(N, 0u);
    if (W < 3u || H < 3u) return;

    // 1. Log-luminance scalar field (CMU Lecture 9 p.108: multiplicative contrast invariance!)
    std::vector<float> log_l(N);
    for (size_t i = 0; i < N; ++i) {
        float lum = (0.299f * rgb[i * 3 + 0] + 0.587f * rgb[i * 3 + 1] + 0.114f * rgb[i * 3 + 2]) * (1.0f / 255.0f);
        log_l[i] = std::log(0.06f + lum);
    }

    // 2. Fast 3x3 edge-preserving bilateral smooth (kills grain/speckle, keeps sharp object edges!)
    std::vector<float> sm(N);
    constexpr float kInv2Sig2 = 1.0f / (2.0f * 0.24f * 0.24f);
    for (uint32_t y = 0; y < H; ++y) {
        int y0 = (y > 0) ? static_cast<int>(y) - 1 : 0;
        int y1 = (y + 1 < H) ? static_cast<int>(y) + 1 : static_cast<int>(H) - 1;
        for (uint32_t x = 0; x < W; ++x) {
            int x0 = (x > 0) ? static_cast<int>(x) - 1 : 0;
            int x1 = (x + 1 < W) ? static_cast<int>(x) + 1 : static_cast<int>(W) - 1;
            float c = log_l[static_cast<size_t>(y) * W + x];
            float ws = 0.0f, vs = 0.0f;
            for (int ny = y0; ny <= y1; ++ny) {
                float sy = (ny == static_cast<int>(y)) ? 2.0f : 1.0f;
                const float* row = log_l.data() + static_cast<size_t>(ny) * W;
                for (int nx = x0; nx <= x1; ++nx) {
                    float sx = (nx == static_cast<int>(x)) ? 2.0f : 1.0f;
                    float v  = row[nx];
                    float d  = v - c;
                    float w  = (sx * sy) * std::exp(-(d * d) * kInv2Sig2);
                    ws += w;
                    vs += w * v;
                }
            }
            sm[static_cast<size_t>(y) * W + x] = (ws > 1e-6f) ? (vs / ws) : c;
        }
    }

    // 3. Forward-difference gradient vector field (CMU Lecture 9 p.28-29: gx = I(x+1,y)-I(x,y), gy = I(x,y+1)-I(x,y))
    std::vector<float> gx_arr(N, 0.0f), gy_arr(N, 0.0f), mag_arr(N, 0.0f);
    double sum_m = 0.0, sum_m2 = 0.0;
    for (uint32_t y = 0; y + 1 < H; ++y) {
        const float* r0 = sm.data() + static_cast<size_t>(y) * W;
        const float* r1 = sm.data() + static_cast<size_t>(y + 1) * W;
        for (uint32_t x = 0; x + 1 < W; ++x) {
            float gx = r0[x + 1] - r0[x];
            float gy = r1[x]     - r0[x];
            float m  = std::sqrt(gx * gx + gy * gy);
            size_t idx = static_cast<size_t>(y) * W + x;
            gx_arr[idx]  = gx;
            gy_arr[idx]  = gy;
            mag_arr[idx] = m;
            sum_m  += m;
            sum_m2 += static_cast<double>(m) * m;
        }
    }

    // 4. Adaptive Gradient-Domain Sparsification (CMU Lecture 9 p.136: flat regions -> 0.0, edges glow!)
    double mean_m = sum_m / std::max<size_t>(1, N);
    double sd_m   = std::sqrt(std::max(0.0, sum_m2 / std::max<size_t>(1, N) - mean_m * mean_m));
    float tau     = static_cast<float>(mean_m + 0.30 * sd_m);
    float inv_sc  = 1.0f / static_cast<float>(std::max(1e-4, 2.2 * sd_m));

    const size_t stride = static_cast<size_t>(W) + 1u;
    const size_t sat_sz = stride * (static_cast<size_t>(H) + 1u);
    out.sat_mag.assign(sat_sz, 0.0);
    for (int b = 0; b < 4; ++b) out.sat_dir[b].assign(sat_sz, 0.0);

    constexpr float kPi     = 3.14159265358979323846f;
    constexpr float kInvPi4 = 4.0f / kPi;

    for (uint32_t y = 0; y < H; ++y) {
        double r_mag = 0.0;
        double r_dir[4] = {0.0, 0.0, 0.0, 0.0};
        size_t dst_row = static_cast<size_t>(y + 1u) * stride;
        size_t prv_row = static_cast<size_t>(y) * stride;
        for (uint32_t x = 0; x < W; ++x) {
            size_t idx = static_cast<size_t>(y) * W + x;
            float m = mag_arr[idx];
            float e = 0.0f;
            float d_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (m > tau) {
                e = std::tanh((m - tau) * inv_sc);
                float ang = std::atan2(gy_arr[idx], gx_arr[idx]);
                if (ang < 0.0f) ang += kPi;
                if (ang >= kPi) ang -= kPi;
                float bf = ang * kInvPi4;
                int b0 = static_cast<int>(std::floor(bf)) & 3;
                int b1 = (b0 + 1) & 3;
                float w1 = bf - std::floor(bf);
                d_val[b0] = e * (1.0f - w1);
                d_val[b1] = e * w1;
                out.edge_vis[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(e * 255.0f)), 0, 255));
            }
            r_mag += e;
            out.sat_mag[dst_row + x + 1u] = out.sat_mag[prv_row + x + 1u] + r_mag;
            for (int b = 0; b < 4; ++b) {
                r_dir[b] += d_val[b];
                out.sat_dir[b][dst_row + x + 1u] = out.sat_dir[b][prv_row + x + 1u] + r_dir[b];
            }
        }
    }
}

// O(1) sparse edge energy check inside any rectangle [fx0..fx1, fy0..fy1]
// Enables instant skipping of flat empty background regions (CMU Lecture 9 p.140/151)!
inline float query_sparse_edge_density(const GradEdgeImage& ge,
                                       float fx0, float fy0, float fx1, float fy1) {
    int iW = static_cast<int>(ge.w), iH = static_cast<int>(ge.h);
    int x0 = std::clamp(static_cast<int>(std::floor(fx0)), 0, iW - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(fy0)), 0, iH - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(fx1)), x0 + 1, iW);
    int y1 = std::clamp(static_cast<int>(std::ceil(fy1)), y0 + 1, iH);
    size_t stride = static_cast<size_t>(ge.w) + 1u;
    const double* sat = ge.sat_mag.data();
    double s = sat[static_cast<size_t>(y1) * stride + x1]
             - sat[static_cast<size_t>(y0) * stride + x1]
             - sat[static_cast<size_t>(y1) * stride + x0]
             + sat[static_cast<size_t>(y0) * stride + x0];
    return static_cast<float>(s / static_cast<double>((x1 - x0) * (y1 - y0)));
}

// Extract a 28x28 (4 x 14x14 directional planes) sparse gradient-domain edge shadow patch in O(196)!
// Returns false if the region is essentially empty (no edges -> skip!).
inline bool extract_grad_edge_patch_28x28(const GradEdgeImage& ge,
                                          float fx0, float fy0, float fx1, float fy1,
                                          bool encode_zncc_u8,
                                          float min_edge_density,
                                          uint8_t out_28x28[28 * 28]) {
    std::memset(out_28x28, 0, 28 * 28);
    if (ge.w < 3u || ge.h < 3u) return false;
    int iW = static_cast<int>(ge.w), iH = static_cast<int>(ge.h);
    int x0 = std::clamp(static_cast<int>(std::floor(fx0)), 0, iW - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(fy0)), 0, iH - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(fx1)), x0 + 1, iW);
    int y1 = std::clamp(static_cast<int>(std::ceil(fy1)), y0 + 1, iH);
    int cw = x1 - x0;
    int ch = y1 - y0;
    size_t stride = static_cast<size_t>(ge.w) + 1u;

    // Fast O(1) sparse rejection of flat wall / empty sky windows!
    double tot_e = ge.sat_mag[static_cast<size_t>(y1) * stride + x1]
                 - ge.sat_mag[static_cast<size_t>(y0) * stride + x1]
                 - ge.sat_mag[static_cast<size_t>(y1) * stride + x0]
                 + ge.sat_mag[static_cast<size_t>(y0) * stride + x0];
    if (tot_e / static_cast<double>(cw * ch) < static_cast<double>(min_edge_density)) return false;

    int ix0_arr[14], ix1_arr[14];
    for (int dx = 0; dx < 14; ++dx) {
        int a = std::clamp(x0 + (dx * cw) / 14, x0, x1 - 1);
        int b = std::clamp(x0 + ((dx + 1) * cw + 13) / 14, a + 1, x1);
        ix0_arr[dx] = a;
        ix1_arr[dx] = b;
    }

    float bins[4][14 * 14];
    float cell_e2[14 * 14];
    for (int dy = 0; dy < 14; ++dy) {
        int iy0 = std::clamp(y0 + (dy * ch) / 14, y0, y1 - 1);
        int iy1 = std::clamp(y0 + ((dy + 1) * ch + 13) / 14, iy0 + 1, y1);
        int dh = iy1 - iy0;
        size_t o0 = static_cast<size_t>(iy0) * stride;
        size_t o1 = static_cast<size_t>(iy1) * stride;
        for (int dx = 0; dx < 14; ++dx) {
            int a = ix0_arr[dx], b = ix1_arr[dx];
            double inv_area = 1.0 / static_cast<double>(dh * (b - a));
            int ci = dy * 14 + dx;
            float e2 = 0.0f;
            for (int q = 0; q < 4; ++q) {
                const double* sd = ge.sat_dir[q].data();
                float v = static_cast<float>((sd[o1 + b] - sd[o0 + b] - sd[o1 + a] + sd[o0 + a]) * inv_area);
                bins[q][ci] = v;
                e2 += v * v;
            }
            cell_e2[ci] = e2;
        }
    }

    // Local 3x3 L2-Hys block normalization across the 14x14 directional edge grid
    float feat[784] = {};
    double sf = 0.0, sf2 = 0.0;
    const int qx[4] = {0, 14, 0, 14};
    const int qy[4] = {0, 0, 14, 14};
    for (int cy = 0; cy < 14; ++cy) {
        for (int cx = 0; cx < 14; ++cx) {
            float blk = 1e-4f;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = std::clamp(cy + dy, 0, 13);
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = std::clamp(cx + dx, 0, 13);
                    blk += cell_e2[ny * 14 + nx];
                }
            }
            float inv_n = 1.0f / std::sqrt(blk);
            int ci = cy * 14 + cx;
            for (int q = 0; q < 4; ++q) {
                float v = std::min(0.45f, bins[q][ci] * inv_n) * (1.0f / 0.45f);
                int p = (qy[q] + cy) * 28 + (qx[q] + cx);
                feat[p] = v;
                sf  += v;
                sf2 += static_cast<double>(v) * v;
            }
        }
    }

    if (!encode_zncc_u8) {
        for (int p = 0; p < 784; ++p) {
            out_28x28[p] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(feat[p] * 255.0f)), 0, 255));
        }
    } else {
        double mf = sf / 784.0;
        double sd = std::sqrt(std::max(0.0, sf2 / 784.0 - mf * mf));
        if (sd < 0.02) return false;
        float inv_sd = static_cast<float>(1.0 / sd);
        for (int p = 0; p < 784; ++p) {
            float zn = std::clamp((feat[p] - static_cast<float>(mf)) * inv_sd, -2.8f, 2.8f);
            out_28x28[p] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(128.0f + 42.0f * zn)), 1, 255));
        }
    }
    return true;
}

} // namespace gllm
