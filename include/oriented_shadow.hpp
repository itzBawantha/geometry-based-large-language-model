#pragma once
// =============================================================================
// oriented_shadow.hpp — 4-Orientation Geometric Gradient Shadow (HOG in 28x28)
//
// Zero neural weights. Zero backpropagation. Pure local directional geometry.
//
// Encodes any cropped region (from RGB buffer or summed-area integral table)
// into a 28x28 (784-point) 4-quadrant oriented-gradient shadow lattice:
//   Top-Left     [ 0..13,  0..13] :   0 deg (horizontal edges: brows, mouth)
//   Top-Right    [14..27,  0..13] :  45 deg (diagonal '/' edges: right ear/cheek)
//   Bottom-Left  [ 0..13, 14..27] :  90 deg (vertical edges: nose bridge, pupils)
//   Bottom-Right [14..27, 14..27] : 135 deg (diagonal '\' edges: left ear/cheek)
//
// Unsigned orientation [0, pi) guarantees invariance to dark-fur-on-light-wall
// vs. light-fur-on-dark-wall; local 3x3 L2-Hys block normalization eliminates
// exposure & lighting variance while rejecting flat featureless walls cleanly.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace gllm {

inline void compute_hog_from_16x16_lum(const float lum[16 * 16],
                                       bool encode_zncc_u8,
                                       uint8_t out_28x28[28 * 28]) {
    std::memset(out_28x28, 0, 28 * 28);

    // Reject flat, featureless blank walls/sky cleanly
    double sl = 0.0, sl2 = 0.0;
    for (int i = 0; i < 256; ++i) {
        sl  += lum[i];
        sl2 += static_cast<double>(lum[i]) * lum[i];
    }
    double mean_l = sl / 256.0;
    double var_l  = std::max(0.0, sl2 / 256.0 - mean_l * mean_l);
    double min_sd = encode_zncc_u8 ? 0.025 : 0.005;
    if (std::sqrt(var_l) < min_sd) return;

    // 1. Compute 3x3 Sobel gradients & soft-bin unsigned orientation [0, pi) into 4 bins across 14x14 cells
    float bins[4][14 * 14] = {};
    constexpr float kInvPi4 = 4.0f / 3.14159265358979323846f;
    constexpr float kPi     = 3.14159265358979323846f;

    auto L = [&](int x, int y) -> float { return lum[y * 16 + x]; };
    for (int cy = 0; cy < 14; ++cy) {
        int y = cy + 1;
        for (int cx = 0; cx < 14; ++cx) {
            int x = cx + 1;
            float gx = (L(x + 1, y - 1) + 2.0f * L(x + 1, y) + L(x + 1, y + 1)) -
                       (L(x - 1, y - 1) + 2.0f * L(x - 1, y) + L(x - 1, y + 1));
            float gy = (L(x - 1, y + 1) + 2.0f * L(x, y + 1) + L(x + 1, y + 1)) -
                       (L(x - 1, y - 1) + 2.0f * L(x, y - 1) + L(x + 1, y - 1));
            float mag = std::sqrt(gx * gx + gy * gy);
            if (mag < 1e-6f) continue;

            // Unsigned edge tangent/gradient orientation in [0, pi)
            float ang = std::atan2(gy, gx);
            if (ang < 0.0f) ang += kPi;
            if (ang >= kPi) ang -= kPi;
            float bf = ang * kInvPi4; // [0, 4)
            int b0 = static_cast<int>(std::floor(bf)) & 3;
            int b1 = (b0 + 1) & 3;
            float w1 = bf - std::floor(bf);
            float w0 = 1.0f - w1;
            int idx = cy * 14 + cx;
            bins[b0][idx] += mag * w0;
            bins[b1][idx] += mag * w1;
        }
    }

    // 2. Local 3x3 cell L2-Hys block normalization (Dalal-Triggs HOG geometry)
    float cell_e2[14 * 14];
    for (int i = 0; i < 196; ++i) {
        cell_e2[i] = bins[0][i] * bins[0][i] + bins[1][i] * bins[1][i] +
                     bins[2][i] * bins[2][i] + bins[3][i] * bins[3][i];
    }

    float feat[28 * 28] = {};
    double sf = 0.0, sf2 = 0.0;
    for (int cy = 0; cy < 14; ++cy) {
        for (int cx = 0; cx < 14; ++cx) {
            float block_e2 = 1e-4f;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = std::clamp(cy + dy, 0, 13);
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = std::clamp(cx + dx, 0, 13);
                    block_e2 += cell_e2[ny * 14 + nx];
                }
            }
            float inv_n = 1.0f / std::sqrt(block_e2);
            int ci = cy * 14 + cx;
            const int qx[4] = {0, 14, 0, 14};
            const int qy[4] = {0, 0, 14, 14};
            for (int b = 0; b < 4; ++b) {
                // Clip at 0.45 (L2-Hys) to prevent single specular glare edge from dominating
                float v = std::min(0.45f, bins[b][ci] * inv_n) * (1.0f / 0.45f);
                int p = (qy[b] + cy) * 28 + (qx[b] + cx);
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
        // Zero-mean unit-variance encoding across all 784 directional bins so
        // eval_score_shadows(u8, zero_mean_hog_tmpl) computes exact Zero-Mean HOG (WHO)!
        double mf = sf / 784.0;
        double sd = std::sqrt(std::max(0.0, sf2 / 784.0 - mf * mf));
        if (sd < 0.02) return;
        float inv_sd = static_cast<float>(1.0 / sd);
        for (int p = 0; p < 784; ++p) {
            float zn = std::clamp((feat[p] - static_cast<float>(mf)) * inv_sd, -2.8f, 2.8f);
            out_28x28[p] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(128.0f + 42.0f * zn)), 1, 255));
        }
    }
}

// Extract 4-orientation HOG shadow directly from a summed-area integral luminance table in O(256)!
inline void extract_hog_from_integral(const double* sat, uint32_t W, uint32_t H,
                                      float fx0, float fy0, float fx1, float fy1,
                                      bool encode_zncc_u8,
                                      uint8_t out_28x28[28 * 28]) {
    int iW = static_cast<int>(W);
    int iH = static_cast<int>(H);
    int x0 = std::clamp(static_cast<int>(std::floor(fx0)), 0, iW - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(fy0)), 0, iH - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(fx1)), x0 + 1, iW);
    int y1 = std::clamp(static_cast<int>(std::ceil(fy1)), y0 + 1, iH);
    int cw = x1 - x0;
    int ch = y1 - y0;
    const size_t stride = static_cast<size_t>(W) + 1u;

    int ix0_arr[16], ix1_arr[16];
    for (int dx = 0; dx < 16; ++dx) {
        int a = std::clamp(x0 + (dx * cw) / 16, x0, x1 - 1);
        int b = std::clamp(x0 + ((dx + 1) * cw + 15) / 16, a + 1, x1);
        ix0_arr[dx] = a;
        ix1_arr[dx] = b;
    }

    float lum[16 * 16];
    for (int dy = 0; dy < 16; ++dy) {
        int iy0 = std::clamp(y0 + (dy * ch) / 16, y0, y1 - 1);
        int iy1 = std::clamp(y0 + ((dy + 1) * ch + 15) / 16, iy0 + 1, y1);
        const double* r0 = sat + static_cast<size_t>(iy0) * stride;
        const double* r1 = sat + static_cast<size_t>(iy1) * stride;
        int dh = iy1 - iy0;
        for (int dx = 0; dx < 16; ++dx) {
            int a = ix0_arr[dx], b = ix1_arr[dx];
            double s = r1[b] - r0[b] - r1[a] + r0[a];
            lum[dy * 16 + dx] = static_cast<float>(s / static_cast<double>(dh * (b - a)));
        }
    }
    compute_hog_from_16x16_lum(lum, encode_zncc_u8, out_28x28);
}

// Extract 4-orientation HOG shadow directly from an RGB24 buffer (used during training)
inline void extract_hog_from_rgb(const uint8_t* rgb, uint32_t W, uint32_t H,
                                 int x0, int y0, int x1, int y1,
                                 uint8_t out_28x28[28 * 28]) {
    int iW = static_cast<int>(W);
    int iH = static_cast<int>(H);
    x0 = std::clamp(x0, 0, iW - 1);
    y0 = std::clamp(y0, 0, iH - 1);
    x1 = std::clamp(x1, x0 + 1, iW);
    y1 = std::clamp(y1, y0 + 1, iH);
    int cw = x1 - x0;
    int ch = y1 - y0;

    float lum[16 * 16];
    for (int dy = 0; dy < 16; ++dy) {
        int iy0 = std::clamp(y0 + (dy * ch) / 16, y0, y1 - 1);
        int iy1 = std::clamp(y0 + ((dy + 1) * ch + 15) / 16, iy0 + 1, y1);
        for (int dx = 0; dx < 16; ++dx) {
            int ix0 = std::clamp(x0 + (dx * cw) / 16, x0, x1 - 1);
            int ix1 = std::clamp(x0 + ((dx + 1) * cw + 15) / 16, ix0 + 1, x1);
            double sl = 0.0;
            int n = 0;
            for (int sy = iy0; sy < iy1; ++sy) {
                const uint8_t* row = rgb + static_cast<size_t>(sy) * W * 3u;
                for (int sx = ix0; sx < ix1; ++sx) {
                    sl += (0.299 * row[sx * 3 + 0] + 0.587 * row[sx * 3 + 1] + 0.114 * row[sx * 3 + 2]) * (1.0 / 255.0);
                    n++;
                }
            }
            lum[dy * 16 + dx] = n > 0 ? static_cast<float>(sl / n) : 0.0f;
        }
    }
    compute_hog_from_16x16_lum(lum, false, out_28x28);
}

} // namespace gllm
