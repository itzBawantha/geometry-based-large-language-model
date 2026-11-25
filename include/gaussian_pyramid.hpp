#pragma once

#include "hud_canvas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gllm {

// =============================================================================
// Generalized Multi-Scale Burt-Adelson Gaussian Pyramid (Domain-Agnostic)
//
// Constructs octave & half-octave Gaussian-filtered downsampled layers from any
// W x H input field (224x224 sketch canvas, high-res mammal facial crops, or
// arbitrary 2D sensor grids) and extracts centered canonical query patches +
// 9 micro-shifts + per-hypothesis 64-bit occupancy bitmasks for single-launch
// multi-scale GPU matching without retraining canonical Operational Manifolds!
// =============================================================================

struct PyramidLevel {
    uint32_t width  = 0;
    uint32_t height = 0;
    std::vector<float> pixels; // width * height in [0..1]
};

struct MultiScaleQueryStack {
    uint32_t num_levels = 0;
    uint32_t num_hyps   = 0;              // num_levels * 9 micro-shifts
    std::vector<uint32_t> level_sizes;    // e.g. {28, 40, 56}
    std::vector<float>    raw_hyps;       // num_hyps * 784 floats in [0..1]
    std::vector<float>    self_entropies; // num_hyps * 0.5 * H(x)
    std::vector<uint64_t> coarse_masks;   // num_hyps 64-bit 8x8 occupancy masks
};

// Separable 5-tap Burt-Adelson binomial Gaussian filter [1, 4, 6, 4, 1] / 16
inline std::vector<float> gaussian_blur_5tap(const float* src, uint32_t w, uint32_t h) {
    static constexpr float kTap[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
    std::vector<float> tmp(static_cast<size_t>(w) * h, 0.0f);
    std::vector<float> out(static_cast<size_t>(w) * h, 0.0f);

    for (uint32_t y = 0; y < h; ++y) {
        const float* row = src + static_cast<size_t>(y) * w;
        float* dst = tmp.data() + static_cast<size_t>(y) * w;
        for (uint32_t x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int dx = -2; dx <= 2; ++dx) {
                int sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(w) - 1);
                acc += row[sx] * kTap[dx + 2];
            }
            dst[x] = acc;
        }
    }
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                int sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(h) - 1);
                acc += tmp[static_cast<size_t>(sy) * w + x] * kTap[dy + 2];
            }
            out[static_cast<size_t>(y) * w + x] = acc;
        }
    }
    return out;
}

// Anti-aliased box/area downsampling between Gaussian pyramid octaves
inline std::vector<float> area_downsample_2d(const float* src, uint32_t w, uint32_t h,
                                             uint32_t nw, uint32_t nh) {
    std::vector<float> out(static_cast<size_t>(nw) * nh, 0.0f);
    float sx = static_cast<float>(w) / static_cast<float>(nw);
    float sy = static_cast<float>(h) / static_cast<float>(nh);
    for (uint32_t y = 0; y < nh; ++y) {
        uint32_t y0 = static_cast<uint32_t>(std::floor(y * sy));
        uint32_t y1 = std::min<uint32_t>(h, std::max<uint32_t>(y0 + 1u, static_cast<uint32_t>(std::ceil((y + 1u) * sy))));
        for (uint32_t x = 0; x < nw; ++x) {
            uint32_t x0 = static_cast<uint32_t>(std::floor(x * sx));
            uint32_t x1 = std::min<uint32_t>(w, std::max<uint32_t>(x0 + 1u, static_cast<uint32_t>(std::ceil((x + 1u) * sx))));
            float acc = 0.0f;
            uint32_t cnt = 0;
            for (uint32_t yy = y0; yy < y1; ++yy) {
                const float* row = src + static_cast<size_t>(yy) * w;
                for (uint32_t xx = x0; xx < x1; ++xx) {
                    acc += row[xx];
                    cnt++;
                }
            }
            out[static_cast<size_t>(y) * nw + x] = cnt ? (acc / static_cast<float>(cnt)) : 0.0f;
        }
    }
    return out;
}

// Append one 28x28 pyramid patch + its 9 micro-shifts (dx, dy in {-1, 0, +1}) into the GPU hypothesis stack
inline void append_patch_shifts_28(const float* patch28, uint32_t level_sz, MultiScaleQueryStack& stack) {
    constexpr uint32_t P = 784u;
    stack.level_sizes.push_back(level_sz);
    stack.num_levels++;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            size_t base = stack.raw_hyps.size();
            stack.raw_hyps.resize(base + P, 0.0f);
            float* dst = &stack.raw_hyps[base];
            for (int y = 0; y < 28; ++y) {
                int sy = y - dy;
                if (sy < 0 || sy >= 28) continue;
                for (int x = 0; x < 28; ++x) {
                    int sx = x - dx;
                    if (sx >= 0 && sx < 28) dst[y * 28 + x] = patch28[sy * 28 + sx];
                }
            }
            double ent = 0.0;
            for (uint32_t p = 0; p < P; ++p) {
                double xc = std::clamp(static_cast<double>(dst[p]), 1e-5, 1.0 - 1e-5);
                ent += xc * std::log(xc) + (1.0 - xc) * std::log(1.0 - xc);
            }
            stack.self_entropies.push_back(static_cast<float>(0.5 * ent));
            stack.coarse_masks.push_back(compute_coarse_mask64(dst, 0.24f));
            stack.num_hyps++;
        }
    }
}

// Build a pure Burt-Adelson 3-level Gaussian Pyramid (L28=1.0x, L40=1.43x, L56=2.0x)
// from any high-resolution square canvas/image (e.g. 224x224) with ZERO arbitrary size hacks!
inline MultiScaleQueryStack build_gaussian_pyramid_stack(const float* src, uint32_t w, uint32_t h) {
    MultiScaleQueryStack stack;
    if (w == 28u && h == 28u) {
        append_patch_shifts_28(src, 28u, stack);
        return stack;
    }

    // Octave cascade: src -> 112 -> 56 -> (half-octave 40) & (full octave 28)
    auto b0   = gaussian_blur_5tap(src, w, h);
    auto L112 = area_downsample_2d(b0.data(), w, h, 112u, 112u);
    auto b1   = gaussian_blur_5tap(L112.data(), 112u, 112u);
    auto L56  = area_downsample_2d(b1.data(), 112u, 112u, 56u, 56u);
    auto b2   = gaussian_blur_5tap(L56.data(), 56u, 56u);
    auto L40  = area_downsample_2d(b2.data(), 56u, 56u, 40u, 40u);
    auto L28  = area_downsample_2d(b2.data(), 56u, 56u, 28u, 28u);

    const std::pair<uint32_t, const std::vector<float>*> levels[3] = {
        {28u, &L28}, {40u, &L40}, {56u, &L56}
    };

    for (const auto& lv : levels) {
        uint32_t sz = lv.first;
        const auto& img = *lv.second;
        int minx = static_cast<int>(sz), miny = static_cast<int>(sz), maxx = -1, maxy = -1;
        for (uint32_t y = 0; y < sz; ++y) {
            for (uint32_t x = 0; x < sz; ++x) {
                if (img[static_cast<size_t>(y) * sz + x] > 0.05f) {
                    minx = std::min(minx, static_cast<int>(x));
                    maxx = std::max(maxx, static_cast<int>(x));
                    miny = std::min(miny, static_cast<int>(y));
                    maxy = std::max(maxy, static_cast<int>(y));
                }
            }
        }
        if (maxx < minx || maxy < miny) continue;
        // Skip higher-resolution octave only if the active object physically overflows the 28x28 window!
        int bw = maxx - minx + 1;
        int bh = maxy - miny + 1;
        if (sz > 28u && (bw > 26 || bh > 26)) continue;

        int cx = (minx + maxx) / 2;
        int cy = (miny + maxy) / 2;
        std::array<float, 784> patch{};
        for (int ty = 0; ty < 28; ++ty) {
            int sy = cy - 14 + ty;
            if (sy < 0 || sy >= static_cast<int>(sz)) continue;
            for (int tx = 0; tx < 28; ++tx) {
                int sx = cx - 14 + tx;
                if (sx >= 0 && sx < static_cast<int>(sz)) {
                    patch[static_cast<size_t>(ty) * 28 + tx] = img[static_cast<size_t>(sy) * sz + sx];
                }
            }
        }
        // Light canonical anti-aliasing smoothing + peak normalize [0..1]
        std::array<float, 784> smooth{};
        gaussian_blur_28(patch.data(), smooth.data(), 0.55f);
        float mx = 0.0f;
        for (float v : smooth) mx = std::max(mx, v);
        if (mx > 1e-6f) {
            float inv = 1.0f / mx;
            for (float& v : smooth) v = std::clamp(v * inv, 0.0f, 1.0f);
        }
        append_patch_shifts_28(smooth.data(), sz, stack);
    }
    return stack;
}

} // namespace gllm
