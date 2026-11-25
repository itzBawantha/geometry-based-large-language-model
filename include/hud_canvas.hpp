#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gllm {

struct MatchResultRow {
    uint32_t class_id   = 0;
    uint32_t outer_idx  = 0;
    std::string label;
    float overlap_pct   = 0.0f; // Independent geometric overlap [0..100]
    float rel_prob_pct  = 0.0f; // Relative probability among surviving candidates [0..100]
    bool  pruned        = false;
};

struct MatchSummary {
    bool     has_result     = false;
    bool     is_unknown     = false;
    float    cuda_ms        = 0.0f;
    uint32_t total_classes  = 0;
    uint32_t pruned_count   = 0;
    uint32_t survived_count = 0;
    std::vector<MatchResultRow> ranked;        // Surviving classes sorted by overlap desc
    std::vector<std::string>    pruned_labels; // Labels eliminated at Stage 1
};

// Minimal crisp 5x7 uppercase + digit + symbol bitmap font (bit 4 = leftmost pixel)
inline void glyph_5x7(char ch, uint8_t out[7]) {
    static const uint8_t kBlank[7] = {0,0,0,0,0,0,0};
    std::memcpy(out, kBlank, 7);
    if (ch >= 'a' && ch <= 'z') {
        // Distinct lowercase glyphs for clarity in 62-class readout!
        switch (ch) {
            case 'a': { uint8_t g[7]={0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}; std::memcpy(out,g,7); return; }
            case 'b': { uint8_t g[7]={0x10,0x10,0x1E,0x11,0x11,0x11,0x1E}; std::memcpy(out,g,7); return; }
            case 'c': { uint8_t g[7]={0x00,0x00,0x0E,0x10,0x10,0x10,0x0E}; std::memcpy(out,g,7); return; }
            case 'd': { uint8_t g[7]={0x01,0x01,0x0F,0x11,0x11,0x11,0x0F}; std::memcpy(out,g,7); return; }
            case 'e': { uint8_t g[7]={0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E}; std::memcpy(out,g,7); return; }
            case 'f': { uint8_t g[7]={0x06,0x08,0x1E,0x08,0x08,0x08,0x08}; std::memcpy(out,g,7); return; }
            case 'g': { uint8_t g[7]={0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E}; std::memcpy(out,g,7); return; }
            case 'h': { uint8_t g[7]={0x10,0x10,0x1E,0x11,0x11,0x11,0x11}; std::memcpy(out,g,7); return; }
            case 'i': { uint8_t g[7]={0x04,0x00,0x0C,0x04,0x04,0x04,0x0E}; std::memcpy(out,g,7); return; }
            case 'j': { uint8_t g[7]={0x02,0x00,0x06,0x02,0x02,0x12,0x0C}; std::memcpy(out,g,7); return; }
            case 'k': { uint8_t g[7]={0x10,0x10,0x12,0x14,0x18,0x14,0x12}; std::memcpy(out,g,7); return; }
            case 'l': { uint8_t g[7]={0x0C,0x04,0x04,0x04,0x04,0x04,0x0E}; std::memcpy(out,g,7); return; }
            case 'm': { uint8_t g[7]={0x00,0x00,0x1A,0x15,0x15,0x15,0x15}; std::memcpy(out,g,7); return; }
            case 'n': { uint8_t g[7]={0x00,0x00,0x1E,0x11,0x11,0x11,0x11}; std::memcpy(out,g,7); return; }
            case 'o': { uint8_t g[7]={0x00,0x00,0x0E,0x11,0x11,0x11,0x0E}; std::memcpy(out,g,7); return; }
            case 'p': { uint8_t g[7]={0x00,0x1E,0x11,0x11,0x1E,0x10,0x10}; std::memcpy(out,g,7); return; }
            case 'q': { uint8_t g[7]={0x00,0x0F,0x11,0x11,0x0F,0x01,0x01}; std::memcpy(out,g,7); return; }
            case 'r': { uint8_t g[7]={0x00,0x00,0x16,0x19,0x10,0x10,0x10}; std::memcpy(out,g,7); return; }
            case 's': { uint8_t g[7]={0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E}; std::memcpy(out,g,7); return; }
            case 't': { uint8_t g[7]={0x08,0x08,0x1E,0x08,0x08,0x09,0x06}; std::memcpy(out,g,7); return; }
            case 'u': { uint8_t g[7]={0x00,0x00,0x11,0x11,0x11,0x11,0x0F}; std::memcpy(out,g,7); return; }
            case 'v': { uint8_t g[7]={0x00,0x00,0x11,0x11,0x11,0x0A,0x04}; std::memcpy(out,g,7); return; }
            case 'w': { uint8_t g[7]={0x00,0x00,0x11,0x11,0x15,0x15,0x0A}; std::memcpy(out,g,7); return; }
            case 'x': { uint8_t g[7]={0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}; std::memcpy(out,g,7); return; }
            case 'y': { uint8_t g[7]={0x00,0x11,0x11,0x11,0x0F,0x01,0x0E}; std::memcpy(out,g,7); return; }
            case 'z': { uint8_t g[7]={0x00,0x00,0x1F,0x02,0x04,0x08,0x1F}; std::memcpy(out,g,7); return; }
        }
    }
    switch (ch) {
        case '0': { uint8_t g[7]={0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case '1': { uint8_t g[7]={0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}; std::memcpy(out,g,7); break; }
        case '2': { uint8_t g[7]={0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}; std::memcpy(out,g,7); break; }
        case '3': { uint8_t g[7]={0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}; std::memcpy(out,g,7); break; }
        case '4': { uint8_t g[7]={0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}; std::memcpy(out,g,7); break; }
        case '5': { uint8_t g[7]={0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case '6': { uint8_t g[7]={0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case '7': { uint8_t g[7]={0x1F,0x01,0x02,0x04,0x08,0x08,0x08}; std::memcpy(out,g,7); break; }
        case '8': { uint8_t g[7]={0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case '9': { uint8_t g[7]={0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}; std::memcpy(out,g,7); break; }
        case 'A': { uint8_t g[7]={0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}; std::memcpy(out,g,7); break; }
        case 'B': { uint8_t g[7]={0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}; std::memcpy(out,g,7); break; }
        case 'C': { uint8_t g[7]={0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case 'D': { uint8_t g[7]={0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}; std::memcpy(out,g,7); break; }
        case 'E': { uint8_t g[7]={0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}; std::memcpy(out,g,7); break; }
        case 'F': { uint8_t g[7]={0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}; std::memcpy(out,g,7); break; }
        case 'G': { uint8_t g[7]={0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}; std::memcpy(out,g,7); break; }
        case 'H': { uint8_t g[7]={0x11,0x11,0x11,0x1F,0x11,0x11,0x11}; std::memcpy(out,g,7); break; }
        case 'I': { uint8_t g[7]={0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}; std::memcpy(out,g,7); break; }
        case 'J': { uint8_t g[7]={0x07,0x02,0x02,0x02,0x12,0x12,0x0C}; std::memcpy(out,g,7); break; }
        case 'K': { uint8_t g[7]={0x11,0x12,0x14,0x18,0x14,0x12,0x11}; std::memcpy(out,g,7); break; }
        case 'L': { uint8_t g[7]={0x10,0x10,0x10,0x10,0x10,0x10,0x1F}; std::memcpy(out,g,7); break; }
        case 'M': { uint8_t g[7]={0x11,0x1B,0x15,0x15,0x11,0x11,0x11}; std::memcpy(out,g,7); break; }
        case 'N': { uint8_t g[7]={0x11,0x19,0x15,0x13,0x11,0x11,0x11}; std::memcpy(out,g,7); break; }
        case 'O': { uint8_t g[7]={0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case 'P': { uint8_t g[7]={0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}; std::memcpy(out,g,7); break; }
        case 'Q': { uint8_t g[7]={0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}; std::memcpy(out,g,7); break; }
        case 'R': { uint8_t g[7]={0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}; std::memcpy(out,g,7); break; }
        case 'S': { uint8_t g[7]={0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}; std::memcpy(out,g,7); break; }
        case 'T': { uint8_t g[7]={0x1F,0x04,0x04,0x04,0x04,0x04,0x04}; std::memcpy(out,g,7); break; }
        case 'U': { uint8_t g[7]={0x11,0x11,0x11,0x11,0x11,0x11,0x0E}; std::memcpy(out,g,7); break; }
        case 'V': { uint8_t g[7]={0x11,0x11,0x11,0x11,0x11,0x0A,0x04}; std::memcpy(out,g,7); break; }
        case 'W': { uint8_t g[7]={0x11,0x11,0x11,0x15,0x15,0x1B,0x11}; std::memcpy(out,g,7); break; }
        case 'X': { uint8_t g[7]={0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}; std::memcpy(out,g,7); break; }
        case 'Y': { uint8_t g[7]={0x11,0x11,0x0A,0x04,0x04,0x04,0x04}; std::memcpy(out,g,7); break; }
        case 'Z': { uint8_t g[7]={0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}; std::memcpy(out,g,7); break; }
        case '.': { uint8_t g[7]={0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}; std::memcpy(out,g,7); break; }
        case ',': { uint8_t g[7]={0x00,0x00,0x00,0x00,0x04,0x04,0x08}; std::memcpy(out,g,7); break; }
        case ':': { uint8_t g[7]={0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}; std::memcpy(out,g,7); break; }
        case '-': { uint8_t g[7]={0x00,0x00,0x00,0x1F,0x00,0x00,0x00}; std::memcpy(out,g,7); break; }
        case '/': { uint8_t g[7]={0x01,0x02,0x02,0x04,0x08,0x08,0x10}; std::memcpy(out,g,7); break; }
        case '%': { uint8_t g[7]={0x19,0x19,0x02,0x04,0x08,0x13,0x13}; std::memcpy(out,g,7); break; }
        case '#': { uint8_t g[7]={0x0A,0x1F,0x0A,0x0A,0x1F,0x0A,0x00}; std::memcpy(out,g,7); break; }
        case '(': { uint8_t g[7]={0x02,0x04,0x08,0x08,0x08,0x04,0x02}; std::memcpy(out,g,7); break; }
        case ')': { uint8_t g[7]={0x08,0x04,0x02,0x02,0x02,0x04,0x08}; std::memcpy(out,g,7); break; }
        case '[': { uint8_t g[7]={0x0E,0x08,0x08,0x08,0x08,0x08,0x0E}; std::memcpy(out,g,7); break; }
        case ']': { uint8_t g[7]={0x0E,0x02,0x02,0x02,0x02,0x02,0x0E}; std::memcpy(out,g,7); break; }
        case '\'':{ uint8_t g[7]={0x04,0x04,0x08,0x00,0x00,0x00,0x00}; std::memcpy(out,g,7); break; }
        case '!': { uint8_t g[7]={0x04,0x04,0x04,0x04,0x04,0x00,0x04}; std::memcpy(out,g,7); break; }
        case '?': { uint8_t g[7]={0x0E,0x11,0x01,0x02,0x04,0x00,0x04}; std::memcpy(out,g,7); break; }
        case '|': { uint8_t g[7]={0x04,0x04,0x04,0x04,0x04,0x04,0x04}; std::memcpy(out,g,7); break; }
        default: break;
    }
}

struct Rect { int x, y, w, h; bool contains(int px, int py) const { return px >= x && px < x + w && py >= y && py < y + h; } };

// Compute 64-bit (8x8 coarse occupancy) bitmask for any 28x28 float image [0..1]
inline uint64_t compute_coarse_mask64(const float* img28, float threshold) {
    uint64_t mask = 0;
    for (int gy = 0; gy < 8; ++gy) {
        int y0 = (gy * 28) / 8;
        int y1 = ((gy + 1) * 28) / 8;
        for (int gx = 0; gx < 8; ++gx) {
            int x0 = (gx * 28) / 8;
            int x1 = ((gx + 1) * 28) / 8;
            float peak = 0.0f;
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x)
                    peak = std::max(peak, img28[y * 28 + x]);
            if (peak > threshold) mask |= (1ULL << (gy * 8 + gx));
        }
    }
    return mask;
}

// Separable 5-tap Gaussian blur on a 28x28 float grid (matches EMNIST stroke smoothing)
inline void gaussian_blur_28(const float* src, float* dst, float sigma = 0.80f) {
    if (sigma <= 0.01f) {
        std::memcpy(dst, src, 784 * sizeof(float));
        return;
    }
    float k[5], ks = 0.0f;
    for (int i = -2; i <= 2; ++i) {
        k[i + 2] = std::exp(-0.5f * (i / sigma) * (i / sigma));
        ks += k[i + 2];
    }
    for (int i = 0; i < 5; ++i) k[i] /= ks;

    float tmp[784] = {};
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float s = 0.0f;
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = x + dx;
                if (xx >= 0 && xx < 28) s += src[y * 28 + xx] * k[dx + 2];
            }
            tmp[y * 28 + x] = s;
        }
    }
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float s = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                int yy = y + dy;
                if (yy >= 0 && yy < 28) s += tmp[yy * 28 + x] * k[dy + 2];
            }
            dst[y * 28 + x] = s;
        }
    }
}

// Canonical aspect-preserving tight-bbox normalizer into target_box x target_box inside 28x28
// Eliminates scale bias between MNIST (20x20), EMNIST (24x24), and hand-drawn sketches!
inline void canonicalize_28(const float* src, float* dst, float target_box = 22.0f, float thr_frac = 0.22f) {
    std::fill(dst, dst + 784, 0.0f);
    float mx = 0.0f;
    for (int i = 0; i < 784; ++i) mx = std::max(mx, src[i]);
    if (mx < 1e-6f) return;

    float thr = mx * thr_frac;
    int minx = 28, miny = 28, maxx = -1, maxy = -1;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            if (src[y * 28 + x] > thr) {
                minx = std::min(minx, x); maxx = std::max(maxx, x);
                miny = std::min(miny, y); maxy = std::max(maxy, y);
            }
        }
    }
    if (maxx < minx || maxy < miny) return;

    int bw = maxx - minx + 1;
    int bh = maxy - miny + 1;
    float scale = target_box / static_cast<float>(std::max(bw, bh));
    float ox = (28.0f - bw * scale) * 0.5f;
    float oy = (28.0f - bh * scale) * 0.5f;

    auto sample = [&](int px, int py) {
        return (px >= 0 && px < 28 && py >= 0 && py < 28) ? src[py * 28 + px] : 0.0f;
    };

    float omx = 0.0f;
    for (int ty = 0; ty < 28; ++ty) {
        for (int tx = 0; tx < 28; ++tx) {
            float sx = (tx + 0.5f - ox) / scale + minx - 0.5f;
            float sy = (ty + 0.5f - oy) / scale + miny - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            int y0 = static_cast<int>(std::floor(sy));
            float wx = sx - x0, wy = sy - y0;
            float v = (sample(x0, y0) * (1.0f - wx) + sample(x0 + 1, y0) * wx) * (1.0f - wy)
                    + (sample(x0, y0 + 1) * (1.0f - wx) + sample(x0 + 1, y0 + 1) * wx) * wy;
            v = std::clamp(v, 0.0f, 1.0f);
            dst[ty * 28 + tx] = v;
            omx = std::max(omx, v);
        }
    }
    if (omx > 1e-6f) {
        float inv = 1.0f / omx;
        for (int i = 0; i < 784; ++i) dst[i] *= inv;
    }
}

class SketchSidebarHUD {
public:
    static constexpr int kPanelW  = 336;
    static constexpr int kPanelH  = 840;
    static constexpr int kCanvasS = 224;

    Rect canvas_rect  = {20, 46, kCanvasS, kCanvasS};
    Rect preview_rect = {256, 46, 62, 62};
    Rect btn_capture  = {20, 282, 144, 34};
    Rect btn_clear    = {174, 282, 144, 34};
    Rect row_rects[10];

    std::vector<float>   canvas;       // 224 x 224 float [0..1]
    std::array<float, 784> norm28{};   // Auto-centered 28x28 [0..1] sent to CUDA
    std::vector<uint8_t> rgba;         // kPanelW x kPanelH x 4 RGBA8 texture
    bool dirty_tex = true;
    bool hover_capture = false;
    bool hover_clear   = false;
    int  hover_row     = -1;

    SketchSidebarHUD() : canvas(kCanvasS * kCanvasS, 0.0f), rgba(kPanelW * kPanelH * 4, 0) {
        for (int i = 0; i < 10; ++i) {
            row_rects[i] = {18, 462 + i * 32, 300, 28};
        }
    }

    void clear() {
        std::fill(canvas.begin(), canvas.end(), 0.0f);
        norm28.fill(0.0f);
        dirty_tex = true;
    }

    void load_raw_28x28(const uint8_t* px28) {
        clear();
        std::array<float, 784> raw{};
        for (int i = 0; i < 784; ++i) raw[i] = static_cast<float>(px28[i]) / 255.0f;
        for (int y = 0; y < kCanvasS; ++y) {
            int sy = (y * 28) / kCanvasS;
            for (int x = 0; x < kCanvasS; ++x) {
                int sx = (x * 28) / kCanvasS;
                canvas[y * kCanvasS + x] = raw[sy * 28 + sx];
            }
        }
        canonicalize_28(raw.data(), norm28.data(), 22.0f, 0.10f);
        dirty_tex = true;
    }

    // Smooth soft-brush dab & stroke interpolation inside the 224x224 canvas
    void draw_stroke(float x0, float y0, float x1, float y1, float radius = 10.5f) {
        float dx = x1 - x0, dy = y1 - y0;
        float dist = std::sqrt(dx * dx + dy * dy);
        int steps = std::max(1, static_cast<int>(dist / 2.0f));
        for (int s = 0; s <= steps; ++s) {
            float t = static_cast<float>(s) / steps;
            float cx = x0 + dx * t;
            float cy = y0 + dy * t;
            int ix0 = std::max(0, static_cast<int>(cx - radius - 1));
            int ix1 = std::min(kCanvasS - 1, static_cast<int>(cx + radius + 1));
            int iy0 = std::max(0, static_cast<int>(cy - radius - 1));
            int iy1 = std::min(kCanvasS - 1, static_cast<int>(cy + radius + 1));
            for (int y = iy0; y <= iy1; ++y) {
                for (int x = ix0; x <= ix1; ++x) {
                    float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    if (d < radius) {
                        float v = 1.0f - std::pow(d / radius, 2.2f);
                        float& dst = canvas[y * kCanvasS + x];
                        dst = std::min(1.0f, std::max(dst, v));
                    }
                }
            }
        }
        update_normalized_28x28();
        dirty_tex = true;
    }

    // Raw-proportion canvas downsampling (NO auto-zoom!):
    //   Maps the actual 224x224 canvas space directly at fixed 8:1 scale (1 canvas block = 1 pixel)
    //   with translation-only centering so tiny dots/partial strokes keep their true small footprint
    //   while full drawings align cleanly with the 28x28 geometric shadows.
    bool update_normalized_28x28() {
        int minx = kCanvasS, miny = kCanvasS, maxx = -1, maxy = -1;
        for (int y = 0; y < kCanvasS; ++y) {
            for (int x = 0; x < kCanvasS; ++x) {
                if (canvas[y * kCanvasS + x] > 0.05f) {
                    minx = std::min(minx, x); maxx = std::max(maxx, x);
                    miny = std::min(miny, y); maxy = std::max(maxy, y);
                }
            }
        }
        norm28.fill(0.0f);
        if (maxx < minx || maxy < miny) return false;

        // Strictly fixed 1:8 scale (224 -> 28) — zero zoom!
        constexpr float kScale = 28.0f / static_cast<float>(kCanvasS); // 0.125
        float cx_can  = 0.5f * static_cast<float>(minx + maxx + 1);
        float cy_can  = 0.5f * static_cast<float>(miny + maxy + 1);
        float shift_x = 14.0f - cx_can * kScale;
        float shift_y = 14.0f - cy_can * kScale;
        shift_x = std::clamp(shift_x, -minx * kScale, 28.0f - (maxx + 1) * kScale);
        shift_y = std::clamp(shift_y, -miny * kScale, 28.0f - (maxy + 1) * kScale);

        std::array<float, 784> tmp{};
        for (int ty = 0; ty < 28; ++ty) {
            for (int tx = 0; tx < 28; ++tx) {
                float sx0 = (static_cast<float>(tx) - shift_x) / kScale;
                float sy0 = (static_cast<float>(ty) - shift_y) / kScale;
                int ix0 = static_cast<int>(std::floor(sx0));
                int iy0 = static_cast<int>(std::floor(sy0));
                // Exact 8x8 canvas-block area average (preserves true canvas-space occupancy)
                float acc = 0.0f;
                for (int ky = 0; ky < 8; ++ky) {
                    int iy = iy0 + ky;
                    if (iy < 0 || iy >= kCanvasS) continue;
                    const float* row = &canvas[iy * kCanvasS];
                    for (int kx = 0; kx < 8; ++kx) {
                        int ix = ix0 + kx;
                        if (ix >= 0 && ix < kCanvasS) acc += row[ix];
                    }
                }
                tmp[ty * 28 + tx] = acc * (1.0f / 64.0f);
            }
        }

        // Light anti-aliasing smoothing (sigma=0.65) + peak normalize to [0..1]
        gaussian_blur_28(tmp.data(), norm28.data(), 0.65f);
        float mx = 0.0f;
        for (int i = 0; i < 784; ++i) mx = std::max(mx, norm28[i]);
        if (mx > 1e-6f) {
            float inv = 1.0f / mx;
            for (int i = 0; i < 784; ++i) norm28[i] = std::clamp(norm28[i] * inv, 0.0f, 1.0f);
        }
        return true;
    }

    // Software rasterizer into kPanelW x kPanelH RGBA8 texture
    void put_px(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        if (x < 0 || x >= kPanelW || y < 0 || y >= kPanelH) return;
        size_t i = (static_cast<size_t>(y) * kPanelW + x) * 4;
        rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = a;
    }

    void fill_rect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        int x0 = std::max(0, x), y0 = std::max(0, y);
        int x1 = std::min(kPanelW, x + w), y1 = std::min(kPanelH, y + h);
        for (int py = y0; py < y1; ++py)
            for (int px = x0; px < x1; ++px)
                put_px(px, py, r, g, b, a);
    }

    void stroke_rect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        fill_rect(x, y, w, 1, r, g, b, a);
        fill_rect(x, y + h - 1, w, 1, r, g, b, a);
        fill_rect(x, y, 1, h, r, g, b, a);
        fill_rect(x + w - 1, y, 1, h, r, g, b, a);
    }

    void draw_text(int x, int y, const std::string& s, int scale, uint8_t r, uint8_t g, uint8_t b) {
        int cx = x;
        for (char ch : s) {
            uint8_t rows[7];
            glyph_5x7(ch, rows);
            for (int ry = 0; ry < 7; ++ry) {
                for (int rx = 0; rx < 5; ++rx) {
                    if (rows[ry] & (1u << (4 - rx))) {
                        fill_rect(cx + rx * scale, y + ry * scale, scale, scale, r, g, b, 255);
                    }
                }
            }
            cx += 6 * scale;
        }
    }

    void rebuild_texture(const MatchSummary& ms, uint32_t focused_class, uint32_t num_classes, uint32_t num_regions = 0) {
        // Semi-transparent dark glass sidebar background
        fill_rect(0, 0, kPanelW, kPanelH, 10, 15, 28, 228);
        fill_rect(kPanelW - 2, 0, 2, kPanelH, 42, 96, 148, 200);

        draw_text(20, 16, "GLLM SKETCH INFERENCE", 2, 125, 220, 255);

        // 1. Drawing canvas (224x224) + subtle 22x22 canonical scale guide frame (176x176)
        fill_rect(canvas_rect.x, canvas_rect.y, canvas_rect.w, canvas_rect.h, 8, 12, 22, 255);
        stroke_rect(canvas_rect.x + 24, canvas_rect.y + 24, 176, 176, 22, 38, 62, 255);
        for (int y = 0; y < kCanvasS; ++y) {
            for (int x = 0; x < kCanvasS; ++x) {
                float v = canvas[y * kCanvasS + x];
                if (v > 0.01f) {
                    uint8_t cr = static_cast<uint8_t>(std::min(255.0f, 40.0f + 215.0f * v));
                    uint8_t cg = static_cast<uint8_t>(std::min(255.0f, 160.0f + 95.0f * v));
                    uint8_t cb = 255;
                    put_px(canvas_rect.x + x, canvas_rect.y + y, cr, cg, cb, 255);
                }
            }
        }
        stroke_rect(canvas_rect.x - 1, canvas_rect.y - 1, canvas_rect.w + 2, canvas_rect.h + 2, 65, 165, 230, 255);

        // 2. Live 28x28 normalized preview
        draw_text(preview_rect.x, preview_rect.y + 70, "28x28", 1, 130, 175, 210);
        draw_text(preview_rect.x, preview_rect.y + 82, "CUDA", 1, 130, 175, 210);
        draw_text(preview_rect.x, preview_rect.y + 94, "INPUT", 1, 130, 175, 210);
        fill_rect(preview_rect.x, preview_rect.y, preview_rect.w, preview_rect.h, 6, 10, 18, 255);
        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                uint8_t v = static_cast<uint8_t>(norm28[y * 28 + x] * 255.0f);
                if (v > 4) fill_rect(preview_rect.x + 3 + x * 2, preview_rect.y + 3 + y * 2, 2, 2, v, static_cast<uint8_t>(std::min(255, v + 30)), 255, 255);
            }
        }
        stroke_rect(preview_rect.x, preview_rect.y, preview_rect.w, preview_rect.h, 60, 120, 175, 255);

        // 3. Buttons: [ CAPTURE ] and [ CLEAR ]
        fill_rect(btn_capture.x, btn_capture.y, btn_capture.w, btn_capture.h,
                  hover_capture ? 32 : 18, hover_capture ? 148 : 118, hover_capture ? 175 : 148, 255);
        stroke_rect(btn_capture.x, btn_capture.y, btn_capture.w, btn_capture.h, 110, 235, 255, 255);
        draw_text(btn_capture.x + 30, btn_capture.y + 10, "CAPTURE", 2, 245, 255, 255);

        fill_rect(btn_clear.x, btn_clear.y, btn_clear.w, btn_clear.h,
                  hover_clear ? 70 : 42, hover_clear ? 82 : 52, hover_clear ? 105 : 74, 255);
        stroke_rect(btn_clear.x, btn_clear.y, btn_clear.w, btn_clear.h, 120, 145, 180, 255);
        draw_text(btn_clear.x + 42, btn_clear.y + 10, "CLEAR", 2, 220, 232, 248);

        // 4. Match Summary Card
        fill_rect(18, 330, 300, 96, 16, 24, 42, 245);
        stroke_rect(18, 330, 300, 96, 52, 92, 138, 255);

        if (!ms.has_result) {
            draw_text(30, 348, "DRAW ANY DIGIT OR LETTER", 1, 175, 205, 235);
            draw_text(30, 366, "THEN CLICK CAPTURE (ENTER)", 1, 125, 195, 245);
            char buf[64];
            if (num_regions > num_classes) {
                std::snprintf(buf, sizeof(buf), "MODEL: %u CLASSES (%u SHADOWS)", num_classes, num_regions);
            } else {
                std::snprintf(buf, sizeof(buf), "MODEL: %u CLASSES IN 1B SPHERE", num_classes);
            }
            draw_text(30, 392, buf, 1, 135, 160, 195);
        } else {
            char hdr[64], sub1[64], sub2[64];
            if (ms.is_unknown || ms.ranked.empty()) {
                draw_text(28, 342, "UNKNOWN / UNTRAINED", 2, 255, 120, 90);
                float top_ov = ms.ranked.empty() ? 0.0f : ms.ranked[0].overlap_pct;
                std::snprintf(sub1, sizeof(sub1), "MAX OVERLAP: %.1f%% (<28%%)", top_ov);
                draw_text(28, 368, sub1, 1, 240, 180, 140);
            } else {
                const auto& w = ms.ranked[0];
                std::snprintf(hdr, sizeof(hdr), "MATCH: '%s'  #%.1f", w.label.c_str(), static_cast<float>(w.outer_idx));
                draw_text(28, 342, hdr, 2, 120, 255, 195);
                std::snprintf(sub1, sizeof(sub1), "OVERLAP %.1f%% | PROB %.1f%% | %.3f ms",
                              w.overlap_pct, w.rel_prob_pct, ms.cuda_ms);
                draw_text(28, 368, sub1, 1, 200, 235, 255);
            }
            std::snprintf(sub2, sizeof(sub2), "PRUNED (STAGE 1): %u / %u CLASSES",
                          ms.pruned_count, ms.total_classes);
            draw_text(28, 386, sub2, 1, 255, 185, 110);

            // Compact preview of pruned labels
            std::string plist = "DROPPED: ";
            for (size_t i = 0; i < ms.pruned_labels.size() && plist.size() < 42; ++i) {
                if (i > 0) plist += " ";
                plist += ms.pruned_labels[i];
            }
            if (ms.pruned_labels.size() > 14) plist += "..";
            draw_text(28, 404, plist, 1, 150, 165, 190);
        }

        // 5. Top-10 Ranked Overlap Bars
        draw_text(20, 442, "TOP 10 OVERLAP MATCHES (CLICK ROW):", 1, 145, 195, 235);
        for (int i = 0; i < 10; ++i) {
            const Rect& rr = row_rects[i];
            bool has_row = ms.has_result && i < static_cast<int>(ms.ranked.size());
            bool is_foc  = has_row && (ms.ranked[i].class_id == focused_class);
            uint8_t bg_r = is_foc ? 28 : (hover_row == i ? 24 : 14);
            uint8_t bg_g = is_foc ? 56 : (hover_row == i ? 38 : 20);
            uint8_t bg_b = is_foc ? 88 : (hover_row == i ? 60 : 35);
            fill_rect(rr.x, rr.y, rr.w, rr.h, bg_r, bg_g, bg_b, 235);
            stroke_rect(rr.x, rr.y, rr.w, rr.h, is_foc ? 95 : 34, is_foc ? 210 : 56, is_foc ? 255 : 86, 255);

            if (has_row) {
                const auto& row = ms.ranked[i];
                char left_txt[32], right_txt[32];
                std::snprintf(left_txt, sizeof(left_txt), "#%-2d %s", i + 1, row.label.c_str());
                draw_text(rr.x + 8, rr.y + 9, left_txt, 1, 240, 248, 255);

                // Overlap bar
                int bar_x = rr.x + 68, bar_y = rr.y + 8, bar_w = 112, bar_h = 11;
                fill_rect(bar_x, bar_y, bar_w, bar_h, 24, 34, 52, 255);
                int fill_w = std::clamp(static_cast<int>(bar_w * (row.overlap_pct / 100.0f)), 0, bar_w);
                uint8_t br = (i == 0) ? 75 : 55;
                uint8_t bg = (i == 0) ? 235 : 175;
                uint8_t bb = (i == 0) ? 185 : 240;
                fill_rect(bar_x, bar_y, fill_w, bar_h, br, bg, bb, 255);

                std::snprintf(right_txt, sizeof(right_txt), "%4.1f%% (%2.0f%%)", row.overlap_pct, row.rel_prob_pct);
                draw_text(rr.x + 188, rr.y + 9, right_txt, 1, 200, 230, 250);
            }
        }

        // Footer hint
        draw_text(20, 800, "DRAG RIGHT: ORBIT 3D | Z: ZOOM", 1, 115, 150, 185);
        draw_text(20, 816, "H: HEATMAP | UP/DN: OUTER SHELL", 1, 115, 150, 185);
        dirty_tex = false;
    }
};

} // namespace gllm
