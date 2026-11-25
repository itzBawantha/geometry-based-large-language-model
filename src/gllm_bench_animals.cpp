// =============================================================================
// gllm_bench_animals — Zero-BBox Pure-Image Bottom-Up Cat vs. Dog Benchmark
//
// Evaluates a trained hierarchical .gllm animals model strictly on RAW,
// UNCROPPED unseen images — ZERO ground-truth bounding boxes, ZERO landmarks
// supplied at inference time!
//
// Benchmarks two pure-image zero-bbox inference pipelines on CUDA:
//   [Mode 1] Unconstrained Global Multi-Scale Scan (0 bboxes, 0 layout priors)
//            Slides 28x28 contour windows across the entire raw image over 5
//            scales, scoring all 6 child feature manifolds on GPU to detect
//            which species' shadows best match anywhere in the image.
//   [Mode 2] Hierarchical Bottom-Up Constellation Scan (0 bboxes at test time!)
//            Stage 1: Multi-scale GPU scan proposes Top-12 candidate face boxes
//                     via L2 cats/face + dogs/face shadow overlap.
//            Stage 2: Inside each candidate face, GPU searches local 3x3x2
//                     grids for all 6 L3 child manifolds (ear_right, ear_left,
//                     eye_right, eye_left, nose, mouth) using species-neutral
//                     anatomical relative geometry from the train split only.
//            Stage 3: Bottom-up child-manifold consensus locks onto the best
//                     face, counts how many of the 6 child manifolds map to
//                     Cat vs. Dog, and fuses L3 -> L2 -> L1 shadow margins!
//
// Also audits zero-bbox face localization IoU against ground-truth face boxes
// (diagnostic only, never used for inference) and optionally exports a 4x4
// visual verification sheet showing detected faces & 6 child features.
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"
#include "../include/oriented_shadow.hpp"

#include <png.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static constexpr uint32_t kPatchDim = 28u;
static constexpr uint32_t kPatchPix = kPatchDim * kPatchDim; // 784
static constexpr uint32_t kNumComps = 8u; // 0:full, 1:face, 2..7: 6 facial features

static const std::array<std::string, kNumComps> kCompNames = {
    "full", "face", "ear_right", "ear_left", "eye_right", "eye_left", "nose", "mouth"
};

struct BBox {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

struct ManifestRow {
    bool is_test = false;
    int  species = 0; // 0 = cat, 1 = dog
    std::string image_path;
    BBox face_box;
    std::array<BBox, 6> feat_boxes;
};

struct RelPrior {
    float cx = 0.5f, cy = 0.5f, w = 0.25f, h = 0.25f;
};

struct DecodedImg {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgb;       // w * h * 3 (only kept for viz samples or small RAM)
    std::vector<double>  integ_lum; // (w+1) * (h+1) summed-area table of luminance [0..1]
    bool ok = false;
};

struct CandidateBox {
    float x0, y0, x1, y1;
    float score;
};

static bool parse_bbox(const std::string& s, BBox& b) {
    float v[4] = {};
    int k = 0;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',') && k < 4) v[k++] = std::strtof(tok.c_str(), nullptr);
    if (k != 4) return false;
    b.x0 = v[0]; b.y0 = v[1]; b.x1 = v[2]; b.y1 = v[3];
    return true;
}

static bool load_manifest(const std::string& path, std::vector<ManifestRow>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cols;
        size_t start = 0;
        while (true) {
            size_t tab = line.find('\t', start);
            if (tab == std::string::npos) { cols.push_back(line.substr(start)); break; }
            cols.push_back(line.substr(start, tab - start));
            start = tab + 1;
        }
        if (cols.size() < 10) continue;
        ManifestRow r;
        r.is_test    = (cols[0] == "test");
        r.species    = (cols[1] == "dog") ? 1 : 0;
        r.image_path = cols[2];
        if (!parse_bbox(cols[3], r.face_box)) continue;
        bool ok = true;
        for (int f = 0; f < 6; ++f) {
            if (!parse_bbox(cols[4 + f], r.feat_boxes[f])) { ok = false; break; }
        }
        if (ok) out.push_back(std::move(r));
    }
    return !out.empty();
}

static bool load_png_integral(const std::string& path, bool keep_rgb, DecodedImg& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    png_byte sig[8];
    if (std::fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8) != 0) {
        std::fclose(fp);
        return false;
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); std::fclose(fp); return false; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return false;
    }
    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_uint_32 w = png_get_image_width(png, info);
    png_uint_32 h = png_get_image_height(png, info);
    int bit_depth  = png_get_bit_depth(png, info);
    int color_type = png_get_color_type(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (color_type & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
    png_read_update_info(png, info);

    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3u);
    std::vector<png_bytep> rows(h);
    for (png_uint_32 y = 0; y < h; ++y) rows[y] = rgb.data() + static_cast<size_t>(y) * w * 3u;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    out.w = w;
    out.h = h;
    const size_t stride = static_cast<size_t>(w) + 1u;
    out.integ_lum.assign(stride * (static_cast<size_t>(h) + 1u), 0.0);
    for (uint32_t y = 0; y < h; ++y) {
        double row_sum = 0.0;
        const uint8_t* rp = rgb.data() + static_cast<size_t>(y) * w * 3u;
        double* dst_row   = out.integ_lum.data() + static_cast<size_t>(y + 1u) * stride;
        const double* prv = out.integ_lum.data() + static_cast<size_t>(y) * stride;
        for (uint32_t x = 0; x < w; ++x) {
            double l = (0.299 * rp[x * 3 + 0] + 0.587 * rp[x * 3 + 1] + 0.114 * rp[x * 3 + 2]) * (1.0 / 255.0);
            row_sum += l;
            dst_row[x + 1u] = prv[x + 1u] + row_sum;
        }
    }
    if (keep_rgb) out.rgb = std::move(rgb);
    out.ok = (w > 0 && h > 0);
    return out.ok;
}

// Ultra-fast O(784) exact area-averaged 28x28 geometric contour shadow extraction via integral image!
// Identical math to gllm_animals_train --shadow-mode contour --no-aspect (fills inner 26x26 cleanly).
static void extract_contour_fast(const DecodedImg& img,
                                 float fx0, float fy0, float fx1, float fy1,
                                 uint8_t* out_shadow) {
    std::memset(out_shadow, 0, kPatchPix);
    int W = static_cast<int>(img.w);
    int H = static_cast<int>(img.h);
    int x0 = std::clamp(static_cast<int>(std::floor(fx0)), 0, W - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(fy0)), 0, H - 1);
    int x1 = std::clamp(static_cast<int>(std::ceil(fx1)), x0 + 1, W);
    int y1 = std::clamp(static_cast<int>(std::ceil(fy1)), y0 + 1, H);
    int cw = x1 - x0;
    int ch = y1 - y0;

    constexpr int kInner = 26;
    constexpr int ox = 1, oy = 1;
    const size_t stride = static_cast<size_t>(W) + 1u;
    const double* sat = img.integ_lum.data();

    float lum[kInner * kInner];
    double sum_l = 0.0, sum_l2 = 0.0;
    int ix0_arr[kInner], ix1_arr[kInner];
    for (int dx = 0; dx < kInner; ++dx) {
        int a = std::clamp(x0 + (dx * cw) / kInner, x0, x1 - 1);
        int b = std::clamp(x0 + ((dx + 1) * cw + kInner - 1) / kInner, a + 1, x1);
        ix0_arr[dx] = a;
        ix1_arr[dx] = b;
    }

    for (int dy = 0; dy < kInner; ++dy) {
        int iy0 = std::clamp(y0 + (dy * ch) / kInner, y0, y1 - 1);
        int iy1 = std::clamp(y0 + ((dy + 1) * ch + kInner - 1) / kInner, iy0 + 1, y1);
        const double* r0 = sat + static_cast<size_t>(iy0) * stride;
        const double* r1 = sat + static_cast<size_t>(iy1) * stride;
        int dh = iy1 - iy0;
        for (int dx = 0; dx < kInner; ++dx) {
            int ix0 = ix0_arr[dx];
            int ix1 = ix1_arr[dx];
            double area = static_cast<double>(dh * (ix1 - ix0));
            double s = r1[ix1] - r0[ix1] - r1[ix0] + r0[ix0];
            float l = static_cast<float>(s / area);
            lum[dy * kInner + dx] = l;
            sum_l  += l;
            sum_l2 += static_cast<double>(l) * l;
        }
    }

    constexpr double inv_n = 1.0 / (kInner * kInner);
    float mean_l = static_cast<float>(sum_l * inv_n);
    float var_l  = static_cast<float>(std::max(0.0, sum_l2 * inv_n - static_cast<double>(mean_l) * mean_l));
    float raw_sd = std::sqrt(var_l);
    // Reject flat, featureless blank walls/sky/solid backgrounds cleanly (ZNCC = 0)
    if (raw_sd < 0.035f) return;
    float inv_sd = 1.0f / raw_sd;

    float zf[kInner * kInner];
    for (int i = 0; i < kInner * kInner; ++i) {
        zf[i] = std::clamp((lum[i] - mean_l) * inv_sd, -3.0f, 3.0f);
    }

    auto sample_z = [&](int x, int y) -> float {
        x = std::clamp(x, 0, kInner - 1);
        y = std::clamp(y, 0, kInner - 1);
        return zf[y * kInner + x];
    };

    float raw_val[kInner * kInner];
    double sv = 0.0, sv2 = 0.0;
    for (int dy = 0; dy < kInner; ++dy) {
        for (int dx = 0; dx < kInner; ++dx) {
            float tl = sample_z(dx - 1, dy - 1), tc = sample_z(dx, dy - 1), tr = sample_z(dx + 1, dy - 1);
            float ml = sample_z(dx - 1, dy    ),                             mr = sample_z(dx + 1, dy    );
            float bl = sample_z(dx - 1, dy + 1), bc = sample_z(dx, dy + 1), br = sample_z(dx + 1, dy + 1);
            float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
            float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
            float gmag = std::sqrt(gx * gx + gy * gy) * 0.25f;
            float edge = std::tanh(gmag * 0.85f);
            float dark = std::clamp(-zf[dy * kInner + dx] * 0.42f, 0.0f, 1.0f);
            float val  = std::clamp(0.16f + 0.52f * edge + 0.32f * dark, 0.0f, 1.0f);
            raw_val[dy * kInner + dx] = val;
            sv  += val;
            sv2 += static_cast<double>(val) * val;
        }
    }

    // Zero-mean unit-variance encoding on 26x26 interior so dot(u8, zero_mean_tmpl) / ||u8|| == exact ZNCC!
    float mv = static_cast<float>(sv * inv_n);
    float v_sd = std::sqrt(static_cast<float>(std::max(0.0, sv2 * inv_n - static_cast<double>(mv) * mv)));
    if (v_sd < 0.02f) return;
    float inv_vsd = 1.0f / v_sd;
    for (int dy = 0; dy < kInner; ++dy) {
        uint8_t* dst = out_shadow + static_cast<size_t>(oy + dy) * kPatchDim + ox;
        for (int dx = 0; dx < kInner; ++dx) {
            float zn = std::clamp((raw_val[dy * kInner + dx] - mv) * inv_vsd, -2.8f, 2.8f);
            dst[dx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(128.0f + 42.0f * zn)), 1, 255));
        }
    }
}

static float box_iou(const BBox& a, const BBox& b) {
    float ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    float ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    float iw = std::max(0.0f, ix1 - ix0), ih = std::max(0.0f, iy1 - iy0);
    float inter = iw * ih;
    float aa = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
    float ba = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
    return (aa + ba - inter > 1e-5f) ? (inter / (aa + ba - inter)) : 0.0f;
}

// Generate multi-scale sliding face proposal windows over a raw uncropped image (ZERO ground-truth bboxes!)
static std::vector<BBox> make_face_proposals(uint32_t W, uint32_t H) {
    std::vector<BBox> out;
    float fw = static_cast<float>(W);
    float fh = static_cast<float>(H);
    float mn = std::min(fw, fh);
    // Cover face sizes from 28% to 94% of min(W,H) with overlapping grid steps
    const float scales[5] = {0.28f, 0.40f, 0.55f, 0.72f, 0.92f};
    for (float sc : scales) {
        float side = mn * sc;
        float step = std::max(12.0f, side * 0.22f);
        int nx = std::max(1, static_cast<int>(std::floor((fw - side) / step)) + 1);
        int ny = std::max(1, static_cast<int>(std::floor((fh - side) / step)) + 1);
        float rx = (nx > 1) ? (fw - side) / static_cast<float>(nx - 1) : 0.0f;
        float ry = (ny > 1) ? (fh - side) / static_cast<float>(ny - 1) : 0.0f;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                float x0 = (nx > 1) ? ix * rx : 0.5f * (fw - side);
                float y0 = (ny > 1) ? iy * ry : 0.5f * (fh - side);
                out.push_back({x0, y0, x0 + side, y0 + side});
            }
        }
    }
    // Also add full-image centered square & full frame itself for tight close-up portraits
    out.push_back({0.0f, 0.0f, fw, fh});
    return out;
}

// Generate unconstrained global multi-scale windows for Mode 1 (blind whole-image feature scan)
static std::vector<BBox> make_global_feature_windows(uint32_t W, uint32_t H) {
    std::vector<BBox> out;
    float fw = static_cast<float>(W);
    float fh = static_cast<float>(H);
    float mn = std::min(fw, fh);
    const float scales[4] = {0.14f, 0.22f, 0.34f, 0.50f};
    for (float sc : scales) {
        float side = mn * sc;
        float step = std::max(10.0f, side * 0.45f);
        int nx = std::max(1, static_cast<int>(std::floor((fw - side) / step)) + 1);
        int ny = std::max(1, static_cast<int>(std::floor((fh - side) / step)) + 1);
        float rx = (nx > 1) ? (fw - side) / static_cast<float>(nx - 1) : 0.0f;
        float ry = (ny > 1) ? (fh - side) / static_cast<float>(ny - 1) : 0.0f;
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                float x0 = (nx > 1) ? ix * rx : 0.5f * (fw - side);
                float y0 = (ny > 1) ? iy * ry : 0.5f * (fh - side);
                out.push_back({x0, y0, x0 + side, y0 + side});
            }
        }
    }
    return out;
}

// Draw a rectangle into an RGB24 canvas for the visual zero-bbox verification sheet
static void draw_rect(std::vector<uint8_t>& img, int W, int H,
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b, int thick = 2) {
    x0 = std::clamp(x0, 0, W - 1); x1 = std::clamp(x1, 0, W - 1);
    y0 = std::clamp(y0, 0, H - 1); y1 = std::clamp(y1, 0, H - 1);
    for (int t = 0; t < thick; ++t) {
        int a0 = std::clamp(x0 + t, 0, W - 1), a1 = std::clamp(x1 - t, 0, W - 1);
        int b0 = std::clamp(y0 + t, 0, H - 1), b1 = std::clamp(y1 - t, 0, H - 1);
        for (int x = a0; x <= a1; ++x) {
            size_t i0 = (static_cast<size_t>(b0) * W + x) * 3u;
            size_t i1 = (static_cast<size_t>(b1) * W + x) * 3u;
            img[i0] = r; img[i0 + 1] = g; img[i0 + 2] = b;
            img[i1] = r; img[i1 + 1] = g; img[i1 + 2] = b;
        }
        for (int y = b0; y <= b1; ++y) {
            size_t i0 = (static_cast<size_t>(y) * W + a0) * 3u;
            size_t i1 = (static_cast<size_t>(y) * W + a1) * 3u;
            img[i0] = r; img[i0 + 1] = g; img[i0 + 2] = b;
            img[i1] = r; img[i1 + 1] = g; img[i1 + 2] = b;
        }
    }
}

static void extract_shadow_fast(const DecodedImg& img,
                                float fx0, float fy0, float fx1, float fy1,
                                bool use_hog,
                                uint8_t* out_shadow) {
    if (use_hog) {
        gllm::extract_hog_from_integral(img.integ_lum.data(), img.w, img.h,
                                        fx0, fy0, fx1, fy1, true, out_shadow);
        return;
    }
    extract_contour_fast(img, fx0, fy0, fx1, fy1, out_shadow);
}

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_bench_animals --model <models/animals_trained.gllm> --manifest <data/cat_dog_split.tsv>\n"
              << "                     [--shadow-mode hog|contour] [--top-faces 12] [--viz-out /tmp/zero_bbox_audit.ppm]\n";
}

int main(int argc, char** argv) {
    std::string model_path    = "models/animals_trained.gllm";
    std::string manifest_path = "data/cat_dog_split.tsv";
    std::string viz_out_path;
    bool use_hog = false;
    int top_faces_k = 12;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (a == "--top-faces" && i + 1 < argc) top_faces_k = std::clamp(std::atoi(argv[++i]), 1, 24);
        else if (a == "--viz-out" && i + 1 < argc) viz_out_path = argv[++i];
        else if (a == "--shadow-mode" && i + 1 < argc) use_hog = (std::string(argv[++i]) == "hog");
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    // 1. Load trained .gllm model & extract L2-normalized shadow templates for all 16 animal compartments
    gllm::GeometricModel model;
    if (!model.load(model_path)) {
        std::cerr << "[ERROR] Cannot load model: " << model_path << "\n";
        return 1;
    }

    std::vector<uint32_t> comp_regs[2][kNumComps];
    for (uint32_t r = 0; r < model.classes.size(); ++r) {
        const auto& c = model.classes[r];
        if (c.width != kPatchDim || c.height != kPatchDim) continue;
        for (int sp = 0; sp < 2; ++sp) {
            std::string prefix = (sp == 0) ? "cat_" : "dog_";
            for (uint32_t comp = 0; comp < kNumComps; ++comp) {
                if (c.label == prefix + kCompNames[comp]) {
                    comp_regs[sp][comp].push_back(r);
                }
            }
        }
    }
    for (int sp = 0; sp < 2; ++sp) {
        for (uint32_t comp = 0; comp < kNumComps; ++comp) {
            if (comp_regs[sp][comp].empty()) {
                std::cerr << "[ERROR] Model missing templates for " << (sp == 0 ? "cat_" : "dog_")
                          << kCompNames[comp] << "\n";
                return 1;
            }
        }
    }

    // Pack L2-normalized template tables per compartment: [cat templates..., dog templates...]
    // PLUS 1 pure universal canonical archetype template per compartment (exact sum of all 5,129 train shadows!)
    // Why: Averaging all faces/parts cancels random background clutter to ZERO for rock-solid Stage-1 localization,
    // while keeping all K=16 fine-grained species clusters for Stage-3 Cat vs. Dog identification!
    std::array<std::vector<float>, kNumComps> comp_tmpls;
    std::array<uint32_t, kNumComps> comp_ncat{}, comp_ndog{};
    std::array<std::vector<float>, kNumComps> comp_canon_tmpl; // 2 templates: [0]=cat_avg, [1]=dog_avg
    std::array<std::vector<float>, kNumComps> comp_univ_tmpl;  // 1 template: universal animal_avg (cat+dog)
    auto make_zncc_vec = [&](const std::vector<double>& raw, float* dst) {
        uint32_t lo = use_hog ? 0u : 1u;
        uint32_t hi = use_hog ? 27u : 26u;
        double mean = 0.0;
        int cnt = 0;
        for (uint32_t y = lo; y <= hi; ++y)
            for (uint32_t x = lo; x <= hi; ++x) { mean += raw[y * kPatchDim + x]; cnt++; }
        mean /= std::max(1, cnt);
        double n2 = 0.0;
        for (uint32_t y = lo; y <= hi; ++y)
            for (uint32_t x = lo; x <= hi; ++x) {
                double d = raw[y * kPatchDim + x] - mean;
                n2 += d * d;
            }
        float inv = n2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
        std::fill(dst, dst + kPatchPix, 0.0f);
        for (uint32_t y = lo; y <= hi; ++y)
            for (uint32_t x = lo; x <= hi; ++x) {
                size_t p = y * kPatchDim + x;
                dst[p] = static_cast<float>(raw[p] - mean) * inv;
            }
    };
    for (uint32_t comp = 0; comp < kNumComps; ++comp) {
        uint32_t nc = static_cast<uint32_t>(comp_regs[0][comp].size());
        uint32_t nd = static_cast<uint32_t>(comp_regs[1][comp].size());
        comp_ncat[comp] = nc;
        comp_ndog[comp] = nd;
        comp_tmpls[comp].resize(static_cast<size_t>(nc + nd) * kPatchPix, 0.0f);
        comp_canon_tmpl[comp].resize(2u * kPatchPix, 0.0f);
        comp_univ_tmpl[comp].resize(kPatchPix, 0.0f);

        std::vector<double> sum_cat(kPatchPix, 0.0), sum_dog(kPatchPix, 0.0), sum_all(kPatchPix, 0.0);
        uint64_t tot_c = 0, tot_d = 0;
        auto pack = [&](uint32_t slot, uint32_t r, int sp) {
            const auto& rec = model.classes[r];
            std::vector<double> rv(kPatchPix);
            for (uint32_t p = 0; p < kPatchPix; ++p) {
                rv[p] = static_cast<double>(rec.shadow_sum[p]);
                if (sp == 0) sum_cat[p] += rv[p]; else sum_dog[p] += rv[p];
            }
            if (sp == 0) tot_c += rec.train_count; else tot_d += rec.train_count;
            make_zncc_vec(rv, &comp_tmpls[comp][static_cast<size_t>(slot) * kPatchPix]);
        };
        for (uint32_t k = 0; k < nc; ++k) pack(k, comp_regs[0][comp][k], 0);
        for (uint32_t k = 0; k < nd; ++k) pack(nc + k, comp_regs[1][comp][k], 1);

        // Equal 50/50 cat+dog universal archetype (eliminates 2:1 dog dataset imbalance during localization!)
        for (uint32_t p = 0; p < kPatchPix; ++p) {
            sum_cat[p] /= std::max<uint64_t>(1, tot_c);
            sum_dog[p] /= std::max<uint64_t>(1, tot_d);
            sum_all[p]  = 0.5 * (sum_cat[p] + sum_dog[p]);
        }
        make_zncc_vec(sum_cat, &comp_canon_tmpl[comp][0]);
        make_zncc_vec(sum_dog, &comp_canon_tmpl[comp][kPatchPix]);
        make_zncc_vec(sum_all, &comp_univ_tmpl[comp][0]);
    }

    // 2. Load manifest: compute species-neutral relative anatomical priors strictly on TRAIN split,
    //    and collect all 1,283 strictly unseen TEST image paths!
    std::vector<ManifestRow> all_rows;
    if (!load_manifest(manifest_path, all_rows)) {
        std::cerr << "[ERROR] Cannot load manifest: " << manifest_path << "\n";
        return 1;
    }

    // Species-neutral 50/50 averaged relative geometry of the 6 features inside a face box (TRAIN ONLY!)
    std::array<RelPrior, 6> feat_prior{};
    std::vector<ManifestRow> train_cat_rows, train_dog_rows;
    {
        double sum_cx[2][6]{}, sum_cy[2][6]{}, sum_w[2][6]{}, sum_h[2][6]{};
        size_t cnt[2]{};
        for (const auto& r : all_rows) {
            if (r.is_test) continue; // STRICTLY TRAIN ONLY!
            if (r.species == 0) train_cat_rows.push_back(r);
            else train_dog_rows.push_back(r);
            float fw = std::max(1.0f, r.face_box.x1 - r.face_box.x0);
            float fh = std::max(1.0f, r.face_box.y1 - r.face_box.y0);
            cnt[r.species]++;
            for (int f = 0; f < 6; ++f) {
                const auto& fb = r.feat_boxes[f];
                double rcx = (0.5 * (fb.x0 + fb.x1) - r.face_box.x0) / fw;
                double rcy = (0.5 * (fb.y0 + fb.y1) - r.face_box.y0) / fh;
                double rw  = std::max(0.05, static_cast<double>(fb.x1 - fb.x0) / fw);
                double rh  = std::max(0.05, static_cast<double>(fb.y1 - fb.y0) / fh);
                sum_cx[r.species][f] += rcx;
                sum_cy[r.species][f] += rcy;
                sum_w[r.species][f]  += rw;
                sum_h[r.species][f]  += rh;
            }
        }
        std::cout << "[PRIOR]  Species-neutral anatomical geometry inside face (computed strictly on train split):\n";
        for (int f = 0; f < 6; ++f) {
            double cx = 0.5 * (sum_cx[0][f] / cnt[0] + sum_cx[1][f] / cnt[1]);
            double cy = 0.5 * (sum_cy[0][f] / cnt[0] + sum_cy[1][f] / cnt[1]);
            double rw = 0.5 * (sum_w[0][f]  / cnt[0] + sum_w[1][f]  / cnt[1]);
            double rh = 0.5 * (sum_h[0][f]  / cnt[0] + sum_h[1][f]  / cnt[1]);
            feat_prior[f] = {static_cast<float>(cx), static_cast<float>(cy),
                             static_cast<float>(rw), static_cast<float>(rh)};
            std::printf("         %-10s : rel_center=(%.2f, %.2f)  rel_size=(%.2f x %.2f)\n",
                        kCompNames[2 + f].c_str(), cx, cy, rw, rh);
        }
    }

    // Build K=32 Background / Non-Face Negative Shadow Templates strictly from TRAIN non-face crops!
    // Enables pure-geometry foreground-vs-background discrimination: face_ev = max(cat,dog) - max(bg)
    std::vector<float> bg_tmpls;
    constexpr uint32_t kBgCount = 32u;
    {
        std::vector<ManifestRow> bg_srcs;
        for (size_t i = 0; i < std::min<size_t>(128, train_cat_rows.size()); ++i) bg_srcs.push_back(train_cat_rows[i * 7 % train_cat_rows.size()]);
        for (size_t i = 0; i < std::min<size_t>(128, train_dog_rows.size()); ++i) bg_srcs.push_back(train_dog_rows[i * 11 % train_dog_rows.size()]);
        std::vector<std::array<float, kPatchPix>> raw_bg;
        raw_bg.reserve(512);
        for (const auto& r : bg_srcs) {
            DecodedImg d;
            if (!load_png_integral(r.image_path, false, d)) continue;
            float fw = static_cast<float>(d.w), fh = static_cast<float>(d.h);
            float side = std::min(fw, fh) * 0.35f;
            const float cx_cands[4] = {0.15f, 0.85f, 0.15f, 0.85f};
            const float cy_cands[4] = {0.15f, 0.15f, 0.85f, 0.85f};
            for (int k = 0; k < 4; ++k) {
                BBox b{cx_cands[k] * fw - 0.5f * side, cy_cands[k] * fh - 0.5f * side,
                       cx_cands[k] * fw + 0.5f * side, cy_cands[k] * fh + 0.5f * side};
                if (box_iou(b, r.face_box) > 0.05f) continue;
                uint8_t sh[kPatchPix];
                extract_shadow_fast(d, b.x0, b.y0, b.x1, b.y1, use_hog, sh);
                uint32_t lo = use_hog ? 0u : 1u;
                uint32_t hi = use_hog ? 27u : 26u;
                double mean = 0.0, n2 = 0.0;
                int bcnt = 0;
                for (uint32_t y = lo; y <= hi; ++y)
                    for (uint32_t x = lo; x <= hi; ++x) { mean += sh[y * kPatchDim + x]; bcnt++; }
                mean /= std::max(1, bcnt);
                std::array<float, kPatchPix> v{};
                for (uint32_t y = lo; y <= hi; ++y)
                    for (uint32_t x = lo; x <= hi; ++x) {
                        double dv = sh[y * kPatchDim + x] - mean;
                        v[y * kPatchDim + x] = static_cast<float>(dv);
                        n2 += dv * dv;
                    }
                if (n2 < 100.0) continue;
                float inv = static_cast<float>(1.0 / std::sqrt(n2));
                for (auto& e : v) e *= inv;
                raw_bg.push_back(v);
                if (raw_bg.size() >= 512) break;
            }
            if (raw_bg.size() >= 512) break;
        }
        // Farthest-point select kBgCount diverse background templates
        uint32_t nb = std::min<uint32_t>(kBgCount, static_cast<uint32_t>(raw_bg.size()));
        bg_tmpls.resize(static_cast<size_t>(std::max(1u, nb)) * kPatchPix, 0.0f);
        if (nb > 0) {
            std::memcpy(&bg_tmpls[0], raw_bg[0].data(), kPatchPix * sizeof(float));
            std::vector<float> max_s(raw_bg.size(), -1e9f);
            for (uint32_t c = 1; c < nb; ++c) {
                const float* prev = &bg_tmpls[static_cast<size_t>(c - 1) * kPatchPix];
                size_t far_i = 0; float min_v = 1e9f;
                for (size_t s = 0; s < raw_bg.size(); ++s) {
                    float dp = 0.0f;
                    for (uint32_t p = 0; p < kPatchPix; ++p) dp += raw_bg[s][p] * prev[p];
                    if (dp > max_s[s]) max_s[s] = dp;
                    if (max_s[s] < min_v) { min_v = max_s[s]; far_i = s; }
                }
                std::memcpy(&bg_tmpls[static_cast<size_t>(c) * kPatchPix], raw_bg[far_i].data(), kPatchPix * sizeof(float));
            }
        }
        std::printf("[BG-NEG] Built %u diverse non-face background templates from %zu train crops\n", nb, raw_bg.size());
    }

    std::vector<ManifestRow> test_rows;
    for (const auto& r : all_rows) {
        if (r.is_test) test_rows.push_back(r);
    }
    const uint32_t N_test = static_cast<uint32_t>(test_rows.size());

    // Pick 8 cats & 8 dogs for optional visual verification sheet (keep their RGB pixels)
    std::vector<bool> keep_rgb(N_test, false);
    std::vector<uint32_t> viz_indices;
    if (!viz_out_path.empty()) {
        int c_cnt = 0, d_cnt = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            if (test_rows[i].species == 0 && c_cnt < 8) { keep_rgb[i] = true; viz_indices.push_back(i); c_cnt++; }
            if (test_rows[i].species == 1 && d_cnt < 8) { keep_rgb[i] = true; viz_indices.push_back(i); d_cnt++; }
        }
    }

    // 3. Decode all 1,283 strictly unseen RAW PNG images across 16 CPU threads (NO bboxes used!)
    auto t0 = std::chrono::steady_clock::now();
    uint32_t n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<DecodedImg> imgs(N_test);
    std::atomic<uint32_t> next_i{0};
    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < n_threads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                uint32_t i = next_i.fetch_add(1);
                if (i >= N_test) break;
                load_png_integral(test_rows[i].image_path, keep_rgb[i], imgs[i]);
            }
        });
    }
    for (auto& w : workers) w.join();
    double dec_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[LOAD]   Decoded %u strictly unseen RAW uncropped PNGs in %.2fs (ZERO bboxes passed to inference!)\n",
                N_test, dec_sec);

    // 4. Initialize CUDA Driver & upload all 8 compartment template tables once to VRAM
    CU_CHECK(cuInit(0));
    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    char gpu_name[256] = {};
    CU_CHECK(cuDeviceGetName(gpu_name, sizeof(gpu_name), dev));
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod = gllm::load_gllm_ptx_module();
    CUfunction fn_eval = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_eval, mod, "eval_score_shadows"));
    CUevent ev_s = nullptr, ev_e = nullptr;
    CU_CHECK(cuEventCreate(&ev_s, 0));
    CU_CHECK(cuEventCreate(&ev_e, 0));

    std::array<CUdeviceptr, kNumComps> d_tmpls{}, d_canon{}, d_univ{};
    for (uint32_t comp = 0; comp < kNumComps; ++comp) {
        CU_CHECK(cuMemAlloc_v2(&d_tmpls[comp], comp_tmpls[comp].size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_tmpls[comp], comp_tmpls[comp].data(), comp_tmpls[comp].size() * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&d_canon[comp], comp_canon_tmpl[comp].size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_canon[comp], comp_canon_tmpl[comp].data(), comp_canon_tmpl[comp].size() * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&d_univ[comp],  comp_univ_tmpl[comp].size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_univ[comp],  comp_univ_tmpl[comp].data(), comp_univ_tmpl[comp].size() * sizeof(float)));
    }
    uint32_t bg_cnt = static_cast<uint32_t>(bg_tmpls.size() / kPatchPix);
    CUdeviceptr d_bg = 0;
    CU_CHECK(cuMemAlloc_v2(&d_bg, bg_tmpls.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_bg, bg_tmpls.data(), bg_tmpls.size() * sizeof(float)));
    std::cout << "[CUDA]   Device: " << gpu_name << " | Loaded " << model.classes.size()
              << " trained animal shadow templates + canonical archetypes + " << bg_cnt << " background templates\n";

    // Helper: score batch against canonical archetype templates (1 universal OR 2 cat/dog avg) on GPU
    auto gpu_score_ptr = [&](const std::vector<uint8_t>& patches_u8, uint32_t M,
                             CUdeviceptr d_t, uint32_t T,
                             std::vector<float>& out_scores, float& gpu_ms) {
        out_scores.resize(static_cast<size_t>(M) * T);
        constexpr uint32_t CHUNK = 65536u;
        CUdeviceptr d_in = 0, d_sc = 0;
        uint32_t alloc_m = std::min(M, CHUNK);
        CU_CHECK(cuMemAlloc_v2(&d_in, static_cast<size_t>(alloc_m) * kPatchPix));
        CU_CHECK(cuMemAlloc_v2(&d_sc, static_cast<size_t>(alloc_m) * T * sizeof(float)));
        for (uint32_t off = 0; off < M; off += CHUNK) {
            uint32_t cnt = std::min(CHUNK, M - off);
            CU_CHECK(cuMemcpyHtoD_v2(d_in, &patches_u8[static_cast<size_t>(off) * kPatchPix],
                                     static_cast<size_t>(cnt) * kPatchPix));
            uint32_t pp = kPatchPix, pairs = cnt * T;
            void* args[] = {&d_in, &d_t, &d_sc, &pp, &T, &cnt};
            CU_CHECK(cuEventRecord(ev_s, nullptr));
            CU_CHECK(cuLaunchKernel(fn_eval, (pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
            CU_CHECK(cuEventRecord(ev_e, nullptr));
            CU_CHECK(cuEventSynchronize(ev_e));
            float ms = 0.0f;
            CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
            gpu_ms += ms;
            CU_CHECK(cuMemcpyDtoH_v2(&out_scores[static_cast<size_t>(off) * T], d_sc,
                                     static_cast<size_t>(cnt) * T * sizeof(float)));
        }
        CU_CHECK(cuMemFree_v2(d_in));
        CU_CHECK(cuMemFree_v2(d_sc));
    };

    // Helper: score batch against background negative templates -> max_bg[i]
    auto gpu_score_bg = [&](const std::vector<uint8_t>& patches_u8, uint32_t M,
                            std::vector<float>& out_bg, float& gpu_ms) {
        out_bg.assign(M, 0.0f);
        if (bg_cnt == 0) return;
        constexpr uint32_t CHUNK = 65536u;
        CUdeviceptr d_in = 0, d_sc = 0;
        uint32_t alloc_m = std::min(M, CHUNK);
        CU_CHECK(cuMemAlloc_v2(&d_in, static_cast<size_t>(alloc_m) * kPatchPix));
        CU_CHECK(cuMemAlloc_v2(&d_sc, static_cast<size_t>(alloc_m) * bg_cnt * sizeof(float)));
        std::vector<float> h_sc(static_cast<size_t>(alloc_m) * bg_cnt);
        for (uint32_t off = 0; off < M; off += CHUNK) {
            uint32_t cnt = std::min(CHUNK, M - off);
            CU_CHECK(cuMemcpyHtoD_v2(d_in, &patches_u8[static_cast<size_t>(off) * kPatchPix],
                                     static_cast<size_t>(cnt) * kPatchPix));
            uint32_t pp = kPatchPix, pairs = cnt * bg_cnt;
            void* args[] = {&d_in, &d_bg, &d_sc, &pp, &bg_cnt, &cnt};
            CU_CHECK(cuEventRecord(ev_s, nullptr));
            CU_CHECK(cuLaunchKernel(fn_eval, (pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
            CU_CHECK(cuEventRecord(ev_e, nullptr));
            CU_CHECK(cuEventSynchronize(ev_e));
            float ms = 0.0f;
            CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
            gpu_ms += ms;
            CU_CHECK(cuMemcpyDtoH_v2(h_sc.data(), d_sc, static_cast<size_t>(cnt) * bg_cnt * sizeof(float)));
            for (uint32_t i = 0; i < cnt; ++i) {
                const float* row = &h_sc[static_cast<size_t>(i) * bg_cnt];
                float mb = -1e9f;
                for (uint32_t k = 0; k < bg_cnt; ++k) mb = std::max(mb, row[k]);
                out_bg[off + i] = mb;
            }
        }
        CU_CHECK(cuMemFree_v2(d_in));
        CU_CHECK(cuMemFree_v2(d_sc));
    };

    // Helper: score a batch of M patches against compartment `comp` templates on GPU -> best_cat[i], best_dog[i]
    auto gpu_score_comp = [&](const std::vector<uint8_t>& patches_u8, uint32_t M, uint32_t comp,
                              std::vector<float>& out_cat, std::vector<float>& out_dog, float& gpu_ms) {
        uint32_t nc = comp_ncat[comp], nd = comp_ndog[comp], T = nc + nd;
        out_cat.resize(M);
        out_dog.resize(M);
        constexpr uint32_t CHUNK = 65536u;
        CUdeviceptr d_in = 0, d_sc = 0;
        uint32_t alloc_m = std::min(M, CHUNK);
        CU_CHECK(cuMemAlloc_v2(&d_in, static_cast<size_t>(alloc_m) * kPatchPix));
        CU_CHECK(cuMemAlloc_v2(&d_sc, static_cast<size_t>(alloc_m) * T * sizeof(float)));
        std::vector<float> h_sc(static_cast<size_t>(alloc_m) * T);

        for (uint32_t off = 0; off < M; off += CHUNK) {
            uint32_t cnt = std::min(CHUNK, M - off);
            CU_CHECK(cuMemcpyHtoD_v2(d_in, &patches_u8[static_cast<size_t>(off) * kPatchPix],
                                     static_cast<size_t>(cnt) * kPatchPix));
            uint32_t pp = kPatchPix, pairs = cnt * T;
            void* args[] = {&d_in, &d_tmpls[comp], &d_sc, &pp, &T, &cnt};
            CU_CHECK(cuEventRecord(ev_s, nullptr));
            CU_CHECK(cuLaunchKernel(fn_eval, (pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
            CU_CHECK(cuEventRecord(ev_e, nullptr));
            CU_CHECK(cuEventSynchronize(ev_e));
            float ms = 0.0f;
            CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
            gpu_ms += ms;
            CU_CHECK(cuMemcpyDtoH_v2(h_sc.data(), d_sc, static_cast<size_t>(cnt) * T * sizeof(float)));
            for (uint32_t i = 0; i < cnt; ++i) {
                const float* row = &h_sc[static_cast<size_t>(i) * T];
                float bc = -1e9f, bd = -1e9f;
                for (uint32_t k = 0; k < nc; ++k) bc = std::max(bc, row[k]);
                for (uint32_t k = 0; k < nd; ++k) bd = std::max(bd, row[nc + k]);
                out_cat[off + i] = bc;
                out_dog[off + i] = bd;
            }
        }
        CU_CHECK(cuMemFree_v2(d_in));
        CU_CHECK(cuMemFree_v2(d_sc));
    };

    auto print_row = [&](const char* name, const std::vector<float>& margins) {
        uint32_t cc = 0, cd = 0, dc = 0, dd = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            int tr = test_rows[i].species;
            int pr = (margins[i] >= 0.0f) ? 0 : 1;
            if (tr == 0 && pr == 0) cc++;
            else if (tr == 0 && pr == 1) cd++;
            else if (tr == 1 && pr == 0) dc++;
            else dd++;
        }
        double acc = 100.0 * (cc + dd) / std::max(1u, N_test);
        double rc  = 100.0 * cc / std::max(1u, cc + cd);
        double rd  = 100.0 * dd / std::max(1u, dc + dd);
        std::printf("  %-40s | Acc: %5.2f%% | Bal: %5.2f%% | Cat: %5.2f%% (%3u/%3u) | Dog: %5.2f%% (%3u/%3u)\n",
                    name, acc, 0.5 * (rc + rd), rc, cc, cc + cd, rd, dd, dc + dd);
    };

    // =========================================================================
    // [MODE 1] Pure Blind Global Multi-Scale Scan (0 BBoxes, 0 Layout Priors!)
    // =========================================================================
    float mode1_gpu_ms = 0.0f;
    std::vector<uint32_t> m1_off(N_test + 1, 0);
    std::vector<std::vector<BBox>> m1_boxes(N_test);
    for (uint32_t i = 0; i < N_test; ++i) {
        m1_boxes[i] = make_global_feature_windows(imgs[i].w, imgs[i].h);
        m1_off[i + 1] = m1_off[i] + static_cast<uint32_t>(m1_boxes[i].size());
    }
    uint32_t total_m1_wins = m1_off[N_test];
    std::vector<uint8_t> m1_patches(static_cast<size_t>(total_m1_wins) * kPatchPix);
    {
        std::atomic<uint32_t> nxt{0};
        std::vector<std::thread> ths;
        for (uint32_t t = 0; t < n_threads; ++t) {
            ths.emplace_back([&]() {
                while (true) {
                    uint32_t i = nxt.fetch_add(1);
                    if (i >= N_test) break;
                    uint32_t base = m1_off[i];
                    for (size_t w = 0; w < m1_boxes[i].size(); ++w) {
                        const auto& b = m1_boxes[i][w];
                        extract_shadow_fast(imgs[i], b.x0, b.y0, b.x1, b.y1, use_hog,
                                            &m1_patches[static_cast<size_t>(base + w) * kPatchPix]);
                    }
                }
            });
        }
        for (auto& th : ths) th.join();
    }

    std::array<std::vector<float>, 6> m1_feat_margin;
    for (int f = 0; f < 6; ++f) {
        std::vector<float> sc_cat, sc_dog;
        gpu_score_comp(m1_patches, total_m1_wins, 2u + f, sc_cat, sc_dog, mode1_gpu_ms);
        m1_feat_margin[f].resize(N_test);
        for (uint32_t i = 0; i < N_test; ++i) {
            float bc = -1e9f, bd = -1e9f;
            for (uint32_t w = m1_off[i]; w < m1_off[i + 1]; ++w) {
                bc = std::max(bc, sc_cat[w]);
                bd = std::max(bd, sc_dog[w]);
            }
            m1_feat_margin[f][i] = bc - bd;
        }
    }
    std::vector<uint8_t>().swap(m1_patches);

    // =========================================================================
    // [MODE 2] Hierarchical Bottom-Up Constellation Scan (ZERO Test BBoxes!)
    //   Stage 1: Multi-scale GPU face proposal scan (L2 face manifold)
    //   Stage 2: Local 3x3 x 2-scale GPU search for all 6 L3 child manifolds
    //   Stage 3: Bottom-up child consensus locks onto face + species vote!
    // =========================================================================
    float mode2_gpu_ms = 0.0f;
    auto t_m2_start = std::chrono::steady_clock::now();

    // Stage 1: Score all multi-scale face proposals across all 1,283 raw images on GPU!
    std::vector<uint32_t> f_off(N_test + 1, 0);
    std::vector<std::vector<BBox>> f_props(N_test);
    for (uint32_t i = 0; i < N_test; ++i) {
        f_props[i] = make_face_proposals(imgs[i].w, imgs[i].h);
        f_off[i + 1] = f_off[i] + static_cast<uint32_t>(f_props[i].size());
    }
    uint32_t total_f_props = f_off[N_test];
    std::vector<uint8_t> f_patches(static_cast<size_t>(total_f_props) * kPatchPix);
    {
        std::atomic<uint32_t> nxt{0};
        std::vector<std::thread> ths;
        for (uint32_t t = 0; t < n_threads; ++t) {
            ths.emplace_back([&]() {
                while (true) {
                    uint32_t i = nxt.fetch_add(1);
                    if (i >= N_test) break;
                    uint32_t base = f_off[i];
                    for (size_t w = 0; w < f_props[i].size(); ++w) {
                        const auto& b = f_props[i][w];
                        extract_shadow_fast(imgs[i], b.x0, b.y0, b.x1, b.y1, use_hog,
                                            &f_patches[static_cast<size_t>(base + w) * kPatchPix]);
                    }
                }
            });
        }
        for (auto& th : ths) th.join();
    }

    // Stage 1 Localization: Use pure canonical universal face archetype (cancels background noise!)
    // while ALSO scoring fine-grained K=16 cat/dog face templates for species classification!
    std::vector<float> fp_cat, fp_dog, fp_canon, fp_univ, fp_bg;
    gpu_score_comp(f_patches, total_f_props, 1u, fp_cat, fp_dog, mode2_gpu_ms);
    gpu_score_ptr(f_patches, total_f_props, d_canon[1], 2u, fp_canon, mode2_gpu_ms);
    gpu_score_ptr(f_patches, total_f_props, d_univ[1],  1u, fp_univ,  mode2_gpu_ms);
    gpu_score_bg(f_patches, total_f_props, fp_bg, mode2_gpu_ms);
    std::vector<uint8_t>().swap(f_patches);

    // NMS to select Top-K diverse candidate face regions per raw image using universal face archetype prominence!
    struct FaceCand {
        BBox box;
        float cat_s, dog_s, canon_cat, canon_dog, best_s;
    };
    std::vector<std::vector<FaceCand>> top_cands(N_test);
    uint32_t total_top_cands = 0;
    std::vector<uint32_t> cand_off(N_test + 1, 0);
    for (uint32_t i = 0; i < N_test; ++i) {
        std::vector<FaceCand> all_c;
        all_c.reserve(f_props[i].size());
        for (size_t w = 0; w < f_props[i].size(); ++w) {
            uint32_t gi = f_off[i] + static_cast<uint32_t>(w);
            float cs = fp_cat[gi], ds = fp_dog[gi];
            float cc = fp_canon[gi * 2u + 0], cd = fp_canon[gi * 2u + 1];
            // Pure universal face archetype beats cluttered backgrounds far cleaner than max over 16 noisy clusters!
            float fg_score = std::max({fp_univ[gi], cc, cd}) - 0.55f * fp_bg[gi];
            all_c.push_back({f_props[i][w], cs, ds, cc, cd, fg_score});
        }
        std::sort(all_c.begin(), all_c.end(), [](const FaceCand& a, const FaceCand& b) {
            return a.best_s > b.best_s;
        });
        for (const auto& c : all_c) {
            bool dup = false;
            for (const auto& kept : top_cands[i]) {
                if (box_iou(c.box, kept.box) > 0.65f) { dup = true; break; }
            }
            if (!dup) {
                top_cands[i].push_back(c);
                if (static_cast<int>(top_cands[i].size()) >= top_faces_k) break;
            }
        }
        cand_off[i + 1] = cand_off[i] + static_cast<uint32_t>(top_cands[i].size());
    }
    total_top_cands = cand_off[N_test];

    // Stage 2: Inside EVERY Top-K candidate face, search a local 3x3 shift x 2 scale grid (18 windows)
    // for EACH of the 6 L3 child manifolds on GPU!
    constexpr uint32_t kLocalGrid = 18u; // 3x3 spatial shifts x 2 scales (0.85x, 1.15x)
    const float shift_d[3] = {-0.08f, 0.0f, +0.08f};
    const float scale_d[2] = {0.85f, 1.15f};

    std::array<std::vector<float>, 6> cand_feat_cat, cand_feat_dog, cand_feat_loc;
    std::array<std::vector<BBox>,  6> cand_feat_box;
    for (int f = 0; f < 6; ++f) {
        uint32_t M = total_top_cands * kLocalGrid;
        std::vector<uint8_t> fp(static_cast<size_t>(M) * kPatchPix);
        std::vector<BBox> boxes(M);
        const auto& pr = feat_prior[f];

        std::atomic<uint32_t> nxt{0};
        std::vector<std::thread> ths;
        for (uint32_t t = 0; t < n_threads; ++t) {
            ths.emplace_back([&]() {
                while (true) {
                    uint32_t i = nxt.fetch_add(1);
                    if (i >= N_test) break;
                    const auto& img = imgs[i];
                    for (uint32_t ci = 0; ci < top_cands[i].size(); ++ci) {
                        uint32_t gcand = cand_off[i] + ci;
                        const auto& fb = top_cands[i][ci].box;
                        float fw = fb.x1 - fb.x0;
                        float fh = fb.y1 - fb.y0;
                        uint32_t lid = 0;
                        for (int si = 0; si < 2; ++si) {
                            float bw = fw * pr.w * scale_d[si];
                            float bh = fh * pr.h * scale_d[si];
                            for (int dy = 0; dy < 3; ++dy) {
                                for (int dx = 0; dx < 3; ++dx) {
                                    float cx = fb.x0 + fw * (pr.cx + shift_d[dx]);
                                    float cy = fb.y0 + fh * (pr.cy + shift_d[dy]);
                                    BBox b{cx - 0.5f * bw, cy - 0.5f * bh, cx + 0.5f * bw, cy + 0.5f * bh};
                                    uint32_t slot = gcand * kLocalGrid + lid;
                                    boxes[slot] = b;
                                    extract_shadow_fast(img, b.x0, b.y0, b.x1, b.y1, use_hog,
                                                        &fp[static_cast<size_t>(slot) * kPatchPix]);
                                    lid++;
                                }
                            }
                        }
                    }
                }
            });
        }
        for (auto& th : ths) th.join();

        std::vector<float> sc_cat, sc_dog, sc_canon, sc_univ, sc_bg;
        gpu_score_comp(fp, M, 2u + f, sc_cat, sc_dog, mode2_gpu_ms);
        gpu_score_ptr(fp, M, d_canon[2u + f], 2u, sc_canon, mode2_gpu_ms);
        gpu_score_ptr(fp, M, d_univ[2u + f],  1u, sc_univ,  mode2_gpu_ms);
        gpu_score_bg(fp, M, sc_bg, mode2_gpu_ms);
        cand_feat_cat[f].resize(total_top_cands);
        cand_feat_dog[f].resize(total_top_cands);
        cand_feat_loc[f].resize(total_top_cands);
        cand_feat_box[f].resize(total_top_cands);
        for (uint32_t gcand = 0; gcand < total_top_cands; ++gcand) {
            uint32_t best_slot = gcand * kLocalGrid;
            float best_loc = -1e9f;
            for (uint32_t lid = 0; lid < kLocalGrid; ++lid) {
                uint32_t slot = gcand * kLocalGrid + lid;
                float cc = sc_canon[slot * 2u + 0], cd = sc_canon[slot * 2u + 1];
                float loc = std::max({sc_univ[slot], cc, cd}) - 0.35f * sc_bg[slot];
                if (loc > best_loc) { best_loc = loc; best_slot = slot; }
            }
            // At the cleanly localized child feature window, compare canonical + K=16 species shadows!
            cand_feat_loc[f][gcand] = best_loc;
            cand_feat_cat[f][gcand] = 0.5f * sc_canon[best_slot * 2u + 0] + 0.5f * sc_cat[best_slot];
            cand_feat_dog[f][gcand] = 0.5f * sc_canon[best_slot * 2u + 1] + 0.5f * sc_dog[best_slot];
            cand_feat_box[f][gcand] = boxes[best_slot];
        }
    }

    // Score L1 Full Image (no bbox needed — full raw image!)
    std::vector<uint8_t> full_patches(static_cast<size_t>(N_test) * kPatchPix);
    for (uint32_t i = 0; i < N_test; ++i) {
        extract_shadow_fast(imgs[i], 0.0f, 0.0f, static_cast<float>(imgs[i].w), static_cast<float>(imgs[i].h), use_hog,
                            &full_patches[static_cast<size_t>(i) * kPatchPix]);
    }
    std::vector<float> full_cat, full_dog;
    gpu_score_comp(full_patches, N_test, 0u, full_cat, full_dog, mode2_gpu_ms);

    // Stage 3: Bottom-Up Child-Manifold Consensus Lock!
    // Select winning candidate face per image maximizing (L2_face + mean of 6 L3 child manifold matches)
    std::array<std::vector<float>, 6> m2_feat_margin;
    for (int f = 0; f < 6; ++f) m2_feat_margin[f].resize(N_test);
    std::vector<float> m2_face_margin(N_test), m2_full_margin(N_test);
    std::vector<float> m2_vote_margin(N_test), m2_l3_5(N_test), m2_l3_6(N_test), m2_l32(N_test), m2_l321(N_test);
    std::vector<BBox>  det_face_box(N_test);
    std::vector<std::array<BBox, 6>> det_feat_boxes(N_test);
    std::vector<int>   det_cat_votes(N_test, 0);

    double sum_iou = 0.0;
    uint32_t iou_ge_30 = 0, iou_ge_50 = 0;
    uint32_t vote_hist_cat[7]{}, vote_hist_dog[7]{};

    for (uint32_t i = 0; i < N_test; ++i) {
        uint32_t best_ci = 0;
        float best_lock = -1e9f;
        for (uint32_t ci = 0; ci < top_cands[i].size(); ++ci) {
            uint32_t gcand = cand_off[i] + ci;
            float f_ev = top_cands[i][ci].best_s;
            float child_ev = 0.0f;
            for (int f = 0; f < 6; ++f) child_ev += cand_feat_loc[f][gcand];
            float total_ev = f_ev + child_ev / 6.0f;
            if (total_ev > best_lock) { best_lock = total_ev; best_ci = ci; }
        }
        uint32_t gbest = cand_off[i] + best_ci;
        det_face_box[i] = top_cands[i][best_ci].box;
        float iou = box_iou(det_face_box[i], test_rows[i].face_box);
        sum_iou += iou;
        if (iou >= 0.30f) iou_ge_30++;
        if (iou >= 0.50f) iou_ge_50++;

        int cv = 0;
        float sum5 = 0.0f, sum6 = 0.0f;
        for (int f = 0; f < 6; ++f) {
            float m = cand_feat_cat[f][gbest] - cand_feat_dog[f][gbest];
            m2_feat_margin[f][i] = m;
            det_feat_boxes[i][f] = cand_feat_box[f][gbest];
            if (m >= 0.0f) cv++;
            sum6 += m;
            if (f != 4) sum5 += m; // excl. nose
        }
        det_cat_votes[i] = cv;
        if (test_rows[i].species == 0) vote_hist_cat[cv]++;
        else vote_hist_dog[cv]++;

        m2_face_margin[i] = 0.5f * (top_cands[i][best_ci].canon_cat - top_cands[i][best_ci].canon_dog) +
                            0.5f * (top_cands[i][best_ci].cat_s - top_cands[i][best_ci].dog_s);
        m2_full_margin[i] = full_cat[i] - full_dog[i];
        // Tie-break 3-3 votes using the bottom-up L3 sum6 margin
        m2_vote_margin[i] = (cv > 3) ? 1.0f : (cv < 3 ? -1.0f : sum6);
        m2_l3_5[i]        = sum5;
        m2_l3_6[i]        = sum6;
        m2_l32[i]         = sum6 + m2_face_margin[i];
        m2_l321[i]        = sum6 + m2_face_margin[i] + m2_full_margin[i];
    }
    double m2_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_m2_start).count();

    // =========================================================================
    // Print Full Pure-Image Zero-BBox Benchmark Report
    // =========================================================================
    std::cout << "\n==============================================================================================================\n";
    std::printf("  PURE-IMAGE ZERO-BBOX CAT vs. DOG BENCHMARK (%u strictly unseen RAW uncropped images | 0 test bboxes!)\n", N_test);
    std::cout << "==============================================================================================================\n";

    std::cout << "\n[MODE 1] Unconstrained Global Multi-Scale Feature Scan (0 bboxes, 0 spatial priors, "
              << total_m1_wins << " GPU windows):\n";
    std::vector<float> m1_sum5(N_test, 0.0f), m1_sum6(N_test, 0.0f), m1_votes(N_test, 0.0f);
    for (int f = 0; f < 6; ++f) {
        std::string nm = "Global Scan L3: " + kCompNames[2 + f];
        print_row(nm.c_str(), m1_feat_margin[f]);
        for (uint32_t i = 0; i < N_test; ++i) {
            m1_sum6[i] += m1_feat_margin[f][i];
            if (f != 4) m1_sum5[i] += m1_feat_margin[f][i];
        }
    }
    for (uint32_t i = 0; i < N_test; ++i) {
        int cv = 0;
        for (int f = 0; f < 6; ++f) if (m1_feat_margin[f][i] >= 0.0f) cv++;
        m1_votes[i] = (cv > 3) ? 1.0f : (cv < 3 ? -1.0f : m1_sum6[i]);
    }
    print_row("Mode 1: 6-Feature Majority Vote",      m1_votes);
    print_row("Mode 1: L3 5 Features (excl. nose)",   m1_sum5);
    print_row("Mode 1: L3 All 6 Features Fused",      m1_sum6);
    std::printf("  [MODE 1 GPU] %u sliding windows scored in %.2f ms (%.0f raw imgs/sec)\n",
                total_m1_wins, mode1_gpu_ms, N_test * 1000.0 / std::max(0.01f, mode1_gpu_ms));

    std::cout << "\n[MODE 2] Hierarchical Bottom-Up Constellation Scan (ZERO test bboxes! Face->6 Child Manifolds Lock):\n";
    std::printf("  [LOCALIZATION] Zero-BBox Face Lock -> Mean IoU: %.3f | IoU>=0.30: %.1f%% (%u/%u) | IoU>=0.50: %.1f%% (%u/%u)\n",
                sum_iou / N_test,
                100.0 * iou_ge_30 / N_test, iou_ge_30, N_test,
                100.0 * iou_ge_50 / N_test, iou_ge_50, N_test);
    for (int f = 0; f < 6; ++f) {
        std::string nm = "Detected L3 Feature: " + kCompNames[2 + f];
        print_row(nm.c_str(), m2_feat_margin[f]);
    }
    print_row("Detected L3: 6-Feature Vote Count",    m2_vote_margin);
    print_row("Detected L3: 5 Features (excl. nose)", m2_l3_5);
    print_row("Detected L3: All 6 Features Fused",    m2_l3_6);
    print_row("Detected L2: Locked Face Only",        m2_face_margin);
    print_row("Detected L3 -> L2 (Features + Face)",  m2_l32);
    print_row("Raw L1: Full Image Only",              m2_full_margin);
    print_row("BOTTOM-UP FULL (L3 -> L2 -> L1)",      m2_l321);

    std::cout << "\n[CHILD-MANIFOLD VOTE DISTRIBUTION] How many of the 6 detected child features mapped to CAT (0..6):\n";
    std::printf("  True CAT (%3u imgs): ", 416u);
    for (int k = 0; k <= 6; ++k) std::printf("%d/6:%3u  ", k, vote_hist_cat[k]);
    std::printf("\n  True DOG (%3u imgs): ", 867u);
    for (int k = 0; k <= 6; ++k) std::printf("%d/6:%3u  ", k, vote_hist_dog[k]);
    std::printf("\n");

    std::printf("\n[SPEED]  Mode 2 GPU Kernel Time: %.2f ms (%.0f raw imgs/sec) | End-to-End Wall (16-core CPU + GPU): %.1f ms (%.0f raw imgs/sec)\n",
                mode2_gpu_ms, N_test * 1000.0 / std::max(0.01f, mode2_gpu_ms),
                m2_wall_ms,   N_test * 1000.0 / std::max(0.1, m2_wall_ms));

    // Diagnostic: Accuracy specifically when zero-bbox lock found the face (IoU >= 0.30) vs missed!
    {
        uint32_t loc_n = 0, loc_ok_6 = 0, loc_ok_bal_c = 0, loc_c = 0, loc_ok_bal_d = 0, loc_d = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            if (box_iou(det_face_box[i], test_rows[i].face_box) >= 0.30f) {
                loc_n++;
                // Check balanced midpoint (zero-param raw vs median-centered diagnostic)
                int pr = (m2_l3_6[i] >= 0.0f) ? 0 : 1;
                if (pr == test_rows[i].species) loc_ok_6++;
                if (test_rows[i].species == 0) { loc_c++; if (pr == 0) loc_ok_bal_c++; }
                else { loc_d++; if (pr == 1) loc_ok_bal_d++; }
            }
        }
        std::printf("[DIAG]   On localized subset (IoU>=0.30: %u/%u imgs): Raw L3-6 Acc = %.1f%% (Cat %u/%u, Dog %u/%u)\n",
                    loc_n, N_test, loc_n ? 100.0 * loc_ok_6 / loc_n : 0.0,
                    loc_ok_bal_c, loc_c, loc_ok_bal_d, loc_d);
    }

    // ROC-AUC diagnostic (threshold-independent separability of raw zero-bbox shadow margins — zero calibration bias!)
    auto compute_auc = [&](const std::vector<float>& margins) -> double {
        std::vector<std::pair<float, int>> v(N_test);
        uint64_t n_cat = 0, n_dog = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            v[i] = {margins[i], test_rows[i].species};
            if (test_rows[i].species == 0) n_cat++; else n_dog++;
        }
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        uint64_t dogs_below = 0, wins = 0;
        for (const auto& p : v) {
            if (p.second == 1) dogs_below++;
            else wins += dogs_below;
        }
        return (n_cat * n_dog > 0) ? (static_cast<double>(wins) / static_cast<double>(n_cat * n_dog)) : 0.5;
    };
    std::printf("[AUC]    Threshold-Free Cat-vs-Dog Separability (ROC-AUC, 0.50=chance, 1.00=perfect):\n");
    std::printf("         Mode 1 Global L3-6 AUC : %.4f | Mode 2 Locked L3-6 AUC : %.4f | L2 Face AUC : %.4f | L1 Full AUC : %.4f | Full L3->L2->L1 AUC : %.4f\n",
                compute_auc(m1_sum6), compute_auc(m2_l3_6), compute_auc(m2_face_margin), compute_auc(m2_full_margin), compute_auc(m2_l321));

    // Optional 4x4 visual verification mosaic of zero-bbox detections!
    if (!viz_out_path.empty() && !viz_indices.empty()) {
        constexpr int kCell = 240, kCols = 4, kRows = 4;
        const int CW = kCell * kCols, CH = kCell * kRows;
        std::vector<uint8_t> canvas(static_cast<size_t>(CW) * CH * 3u, 18u);
        const uint8_t fcol[6][3] = {
            {60, 220, 255}, {255, 170, 50}, {140, 110, 255},
            {90, 255, 120}, {255, 90, 140}, {80, 190, 255}
        };
        for (size_t idx = 0; idx < std::min<size_t>(16, viz_indices.size()); ++idx) {
            uint32_t i = viz_indices[idx];
            const auto& img = imgs[i];
            if (img.rgb.empty()) continue;
            int gx = static_cast<int>(idx % kCols) * kCell;
            int gy = static_cast<int>(idx / kCols) * kCell;
            float sx = static_cast<float>(kCell - 8) / static_cast<float>(img.w);
            float sy = static_cast<float>(kCell - 8) / static_cast<float>(img.h);
            float sc = std::min(sx, sy);
            int dw = std::max(1, static_cast<int>(std::lround(img.w * sc)));
            int dh = std::max(1, static_cast<int>(std::lround(img.h * sc)));
            int ox = gx + (kCell - dw) / 2;
            int oy = gy + (kCell - dh) / 2;
            for (int y = 0; y < dh; ++y) {
                int py = std::clamp(static_cast<int>(y / sc), 0, static_cast<int>(img.h) - 1);
                for (int x = 0; x < dw; ++x) {
                    int px = std::clamp(static_cast<int>(x / sc), 0, static_cast<int>(img.w) - 1);
                    size_t si = (static_cast<size_t>(py) * img.w + px) * 3u;
                    size_t di = (static_cast<size_t>(oy + y) * CW + (ox + x)) * 3u;
                    canvas[di] = img.rgb[si]; canvas[di + 1] = img.rgb[si + 1]; canvas[di + 2] = img.rgb[si + 2];
                }
            }
            // Border color: bright green if predicted species == truth, red if wrong!
            int pred = (m2_l321[i] >= 0.0f) ? 0 : 1;
            bool ok  = (pred == test_rows[i].species);
            draw_rect(canvas, CW, CH, gx + 1, gy + 1, gx + kCell - 2, gy + kCell - 2,
                      ok ? 40 : 240, ok ? 225 : 60, ok ? 95 : 60, 3);
            // Draw zero-bbox detected face box (bright yellow) & 6 detected child features (distinct colors)
            const auto& fb = det_face_box[i];
            draw_rect(canvas, CW, CH,
                      ox + static_cast<int>(fb.x0 * sc), oy + static_cast<int>(fb.y0 * sc),
                      ox + static_cast<int>(fb.x1 * sc), oy + static_cast<int>(fb.y1 * sc),
                      255, 235, 50, 2);
            for (int f = 0; f < 6; ++f) {
                const auto& b = det_feat_boxes[i][f];
                draw_rect(canvas, CW, CH,
                          ox + static_cast<int>(b.x0 * sc), oy + static_cast<int>(b.y0 * sc),
                          ox + static_cast<int>(b.x1 * sc), oy + static_cast<int>(b.y1 * sc),
                          fcol[f][0], fcol[f][1], fcol[f][2], 2);
            }
        }
        std::ofstream pf(viz_out_path, std::ios::binary);
        pf << "P6\n" << CW << " " << CH << "\n255\n";
        pf.write(reinterpret_cast<const char*>(canvas.data()), static_cast<std::streamsize>(canvas.size()));
        std::cout << "[VIZ]    Saved 4x4 zero-bbox detection verification sheet to " << viz_out_path << "\n";
    }

    for (uint32_t comp = 0; comp < kNumComps; ++comp) {
        CU_CHECK(cuMemFree_v2(d_tmpls[comp]));
        CU_CHECK(cuMemFree_v2(d_canon[comp]));
        CU_CHECK(cuMemFree_v2(d_univ[comp]));
    }
    CU_CHECK(cuMemFree_v2(d_bg));
    CU_CHECK(cuEventDestroy_v2(ev_s));
    CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
