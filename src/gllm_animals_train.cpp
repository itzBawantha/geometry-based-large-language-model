// =============================================================================
// gllm_animals_train — Full CatFLW + DogFLW Hierarchical Multi-Shadow Trainer
//                      & Bottom-Up Unseen Cat vs. Dog GPU Evaluator
//
// Zero neural weights. Zero backpropagation. 100% pure geometric shadow overlap
// + GPU Spherical K-Means++ clustering inside the 1-billion-point 3D sphere!
//
// Builds the exact 4-level, 17-manifold, 10-concept hierarchical 3D bounding-box
// animals model:
//   Level 0 (Root):     #0  animals            <-> Outer #0 ('animals')
//   Level 1 (Species):  #1  cats               <-> Outer #1 ('cat')   [K full-body shadows]
//                       #2  dogs               <-> Outer #2 ('dog')   [K full-body shadows]
//   Level 2 (Face):     #3  cats/face          <-> Outer #3 ('face' - SHARED!) [K face shadows]
//                       #4  dogs/face          <-> Outer #3 ('face' - SHARED!) [K face shadows]
//   Level 3 (Features): #5..#10  cats/face/{ear_right,ear_left,eye_right,eye_left,nose,mouth}
//                       #11..#16 dogs/face/{ear_right,ear_left,eye_right,eye_left,nose,mouth}
//                       <-> SHARED Outer #4..#9 ('ear_right'..'mouth') [K shadows each]
//
// Uses exact dataset face & polygon bounding boxes, decodes all 6,412 PNGs via
// multi-threaded libpng, projects each crop to a lighting- & fur-invariant
// aspect-preserving 28x28 geometric shadow + true-RGB cluster prototype, runs
// GPU Spherical K-Means++ on CUDA, saves .gllm v5, and audits bottom-up
// (Features -> Face -> Full Body) unseen Cat vs. Dog classification on GPU.
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"
#include "../include/grad_domain.hpp"
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

static constexpr uint32_t kPatchDim  = 28u;
static constexpr uint32_t kPatchPix  = kPatchDim * kPatchDim; // 784
static constexpr uint32_t kNumComps  = 8u; // 0:full, 1:face, 2..7: 6 facial features

static const std::array<std::string, kNumComps> kCompNames = {
    "full", "face", "ear_right", "ear_left", "eye_right", "eye_left", "nose", "mouth"
};

enum class ShadowMode { kContour = 0, kSobel = 1, kLuma = 2, kHog = 3, kGrad = 4 };

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

struct RgbBuf {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgb; // w * h * 3
};

struct SamplePatches {
    std::array<std::array<uint8_t, kPatchPix>, kNumComps> shadow{};
    std::array<std::array<uint8_t, kPatchPix * 3u>, kNumComps> rgb{};
    std::array<std::array<uint8_t, kPatchPix>, kNumComps> mask{};
    bool ok = false;
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
        r.is_test = (cols[0] == "test");
        r.species = (cols[1] == "dog") ? 1 : 0;
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

static bool load_png_rgb(const std::string& path, RgbBuf& out) {
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

    out.w = w;
    out.h = h;
    out.rgb.resize(static_cast<size_t>(w) * h * 3u);
    std::vector<png_bytep> rows(h);
    for (png_uint_32 y = 0; y < h; ++y) {
        rows[y] = out.rgb.data() + static_cast<size_t>(y) * w * 3u;
    }
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return (w > 0 && h > 0);
}

// Crop exact bbox from RGB image, fit into 28x28 (aspect-preserving or square stretch),
// and compute lighting/fur-invariant geometric contour shadow + true RGB!
static void extract_patch_28x28(const RgbBuf& img,
                                int x0, int y0, int x1, int y1,
                                ShadowMode mode,
                                bool preserve_aspect,
                                std::array<uint8_t, kPatchPix>& out_shadow,
                                std::array<uint8_t, kPatchPix * 3u>& out_rgb,
                                std::array<uint8_t, kPatchPix>& out_mask) {
    out_shadow.fill(0);
    out_rgb.fill(0);
    out_mask.fill(0);

    int W = static_cast<int>(img.w);
    int H = static_cast<int>(img.h);
    x0 = std::clamp(x0, 0, W - 1);
    y0 = std::clamp(y0, 0, H - 1);
    x1 = std::clamp(x1, x0 + 1, W);
    y1 = std::clamp(y1, y0 + 1, H);
    int cw = x1 - x0;
    int ch = y1 - y0;

    // Fit into inner 26x26 canvas (aspect-preserving preserves shape proportion; square tests pure internal texture/edges!)
    constexpr int kInner = 26;
    int fw = kInner, fh = kInner;
    if (preserve_aspect) {
        if (cw >= ch) {
            fw = kInner;
            fh = std::max(2, static_cast<int>(std::lround(static_cast<double>(ch) * kInner / cw)));
        } else {
            fh = kInner;
            fw = std::max(2, static_cast<int>(std::lround(static_cast<double>(cw) * kInner / ch)));
        }
    }
    int ox = (static_cast<int>(kPatchDim) - fw) / 2;
    int oy = (static_cast<int>(kPatchDim) - fh) / 2;

    std::array<float, kPatchPix> lum{};
    double sum_l = 0.0, sum_l2 = 0.0;
    int cnt = 0;

    for (int dy = 0; dy < fh; ++dy) {
        double sy0 = y0 + (static_cast<double>(dy) * ch) / fh;
        double sy1 = y0 + (static_cast<double>(dy + 1) * ch) / fh;
        int iy0 = std::clamp(static_cast<int>(std::floor(sy0)), y0, y1 - 1);
        int iy1 = std::clamp(static_cast<int>(std::ceil(sy1)), iy0 + 1, y1);
        for (int dx = 0; dx < fw; ++dx) {
            double sx0 = x0 + (static_cast<double>(dx) * cw) / fw;
            double sx1 = x0 + (static_cast<double>(dx + 1) * cw) / fw;
            int ix0 = std::clamp(static_cast<int>(std::floor(sx0)), x0, x1 - 1);
            int ix1 = std::clamp(static_cast<int>(std::ceil(sx1)), ix0 + 1, x1);

            double sr = 0.0, sg = 0.0, sb = 0.0;
            int n = 0;
            for (int sy = iy0; sy < iy1; ++sy) {
                const uint8_t* row = img.rgb.data() + static_cast<size_t>(sy) * W * 3u;
                for (int sx = ix0; sx < ix1; ++sx) {
                    sr += row[sx * 3 + 0];
                    sg += row[sx * 3 + 1];
                    sb += row[sx * 3 + 2];
                    n++;
                }
            }
            float inv = n > 0 ? (1.0f / static_cast<float>(n)) : 1.0f;
            float r = static_cast<float>(sr) * inv;
            float g = static_cast<float>(sg) * inv;
            float b = static_cast<float>(sb) * inv;

            int px = ox + dx;
            int py = oy + dy;
            size_t idx = static_cast<size_t>(py) * kPatchDim + px;
            out_rgb[idx * 3 + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(r)), 0, 255));
            out_rgb[idx * 3 + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(g)), 0, 255));
            out_rgb[idx * 3 + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(b)), 0, 255));
            out_mask[idx] = 1u;

            float l = (0.299f * r + 0.587f * g + 0.114f * b) * (1.0f / 255.0f);
            lum[idx] = l;
            sum_l  += l;
            sum_l2 += static_cast<double>(l) * l;
            cnt++;
        }
    }

    if (mode == ShadowMode::kHog) {
        gllm::extract_hog_from_rgb(img.rgb.data(), img.w, img.h, x0, y0, x1, y1, out_shadow.data());
        return;
    }

    if (cnt <= 0) return;
    float mean_l = static_cast<float>(sum_l / cnt);
    float var_l  = static_cast<float>(std::max(0.0, sum_l2 / cnt - static_cast<double>(mean_l) * mean_l));
    float inv_sd = 1.0f / std::max(0.05f, std::sqrt(var_l));

    // Normalized interior luminance z-field (fur-color & exposure invariant!)
    std::array<float, kPatchPix> zf{};
    for (int dy = 0; dy < fh; ++dy) {
        for (int dx = 0; dx < fw; ++dx) {
            size_t idx = static_cast<size_t>(oy + dy) * kPatchDim + (ox + dx);
            zf[idx] = std::clamp((lum[idx] - mean_l) * inv_sd, -3.0f, 3.0f);
        }
    }

    auto sample_z = [&](int x, int y) -> float {
        x = std::clamp(x, ox, ox + fw - 1);
        y = std::clamp(y, oy, oy + fh - 1);
        return zf[static_cast<size_t>(y) * kPatchDim + x];
    };

    for (int dy = 0; dy < fh; ++dy) {
        int py = oy + dy;
        for (int dx = 0; dx < fw; ++dx) {
            int px = ox + dx;
            size_t idx = static_cast<size_t>(py) * kPatchDim + px;
            float val = 0.0f;
            if (mode == ShadowMode::kLuma) {
                val = std::clamp(lum[idx], 0.02f, 1.0f);
            } else {
                float tl = sample_z(px - 1, py - 1), tc = sample_z(px, py - 1), tr = sample_z(px + 1, py - 1);
                float ml = sample_z(px - 1, py    ),                             mr = sample_z(px + 1, py    );
                float bl = sample_z(px - 1, py + 1), bc = sample_z(px, py + 1), br = sample_z(px + 1, py + 1);
                float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
                float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
                float gmag = std::sqrt(gx * gx + gy * gy) * 0.25f;
                float edge = std::tanh(gmag * 0.85f);
                if (mode == ShadowMode::kSobel) {
                    val = std::clamp(0.10f + 0.90f * edge, 0.0f, 1.0f);
                } else {
                    // Geometric contour + dark anatomical cavity (pupils, nostrils, mouth line, inner ear)
                    // + subtle silhouette presence preserving exact bounding-box aspect ratio
                    float dark = std::clamp(-zf[idx] * 0.42f, 0.0f, 1.0f);
                    val = std::clamp(0.16f + 0.52f * edge + 0.32f * dark, 0.0f, 1.0f);
                }
            }
            out_shadow[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(val * 255.0f)), 1, 255));
        }
    }
}

static bool normalize_u8(const uint8_t* px, uint32_t n, float* out) {
    double s2 = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double v = static_cast<double>(px[i]) * (1.0 / 255.0);
        s2 += v * v;
    }
    if (s2 < 1e-8) { std::fill(out, out + n, 0.0f); return false; }
    float inv = static_cast<float>(1.0 / std::sqrt(s2));
    for (uint32_t i = 0; i < n; ++i) out[i] = static_cast<float>(px[i]) * (1.0f / 255.0f) * inv;
    return true;
}

// Place all 10 animals concept nodes cleanly in 3 structured arcs on upper-front crown of R = 1.0
static void place_concept_crown_point(uint32_t idx, float& ox, float& oy, float& oz) {
    float tx = 0.0f, ty = 0.88f;
    switch (idx) {
        case 0u: tx =  0.00f; ty = 0.93f; break; // animals
        case 1u: tx = -0.36f; ty = 0.82f; break; // cat
        case 2u: tx =  0.36f; ty = 0.82f; break; // dog
        case 3u: tx =  0.00f; ty = 0.78f; break; // face (shared)
        case 4u: tx = -0.50f; ty = 0.64f; break; // ear_right (shared)
        case 5u: tx = -0.30f; ty = 0.62f; break; // ear_left  (shared)
        case 6u: tx = -0.10f; ty = 0.60f; break; // eye_right (shared)
        case 7u: tx =  0.10f; ty = 0.60f; break; // eye_left  (shared)
        case 8u: tx =  0.30f; ty = 0.62f; break; // nose      (shared)
        case 9u: tx =  0.50f; ty = 0.64f; break; // mouth     (shared)
        default: tx = 0.0f; ty = 0.55f; break;
    }
    float xy2 = tx * tx + ty * ty;
    if (xy2 > 0.96f) { float s = std::sqrt(0.96f / xy2); tx *= s; ty *= s; xy2 = tx * tx + ty * ty; }
    ox = tx; oy = ty; oz = std::sqrt(std::max(0.0f, 1.0f - xy2));
}

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_animals_train --manifest <data/cat_dog_split.tsv> --model <models/animals_trained.gllm>\n"
              << "                     [--clusters 16] [--iters 15] [--shadow-mode contour|sobel|luma] [--grid 1241]\n";
}

int main(int argc, char** argv) {
    std::string manifest_path = "data/cat_dog_split.tsv";
    std::string model_path    = "models/animals_trained.gllm";
    uint32_t num_clusters     = 16u;
    uint32_t cluster_iters    = 15u;
    uint32_t grid_dim         = 1241u;
    ShadowMode shadow_mode    = ShadowMode::kContour;
    std::string mode_str      = "contour";
    bool preserve_aspect      = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--clusters" && i + 1 < argc) num_clusters = std::clamp<uint32_t>(std::atoi(argv[++i]), 1u, 16u);
        else if (a == "--iters" && i + 1 < argc) cluster_iters = std::max<uint32_t>(1u, std::atoi(argv[++i]));
        else if (a == "--grid" && i + 1 < argc) grid_dim = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--no-aspect") preserve_aspect = false;
        else if (a == "--shadow-mode" && i + 1 < argc) {
            mode_str = argv[++i];
            if (mode_str == "sobel") shadow_mode = ShadowMode::kSobel;
            else if (mode_str == "luma") shadow_mode = ShadowMode::kLuma;
            else if (mode_str == "hog") shadow_mode = ShadowMode::kHog;
            else if (mode_str == "grad") shadow_mode = ShadowMode::kGrad;
            else shadow_mode = ShadowMode::kContour;
        } else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<ManifestRow> rows;
    if (!load_manifest(manifest_path, rows)) {
        std::cerr << "[ERROR] Failed to load manifest: " << manifest_path << "\n";
        return 1;
    }

    uint32_t n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "[LOAD]   Decoding " << rows.size() << " CatFLW+DogFLW PNGs across "
              << n_threads << " CPU threads (exact dataset bboxes, shadow-mode=" << mode_str << ")...\n";

    std::vector<SamplePatches> patches(rows.size());
    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> decode_fail{0};
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (uint32_t t = 0; t < n_threads; ++t) {
        workers.emplace_back([&]() {
            RgbBuf img;
            while (true) {
                size_t idx = next_idx.fetch_add(1);
                if (idx >= rows.size()) break;
                const auto& r = rows[idx];
                if (!load_png_rgb(r.image_path, img)) { decode_fail.fetch_add(1); continue; }
                auto& sp = patches[idx];
                gllm::GradEdgeImage ge;
                if (shadow_mode == ShadowMode::kGrad) {
                    gllm::build_grad_edge_image(img.rgb.data(), img.w, img.h, ge);
                }
                auto do_patch = [&](int comp_idx, int x0, int y0, int x1, int y1) {
                    extract_patch_28x28(img, x0, y0, x1, y1,
                                        (shadow_mode == ShadowMode::kGrad) ? ShadowMode::kContour : shadow_mode,
                                        preserve_aspect, sp.shadow[comp_idx], sp.rgb[comp_idx], sp.mask[comp_idx]);
                    if (shadow_mode == ShadowMode::kGrad) {
                        gllm::extract_grad_edge_patch_28x28(ge, static_cast<float>(x0), static_cast<float>(y0),
                                                            static_cast<float>(x1), static_cast<float>(y1),
                                                            false, 0.0f, sp.shadow[comp_idx].data());
                    }
                };
                // 0: full body
                do_patch(0, 0, 0, static_cast<int>(img.w), static_cast<int>(img.h));
                // 1: exact dataset face bbox
                do_patch(1,
                         static_cast<int>(std::floor(r.face_box.x0)),
                         static_cast<int>(std::floor(r.face_box.y0)),
                         static_cast<int>(std::ceil(r.face_box.x1)),
                         static_cast<int>(std::ceil(r.face_box.y1)));
                // 2..7: exact dataset feature polygon bboxes
                for (int f = 0; f < 6; ++f) {
                    const auto& fb = r.feat_boxes[f];
                    do_patch(2 + f,
                             static_cast<int>(std::floor(fb.x0)),
                             static_cast<int>(std::floor(fb.y0)),
                             static_cast<int>(std::ceil(fb.x1)),
                             static_cast<int>(std::ceil(fb.y1)));
                }
                sp.ok = true;
            }
        });
    }
    for (auto& w : workers) w.join();

    auto t_dec = std::chrono::steady_clock::now();
    double dec_sec = std::chrono::duration<double>(t_dec - t0).count();

    std::vector<size_t> train_ids[2], test_ids;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!patches[i].ok) continue;
        if (rows[i].is_test) test_ids.push_back(i);
        else train_ids[rows[i].species].push_back(i);
    }
    std::printf("[LOAD]   Decoded %zu valid images in %.2fs (failed=%zu) | Train: %zu cats, %zu dogs | Unseen Holdout: %zu\n",
                train_ids[0].size() + train_ids[1].size() + test_ids.size(),
                dec_sec, decode_fail.load(),
                train_ids[0].size(), train_ids[1].size(), test_ids.size());

    // -------------------------------------------------------------------------
    // 2. Initialize CUDA Driver & Load PTX Kernels
    // -------------------------------------------------------------------------
    CU_CHECK(cuInit(0));
    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    char gpu_name[256] = {};
    CU_CHECK(cuDeviceGetName(gpu_name, sizeof(gpu_name), dev));
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&ctx, 0, dev));

    CUmodule mod = gllm::load_gllm_ptx_module();
    CUfunction fn_train = nullptr, fn_eval = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_train, mod, "train_accumulate_shadows"));
    CU_CHECK(cuModuleGetFunction(&fn_eval,  mod, "eval_score_shadows"));
    CUevent ev_s = nullptr, ev_e = nullptr;
    CU_CHECK(cuEventCreate(&ev_s, 0));
    CU_CHECK(cuEventCreate(&ev_e, 0));
    std::cout << "[CUDA]   GPU: " << gpu_name << " | K=" << num_clusters
              << " clusters/compartment x 16 compartments = " << (16u * num_clusters) << " sub-shadow regions\n";

    // -------------------------------------------------------------------------
    // 3. Build Exact 4-Level 17-Manifold 3D Bounding-Box Hierarchy
    // -------------------------------------------------------------------------
    gllm::GeometricModel model;
    model.lattice_grid = grid_dim;
    int64_t mid    = static_cast<int64_t>(grid_dim / 2u);
    int64_t cat_cx = mid - 105;
    int64_t dog_cx = mid + 105;
    int64_t z_full = mid - 72; // 16 slices * dz=3 -> [548..593]
    int64_t z_face = mid - 8;  // 16 slices * dz=3 -> [612..657]
    int64_t z_feat = mid + 56; // 16 slices * dz=3 -> [676..721]
    int64_t dz_step = (num_clusters <= 8u) ? 4 : 3;

    uint32_t m_animals  = model.create_hierarchical_manifold("animals",   "animals", "root_animals_manifold",    gllm::kInvalidId, mid,    mid, mid,    1u);
    uint32_t m_cats     = model.create_hierarchical_manifold("cats",      "cat",     "species_cat_manifold",     m_animals,        cat_cx, mid, mid,    1u);
    uint32_t m_dogs     = model.create_hierarchical_manifold("dogs",      "dog",     "species_dog_manifold",     m_animals,        dog_cx, mid, mid,    1u);
    uint32_t m_cat_face = model.create_hierarchical_manifold("cats/face", "face",    "cat_facial_region_shadow", m_cats,           cat_cx, mid, z_face, 1u);
    uint32_t m_dog_face = model.create_hierarchical_manifold("dogs/face", "face",    "dog_facial_region_shadow", m_dogs,           dog_cx, mid, z_face, 1u);

    std::array<uint32_t, kNumComps> comp_manifold[2];
    comp_manifold[0][0] = m_cats;
    comp_manifold[0][1] = m_cat_face;
    comp_manifold[1][0] = m_dogs;
    comp_manifold[1][1] = m_dog_face;
    for (int f = 0; f < 6; ++f) {
        const auto& fn = kCompNames[2 + f];
        comp_manifold[0][2 + f] = model.create_hierarchical_manifold(
            "cats/face/" + fn, fn, "cat_feature_" + fn, m_cat_face, cat_cx, mid, z_feat, 1u);
    }
    for (int f = 0; f < 6; ++f) {
        const auto& fn = kCompNames[2 + f];
        comp_manifold[1][2 + f] = model.create_hierarchical_manifold(
            "dogs/face/" + fn, fn, "dog_feature_" + fn, m_dog_face, dog_cx, mid, z_feat, 1u);
    }

    for (auto& m : model.manifolds) {
        if (m.outer_idx != gllm::kInvalidId) place_concept_crown_point(m.outer_idx, m.outer_x, m.outer_y, m.outer_z);
        m.inference_flags = gllm::pack_inference_flags(true, gllm::InferenceModality::kRgbPyramid);
    }

    // Anatomical XY centers for the 6 facial features (strictly non-overlapping 28x28 boxes + 2-voxel pad!)
    auto comp_center = [&](int sp, uint32_t comp, int64_t& cx, int64_t& cy, int64_t& z0) {
        int64_t bx = (sp == 0) ? cat_cx : dog_cx;
        switch (comp) {
            case 0u: cx = bx;      cy = mid;      z0 = z_full; break; // full
            case 1u: cx = bx;      cy = mid;      z0 = z_face; break; // face
            case 2u: cx = bx - 24; cy = mid + 56; z0 = z_feat; break; // ear_right
            case 3u: cx = bx + 24; cy = mid + 56; z0 = z_feat; break; // ear_left
            case 4u: cx = bx - 24; cy = mid + 16; z0 = z_feat; break; // eye_right
            case 5u: cx = bx + 24; cy = mid + 16; z0 = z_feat; break; // eye_left
            case 6u: cx = bx;      cy = mid - 24; z0 = z_feat; break; // nose
            case 7u: cx = bx;      cy = mid - 64; z0 = z_feat; break; // mouth
        }
    };

    // -------------------------------------------------------------------------
    // 4. GPU Spherical K-Means++ Multi-Shadow Training across all 16 Compartments
    // -------------------------------------------------------------------------
    float total_train_gpu_ms = 0.0f;
    // Keep region index ranges [species][comp] for instant GPU evaluation right after saving!
    std::vector<uint32_t> comp_region_ids[2][kNumComps];

    std::cout << "\n[TRAIN]  Running GPU Spherical K-Means++ across all 16 animal compartments:\n";
    for (int sp = 0; sp < 2; ++sp) {
        const char* sp_name = (sp == 0) ? "cat" : "dog";
        const auto& ids = train_ids[sp];
        uint32_t M = static_cast<uint32_t>(ids.size());

        for (uint32_t comp = 0; comp < kNumComps; ++comp) {
            uint32_t K = std::min<uint32_t>(num_clusters, std::max<uint32_t>(1u, M / 8u));
            std::vector<uint8_t> cls_px(static_cast<size_t>(M) * kPatchPix);
            for (uint32_t i = 0; i < M; ++i) {
                std::memcpy(&cls_px[static_cast<size_t>(i) * kPatchPix],
                            patches[ids[i]].shadow[comp].data(), kPatchPix);
            }

            // Deterministic farthest-point spherical seeding (strictly over non-zero vectors!)
            std::vector<float> pool_vecs;
            pool_vecs.reserve(static_cast<size_t>(std::min<uint32_t>(M, 1024u)) * kPatchPix);
            std::vector<float> tmp_v(kPatchPix);
            uint32_t stride_n = std::min<uint32_t>(M, 1024u);
            for (uint32_t s = 0; s < stride_n; ++s) {
                uint32_t idx = static_cast<uint32_t>((static_cast<uint64_t>(s) * M) / stride_n);
                if (normalize_u8(&cls_px[static_cast<size_t>(idx) * kPatchPix], kPatchPix, tmp_v.data())) {
                    pool_vecs.insert(pool_vecs.end(), tmp_v.begin(), tmp_v.end());
                }
            }
            uint32_t pool_n = static_cast<uint32_t>(pool_vecs.size() / kPatchPix);
            if (pool_n < K) K = std::max(1u, pool_n);
            std::vector<float> h_cents(static_cast<size_t>(K) * kPatchPix, 0.0f);
            if (pool_n > 0) std::memcpy(&h_cents[0], &pool_vecs[0], kPatchPix * sizeof(float));
            std::vector<float> max_sim(pool_n, -1.0f);
            for (uint32_t c = 1; c < K; ++c) {
                const float* prev = &h_cents[static_cast<size_t>(c - 1) * kPatchPix];
                uint32_t far_s = 0;
                float min_s = 1e9f;
                for (uint32_t s = 0; s < pool_n; ++s) {
                    const float* v = &pool_vecs[static_cast<size_t>(s) * kPatchPix];
                    float dot = 0.0f;
                    for (uint32_t p = 0; p < kPatchPix; ++p) dot += v[p] * prev[p];
                    if (dot > max_sim[s]) max_sim[s] = dot;
                    if (max_sim[s] < min_s) { min_s = max_sim[s]; far_s = s; }
                }
                std::memcpy(&h_cents[static_cast<size_t>(c) * kPatchPix],
                            &pool_vecs[static_cast<size_t>(far_s) * kPatchPix], kPatchPix * sizeof(float));
            }

            // GPU Spherical K-Means iterations via eval_score_shadows + train_accumulate_shadows
            CUdeviceptr d_imgs = 0, d_cents = 0, d_scores = 0, d_assign = 0, d_sum = 0, d_cnt = 0;
            CU_CHECK(cuMemAlloc_v2(&d_imgs,   cls_px.size()));
            CU_CHECK(cuMemcpyHtoD_v2(d_imgs,  cls_px.data(), cls_px.size()));
            CU_CHECK(cuMemAlloc_v2(&d_cents,  h_cents.size() * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_scores, static_cast<size_t>(M) * K * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_assign, static_cast<size_t>(M) * sizeof(uint32_t)));
            CU_CHECK(cuMemAlloc_v2(&d_sum,    static_cast<size_t>(K) * kPatchPix * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_cnt,    static_cast<size_t>(K) * sizeof(uint32_t)));

            std::vector<float>    h_scores(static_cast<size_t>(M) * K);
            std::vector<uint32_t> h_assign(M, 0u);
            std::vector<float>    h_sum(static_cast<size_t>(K) * kPatchPix, 0.0f);
            std::vector<uint32_t> h_cnt(K, 0u);

            for (uint32_t it = 0; it < cluster_iters; ++it) {
                CU_CHECK(cuMemcpyHtoD_v2(d_cents, h_cents.data(), h_cents.size() * sizeof(float)));
                uint32_t pp = kPatchPix, pairs = M * K;
                void* e_args[] = {&d_imgs, &d_cents, &d_scores, &pp, &K, &M};
                CU_CHECK(cuEventRecord(ev_s, nullptr));
                CU_CHECK(cuLaunchKernel(fn_eval, (pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, e_args, nullptr));
                CU_CHECK(cuEventRecord(ev_e, nullptr));
                CU_CHECK(cuEventSynchronize(ev_e));
                float ms = 0.0f;
                CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
                total_train_gpu_ms += ms;

                CU_CHECK(cuMemcpyDtoH_v2(h_scores.data(), d_scores, h_scores.size() * sizeof(float)));
                uint32_t worst_i = 0;
                float worst_v = 1e9f;
                for (uint32_t i = 0; i < M; ++i) {
                    uint32_t bk = 0; float bv = -1e9f;
                    for (uint32_t k = 0; k < K; ++k) {
                        float s = h_scores[static_cast<size_t>(i) * K + k];
                        if (s > bv) { bv = s; bk = k; }
                    }
                    h_assign[i] = bk;
                    if (bv < worst_v) { worst_v = bv; worst_i = i; }
                }

                CU_CHECK(cuMemcpyHtoD_v2(d_assign, h_assign.data(), h_assign.size() * sizeof(uint32_t)));
                CU_CHECK(cuMemsetD32_v2(d_sum, 0u, static_cast<size_t>(K) * kPatchPix));
                CU_CHECK(cuMemsetD32_v2(d_cnt, 0u, K));
                uint32_t start = 0, batch = M, tw = M * kPatchPix;
                void* t_args[] = {&d_imgs, &d_assign, &d_sum, &d_cnt, &pp, &K, &start, &batch};
                CU_CHECK(cuEventRecord(ev_s, nullptr));
                CU_CHECK(cuLaunchKernel(fn_train, (tw + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, t_args, nullptr));
                CU_CHECK(cuEventRecord(ev_e, nullptr));
                CU_CHECK(cuEventSynchronize(ev_e));
                CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
                total_train_gpu_ms += ms;

                CU_CHECK(cuMemcpyDtoH_v2(h_sum.data(), d_sum, h_sum.size() * sizeof(float)));
                CU_CHECK(cuMemcpyDtoH_v2(h_cnt.data(), d_cnt, h_cnt.size() * sizeof(uint32_t)));

                for (uint32_t k = 0; k < K; ++k) {
                    float* dst = &h_cents[static_cast<size_t>(k) * kPatchPix];
                    if (h_cnt[k] == 0u) {
                        uint32_t pick = (worst_i + k * 97u) % M;
                        if (!normalize_u8(&cls_px[static_cast<size_t>(pick) * kPatchPix], kPatchPix, dst) && pool_n > 0) {
                            std::memcpy(dst, &pool_vecs[static_cast<size_t>(k % pool_n) * kPatchPix], kPatchPix * sizeof(float));
                        }
                        continue;
                    }
                    double n2 = 0.0;
                    const float* sm = &h_sum[static_cast<size_t>(k) * kPatchPix];
                    for (uint32_t p = 0; p < kPatchPix; ++p) n2 += static_cast<double>(sm[p]) * sm[p];
                    float inv = static_cast<float>(1.0 / std::sqrt(n2 + 1e-12));
                    for (uint32_t p = 0; p < kPatchPix; ++p) dst[p] = sm[p] * inv;
                }
            }

            CU_CHECK(cuMemFree_v2(d_imgs)); CU_CHECK(cuMemFree_v2(d_cents));
            CU_CHECK(cuMemFree_v2(d_scores)); CU_CHECK(cuMemFree_v2(d_assign));
            CU_CHECK(cuMemFree_v2(d_sum)); CU_CHECK(cuMemFree_v2(d_cnt));

            // Accumulate true average RGB prototype per cluster (for crisp 3D RGB visualization + H heatmap toggle!)
            std::vector<double> rgb_r(static_cast<size_t>(K) * kPatchPix, 0.0);
            std::vector<double> rgb_g(static_cast<size_t>(K) * kPatchPix, 0.0);
            std::vector<double> rgb_b(static_cast<size_t>(K) * kPatchPix, 0.0);
            std::vector<uint32_t> mask_cnt(static_cast<size_t>(K) * kPatchPix, 0u);
            for (uint32_t i = 0; i < M; ++i) {
                uint32_t k = h_assign[i];
                const auto& rgb  = patches[ids[i]].rgb[comp];
                const auto& msk  = patches[ids[i]].mask[comp];
                size_t base = static_cast<size_t>(k) * kPatchPix;
                for (uint32_t p = 0; p < kPatchPix; ++p) {
                    if (msk[p]) {
                        rgb_r[base + p] += rgb[p * 3 + 0];
                        rgb_g[base + p] += rgb[p * 3 + 1];
                        rgb_b[base + p] += rgb[p * 3 + 2];
                        mask_cnt[base + p]++;
                    }
                }
            }

            // Place all non-empty K cluster slices neatly stacked along Z inside the compartment's 3D box!
            int64_t cx, cy, z0;
            comp_center(sp, comp, cx, cy, z0);
            uint32_t mid_id = comp_manifold[sp][comp];
            const auto* mrec = model.get_manifold(mid_id);
            uint32_t placed = 0;
            for (uint32_t k = 0; k < K; ++k) {
                if (h_cnt[k] == 0u) continue;
                int64_t ox = cx - static_cast<int64_t>(kPatchDim / 2u);
                int64_t oy = cy - static_cast<int64_t>(kPatchDim / 2u);
                int64_t oz = z0 + static_cast<int64_t>(placed) * dz_step;
                if (!model.inside_sphere(ox, oy, oz, kPatchDim, kPatchDim, 0.90f) ||
                    model.overlaps_any(static_cast<uint32_t>(ox), static_cast<uint32_t>(oy),
                                       static_cast<uint32_t>(oz), kPatchDim, kPatchDim, 1u, 2u)) {
                    std::cerr << "[ERROR] Overlap/bounds failure placing " << sp_name << "_" << kCompNames[comp]
                              << " cluster #" << k << " at (" << ox << "," << oy << "," << oz << ")\n";
                    return 1;
                }
                gllm::ClassRecord rec;
                rec.label       = std::string(sp_name) + "_" + kCompNames[comp];
                rec.ox          = static_cast<uint32_t>(ox);
                rec.oy          = static_cast<uint32_t>(oy);
                rec.oz          = static_cast<uint32_t>(oz);
                rec.width       = kPatchDim;
                rec.height      = kPatchDim;
                rec.manifold_id = mid_id;
                rec.outer_idx   = mrec->outer_idx;
                rec.outer_x     = mrec->outer_x;
                rec.outer_y     = mrec->outer_y;
                rec.outer_z     = mrec->outer_z;
                rec.train_count = h_cnt[k];
                rec.channels    = 3u;
                rec.shadow_sum.resize(kPatchPix);
                rec.shadow_rgb.resize(kPatchPix, 0u);

                size_t base = static_cast<size_t>(k) * kPatchPix;
                for (uint32_t p = 0; p < kPatchPix; ++p) {
                    rec.shadow_sum[p] = h_sum[base + p];
                    uint32_t mc = mask_cnt[base + p];
                    if (mc * 4u >= h_cnt[k] && mc > 0u) {
                        uint32_t rr = static_cast<uint32_t>(std::clamp(std::lround(rgb_r[base + p] / mc), 0l, 255l));
                        uint32_t gg = static_cast<uint32_t>(std::clamp(std::lround(rgb_g[base + p] / mc), 0l, 255l));
                        uint32_t bb = static_cast<uint32_t>(std::clamp(std::lround(rgb_b[base + p] / mc), 0l, 255l));
                        rec.shadow_rgb[p] = 0x01000000u | (rr << 16) | (gg << 8) | bb;
                    }
                }
                uint32_t reg_idx = static_cast<uint32_t>(model.classes.size());
                comp_region_ids[sp][comp].push_back(reg_idx);
                model.classes.push_back(std::move(rec));
                placed++;
            }
            std::printf("  [%s] %-16s -> Manifold #%-2u | %2u sub-shadows | %4u imgs | Outer #%u ('%s')\n",
                        sp == 0 ? "CAT" : "DOG", (std::string(sp_name) + "_" + kCompNames[comp]).c_str(),
                        mid_id, placed, M, mrec->outer_idx, mrec->concept_label.c_str());
        }
    }

    // Recompute exact bottom-up 3D bounding boxes across all 4 levels & save .gllm v5!
    model.recompute_manifold_envelopes();
    if (!model.save(model_path)) {
        std::cerr << "[ERROR] Failed to save model: " << model_path << "\n";
        return 1;
    }
    std::printf("\n[SAVED]  %s | %zu manifolds | %zu sub-shadow regions | GPU K-Means time: %.2f ms\n",
                model_path.c_str(), model.manifolds.size(), model.classes.size(), total_train_gpu_ms);

    // -------------------------------------------------------------------------
    // 5. Bottom-Up Hierarchical Unseen Cat vs. Dog GPU Evaluation (1,283 holdout!)
    //    Scores strictly against the saved L2-normalized shadow templates via CUDA!
    // -------------------------------------------------------------------------
    const uint32_t N_test = static_cast<uint32_t>(test_ids.size());
    std::array<std::vector<float>, kNumComps> test_margin_raw;
    std::array<std::vector<float>, kNumComps> test_margin_cal;
    std::array<double, kNumComps> comp_train_bias{};
    std::array<double, kNumComps> comp_train_sd{};
    float total_eval_gpu_ms = 0.0f;

    // Combine train IDs for zero-leakage calibration statistics (mean/std of raw margin on train only)
    std::vector<size_t> all_train_ids;
    all_train_ids.reserve(train_ids[0].size() + train_ids[1].size());
    all_train_ids.insert(all_train_ids.end(), train_ids[0].begin(), train_ids[0].end());
    all_train_ids.insert(all_train_ids.end(), train_ids[1].begin(), train_ids[1].end());
    const uint32_t N_train = static_cast<uint32_t>(all_train_ids.size());

    auto score_batch_gpu = [&](const std::vector<size_t>& sample_ids, uint32_t comp,
                               const std::vector<float>& tmpls, uint32_t n_cat, uint32_t n_dog,
                               std::vector<float>& out_margin, float* gpu_ms_accum) {
        uint32_t N = static_cast<uint32_t>(sample_ids.size());
        uint32_t T = n_cat + n_dog;
        std::vector<uint8_t> imgs(static_cast<size_t>(N) * kPatchPix);
        for (uint32_t i = 0; i < N; ++i) {
            std::memcpy(&imgs[static_cast<size_t>(i) * kPatchPix],
                        patches[sample_ids[i]].shadow[comp].data(), kPatchPix);
        }
        CUdeviceptr d_i = 0, d_t = 0, d_s = 0;
        CU_CHECK(cuMemAlloc_v2(&d_i, imgs.size()));
        CU_CHECK(cuMemcpyHtoD_v2(d_i, imgs.data(), imgs.size()));
        CU_CHECK(cuMemAlloc_v2(&d_t, tmpls.size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_t, tmpls.data(), tmpls.size() * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&d_s, static_cast<size_t>(N) * T * sizeof(float)));

        uint32_t pp = kPatchPix, pairs = N * T;
        void* args[] = {&d_i, &d_t, &d_s, &pp, &T, &N};
        CU_CHECK(cuEventRecord(ev_s, nullptr));
        CU_CHECK(cuLaunchKernel(fn_eval, (pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        CU_CHECK(cuEventRecord(ev_e, nullptr));
        CU_CHECK(cuEventSynchronize(ev_e));
        float ms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
        if (gpu_ms_accum) *gpu_ms_accum += ms;

        std::vector<float> scores(static_cast<size_t>(N) * T);
        CU_CHECK(cuMemcpyDtoH_v2(scores.data(), d_s, scores.size() * sizeof(float)));
        CU_CHECK(cuMemFree_v2(d_i)); CU_CHECK(cuMemFree_v2(d_t)); CU_CHECK(cuMemFree_v2(d_s));

        out_margin.resize(N);
        for (uint32_t i = 0; i < N; ++i) {
            const float* row = &scores[static_cast<size_t>(i) * T];
            float best_cat = -1e9f, best_dog = -1e9f;
            for (uint32_t k = 0; k < n_cat; ++k) best_cat = std::max(best_cat, row[k]);
            for (uint32_t k = 0; k < n_dog; ++k) best_dog = std::max(best_dog, row[n_cat + k]);
            out_margin[i] = best_cat - best_dog; // > 0 => CAT, < 0 => DOG
        }
    };

    for (uint32_t comp = 0; comp < kNumComps; ++comp) {
        const auto& c_regs = comp_region_ids[0][comp];
        const auto& d_regs = comp_region_ids[1][comp];
        uint32_t n_cat = static_cast<uint32_t>(c_regs.size());
        uint32_t n_dog = static_cast<uint32_t>(d_regs.size());
        std::vector<float> tmpls(static_cast<size_t>(n_cat + n_dog) * kPatchPix, 0.0f);
        auto pack_tmpl = [&](uint32_t dst_slot, uint32_t reg_idx) {
            const auto& rec = model.classes[reg_idx];
            double n2 = 0.0;
            for (uint32_t p = 0; p < kPatchPix; ++p) n2 += static_cast<double>(rec.shadow_sum[p]) * rec.shadow_sum[p];
            float inv = n2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
            for (uint32_t p = 0; p < kPatchPix; ++p) tmpls[static_cast<size_t>(dst_slot) * kPatchPix + p] = rec.shadow_sum[p] * inv;
        };
        for (uint32_t k = 0; k < n_cat; ++k) pack_tmpl(k, c_regs[k]);
        for (uint32_t k = 0; k < n_dog; ++k) pack_tmpl(n_cat + k, d_regs[k]);

        // Zero-leakage balanced midpoint calibration on TRAINING split only:
        // midpoint = 0.5 * (mean_train_cat_margin + mean_train_dog_margin)
        std::vector<float> tr_m;
        score_batch_gpu(all_train_ids, comp, tmpls, n_cat, n_dog, tr_m, nullptr);
        double mc = 0.0, md = 0.0;
        size_t nc = 0, nd = 0;
        for (uint32_t i = 0; i < N_train; ++i) {
            if (rows[all_train_ids[i]].species == 0) { mc += tr_m[i]; nc++; }
            else { md += tr_m[i]; nd++; }
        }
        mc /= std::max<size_t>(1, nc);
        md /= std::max<size_t>(1, nd);
        double mid_bias = 0.5 * (mc + md);
        double var_sum = 0.0;
        for (uint32_t i = 0; i < N_train; ++i) {
            double d = tr_m[i] - mid_bias;
            var_sum += d * d;
        }
        double sd = std::max(1e-5, std::sqrt(var_sum / std::max<uint32_t>(1u, N_train)));
        comp_train_bias[comp] = mid_bias;
        comp_train_sd[comp]   = sd;

        // Strictly unseen holdout GPU scoring!
        score_batch_gpu(test_ids, comp, tmpls, n_cat, n_dog, test_margin_raw[comp], &total_eval_gpu_ms);
        test_margin_cal[comp].resize(N_test);
        for (uint32_t i = 0; i < N_test; ++i) {
            test_margin_cal[comp][i] = static_cast<float>((test_margin_raw[comp][i] - mid_bias) / sd);
        }
    }

    auto print_eval_row = [&](const char* label, const std::vector<float>& margins) {
        uint32_t cc = 0, cd = 0, dc = 0, dd = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            int truth = rows[test_ids[i]].species; // 0=cat, 1=dog
            int pred  = (margins[i] >= 0.0f) ? 0 : 1;
            if (truth == 0 && pred == 0) cc++;
            else if (truth == 0 && pred == 1) cd++;
            else if (truth == 1 && pred == 0) dc++;
            else dd++;
        }
        double acc = 100.0 * (cc + dd) / std::max(1u, N_test);
        double r_cat = 100.0 * cc / std::max(1u, cc + cd);
        double r_dog = 100.0 * dd / std::max(1u, dc + dd);
        double bal = 0.5 * (r_cat + r_dog);
        std::printf("  %-34s | Acc: %5.2f%% | Bal: %5.2f%% | Cat: %5.2f%% (%3u/%3u) | Dog: %5.2f%% (%3u/%3u)\n",
                    label, acc, bal, r_cat, cc, cc + cd, r_dog, dd, dc + dd);
    };

    auto build_fused = [&](const std::array<std::vector<float>, kNumComps>& src,
                           const std::vector<uint32_t>& comps) {
        std::vector<float> out(N_test, 0.0f);
        for (uint32_t i = 0; i < N_test; ++i) {
            float s = 0.0f;
            for (uint32_t c : comps) s += src[c][i];
            out[i] = s;
        }
        return out;
    };

    std::cout << "\n========================================================================================================\n";
    std::printf("  BOTTOM-UP HIERARCHICAL UNSEEN CAT vs. DOG GPU AUDIT (%u unseen holdout images | K=%u | mode=%s)\n",
                N_test, num_clusters, mode_str.c_str());
    std::cout << "========================================================================================================\n";

    std::cout << "\n[A] Pure Raw Zero-Parameter Cosine Shadow Overlap (best_cat_shadow - best_dog_shadow >= 0):\n";
    for (uint32_t f = 2; f < 8; ++f) {
        std::string nm = "L3 Feature: " + kCompNames[f];
        print_eval_row(nm.c_str(), test_margin_raw[f]);
    }
    print_eval_row("L3 5 FEATURES (excl. nose)",        build_fused(test_margin_raw, {2, 3, 4, 5, 7}));
    print_eval_row("L3 ALL 6 FEATURES FUSED",           build_fused(test_margin_raw, {2, 3, 4, 5, 6, 7}));
    print_eval_row("L2 Face Manifold Only",             test_margin_raw[1]);
    print_eval_row("L3(5)+L2 (excl. nose) -> Face",     build_fused(test_margin_raw, {1, 2, 3, 4, 5, 7}));
    print_eval_row("L3+L2 Features -> Face Fusion",     build_fused(test_margin_raw, {1, 2, 3, 4, 5, 6, 7}));
    print_eval_row("L1 Full Body Manifold Only",        test_margin_raw[0]);
    print_eval_row("BOTTOM-UP (excl. nose, 7 parts)",   build_fused(test_margin_raw, {0, 1, 2, 3, 4, 5, 7}));
    print_eval_row("BOTTOM-UP FULL (L3 -> L2 -> L1)",   build_fused(test_margin_raw, {0, 1, 2, 3, 4, 5, 6, 7}));

    std::cout << "\n[B] Train-Centered Equal-Variance Bottom-Up Fusion (zero test leakage; calibrated on train split only):\n";
    for (uint32_t f = 2; f < 8; ++f) {
        std::string nm = "L3 Feature: " + kCompNames[f];
        print_eval_row(nm.c_str(), test_margin_cal[f]);
    }
    print_eval_row("L3 ALL 6 FEATURES FUSED",           build_fused(test_margin_cal, {2, 3, 4, 5, 6, 7}));
    print_eval_row("L2 Face Manifold Only",             test_margin_cal[1]);
    print_eval_row("L3+L2 Features -> Face Fusion",     build_fused(test_margin_cal, {1, 2, 3, 4, 5, 6, 7}));
    print_eval_row("L1 Full Body Manifold Only",        test_margin_cal[0]);
    print_eval_row("BOTTOM-UP FULL (L3 -> L2 -> L1)",   build_fused(test_margin_cal, {0, 1, 2, 3, 4, 5, 6, 7}));

    double imgs_per_sec = (total_eval_gpu_ms > 0.0f) ? (static_cast<double>(N_test) * 1000.0 / total_eval_gpu_ms) : 0.0;
    std::printf("\n[SPEED]  Full 8-compartment GPU inference across %u unseen images: %.2f ms (%.0f full-hierarchy imgs/sec!)\n",
                N_test, total_eval_gpu_ms, imgs_per_sec);

    CU_CHECK(cuEventDestroy_v2(ev_s));
    CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
