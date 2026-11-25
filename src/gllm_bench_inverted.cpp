// =============================================================================
// gllm_bench_inverted — Inverted Child-First Gradient-Domain Bottom-Up Benchmark
//
// Zero neural weights. Zero backpropagation. ZERO test bounding boxes!
//
// Inverted Pipeline (Child Features FIRST -> Derived Bounding Boxes!):
//   Step 1: Convert raw uncropped image into a Gradient-Domain Sparse Edge Map
//           (CMU 15-463 Lecture 9: log-luminance + edge-preserving bilateral +
//            forward-difference gradient field + adaptive sparsification).
//   Step 2: Skip empty flat black regions in O(1); scan ALL 6 lowest child
//           feature manifolds (ear_right, ear_left, eye_right, eye_left, nose,
//           mouth) directly across the active edge map on CUDA!
//   Step 3: Generalized Hough Geometric Constellation Voting — EVERY detected
//           child edge match projects backward to vote for an implied parent
//           face constellation; the highest cross-feature consensus CREATES
//           the 6 verified child feature boxes + enclosing parent face box!
//   Step 4: Confirm parent L2 face + L1 full-image edge overlap and classify
//           Cat vs. Dog strictly bottom-up!
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"
#include "../include/grad_domain.hpp"

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

static constexpr uint32_t kPatchPix = 784u;
static constexpr uint32_t kNumComps = 8u;
static const std::array<std::string, kNumComps> kCompNames = {
    "full", "face", "ear_right", "ear_left", "eye_right", "eye_left", "nose", "mouth"
};

struct BBox { float x0 = 0, y0 = 0, x1 = 0, y1 = 0; };
struct RelPrior { float cx = 0.5f, cy = 0.5f, w = 0.25f, h = 0.25f; };

struct ManifestRow {
    bool is_test = false;
    int  species = 0;
    std::string image_path;
    BBox face_box;
    std::array<BBox, 6> feat_boxes;
};

struct RawSample {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgb;
    gllm::GradEdgeImage  ge;
    bool ok = false;
};

struct ChildDet {
    float cx, cy, bw, bh;
    float univ_s, cat_s, dog_s;
};

static bool parse_bbox(const std::string& s, BBox& b) {
    float v[4] = {};
    int k = 0;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',') && k < 4) v[k++] = std::strtof(tok.c_str(), nullptr);
    if (k != 4) return false;
    b = {v[0], v[1], v[2], v[3]};
    return true;
}

static bool load_manifest(const std::string& path, std::vector<ManifestRow>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cols;
        size_t s = 0;
        while (true) {
            size_t t = line.find('\t', s);
            if (t == std::string::npos) { cols.push_back(line.substr(s)); break; }
            cols.push_back(line.substr(s, t - s));
            s = t + 1;
        }
        if (cols.size() < 10) continue;
        ManifestRow r;
        r.is_test = (cols[0] == "test");
        r.species = (cols[1] == "dog") ? 1 : 0;
        r.image_path = cols[2];
        if (!parse_bbox(cols[3], r.face_box)) continue;
        bool ok = true;
        for (int f = 0; f < 6; ++f) if (!parse_bbox(cols[4 + f], r.feat_boxes[f])) { ok = false; break; }
        if (ok) out.push_back(std::move(r));
    }
    return !out.empty();
}

static bool load_png_grad(const std::string& path, bool keep_rgb, RawSample& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    png_byte sig[8];
    if (std::fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8) != 0) { std::fclose(fp); return false; }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); std::fclose(fp); return false; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, nullptr); std::fclose(fp); return false; }
    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_uint_32 w = png_get_image_width(png, info);
    png_uint_32 h = png_get_image_height(png, info);
    int bd = png_get_bit_depth(png, info), ct = png_get_color_type(png, info);
    if (bd == 16) png_set_strip_16(png);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_GRAY && bd < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (ct & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
    png_read_update_info(png, info);

    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3u);
    std::vector<png_bytep> rows(h);
    for (png_uint_32 y = 0; y < h; ++y) rows[y] = rgb.data() + static_cast<size_t>(y) * w * 3u;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    out.w = w; out.h = h;
    gllm::build_grad_edge_image(rgb.data(), w, h, out.ge);
    if (keep_rgb) out.rgb = std::move(rgb);
    out.ok = (w > 2 && h > 2);
    return out.ok;
}

static float box_iou(const BBox& a, const BBox& b) {
    float ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    float ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    float inter = std::max(0.0f, ix1 - ix0) * std::max(0.0f, iy1 - iy0);
    float aa = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
    float ba = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
    return (aa + ba - inter > 1e-5f) ? (inter / (aa + ba - inter)) : 0.0f;
}

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

int main(int argc, char** argv) {
    std::string model_path    = "models/animals_grad.gllm";
    std::string manifest_path = "data/cat_dog_split.tsv";
    std::string viz_out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (a == "--viz-out" && i + 1 < argc) viz_out_path = argv[++i];
    }

    // 1. Load trained gradient-domain .gllm model & build Zero-Mean templates + canonical archetypes
    gllm::GeometricModel model;
    if (!model.load(model_path)) { std::cerr << "[ERROR] Cannot load " << model_path << "\n"; return 1; }

    std::vector<uint32_t> comp_regs[2][kNumComps];
    for (uint32_t r = 0; r < model.classes.size(); ++r) {
        const auto& c = model.classes[r];
        if (c.width != 28u || c.height != 28u) continue;
        for (int sp = 0; sp < 2; ++sp) {
            std::string pfx = (sp == 0) ? "cat_" : "dog_";
            for (uint32_t comp = 0; comp < kNumComps; ++comp) {
                if (c.label == pfx + kCompNames[comp]) comp_regs[sp][comp].push_back(r);
            }
        }
    }

    auto make_zncc = [](const std::vector<double>& raw, float* dst) {
        double mean = 0.0;
        for (uint32_t p = 0; p < kPatchPix; ++p) mean += raw[p];
        mean /= kPatchPix;
        double n2 = 0.0;
        for (uint32_t p = 0; p < kPatchPix; ++p) { double d = raw[p] - mean; n2 += d * d; }
        float inv = n2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
        for (uint32_t p = 0; p < kPatchPix; ++p) dst[p] = static_cast<float>(raw[p] - mean) * inv;
    };

    std::array<std::vector<float>, kNumComps> comp_tmpls, comp_canon, comp_univ;
    std::array<uint32_t, kNumComps> comp_ncat{}, comp_ndog{};
    for (uint32_t comp = 0; comp < kNumComps; ++comp) {
        uint32_t nc = static_cast<uint32_t>(comp_regs[0][comp].size());
        uint32_t nd = static_cast<uint32_t>(comp_regs[1][comp].size());
        comp_ncat[comp] = nc; comp_ndog[comp] = nd;
        comp_tmpls[comp].resize(static_cast<size_t>(nc + nd) * kPatchPix);
        comp_canon[comp].resize(2u * kPatchPix);
        comp_univ[comp].resize(kPatchPix);
        std::vector<double> sc(kPatchPix, 0.0), sd(kPatchPix, 0.0), sa(kPatchPix, 0.0);
        uint64_t tc = 0, td = 0;
        for (uint32_t k = 0; k < nc; ++k) {
            const auto& rec = model.classes[comp_regs[0][comp][k]];
            std::vector<double> rv(kPatchPix);
            for (uint32_t p = 0; p < kPatchPix; ++p) { rv[p] = rec.shadow_sum[p]; sc[p] += rv[p]; }
            tc += rec.train_count;
            make_zncc(rv, &comp_tmpls[comp][static_cast<size_t>(k) * kPatchPix]);
        }
        for (uint32_t k = 0; k < nd; ++k) {
            const auto& rec = model.classes[comp_regs[1][comp][k]];
            std::vector<double> rv(kPatchPix);
            for (uint32_t p = 0; p < kPatchPix; ++p) { rv[p] = rec.shadow_sum[p]; sd[p] += rv[p]; }
            td += rec.train_count;
            make_zncc(rv, &comp_tmpls[comp][static_cast<size_t>(nc + k) * kPatchPix]);
        }
        for (uint32_t p = 0; p < kPatchPix; ++p) {
            sc[p] /= std::max<uint64_t>(1, tc);
            sd[p] /= std::max<uint64_t>(1, td);
            sa[p]  = 0.5 * (sc[p] + sd[p]);
        }
        make_zncc(sc, &comp_canon[comp][0]);
        make_zncc(sd, &comp_canon[comp][kPatchPix]);
        make_zncc(sa, &comp_univ[comp][0]);
    }

    // 2. Load manifest: compute species-neutral relative geometry strictly on TRAIN split, load TEST raw images
    std::vector<ManifestRow> all_rows, test_rows;
    if (!load_manifest(manifest_path, all_rows)) return 1;
    std::array<RelPrior, 6> prior{};
    {
        double scx[2][6]{}, scy[2][6]{}, sw[2][6]{}, sh[2][6]{};
        size_t cnt[2]{};
        for (const auto& r : all_rows) {
            if (r.is_test) { test_rows.push_back(r); continue; }
            float fw = std::max(1.0f, r.face_box.x1 - r.face_box.x0);
            float fh = std::max(1.0f, r.face_box.y1 - r.face_box.y0);
            cnt[r.species]++;
            for (int f = 0; f < 6; ++f) {
                const auto& fb = r.feat_boxes[f];
                scx[r.species][f] += (0.5 * (fb.x0 + fb.x1) - r.face_box.x0) / fw;
                scy[r.species][f] += (0.5 * (fb.y0 + fb.y1) - r.face_box.y0) / fh;
                sw[r.species][f]  += std::max(0.05, static_cast<double>(fb.x1 - fb.x0) / fw);
                sh[r.species][f]  += std::max(0.05, static_cast<double>(fb.y1 - fb.y0) / fh);
            }
        }
        for (int f = 0; f < 6; ++f) {
            prior[f] = {
                static_cast<float>(0.5 * (scx[0][f] / cnt[0] + scx[1][f] / cnt[1])),
                static_cast<float>(0.5 * (scy[0][f] / cnt[0] + scy[1][f] / cnt[1])),
                static_cast<float>(0.5 * (sw[0][f]  / cnt[0] + sw[1][f]  / cnt[1])),
                static_cast<float>(0.5 * (sh[0][f]  / cnt[0] + sh[1][f]  / cnt[1]))
            };
        }
    }
    const uint32_t N_test = static_cast<uint32_t>(test_rows.size());
    std::vector<bool> keep_rgb(N_test, false);
    std::vector<uint32_t> viz_ids;
    if (!viz_out_path.empty()) {
        int cc = 0, dc = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            if (test_rows[i].species == 0 && cc < 4) { keep_rgb[i] = true; viz_ids.push_back(i); cc++; }
            if (test_rows[i].species == 1 && dc < 4) { keep_rgb[i] = true; viz_ids.push_back(i); dc++; }
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    uint32_t n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<RawSample> samples(N_test);
    std::atomic<uint32_t> nxt{0};
    std::vector<std::thread> ths;
    for (uint32_t t = 0; t < n_threads; ++t) {
        ths.emplace_back([&]() {
            while (true) {
                uint32_t i = nxt.fetch_add(1);
                if (i >= N_test) break;
                load_png_grad(test_rows[i].image_path, keep_rgb[i], samples[i]);
            }
        });
    }
    for (auto& th : ths) th.join();
    double dec_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[GRAD]   Built gradient-domain sparse edge maps for %u raw unseen images in %.2fs (0 test bboxes!)\n",
                N_test, dec_sec);

    // 3. CUDA setup: pack all 6 child universal+canon templates into one fast batch & full K=16 templates
    CU_CHECK(cuInit(0));
    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod = gllm::load_gllm_ptx_module();
    CUfunction fn_eval = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_eval, mod, "eval_score_shadows"));
    CUevent ev_s = nullptr, ev_e = nullptr;
    CU_CHECK(cuEventCreate(&ev_s, 0));
    CU_CHECK(cuEventCreate(&ev_e, 0));

    // Pack 18 child-first detection templates (6 features x [univ, cat_avg, dog_avg]) into ONE VRAM buffer!
    // One single CUDA kernel launch cross-checks EVERY active edge patch against ALL 6 child feature manifolds!
    std::vector<float> h_child18(18u * kPatchPix);
    for (int f = 0; f < 6; ++f) {
        std::memcpy(&h_child18[static_cast<size_t>(f * 3 + 0) * kPatchPix], comp_univ[2 + f].data(), kPatchPix * sizeof(float));
        std::memcpy(&h_child18[static_cast<size_t>(f * 3 + 1) * kPatchPix], &comp_canon[2 + f][0],   kPatchPix * sizeof(float));
        std::memcpy(&h_child18[static_cast<size_t>(f * 3 + 2) * kPatchPix], &comp_canon[2 + f][kPatchPix], kPatchPix * sizeof(float));
    }
    CUdeviceptr d_child18 = 0;
    CU_CHECK(cuMemAlloc_v2(&d_child18, h_child18.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_child18, h_child18.data(), h_child18.size() * sizeof(float)));

    std::array<CUdeviceptr, kNumComps> d_tmpls{}, d_canon{};
    for (uint32_t c = 0; c < kNumComps; ++c) {
        CU_CHECK(cuMemAlloc_v2(&d_tmpls[c], comp_tmpls[c].size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_tmpls[c], comp_tmpls[c].data(), comp_tmpls[c].size() * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&d_canon[c], comp_canon[c].size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_canon[c], comp_canon[c].data(), comp_canon[c].size() * sizeof(float)));
    }

    auto gpu_score_ptr = [&](const std::vector<uint8_t>& patches, uint32_t M,
                             CUdeviceptr d_t, uint32_t T,
                             std::vector<float>& out, float& gpu_ms) {
        out.resize(static_cast<size_t>(M) * T);
        constexpr uint32_t CHUNK = 65536u;
        CUdeviceptr d_in = 0, d_sc = 0;
        uint32_t am = std::min(M, CHUNK);
        CU_CHECK(cuMemAlloc_v2(&d_in, static_cast<size_t>(am) * kPatchPix));
        CU_CHECK(cuMemAlloc_v2(&d_sc, static_cast<size_t>(am) * T * sizeof(float)));
        for (uint32_t off = 0; off < M; off += CHUNK) {
            uint32_t cnt = std::min(CHUNK, M - off);
            CU_CHECK(cuMemcpyHtoD_v2(d_in, &patches[static_cast<size_t>(off) * kPatchPix],
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
            CU_CHECK(cuMemcpyDtoH_v2(&out[static_cast<size_t>(off) * T], d_sc,
                                     static_cast<size_t>(cnt) * T * sizeof(float)));
        }
        CU_CHECK(cuMemFree_v2(d_in));
        CU_CHECK(cuMemFree_v2(d_sc));
    };

    // =========================================================================
    // STEP 2 (INVERTED — CHILD FEATURES FIRST!):
    //   Scan active sparse-edge windows across all 6 lowest child feature scales!
    //   Skip flat empty black regions in O(1) — zero blind face boxes generated!
    // =========================================================================
    auto t_inv0 = std::chrono::steady_clock::now();
    std::vector<std::vector<BBox>> img_wins(N_test);
    std::vector<std::vector<uint8_t>> img_patches(N_test);
    std::atomic<uint64_t> total_cand_wins{0}, skipped_empty_wins{0};
    {
        std::atomic<uint32_t> ni{0};
        std::vector<std::thread> ws;
        for (uint32_t t = 0; t < n_threads; ++t) {
            ws.emplace_back([&]() {
                uint8_t buf[kPatchPix];
                while (true) {
                    uint32_t i = ni.fetch_add(1);
                    if (i >= N_test) break;
                    const auto& ge = samples[i].ge;
                    float fw = static_cast<float>(ge.w), fh = static_cast<float>(ge.h);
                    float mn = std::min(fw, fh);
                    // Multi-scale child feature grid covering small eyes (6%) up to large ears/muzzles (34%)
                    const float scales[5] = {0.07f, 0.11f, 0.17f, 0.25f, 0.35f};
                    uint64_t tot_w = 0, sk_w = 0;
                    for (float sc : scales) {
                        float side = mn * sc;
                        float step = std::max(8.0f, side * 0.42f);
                        int nx = std::max(1, static_cast<int>(std::floor((fw - side) / step)) + 1);
                        int ny = std::max(1, static_cast<int>(std::floor((fh - side) / step)) + 1);
                        float rx = (nx > 1) ? (fw - side) / static_cast<float>(nx - 1) : 0.0f;
                        float ry = (ny > 1) ? (fh - side) / static_cast<float>(ny - 1) : 0.0f;
                        for (int iy = 0; iy < ny; ++iy) {
                            for (int ix = 0; ix < nx; ++ix) {
                                tot_w++;
                                float x0 = (nx > 1) ? ix * rx : 0.5f * (fw - side);
                                float y0 = (ny > 1) ? iy * ry : 0.5f * (fh - side);
                                if (!gllm::extract_grad_edge_patch_28x28(ge, x0, y0, x0 + side, y0 + side,
                                                                         true, 0.015f, buf)) {
                                    sk_w++;
                                    continue; // Sparse flat region skipped in O(1)!
                                }
                                img_wins[i].push_back({x0, y0, x0 + side, y0 + side});
                                img_patches[i].insert(img_patches[i].end(), buf, buf + kPatchPix);
                            }
                        }
                    }
                    total_cand_wins.fetch_add(tot_w);
                    skipped_empty_wins.fetch_add(sk_w);
                }
            });
        }
        for (auto& w : ws) w.join();
    }

    // Flatten active sparse edge windows for ONE massive parallel GPU child-feature cross-check!
    std::vector<uint32_t> w_off(N_test + 1, 0);
    for (uint32_t i = 0; i < N_test; ++i) w_off[i + 1] = w_off[i] + static_cast<uint32_t>(img_wins[i].size());
    uint32_t M_active = w_off[N_test];
    std::vector<uint8_t> flat_patches(static_cast<size_t>(M_active) * kPatchPix);
    for (uint32_t i = 0; i < N_test; ++i) {
        if (!img_patches[i].empty()) {
            std::memcpy(&flat_patches[static_cast<size_t>(w_off[i]) * kPatchPix],
                        img_patches[i].data(), img_patches[i].size());
            std::vector<uint8_t>().swap(img_patches[i]);
        }
    }

    float gpu_ms = 0.0f;
    std::vector<float> child18_scores;
    gpu_score_ptr(flat_patches, M_active, d_child18, 18u, child18_scores, gpu_ms);
    std::vector<uint8_t>().swap(flat_patches);

    std::printf("[SPARSE] Total windows: %llu | Skipped flat empty black regions: %llu (%.1f%%) | Active scored on GPU: %u (%.1f ms)\n",
                static_cast<unsigned long long>(total_cand_wins.load()),
                static_cast<unsigned long long>(skipped_empty_wins.load()),
                100.0 * skipped_empty_wins.load() / std::max<uint64_t>(1, total_cand_wins.load()),
                M_active, gpu_ms);

    // =========================================================================
    // STEP 3 (BOTTOM-UP BOUNDING BOX CREATION VIA GENERALIZED HOUGH VOTING!):
    //   From the Top-40 child feature edge detections per feature f=0..5,
    //   EVERY child detection votes for an implied parent face box!
    //   Where all 6 child feature detections geometrically converge CREATES
    //   the derived face box and 6 verified child feature bounding boxes!
    // =========================================================================
    std::vector<BBox> derived_face(N_test);
    std::vector<std::array<BBox, 6>> derived_feats(N_test);
    std::vector<int> derived_matched_cnt(N_test, 0);
    {
        std::atomic<uint32_t> ni{0};
        std::vector<std::thread> ws;
        for (uint32_t t = 0; t < n_threads; ++t) {
            ws.emplace_back([&]() {
                while (true) {
                    uint32_t i = ni.fetch_add(1);
                    if (i >= N_test) break;
                    const auto& wins = img_wins[i];
                    uint32_t base = w_off[i];
                    float fw_img = static_cast<float>(samples[i].w);
                    float fh_img = static_cast<float>(samples[i].h);
                    float mn_img = std::min(fw_img, fh_img);

                    // Extract Top-40 NMS child edge detections per feature f=0..5
                    std::array<std::vector<ChildDet>, 6> dets;
                    for (int f = 0; f < 6; ++f) {
                        std::vector<ChildDet> cand;
                        cand.reserve(wins.size());
                        for (size_t wi = 0; wi < wins.size(); ++wi) {
                            const float* sc = &child18_scores[static_cast<size_t>(base + wi) * 18u + f * 3u];
                            float u = std::max({sc[0], sc[1], sc[2]});
                            if (u <= 0.05f) continue;
                            const auto& b = wins[wi];
                            cand.push_back({0.5f * (b.x0 + b.x1), 0.5f * (b.y0 + b.y1),
                                            b.x1 - b.x0, b.y1 - b.y0, u, sc[1], sc[2]});
                        }
                        std::sort(cand.begin(), cand.end(), [](const ChildDet& a, const ChildDet& b) {
                            return a.univ_s > b.univ_s;
                        });
                        for (const auto& c : cand) {
                            bool dup = false;
                            BBox cb{c.cx - 0.5f * c.bw, c.cy - 0.5f * c.bh, c.cx + 0.5f * c.bw, c.cy + 0.5f * c.bh};
                            for (const auto& k : dets[f]) {
                                BBox kb{k.cx - 0.5f * k.bw, k.cy - 0.5f * k.bh, k.cx + 0.5f * k.bw, k.cy + 0.5f * k.bh};
                                if (box_iou(cb, kb) > 0.45f) { dup = true; break; }
                            }
                            if (!dup) { dets[f].push_back(c); if (dets[f].size() >= 36u) break; }
                        }
                    }

                    // Every detected child feature votes for an implied parent face constellation!
                    float best_score = -1e9f;
                    BBox  best_fb{0, 0, fw_img, fh_img};
                    std::array<BBox, 6> best_parts{};
                    int best_hits = 0;

                    for (int f = 0; f < 6; ++f) {
                        float rel_sz = std::sqrt(prior[f].w * prior[f].h);
                        for (size_t di = 0; di < std::min<size_t>(18u, dets[f].size()); ++di) {
                            const auto& d = dets[f][di];
                            float fsize = std::clamp(std::sqrt(d.bw * d.bh) / rel_sz, 0.24f * mn_img, 1.05f * mn_img);
                            float fcx   = d.cx - (prior[f].cx - 0.5f) * fsize;
                            float fcy   = d.cy - (prior[f].cy - 0.5f) * fsize;
                            if (fcx < 0.0f || fcx > fw_img || fcy < 0.0f || fcy > fh_img) continue;

                            // Gather best geometrically consistent detections from ALL 6 child feature lists!
                            float consensus = 0.0f;
                            int hits = 0;
                            std::array<BBox, 6> parts{};
                            for (int k = 0; k < 6; ++k) {
                                float ex = fcx + (prior[k].cx - 0.5f) * fsize;
                                float ey = fcy + (prior[k].cy - 0.5f) * fsize;
                                float ew = prior[k].w * fsize;
                                float eh = prior[k].h * fsize;
                                float esz = std::sqrt(ew * eh);
                                parts[k] = {ex - 0.5f * ew, ey - 0.5f * eh, ex + 0.5f * ew, ey + 0.5f * eh};

                                float best_k = 0.0f;
                                const ChildDet* hit_d = nullptr;
                                for (const auto& dk : dets[k]) {
                                    float dist = std::hypot(dk.cx - ex, dk.cy - ey);
                                    if (dist > 0.22f * fsize) continue;
                                    float dsz = std::sqrt(dk.bw * dk.bh);
                                    float ratio = dsz / std::max(1.0f, esz);
                                    if (ratio < 0.45f || ratio > 2.25f) continue;
                                    float geom_w = std::exp(-2.5f * (dist / (0.22f * fsize)) * (dist / (0.22f * fsize)));
                                    float sc = dk.univ_s * geom_w;
                                    if (sc > best_k) { best_k = sc; hit_d = &dk; }
                                }
                                if (hit_d) {
                                    hits++;
                                    consensus += best_k;
                                    // Refine part box directly from the verified child edge detection!
                                    parts[k] = {hit_d->cx - 0.5f * hit_d->bw, hit_d->cy - 0.5f * hit_d->bh,
                                                hit_d->cx + 0.5f * hit_d->bw, hit_d->cy + 0.5f * hit_d->bh};
                                }
                            }
                            if (consensus > best_score) {
                                best_score = consensus;
                                best_hits  = hits;
                                best_fb    = {fcx - 0.5f * fsize, fcy - 0.5f * fsize,
                                              fcx + 0.5f * fsize, fcy + 0.5f * fsize};
                                best_parts = parts;
                            }
                        }
                    }
                    derived_face[i]        = best_fb;
                    derived_feats[i]       = best_parts;
                    derived_matched_cnt[i] = best_hits;
                }
            });
        }
        for (auto& w : ws) w.join();
    }

    // =========================================================================
    // STEP 4: Score all 6 bottom-up derived child feature boxes + derived face
    //         box + full edge map on GPU against full K=16 + canonical templates!
    // =========================================================================
    std::array<std::vector<float>, kNumComps> margins;
    for (uint32_t comp = 0; comp < kNumComps; ++comp) {
        std::vector<uint8_t> patches(static_cast<size_t>(N_test) * kPatchPix);
        for (uint32_t i = 0; i < N_test; ++i) {
            BBox b = (comp == 0u) ? BBox{0, 0, float(samples[i].w), float(samples[i].h)}
                   : ((comp == 1u) ? derived_face[i] : derived_feats[i][comp - 2u]);
            gllm::extract_grad_edge_patch_28x28(samples[i].ge, b.x0, b.y0, b.x1, b.y1,
                                                true, 0.0f, &patches[static_cast<size_t>(i) * kPatchPix]);
        }
        uint32_t nc = comp_ncat[comp], nd = comp_ndog[comp];
        std::vector<float> s_all, s_can;
        gpu_score_ptr(patches, N_test, d_tmpls[comp], nc + nd, s_all, gpu_ms);
        gpu_score_ptr(patches, N_test, d_canon[comp], 2u, s_can, gpu_ms);
        margins[comp].resize(N_test);
        for (uint32_t i = 0; i < N_test; ++i) {
            float bc = -1e9f, bd = -1e9f;
            for (uint32_t k = 0; k < nc; ++k) bc = std::max(bc, s_all[static_cast<size_t>(i) * (nc + nd) + k]);
            for (uint32_t k = 0; k < nd; ++k) bd = std::max(bd, s_all[static_cast<size_t>(i) * (nc + nd) + nc + k]);
            float c_sc = 0.5f * s_can[i * 2u] + 0.5f * bc;
            float d_sc = 0.5f * s_can[i * 2u + 1] + 0.5f * bd;
            margins[comp][i] = c_sc - d_sc;
        }
    }
    double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_inv0).count();

    // Diagnostics & Full Report
    double sum_iou = 0.0;
    uint32_t iou30 = 0, iou50 = 0;
    uint64_t sum_hits = 0;
    for (uint32_t i = 0; i < N_test; ++i) {
        float iou = box_iou(derived_face[i], test_rows[i].face_box);
        sum_iou += iou;
        if (iou >= 0.30f) iou30++;
        if (iou >= 0.50f) iou50++;
        sum_hits += derived_matched_cnt[i];
    }

    auto print_row = [&](const char* name, const std::vector<float>& m) {
        uint32_t cc = 0, cd = 0, dc = 0, dd = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            int tr = test_rows[i].species, pr = (m[i] >= 0.0f) ? 0 : 1;
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
    auto compute_auc = [&](const std::vector<float>& m) -> double {
        std::vector<std::pair<float, int>> v(N_test);
        uint64_t nc = 0, nd = 0;
        for (uint32_t i = 0; i < N_test; ++i) {
            v[i] = {m[i], test_rows[i].species};
            if (test_rows[i].species == 0) nc++; else nd++;
        }
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        uint64_t db = 0, w = 0;
        for (const auto& p : v) { if (p.second == 1) db++; else w += db; }
        return (nc * nd > 0) ? (static_cast<double>(w) / static_cast<double>(nc * nd)) : 0.5;
    };

    std::vector<float> l3_5(N_test), l3_6(N_test), l32(N_test), l321(N_test), votes(N_test);
    for (uint32_t i = 0; i < N_test; ++i) {
        int cv = 0;
        float s5 = 0.0f, s6 = 0.0f;
        for (int f = 0; f < 6; ++f) {
            float mv = margins[2 + f][i];
            if (mv >= 0.0f) cv++;
            s6 += mv;
            if (f != 4) s5 += mv;
        }
        l3_5[i]  = s5;
        l3_6[i]  = s6;
        // Equal 1:1:1 hierarchy-level fusion: mean(L3 6 child features) + L2 derived face + L1 full edge map!
        l32[i]   = (s6 / 6.0f) + margins[1][i];
        l321[i]  = (s6 / 6.0f) + margins[1][i] + margins[0][i];
        votes[i] = (cv > 3) ? 1.0f : (cv < 3 ? -1.0f : s6);
    }

    std::cout << "\n==============================================================================================================\n";
    std::printf("  INVERTED CHILD-FIRST GRADIENT-DOMAIN BENCHMARK (%u strictly unseen RAW images | 0 test bboxes!)\n", N_test);
    std::cout << "==============================================================================================================\n";
    std::printf("  [DERIVED BOXES] Avg child edge hits/img: %.2f/6 | Face Mean IoU: %.3f | IoU>=0.30: %.1f%% (%u) | IoU>=0.50: %.1f%% (%u)\n",
                static_cast<double>(sum_hits) / N_test, sum_iou / N_test,
                100.0 * iou30 / N_test, iou30, 100.0 * iou50 / N_test, iou50);
    for (int f = 0; f < 6; ++f) {
        std::string nm = "Bottom-Up L3: " + kCompNames[2 + f];
        print_row(nm.c_str(), margins[2 + f]);
    }
    print_row("Bottom-Up L3: 6-Feature Vote Count",    votes);
    print_row("Bottom-Up L3: 5 Features (excl. nose)", l3_5);
    print_row("Bottom-Up L3: All 6 Features Fused",    l3_6);
    print_row("Derived L2: Face Edge Box Only",        margins[1]);
    print_row("Bottom-Up L3 -> L2 (Child + Face)",     l32);
    print_row("Raw L1: Full Gradient Edge Map Only",   margins[0]);
    print_row("INVERTED FULL (L3 -> L2 -> L1)",        l321);
    std::printf("  [ROC-AUC] L3-5: %.4f | L3-6: %.4f | L2 Face: %.4f | L1 Full: %.4f | Full Inverted L3->L2->L1: %.4f\n",
                compute_auc(l3_5), compute_auc(l3_6), compute_auc(margins[1]), compute_auc(margins[0]), compute_auc(l321));
    std::printf("  [SPEED]   GPU: %.1f ms (%.0f raw imgs/sec) | End-to-End 16-Core+GPU: %.1f ms (%.0f raw imgs/sec)\n",
                gpu_ms, N_test * 1000.0 / std::max(0.1f, gpu_ms), wall_ms, N_test * 1000.0 / std::max(0.1, wall_ms));

    // Export Side-by-Side Visual Verification Sheet: [Raw Photo + Derived Boxes | Gradient-Domain Edge Map + Derived Boxes]!
    if (!viz_out_path.empty() && !viz_ids.empty()) {
        constexpr int kCell = 240, kCols = 4, kRows = 4; // 8 samples x 2 columns (Photo | Edge Map) = 16 cells
        const int CW = kCell * kCols, CH = kCell * kRows;
        std::vector<uint8_t> canvas(static_cast<size_t>(CW) * CH * 3u, 12u);
        const uint8_t fcol[6][3] = {
            {60, 220, 255}, {255, 170, 50}, {140, 110, 255},
            {90, 255, 120}, {255, 90, 140}, {80, 190, 255}
        };
        for (size_t s = 0; s < std::min<size_t>(8, viz_ids.size()); ++s) {
            uint32_t i = viz_ids[s];
            const auto& smp = samples[i];
            if (smp.rgb.empty()) continue;
            for (int mode = 0; mode < 2; ++mode) { // 0 = raw photo, 1 = gradient-domain sparse edge map!
                size_t cell = s * 2u + mode;
                int gx = static_cast<int>(cell % kCols) * kCell;
                int gy = static_cast<int>(cell / kCols) * kCell;
                float sc = std::min(static_cast<float>(kCell - 8) / smp.w, static_cast<float>(kCell - 8) / smp.h);
                int dw = std::max(1, static_cast<int>(std::lround(smp.w * sc)));
                int dh = std::max(1, static_cast<int>(std::lround(smp.h * sc)));
                int ox = gx + (kCell - dw) / 2;
                int oy = gy + (kCell - dh) / 2;
                for (int y = 0; y < dh; ++y) {
                    int py = std::clamp(static_cast<int>(y / sc), 0, static_cast<int>(smp.h) - 1);
                    for (int x = 0; x < dw; ++x) {
                        int px = std::clamp(static_cast<int>(x / sc), 0, static_cast<int>(smp.w) - 1);
                        size_t di = (static_cast<size_t>(oy + y) * CW + (ox + x)) * 3u;
                        if (mode == 0) {
                            size_t si = (static_cast<size_t>(py) * smp.w + px) * 3u;
                            canvas[di] = smp.rgb[si]; canvas[di + 1] = smp.rgb[si + 1]; canvas[di + 2] = smp.rgb[si + 2];
                        } else {
                            uint8_t ev = smp.ge.edge_vis[static_cast<size_t>(py) * smp.w + px];
                            canvas[di] = ev; canvas[di + 1] = static_cast<uint8_t>(std::min(255, int(ev) + ev / 6)); canvas[di + 2] = 255u * (ev > 8 ? 1 : 0);
                            if (ev <= 8) { canvas[di] = 6; canvas[di + 1] = 10; canvas[di + 2] = 18; }
                        }
                    }
                }
                int pr = (l321[i] >= 0.0f) ? 0 : 1;
                bool ok = (pr == test_rows[i].species);
                draw_rect(canvas, CW, CH, gx + 1, gy + 1, gx + kCell - 2, gy + kCell - 2,
                          ok ? 40 : 240, ok ? 225 : 65, ok ? 95 : 65, 3);
                const auto& fb = derived_face[i];
                draw_rect(canvas, CW, CH, ox + int(fb.x0 * sc), oy + int(fb.y0 * sc),
                          ox + int(fb.x1 * sc), oy + int(fb.y1 * sc), 255, 235, 50, 2);
                for (int f = 0; f < 6; ++f) {
                    const auto& b = derived_feats[i][f];
                    draw_rect(canvas, CW, CH, ox + int(b.x0 * sc), oy + int(b.y0 * sc),
                              ox + int(b.x1 * sc), oy + int(b.y1 * sc), fcol[f][0], fcol[f][1], fcol[f][2], 2);
                }
            }
        }
        std::ofstream pf(viz_out_path, std::ios::binary);
        pf << "P6\n" << CW << " " << CH << "\n255\n";
        pf.write(reinterpret_cast<const char*>(canvas.data()), static_cast<std::streamsize>(canvas.size()));
        std::cout << "[VIZ]    Saved Photo | Gradient-Domain Edge Map side-by-side sheet to " << viz_out_path << "\n";
    }

    CU_CHECK(cuMemFree_v2(d_child18));
    for (uint32_t c = 0; c < kNumComps; ++c) { CU_CHECK(cuMemFree_v2(d_tmpls[c])); CU_CHECK(cuMemFree_v2(d_canon[c])); }
    CU_CHECK(cuEventDestroy_v2(ev_s)); CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod)); CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
