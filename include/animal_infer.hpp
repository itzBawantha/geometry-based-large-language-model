#pragma once
// =============================================================================
// animal_infer.hpp — Real-Time CUDA Inverted Child-First Gradient-Domain Engine
//
// Zero neural weights. Zero backpropagation. Pure Gradient-Domain Sparse Edge
// Maps (CMU 15-463 Lecture 9) + Inverted Child-First Bottom-Up Detection!
//
// Pipeline on ANY dropped/loaded image (.png, .jpg, .webp, .bmp):
//   1. Convert raw image -> Gradient-Domain Sparse Edge Map (log-luminance +
//      bilateral smooth + forward-difference gradient + adaptive sparsification).
//   2. Skip empty black flat regions in O(1); match ALL 6 lowest child feature
//      manifolds (ear_right, ear_left, eye_right, eye_left, nose, mouth) across
//      active sparse edge windows on CUDA!
//   3. Generalized Hough Geometric Constellation Voting: EVERY detected child
//      edge match votes backward for its implied parent face; the highest
//      cross-feature convergence CREATES the 6 verified child feature boxes +
//      enclosing parent face box strictly bottom-up!
//   4. Fuse equal-weight bottom-up hierarchy levels:
//        mean(L3 6 child features) + L2 derived face + L1 full edge map
//      and illuminate the matched manifold & outer-shell ray live in 3D!
// =============================================================================

#include "cuda_driver_min.hpp"
#include "gllm_model.hpp"
#include "grad_domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gllm {

struct AnimalBox {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

struct AnimalCompResult {
    std::string name;
    uint32_t    outer_idx = 0;
    float       cat_score = 0.0f;
    float       dog_score = 0.0f;
    float       margin    = 0.0f;
    bool        is_cat    = false;
    AnimalBox   box{};
};

struct AnimalInferResult {
    bool        valid           = false;
    bool        pred_is_cat     = false;
    bool        used_manual_box = false;
    std::string image_path;
    std::string truth_species;
    float       gpu_ms          = 0.0f;
    float       cat_prob        = 50.0f;
    float       dog_prob        = 50.0f;
    int         cat_votes       = 0;
    int         child_hits      = 0; // How many of 6 child manifolds converged bottom-up
    uint32_t    winner_outer_idx = 0;
    uint32_t    cat_outer_idx    = 0;
    uint32_t    dog_outer_idx    = 0;
    AnimalBox   face_box{};
    std::array<AnimalCompResult, 8> comps;
    std::array<float, 784> face_hog_preview{};
    uint32_t    img_w = 0, img_h = 0;
    std::vector<uint8_t> img_rgb;      // Raw color photo RGB24
    std::vector<uint8_t> edge_rgb;     // Glowing gradient-domain sparse edge map RGB24!
};

class AnimalInferenceEngine {
public:
    static constexpr uint32_t kNumComps = 8u;
    static constexpr uint32_t kPatchPix = 784u;

    bool ready = false;
    uint32_t cat_outer_idx = 0;
    uint32_t dog_outer_idx = 0;
    std::array<uint32_t, kNumComps> comp_outer_idx{};
    std::array<uint32_t, kNumComps> comp_ncat{}, comp_ndog{};
    std::array<CUdeviceptr, kNumComps> d_tmpls{};
    std::array<CUdeviceptr, kNumComps> d_canon{};
    CUdeviceptr d_child18 = 0;
    CUfunction fn_eval = nullptr;
    CUevent ev_s = nullptr, ev_e = nullptr;

    struct RelPrior { float cx, cy, w, h; };
    std::array<RelPrior, 6> feat_prior = {{
        {0.24f, 0.33f, 0.24f, 0.37f}, // ear_right
        {0.76f, 0.34f, 0.24f, 0.37f}, // ear_left
        {0.36f, 0.50f, 0.09f, 0.08f}, // eye_right
        {0.63f, 0.50f, 0.09f, 0.08f}, // eye_left
        {0.50f, 0.68f, 0.24f, 0.16f}, // nose
        {0.50f, 0.78f, 0.30f, 0.14f}  // mouth
    }};

    static const std::array<std::string, kNumComps>& comp_names() {
        static const std::array<std::string, kNumComps> kNames = {
            "full", "face", "ear_right", "ear_left", "eye_right", "eye_left", "nose", "mouth"
        };
        return kNames;
    }

    bool init(const GeometricModel& model, CUmodule ptx_mod) {
        release();
        const auto& names = comp_names();
        std::vector<uint32_t> regs[2][kNumComps];
        for (uint32_t r = 0; r < model.classes.size(); ++r) {
            const auto& c = model.classes[r];
            if (c.width != 28u || c.height != 28u) continue;
            for (int sp = 0; sp < 2; ++sp) {
                std::string pfx = (sp == 0) ? "cat_" : "dog_";
                for (uint32_t comp = 0; comp < kNumComps; ++comp) {
                    if (c.label == pfx + names[comp]) regs[sp][comp].push_back(r);
                }
            }
        }
        for (int sp = 0; sp < 2; ++sp) {
            for (uint32_t comp = 0; comp < kNumComps; ++comp) {
                if (regs[sp][comp].empty()) return false;
            }
        }
        if (cuModuleGetFunction(&fn_eval, ptx_mod, "eval_score_shadows") != CUDA_SUCCESS) return false;
        cuEventCreate(&ev_s, 0);
        cuEventCreate(&ev_e, 0);

        cat_outer_idx = model.classes[regs[0][0][0]].outer_idx;
        dog_outer_idx = model.classes[regs[1][0][0]].outer_idx;

        auto make_zncc = [](const std::vector<double>& raw, float* dst) {
            double mean = 0.0;
            for (uint32_t p = 0; p < kPatchPix; ++p) mean += raw[p];
            mean /= kPatchPix;
            double n2 = 0.0;
            for (uint32_t p = 0; p < kPatchPix; ++p) { double d = raw[p] - mean; n2 += d * d; }
            float inv = n2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
            for (uint32_t p = 0; p < kPatchPix; ++p) dst[p] = static_cast<float>(raw[p] - mean) * inv;
        };

        std::vector<float> h_child18(18u * kPatchPix, 0.0f);
        for (uint32_t comp = 0; comp < kNumComps; ++comp) {
            uint32_t nc = static_cast<uint32_t>(regs[0][comp].size());
            uint32_t nd = static_cast<uint32_t>(regs[1][comp].size());
            comp_ncat[comp] = nc; comp_ndog[comp] = nd;
            comp_outer_idx[comp] = model.classes[regs[0][comp][0]].outer_idx;

            std::vector<float> tmpls(static_cast<size_t>(nc + nd) * kPatchPix, 0.0f);
            std::vector<float> canon(2u * kPatchPix, 0.0f);
            std::vector<float> univ(kPatchPix, 0.0f);
            std::vector<double> sc(kPatchPix, 0.0), sd(kPatchPix, 0.0), sa(kPatchPix, 0.0);
            uint64_t tc = 0, td = 0;

            auto pack = [&](uint32_t slot, uint32_t r, int sp) {
                const auto& rec = model.classes[r];
                std::vector<double> rv(kPatchPix);
                for (uint32_t p = 0; p < kPatchPix; ++p) {
                    rv[p] = static_cast<double>(rec.shadow_sum[p]);
                    if (sp == 0) sc[p] += rv[p]; else sd[p] += rv[p];
                }
                if (sp == 0) tc += rec.train_count; else td += rec.train_count;
                make_zncc(rv, &tmpls[static_cast<size_t>(slot) * kPatchPix]);
            };
            for (uint32_t k = 0; k < nc; ++k) pack(k, regs[0][comp][k], 0);
            for (uint32_t k = 0; k < nd; ++k) pack(nc + k, regs[1][comp][k], 1);

            for (uint32_t p = 0; p < kPatchPix; ++p) {
                sc[p] /= std::max<uint64_t>(1, tc);
                sd[p] /= std::max<uint64_t>(1, td);
                sa[p]  = 0.5 * (sc[p] + sd[p]);
            }
            make_zncc(sc, &canon[0]);
            make_zncc(sd, &canon[kPatchPix]);
            make_zncc(sa, &univ[0]);

            if (comp >= 2u) {
                uint32_t f = comp - 2u;
                std::memcpy(&h_child18[static_cast<size_t>(f * 3 + 0) * kPatchPix], univ.data(), kPatchPix * sizeof(float));
                std::memcpy(&h_child18[static_cast<size_t>(f * 3 + 1) * kPatchPix], &canon[0],   kPatchPix * sizeof(float));
                std::memcpy(&h_child18[static_cast<size_t>(f * 3 + 2) * kPatchPix], &canon[kPatchPix], kPatchPix * sizeof(float));
            }
            CU_CHECK(cuMemAlloc_v2(&d_tmpls[comp], tmpls.size() * sizeof(float)));
            CU_CHECK(cuMemcpyHtoD_v2(d_tmpls[comp], tmpls.data(), tmpls.size() * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_canon[comp], canon.size() * sizeof(float)));
            CU_CHECK(cuMemcpyHtoD_v2(d_canon[comp], canon.data(), canon.size() * sizeof(float)));
        }
        CU_CHECK(cuMemAlloc_v2(&d_child18, h_child18.size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_child18, h_child18.data(), h_child18.size() * sizeof(float)));
        ready = true;
        return true;
    }

    void release() {
        if (d_child18) { cuMemFree_v2(d_child18); d_child18 = 0; }
        for (uint32_t c = 0; c < kNumComps; ++c) {
            if (d_tmpls[c]) { cuMemFree_v2(d_tmpls[c]); d_tmpls[c] = 0; }
            if (d_canon[c]) { cuMemFree_v2(d_canon[c]); d_canon[c] = 0; }
        }
        if (ev_s) { cuEventDestroy_v2(ev_s); ev_s = nullptr; }
        if (ev_e) { cuEventDestroy_v2(ev_e); ev_e = nullptr; }
        ready = false;
    }

    ~AnimalInferenceEngine() { release(); }

    static bool decode_any_image_rgb(const std::string& path, uint32_t& out_w, uint32_t& out_h, std::vector<uint8_t>& out_rgb) {
        std::string esc;
        esc.reserve(path.size() + 8);
        for (char c : path) { if (c == '\'') esc += "'\\''"; else esc += c; }
        std::string cmd = "ffmpeg -v error -i '" + esc + "' -vf format=rgb24 -f image2pipe -vcodec ppm -";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return false;
        std::vector<uint8_t> buf;
        uint8_t chunk[65536];
        size_t nr = 0;
        while ((nr = std::fread(chunk, 1, sizeof(chunk), pipe)) > 0) buf.insert(buf.end(), chunk, chunk + nr);
        if (pclose(pipe) != 0 || buf.size() < 12) return false;

        size_t pos = 0;
        auto skip_ws = [&]() {
            while (pos < buf.size()) {
                uint8_t c = buf[pos];
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { pos++; continue; }
                if (c == '#') { while (pos < buf.size() && buf[pos] != '\n') pos++; continue; }
                break;
            }
        };
        auto read_tok = [&]() -> std::string {
            skip_ws();
            size_t s = pos;
            while (pos < buf.size() && buf[pos] != ' ' && buf[pos] != '\t' && buf[pos] != '\r' && buf[pos] != '\n') pos++;
            return std::string(reinterpret_cast<const char*>(buf.data() + s), pos - s);
        };
        if (read_tok() != "P6") return false;
        out_w = static_cast<uint32_t>(std::stoul(read_tok()));
        out_h = static_cast<uint32_t>(std::stoul(read_tok()));
        uint32_t maxv = static_cast<uint32_t>(std::stoul(read_tok()));
        if (pos < buf.size()) pos++;
        size_t bytes = static_cast<size_t>(out_w) * out_h * 3u;
        if (out_w == 0 || out_h == 0 || maxv != 255 || pos + bytes > buf.size()) return false;
        out_rgb.assign(buf.begin() + pos, buf.begin() + pos + bytes);
        return true;
    }

    static float box_iou(const AnimalBox& a, const AnimalBox& b) {
        float ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
        float ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
        float inter = std::max(0.0f, ix1 - ix0) * std::max(0.0f, iy1 - iy0);
        float aa = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
        float ba = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
        return (aa + ba - inter > 1e-5f) ? (inter / (aa + ba - inter)) : 0.0f;
    }

    void gpu_score(const std::vector<uint8_t>& patches, uint32_t M,
                   CUdeviceptr d_t, uint32_t T,
                   std::vector<float>& out, float& gpu_ms) const {
        out.resize(static_cast<size_t>(M) * T);
        if (M == 0u) return;
        CUdeviceptr d_in = 0, d_sc = 0;
        CU_CHECK(cuMemAlloc_v2(&d_in, static_cast<size_t>(M) * kPatchPix));
        CU_CHECK(cuMemcpyHtoD_v2(d_in, patches.data(), static_cast<size_t>(M) * kPatchPix));
        CU_CHECK(cuMemAlloc_v2(&d_sc, static_cast<size_t>(M) * T * sizeof(float)));
        uint32_t pp = kPatchPix, pairs = M * T;
        void* args[] = {&d_in, &d_t, &d_sc, &pp, &T, &M};
        CU_CHECK(cuEventRecord(ev_s, nullptr));
        CU_CHECK(cuLaunchKernel(fn_eval, (pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        CU_CHECK(cuEventRecord(ev_e, nullptr));
        CU_CHECK(cuEventSynchronize(ev_e));
        float ms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
        gpu_ms += ms;
        CU_CHECK(cuMemcpyDtoH_v2(out.data(), d_sc, out.size() * sizeof(float)));
        CU_CHECK(cuMemFree_v2(d_in));
        CU_CHECK(cuMemFree_v2(d_sc));
    }

    // Run Inverted Child-First Gradient-Domain CUDA Inference on raw RGB image!
    AnimalInferResult infer_rgb(const std::vector<uint8_t>& rgb, uint32_t W, uint32_t H,
                                const std::string& path_label,
                                const AnimalBox* manual_face_box = nullptr,
                                const std::string& truth = "") const {
        AnimalInferResult res;
        if (!ready || W < 4u || H < 4u || rgb.size() < static_cast<size_t>(W) * H * 3u) return res;
        res.image_path    = path_label;
        res.truth_species = truth;
        res.cat_outer_idx = cat_outer_idx;
        res.dog_outer_idx = dog_outer_idx;
        res.img_w = W; res.img_h = H;
        res.img_rgb = rgb;

        // 1. Build Gradient-Domain Sparse Edge Map (CMU Lecture 9!)
        GradEdgeImage ge;
        build_grad_edge_image(rgb.data(), W, H, ge);
        res.edge_rgb.resize(static_cast<size_t>(W) * H * 3u);
        for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) {
            uint8_t ev = ge.edge_vis[i];
            if (ev <= 8u) {
                res.edge_rgb[i * 3 + 0] = 6; res.edge_rgb[i * 3 + 1] = 10; res.edge_rgb[i * 3 + 2] = 18;
            } else {
                res.edge_rgb[i * 3 + 0] = ev;
                res.edge_rgb[i * 3 + 1] = static_cast<uint8_t>(std::min(255, int(ev) + ev / 5));
                res.edge_rgb[i * 3 + 2] = 255;
            }
        }

        float gpu_ms = 0.0f;
        float fw_img = static_cast<float>(W), fh_img = static_cast<float>(H), mn_img = std::min(fw_img, fh_img);
        AnimalBox derived_face{0, 0, fw_img, fh_img};
        std::array<AnimalBox, 6> derived_parts{};
        int derived_hits = 0;

        if (manual_face_box && (manual_face_box->x1 - manual_face_box->x0) > 8.0f &&
            (manual_face_box->y1 - manual_face_box->y0) > 8.0f) {
            res.used_manual_box = true;
            derived_face = *manual_face_box;
            derived_hits = 6;
            float fw = derived_face.x1 - derived_face.x0, fh = derived_face.y1 - derived_face.y0;
            for (int f = 0; f < 6; ++f) {
                const auto& pr = feat_prior[f];
                float cx = derived_face.x0 + fw * pr.cx, cy = derived_face.y0 + fh * pr.cy;
                float bw = fw * pr.w, bh = fh * pr.h;
                derived_parts[f] = {cx - 0.5f * bw, cy - 0.5f * bh, cx + 0.5f * bw, cy + 0.5f * bh};
            }
        } else {
            // 2. INVERTED CHILD-FIRST SCAN: match ALL 6 lowest child feature manifolds over active sparse edges!
            std::vector<AnimalBox> wins;
            std::vector<uint8_t> patches;
            uint8_t buf[kPatchPix];
            const float scales[5] = {0.07f, 0.11f, 0.17f, 0.25f, 0.35f};
            for (float sc : scales) {
                float side = mn_img * sc;
                float step = std::max(8.0f, side * 0.42f);
                int nx = std::max(1, static_cast<int>(std::floor((fw_img - side) / step)) + 1);
                int ny = std::max(1, static_cast<int>(std::floor((fh_img - side) / step)) + 1);
                float rx = (nx > 1) ? (fw_img - side) / static_cast<float>(nx - 1) : 0.0f;
                float ry = (ny > 1) ? (fh_img - side) / static_cast<float>(ny - 1) : 0.0f;
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        float x0 = (nx > 1) ? ix * rx : 0.5f * (fw_img - side);
                        float y0 = (ny > 1) ? iy * ry : 0.5f * (fh_img - side);
                        if (!extract_grad_edge_patch_28x28(ge, x0, y0, x0 + side, y0 + side, true, 0.015f, buf))
                            continue;
                        wins.push_back({x0, y0, x0 + side, y0 + side});
                        patches.insert(patches.end(), buf, buf + kPatchPix);
                    }
                }
            }

            uint32_t M = static_cast<uint32_t>(wins.size());
            std::vector<float> c18;
            gpu_score(patches, M, d_child18, 18u, c18, gpu_ms);

            struct CDet { float cx, cy, bw, bh, univ_s; };
            std::array<std::vector<CDet>, 6> dets;
            for (int f = 0; f < 6; ++f) {
                std::vector<CDet> cand;
                cand.reserve(M);
                for (uint32_t wi = 0; wi < M; ++wi) {
                    const float* sc = &c18[static_cast<size_t>(wi) * 18u + f * 3u];
                    float u = std::max({sc[0], sc[1], sc[2]});
                    if (u <= 0.05f) continue;
                    const auto& b = wins[wi];
                    cand.push_back({0.5f * (b.x0 + b.x1), 0.5f * (b.y0 + b.y1), b.x1 - b.x0, b.y1 - b.y0, u});
                }
                std::sort(cand.begin(), cand.end(), [](const CDet& a, const CDet& b) { return a.univ_s > b.univ_s; });
                for (const auto& c : cand) {
                    bool dup = false;
                    AnimalBox cb{c.cx - 0.5f * c.bw, c.cy - 0.5f * c.bh, c.cx + 0.5f * c.bw, c.cy + 0.5f * c.bh};
                    for (const auto& k : dets[f]) {
                        AnimalBox kb{k.cx - 0.5f * k.bw, k.cy - 0.5f * k.bh, k.cx + 0.5f * k.bw, k.cy + 0.5f * k.bh};
                        if (box_iou(cb, kb) > 0.45f) { dup = true; break; }
                    }
                    if (!dup) { dets[f].push_back(c); if (dets[f].size() >= 36u) break; }
                }
            }

            // 3. Generalized Hough Geometric Constellation Voting: Child detections CREATE the bounding boxes!
            float best_cons = -1e9f;
            for (int f = 0; f < 6; ++f) {
                float rel_sz = std::sqrt(feat_prior[f].w * feat_prior[f].h);
                for (size_t di = 0; di < std::min<size_t>(18u, dets[f].size()); ++di) {
                    const auto& d = dets[f][di];
                    float fsize = std::clamp(std::sqrt(d.bw * d.bh) / rel_sz, 0.24f * mn_img, 1.05f * mn_img);
                    float fcx   = d.cx - (feat_prior[f].cx - 0.5f) * fsize;
                    float fcy   = d.cy - (feat_prior[f].cy - 0.5f) * fsize;
                    if (fcx < 0.0f || fcx > fw_img || fcy < 0.0f || fcy > fh_img) continue;

                    float cons = 0.0f;
                    int hits = 0;
                    std::array<AnimalBox, 6> parts{};
                    for (int k = 0; k < 6; ++k) {
                        float ex = fcx + (feat_prior[k].cx - 0.5f) * fsize;
                        float ey = fcy + (feat_prior[k].cy - 0.5f) * fsize;
                        float ew = feat_prior[k].w * fsize, eh = feat_prior[k].h * fsize;
                        float esz = std::sqrt(ew * eh);
                        parts[k] = {ex - 0.5f * ew, ey - 0.5f * eh, ex + 0.5f * ew, ey + 0.5f * eh};
                        float best_k = 0.0f;
                        const CDet* hd = nullptr;
                        for (const auto& dk : dets[k]) {
                            float dist = std::hypot(dk.cx - ex, dk.cy - ey);
                            if (dist > 0.22f * fsize) continue;
                            float ratio = std::sqrt(dk.bw * dk.bh) / std::max(1.0f, esz);
                            if (ratio < 0.45f || ratio > 2.25f) continue;
                            float gw = std::exp(-2.5f * (dist / (0.22f * fsize)) * (dist / (0.22f * fsize)));
                            float sc = dk.univ_s * gw;
                            if (sc > best_k) { best_k = sc; hd = &dk; }
                        }
                        if (hd) {
                            hits++;
                            cons += best_k;
                            parts[k] = {hd->cx - 0.5f * hd->bw, hd->cy - 0.5f * hd->bh,
                                        hd->cx + 0.5f * hd->bw, hd->cy + 0.5f * hd->bh};
                        }
                    }
                    if (cons > best_cons) {
                        best_cons = cons;
                        derived_hits = hits;
                        derived_face = {fcx - 0.5f * fsize, fcy - 0.5f * fsize, fcx + 0.5f * fsize, fcy + 0.5f * fsize};
                        derived_parts = parts;
                    }
                }
            }
        }

        // 4. Score all 8 hierarchy compartments (derived child boxes -> derived face -> full edge map) on CUDA!
        const auto& names = comp_names();
        res.face_box   = derived_face;
        res.child_hits = derived_hits;
        int cat_votes = 0;
        float sum6 = 0.0f;
        for (uint32_t comp = 0; comp < kNumComps; ++comp) {
            AnimalBox b = (comp == 0u) ? AnimalBox{0, 0, fw_img, fh_img}
                        : ((comp == 1u) ? derived_face : derived_parts[comp - 2u]);
            std::vector<uint8_t> p28(kPatchPix);
            extract_grad_edge_patch_28x28(ge, b.x0, b.y0, b.x1, b.y1, true, 0.0f, p28.data());
            uint32_t nc = comp_ncat[comp], nd = comp_ndog[comp];
            std::vector<float> s_all, s_can;
            gpu_score(p28, 1u, d_tmpls[comp], nc + nd, s_all, gpu_ms);
            gpu_score(p28, 1u, d_canon[comp], 2u, s_can, gpu_ms);
            float bc = -1e9f, bd = -1e9f;
            for (uint32_t k = 0; k < nc; ++k) bc = std::max(bc, s_all[k]);
            for (uint32_t k = 0; k < nd; ++k) bd = std::max(bd, s_all[nc + k]);
            float cs = 0.5f * s_can[0] + 0.5f * bc;
            float ds = 0.5f * s_can[1] + 0.5f * bd;
            float m  = cs - ds;
            res.comps[comp] = {names[comp], (comp == 0u) ? cat_outer_idx : comp_outer_idx[comp], cs, ds, m, m >= 0.0f, b};
            if (comp >= 2u) { if (m >= 0.0f) cat_votes++; sum6 += m; }
        }

        // Equal 1:1:1 hierarchy-level fusion: mean(L3 6 child features) + L2 derived face + L1 full edge map!
        float fused = (sum6 / 6.0f) + res.comps[1].margin + res.comps[0].margin;
        float p_cat = 100.0f / (1.0f + std::exp(-fused * 10.0f));
        res.cat_prob         = std::clamp(p_cat, 1.0f, 99.0f);
        res.dog_prob         = 100.0f - res.cat_prob;
        res.pred_is_cat      = (fused >= 0.0f);
        res.cat_votes        = cat_votes;
        res.winner_outer_idx = res.pred_is_cat ? cat_outer_idx : dog_outer_idx;
        res.gpu_ms           = gpu_ms;

        uint8_t vis28[784];
        extract_grad_edge_patch_28x28(ge, derived_face.x0, derived_face.y0, derived_face.x1, derived_face.y1,
                                      false, 0.0f, vis28);
        for (uint32_t p = 0; p < 784u; ++p) res.face_hog_preview[p] = static_cast<float>(vis28[p]) / 255.0f;
        res.valid = true;
        return res;
    }

    AnimalInferResult infer_file(const std::string& path, const std::string& truth = "") const {
        uint32_t w = 0, h = 0;
        std::vector<uint8_t> rgb;
        if (!decode_any_image_rgb(path, w, h, rgb)) return AnimalInferResult{};
        return infer_rgb(rgb, w, h, path, nullptr, truth);
    }
};

} // namespace gllm
