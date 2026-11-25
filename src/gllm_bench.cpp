// =============================================================================
// gllm_bench — Reproducible Full Holdout, Live CUDA Matcher & OOD Benchmark CLI
//
// Verifies:
//   1. Spatial Integrity: 0 overlaps across all allocated 3D sphere subspaces
//   2. Full GPU Cosine Shadow-Overlap Holdout (MNIST 10k + EMNIST ByClass 58.4k)
//      reporting Exact 62-Way Top-1, Top-3, Visually-Equiv, 10-Way Digits & Speed
//   3. Live 2-Stage CUDA Matcher (64-bit Bitmask + 9-Shift Symmetric Bernoulli x
//      Continuous Soft-IoU + <28% UNKNOWN gate) on stratified holdout + OOD shapes
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"
#include "../include/hud_canvas.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

static bool visual_equiv(const std::string& a, const std::string& b) {
    if (a == b) return true;
    if (a.size() == 1 && b.size() == 1) {
        char ca = a[0], cb = b[0];
        if (std::tolower(static_cast<unsigned char>(ca)) ==
            std::tolower(static_cast<unsigned char>(cb))) {
            return true;
        }
        auto grp = [](char c) -> int {
            if (c == '0' || c == 'O' || c == 'o') return 1;
            if (c == '1' || c == 'I' || c == 'l') return 2;
            return 0;
        };
        int ga = grp(ca), gb = grp(cb);
        if (ga != 0 && ga == gb) return true;
    }
    return false;
}

static void print_usage() {
    std::cout
        << "Usage:\n"
        << "  gllm_bench --model <model.gllm> [--mnist-dir <dir>] [--emnist-dir <dir>]\n"
        << "             [--live-per-class <N>] [--unknown-thresh <pct>]\n\n"
        << "Options:\n"
        << "  --model <path.gllm>     Trained GLLM checkpoint (required)\n"
        << "  --mnist-dir <dir>       Directory with MNIST t10k-*-ubyte.gz test files\n"
        << "  --emnist-dir <dir>      Directory with EMNIST ByClass test-*-ubyte.gz files\n"
        << "  --live-per-class <N>    Stratified samples/class for live CUDA matcher (default: 100)\n"
        << "  --unknown-thresh <pct>  Open-world UNKNOWN threshold % (default: 28.0)\n";
}

int main(int argc, char** argv) {
    std::string model_path, mnist_dir, emnist_dir;
    int live_per_class = 100;
    float unknown_thresh = 28.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--mnist-dir" && i + 1 < argc) mnist_dir = argv[++i];
        else if (a == "--emnist-dir" && i + 1 < argc) emnist_dir = argv[++i];
        else if (a == "--live-per-class" && i + 1 < argc) live_per_class = std::max(0, std::atoi(argv[++i]));
        else if (a == "--unknown-thresh" && i + 1 < argc) unknown_thresh = static_cast<float>(std::atof(argv[++i]));
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    if (model_path.empty()) {
        print_usage();
        return 1;
    }

    gllm::GeometricModel model;
    if (!model.load(model_path)) {
        std::cerr << "[ERROR] Cannot load model: " << model_path << "\n";
        return 1;
    }

    const uint32_t R = static_cast<uint32_t>(model.classes.size());
    if (R == 0) {
        std::cerr << "[ERROR] Model contains 0 allocated regions.\n";
        return 1;
    }
    const uint32_t P = model.classes[0].width * model.classes[0].height;

    // -------------------------------------------------------------------------
    // 1. Spatial Integrity Audit (Verify 0 Overlaps in 3D Sphere Lattice)
    // -------------------------------------------------------------------------
    uint32_t overlaps = 0;
    std::set<uint32_t> unique_outer_set;
    std::set<std::string> unique_label_set;
    for (uint32_t i = 0; i < R; ++i) {
        const auto& a = model.classes[i];
        unique_outer_set.insert(a.outer_idx);
        unique_label_set.insert(a.label);
        uint32_t ax1 = a.ox + a.width, ay1 = a.oy + a.height, az1 = a.oz + 1u;
        for (uint32_t j = i + 1; j < R; ++j) {
            const auto& b = model.classes[j];
            uint32_t bx1 = b.ox + b.width, by1 = b.oy + b.height, bz1 = b.oz + 1u;
            if (a.ox < bx1 && ax1 > b.ox && a.oy < by1 && ay1 > b.oy && a.oz < bz1 && az1 > b.oz) {
                overlaps++;
            }
        }
    }
    std::printf("[AUDIT]  Model: %s\n", model_path.c_str());
    std::printf("         Implicit Lattice: %u^3 (~%llu sphere pts) | Outer Shell: %u pts\n",
                model.lattice_grid,
                static_cast<unsigned long long>(model.approx_sphere_lattice_points()),
                model.outer_shell_pts);
    std::printf("         Unique Classes: %zu | Unique Outer Nodes: %zu | Sub-Shadow Regions: %u | Spatial Overlaps: %u (%s)\n",
                unique_label_set.size(), unique_outer_set.size(), R, overlaps,
                overlaps == 0 ? "PASS" : "FAIL");
    if (overlaps > 0) return 2;

    // Initialize CUDA Driver API & load PTX kernels
    CU_CHECK(cuInit(0));
    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    char dev_name[128] = {};
    CU_CHECK(cuDeviceGetName(dev_name, sizeof(dev_name), dev));
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod = gllm::load_gllm_ptx_module();
    CUfunction fn_eval = nullptr, fn_match = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_eval,  mod, "eval_score_shadows"));
    CU_CHECK(cuModuleGetFunction(&fn_match, mod, "match_sketch_pruned"));
    std::printf("[CUDA]   Device: %s\n", dev_name);

    // Build lookup maps & L2-normalized shadow templates
    std::vector<float> h_tmpl(static_cast<size_t>(R) * P, 0.0f);
    std::vector<uint32_t> unique_oids;
    std::map<uint32_t, std::string> oid_to_lbl;
    uint32_t max_oid = 0;
    for (uint32_t r = 0; r < R; ++r) {
        const auto& rec = model.classes[r];
        if (!oid_to_lbl.count(rec.outer_idx)) {
            oid_to_lbl[rec.outer_idx] = rec.label;
            unique_oids.push_back(rec.outer_idx);
            max_oid = std::max(max_oid, rec.outer_idx);
        }
        double n2 = 0.0;
        for (uint32_t p = 0; p < P; ++p) n2 += static_cast<double>(rec.shadow_sum[p]) * rec.shadow_sum[p];
        float inv = n2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
        for (uint32_t p = 0; p < P; ++p) h_tmpl[static_cast<size_t>(r) * P + p] = rec.shadow_sum[p] * inv;
    }

    // Load requested holdout datasets
    std::vector<std::string> digit_names = {"0","1","2","3","4","5","6","7","8","9"};
    std::vector<std::string> letter_names = {
        "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
        "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"
    };

    bool has_mnist = !mnist_dir.empty();
    bool has_emnist = !emnist_dir.empty();
    gllm::LabeledImageBatch ds_mnist, ds_emnist;
    if (has_mnist) {
        ds_mnist = gllm::load_idx_dataset(
            mnist_dir + "/t10k-images-idx3-ubyte.gz",
            mnist_dir + "/t10k-labels-idx1-ubyte.gz",
            digit_names, 0, false, 0, 9);
    }
    if (has_emnist) {
        ds_emnist = gllm::load_idx_dataset(
            emnist_dir + "/emnist-byclass-test-images-idx3-ubyte.gz",
            emnist_dir + "/emnist-byclass-test-labels-idx1-ubyte.gz",
            letter_names, 10, true, 10, 61);
    }

    // -------------------------------------------------------------------------
    // 2. Full Holdout GPU Evaluation (eval_score_shadows, max-pooled per class)
    // -------------------------------------------------------------------------
    const uint32_t CHUNK = 16384u;
    CUdeviceptr d_tmpl = 0, d_imgs = 0, d_scores = 0;
    CU_CHECK(cuMemAlloc_v2(&d_tmpl, h_tmpl.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_tmpl, h_tmpl.data(), h_tmpl.size() * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_imgs, static_cast<size_t>(CHUNK) * P));
    CU_CHECK(cuMemAlloc_v2(&d_scores, static_cast<size_t>(CHUNK) * R * sizeof(float)));

    uint64_t comb_n = 0, comb_top1 = 0, comb_top3 = 0, comb_equiv = 0;
    double comb_ms = 0.0;

    auto eval_full = [&](const char* title, const gllm::LabeledImageBatch& ds, bool report_10way) {
        uint32_t N = ds.count;
        uint32_t top1 = 0, top3 = 0, top1_equiv = 0, top1_10way = 0;
        std::map<std::string, std::pair<uint32_t, uint32_t>> per_cls;
        std::vector<float> h_sc(static_cast<size_t>(CHUNK) * R);
        std::vector<float> best_by_oid(max_oid + 1u);
        CUevent e0 = nullptr, e1 = nullptr;
        CU_CHECK(cuEventCreate(&e0, 0));
        CU_CHECK(cuEventCreate(&e1, 0));
        float total_ms = 0.0f;

        for (uint32_t off = 0; off < N; off += CHUNK) {
            uint32_t cnt = std::min(CHUNK, N - off);
            CU_CHECK(cuMemcpyHtoD_v2(d_imgs, &ds.pixels[static_cast<size_t>(off) * P], static_cast<size_t>(cnt) * P));
            uint32_t pp = P, nc = R, ni = cnt;
            void* args[] = {&d_imgs, &d_tmpl, &d_scores, &pp, &nc, &ni};
            CU_CHECK(cuEventRecord(e0, nullptr));
            CU_CHECK(cuLaunchKernel(fn_eval, (cnt * R + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
            CU_CHECK(cuEventRecord(e1, nullptr));
            CU_CHECK(cuEventSynchronize(e1));
            float ms = 0.0f;
            CU_CHECK(cuEventElapsedTime(&ms, e0, e1));
            total_ms += ms;
            CU_CHECK(cuMemcpyDtoH_v2(h_sc.data(), d_scores, static_cast<size_t>(cnt) * R * sizeof(float)));

            for (uint32_t i = 0; i < cnt; ++i) {
                const std::string& truth = ds.labels[off + i];
                const float* row = &h_sc[static_cast<size_t>(i) * R];
                std::fill(best_by_oid.begin(), best_by_oid.end(), -1e9f);
                for (uint32_t r = 0; r < R; ++r) {
                    uint32_t oid = model.classes[r].outer_idx;
                    if (row[r] > best_by_oid[oid]) best_by_oid[oid] = row[r];
                }
                std::vector<std::pair<float, uint32_t>> ranked;
                ranked.reserve(unique_oids.size());
                for (uint32_t oid : unique_oids) ranked.push_back({best_by_oid[oid], oid});
                size_t topk = std::min<size_t>(3, ranked.size());
                std::partial_sort(ranked.begin(), ranked.begin() + topk, ranked.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                const std::string& p1 = oid_to_lbl[ranked[0].second];
                per_cls[truth].second++;
                if (p1 == truth) { top1++; per_cls[truth].first++; }
                if (visual_equiv(p1, truth)) top1_equiv++;
                for (size_t k = 0; k < topk; ++k) {
                    if (oid_to_lbl[ranked[k].second] == truth) { top3++; break; }
                }
                if (report_10way) {
                    float b10 = -1e9f; uint32_t best_d = 0;
                    for (uint32_t d = 0; d < std::min<uint32_t>(10u, max_oid + 1u); ++d) {
                        if (oid_to_lbl.count(d) && best_by_oid[d] > b10) { b10 = best_by_oid[d]; best_d = d; }
                    }
                    if (oid_to_lbl[best_d] == truth) top1_10way++;
                }
            }
        }
        CU_CHECK(cuEventDestroy_v2(e0));
        CU_CHECK(cuEventDestroy_v2(e1));

        comb_n += N; comb_top1 += top1; comb_top3 += top3; comb_equiv += top1_equiv; comb_ms += total_ms;
        std::printf("\n[HOLDOUT] %s (%u unseen imgs | %.1f ms = %.0f imgs/s)\n",
                    title, N, total_ms, N / (total_ms / 1000.0));
        std::printf("          Exact Top-1:               %5u / %u (%.2f%%)\n", top1, N, 100.0 * top1 / N);
        std::printf("          Exact Top-3:               %5u / %u (%.2f%%)\n", top3, N, 100.0 * top3 / N);
        std::printf("          Visually-Equiv (case/0O/1Il): %5u / %u (%.2f%%)\n", top1_equiv, N, 100.0 * top1_equiv / N);
        if (report_10way) {
            std::printf("          Digits-Only (10-Way) Top-1: %5u / %u (%.2f%%)\n", top1_10way, N, 100.0 * top1_10way / N);
        }
        std::vector<std::pair<double, std::string>> cls_acc;
        for (const auto& kv : per_cls) {
            cls_acc.push_back({100.0 * kv.second.first / kv.second.second, kv.first});
        }
        std::sort(cls_acc.begin(), cls_acc.end());
        std::printf("          Hardest 5:   ");
        for (size_t k = 0; k < std::min<size_t>(5, cls_acc.size()); ++k)
            std::printf("'%s'(%.1f%%) ", cls_acc[k].second.c_str(), cls_acc[k].first);
        std::printf("\n          Strongest 5: ");
        for (size_t k = cls_acc.size() > 5 ? cls_acc.size() - 5 : 0; k < cls_acc.size(); ++k)
            std::printf("'%s'(%.1f%%) ", cls_acc[k].second.c_str(), cls_acc[k].first);
        std::printf("\n");
    };

    if (has_mnist)  eval_full("MNIST Test Digits (0-9)", ds_mnist, true);
    if (has_emnist) eval_full("EMNIST ByClass Test Letters (A-Z, a-z)", ds_emnist, false);
    if (has_mnist && has_emnist) {
        std::printf("\n[COMBINED HOLDOUT] %llu unseen images | %.1f ms (%.0f imgs/s)\n",
                    static_cast<unsigned long long>(comb_n), comb_ms, comb_n / (comb_ms / 1000.0));
        std::printf("          Exact Top-1:               %llu / %llu (%.2f%%)\n",
                    static_cast<unsigned long long>(comb_top1), static_cast<unsigned long long>(comb_n), 100.0 * comb_top1 / comb_n);
        std::printf("          Visually-Equiv Top-1:      %llu / %llu (%.2f%%)\n",
                    static_cast<unsigned long long>(comb_equiv), static_cast<unsigned long long>(comb_n), 100.0 * comb_equiv / comb_n);
        std::printf("          Exact Top-3:               %llu / %llu (%.2f%%)\n",
                    static_cast<unsigned long long>(comb_top3), static_cast<unsigned long long>(comb_n), 100.0 * comb_top3 / comb_n);
    }

    // -------------------------------------------------------------------------
    // 3. Live 2-Stage CUDA Matcher (28x28: 64-bit Bitmask + 9-Shift Bernoulli x Soft-IoU)
    // -------------------------------------------------------------------------
    if (P == 784u) {
        std::vector<uint64_t> h_masks(R, 0ULL);
        std::vector<float> h_lo(static_cast<size_t>(R) * P, 0.0f);
        std::vector<float> h_pr(static_cast<size_t>(R) * P, 0.0f);
        std::vector<float> h_bias(R, 0.0f);
        for (uint32_t r = 0; r < R; ++r) {
            const auto& rec = model.classes[r];
            float inv = rec.train_count ? (1.0f / static_cast<float>(rec.train_count)) : 0.0f;
            std::vector<float> avg(P);
            for (uint32_t p = 0; p < P; ++p) avg[p] = rec.shadow_sum[p] * inv;
            float pnorm[784] = {};
            gllm::canonicalize_28(avg.data(), pnorm, 22.0f, 0.22f);
            h_masks[r] = gllm::compute_coarse_mask64(pnorm, 0.22f);
            double bs = 0.0, lmax = 0.0;
            for (uint32_t p = 0; p < P; ++p) {
                h_pr[static_cast<size_t>(r) * P + p] = std::clamp(pnorm[p], 0.0f, 1.0f);
                double pr = std::clamp(static_cast<double>(pnorm[p]) * 0.90, 0.015, 0.96);
                h_lo[static_cast<size_t>(r) * P + p] = static_cast<float>(std::log(pr / (1.0 - pr)));
                bs   += std::log(1.0 - pr);
                lmax += pr * std::log(pr) + (1.0 - pr) * std::log(1.0 - pr);
            }
            h_bias[r] = static_cast<float>(bs - 0.5 * lmax);
        }

        const uint32_t kNumShifts = 9u;
        CUdeviceptr dm = 0, dlo = 0, dpr = 0, db = 0, dsk = 0, dent = 0, dov = 0, dst = 0;
        CU_CHECK(cuMemAlloc_v2(&dm, R * sizeof(uint64_t)));
        CU_CHECK(cuMemcpyHtoD_v2(dm, h_masks.data(), R * sizeof(uint64_t)));
        CU_CHECK(cuMemAlloc_v2(&dlo, h_lo.size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(dlo, h_lo.data(), h_lo.size() * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&dpr, h_pr.size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(dpr, h_pr.data(), h_pr.size() * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&db, R * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(db, h_bias.data(), R * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&dsk, static_cast<size_t>(kNumShifts) * P * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&dent, kNumShifts * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&dov, R * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&dst, R * sizeof(uint32_t)));

        CUevent me0 = nullptr, me1 = nullptr;
        CU_CHECK(cuEventCreate(&me0, 0));
        CU_CHECK(cuEventCreate(&me1, 0));

        std::vector<float> h_raw(static_cast<size_t>(kNumShifts) * P);
        std::vector<float> h_ent(kNumShifts), h_ov(R);
        std::vector<uint32_t> h_st(R);
        std::vector<float> best_by_oid(max_oid + 1u);
        float prune_ratio = 0.68f;
        float inv_t_ln2   = static_cast<float>(1.0 / (200.0 * std::log(2.0)));

        auto run_live = [&](const float* norm28, float& out_ms) {
            uint32_t s = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float* dp = &h_raw[static_cast<size_t>(s) * P];
                    std::fill(dp, dp + P, 0.0f);
                    for (int y = 0; y < 28; ++y) {
                        int sy = y - dy;
                        if (sy < 0 || sy >= 28) continue;
                        for (int x = 0; x < 28; ++x) {
                            int sx = x - dx;
                            if (sx >= 0 && sx < 28) dp[y * 28 + x] = norm28[sy * 28 + sx];
                        }
                    }
                    double e = 0.0;
                    for (uint32_t p = 0; p < P; ++p) {
                        double xc = std::clamp(static_cast<double>(dp[p]), 1e-5, 1.0 - 1e-5);
                        e += xc * std::log(xc) + (1.0 - xc) * std::log(1.0 - xc);
                    }
                    h_ent[s++] = static_cast<float>(0.5 * e);
                }
            }
            uint64_t qm = gllm::compute_coarse_mask64(norm28, 0.24f);
            CU_CHECK(cuMemcpyHtoD_v2(dsk, h_raw.data(), h_raw.size() * sizeof(float)));
            CU_CHECK(cuMemcpyHtoD_v2(dent, h_ent.data(), h_ent.size() * sizeof(float)));
            uint32_t pp = P, nc = R, ns = kNumShifts;
            void* args[] = {&dsk, &dent, &qm, &dm, &dlo, &dpr, &db, &dov, &dst, &pp, &nc, &ns, &prune_ratio, &inv_t_ln2};
            CU_CHECK(cuEventRecord(me0, nullptr));
            CU_CHECK(cuLaunchKernel(fn_match, (R + 63u) / 64u, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
            CU_CHECK(cuEventRecord(me1, nullptr));
            CU_CHECK(cuEventSynchronize(me1));
            CU_CHECK(cuEventElapsedTime(&out_ms, me0, me1));
            CU_CHECK(cuMemcpyDtoH_v2(h_ov.data(), dov, R * sizeof(float)));
            CU_CHECK(cuMemcpyDtoH_v2(h_st.data(), dst, R * sizeof(uint32_t)));

            std::fill(best_by_oid.begin(), best_by_oid.end(), -1.0f);
            for (uint32_t r = 0; r < R; ++r) {
                if (h_st[r]) {
                    uint32_t oid = model.classes[r].outer_idx;
                    if (h_ov[r] > best_by_oid[oid]) best_by_oid[oid] = h_ov[r];
                }
            }
            std::vector<std::pair<float, uint32_t>> ranked;
            for (uint32_t oid : unique_oids) {
                if (best_by_oid[oid] >= 0.0f) ranked.push_back({best_by_oid[oid], oid});
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            bool is_unk = ranked.empty() || (ranked[0].first * 100.0f < unknown_thresh);
            return std::make_pair(std::move(ranked), is_unk);
        };

        auto eval_live_ds = [&](const char* title, const gllm::LabeledImageBatch& ds, bool report_10way) {
            if (live_per_class <= 0) return;
            std::map<std::string, int> seen;
            uint32_t tot = 0, top1 = 0, top3 = 0, equiv = 0, t10 = 0, known_cnt = 0;
            double sum_ms = 0.0, sum_conf = 0.0;
            std::vector<float> raw(P), can(P);
            for (uint32_t i = 0; i < ds.count; ++i) {
                const std::string& truth = ds.labels[i];
                if (seen[truth] >= live_per_class) continue;
                seen[truth]++;
                for (uint32_t p = 0; p < P; ++p) raw[p] = ds.pixels[static_cast<size_t>(i) * P + p] / 255.0f;
                gllm::canonicalize_28(raw.data(), can.data(), 22.0f, 0.10f);
                float ms = 0.0f;
                auto res = run_live(can.data(), ms);
                tot++; sum_ms += ms;
                const auto& ranked = res.first;
                if (!ranked.empty()) {
                    sum_conf += ranked[0].first * 100.0;
                    const std::string& p1 = oid_to_lbl[ranked[0].second];
                    if (p1 == truth) top1++;
                    if (visual_equiv(p1, truth)) equiv++;
                    for (size_t k = 0; k < std::min<size_t>(3, ranked.size()); ++k) {
                        if (oid_to_lbl[ranked[k].second] == truth) { top3++; break; }
                    }
                    if (!res.second) known_cnt++;
                    if (report_10way) {
                        for (const auto& kv : ranked) {
                            if (kv.second < 10u) { if (oid_to_lbl[kv.second] == truth) t10++; break; }
                        }
                    }
                }
            }
            std::printf("\n[LIVE MATCHER: 9-Shift Bernoulli x Soft-IoU] %s (%u stratified imgs | avg %.3f ms/query)\n",
                        title, tot, sum_ms / std::max(1u, tot));
            std::printf("          Exact Top-1:               %4u / %u (%.2f%%)\n", top1, tot, 100.0 * top1 / std::max(1u, tot));
            std::printf("          Visually-Equiv Top-1:      %4u / %u (%.2f%%)\n", equiv, tot, 100.0 * equiv / std::max(1u, tot));
            std::printf("          Exact Top-3:               %4u / %u (%.2f%%)\n", top3, tot, 100.0 * top3 / std::max(1u, tot));
            if (report_10way) {
                std::printf("          Digits-Only (10-Way) Top-1: %4u / %u (%.2f%%)\n", t10, tot, 100.0 * t10 / std::max(1u, tot));
            }
            std::printf("          Accepted (>=%.1f%%):        %4u / %u (%.1f%%) | Avg Top Score: %.1f%%\n",
                        unknown_thresh, known_cnt, tot, 100.0 * known_cnt / std::max(1u, tot),
                        sum_conf / std::max(1u, tot));
        };

        if (has_mnist)  eval_live_ds("MNIST Digits", ds_mnist, true);
        if (has_emnist) eval_live_ds("EMNIST Letters", ds_emnist, false);

        // Open-World OOD Rejection Test (25 single dots + 25 short partial strokes on raw 224x224 canvas)
        gllm::SketchSidebarHUD hud;
        uint32_t dot_rej = 0, dash_rej = 0;
        double dot_max = 0.0, dash_max = 0.0;
        for (int k = 0; k < 25; ++k) {
            hud.clear();
            float x = 40.0f + (k % 5) * 35.0f, y = 40.0f + (k / 5) * 35.0f;
            hud.draw_stroke(x, y, x, y, 8.0f + (k % 3) * 2.5f);
            float ms = 0.0f;
            auto res = run_live(hud.norm28.data(), ms);
            if (res.second) dot_rej++;
            if (!res.first.empty()) dot_max = std::max(dot_max, static_cast<double>(res.first[0].first * 100.0));
        }
        for (int k = 0; k < 25; ++k) {
            hud.clear();
            float ang = k * 0.25f, len = 18.0f + (k % 5) * 7.0f;
            hud.draw_stroke(112.0f - len * std::cos(ang), 112.0f - len * std::sin(ang),
                            112.0f + len * std::cos(ang), 112.0f + len * std::sin(ang));
            float ms = 0.0f;
            auto res = run_live(hud.norm28.data(), ms);
            if (res.second) dash_rej++;
            if (!res.first.empty()) dash_max = std::max(dash_max, static_cast<double>(res.first[0].first * 100.0));
        }
        std::printf("\n[OPEN-WORLD OOD REJECTION] (< %.1f%% -> UNKNOWN)\n", unknown_thresh);
        std::printf("          Single Click Dots:         %2u / 25 rejected (%.1f%%) | Max Score: %.1f%%\n",
                    dot_rej, 100.0 * dot_rej / 25.0, dot_max);
        std::printf("          Short Partial Strokes:     %2u / 25 rejected (%.1f%%) | Max Score: %.1f%%\n",
                    dash_rej, 100.0 * dash_rej / 25.0, dash_max);

        CU_CHECK(cuEventDestroy_v2(me0));
        CU_CHECK(cuEventDestroy_v2(me1));
        CU_CHECK(cuMemFree_v2(dm));
        CU_CHECK(cuMemFree_v2(dlo));
        CU_CHECK(cuMemFree_v2(dpr));
        CU_CHECK(cuMemFree_v2(db));
        CU_CHECK(cuMemFree_v2(dsk));
        CU_CHECK(cuMemFree_v2(dent));
        CU_CHECK(cuMemFree_v2(dov));
        CU_CHECK(cuMemFree_v2(dst));
    }

    CU_CHECK(cuMemFree_v2(d_tmpl));
    CU_CHECK(cuMemFree_v2(d_imgs));
    CU_CHECK(cuMemFree_v2(d_scores));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
