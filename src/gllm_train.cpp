// =============================================================================
// gllm_train — Standalone CUDA Multi-Shadow Geometric Trainer CLI
//
// Dynamically allocates unique, non-overlapping WxH spaces inside a massive
// implicit 3D sphere lattice (default 1241^3 -> ~1 billion lattice points!)
// and binds each class to an available neighboring outer-shell point on the
// 1.2M outer sphere.
//
// Universal Multi-Shadow Clustering (--clusters K):
//   When K > 1, runs GPU-accelerated Spherical K-Means++ inside each class to
//   separate distinct geometric variants (e.g., crossed vs plain '7', open vs
//   closed '4', or future erect vs floppy animal ears / slit vs round eyes)
//   into K guaranteed non-overlapping WxH lattice subspaces that all converge
//   to the class's single shared outer-shell point!
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_train --model <path.gllm> --idx-images <img.gz> --idx-labels <lbl.gz>\n"
              << "             [--grid 1241] [--clusters 1] [--cluster-iters 10]\n"
              << "             [--manifold handwritten/numbers] [--manifold-label handwritten_numbers]\n"
              << "             [--manifold-concept numbers] [--parent-manifold handwritten]\n"
              << "             [--parent-concept handwritten] [--parent-label handwritten_alphanumeric]\n"
              << "             [--envelope-box] [--anchor x,y,z]\n"
              << "             [--label-names 0,1,...,9] [--label-offset 0] [--transpose]\n"
              << "             [--min-raw-label N] [--max-raw-label M]\n"
              << "  gllm_train --model <path.gllm> --folder <class_subfolders_dir> [--grid 1241] [--clusters 1]\n"
              << "             [--manifold <name>] [--manifold-label <cuda_semantic_label>]\n";
}

// L2-normalize an 8-bit image into float vector; returns false if empty
static bool normalize_u8_image(const uint8_t* px, uint32_t pix, float* out) {
    double norm2 = 0.0;
    for (uint32_t p = 0; p < pix; ++p) {
        double v = static_cast<double>(px[p]) * (1.0 / 255.0);
        norm2 += v * v;
    }
    if (norm2 < 1e-8) {
        std::fill(out, out + pix, 0.0f);
        return false;
    }
    float inv = static_cast<float>(1.0 / std::sqrt(norm2));
    for (uint32_t p = 0; p < pix; ++p) {
        out[p] = static_cast<float>(px[p]) * (1.0f / 255.0f) * inv;
    }
    return true;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string idx_images, idx_labels, folder_dir;
    std::vector<std::string> label_names;
    int label_offset  = 0;
    int min_raw_label = -1;
    int max_raw_label = -1;
    bool transpose    = false;
    uint32_t grid_dim      = 1241; // ~1,000,722,000 implicit points inside the unit sphere
    uint32_t num_clusters  = 1;    // K=1 single averaged shadow; K>1 multi-shadow geometric clustering
    uint32_t cluster_iters = 10;
    std::string manifold_name   = "fundamental_text";
    std::string manifold_label  = "handwritten_letters_and_numbers";
    std::string manifold_concept;
    std::string parent_manifold_name;
    std::string parent_concept;
    std::string parent_label;
    bool envelope_box = false;
    bool has_anchor   = false;
    uint32_t anchor_x = 620, anchor_y = 620, anchor_z = 620;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--idx-images" && i + 1 < argc) idx_images = argv[++i];
        else if (a == "--idx-labels" && i + 1 < argc) idx_labels = argv[++i];
        else if (a == "--folder" && i + 1 < argc) folder_dir = argv[++i];
        else if (a == "--label-names" && i + 1 < argc) label_names = split_csv(argv[++i]);
        else if (a == "--label-offset" && i + 1 < argc) label_offset = std::atoi(argv[++i]);
        else if (a == "--min-raw-label" && i + 1 < argc) min_raw_label = std::atoi(argv[++i]);
        else if (a == "--max-raw-label" && i + 1 < argc) max_raw_label = std::atoi(argv[++i]);
        else if (a == "--grid" && i + 1 < argc) grid_dim = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--clusters" && i + 1 < argc) num_clusters = std::max<uint32_t>(1u, static_cast<uint32_t>(std::atoi(argv[++i])));
        else if (a == "--cluster-iters" && i + 1 < argc) cluster_iters = std::max<uint32_t>(1u, static_cast<uint32_t>(std::atoi(argv[++i])));
        else if (a == "--manifold" && i + 1 < argc) manifold_name = argv[++i];
        else if (a == "--manifold-label" && i + 1 < argc) manifold_label = argv[++i];
        else if (a == "--manifold-concept" && i + 1 < argc) manifold_concept = argv[++i];
        else if (a == "--parent-manifold" && i + 1 < argc) parent_manifold_name = argv[++i];
        else if (a == "--parent-concept" && i + 1 < argc) parent_concept = argv[++i];
        else if (a == "--parent-label" && i + 1 < argc) parent_label = argv[++i];
        else if (a == "--envelope-box") envelope_box = true;
        else if (a == "--anchor" && i + 1 < argc) {
            auto parts = split_csv(argv[++i]);
            if (parts.size() == 3) {
                anchor_x = static_cast<uint32_t>(std::atoi(parts[0].c_str()));
                anchor_y = static_cast<uint32_t>(std::atoi(parts[1].c_str()));
                anchor_z = static_cast<uint32_t>(std::atoi(parts[2].c_str()));
                has_anchor = true;
            }
        }
        else if (a == "--transpose") transpose = true;
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
    }

    if (model_path.empty() || (folder_dir.empty() && (idx_images.empty() || idx_labels.empty()))) {
        print_usage();
        return 1;
    }

    // 1. Load existing .gllm model for incremental non-overlapping training, or create new
    gllm::GeometricModel model;
    bool loaded_existing = model.load(model_path);
    if (loaded_existing) {
        std::cout << "[MODEL]  Loaded existing model: " << model_path
                  << " (" << model.unique_class_count() << " unique classes, "
                  << model.classes.size() << " sub-shadow regions, grid="
                  << model.lattice_grid << "^3 -> ~" << model.approx_sphere_lattice_points()
                  << " sphere lattice pts)\n";
    } else {
        model.lattice_grid = grid_dim;
        std::cout << "[MODEL]  Initializing new geometric sphere model: grid="
                  << model.lattice_grid << "^3 -> ~" << model.approx_sphere_lattice_points()
                  << " implicit lattice pts | outer shell=" << model.outer_shell_pts << " pts\n";
    }

    // 2. Load dataset (IDX .gz or folder-of-PGM/PPM)
    gllm::LabeledImageBatch ds = !folder_dir.empty()
        ? gllm::load_folder_dataset(folder_dir)
        : gllm::load_idx_dataset(idx_images, idx_labels, label_names, label_offset, transpose, min_raw_label, max_raw_label);

    if (ds.count == 0) {
        std::cerr << "[ERROR] Dataset contains 0 images.\n";
        return 1;
    }
    uint32_t pix_per_img = ds.width * ds.height;
    std::cout << "[DATA]   Loaded " << ds.count << " images (" << ds.width << "x" << ds.height
              << ", " << ds.unique_labels.size() << " classes | clusters/class=" << num_clusters << ")\n";

    // 3. Initialize CUDA Driver & Load PTX Kernels
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

    constexpr uint32_t kMaxBatch = 65536u;
    float total_gpu_ms = 0.0f;

    uint32_t parent_id = gllm::kInvalidId;
    uint32_t mid_lat = model.lattice_grid / 2u;
    if (!parent_manifold_name.empty()) {
        int pi = model.find_manifold_by_name(parent_manifold_name);
        if (pi >= 0) {
            parent_id = model.manifolds[pi].manifold_id;
        } else {
            uint32_t pid = 0u;
            for (const auto& m : model.manifolds) pid = std::max(pid, m.manifold_id + 1u);
            gllm::ManifoldRecord pm;
            pm.manifold_id        = pid;
            pm.parent_manifold_id = gllm::kInvalidId;
            pm.depth              = 0u;
            pm.envelope_mode      = envelope_box ? 1u : 0u;
            pm.name               = parent_manifold_name;
            pm.concept_label      = parent_concept.empty() ? parent_manifold_name : parent_concept;
            pm.semantic_label     = parent_label.empty() ? pm.concept_label : parent_label;
            pm.anchor_x           = mid_lat; pm.anchor_y = mid_lat; pm.anchor_z = mid_lat;
            pm.outer_idx          = gllm::kInvalidId; // Bound to outer shell AFTER classes so digits keep 0..9!
            model.manifolds.push_back(std::move(pm));
            parent_id = pid;
        }
    }

    // Clear manifold-only outer indices before allocating new class nodes so classes stay contiguous (0..61)
    // and manifold concept nodes are cleanly placed immediately after them (#62, #63, #64...)
    if (envelope_box || !manifold_concept.empty() || !parent_manifold_name.empty()) {
        for (auto& m : model.manifolds) m.outer_idx = gllm::kInvalidId;
    }

    uint32_t target_manifold_id = model.find_or_create_manifold(manifold_name, manifold_label);
    for (auto& m : model.manifolds) {
        if (m.manifold_id == target_manifold_id) {
            if (parent_id != gllm::kInvalidId) {
                m.parent_manifold_id = parent_id;
                if (const auto* pm = model.get_manifold(parent_id)) m.depth = pm->depth + 1u;
            }
            if (envelope_box) m.envelope_mode = 1u;
            if (!manifold_concept.empty()) m.concept_label = manifold_concept;
            if (has_anchor) { m.anchor_x = anchor_x; m.anchor_y = anchor_y; m.anchor_z = anchor_z; }
            break;
        }
    }
    std::cout << "[MANIFOLD] Active Operational Manifold #" << target_manifold_id
              << ": name='" << manifold_name << "' | CUDA semantic_label='" << manifold_label << "'\n";

    if (num_clusters <= 1u) {
        // =====================================================================
        // Mode A: Single Averaged Shadow per Class (100% backwards-compatible)
        // =====================================================================
        std::vector<int> local_to_model(ds.unique_labels.size(), -1);
        std::map<std::string, uint32_t> label_to_local;
        for (size_t i = 0; i < ds.unique_labels.size(); ++i) {
            const auto& lbl = ds.unique_labels[i];
            bool existed = (model.find_class(lbl) >= 0);
            int mid = model.register_or_get_class(lbl, ds.width, ds.height, target_manifold_id);
            local_to_model[i] = mid;
            label_to_local[lbl] = static_cast<uint32_t>(i);
            const auto& rec = model.classes[static_cast<size_t>(mid)];
            std::printf("  [%s] Class '%-8s' -> Lattice [%4u..%4u, %4u..%4u, z=%4u] <-> Outer #%u (%+.3f,%+.3f,%+.3f)\n",
                        existed ? "REUSE" : "ALLOC", rec.label.c_str(),
                        rec.ox, rec.ox + rec.width - 1, rec.oy, rec.oy + rec.height - 1, rec.oz,
                        rec.outer_idx, rec.outer_x, rec.outer_y, rec.outer_z);
        }

        uint32_t num_local_classes = static_cast<uint32_t>(ds.unique_labels.size());
        std::vector<uint32_t> h_class_ids(ds.count);
        for (uint32_t i = 0; i < ds.count; ++i) h_class_ids[i] = label_to_local[ds.labels[i]];

        CUdeviceptr d_imgs = 0, d_cids = 0, d_sum = 0, d_cnt = 0;
        CU_CHECK(cuMemAlloc_v2(&d_imgs, ds.pixels.size()));
        CU_CHECK(cuMemcpyHtoD_v2(d_imgs, ds.pixels.data(), ds.pixels.size()));
        CU_CHECK(cuMemAlloc_v2(&d_cids, h_class_ids.size() * sizeof(uint32_t)));
        CU_CHECK(cuMemcpyHtoD_v2(d_cids, h_class_ids.data(), h_class_ids.size() * sizeof(uint32_t)));
        CU_CHECK(cuMemAlloc_v2(&d_sum,  static_cast<size_t>(num_local_classes) * pix_per_img * sizeof(float)));
        CU_CHECK(cuMemAlloc_v2(&d_cnt,  static_cast<size_t>(num_local_classes) * sizeof(uint32_t)));
        CU_CHECK(cuMemsetD32_v2(d_sum, 0u, static_cast<size_t>(num_local_classes) * pix_per_img));
        CU_CHECK(cuMemsetD32_v2(d_cnt, 0u, num_local_classes));

        for (uint32_t start = 0; start < ds.count; start += kMaxBatch) {
            uint32_t batch = std::min(kMaxBatch, ds.count - start);
            uint32_t total_work = batch * pix_per_img;
            void* params[] = {&d_imgs, &d_cids, &d_sum, &d_cnt, &pix_per_img, &num_local_classes, &start, &batch};
            CU_CHECK(cuEventRecord(ev_s, nullptr));
            CU_CHECK(cuLaunchKernel(fn_train, (total_work + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
            CU_CHECK(cuEventRecord(ev_e, nullptr));
            CU_CHECK(cuEventSynchronize(ev_e));
            float ms = 0.0f;
            CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));
            total_gpu_ms += ms;
        }

        std::vector<float> h_sum(static_cast<size_t>(num_local_classes) * pix_per_img);
        std::vector<uint32_t> h_cnt(num_local_classes);
        CU_CHECK(cuMemcpyDtoH_v2(h_sum.data(), d_sum, h_sum.size() * sizeof(float)));
        CU_CHECK(cuMemcpyDtoH_v2(h_cnt.data(), d_cnt, h_cnt.size() * sizeof(uint32_t)));

        for (uint32_t i = 0; i < num_local_classes; ++i) {
            auto& rec = model.classes[static_cast<size_t>(local_to_model[i])];
            rec.train_count += h_cnt[i];
            float peak = 0.0f;
            uint32_t active = 0;
            for (uint32_t p = 0; p < pix_per_img; ++p) {
                rec.shadow_sum[p] += h_sum[static_cast<size_t>(i) * pix_per_img + p];
                float avg = rec.train_count > 0 ? rec.shadow_sum[p] / static_cast<float>(rec.train_count) : 0.0f;
                if (avg > 0.004f) active++;
                if (avg > peak) peak = avg;
            }
            std::printf("  Class '%-8s': total=%6u shadows | active=%4u/%u pts | peak=%.1f%%\n",
                        rec.label.c_str(), rec.train_count, active, pix_per_img, peak * 100.0f);
        }

        CU_CHECK(cuMemFree_v2(d_imgs));
        CU_CHECK(cuMemFree_v2(d_cids));
        CU_CHECK(cuMemFree_v2(d_sum));
        CU_CHECK(cuMemFree_v2(d_cnt));
    } else {
        // =====================================================================
        // Mode B: Universal GPU Spherical K-Means++ Multi-Shadow Clustering!
        //   Separates each class into K distinct geometric structural variants,
        //   each allocated its own non-overlapping WxH lattice region converging
        //   to the class's single shared outer-shell point!
        // =====================================================================
        std::map<std::string, std::vector<uint32_t>> cls_indices;
        for (uint32_t i = 0; i < ds.count; ++i) cls_indices[ds.labels[i]].push_back(i);

        for (const auto& lbl : ds.unique_labels) {
            const auto& idxs = cls_indices[lbl];
            uint32_t M = static_cast<uint32_t>(idxs.size());
            if (M == 0) continue;
            uint32_t K = std::min<uint32_t>(num_clusters, std::max<uint32_t>(1u, M / 8u));
            if (M < num_clusters) K = M;

            // Pack contiguous class pixel buffer
            std::vector<uint8_t> cls_px(static_cast<size_t>(M) * pix_per_img);
            for (uint32_t i = 0; i < M; ++i) {
                const uint8_t* src = &ds.pixels[static_cast<size_t>(idxs[i]) * pix_per_img];
                std::memcpy(&cls_px[static_cast<size_t>(i) * pix_per_img], src, pix_per_img);
            }

            // Deterministic Farthest-Point Spherical Seeding (on up to 2048 strided samples)
            uint32_t pool_n = std::min<uint32_t>(M, 2048u);
            std::vector<uint32_t> pool_ids(pool_n);
            std::vector<float> pool_vecs(static_cast<size_t>(pool_n) * pix_per_img);
            for (uint32_t s = 0; s < pool_n; ++s) {
                uint32_t idx = static_cast<uint32_t>((static_cast<uint64_t>(s) * M) / pool_n);
                pool_ids[s] = idx;
                normalize_u8_image(&cls_px[static_cast<size_t>(idx) * pix_per_img], pix_per_img,
                                   &pool_vecs[static_cast<size_t>(s) * pix_per_img]);
            }

            std::vector<float> h_centroids(static_cast<size_t>(K) * pix_per_img, 0.0f);
            // Seed 0: first non-empty image in pool
            uint32_t first_s = 0;
            for (uint32_t s = 0; s < pool_n; ++s) {
                float e = 0.0f;
                for (uint32_t p = 0; p < pix_per_img; ++p) e += pool_vecs[static_cast<size_t>(s) * pix_per_img + p];
                if (e > 0.1f) { first_s = s; break; }
            }
            std::memcpy(&h_centroids[0], &pool_vecs[static_cast<size_t>(first_s) * pix_per_img], pix_per_img * sizeof(float));

            std::vector<float> max_sim(pool_n, -1.0f);
            for (uint32_t c = 1; c < K; ++c) {
                const float* prev_c = &h_centroids[static_cast<size_t>(c - 1) * pix_per_img];
                uint32_t far_s = 0;
                float min_max_sim = 1e9f;
                for (uint32_t s = 0; s < pool_n; ++s) {
                    const float* v = &pool_vecs[static_cast<size_t>(s) * pix_per_img];
                    float dot = 0.0f;
                    for (uint32_t p = 0; p < pix_per_img; ++p) dot += v[p] * prev_c[p];
                    if (dot > max_sim[s]) max_sim[s] = dot;
                    if (max_sim[s] < min_max_sim) {
                        min_max_sim = max_sim[s];
                        far_s = s;
                    }
                }
                std::memcpy(&h_centroids[static_cast<size_t>(c) * pix_per_img],
                            &pool_vecs[static_cast<size_t>(far_s) * pix_per_img], pix_per_img * sizeof(float));
            }

            // Upload class pixels & run GPU Spherical K-Means!
            CUdeviceptr d_cimgs = 0, d_cents = 0, d_scores = 0, d_assign = 0, d_sum = 0, d_cnt = 0;
            CU_CHECK(cuMemAlloc_v2(&d_cimgs,  cls_px.size()));
            CU_CHECK(cuMemcpyHtoD_v2(d_cimgs, cls_px.data(), cls_px.size()));
            CU_CHECK(cuMemAlloc_v2(&d_cents,  static_cast<size_t>(K) * pix_per_img * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_scores, static_cast<size_t>(M) * K * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_assign, static_cast<size_t>(M) * sizeof(uint32_t)));
            CU_CHECK(cuMemAlloc_v2(&d_sum,    static_cast<size_t>(K) * pix_per_img * sizeof(float)));
            CU_CHECK(cuMemAlloc_v2(&d_cnt,    static_cast<size_t>(K) * sizeof(uint32_t)));

            std::vector<float>    h_scores(static_cast<size_t>(M) * K);
            std::vector<uint32_t> h_assign(M, 0u);
            std::vector<float>    h_sum(static_cast<size_t>(K) * pix_per_img, 0.0f);
            std::vector<uint32_t> h_cnt(K, 0u);

            for (uint32_t it = 0; it < cluster_iters; ++it) {
                CU_CHECK(cuMemcpyHtoD_v2(d_cents, h_centroids.data(), h_centroids.size() * sizeof(float)));
                uint32_t total_pairs = M * K;
                void* e_params[] = {&d_cimgs, &d_cents, &d_scores, &pix_per_img, &K, &M};
                CU_CHECK(cuEventRecord(ev_s, nullptr));
                CU_CHECK(cuLaunchKernel(fn_eval, (total_pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, e_params, nullptr));
                CU_CHECK(cuEventRecord(ev_e, nullptr));
                CU_CHECK(cuEventSynchronize(ev_e));
                float ms1 = 0.0f;
                CU_CHECK(cuEventElapsedTime(&ms1, ev_s, ev_e));
                total_gpu_ms += ms1;

                CU_CHECK(cuMemcpyDtoH_v2(h_scores.data(), d_scores, h_scores.size() * sizeof(float)));
                uint32_t worst_img = 0;
                float worst_best = 1e9f;
                for (uint32_t i = 0; i < M; ++i) {
                    uint32_t best_k = 0;
                    float best_v = -1e9f;
                    for (uint32_t k = 0; k < K; ++k) {
                        float s = h_scores[static_cast<size_t>(i) * K + k];
                        if (s > best_v) { best_v = s; best_k = k; }
                    }
                    h_assign[i] = best_k;
                    if (best_v < worst_best) { worst_best = best_v; worst_img = i; }
                }

                CU_CHECK(cuMemcpyHtoD_v2(d_assign, h_assign.data(), h_assign.size() * sizeof(uint32_t)));
                CU_CHECK(cuMemsetD32_v2(d_sum, 0u, static_cast<size_t>(K) * pix_per_img));
                CU_CHECK(cuMemsetD32_v2(d_cnt, 0u, K));
                for (uint32_t start = 0; start < M; start += kMaxBatch) {
                    uint32_t batch = std::min(kMaxBatch, M - start);
                    uint32_t tw = batch * pix_per_img;
                    void* t_params[] = {&d_cimgs, &d_assign, &d_sum, &d_cnt, &pix_per_img, &K, &start, &batch};
                    CU_CHECK(cuEventRecord(ev_s, nullptr));
                    CU_CHECK(cuLaunchKernel(fn_train, (tw + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, t_params, nullptr));
                    CU_CHECK(cuEventRecord(ev_e, nullptr));
                    CU_CHECK(cuEventSynchronize(ev_e));
                    float ms2 = 0.0f;
                    CU_CHECK(cuEventElapsedTime(&ms2, ev_s, ev_e));
                    total_gpu_ms += ms2;
                }

                CU_CHECK(cuMemcpyDtoH_v2(h_sum.data(), d_sum, h_sum.size() * sizeof(float)));
                CU_CHECK(cuMemcpyDtoH_v2(h_cnt.data(), d_cnt, h_cnt.size() * sizeof(uint32_t)));

                // Update L2-normalized centroids for next iteration
                for (uint32_t k = 0; k < K; ++k) {
                    float* dst = &h_centroids[static_cast<size_t>(k) * pix_per_img];
                    if (h_cnt[k] == 0u) {
                        normalize_u8_image(&cls_px[static_cast<size_t>(worst_img) * pix_per_img], pix_per_img, dst);
                        continue;
                    }
                    double n2 = 0.0;
                    const float* sm = &h_sum[static_cast<size_t>(k) * pix_per_img];
                    for (uint32_t p = 0; p < pix_per_img; ++p) n2 += static_cast<double>(sm[p]) * sm[p];
                    float inv_n = static_cast<float>(1.0 / std::sqrt(n2 + 1e-12));
                    for (uint32_t p = 0; p < pix_per_img; ++p) dst[p] = sm[p] * inv_n;
                }
            }

            // Allocate one guaranteed non-overlapping 3D lattice space per non-empty cluster
            // All clusters of this class share ONE outer-shell point!
            uint32_t placed_k = 0;
            float avg_peak = 0.0f;
            uint32_t shared_outer = 0;
            for (uint32_t k = 0; k < K; ++k) {
                if (h_cnt[k] == 0u) continue;
                int mid = model.allocate_subshadow(lbl, ds.width, ds.height, target_manifold_id);
                auto& rec = model.classes[static_cast<size_t>(mid)];
                rec.train_count = h_cnt[k];
                float pk = 0.0f;
                for (uint32_t p = 0; p < pix_per_img; ++p) {
                    rec.shadow_sum[p] = h_sum[static_cast<size_t>(k) * pix_per_img + p];
                    pk = std::max(pk, rec.shadow_sum[p] / static_cast<float>(rec.train_count));
                }
                avg_peak += pk;
                shared_outer = rec.outer_idx;
                placed_k++;
            }
            if (placed_k > 0) avg_peak /= static_cast<float>(placed_k);
            std::printf("  [CLUST] Class '%-6s' (%6u imgs) -> %2u non-overlapping sub-shadows <-> Outer #%-3u | avg peak=%.1f%%\n",
                        lbl.c_str(), M, placed_k, shared_outer, avg_peak * 100.0f);

            CU_CHECK(cuMemFree_v2(d_cimgs));
            CU_CHECK(cuMemFree_v2(d_cents));
            CU_CHECK(cuMemFree_v2(d_scores));
            CU_CHECK(cuMemFree_v2(d_assign));
            CU_CHECK(cuMemFree_v2(d_sum));
            CU_CHECK(cuMemFree_v2(d_cnt));
        }
    }

    // Bind hierarchical 3D-box manifolds to additional outer-shell concept nodes (placed AFTER class indices!)
    // So existing digit/letter nodes (0..61) stay untouched while e.g. handwritten/numbers/letters get #62, #63, #64.
    if (envelope_box || !manifold_concept.empty() || !parent_manifold_name.empty()) {
        // Place hierarchical manifold concept nodes cleanly along a raised upper crown arc above the character cap
        auto place_manifold_crown = [](uint32_t rank, float& ox, float& oy, float& oz) {
            float tx = 0.0f, ty = 0.93f;
            if (rank == 0u)      { tx =  0.00f; ty = 0.94f; } // root (e.g. handwritten)
            else if (rank == 1u) { tx = -0.34f; ty = 0.87f; } // left child (e.g. numbers)
            else if (rank == 2u) { tx =  0.34f; ty = 0.87f; } // right child (e.g. letters)
            else {
                constexpr float kGolden = 2.39996322972865332f;
                tx = 0.38f * std::cos(rank * kGolden);
                ty = 0.85f - 0.08f * std::sin(rank * kGolden);
            }
            float xy2 = tx * tx + ty * ty;
            if (xy2 > 0.96f) { float s = std::sqrt(0.96f / xy2); tx *= s; ty *= s; xy2 = tx * tx + ty * ty; }
            ox = tx; oy = ty; oz = std::sqrt(std::max(0.0f, 1.0f - xy2));
        };
        // Re-anchor any already-bound manifold outer indices if new classes were appended after them
        uint32_t max_cls_outer = 0u;
        bool any_cls = false;
        for (const auto& c : model.classes) {
            if (!any_cls || c.outer_idx > max_cls_outer) max_cls_outer = c.outer_idx;
            any_cls = true;
        }
        uint32_t next_man_outer = any_cls ? (max_cls_outer + 1u) : 0u;
        uint32_t rank = 0u;
        std::map<std::string, uint32_t> concept_to_oid;
        for (auto& m : model.manifolds) {
            if (m.envelope_mode != 1u && m.concept_label.empty()) continue;
            std::string cw = m.concept_label.empty() ? m.name : m.concept_label;
            m.concept_label = cw;
            auto it = concept_to_oid.find(cw);
            if (it != concept_to_oid.end()) {
                m.outer_idx = it->second;
                for (const auto& prev : model.manifolds) {
                    if (prev.outer_idx == m.outer_idx) {
                        m.outer_x = prev.outer_x; m.outer_y = prev.outer_y; m.outer_z = prev.outer_z;
                        break;
                    }
                }
            } else {
                m.outer_idx = next_man_outer++;
                concept_to_oid[cw] = m.outer_idx;
                place_manifold_crown(rank++, m.outer_x, m.outer_y, m.outer_z);
            }
        }
    }

    // Recompute exact tight-fitting Operational Manifold envelopes before saving
    model.recompute_manifold_envelopes();
    for (const auto& m : model.manifolds) {
        std::printf("[MANIFOLD] #%u '%s' (CUDA: '%s', outer=#%s '%s', %s) | tight lattice [%u..%u, %u..%u, %u..%u] | maxR=%.4f | vol=%.4f%% of 1B sphere\n",
                    m.manifold_id, m.name.c_str(), m.semantic_label.c_str(),
                    (m.outer_idx == gllm::kInvalidId ? "-" : std::to_string(m.outer_idx).c_str()),
                    m.concept_label.c_str(),
                    (m.envelope_mode == 1u ? "3D-Box" : "Deformed"),
                    m.min_lx, m.max_lx, m.min_ly, m.max_ly, m.min_lz, m.max_lz,
                    m.max_radius, m.volume_fraction * 100.0f);
    }

    std::cout << "\n[CUDA]   Total GPU kernel time on " << gpu_name << ": " << total_gpu_ms << " ms\n";

    if (!model.save(model_path)) {
        std::cerr << "[ERROR] Failed to save model to " << model_path << "\n";
        return 1;
    }
    std::cout << "[SAVED]  Model saved to " << model_path
              << " (unique classes: " << model.unique_class_count()
              << ", total non-overlapping sub-shadow regions: " << model.classes.size() << ")\n";

    CU_CHECK(cuEventDestroy_v2(ev_s));
    CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
