// =============================================================================
// gllm_eval — Standalone CUDA Shadow-Overlap Evaluator CLI
//
// Loads any trained .gllm geometric sphere model and evaluates an unseen test
// dataset (e.g., 10,000 MNIST test images or EMNIST letters) directly on the
// NVIDIA GPU using cosine-normalized geometric shadow overlap scoring.
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"

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
              << "  gllm_eval --model <path.gllm> --idx-images <test_img.gz> --idx-labels <test_lbl.gz>\n"
              << "            [--label-names ...] [--label-offset 0] [--transpose]\n"
              << "  gllm_eval --model <path.gllm> --folder <test_class_subfolders_dir>\n";
}

int main(int argc, char** argv) {
    std::string model_path, idx_images, idx_labels, folder_dir;
    std::vector<std::string> label_names;
    int label_offset = 0;
    int min_raw_label = -1;
    int max_raw_label = -1;
    bool transpose = false;

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
        else if (a == "--transpose") transpose = true;
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
    }

    if (model_path.empty() || (folder_dir.empty() && (idx_images.empty() || idx_labels.empty()))) {
        print_usage();
        return 1;
    }

    gllm::GeometricModel model;
    if (!model.load(model_path)) {
        std::cerr << "[ERROR] Cannot load model: " << model_path << "\n";
        return 1;
    }

    gllm::LabeledImageBatch ds = !folder_dir.empty()
        ? gllm::load_folder_dataset(folder_dir)
        : gllm::load_idx_dataset(idx_images, idx_labels, label_names, label_offset, transpose, min_raw_label, max_raw_label);

    // Collect all trained classes in the model that match the test image dimensions
    std::vector<size_t> eval_class_indices;
    for (size_t i = 0; i < model.classes.size(); ++i) {
        if (model.classes[i].width == ds.width && model.classes[i].height == ds.height &&
            model.classes[i].train_count > 0) {
            eval_class_indices.push_back(i);
        }
    }
    if (eval_class_indices.empty()) {
        std::cerr << "[ERROR] Model has no trained classes matching " << ds.width << "x" << ds.height << "\n";
        return 1;
    }

    uint32_t pix = ds.width * ds.height;
    uint32_t R   = static_cast<uint32_t>(eval_class_indices.size()); // total sub-shadow regions
    uint32_t U   = model.unique_class_count();

    // Build L2-normalized shadow templates for GPU cosine-similarity scoring across ALL sub-shadows
    std::vector<float> h_templates(static_cast<size_t>(R) * pix, 0.0f);
    for (uint32_t r = 0; r < R; ++r) {
        const auto& rec = model.classes[eval_class_indices[r]];
        double norm2 = 0.0;
        for (uint32_t p = 0; p < pix; ++p) {
            double v = static_cast<double>(rec.shadow_sum[p]) / static_cast<double>(rec.train_count);
            norm2 += v * v;
        }
        float inv_norm = static_cast<float>(1.0 / std::sqrt(norm2 + 1e-12));
        for (uint32_t p = 0; p < pix; ++p) {
            float avg = rec.shadow_sum[p] / static_cast<float>(rec.train_count);
            h_templates[static_cast<size_t>(r) * pix + p] = avg * inv_norm;
        }
    }

    // Run CUDA Evaluation Kernel on GPU (chunked over test images so N * R fits cleanly in VRAM!)
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

    uint32_t N = ds.count;
    uint32_t chunk_imgs = std::max<uint32_t>(1u, std::min<uint32_t>(N, 16384u));

    CUdeviceptr d_imgs = 0, d_tmpls = 0, d_scores = 0;
    CU_CHECK(cuMemAlloc_v2(&d_imgs,   static_cast<size_t>(chunk_imgs) * pix));
    CU_CHECK(cuMemAlloc_v2(&d_tmpls,  h_templates.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_tmpls, h_templates.data(), h_templates.size() * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_scores, static_cast<size_t>(chunk_imgs) * R * sizeof(float)));

    std::vector<float> h_scores(static_cast<size_t>(chunk_imgs) * R);
    float gpu_ms = 0.0f;
    uint32_t correct = 0;
    std::map<std::string, uint32_t> cls_total, cls_ok;

    for (uint32_t start = 0; start < N; start += chunk_imgs) {
        uint32_t batch = std::min(chunk_imgs, N - start);
        CU_CHECK(cuMemcpyHtoD_v2(d_imgs, &ds.pixels[static_cast<size_t>(start) * pix], static_cast<size_t>(batch) * pix));

        uint32_t total_pairs = batch * R;
        void* params[] = {&d_imgs, &d_tmpls, &d_scores, &pix, &R, &batch};

        CU_CHECK(cuEventRecord(ev_s, nullptr));
        CU_CHECK(cuLaunchKernel(fn_eval, (total_pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
        CU_CHECK(cuEventRecord(ev_e, nullptr));
        CU_CHECK(cuEventSynchronize(ev_e));
        float cms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&cms, ev_s, ev_e));
        gpu_ms += cms;

        CU_CHECK(cuMemcpyDtoH_v2(h_scores.data(), d_scores, static_cast<size_t>(batch) * R * sizeof(float)));

        // Max-pool over sub-shadow variants: winning sub-shadow votes for its class label!
        for (uint32_t i = 0; i < batch; ++i) {
            uint32_t best_r = 0;
            float best_s = -1e9f;
            for (uint32_t r = 0; r < R; ++r) {
                float s = h_scores[static_cast<size_t>(i) * R + r];
                if (s > best_s) { best_s = s; best_r = r; }
            }
            const std::string& pred = model.classes[eval_class_indices[best_r]].label;
            const std::string& gt   = ds.labels[start + i];
            cls_total[gt]++;
            if (pred == gt) {
                correct++;
                cls_ok[gt]++;
            }
        }
    }

    double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(N);
    std::printf("[EVAL]   Model: %s (%u unique classes / %u sub-shadow regions evaluated against %u test images)\n",
                model_path.c_str(), U, R, N);
    std::printf("[CUDA]   GPU Inference: %.3f ms on %s (%.0f imgs/sec)\n",
                gpu_ms, gpu_name, N / (gpu_ms / 1000.0));
    std::printf("[RESULT] Overall Pure Shadow-Overlap Accuracy: %u / %u (%.2f%%)\n",
                correct, N, acc);
    for (const auto& [lbl, tot] : cls_total) {
        uint32_t ok = cls_ok[lbl];
        std::printf("  Class '%-8s': %4u / %4u (%.1f%%)\n",
                    lbl.c_str(), ok, tot, 100.0 * ok / std::max(1u, tot));
    }

    CU_CHECK(cuMemFree_v2(d_imgs));
    CU_CHECK(cuMemFree_v2(d_tmpls));
    CU_CHECK(cuMemFree_v2(d_scores));
    CU_CHECK(cuEventDestroy_v2(ev_s));
    CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
