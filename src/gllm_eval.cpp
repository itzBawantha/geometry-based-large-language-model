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
    bool transpose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--idx-images" && i + 1 < argc) idx_images = argv[++i];
        else if (a == "--idx-labels" && i + 1 < argc) idx_labels = argv[++i];
        else if (a == "--folder" && i + 1 < argc) folder_dir = argv[++i];
        else if (a == "--label-names" && i + 1 < argc) label_names = split_csv(argv[++i]);
        else if (a == "--label-offset" && i + 1 < argc) label_offset = std::atoi(argv[++i]);
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
        : gllm::load_idx_dataset(idx_images, idx_labels, label_names, label_offset, transpose);

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
    uint32_t C   = static_cast<uint32_t>(eval_class_indices.size());

    // Build L2-normalized shadow templates for GPU cosine-similarity scoring
    std::vector<float> h_templates(static_cast<size_t>(C) * pix, 0.0f);
    for (uint32_t c = 0; c < C; ++c) {
        const auto& rec = model.classes[eval_class_indices[c]];
        double norm2 = 0.0;
        for (uint32_t p = 0; p < pix; ++p) {
            double v = static_cast<double>(rec.shadow_sum[p]) / static_cast<double>(rec.train_count);
            norm2 += v * v;
        }
        float inv_norm = static_cast<float>(1.0 / std::sqrt(norm2 + 1e-12));
        for (uint32_t p = 0; p < pix; ++p) {
            float avg = rec.shadow_sum[p] / static_cast<float>(rec.train_count);
            h_templates[static_cast<size_t>(c) * pix + p] = avg * inv_norm;
        }
    }

    // Run CUDA Evaluation Kernel on RTX A3000
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

    CUdeviceptr d_imgs = 0, d_tmpls = 0, d_scores = 0;
    CU_CHECK(cuMemAlloc_v2(&d_imgs,   ds.pixels.size()));
    CU_CHECK(cuMemcpyHtoD_v2(d_imgs,  ds.pixels.data(), ds.pixels.size()));
    CU_CHECK(cuMemAlloc_v2(&d_tmpls,  h_templates.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_tmpls, h_templates.data(), h_templates.size() * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_scores, static_cast<size_t>(ds.count) * C * sizeof(float)));

    uint32_t N = ds.count;
    uint32_t total_pairs = N * C;
    void* params[] = {&d_imgs, &d_tmpls, &d_scores, &pix, &C, &N};

    CU_CHECK(cuEventRecord(ev_s, nullptr));
    CU_CHECK(cuLaunchKernel(fn_eval, (total_pairs + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
    CU_CHECK(cuEventRecord(ev_e, nullptr));
    CU_CHECK(cuEventSynchronize(ev_e));

    float gpu_ms = 0.0f;
    CU_CHECK(cuEventElapsedTime(&gpu_ms, ev_s, ev_e));

    std::vector<float> h_scores(static_cast<size_t>(N) * C);
    CU_CHECK(cuMemcpyDtoH_v2(h_scores.data(), d_scores, h_scores.size() * sizeof(float)));

    // Compute exact accuracy against ground-truth labels
    uint32_t correct = 0;
    std::map<std::string, uint32_t> cls_total, cls_ok;
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t best_c = 0;
        float best_s = -1e9f;
        for (uint32_t c = 0; c < C; ++c) {
            float s = h_scores[static_cast<size_t>(i) * C + c];
            if (s > best_s) { best_s = s; best_c = c; }
        }
        const std::string& pred = model.classes[eval_class_indices[best_c]].label;
        const std::string& gt   = ds.labels[i];
        cls_total[gt]++;
        if (pred == gt) {
            correct++;
            cls_ok[gt]++;
        }
    }

    double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(N);
    std::printf("[EVAL]   Model: %s (%u classes evaluated against %u test images)\n",
                model_path.c_str(), C, N);
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
