// =============================================================================
// gllm_train — Standalone CUDA Shadow Trainer CLI
//
// Dynamically allocates unique, non-overlapping WxH spaces inside a massive
// implicit 3D sphere lattice (default 1241^3 -> ~1 billion lattice points!)
// and binds each class to an available neighboring outer-shell point on the
// 1.2M outer sphere. Supports incremental multi-dataset training: run once on
// MNIST digits (0..9), then run again on EMNIST letters (A..Z) or custom image
// folders into the SAME .gllm model file without ever overlapping prior classes!
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
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
              << "             [--grid 1241] [--label-names 0,1,...,9] [--label-offset 0] [--transpose]\n"
              << "  gllm_train --model <path.gllm> --folder <class_subfolders_dir> [--grid 1241]\n";
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string idx_images, idx_labels, folder_dir;
    std::vector<std::string> label_names;
    int label_offset = 0;
    int min_raw_label = -1;
    int max_raw_label = -1;
    bool transpose = false;
    uint32_t grid_dim = 1241; // ~1,000,722,000 implicit points inside the unit sphere

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
                  << " (" << model.classes.size() << " existing classes, grid="
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
    std::cout << "[DATA]   Loaded " << ds.count << " images (" << ds.width << "x" << ds.height
              << ", " << ds.unique_labels.size() << " classes)\n";

    // 3. Dynamically allocate guaranteed non-overlapping WxH lattice spaces + outer points
    uint32_t pix_per_img = ds.width * ds.height;
    std::vector<int> local_to_model(ds.unique_labels.size(), -1);
    std::map<std::string, uint32_t> label_to_local;
    for (size_t i = 0; i < ds.unique_labels.size(); ++i) {
        const auto& lbl = ds.unique_labels[i];
        bool existed = (model.find_class(lbl) >= 0);
        int mid = model.register_or_get_class(lbl, ds.width, ds.height);
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
    for (uint32_t i = 0; i < ds.count; ++i) {
        h_class_ids[i] = label_to_local[ds.labels[i]];
    }

    // 4. Execute CUDA Shadow Accumulation Kernel on NVIDIA GPU
    CU_CHECK(cuInit(0));
    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    char gpu_name[256] = {};
    CU_CHECK(cuDeviceGetName(gpu_name, sizeof(gpu_name), dev));
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&ctx, 0, dev));

    CUmodule mod = gllm::load_gllm_ptx_module();
    CUfunction fn_train = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_train, mod, "train_accumulate_shadows"));

    CUevent ev_s = nullptr, ev_e = nullptr;
    CU_CHECK(cuEventCreate(&ev_s, 0));
    CU_CHECK(cuEventCreate(&ev_e, 0));

    CUdeviceptr d_imgs = 0, d_cids = 0, d_sum = 0, d_cnt = 0;
    CU_CHECK(cuMemAlloc_v2(&d_imgs, ds.pixels.size()));
    CU_CHECK(cuMemcpyHtoD_v2(d_imgs, ds.pixels.data(), ds.pixels.size()));
    CU_CHECK(cuMemAlloc_v2(&d_cids, h_class_ids.size() * sizeof(uint32_t)));
    CU_CHECK(cuMemcpyHtoD_v2(d_cids, h_class_ids.data(), h_class_ids.size() * sizeof(uint32_t)));
    CU_CHECK(cuMemAlloc_v2(&d_sum,  static_cast<size_t>(num_local_classes) * pix_per_img * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_cnt,  static_cast<size_t>(num_local_classes) * sizeof(uint32_t)));
    CU_CHECK(cuMemsetD32_v2(d_sum, 0u, static_cast<size_t>(num_local_classes) * pix_per_img));
    CU_CHECK(cuMemsetD32_v2(d_cnt, 0u, num_local_classes));

    // Chunk GPU launches at 65,536 images per kernel call so datasets of 1M+ images never overflow 32-bit tid
    float gpu_ms = 0.0f;
    constexpr uint32_t kMaxBatch = 65536u;
    for (uint32_t start = 0; start < ds.count; start += kMaxBatch) {
        uint32_t batch = std::min(kMaxBatch, ds.count - start);
        uint32_t total_work = batch * pix_per_img;
        void* params[] = {
            &d_imgs, &d_cids, &d_sum, &d_cnt,
            &pix_per_img, &num_local_classes, &start, &batch
        };
        CU_CHECK(cuEventRecord(ev_s, nullptr));
        CU_CHECK(cuLaunchKernel(fn_train, (total_work + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
        CU_CHECK(cuEventRecord(ev_e, nullptr));
        CU_CHECK(cuEventSynchronize(ev_e));
        float chunk_ms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&chunk_ms, ev_s, ev_e));
        gpu_ms += chunk_ms;
    }

    std::vector<float> h_sum(static_cast<size_t>(num_local_classes) * pix_per_img);
    std::vector<uint32_t> h_cnt(num_local_classes);
    CU_CHECK(cuMemcpyDtoH_v2(h_sum.data(), d_sum, h_sum.size() * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH_v2(h_cnt.data(), d_cnt, h_cnt.size() * sizeof(uint32_t)));

    // 5. Merge accumulated shadows cleanly into the model's ClassRecords (supports continual adds!)
    std::cout << "\n[CUDA]   Accumulated " << ds.count << " shadows on " << gpu_name
              << " in " << gpu_ms << " ms (" << (ds.count / (gpu_ms / 1000.0)) << " imgs/sec)\n";

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
        std::printf("  Class '%-8s': total=%6u shadows | active=%4u/%u pts | peak overlap=%.1f%%\n",
                    rec.label.c_str(), rec.train_count, active, pix_per_img, peak * 100.0f);
    }

    if (!model.save(model_path)) {
        std::cerr << "[ERROR] Failed to save model to " << model_path << "\n";
        return 1;
    }
    std::cout << "[SAVED]  Model saved to " << model_path
              << " (total classes in sphere: " << model.classes.size() << ")\n";

    CU_CHECK(cuMemFree_v2(d_imgs));
    CU_CHECK(cuMemFree_v2(d_cids));
    CU_CHECK(cuMemFree_v2(d_sum));
    CU_CHECK(cuMemFree_v2(d_cnt));
    CU_CHECK(cuEventDestroy_v2(ev_s));
    CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(ctx));
    return 0;
}
