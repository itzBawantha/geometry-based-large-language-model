// =============================================================================
// gllm_inspect — Lookup Table Inspector & Exporter for .gllm Checkpoints
//
// Reads the serialized lookup table embedded in any .gllm checkpoint and
// displays or queries the exact mapping between:
//   Outer-Shell Point Scalar Value <-> Class Label <-> Outer (x, y, z)
//   <-> Unique 3D Lattice Space [ox..ox+w-1, oy..oy+h-1, z=oz] <-> Shadow Stats
//
// Usage:
//   gllm_inspect --model models/sphere_1b.gllm [--json | --csv]
//   gllm_inspect --model models/sphere_1b.gllm --value 3
//   gllm_inspect --model models/sphere_1b.gllm --label A
// =============================================================================

#include "../include/gllm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_inspect --model <path.gllm> [--json | --csv]\n"
              << "  gllm_inspect --model <path.gllm> --value <outer_scalar_id>\n"
              << "  gllm_inspect --model <path.gllm> --label <class_label>\n";
}

struct RowStats {
    uint32_t active_pts = 0;
    float peak_overlap  = 0.0f;
};

static RowStats compute_stats(const gllm::ClassRecord& rec) {
    RowStats st;
    if (rec.train_count == 0) return st;
    float inv = 1.0f / static_cast<float>(rec.train_count);
    for (float s : rec.shadow_sum) {
        float avg = s * inv;
        if (avg > 0.004f) st.active_pts++;
        if (avg > st.peak_overlap) st.peak_overlap = avg;
    }
    return st;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string query_label;
    int query_value = -1;
    bool as_json = false;
    bool as_csv  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--value" && i + 1 < argc) query_value = std::atoi(argv[++i]);
        else if (a == "--label" && i + 1 < argc) query_label = argv[++i];
        else if (a == "--json") as_json = true;
        else if (a == "--csv") as_csv = true;
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
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

    auto matches = [&](size_t idx, const gllm::ClassRecord& rec) {
        if (query_value >= 0 && static_cast<int>(rec.outer_idx) != query_value && static_cast<int>(idx) != query_value)
            return false;
        if (!query_label.empty() && rec.label != query_label)
            return false;
        return true;
    };

    if (as_json) {
        auto mtable = model.build_gpu_manifold_table();
        std::cout << "{\n"
                  << "  \"model\": \"" << model_path << "\",\n"
                  << "  \"lattice_grid\": " << model.lattice_grid << ",\n"
                  << "  \"approx_sphere_lattice_points\": " << model.approx_sphere_lattice_points() << ",\n"
                  << "  \"outer_shell_points\": " << model.outer_shell_pts << ",\n"
                  << "  \"operational_manifolds\": [\n";
        for (size_t i = 0; i < mtable.size(); ++i) {
            const auto& d = mtable[i];
            const auto* mr = model.get_manifold(d.manifold_id);
            if (i > 0) std::cout << ",\n";
            std::printf(
                "    {\"manifold_id\": %u, \"name\": \"%s\", \"cuda_semantic_label\": \"%s\", "
                "\"unique_classes\": %u, \"subshadow_regions\": %u, "
                "\"center_xyz\": [%.5f, %.5f, %.5f], \"max_radius\": %.5f, \"sphere_volume_pct\": %.5f, "
                "\"lattice_bounds\": [[%u, %u], [%u, %u], [%u, %u]]}",
                d.manifold_id, d.name, d.semantic_label, d.class_count, d.region_count,
                d.center_x, d.center_y, d.center_z, d.max_radius, d.volume_fraction * 100.0f,
                mr ? mr->min_lx : 0u, mr ? mr->max_lx : 0u,
                mr ? mr->min_ly : 0u, mr ? mr->max_ly : 0u,
                mr ? mr->min_lz : 0u, mr ? mr->max_lz : 0u
            );
        }
        std::cout << "\n  ],\n"
                  << "  \"assigned_classes\": " << model.classes.size() << ",\n"
                  << "  \"lookup_table\": [\n";
        bool first = true;
        for (size_t i = 0; i < model.classes.size(); ++i) {
            const auto& c = model.classes[i];
            if (!matches(i, c)) continue;
            RowStats st = compute_stats(c);
            const auto* mr = model.get_manifold(c.manifold_id);
            if (!first) std::cout << ",\n";
            first = false;
            std::printf(
                "    {\"outer_scalar\": %.1f, \"outer_idx\": %u, \"label\": \"%s\", "
                "\"manifold_id\": %u, \"manifold\": \"%s\", "
                "\"outer_xyz\": [%.5f, %.5f, %.5f], "
                "\"lattice_origin\": [%u, %u, %u], \"size\": [%u, %u], "
                "\"train_shadows\": %u, \"active_pts\": %u, \"peak_overlap_pct\": %.2f}",
                static_cast<float>(c.outer_idx), c.outer_idx, c.label.c_str(),
                c.manifold_id, mr ? mr->name.c_str() : "default",
                c.outer_x, c.outer_y, c.outer_z,
                c.ox, c.oy, c.oz, c.width, c.height,
                c.train_count, st.active_pts, st.peak_overlap * 100.0f
            );
        }
        std::cout << "\n  ]\n}\n";
        return 0;
    }

    if (as_csv) {
        std::cout << "outer_scalar,outer_idx,label,manifold_id,manifold,outer_x,outer_y,outer_z,ox,oy,oz,width,height,train_shadows,active_pts,peak_overlap_pct\n";
        for (size_t i = 0; i < model.classes.size(); ++i) {
            const auto& c = model.classes[i];
            if (!matches(i, c)) continue;
            RowStats st = compute_stats(c);
            const auto* mr = model.get_manifold(c.manifold_id);
            std::printf("%.1f,%u,%s,%u,%s,%.5f,%.5f,%.5f,%u,%u,%u,%u,%u,%u,%u,%.2f\n",
                        static_cast<float>(c.outer_idx), c.outer_idx, c.label.c_str(),
                        c.manifold_id, mr ? mr->name.c_str() : "default",
                        c.outer_x, c.outer_y, c.outer_z,
                        c.ox, c.oy, c.oz, c.width, c.height,
                        c.train_count, st.active_pts, st.peak_overlap * 100.0f);
        }
        return 0;
    }

    // Crisp human-readable lookup table + Operational Manifolds summary
    std::printf("GLLM Lookup Table | %s\n", model_path.c_str());
    std::printf("Sphere Lattice: %u^3 (~%llu implicit pts) | Outer Shell: %u unique classes (%zu sub-shadow regions) / %u pts\n\n",
                model.lattice_grid,
                static_cast<unsigned long long>(model.approx_sphere_lattice_points()),
                model.unique_class_count(), model.classes.size(), model.outer_shell_pts);

    auto mtable = model.build_gpu_manifold_table();
    if (!mtable.empty()) {
        std::printf("Operational Manifolds (%zu registered in CUDA metadata):\n", mtable.size());
        std::printf("  %-4s | %-18s | %-32s | %-7s | %-7s | %-28s | %-8s\n",
                    "ID", "Manifold Name", "CUDA Semantic Label", "Classes", "Regions", "Tight Lattice Bounds", "Vol %");
        std::printf("  -----+--------------------+----------------------------------+---------+---------+------------------------------+---------\n");
        for (const auto& d : mtable) {
            const auto* mr = model.get_manifold(d.manifold_id);
            char bbuf[64];
            std::snprintf(bbuf, sizeof(bbuf), "[%u..%u,%u..%u,%u..%u]",
                          mr ? mr->min_lx : 0u, mr ? mr->max_lx : 0u,
                          mr ? mr->min_ly : 0u, mr ? mr->max_ly : 0u,
                          mr ? mr->min_lz : 0u, mr ? mr->max_lz : 0u);
            std::printf("  #%-3u | %-18s | %-32s | %-7u | %-7u | %-28s | %6.4f%%\n",
                        d.manifold_id, d.name, d.semantic_label,
                        d.class_count, d.region_count, bbuf, d.volume_fraction * 100.0f);
        }
        std::printf("\n");
    }

    std::printf("%-8s | %-10s | %-25s | %-28s | %-7s | %-10s | %-8s\n",
                "Scalar", "Label", "Outer Point (x, y, z)", "Lattice Space [x, y, z]",
                "Size", "Shadows", "Peak");
    std::printf("---------+------------+---------------------------+------------------------------+---------+------------+---------\n");

    int shown = 0;
    for (size_t i = 0; i < model.classes.size(); ++i) {
        const auto& c = model.classes[i];
        if (!matches(i, c)) continue;
        RowStats st = compute_stats(c);
        char outer_buf[64], lat_buf[64], sz_buf[32];
        std::snprintf(outer_buf, sizeof(outer_buf), "(%+.3f, %+.3f, %+.3f)", c.outer_x, c.outer_y, c.outer_z);
        std::snprintf(lat_buf, sizeof(lat_buf), "[%4u..%4u,%4u..%4u,z=%4u]",
                      c.ox, c.ox + c.width - 1, c.oy, c.oy + c.height - 1, c.oz);
        std::snprintf(sz_buf, sizeof(sz_buf), "%ux%u", c.width, c.height);
        std::printf("%-8.1f | %-10s | %-25s | %-28s | %-7s | %-10u | %5.1f%%\n",
                    static_cast<float>(c.outer_idx), c.label.c_str(),
                    outer_buf, lat_buf, sz_buf, c.train_count, st.peak_overlap * 100.0f);
        shown++;
    }
    if (shown == 0) {
        std::cout << "(No matching entry found in lookup table.)\n";
    }
    return 0;
}
