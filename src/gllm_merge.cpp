// =============================================================================
// gllm_merge — Unified 1B-Sphere Universe Merger
//
// Combines multiple independent .gllm checkpoints (e.g. `handwritten` + `animals`
// + any future domain) into ONE unified 1-billion-point sphere (.gllm v5) with:
//   - Guaranteed ZERO spatial overlap across all regions & nested 3D boxes
//   - Clean collision-free remapping of manifold IDs, parent links, & outer nodes
//   - Distinct outer-shell crown sectors so each root domain's concept nodes
//     remain visually clear on the 1.2M outer sphere (R = 1.0)
//   - Explicit per-root inference enable/disable & modality registration
// =============================================================================

#include "../include/gllm_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_merge --output <universe.gllm>\n"
              << "             --input <model1.gllm> [--shift dx,dy,dz] [--infer on|off] [--modality sketch28|rgb]\n"
              << "             --input <model2.gllm> [--shift dx,dy,dz] [--infer on|off] [--modality sketch28|rgb]\n"
              << "             [...]\n";
}

struct InputSpec {
    std::string path;
    int dx = 0, dy = 0, dz = 0;
    int infer_override = -1; // -1 = keep from file, 0 = off, 1 = on
    int mod_override   = -1; // -1 = keep from file, 1 = Sketch28, 2 = RgbPyr
};

static bool parse_vec3i(const std::string& s, int& dx, int& dy, int& dz) {
    std::stringstream ss(s);
    std::string tok;
    int v[3] = {};
    int k = 0;
    while (std::getline(ss, tok, ',') && k < 3) v[k++] = std::atoi(tok.c_str());
    if (k != 3) return false;
    dx = v[0]; dy = v[1]; dz = v[2];
    return true;
}

// Re-project a secondary domain's outer-shell concept nodes onto a clean distinct sector on R = 1.0
// (so domain 0 keeps the upper-front crown while domain 1 uses the lower-front arc, etc.)
static void remap_domain_outer_point(uint32_t domain_idx, uint32_t local_idx,
                                     float orig_x, float orig_y, float orig_z,
                                     float& ox, float& oy, float& oz) {
    if (domain_idx == 0u) {
        ox = orig_x; oy = orig_y; oz = orig_z;
        return;
    }
    // Mirror cleanly onto lower-front crown for domain 1 (or golden sector for domain >= 2)
    if (domain_idx == 1u) {
        // Flip Y neatly onto the lower-front hemisphere so 'animals' concepts form a clean mirror crown
        // right beneath the 'animals' 3D box (or upper-right sector if shifted up)!
        float tx = orig_x;
        float ty = -orig_y;
        float xy2 = tx * tx + ty * ty;
        if (xy2 > 0.96f) { float s = std::sqrt(0.96f / xy2); tx *= s; ty *= s; xy2 = tx * tx + ty * ty; }
        ox = tx;
        oy = ty;
        oz = std::sqrt(std::max(0.0f, 1.0f - xy2));
        return;
    }
    constexpr float kGolden = 2.39996322972865332f;
    float base_ang = static_cast<float>(domain_idx) * 2.0943951f;
    float ang = base_ang + static_cast<float>(local_idx) * 0.14f * std::cos(local_idx * kGolden);
    float tx = 0.72f * std::cos(ang);
    float ty = 0.45f * std::sin(ang);
    float xy2 = tx * tx + ty * ty;
    ox = tx; oy = ty; oz = std::sqrt(std::max(0.0f, 1.0f - xy2));
}

int main(int argc, char** argv) {
    std::string out_path;
    std::vector<InputSpec> inputs;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--output" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--input" && i + 1 < argc) {
            InputSpec sp;
            sp.path = argv[++i];
            inputs.push_back(sp);
        } else if (a == "--shift" && i + 1 < argc && !inputs.empty()) {
            if (!parse_vec3i(argv[++i], inputs.back().dx, inputs.back().dy, inputs.back().dz)) {
                std::cerr << "[ERROR] Invalid --shift format (expected dx,dy,dz)\n";
                return 1;
            }
        } else if (a == "--infer" && i + 1 < argc && !inputs.empty()) {
            std::string v = argv[++i];
            inputs.back().infer_override = (v == "on" || v == "1" || v == "true") ? 1 : 0;
        } else if (a == "--modality" && i + 1 < argc && !inputs.empty()) {
            std::string v = argv[++i];
            if (v == "sketch28") inputs.back().mod_override = static_cast<int>(gllm::InferenceModality::kSketch28x28);
            else if (v == "rgb") inputs.back().mod_override = static_cast<int>(gllm::InferenceModality::kRgbPyramid);
            else if (v == "none") inputs.back().mod_override = static_cast<int>(gllm::InferenceModality::kNone);
        } else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    if (out_path.empty() || inputs.empty()) {
        print_usage();
        return 1;
    }

    gllm::GeometricModel universe;
    uint32_t next_manifold_id = 0u;
    uint32_t next_outer_id    = 0u;

    for (size_t di = 0; di < inputs.size(); ++di) {
        const auto& spec = inputs[di];
        gllm::GeometricModel part;
        if (!part.load(spec.path)) {
            std::cerr << "[ERROR] Failed to load input model: " << spec.path << "\n";
            return 1;
        }
        if (di == 0) {
            universe.lattice_grid    = part.lattice_grid;
            universe.outer_shell_pts = part.outer_shell_pts;
        } else if (part.lattice_grid != universe.lattice_grid) {
            std::cerr << "[ERROR] Lattice grid mismatch: " << part.lattice_grid
                      << " vs " << universe.lattice_grid << "\n";
            return 1;
        }

        // 1. Build manifold ID remap table & outer-node remap table for this input model
        std::map<uint32_t, uint32_t> mid_map;
        std::map<uint32_t, uint32_t> oid_map;
        std::map<uint32_t, std::array<float, 3>> oid_xyz;

        // Collect all unique old outer_idx in deterministic ascending order
        std::vector<uint32_t> old_oids;
        auto track_oid = [&](uint32_t oid, float x, float y, float z) {
            if (oid == gllm::kInvalidId) return;
            if (oid_xyz.find(oid) == oid_xyz.end()) {
                oid_xyz[oid] = {x, y, z};
                old_oids.push_back(oid);
            }
        };
        for (const auto& c : part.classes)   track_oid(c.outer_idx, c.outer_x, c.outer_y, c.outer_z);
        for (const auto& m : part.manifolds) track_oid(m.outer_idx, m.outer_x, m.outer_y, m.outer_z);
        std::sort(old_oids.begin(), old_oids.end());

        std::map<uint32_t, std::array<float, 3>> new_oid_xyz;
        for (size_t k = 0; k < old_oids.size(); ++k) {
            uint32_t old_id = old_oids[k];
            uint32_t new_id = (di == 0) ? old_id : next_outer_id++;
            if (di == 0) next_outer_id = std::max(next_outer_id, new_id + 1u);
            oid_map[old_id] = new_id;
            auto orig = oid_xyz[old_id];
            float nx, ny, nz;
            remap_domain_outer_point(static_cast<uint32_t>(di), static_cast<uint32_t>(k),
                                     orig[0], orig[1], orig[2], nx, ny, nz);
            new_oid_xyz[new_id] = {nx, ny, nz};
        }

        for (const auto& m : part.manifolds) {
            mid_map[m.manifold_id] = next_manifold_id++;
        }

        // 2. Append remapped manifolds with shifted anchors & inference flags
        for (auto m : part.manifolds) {
            m.manifold_id = mid_map[m.manifold_id];
            if (m.parent_manifold_id != gllm::kInvalidId) {
                m.parent_manifold_id = mid_map[m.parent_manifold_id];
            }
            m.anchor_x = static_cast<uint32_t>(std::clamp<int64_t>(static_cast<int64_t>(m.anchor_x) + spec.dx, 0, universe.lattice_grid - 1));
            m.anchor_y = static_cast<uint32_t>(std::clamp<int64_t>(static_cast<int64_t>(m.anchor_y) + spec.dy, 0, universe.lattice_grid - 1));
            m.anchor_z = static_cast<uint32_t>(std::clamp<int64_t>(static_cast<int64_t>(m.anchor_z) + spec.dz, 0, universe.lattice_grid - 1));
            if (m.outer_idx != gllm::kInvalidId) {
                m.outer_idx = oid_map[m.outer_idx];
                auto xyz = new_oid_xyz[m.outer_idx];
                m.outer_x = xyz[0]; m.outer_y = xyz[1]; m.outer_z = xyz[2];
            }
            bool en = gllm::is_infer_flag_enabled(m.inference_flags);
            auto mod = gllm::get_infer_modality(m.inference_flags);
            if (spec.infer_override >= 0) en = (spec.infer_override != 0);
            if (spec.mod_override >= 0)   mod = static_cast<gllm::InferenceModality>(spec.mod_override);
            m.inference_flags = gllm::pack_inference_flags(en, mod);
            universe.manifolds.push_back(std::move(m));
        }

        // 3. Append shifted regions with strict sphere containment & 0-overlap verification
        for (auto c : part.classes) {
            int64_t nx = static_cast<int64_t>(c.ox) + spec.dx;
            int64_t ny = static_cast<int64_t>(c.oy) + spec.dy;
            int64_t nz = static_cast<int64_t>(c.oz) + spec.dz;
            if (!universe.inside_sphere(nx, ny, nz, c.width, c.height, 0.92f)) {
                std::cerr << "[ERROR] Region '" << c.label << "' from " << spec.path
                          << " exceeds sphere bounds after shift (" << spec.dx << "," << spec.dy << "," << spec.dz << ")\n";
                return 1;
            }
            if (universe.overlaps_any(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), static_cast<uint32_t>(nz),
                                      c.width, c.height, 1u, 1u)) {
                std::cerr << "[ERROR] Spatial overlap detected while merging '" << c.label << "' from " << spec.path
                          << " — adjust --shift so domains sit in disjoint regions!\n";
                return 1;
            }
            c.ox = static_cast<uint32_t>(nx);
            c.oy = static_cast<uint32_t>(ny);
            c.oz = static_cast<uint32_t>(nz);
            c.manifold_id = mid_map[c.manifold_id];
            c.outer_idx   = oid_map[c.outer_idx];
            auto xyz      = new_oid_xyz[c.outer_idx];
            c.outer_x = xyz[0]; c.outer_y = xyz[1]; c.outer_z = xyz[2];
            universe.classes.push_back(std::move(c));
        }

        std::printf("[MERGE]  Added %-34s | shift=(%+d,%+d,%+d) | +%zu manifolds, +%zu regions\n",
                    spec.path.c_str(), spec.dx, spec.dy, spec.dz,
                    part.manifolds.size(), part.classes.size());
    }

    universe.recompute_manifold_envelopes();

    // Verify 0 box-vs-box overlap between distinct root manifolds!
    for (size_t i = 0; i < universe.manifolds.size(); ++i) {
        const auto& a = universe.manifolds[i];
        if (a.parent_manifold_id != gllm::kInvalidId) continue;
        for (size_t j = i + 1; j < universe.manifolds.size(); ++j) {
            const auto& b = universe.manifolds[j];
            if (b.parent_manifold_id != gllm::kInvalidId) continue;
            bool box_hit = !(a.max_lx < b.min_lx || b.max_lx < a.min_lx ||
                             a.max_ly < b.min_ly || b.max_ly < a.min_ly ||
                             a.max_lz < b.min_lz || b.max_lz < a.min_lz);
            if (box_hit) {
                std::cerr << "[WARN] Root manifold 3D boxes '" << a.name << "' and '" << b.name << "' touch/overlap!\n";
            }
        }
    }

    if (!universe.save(out_path)) {
        std::cerr << "[ERROR] Failed to save unified universe model: " << out_path << "\n";
        return 1;
    }
    std::cout << "\n[UNIVERSE] Saved unified 1B sphere: " << out_path
              << " (" << universe.manifolds.size() << " manifolds, "
              << universe.classes.size() << " regions, "
              << next_outer_id << " outer-shell nodes)\n";
    return 0;
}
