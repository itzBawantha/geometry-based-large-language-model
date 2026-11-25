#pragma once

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace gllm {

constexpr uint32_t kModelMagic   = 0x4D4C4C47u; // "GLLM"
constexpr uint32_t kModelVersion = 4u;          // v4 adds hierarchical 3D AABB manifolds + shared outer-shell concept bindings
constexpr uint32_t kManifoldVerts = 162u;       // 2x-subdivided geodesic icosphere (162 verts, 320 tris)
constexpr uint32_t kInvalidId     = 0xFFFFFFFFu;

// GPU-compatible 32-byte region descriptor (matches kernels/gllm_kernels.ptx)
struct alignas(16) GpuRegionDesc {
    uint32_t ox, oy, oz, width;
    uint32_t height, outer_idx, shadow_offset, class_id;
};

// GPU-compatible 16-byte outer-shell point descriptor
struct alignas(16) GpuOuterPoint {
    float x, y, z, value;
};

// GPU-resident 144-byte Operational Manifold Metadata Register (uploaded directly to CUDA VRAM!)
struct alignas(16) GpuManifoldDesc {
    uint32_t manifold_id;
    uint32_t class_count;
    uint32_t region_count;
    uint32_t parent_manifold_id; // kInvalidId (0xFFFFFFFF) for root manifolds
    float    center_x, center_y, center_z, max_radius;
    float    min_x, min_y, min_z, volume_fraction;
    float    max_x, max_y, max_z;
    uint32_t outer_idx;          // Shared outer-shell concept node ID (kInvalidId if unbound)
    char     name[32];           // e.g. "fundamental_text", "animals/cats/face"
    char     semantic_label[48]; // e.g. "handwritten_letters_and_numbers", "face"
};
static_assert(sizeof(GpuManifoldDesc) == 144, "GpuManifoldDesc must be 144 bytes for CUDA alignment");

struct GeodesicSphereMesh {
    std::vector<float> dirs;     // 162 * 3 unit directions (x, y, z)
    std::vector<uint32_t> tris;  // 320 * 3 triangle vertex indices
    std::vector<uint32_t> edges; // 480 * 2 unique wireframe edge vertex indices
    std::vector<std::vector<uint32_t>> adj; // 162 neighbor lists
};

// Deterministic 2x-subdivided unit geodesic icosphere (162 verts, 320 tris, 480 edges)
inline const GeodesicSphereMesh& unit_geodesic_sphere() {
    static GeodesicSphereMesh mesh = []() {
        GeodesicSphereMesh m;
        const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
        auto add_v = [&](float x, float y, float z) -> uint32_t {
            float l = std::sqrt(x * x + y * y + z * z);
            uint32_t id = static_cast<uint32_t>(m.dirs.size() / 3);
            m.dirs.push_back(x / l);
            m.dirs.push_back(y / l);
            m.dirs.push_back(z / l);
            return id;
        };
        add_v(-1,  t, 0); add_v( 1,  t, 0); add_v(-1, -t, 0); add_v( 1, -t, 0);
        add_v( 0, -1,  t); add_v( 0,  1,  t); add_v( 0, -1, -t); add_v( 0,  1, -t);
        add_v( t,  0, -1); add_v( t,  0,  1); add_v(-t,  0, -1); add_v(-t,  0,  1);

        std::vector<uint32_t> tris = {
            0,11,5,  0,5,1,  0,1,7,  0,7,10, 0,10,11,
            1,5,9,   5,11,4, 11,10,2, 10,7,6, 7,1,8,
            3,9,4,   3,4,2,  3,2,6,  3,6,8,  3,8,9,
            4,9,5,   2,4,11, 6,2,10, 8,6,7,  9,8,1
        };
        for (int sub = 0; sub < 2; ++sub) {
            std::map<uint64_t, uint32_t> mid_cache;
            auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
                uint64_t key = (a < b) ? ((uint64_t(a) << 32) | b) : ((uint64_t(b) << 32) | a);
                auto it = mid_cache.find(key);
                if (it != mid_cache.end()) return it->second;
                uint32_t nid = add_v(
                    0.5f * (m.dirs[a * 3 + 0] + m.dirs[b * 3 + 0]),
                    0.5f * (m.dirs[a * 3 + 1] + m.dirs[b * 3 + 1]),
                    0.5f * (m.dirs[a * 3 + 2] + m.dirs[b * 3 + 2]));
                mid_cache[key] = nid;
                return nid;
            };
            std::vector<uint32_t> next;
            next.reserve(tris.size() * 4);
            for (size_t i = 0; i < tris.size(); i += 3) {
                uint32_t a = tris[i], b = tris[i + 1], c = tris[i + 2];
                uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
                uint32_t sub_t[12] = {a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca};
                next.insert(next.end(), sub_t, sub_t + 12);
            }
            tris.swap(next);
        }
        m.tris = std::move(tris);
        size_t nv = m.dirs.size() / 3;
        m.adj.resize(nv);
        std::map<uint64_t, bool> seen_e;
        auto add_edge = [&](uint32_t a, uint32_t b) {
            uint64_t key = (a < b) ? ((uint64_t(a) << 32) | b) : ((uint64_t(b) << 32) | a);
            if (!seen_e[key]) {
                seen_e[key] = true;
                m.edges.push_back(a);
                m.edges.push_back(b);
                m.adj[a].push_back(b);
                m.adj[b].push_back(a);
            }
        };
        for (size_t i = 0; i < m.tris.size(); i += 3) {
            add_edge(m.tris[i], m.tris[i + 1]);
            add_edge(m.tris[i + 1], m.tris[i + 2]);
            add_edge(m.tris[i + 2], m.tris[i]);
        }
        return m;
    }();
    return mesh;
}

struct ManifoldRecord {
    uint32_t manifold_id = 0;
    uint32_t parent_manifold_id = kInvalidId; // kInvalidId = root manifold
    uint32_t depth = 0;                       // Hierarchy depth (0=root, 1=species, 2=face, 3=feature)
    uint32_t envelope_mode = 0;               // 0 = geodesic deformed sphere, 1 = exact 3D Bounding Box (AABB)
    uint32_t outer_idx = kInvalidId;          // Outer-shell concept node ID (shared across matching concept_label!)
    float    outer_x = 0.0f, outer_y = 0.0f, outer_z = 1.0f;
    std::string concept_label;                // Outer-shell concept word (e.g. "animals", "cat", "dog", "face")
    std::string name = "fundamental_text";
    std::string semantic_label = "handwritten_letters_and_numbers";
    uint32_t anchor_x = 620, anchor_y = 620, anchor_z = 620; // Lattice packing center
    float center_x = 0.0f, center_y = 0.0f, center_z = 0.0f; // World center of enclosed subspaces
    float min_x = 0.0f, min_y = 0.0f, min_z = 0.0f;          // Tight world AABB min
    float max_x = 0.0f, max_y = 0.0f, max_z = 0.0f;          // Tight world AABB max
    float max_radius = 0.0f;                                 // Tight max radius / box half-diagonal
    float volume_fraction = 0.0f;                            // Fraction of 1B unit sphere volume used
    uint32_t min_lx = 0, min_ly = 0, min_lz = 0;             // Tight lattice AABB min
    uint32_t max_lx = 0, max_ly = 0, max_lz = 0;             // Tight lattice AABB max
    float deformed_radii[kManifoldVerts] = {};               // Exact tight deformed-sphere envelope radii
};

struct ClassRecord {
    std::string label;
    uint32_t ox = 0, oy = 0, oz = 0;   // Unique non-overlapping origin in G^3 lattice
    uint32_t width = 28, height = 28;  // Exact 1-to-1 pixel dimensions
    uint32_t outer_idx = 0;            // Assigned point index on the 1.2M outer shell
    float outer_x = 0.0f, outer_y = 0.0f, outer_z = 1.0f;
    uint32_t train_count = 0;          // Total shadows accumulated into this region
    uint32_t manifold_id = 0;          // Operational Manifold membership ID
    uint32_t channels    = 1;          // 1 = grayscale luminance only, 3 = luminance + packed RGB (0x01RRGGBB)
    std::vector<float>    shadow_sum;  // width * height raw accumulated shadow totals
    std::vector<uint32_t> shadow_rgb;  // width * height packed 0x01RRGGBB (when channels == 3)
};

struct GeometricModel {
    uint32_t lattice_grid    = 1241;    // 1241^3 sphere -> ~1,000,722,000 implicit lattice points!
    uint32_t outer_shell_pts = 1200000; // Permanent 1.2M outer spherical shell capacity
    std::vector<ManifoldRecord> manifolds;
    std::vector<ClassRecord> classes;

    int find_manifold_by_name(const std::string& name) const {
        for (size_t i = 0; i < manifolds.size(); ++i) {
            if (manifolds[i].name == name) return static_cast<int>(i);
        }
        return -1;
    }

    const ManifoldRecord* get_manifold(uint32_t mid) const {
        for (const auto& m : manifolds) {
            if (m.manifold_id == mid) return &m;
        }
        return nullptr;
    }

    // Find or allocate a new non-overlapping Operational Manifold anchor inside the 1B sphere
    uint32_t find_or_create_manifold(const std::string& name, const std::string& semantic_label) {
        int existing = find_manifold_by_name(name);
        if (existing >= 0) {
            if (!semantic_label.empty()) manifolds[existing].semantic_label = semantic_label;
            return manifolds[existing].manifold_id;
        }

        uint32_t new_id = 0;
        for (const auto& m : manifolds) new_id = std::max(new_id, m.manifold_id + 1u);

        ManifoldRecord m;
        m.manifold_id    = new_id;
        m.name           = name.empty() ? "default_manifold" : name;
        m.semantic_label = semantic_label.empty() ? m.name : semantic_label;

        // First manifold anchors dead-center at (G/2, G/2, G/2); subsequent manifolds place
        // cleanly at golden-spiral non-overlapping offsets outside existing manifold envelopes!
        uint32_t mid = lattice_grid / 2u;
        if (manifolds.empty()) {
            m.anchor_x = mid; m.anchor_y = mid; m.anchor_z = mid;
        } else {
            constexpr float kGolden = 2.39996322972865332f;
            float max_r_lat = 180.0f;
            for (const auto& em : manifolds) {
                float dx = float(em.max_lx > em.min_lx ? (em.max_lx - em.min_lx) : 120u);
                float dy = float(em.max_ly > em.min_ly ? (em.max_ly - em.min_ly) : 120u);
                float dz = float(em.max_lz > em.min_lz ? (em.max_lz - em.min_lz) : 120u);
                max_r_lat = std::max(max_r_lat, 0.65f * std::sqrt(dx * dx + dy * dy + dz * dz) + 90.0f);
            }
            float ang = static_cast<float>(new_id) * kGolden;
            float elev = (new_id % 2 == 1 ? 0.35f : -0.35f);
            int64_t ax = static_cast<int64_t>(mid) + static_cast<int64_t>(max_r_lat * std::cos(ang) * std::cos(elev));
            int64_t ay = static_cast<int64_t>(mid) + static_cast<int64_t>(max_r_lat * std::sin(ang) * std::cos(elev));
            int64_t az = static_cast<int64_t>(mid) + static_cast<int64_t>(max_r_lat * std::sin(elev));
            m.anchor_x = static_cast<uint32_t>(std::clamp<int64_t>(ax, 120, static_cast<int64_t>(lattice_grid) - 120));
            m.anchor_y = static_cast<uint32_t>(std::clamp<int64_t>(ay, 120, static_cast<int64_t>(lattice_grid) - 120));
            m.anchor_z = static_cast<uint32_t>(std::clamp<int64_t>(az, 120, static_cast<int64_t>(lattice_grid) - 120));
        }
        manifolds.push_back(std::move(m));
        return new_id;
    }

    int find_class(const std::string& label) const {
        for (size_t i = 0; i < classes.size(); ++i) {
            if (classes[i].label == label) return static_cast<int>(i);
        }
        return -1;
    }

    int find_outer_idx_for_label(const std::string& label) const {
        for (const auto& c : classes) {
            if (c.label == label) return static_cast<int>(c.outer_idx);
        }
        return -1;
    }

    uint32_t next_outer_idx() const {
        bool has_any = false;
        uint32_t mx = 0u;
        for (const auto& c : classes) {
            if (!has_any || c.outer_idx > mx) mx = c.outer_idx;
            has_any = true;
        }
        for (const auto& m : manifolds) {
            if (m.outer_idx != kInvalidId) {
                if (!has_any || m.outer_idx > mx) mx = m.outer_idx;
                has_any = true;
            }
        }
        return has_any ? (mx + 1u) : 0u;
    }

    // Find or allocate a shared outer-shell concept node ID for a concept word (e.g. "animals", "cat", "dog", "face")
    uint32_t find_or_assign_concept_outer(const std::string& concept_word, float& ox, float& oy, float& oz) const {
        for (const auto& m : manifolds) {
            if (m.outer_idx != kInvalidId && m.concept_label == concept_word) {
                ox = m.outer_x; oy = m.outer_y; oz = m.outer_z;
                return m.outer_idx;
            }
        }
        for (const auto& c : classes) {
            if (c.label == concept_word) {
                ox = c.outer_x; oy = c.outer_y; oz = c.outer_z;
                return c.outer_idx;
            }
        }
        uint32_t oidx = next_outer_idx();
        assign_outer_point(oidx, ox, oy, oz);
        return oidx;
    }

    // Create a hierarchical 3D Bounding Box manifold bound to a (potentially shared!) outer-shell concept node
    uint32_t create_hierarchical_manifold(const std::string& name,
                                          const std::string& concept_word,
                                          const std::string& semantic_label,
                                          uint32_t parent_id,
                                          uint32_t anchor_x, uint32_t anchor_y, uint32_t anchor_z,
                                          uint32_t envelope_mode = 1u) {
        int existing = find_manifold_by_name(name);
        if (existing >= 0) return manifolds[existing].manifold_id;

        uint32_t new_id = 0u;
        for (const auto& m : manifolds) new_id = std::max(new_id, m.manifold_id + 1u);

        uint32_t depth = 0u;
        if (parent_id != kInvalidId) {
            if (const ManifoldRecord* pm = get_manifold(parent_id)) depth = pm->depth + 1u;
        }

        ManifoldRecord m;
        m.manifold_id        = new_id;
        m.parent_manifold_id = parent_id;
        m.depth              = depth;
        m.envelope_mode      = envelope_mode;
        m.name               = name;
        m.concept_label      = concept_word.empty() ? name : concept_word;
        m.semantic_label     = semantic_label.empty() ? m.concept_label : semantic_label;
        m.anchor_x           = anchor_x;
        m.anchor_y           = anchor_y;
        m.anchor_z           = anchor_z;
        m.outer_idx          = find_or_assign_concept_outer(m.concept_label, m.outer_x, m.outer_y, m.outer_z);
        manifolds.push_back(std::move(m));
        return new_id;
    }

    uint32_t unique_class_count() const {
        std::map<uint32_t, bool> seen;
        for (const auto& c : classes) seen[c.outer_idx] = true;
        return static_cast<uint32_t>(seen.size());
    }

    uint64_t approx_sphere_lattice_points() const {
        double g = static_cast<double>(lattice_grid);
        return static_cast<uint64_t>((3.14159265358979323846 / 6.0) * g * g * g);
    }

    // Check 3D AABB overlap with a clean visual/safety margin against ALL existing classes
    bool overlaps_any(uint32_t ox, uint32_t oy, uint32_t oz,
                      uint32_t w, uint32_t h,
                      uint32_t pad_xy, uint32_t pad_z) const {
        int64_t ax0 = static_cast<int64_t>(ox) - pad_xy;
        int64_t ax1 = static_cast<int64_t>(ox + w - 1) + pad_xy;
        int64_t ay0 = static_cast<int64_t>(oy) - pad_xy;
        int64_t ay1 = static_cast<int64_t>(oy + h - 1) + pad_xy;
        int64_t az0 = static_cast<int64_t>(oz) - pad_z;
        int64_t az1 = static_cast<int64_t>(oz) + pad_z;

        for (const auto& c : classes) {
            int64_t bx0 = c.ox;
            int64_t bx1 = c.ox + c.width - 1;
            int64_t by0 = c.oy;
            int64_t by1 = c.oy + c.height - 1;
            int64_t bz0 = c.oz;
            int64_t bz1 = c.oz;
            bool hit = !(ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0 || az1 < bz0 || bz1 < az0);
            if (hit) return true;
        }
        return false;
    }

    // Check all 4 corners of a W x H x 1 region lie strictly inside sphere radius r_max
    bool inside_sphere(int64_t ox, int64_t oy, int64_t oz,
                       uint32_t w, uint32_t h, float r_max = 0.90f) const {
        if (ox < 0 || oy < 0 || oz < 0) return false;
        if (ox + static_cast<int64_t>(w) > static_cast<int64_t>(lattice_grid)) return false;
        if (oy + static_cast<int64_t>(h) > static_cast<int64_t>(lattice_grid)) return false;
        if (oz >= static_cast<int64_t>(lattice_grid)) return false;

        float step = 2.0f / static_cast<float>(lattice_grid);
        float z = (static_cast<float>(oz) + 0.5f) * step - 1.0f;
        int64_t xs[2] = {ox, ox + static_cast<int64_t>(w) - 1};
        int64_t ys[2] = {oy, oy + static_cast<int64_t>(h) - 1};
        for (int i = 0; i < 2; ++i) {
            float x = (static_cast<float>(xs[i]) + 0.5f) * step - 1.0f;
            for (int j = 0; j < 2; ++j) {
                float y = (static_cast<float>(ys[j]) + 0.5f) * step - 1.0f;
                if (x * x + y * y + z * z > r_max * r_max) return false;
            }
        }
        return true;
    }

    // Assign next unique neighboring outer-shell point on R = 1.0 (golden-angle spherical cap)
    void assign_outer_point(uint32_t idx, float& ox, float& oy, float& oz) const {
        // Tight local neighborhood cap starting on the upper-front crown of the sphere
        // (0, 0.78, 0.625) spiraling smoothly outward so neighboring classes stay clustered!
        constexpr float kGolden = 2.39996322972865332f;
        float r_cap = (idx == 0) ? 0.0f : (0.022f * std::sqrt(static_cast<float>(idx)));
        float ang   = static_cast<float>(idx) * kGolden;
        float tx    = r_cap * std::cos(ang);
        float ty    = 0.78f - r_cap * std::sin(ang);
        // Project cleanly onto exact unit sphere R = 1.0
        float xy2 = tx * tx + ty * ty;
        if (xy2 > 0.96f) {
            float s = std::sqrt(0.96f / xy2);
            tx *= s;
            ty *= s;
            xy2 = tx * tx + ty * ty;
        }
        ox = tx;
        oy = ty;
        oz = std::sqrt(std::max(0.0f, 1.0f - xy2));
    }

    // Minimum-volume Euclidean spherical tight-packing allocator inside the 3D sphere lattice!
    // Packs regions radially outward from target manifold's anchor (anchor_x, anchor_y, anchor_z)
    // using minimum non-overlapping separation (pad_xy=1, pad_z=2) -> minimum volume deformed sphere!
    bool find_free_lattice_slot(uint32_t w, uint32_t h, uint32_t manifold_id,
                                uint32_t& best_ox, uint32_t& best_oy, uint32_t& best_oz) const {
        uint32_t pad_xy = 1u; // Minimum strict non-overlapping XY safety voxel
        uint32_t pad_z  = 2u; // Minimum strict non-overlapping Z slice separation
        uint32_t stride_x = w + pad_xy;
        uint32_t stride_y = h + pad_xy;
        uint32_t stride_z = pad_z + 1u;

        int64_t cx = static_cast<int64_t>(lattice_grid / 2u);
        int64_t cy = static_cast<int64_t>(lattice_grid / 2u);
        int64_t cz = static_cast<int64_t>(lattice_grid / 2u);
        if (const ManifoldRecord* m = get_manifold(manifold_id)) {
            cx = static_cast<int64_t>(m->anchor_x);
            cy = static_cast<int64_t>(m->anchor_y);
            cz = static_cast<int64_t>(m->anchor_z);
        }

        // Build candidate offsets in true 3D Euclidean spherical-distance order around manifold anchor
        // so the allocated cluster naturally forms a compact, minimum-volume sphere!
        struct Cand { int64_t dist2; int dx, dy, dz; };
        constexpr int kMaxXY = 14;
        constexpr int kMaxZ  = 48;
        std::vector<Cand> cands;
        cands.reserve((2 * kMaxXY + 1) * (2 * kMaxXY + 1) * 16);
        // Scale search radius progressively if needed; first pass covers ~6,000 tightly packed slots
        for (int pass = 0; pass < 2; ++pass) {
            int max_xy = (pass == 0) ? 8 : kMaxXY;
            int max_z  = (pass == 0) ? 28 : kMaxZ;
            int64_t max_r2 = int64_t(max_xy * int(stride_x)) * int64_t(max_xy * int(stride_x));
            cands.clear();
            for (int dz = -max_z; dz <= max_z; ++dz) {
                int64_t zoff = int64_t(dz) * int64_t(stride_z);
                int64_t z2 = zoff * zoff;
                if (z2 > max_r2) continue;
                for (int dy = -max_xy; dy <= max_xy; ++dy) {
                    int64_t yoff = int64_t(dy) * int64_t(stride_y);
                    int64_t yz2 = z2 + yoff * yoff;
                    if (yz2 > max_r2) continue;
                    for (int dx = -max_xy; dx <= max_xy; ++dx) {
                        int64_t xoff = int64_t(dx) * int64_t(stride_x);
                        int64_t d2 = yz2 + xoff * xoff;
                        if (d2 <= max_r2) cands.push_back({d2, dx, dy, dz});
                    }
                }
            }
            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
                if (a.dist2 != b.dist2) return a.dist2 < b.dist2;
                if (std::abs(a.dz) != std::abs(b.dz)) return std::abs(a.dz) < std::abs(b.dz);
                if (a.dy != b.dy) return a.dy > b.dy;
                return a.dx < b.dx;
            });
            for (const auto& c : cands) {
                int64_t ox = cx + int64_t(c.dx) * int64_t(stride_x) - int64_t(w / 2u);
                int64_t oy = cy + int64_t(c.dy) * int64_t(stride_y) - int64_t(h / 2u);
                int64_t oz = cz + int64_t(c.dz) * int64_t(stride_z);
                if (!inside_sphere(ox, oy, oz, w, h, 0.90f)) continue;
                if (!overlaps_any(static_cast<uint32_t>(ox),
                                  static_cast<uint32_t>(oy),
                                  static_cast<uint32_t>(oz),
                                  w, h, pad_xy, pad_z)) {
                    best_ox = static_cast<uint32_t>(ox);
                    best_oy = static_cast<uint32_t>(oy);
                    best_oz = static_cast<uint32_t>(oz);
                    return true;
                }
            }
        }
        return false;
    }

    // Recompute exact manifold envelopes (supports both exact 3D AABB bounding boxes AND geodesic deformed spheres)
    // Propagates bounds bottom-up through parent_manifold_id so every parent box strictly encloses all child boxes!
    void recompute_manifold_envelopes() {
        if (manifolds.empty() && !classes.empty()) {
            find_or_create_manifold("fundamental_text", "handwritten_letters_and_numbers");
        }
        const auto& geo = unit_geodesic_sphere();
        float step = 2.0f / static_cast<float>(lattice_grid);
        constexpr double kUnitSphereVol = (4.0 / 3.0) * 3.14159265358979323846;

        // Track direct lattice bounds per manifold first
        std::vector<bool> has_bounds(manifolds.size(), false);
        for (size_t mi = 0; mi < manifolds.size(); ++mi) {
            auto& m = manifolds[mi];
            uint32_t min_lx = lattice_grid, min_ly = lattice_grid, min_lz = lattice_grid;
            uint32_t max_lx = 0, max_ly = 0, max_lz = 0;
            bool any = false;
            for (const auto& c : classes) {
                if (c.manifold_id != m.manifold_id) continue;
                uint32_t x0 = c.ox, x1 = c.ox + c.width - 1u;
                uint32_t y0 = c.oy, y1 = c.oy + c.height - 1u;
                uint32_t z0 = c.oz;
                min_lx = std::min(min_lx, x0); max_lx = std::max(max_lx, x1);
                min_ly = std::min(min_ly, y0); max_ly = std::max(max_ly, y1);
                min_lz = std::min(min_lz, z0); max_lz = std::max(max_lz, z0);
                any = true;
            }
            if (any) {
                if (m.envelope_mode == 1u) {
                    // Tight exact 3D bounding box padding (2 voxels XY, 4 voxels Z so flat planes have crisp 3D depth)
                    uint32_t px = 2u, pz = 4u;
                    m.min_lx = (min_lx > px) ? (min_lx - px) : 0u;
                    m.max_lx = std::min(lattice_grid - 1u, max_lx + px);
                    m.min_ly = (min_ly > px) ? (min_ly - px) : 0u;
                    m.max_ly = std::min(lattice_grid - 1u, max_ly + px);
                    m.min_lz = (min_lz > pz) ? (min_lz - pz) : 0u;
                    m.max_lz = std::min(lattice_grid - 1u, max_lz + pz);
                } else {
                    m.min_lx = min_lx; m.max_lx = max_lx;
                    m.min_ly = min_ly; m.max_ly = max_ly;
                    m.min_lz = min_lz; m.max_lz = max_lz;
                }
                has_bounds[mi] = true;
            }
        }

        // Propagate bounds bottom-up from deepest children to root parents so parent boxes strictly enclose children!
        uint32_t max_depth = 0u;
        for (const auto& m : manifolds) max_depth = std::max(max_depth, m.depth);
        for (int d = static_cast<int>(max_depth); d >= 1; --d) {
            for (size_t ci = 0; ci < manifolds.size(); ++ci) {
                if (static_cast<int>(manifolds[ci].depth) != d || !has_bounds[ci]) continue;
                uint32_t pid = manifolds[ci].parent_manifold_id;
                if (pid == kInvalidId) continue;
                for (size_t pi = 0; pi < manifolds.size(); ++pi) {
                    if (manifolds[pi].manifold_id != pid) continue;
                    const auto& ch = manifolds[ci];
                    auto& pr = manifolds[pi];
                    uint32_t pad = (pr.envelope_mode == 1u) ? 6u : 0u;
                    uint32_t c_minx = (ch.min_lx > pad) ? (ch.min_lx - pad) : 0u;
                    uint32_t c_miny = (ch.min_ly > pad) ? (ch.min_ly - pad) : 0u;
                    uint32_t c_minz = (ch.min_lz > pad) ? (ch.min_lz - pad) : 0u;
                    uint32_t c_maxx = std::min(lattice_grid - 1u, ch.max_lx + pad);
                    uint32_t c_maxy = std::min(lattice_grid - 1u, ch.max_ly + pad);
                    uint32_t c_maxz = std::min(lattice_grid - 1u, ch.max_lz + pad);
                    if (!has_bounds[pi]) {
                        pr.min_lx = c_minx; pr.max_lx = c_maxx;
                        pr.min_ly = c_miny; pr.max_ly = c_maxy;
                        pr.min_lz = c_minz; pr.max_lz = c_maxz;
                        has_bounds[pi] = true;
                    } else {
                        pr.min_lx = std::min(pr.min_lx, c_minx); pr.max_lx = std::max(pr.max_lx, c_maxx);
                        pr.min_ly = std::min(pr.min_ly, c_miny); pr.max_ly = std::max(pr.max_ly, c_maxy);
                        pr.min_lz = std::min(pr.min_lz, c_minz); pr.max_lz = std::max(pr.max_lz, c_maxz);
                    }
                    break;
                }
            }
        }

        for (size_t mi = 0; mi < manifolds.size(); ++mi) {
            auto& m = manifolds[mi];
            if (!has_bounds[mi]) {
                m.center_x = (m.anchor_x + 0.5f) * step - 1.0f;
                m.center_y = (m.anchor_y + 0.5f) * step - 1.0f;
                m.center_z = (m.anchor_z + 0.5f) * step - 1.0f;
                m.max_radius = 0.0f;
                m.volume_fraction = 0.0f;
                std::fill(std::begin(m.deformed_radii), std::end(m.deformed_radii), 0.0f);
                continue;
            }

            if (m.envelope_mode == 1u) {
                // Exact Minimum-Volume 3D Bounding Box (AABB) enclosing 100% of all interior pixels & child boxes!
                m.min_x = static_cast<float>(m.min_lx) * step - 1.0f;
                m.min_y = static_cast<float>(m.min_ly) * step - 1.0f;
                m.min_z = static_cast<float>(m.min_lz) * step - 1.0f;
                m.max_x = static_cast<float>(m.max_lx + 1u) * step - 1.0f;
                m.max_y = static_cast<float>(m.max_ly + 1u) * step - 1.0f;
                m.max_z = static_cast<float>(m.max_lz + 1u) * step - 1.0f;
                m.center_x = 0.5f * (m.min_x + m.max_x);
                m.center_y = 0.5f * (m.min_y + m.max_y);
                m.center_z = 0.5f * (m.min_z + m.max_z);
                float hx = 0.5f * (m.max_x - m.min_x);
                float hy = 0.5f * (m.max_y - m.min_y);
                float hz = 0.5f * (m.max_z - m.min_z);
                m.max_radius = std::sqrt(hx * hx + hy * hy + hz * hz);
                double vol = static_cast<double>(m.max_x - m.min_x) *
                             static_cast<double>(m.max_y - m.min_y) *
                             static_cast<double>(m.max_z - m.min_z);
                m.volume_fraction = static_cast<float>(vol / kUnitSphereVol);
                std::fill(std::begin(m.deformed_radii), std::end(m.deformed_radii), m.max_radius);
                continue;
            }

            // Legacy mode 0: Geodesic deformed-sphere envelope (used by fundamental_text)
            struct Pt3 { float x, y, z; };
            std::vector<Pt3> corners;
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (const auto& c : classes) {
                if (c.manifold_id != m.manifold_id) continue;
                uint32_t x0 = c.ox, x1 = c.ox + c.width - 1u;
                uint32_t y0 = c.oy, y1 = c.oy + c.height - 1u;
                uint32_t z0 = c.oz;
                float wx[2] = {(x0 + 0.5f) * step - 1.0f, (x1 + 0.5f) * step - 1.0f};
                float wy[2] = {(y0 + 0.5f) * step - 1.0f, (y1 + 0.5f) * step - 1.0f};
                float wz    = (z0 + 0.5f) * step - 1.0f;
                for (int ix = 0; ix < 2; ++ix) {
                    for (int iy = 0; iy < 2; ++iy) {
                        corners.push_back({wx[ix], wy[iy], wz});
                        sx += wx[ix]; sy += wy[iy]; sz += wz;
                    }
                }
            }
            m.min_x = (m.min_lx + 0.5f) * step - 1.0f;
            m.min_y = (m.min_ly + 0.5f) * step - 1.0f;
            m.min_z = (m.min_lz + 0.5f) * step - 1.0f;
            m.max_x = (m.max_lx + 0.5f) * step - 1.0f;
            m.max_y = (m.max_ly + 0.5f) * step - 1.0f;
            m.max_z = (m.max_lz + 0.5f) * step - 1.0f;
            m.center_x = static_cast<float>(sx / std::max<size_t>(1, corners.size()));
            m.center_y = static_cast<float>(sy / std::max<size_t>(1, corners.size()));
            m.center_z = static_cast<float>(sz / std::max<size_t>(1, corners.size()));

            std::vector<float> cux(corners.size()), cuy(corners.size()), cuz(corners.size()), cr(corners.size());
            float skin = step * 3.5f;
            for (size_t k = 0; k < corners.size(); ++k) {
                float dx = corners[k].x - m.center_x;
                float dy = corners[k].y - m.center_y;
                float dz = corners[k].z - m.center_z;
                float r  = std::sqrt(dx * dx + dy * dy + dz * dz);
                cr[k] = r;
                if (r > 1e-7f) { cux[k] = dx / r; cuy[k] = dy / r; cuz[k] = dz / r; }
                else { cux[k] = 0.0f; cuy[k] = 0.0f; cuz[k] = 1.0f; }
            }

            float raw_r[kManifoldVerts];
            for (uint32_t v = 0; v < kManifoldVerts; ++v) {
                float vx = geo.dirs[v * 3 + 0];
                float vy = geo.dirs[v * 3 + 1];
                float vz = geo.dirs[v * 3 + 2];
                float best = skin;
                for (size_t k = 0; k < corners.size(); ++k) {
                    float d = vx * cux[k] + vy * cuy[k] + vz * cuz[k];
                    if (d > 0.0f) {
                        float val = cr[k] * d * d * d + skin;
                        if (val > best) best = val;
                    }
                }
                raw_r[v] = best;
            }

            float sm_r[kManifoldVerts];
            for (int pass = 0; pass < 2; ++pass) {
                for (uint32_t v = 0; v < kManifoldVerts; ++v) {
                    float acc = 0.0f;
                    for (uint32_t nb : geo.adj[v]) acc += raw_r[nb];
                    float nb_avg = geo.adj[v].empty() ? raw_r[v] : (acc / geo.adj[v].size());
                    sm_r[v] = 0.65f * raw_r[v] + 0.35f * nb_avg;
                }
                std::memcpy(raw_r, sm_r, sizeof(raw_r));
            }

            for (size_t k = 0; k < corners.size(); ++k) {
                float need = cr[k] + skin;
                int best_t = 0;
                float best_dot = -2.0f;
                for (size_t t = 0; t < geo.tris.size(); t += 3) {
                    uint32_t a = geo.tris[t], b = geo.tris[t + 1], c = geo.tris[t + 2];
                    float tx = geo.dirs[a * 3 + 0] + geo.dirs[b * 3 + 0] + geo.dirs[c * 3 + 0];
                    float ty = geo.dirs[a * 3 + 1] + geo.dirs[b * 3 + 1] + geo.dirs[c * 3 + 1];
                    float tz = geo.dirs[a * 3 + 2] + geo.dirs[b * 3 + 2] + geo.dirs[c * 3 + 2];
                    float tl = std::sqrt(tx * tx + ty * ty + tz * tz);
                    float dp = (tx * cux[k] + ty * cuy[k] + tz * cuz[k]) / std::max(tl, 1e-7f);
                    if (dp > best_dot) { best_dot = dp; best_t = static_cast<int>(t); }
                }
                for (int q = 0; q < 3; ++q) {
                    uint32_t vi = geo.tris[best_t + q];
                    if (raw_r[vi] < need) raw_r[vi] = need;
                }
            }

            float mr = 0.0f;
            for (uint32_t v = 0; v < kManifoldVerts; ++v) {
                m.deformed_radii[v] = raw_r[v];
                if (raw_r[v] > mr) mr = raw_r[v];
            }
            m.max_radius = mr;

            double vol = 0.0;
            for (size_t t = 0; t < geo.tris.size(); t += 3) {
                uint32_t a = geo.tris[t], b = geo.tris[t + 1], c = geo.tris[t + 2];
                float ax = geo.dirs[a * 3 + 0] * raw_r[a], ay = geo.dirs[a * 3 + 1] * raw_r[a], az = geo.dirs[a * 3 + 2] * raw_r[a];
                float bx = geo.dirs[b * 3 + 0] * raw_r[b], by = geo.dirs[b * 3 + 1] * raw_r[b], bz = geo.dirs[b * 3 + 2] * raw_r[b];
                float cxv = geo.dirs[c * 3 + 0] * raw_r[c], cyv = geo.dirs[c * 3 + 1] * raw_r[c], czv = geo.dirs[c * 3 + 2] * raw_r[c];
                double det = ax * (by * czv - bz * cyv) - ay * (bx * czv - bz * cxv) + az * (bx * cyv - by * cxv);
                vol += std::abs(det) / 6.0;
            }
            m.volume_fraction = static_cast<float>(vol / kUnitSphereVol);
        }
    }

    // Build GPU-resident 144-byte GpuManifoldDesc metadata table for direct upload to CUDA VRAM!
    std::vector<GpuManifoldDesc> build_gpu_manifold_table() const {
        std::vector<GpuManifoldDesc> out;
        out.reserve(manifolds.size());
        for (const auto& m : manifolds) {
            GpuManifoldDesc d{};
            d.manifold_id        = m.manifold_id;
            d.parent_manifold_id = m.parent_manifold_id;
            d.center_x           = m.center_x;
            d.center_y           = m.center_y;
            d.center_z           = m.center_z;
            d.max_radius         = m.max_radius;
            d.min_x              = m.min_x;
            d.min_y              = m.min_y;
            d.min_z              = m.min_z;
            d.volume_fraction    = m.volume_fraction;
            d.max_x              = m.max_x;
            d.max_y              = m.max_y;
            d.max_z              = m.max_z;
            d.outer_idx          = m.outer_idx;
            std::strncpy(d.name, m.name.c_str(), sizeof(d.name) - 1);
            std::strncpy(d.semantic_label, m.semantic_label.c_str(), sizeof(d.semantic_label) - 1);
            std::map<uint32_t, bool> seen_cls;
            uint32_t reg_cnt = 0;
            for (const auto& c : classes) {
                if (c.manifold_id == m.manifold_id) {
                    seen_cls[c.outer_idx] = true;
                    reg_cnt++;
                }
            }
            d.class_count  = static_cast<uint32_t>(seen_cls.size());
            d.region_count = reg_cnt;
            out.push_back(d);
        }
        return out;
    }

    // Allocate a brand-new guaranteed non-overlapping sub-shadow W x H x 1 space inside the
    // target Operational Manifold while sharing the class's single outer-shell point!
    int allocate_subshadow(const std::string& label, uint32_t w, uint32_t h, uint32_t manifold_id = 0u) {
        if (manifolds.empty()) {
            find_or_create_manifold("fundamental_text", "handwritten_letters_and_numbers");
        }
        uint32_t best_ox = 0, best_oy = 0, best_oz = 0;
        if (!find_free_lattice_slot(w, h, manifold_id, best_ox, best_oy, best_oz)) {
            std::cerr << "[ERROR] Sphere lattice (grid=" << lattice_grid
                      << ") has no remaining non-overlapping space for " << w << "x" << h << "\n";
            std::exit(1);
        }

        int existing_outer = find_outer_idx_for_label(label);
        uint32_t oidx = (existing_outer >= 0) ? static_cast<uint32_t>(existing_outer) : next_outer_idx();

        ClassRecord rec;
        rec.label       = label;
        rec.ox          = best_ox;
        rec.oy          = best_oy;
        rec.oz          = best_oz;
        rec.width       = w;
        rec.height      = h;
        rec.outer_idx   = oidx;
        rec.manifold_id = manifold_id;
        assign_outer_point(rec.outer_idx, rec.outer_x, rec.outer_y, rec.outer_z);
        rec.train_count = 0;
        rec.shadow_sum.assign(static_cast<size_t>(w) * h, 0.0f);
        classes.push_back(std::move(rec));
        return static_cast<int>(classes.size()) - 1;
    }

    // Dynamically allocate a guaranteed non-overlapping W x H x 1 space inside the target manifold
    int register_or_get_class(const std::string& label, uint32_t w, uint32_t h, uint32_t manifold_id = 0u) {
        int existing = find_class(label);
        if (existing >= 0) {
            if (classes[existing].width != w || classes[existing].height != h) {
                std::cerr << "[ERROR] Class '" << label << "' already exists with size "
                          << classes[existing].width << "x" << classes[existing].height
                          << " (cannot mix with " << w << "x" << h << ")\n";
                std::exit(1);
            }
            return existing;
        }
        return allocate_subshadow(label, w, h, manifold_id);
    }

    bool save(const std::string& path) const {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.good()) return false;

        uint32_t num_classes   = static_cast<uint32_t>(classes.size());
        uint32_t num_manifolds = static_cast<uint32_t>(manifolds.size());
        ofs.write(reinterpret_cast<const char*>(&kModelMagic), 4);
        ofs.write(reinterpret_cast<const char*>(&kModelVersion), 4);
        ofs.write(reinterpret_cast<const char*>(&lattice_grid), 4);
        ofs.write(reinterpret_cast<const char*>(&outer_shell_pts), 4);
        ofs.write(reinterpret_cast<const char*>(&num_classes), 4);
        ofs.write(reinterpret_cast<const char*>(&num_manifolds), 4);

        for (const auto& m : manifolds) {
            char mname[32] = {};
            char msem[48]  = {};
            char mcon[32]  = {};
            std::strncpy(mname, m.name.c_str(), 31);
            std::strncpy(msem, m.semantic_label.c_str(), 47);
            std::strncpy(mcon, m.concept_label.c_str(), 31);
            ofs.write(reinterpret_cast<const char*>(&m.manifold_id), 4);
            ofs.write(mname, 32);
            ofs.write(msem, 48);
            ofs.write(reinterpret_cast<const char*>(&m.anchor_x), 4);
            ofs.write(reinterpret_cast<const char*>(&m.anchor_y), 4);
            ofs.write(reinterpret_cast<const char*>(&m.anchor_z), 4);
            ofs.write(reinterpret_cast<const char*>(&m.center_x), 4);
            ofs.write(reinterpret_cast<const char*>(&m.center_y), 4);
            ofs.write(reinterpret_cast<const char*>(&m.center_z), 4);
            ofs.write(reinterpret_cast<const char*>(&m.min_x), 4);
            ofs.write(reinterpret_cast<const char*>(&m.min_y), 4);
            ofs.write(reinterpret_cast<const char*>(&m.min_z), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_x), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_y), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_z), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_radius), 4);
            ofs.write(reinterpret_cast<const char*>(&m.volume_fraction), 4);
            ofs.write(reinterpret_cast<const char*>(&m.min_lx), 4);
            ofs.write(reinterpret_cast<const char*>(&m.min_ly), 4);
            ofs.write(reinterpret_cast<const char*>(&m.min_lz), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_lx), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_ly), 4);
            ofs.write(reinterpret_cast<const char*>(&m.max_lz), 4);
            ofs.write(reinterpret_cast<const char*>(m.deformed_radii), sizeof(m.deformed_radii));
            // v4 hierarchical metadata & outer-shell concept link
            ofs.write(reinterpret_cast<const char*>(&m.parent_manifold_id), 4);
            ofs.write(reinterpret_cast<const char*>(&m.depth), 4);
            ofs.write(reinterpret_cast<const char*>(&m.envelope_mode), 4);
            ofs.write(reinterpret_cast<const char*>(&m.outer_idx), 4);
            ofs.write(reinterpret_cast<const char*>(&m.outer_x), 4);
            ofs.write(reinterpret_cast<const char*>(&m.outer_y), 4);
            ofs.write(reinterpret_cast<const char*>(&m.outer_z), 4);
            ofs.write(mcon, 32);
        }

        for (const auto& c : classes) {
            char lbl[32] = {};
            std::strncpy(lbl, c.label.c_str(), 31);
            ofs.write(lbl, 32);
            ofs.write(reinterpret_cast<const char*>(&c.ox), 4);
            ofs.write(reinterpret_cast<const char*>(&c.oy), 4);
            ofs.write(reinterpret_cast<const char*>(&c.oz), 4);
            ofs.write(reinterpret_cast<const char*>(&c.width), 4);
            ofs.write(reinterpret_cast<const char*>(&c.height), 4);
            ofs.write(reinterpret_cast<const char*>(&c.outer_idx), 4);
            ofs.write(reinterpret_cast<const char*>(&c.outer_x), 4);
            ofs.write(reinterpret_cast<const char*>(&c.outer_y), 4);
            ofs.write(reinterpret_cast<const char*>(&c.outer_z), 4);
            ofs.write(reinterpret_cast<const char*>(&c.train_count), 4);
            ofs.write(reinterpret_cast<const char*>(&c.manifold_id), 4);
            uint32_t ch = (c.channels == 3u && c.shadow_rgb.size() == c.shadow_sum.size()) ? 3u : 1u;
            ofs.write(reinterpret_cast<const char*>(&ch), 4);
            ofs.write(reinterpret_cast<const char*>(c.shadow_sum.data()),
                      static_cast<std::streamsize>(c.shadow_sum.size() * sizeof(float)));
            if (ch == 3u) {
                ofs.write(reinterpret_cast<const char*>(c.shadow_rgb.data()),
                          static_cast<std::streamsize>(c.shadow_rgb.size() * sizeof(uint32_t)));
            }
        }
        return ofs.good();
    }

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good()) return false;

        uint32_t magic = 0, ver = 0, num_classes = 0, num_manifolds = 0;
        ifs.read(reinterpret_cast<char*>(&magic), 4);
        ifs.read(reinterpret_cast<char*>(&ver), 4);
        if (magic != kModelMagic || ver < 1u || ver > kModelVersion) return false;

        ifs.read(reinterpret_cast<char*>(&lattice_grid), 4);
        ifs.read(reinterpret_cast<char*>(&outer_shell_pts), 4);
        ifs.read(reinterpret_cast<char*>(&num_classes), 4);

        manifolds.clear();
        if (ver >= 2u) {
            ifs.read(reinterpret_cast<char*>(&num_manifolds), 4);
            manifolds.reserve(num_manifolds);
            for (uint32_t i = 0; i < num_manifolds; ++i) {
                ManifoldRecord m;
                char mname[32] = {};
                char msem[48]  = {};
                ifs.read(reinterpret_cast<char*>(&m.manifold_id), 4);
                ifs.read(mname, 32);
                ifs.read(msem, 48);
                m.name = mname;
                m.semantic_label = msem;
                ifs.read(reinterpret_cast<char*>(&m.anchor_x), 4);
                ifs.read(reinterpret_cast<char*>(&m.anchor_y), 4);
                ifs.read(reinterpret_cast<char*>(&m.anchor_z), 4);
                ifs.read(reinterpret_cast<char*>(&m.center_x), 4);
                ifs.read(reinterpret_cast<char*>(&m.center_y), 4);
                ifs.read(reinterpret_cast<char*>(&m.center_z), 4);
                ifs.read(reinterpret_cast<char*>(&m.min_x), 4);
                ifs.read(reinterpret_cast<char*>(&m.min_y), 4);
                ifs.read(reinterpret_cast<char*>(&m.min_z), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_x), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_y), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_z), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_radius), 4);
                ifs.read(reinterpret_cast<char*>(&m.volume_fraction), 4);
                ifs.read(reinterpret_cast<char*>(&m.min_lx), 4);
                ifs.read(reinterpret_cast<char*>(&m.min_ly), 4);
                ifs.read(reinterpret_cast<char*>(&m.min_lz), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_lx), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_ly), 4);
                ifs.read(reinterpret_cast<char*>(&m.max_lz), 4);
                ifs.read(reinterpret_cast<char*>(m.deformed_radii), sizeof(m.deformed_radii));
                if (ver >= 4u) {
                    char mcon[32] = {};
                    ifs.read(reinterpret_cast<char*>(&m.parent_manifold_id), 4);
                    ifs.read(reinterpret_cast<char*>(&m.depth), 4);
                    ifs.read(reinterpret_cast<char*>(&m.envelope_mode), 4);
                    ifs.read(reinterpret_cast<char*>(&m.outer_idx), 4);
                    ifs.read(reinterpret_cast<char*>(&m.outer_x), 4);
                    ifs.read(reinterpret_cast<char*>(&m.outer_y), 4);
                    ifs.read(reinterpret_cast<char*>(&m.outer_z), 4);
                    ifs.read(mcon, 32);
                    m.concept_label = mcon;
                } else {
                    m.parent_manifold_id = kInvalidId;
                    m.depth = 0u;
                    m.envelope_mode = 0u;
                    m.outer_idx = kInvalidId;
                    m.concept_label = m.name;
                }
                manifolds.push_back(std::move(m));
            }
        }

        classes.clear();
        classes.reserve(num_classes);
        for (uint32_t i = 0; i < num_classes; ++i) {
            ClassRecord c;
            char lbl[32] = {};
            ifs.read(lbl, 32);
            c.label = lbl;
            ifs.read(reinterpret_cast<char*>(&c.ox), 4);
            ifs.read(reinterpret_cast<char*>(&c.oy), 4);
            ifs.read(reinterpret_cast<char*>(&c.oz), 4);
            ifs.read(reinterpret_cast<char*>(&c.width), 4);
            ifs.read(reinterpret_cast<char*>(&c.height), 4);
            ifs.read(reinterpret_cast<char*>(&c.outer_idx), 4);
            ifs.read(reinterpret_cast<char*>(&c.outer_x), 4);
            ifs.read(reinterpret_cast<char*>(&c.outer_y), 4);
            ifs.read(reinterpret_cast<char*>(&c.outer_z), 4);
            ifs.read(reinterpret_cast<char*>(&c.train_count), 4);
            if (ver >= 2u) {
                ifs.read(reinterpret_cast<char*>(&c.manifold_id), 4);
            } else {
                c.manifold_id = 0u;
            }
            if (ver >= 3u) {
                ifs.read(reinterpret_cast<char*>(&c.channels), 4);
            } else {
                c.channels = 1u;
            }
            c.shadow_sum.resize(static_cast<size_t>(c.width) * c.height);
            ifs.read(reinterpret_cast<char*>(c.shadow_sum.data()),
                     static_cast<std::streamsize>(c.shadow_sum.size() * sizeof(float)));
            if (ver >= 3u && c.channels == 3u) {
                c.shadow_rgb.resize(static_cast<size_t>(c.width) * c.height);
                ifs.read(reinterpret_cast<char*>(c.shadow_rgb.data()),
                         static_cast<std::streamsize>(c.shadow_rgb.size() * sizeof(uint32_t)));
            } else {
                c.channels = 1u;
                c.shadow_rgb.clear();
            }
            classes.push_back(std::move(c));
        }

        if (ver == 1u && !classes.empty()) {
            recompute_manifold_envelopes();
        }
        return ifs.good();
    }
};

// -----------------------------------------------------------------------------
// Generic Dataset Container (supports IDX .gz AND folder-of-PGM/PPM datasets!)
// -----------------------------------------------------------------------------
struct LabeledImageBatch {
    uint32_t width = 0, height = 0;
    uint32_t count = 0;
    std::vector<uint8_t> pixels;         // count * width * height (8-bit grayscale)
    std::vector<std::string> labels;     // count string labels (e.g. "0".."9", "A".."Z")
    std::vector<std::string> unique_labels;
};

inline uint32_t read_gz_be32(gzFile f) {
    uint8_t b[4] = {};
    gzread(f, b, 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
}

// Load any MNIST / EMNIST / Fashion-MNIST IDX gzipped dataset
// Optional label_names (e.g. "A,B,C,...,Z,a,b,...,z"), label_offset,
// min_raw_label / max_raw_label filter, and transpose=true for EMNIST!
inline LabeledImageBatch load_idx_dataset(const std::string& img_path,
                                          const std::string& lbl_path,
                                          const std::vector<std::string>& label_names = {},
                                          int label_offset = 0,
                                          bool transpose = false,
                                          int min_raw_label = -1,
                                          int max_raw_label = -1) {
    gzFile f_img = gzopen(img_path.c_str(), "rb");
    gzFile f_lbl = gzopen(lbl_path.c_str(), "rb");
    if (!f_img || !f_lbl) {
        std::cerr << "[ERROR] Cannot open IDX files:\n  " << img_path << "\n  " << lbl_path << "\n";
        std::exit(1);
    }
    uint32_t magic_i = read_gz_be32(f_img), n_i = read_gz_be32(f_img),
             rows = read_gz_be32(f_img), cols = read_gz_be32(f_img);
    uint32_t magic_l = read_gz_be32(f_lbl), n_l = read_gz_be32(f_lbl);
    if (magic_i != 2051 || magic_l != 2049 || n_i != n_l) {
        std::cerr << "[ERROR] Invalid IDX header (magic_i=" << magic_i
                  << ", magic_l=" << magic_l << ")\n";
        std::exit(1);
    }

    std::vector<uint8_t> raw_lbls(n_l);
    gzread(f_lbl, raw_lbls.data(), n_l);
    gzclose(f_lbl);

    const size_t img_bytes = static_cast<size_t>(rows) * cols;
    std::vector<uint8_t> raw_pixels(static_cast<size_t>(n_i) * img_bytes);
    size_t total = raw_pixels.size();
    size_t off = 0;
    while (off < total) {
        int chunk = static_cast<int>(std::min<size_t>(1 << 22, total - off));
        int got = gzread(f_img, raw_pixels.data() + off, chunk);
        if (got <= 0) break;
        off += static_cast<size_t>(got);
    }
    gzclose(f_img);

    LabeledImageBatch out;
    out.width  = cols;
    out.height = rows;

    // Count matching images first to reserve cleanly
    uint32_t kept = 0;
    for (uint32_t i = 0; i < n_i; ++i) {
        int r = static_cast<int>(raw_lbls[i]);
        if (min_raw_label >= 0 && r < min_raw_label) continue;
        if (max_raw_label >= 0 && r > max_raw_label) continue;
        kept++;
    }
    out.count = kept;
    out.pixels.resize(static_cast<size_t>(kept) * img_bytes);
    out.labels.resize(kept);

    // Preserve numeric raw-label order for unique_labels (e.g. A..Z then a..z)
    std::map<int, std::string> ordered_unique;
    uint32_t dst_i = 0;
    for (uint32_t i = 0; i < n_i; ++i) {
        int r = static_cast<int>(raw_lbls[i]);
        if (min_raw_label >= 0 && r < min_raw_label) continue;
        if (max_raw_label >= 0 && r > max_raw_label) continue;

        const uint8_t* src = &raw_pixels[static_cast<size_t>(i) * img_bytes];
        uint8_t*       dst = &out.pixels[static_cast<size_t>(dst_i) * img_bytes];
        if (transpose && rows == cols) {
            for (uint32_t y = 0; y < rows; ++y)
                for (uint32_t x = 0; x < cols; ++x)
                    dst[y * cols + x] = src[x * rows + y];
        } else {
            std::memcpy(dst, src, img_bytes);
        }

        int idx = r - label_offset;
        std::string name;
        if (!label_names.empty() && idx >= 0 && idx < static_cast<int>(label_names.size())) {
            name = label_names[static_cast<size_t>(idx)];
        } else {
            name = std::to_string(idx);
        }
        out.labels[dst_i] = name;
        ordered_unique[r] = name;
        dst_i++;
    }
    for (const auto& kv : ordered_unique) {
        out.unique_labels.push_back(kv.second);
    }
    return out;
}

// Read binary PGM (P5) or PPM (P6 -> luminance) image file
inline bool read_pnm_grayscale(const std::string& path,
                               uint32_t& w, uint32_t& h,
                               std::vector<uint8_t>& out_gray) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) return false;
    std::string magic;
    ifs >> magic;
    if (magic != "P5" && magic != "P6") return false;

    auto skip_ws_comments = [&]() {
        while (ifs.good()) {
            int c = ifs.peek();
            if (std::isspace(c)) { ifs.get(); continue; }
            if (c == '#') { std::string line; std::getline(ifs, line); continue; }
            break;
        }
    };
    int maxv = 0;
    skip_ws_comments(); ifs >> w;
    skip_ws_comments(); ifs >> h;
    skip_ws_comments(); ifs >> maxv;
    ifs.get(); // single whitespace byte after maxv
    if (w == 0 || h == 0 || maxv != 255) return false;

    out_gray.resize(static_cast<size_t>(w) * h);
    if (magic == "P5") {
        ifs.read(reinterpret_cast<char*>(out_gray.data()), static_cast<std::streamsize>(out_gray.size()));
    } else {
        std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
        ifs.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        for (size_t i = 0; i < out_gray.size(); ++i) {
            out_gray[i] = static_cast<uint8_t>(
                0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2]
            );
        }
    }
    return ifs.good();
}

// Load any folder-based dataset: <root>/<class_label>/*.pgm (or *.ppm)
inline LabeledImageBatch load_folder_dataset(const std::string& root_dir) {
    namespace fs = std::filesystem;
    LabeledImageBatch out;
    std::vector<fs::path> class_dirs;
    for (const auto& entry : fs::directory_iterator(root_dir)) {
        if (entry.is_directory()) class_dirs.push_back(entry.path());
    }
    std::sort(class_dirs.begin(), class_dirs.end());

    for (const auto& dir : class_dirs) {
        std::string lbl = dir.filename().string();
        std::vector<fs::path> files;
        for (const auto& f : fs::directory_iterator(dir)) {
            if (!f.is_regular_file()) continue;
            auto ext = f.path().extension().string();
            if (ext == ".pgm" || ext == ".ppm") files.push_back(f.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) continue;
        out.unique_labels.push_back(lbl);

        for (const auto& f : files) {
            uint32_t w = 0, h = 0;
            std::vector<uint8_t> gray;
            if (!read_pnm_grayscale(f.string(), w, h, gray)) continue;
            if (out.width == 0) { out.width = w; out.height = h; }
            if (w != out.width || h != out.height) continue;
            out.pixels.insert(out.pixels.end(), gray.begin(), gray.end());
            out.labels.push_back(lbl);
            out.count++;
        }
    }
    return out;
}

} // namespace gllm
