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
constexpr uint32_t kModelVersion = 1u;

// GPU-compatible 32-byte region descriptor (matches kernels/gllm_kernels.ptx)
struct alignas(16) GpuRegionDesc {
    uint32_t ox, oy, oz, width;
    uint32_t height, outer_idx, shadow_offset, class_id;
};

// GPU-compatible 16-byte outer-shell point descriptor
struct alignas(16) GpuOuterPoint {
    float x, y, z, value;
};

struct ClassRecord {
    std::string label;
    uint32_t ox = 0, oy = 0, oz = 0;   // Unique non-overlapping origin in G^3 lattice
    uint32_t width = 28, height = 28;  // Exact 1-to-1 pixel dimensions
    uint32_t outer_idx = 0;            // Assigned point index on the 1.2M outer shell
    float outer_x = 0.0f, outer_y = 0.0f, outer_z = 1.0f;
    uint32_t train_count = 0;          // Total shadows accumulated into this region
    std::vector<float> shadow_sum;     // width * height raw accumulated shadow totals
};

struct GeometricModel {
    uint32_t lattice_grid    = 1241;    // 1241^3 sphere -> ~1,000,722,000 implicit lattice points!
    uint32_t outer_shell_pts = 1200000; // Permanent 1.2M outer spherical shell capacity
    std::vector<ClassRecord> classes;

    int find_class(const std::string& label) const {
        for (size_t i = 0; i < classes.size(); ++i) {
            if (classes[i].label == label) return static_cast<int>(i);
        }
        return -1;
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

    // Dynamically allocate a guaranteed non-overlapping W x H x 1 space inside the sphere
    // and bind it to the next available neighboring outer-shell point!
    int register_or_get_class(const std::string& label, uint32_t w, uint32_t h) {
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

        uint32_t pad_xy = std::max<uint32_t>(4u, w / 4u);
        uint32_t pad_z  = std::max<uint32_t>(4u, std::min<uint32_t>(14u, lattice_grid / 80u));
        uint32_t stride_x = w + pad_xy;
        uint32_t stride_y = h + pad_xy;
        uint32_t stride_z = pad_z + 1u;

        int64_t mid = static_cast<int64_t>(lattice_grid / 2);
        bool placed = false;
        uint32_t best_ox = 0, best_oy = 0, best_oz = 0;

        // Expanding 3D shell search around sphere center -> naturally packs classes
        // tightly near the core first without ever overlapping prior allocations!
        for (int shell = 0; shell <= 64 && !placed; ++shell) {
            int max_dz = std::min(shell, 16);
            for (int dz = 0; dz <= max_dz && !placed; ++dz) {
                int z_signs[2] = {dz, -dz};
                int z_cnt = (dz == 0) ? 1 : 2;
                for (int zs = 0; zs < z_cnt && !placed; ++zs) {
                    int64_t oz = mid + static_cast<int64_t>(z_signs[zs]) * stride_z;
                    for (int dy = shell; dy >= -shell && !placed; --dy) {
                        for (int dx = -shell; dx <= shell && !placed; ++dx) {
                            if (std::max({std::abs(dx), std::abs(dy), dz}) != shell) continue;
                            int64_t ox = mid + static_cast<int64_t>(dx) * stride_x - static_cast<int64_t>(w / 2);
                            int64_t oy = mid + static_cast<int64_t>(dy) * stride_y - static_cast<int64_t>(h / 2);
                            if (!inside_sphere(ox, oy, oz, w, h, 0.90f)) continue;
                            if (!overlaps_any(static_cast<uint32_t>(ox),
                                              static_cast<uint32_t>(oy),
                                              static_cast<uint32_t>(oz),
                                              w, h, pad_xy, pad_z)) {
                                best_ox = static_cast<uint32_t>(ox);
                                best_oy = static_cast<uint32_t>(oy);
                                best_oz = static_cast<uint32_t>(oz);
                                placed = true;
                            }
                        }
                    }
                }
            }
        }

        if (!placed) {
            std::cerr << "[ERROR] Sphere lattice (grid=" << lattice_grid
                      << ") has no remaining non-overlapping space for " << w << "x" << h << "\n";
            std::exit(1);
        }

        ClassRecord rec;
        rec.label     = label;
        rec.ox        = best_ox;
        rec.oy        = best_oy;
        rec.oz        = best_oz;
        rec.width     = w;
        rec.height    = h;
        rec.outer_idx = static_cast<uint32_t>(classes.size());
        assign_outer_point(rec.outer_idx, rec.outer_x, rec.outer_y, rec.outer_z);
        rec.train_count = 0;
        rec.shadow_sum.assign(static_cast<size_t>(w) * h, 0.0f);
        classes.push_back(std::move(rec));
        return static_cast<int>(classes.size()) - 1;
    }

    bool save(const std::string& path) const {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.good()) return false;

        uint32_t num_classes = static_cast<uint32_t>(classes.size());
        ofs.write(reinterpret_cast<const char*>(&kModelMagic), 4);
        ofs.write(reinterpret_cast<const char*>(&kModelVersion), 4);
        ofs.write(reinterpret_cast<const char*>(&lattice_grid), 4);
        ofs.write(reinterpret_cast<const char*>(&outer_shell_pts), 4);
        ofs.write(reinterpret_cast<const char*>(&num_classes), 4);

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
            ofs.write(reinterpret_cast<const char*>(c.shadow_sum.data()),
                      static_cast<std::streamsize>(c.shadow_sum.size() * sizeof(float)));
        }
        return ofs.good();
    }

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good()) return false;

        uint32_t magic = 0, ver = 0, num_classes = 0;
        ifs.read(reinterpret_cast<char*>(&magic), 4);
        ifs.read(reinterpret_cast<char*>(&ver), 4);
        if (magic != kModelMagic || ver != kModelVersion) return false;

        ifs.read(reinterpret_cast<char*>(&lattice_grid), 4);
        ifs.read(reinterpret_cast<char*>(&outer_shell_pts), 4);
        ifs.read(reinterpret_cast<char*>(&num_classes), 4);

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
            c.shadow_sum.resize(static_cast<size_t>(c.width) * c.height);
            ifs.read(reinterpret_cast<char*>(c.shadow_sum.data()),
                     static_cast<std::streamsize>(c.shadow_sum.size() * sizeof(float)));
            classes.push_back(std::move(c));
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
