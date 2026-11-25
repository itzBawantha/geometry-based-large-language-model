// =============================================================================
// gllm_pyramid — Face-Cropped RGB Burt-Adelson Gaussian Pyramid -> 1B Lattice Mapper
//
// Pipeline:
//   1. Decode input image in full RGB24 (PNG/JPG/BMP/PPM/PGM)
//   2. Optional face bounding-box crop (--crop x0,y0,x1,y1) BEFORE Gaussian Pyramid
//   3. Run 5-tap separable binomial Burt-Adelson Gaussian Pyramid ([1,4,6,4,1]/16
//      + 2x area downsample) independently across R, G, and B channels
//   4. Select either all octaves or a single target octave (--select-level 2)
//   5. Map EVERY pixel (x, y) 1-to-1 onto exact 3D lattice coordinates
//      (ox + x, oy + H - 1 - y, oz) inside the 1241^3 (~1B-pt) sphere lattice
//      with true per-point packed RGB (0x01RRGGBB) stored in CUDA VRAM!
// =============================================================================

#include "../include/gaussian_pyramid.hpp"
#include "../include/gllm_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_pyramid --image <path.png|jpg|ppm|pgm> --model <out.gllm>\n"
              << "               [--crop x0,y0,x1,y1] [--levels 5] [--select-level -1]\n"
              << "               [--min-size 16] [--z-step 36] [--grid 1241]\n"
              << "               [--manifold animals_cat_face] [--manifold-label cat_face_rgb_pyramid_L2]\n"
              << "               [--prefix cat_face] [--append]\n";
}

// Decode any image into RGB24 via ffmpeg pipe (or directly if binary P6 PPM / P5 PGM)
static bool load_any_rgb(const std::string& path,
                         uint32_t& out_w, uint32_t& out_h,
                         std::vector<uint8_t>& out_rgb) {
    std::string escaped;
    escaped.reserve(path.size() + 8);
    for (char c : path) {
        if (c == '\'') escaped += "'\\''";
        else escaped += c;
    }
    std::string cmd = "ffmpeg -v error -i '" + escaped + "' -vf format=rgb24 -f image2pipe -vcodec ppm -";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    std::vector<uint8_t> buf;
    uint8_t chunk[65536];
    size_t nread = 0;
    while ((nread = std::fread(chunk, 1, sizeof(chunk), pipe)) > 0) {
        buf.insert(buf.end(), chunk, chunk + nread);
    }
    int rc = pclose(pipe);
    if (rc != 0 || buf.size() < 12) return false;

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

    std::string magic = read_tok();
    if (magic != "P6") return false;
    out_w = static_cast<uint32_t>(std::stoul(read_tok()));
    out_h = static_cast<uint32_t>(std::stoul(read_tok()));
    uint32_t maxv = static_cast<uint32_t>(std::stoul(read_tok()));
    if (pos < buf.size()) pos++; // single whitespace byte after maxv
    size_t need = static_cast<size_t>(out_w) * out_h * 3u;
    if (out_w == 0 || out_h == 0 || maxv != 255 || pos + need > buf.size()) return false;

    out_rgb.assign(buf.begin() + static_cast<std::ptrdiff_t>(pos),
                   buf.begin() + static_cast<std::ptrdiff_t>(pos + need));
    return true;
}

static bool parse_crop(const std::string& s, int& x0, int& y0, int& x1, int& y1) {
    std::stringstream ss(s);
    std::string tok;
    double vals[4] = {};
    int k = 0;
    while (std::getline(ss, tok, ',') && k < 4) {
        vals[k++] = std::atof(tok.c_str());
    }
    if (k != 4) return false;
    x0 = static_cast<int>(std::floor(vals[0]));
    y0 = static_cast<int>(std::floor(vals[1]));
    x1 = static_cast<int>(std::ceil(vals[2]));
    y1 = static_cast<int>(std::ceil(vals[3]));
    return (x1 > x0 && y1 > y0);
}

int main(int argc, char** argv) {
    std::string image_path, model_path, crop_str;
    std::string manifold_name  = "animals_face_pyramid";
    std::string manifold_label = "face_rgb_gaussian_pyramid";
    std::string prefix         = "face";
    uint32_t max_levels  = 5;
    int      select_lv   = -1; // -1 = map all octaves; >= 0 = map ONLY that exact octave (e.g. 2 = Level 2)
    uint32_t min_size    = 16;
    uint32_t z_step      = 36;
    uint32_t grid_dim    = 1241;
    bool     append_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--crop" && i + 1 < argc) crop_str = argv[++i];
        else if (a == "--levels" && i + 1 < argc) max_levels = std::max<uint32_t>(1u, static_cast<uint32_t>(std::atoi(argv[++i])));
        else if (a == "--select-level" && i + 1 < argc) select_lv = std::atoi(argv[++i]);
        else if (a == "--min-size" && i + 1 < argc) min_size = std::max<uint32_t>(8u, static_cast<uint32_t>(std::atoi(argv[++i])));
        else if (a == "--z-step" && i + 1 < argc) z_step = std::max<uint32_t>(3u, static_cast<uint32_t>(std::atoi(argv[++i])));
        else if (a == "--grid" && i + 1 < argc) grid_dim = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--manifold" && i + 1 < argc) manifold_name = argv[++i];
        else if (a == "--manifold-label" && i + 1 < argc) manifold_label = argv[++i];
        else if (a == "--prefix" && i + 1 < argc) prefix = argv[++i];
        else if (a == "--append") append_mode = true;
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    if (image_path.empty() || model_path.empty()) {
        print_usage();
        return 1;
    }

    uint32_t full_w = 0, full_h = 0;
    std::vector<uint8_t> full_rgb;
    if (!load_any_rgb(image_path, full_w, full_h, full_rgb)) {
        std::cerr << "[ERROR] Failed to decode RGB image: " << image_path << "\n";
        return 1;
    }
    std::cout << "[IMAGE]  Loaded RGB24 " << image_path << " (" << full_w << "x" << full_h << ")\n";

    // Stage 1: Optional dataset face bounding-box crop BEFORE Gaussian Pyramid
    int cx0 = 0, cy0 = 0, cx1 = static_cast<int>(full_w), cy1 = static_cast<int>(full_h);
    if (!crop_str.empty()) {
        if (!parse_crop(crop_str, cx0, cy0, cx1, cy1)) {
            std::cerr << "[ERROR] Invalid --crop format (expected x0,y0,x1,y1): " << crop_str << "\n";
            return 1;
        }
        cx0 = std::clamp(cx0, 0, static_cast<int>(full_w) - 1);
        cy0 = std::clamp(cy0, 0, static_cast<int>(full_h) - 1);
        cx1 = std::clamp(cx1, cx0 + 1, static_cast<int>(full_w));
        cy1 = std::clamp(cy1, cy0 + 1, static_cast<int>(full_h));
        std::cout << "[CROP]   Cut face bounding box [" << cx0 << ", " << cy0 << ", " << cx1 << ", " << cy1
                  << "] -> " << (cx1 - cx0) << "x" << (cy1 - cy0) << " RGB face patch\n";
    }

    uint32_t w = static_cast<uint32_t>(cx1 - cx0);
    uint32_t h = static_cast<uint32_t>(cy1 - cy0);
    size_t npix = static_cast<size_t>(w) * h;
    std::vector<float> cur_r(npix), cur_g(npix), cur_b(npix);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t si = (static_cast<size_t>(cy0 + y) * full_w + (cx0 + x)) * 3u;
            size_t di = static_cast<size_t>(y) * w + x;
            cur_r[di] = static_cast<float>(full_rgb[si + 0]) * (1.0f / 255.0f);
            cur_g[di] = static_cast<float>(full_rgb[si + 1]) * (1.0f / 255.0f);
            cur_b[di] = static_cast<float>(full_rgb[si + 2]) * (1.0f / 255.0f);
        }
    }

    // Stage 2: Build RGB Burt-Adelson 5-tap separable binomial Gaussian Pyramid
    struct RgbOctave {
        uint32_t level = 0;
        uint32_t w = 0, h = 0;
        std::vector<float>    lum; // [0..1]
        std::vector<uint32_t> rgb; // packed 0x01RRGGBB
    };
    std::vector<RgbOctave> octaves;
    uint32_t need_levels = (select_lv >= 0) ? std::max(max_levels, static_cast<uint32_t>(select_lv + 1)) : max_levels;
    uint32_t cw = w, ch = h;

    for (uint32_t lv = 0; lv < need_levels; ++lv) {
        size_t n = static_cast<size_t>(cw) * ch;
        RgbOctave oct;
        oct.level = lv;
        oct.w = cw;
        oct.h = ch;
        oct.lum.resize(n);
        oct.rgb.resize(n);
        for (size_t p = 0; p < n; ++p) {
            float rf = std::clamp(cur_r[p], 0.0f, 1.0f);
            float gf = std::clamp(cur_g[p], 0.0f, 1.0f);
            float bf = std::clamp(cur_b[p], 0.0f, 1.0f);
            oct.lum[p] = 0.299f * rf + 0.587f * gf + 0.114f * bf;
            uint32_t rb = static_cast<uint32_t>(std::lround(rf * 255.0f)) & 0xFFu;
            uint32_t gb = static_cast<uint32_t>(std::lround(gf * 255.0f)) & 0xFFu;
            uint32_t bb = static_cast<uint32_t>(std::lround(bf * 255.0f)) & 0xFFu;
            oct.rgb[p] = 0x01000000u | (rb << 16) | (gb << 8) | bb;
        }
        std::printf("  [PYR]  Level %u: %3ux%-3u (%6zu RGB pixels)\n", lv, cw, ch, n);
        octaves.push_back(std::move(oct));

        if (lv + 1 >= need_levels) break;
        if (cw / 2u < min_size || ch / 2u < min_size) break;
        auto br = gllm::gaussian_blur_5tap(cur_r.data(), cw, ch);
        auto bg = gllm::gaussian_blur_5tap(cur_g.data(), cw, ch);
        auto bb = gllm::gaussian_blur_5tap(cur_b.data(), cw, ch);
        uint32_t nw = std::max(1u, cw / 2u);
        uint32_t nh = std::max(1u, ch / 2u);
        cur_r = gllm::area_downsample_2d(br.data(), cw, ch, nw, nh);
        cur_g = gllm::area_downsample_2d(bg.data(), cw, ch, nw, nh);
        cur_b = gllm::area_downsample_2d(bb.data(), cw, ch, nw, nh);
        cw = nw;
        ch = nh;
    }

    // Stage 3: Filter to selected level (if --select-level N was requested)
    std::vector<RgbOctave> selected;
    for (auto& oct : octaves) {
        if (select_lv < 0 || static_cast<int>(oct.level) == select_lv) {
            selected.push_back(std::move(oct));
        }
    }
    if (selected.empty()) {
        std::cerr << "[ERROR] Requested --select-level " << select_lv
                  << " was not generated (image too small for that octave).\n";
        return 1;
    }

    // Stage 4: Map EVERY pixel of selected octave(s) 1-to-1 into the 1241^3 1B sphere lattice
    gllm::GeometricModel model;
    if (append_mode && model.load(model_path)) {
        std::cout << "[MODEL]  Appending to existing model: " << model_path << "\n";
    } else {
        model.lattice_grid = grid_dim;
    }

    uint32_t manifold_id = model.find_or_create_manifold(manifold_name, manifold_label);
    const auto* mrec = model.get_manifold(manifold_id);
    int64_t ax = mrec ? static_cast<int64_t>(mrec->anchor_x) : static_cast<int64_t>(model.lattice_grid / 2u);
    int64_t ay = mrec ? static_cast<int64_t>(mrec->anchor_y) : static_cast<int64_t>(model.lattice_grid / 2u);
    int64_t az = mrec ? static_cast<int64_t>(mrec->anchor_z) : static_cast<int64_t>(model.lattice_grid / 2u);

    int64_t z_span  = static_cast<int64_t>(selected.size() > 1 ? (selected.size() - 1) * z_step : 0);
    int64_t z_start = az - z_span / 2;

    uint64_t total_mapped_pts = 0;
    std::cout << "\n[LATTICE] Mapping " << selected.size()
              << " selected RGB Gaussian Pyramid level(s) 1-to-1 into "
              << model.lattice_grid << "^3 (~" << model.approx_sphere_lattice_points() << " pts) sphere:\n";

    for (size_t i = 0; i < selected.size(); ++i) {
        const auto& oct = selected[i];
        int64_t ox = ax - static_cast<int64_t>(oct.w / 2u);
        int64_t oy = ay - static_cast<int64_t>(oct.h / 2u);
        int64_t oz = z_start + static_cast<int64_t>(i) * static_cast<int64_t>(z_step);

        if (!model.inside_sphere(ox, oy, oz, oct.w, oct.h, 0.90f) ||
            model.overlaps_any(static_cast<uint32_t>(std::max<int64_t>(0, ox)),
                               static_cast<uint32_t>(std::max<int64_t>(0, oy)),
                               static_cast<uint32_t>(std::max<int64_t>(0, oz)),
                               oct.w, oct.h, 1u, 2u)) {
            uint32_t fx = 0, fy = 0, fz = 0;
            if (!model.find_free_lattice_slot(oct.w, oct.h, manifold_id, fx, fy, fz)) {
                std::cerr << "[ERROR] No free non-overlapping slot inside sphere for " << oct.w << "x" << oct.h << "\n";
                return 1;
            }
            ox = fx; oy = fy; oz = fz;
        }

        uint32_t oidx = model.next_outer_idx();
        gllm::ClassRecord rec;
        rec.label       = prefix + "_L" + std::to_string(oct.level) + "_" + std::to_string(oct.w) + "x" + std::to_string(oct.h);
        rec.ox          = static_cast<uint32_t>(ox);
        rec.oy          = static_cast<uint32_t>(oy);
        rec.oz          = static_cast<uint32_t>(oz);
        rec.width       = oct.w;
        rec.height      = oct.h;
        rec.outer_idx   = oidx;
        rec.manifold_id = manifold_id;
        model.assign_outer_point(rec.outer_idx, rec.outer_x, rec.outer_y, rec.outer_z);
        rec.train_count = 1u;
        rec.channels    = 3u;
        rec.shadow_sum  = oct.lum;
        rec.shadow_rgb  = oct.rgb;

        uint64_t npts = static_cast<uint64_t>(oct.w) * oct.h;
        total_mapped_pts += npts;

        std::printf("  [L%u RGB] %-20s | %3ux%-3u = %6llu exact RGB lattice pts | Lattice [%4u..%4u, %4u..%4u, z=%4u] <-> Outer #%u\n",
                    oct.level, rec.label.c_str(), oct.w, oct.h,
                    static_cast<unsigned long long>(npts),
                    rec.ox, rec.ox + rec.width - 1, rec.oy, rec.oy + rec.height - 1, rec.oz,
                    rec.outer_idx);

        model.classes.push_back(std::move(rec));
    }

    model.recompute_manifold_envelopes();
    for (const auto& m : model.manifolds) {
        std::printf("\n[MANIFOLD] #%u '%s' (CUDA: '%s') | tight lattice [%u..%u, %u..%u, %u..%u] | maxR=%.4f | vol=%.5f%% of 1B sphere\n",
                    m.manifold_id, m.name.c_str(), m.semantic_label.c_str(),
                    m.min_lx, m.max_lx, m.min_ly, m.max_ly, m.min_lz, m.max_lz,
                    m.max_radius, m.volume_fraction * 100.0f);
    }

    if (!model.save(model_path)) {
        std::cerr << "[ERROR] Failed to save model: " << model_path << "\n";
        return 1;
    }
    std::cout << "[SAVED]  " << model_path << " (" << total_mapped_pts
              << " exact RGB lattice points stored in .gllm v3)\n";
    return 0;
}
