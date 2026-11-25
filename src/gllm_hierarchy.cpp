// =============================================================================
// gllm_hierarchy — 4-Level Hierarchical 3D Bounding-Box Manifold Builder
//
// Constructs a strict zero-overlap 4-level nested 3D bounding-box manifold
// hierarchy inside the 1-billion-point implicit sphere lattice (1241^3), bound
// to 10 semantic concept nodes on the 1.2M outer shell (R = 1.0):
//
//   Level 0 (Root Box):      animals          <-> Outer #0 ('animals')
//     Level 1 (Box):         cats             <-> Outer #1 ('cat')
//       - Full uncropped cat RGB L2 lattice (in 'cats', outside 'cats/face')
//       Level 2 (Box):       cats/face        <-> Outer #3 ('face' - SHARED!)
//         - Cropped cat face RGB L2 lattice (in 'cats/face', behind features)
//         Level 3 (6 Boxes): cats/face/{ear_right,ear_left,eye_right,eye_left,nose,mouth}
//                            <-> SHARED Outer #4..#9 ('ear_right'..'mouth')
//     Level 1 (Box):         dogs             <-> Outer #2 ('dog')
//       - Full uncropped dog RGB L2 lattice (in 'dogs', outside 'dogs/face')
//       Level 2 (Box):       dogs/face        <-> Outer #3 ('face' - SHARED!)
//         - Cropped dog face RGB L2 lattice (in 'dogs/face', behind features)
//         Level 3 (6 Boxes): dogs/face/{ear_right,ear_left,eye_right,eye_left,nose,mouth}
//                            <-> SHARED Outer #4..#9 ('ear_right'..'mouth')
// =============================================================================

#include "../include/gaussian_pyramid.hpp"
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

static const std::array<std::string, 6> kFeatureNames = {
    "ear_right", "ear_left", "eye_right", "eye_left", "nose", "mouth"
};

static void print_usage() {
    std::cout << "Usage:\n"
              << "  gllm_hierarchy --model <out.gllm>\n"
              << "                 --cat-image <cat.png> --cat-bbox x0,y0,x1,y1\n"
              << "                 --cat-ear-right ... --cat-ear-left ... --cat-eye-right ...\n"
              << "                 --cat-eye-left ... --cat-nose ... --cat-mouth ...\n"
              << "                 --dog-image <dog.png> --dog-bbox x0,y0,x1,y1\n"
              << "                 --dog-ear-right ... --dog-ear-left ... --dog-eye-right ...\n"
              << "                 --dog-eye-left ... --dog-nose ... --dog-mouth ...\n"
              << "                 [--level 2] [--feature-max-dim 44] [--grid 1241]\n";
}

struct RgbImage {
    uint32_t w = 0, h = 0;
    std::vector<float> r, g, b;
};

static bool load_rgb_image(const std::string& path, RgbImage& out) {
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
    if (pclose(pipe) != 0 || buf.size() < 12) return false;

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

    if (read_tok() != "P6") return false;
    out.w = static_cast<uint32_t>(std::stoul(read_tok()));
    out.h = static_cast<uint32_t>(std::stoul(read_tok()));
    uint32_t maxv = static_cast<uint32_t>(std::stoul(read_tok()));
    if (pos < buf.size()) pos++;
    size_t n = static_cast<size_t>(out.w) * out.h;
    if (out.w == 0 || out.h == 0 || maxv != 255 || pos + n * 3u > buf.size()) return false;

    out.r.resize(n);
    out.g.resize(n);
    out.b.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out.r[i] = static_cast<float>(buf[pos + i * 3u + 0]) * (1.0f / 255.0f);
        out.g[i] = static_cast<float>(buf[pos + i * 3u + 1]) * (1.0f / 255.0f);
        out.b[i] = static_cast<float>(buf[pos + i * 3u + 2]) * (1.0f / 255.0f);
    }
    return true;
}

static bool parse_bbox(const std::string& s, int& x0, int& y0, int& x1, int& y1) {
    std::stringstream ss(s);
    std::string tok;
    double v[4] = {};
    int k = 0;
    while (std::getline(ss, tok, ',') && k < 4) v[k++] = std::atof(tok.c_str());
    if (k != 4) return false;
    x0 = static_cast<int>(std::floor(v[0]));
    y0 = static_cast<int>(std::floor(v[1]));
    x1 = static_cast<int>(std::ceil(v[2]));
    y1 = static_cast<int>(std::ceil(v[3]));
    return (x1 > x0 && y1 > y0);
}

static RgbImage crop_rgb(const RgbImage& src, int x0, int y0, int x1, int y1) {
    x0 = std::clamp(x0, 0, static_cast<int>(src.w) - 1);
    y0 = std::clamp(y0, 0, static_cast<int>(src.h) - 1);
    x1 = std::clamp(x1, x0 + 1, static_cast<int>(src.w));
    y1 = std::clamp(y1, y0 + 1, static_cast<int>(src.h));
    RgbImage dst;
    dst.w = static_cast<uint32_t>(x1 - x0);
    dst.h = static_cast<uint32_t>(y1 - y0);
    size_t n = static_cast<size_t>(dst.w) * dst.h;
    dst.r.resize(n); dst.g.resize(n); dst.b.resize(n);
    for (uint32_t y = 0; y < dst.h; ++y) {
        for (uint32_t x = 0; x < dst.w; ++x) {
            size_t si = static_cast<size_t>(y0 + y) * src.w + (x0 + x);
            size_t di = static_cast<size_t>(y) * dst.w + x;
            dst.r[di] = src.r[si];
            dst.g[di] = src.g[si];
            dst.b[di] = src.b[si];
        }
    }
    return dst;
}

static RgbImage pyramid_step(const RgbImage& src) {
    auto br = gllm::gaussian_blur_5tap(src.r.data(), src.w, src.h);
    auto bg = gllm::gaussian_blur_5tap(src.g.data(), src.w, src.h);
    auto bb = gllm::gaussian_blur_5tap(src.b.data(), src.w, src.h);
    uint32_t nw = std::max(1u, src.w / 2u);
    uint32_t nh = std::max(1u, src.h / 2u);
    RgbImage nxt;
    nxt.w = nw; nxt.h = nh;
    nxt.r = gllm::area_downsample_2d(br.data(), src.w, src.h, nw, nh);
    nxt.g = gllm::area_downsample_2d(bg.data(), src.w, src.h, nw, nh);
    nxt.b = gllm::area_downsample_2d(bb.data(), src.w, src.h, nw, nh);
    return nxt;
}

static RgbImage pyramid_octave(const RgbImage& src, uint32_t target_level) {
    RgbImage cur = src;
    for (uint32_t lv = 0; lv < target_level; ++lv) {
        if (cur.w < 16u || cur.h < 16u) break;
        cur = pyramid_step(cur);
    }
    return cur;
}

// Adaptive Gaussian Pyramid octave for facial features:
// steps down via 5-tap Burt-Adelson octaves until max(w,h) <= max_dim while preserving >= 10px min detail
static RgbImage pyramid_feature_fit(const RgbImage& src, uint32_t max_dim) {
    RgbImage cur = src;
    while (std::max(cur.w, cur.h) > max_dim && (cur.w / 2u) >= 10u && (cur.h / 2u) >= 10u) {
        cur = pyramid_step(cur);
    }
    return cur;
}

// Place all 10 semantic concept nodes cleanly in 3 structured arcs on the upper-front crown of R = 1.0:
//   Row 0 (Root):     #0 animals
//   Row 1 (Species):  #1 cat, #3 face (shared), #2 dog
//   Row 2 (Features): #4 ear_right, #5 ear_left, #6 eye_right, #7 eye_left, #8 nose, #9 mouth (all shared!)
static void place_concept_crown_point(uint32_t idx, float& ox, float& oy, float& oz) {
    float tx = 0.0f, ty = 0.88f;
    switch (idx) {
        case 0u: tx =  0.00f; ty = 0.93f; break; // animals
        case 1u: tx = -0.36f; ty = 0.82f; break; // cat
        case 2u: tx =  0.36f; ty = 0.82f; break; // dog
        case 3u: tx =  0.00f; ty = 0.78f; break; // face (shared)
        case 4u: tx = -0.50f; ty = 0.64f; break; // ear_right (shared)
        case 5u: tx = -0.30f; ty = 0.62f; break; // ear_left  (shared)
        case 6u: tx = -0.10f; ty = 0.60f; break; // eye_right (shared)
        case 7u: tx =  0.10f; ty = 0.60f; break; // eye_left  (shared)
        case 8u: tx =  0.30f; ty = 0.62f; break; // nose      (shared)
        case 9u: tx =  0.50f; ty = 0.64f; break; // mouth     (shared)
        default: {
            constexpr float kGolden = 2.39996322972865332f;
            float r = 0.22f * std::sqrt(static_cast<float>(idx));
            tx = r * std::cos(idx * kGolden);
            ty = 0.55f - 0.35f * r * std::sin(idx * kGolden);
            break;
        }
    }
    float xy2 = tx * tx + ty * ty;
    if (xy2 > 0.96f) {
        float s = std::sqrt(0.96f / xy2);
        tx *= s; ty *= s; xy2 = tx * tx + ty * ty;
    }
    ox = tx;
    oy = ty;
    oz = std::sqrt(std::max(0.0f, 1.0f - xy2));
}

static void add_rgb_slice(gllm::GeometricModel& model,
                          const RgbImage& img,
                          const std::string& label,
                          uint32_t manifold_id,
                          int64_t cx, int64_t cy, int64_t cz) {
    int64_t ox = cx - static_cast<int64_t>(img.w / 2u);
    int64_t oy = cy - static_cast<int64_t>(img.h / 2u);
    int64_t oz = cz;

    if (!model.inside_sphere(ox, oy, oz, img.w, img.h, 0.90f) ||
        model.overlaps_any(static_cast<uint32_t>(std::max<int64_t>(0, ox)),
                           static_cast<uint32_t>(std::max<int64_t>(0, oy)),
                           static_cast<uint32_t>(std::max<int64_t>(0, oz)),
                           img.w, img.h, 1u, 2u)) {
        std::cerr << "[ERROR] Slice '" << label << "' (" << img.w << "x" << img.h
                  << ") overlaps or exceeds sphere bounds at (" << ox << "," << oy << "," << oz << ")\n";
        std::exit(1);
    }

    const auto* m = model.get_manifold(manifold_id);
    gllm::ClassRecord rec;
    rec.label       = label;
    rec.ox          = static_cast<uint32_t>(ox);
    rec.oy          = static_cast<uint32_t>(oy);
    rec.oz          = static_cast<uint32_t>(oz);
    rec.width       = img.w;
    rec.height      = img.h;
    rec.manifold_id = manifold_id;
    rec.outer_idx   = (m && m->outer_idx != gllm::kInvalidId) ? m->outer_idx : model.next_outer_idx();
    rec.outer_x     = m ? m->outer_x : 0.0f;
    rec.outer_y     = m ? m->outer_y : 0.78f;
    rec.outer_z     = m ? m->outer_z : 0.625f;
    rec.train_count = 1u;
    rec.channels    = 3u;

    size_t n = static_cast<size_t>(img.w) * img.h;
    rec.shadow_sum.resize(n);
    rec.shadow_rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float rf = std::clamp(img.r[i], 0.0f, 1.0f);
        float gf = std::clamp(img.g[i], 0.0f, 1.0f);
        float bf = std::clamp(img.b[i], 0.0f, 1.0f);
        rec.shadow_sum[i] = 0.299f * rf + 0.587f * gf + 0.114f * bf;
        uint32_t rb = static_cast<uint32_t>(std::lround(rf * 255.0f)) & 0xFFu;
        uint32_t gb = static_cast<uint32_t>(std::lround(gf * 255.0f)) & 0xFFu;
        uint32_t bb = static_cast<uint32_t>(std::lround(bf * 255.0f)) & 0xFFu;
        rec.shadow_rgb[i] = 0x01000000u | (rb << 16) | (gb << 8) | bb;
    }

    std::printf("  [MAP]  %-22s -> Manifold #%-2u | %3ux%-3u (%6zu RGB pts) | Lattice [%4u..%4u, %4u..%4u, z=%4u] -> Outer #%u ('%s')\n",
                rec.label.c_str(), manifold_id, rec.width, rec.height, n,
                rec.ox, rec.ox + rec.width - 1, rec.oy, rec.oy + rec.height - 1, rec.oz,
                rec.outer_idx, m ? m->concept_label.c_str() : "?");
    model.classes.push_back(std::move(rec));
}

// Lay out the 6 facial features in a strictly non-overlapping anatomical constellation inside the face manifold!
//   Row 0 (top):      ear_right (left)  |  ear_left (right)
//   Row 1 (upper):    eye_right (left)  |  eye_left (right)
//   Row 2 (mid-low):         nose (center)
//   Row 3 (bottom):         mouth (center)
static void map_six_features_anatomically(
    gllm::GeometricModel& model,
    const std::string& species_prefix,
    const RgbImage& full_img,
    const std::map<std::string, std::string>& feat_bboxes,
    const std::array<uint32_t, 6>& feat_manifold_ids,
    int64_t cx, int64_t cy, int64_t z_feat,
    uint32_t feature_max_dim
) {
    std::array<RgbImage, 6> imgs;
    for (size_t i = 0; i < 6; ++i) {
        auto it = feat_bboxes.find(kFeatureNames[i]);
        if (it == feat_bboxes.end() || it->second.empty()) {
            std::cerr << "[ERROR] Missing feature bbox for " << species_prefix << "/" << kFeatureNames[i] << "\n";
            std::exit(1);
        }
        int x0, y0, x1, y1;
        if (!parse_bbox(it->second, x0, y0, x1, y1)) {
            std::cerr << "[ERROR] Invalid feature bbox for " << species_prefix << "/" << kFeatureNames[i] << "\n";
            std::exit(1);
        }
        imgs[i] = pyramid_feature_fit(crop_rgb(full_img, x0, y0, x1, y1), feature_max_dim);
    }

    const int64_t gap = 12; // Guaranteed clear spacing between sibling feature 3D boxes (including 2-voxel box pad)
    int64_t h_ears  = std::max(imgs[0].h, imgs[1].h);
    int64_t h_eyes  = std::max(imgs[2].h, imgs[3].h);
    int64_t h_nose  = imgs[4].h;
    int64_t h_mouth = imgs[5].h;

    int64_t total_h = h_ears + h_eyes + h_nose + h_mouth + 3 * gap;
    int64_t top_y   = cy + total_h / 2;

    int64_t y_ears  = top_y - h_ears / 2;
    int64_t y_eyes  = top_y - h_ears - gap - h_eyes / 2;
    int64_t y_nose  = top_y - h_ears - h_eyes - 2 * gap - h_nose / 2;
    int64_t y_mouth = top_y - h_ears - h_eyes - h_nose - 3 * gap - h_mouth / 2;

    int64_t ear_dx = (static_cast<int64_t>(imgs[0].w + imgs[1].w) / 2 + gap + 8) / 2;
    int64_t eye_dx = (static_cast<int64_t>(imgs[2].w + imgs[3].w) / 2 + gap + 6) / 2;

    int64_t xs[6] = { cx - ear_dx, cx + ear_dx, cx - eye_dx, cx + eye_dx, cx,     cx };
    int64_t ys[6] = { y_ears,      y_ears,      y_eyes,      y_eyes,      y_nose, y_mouth };

    for (size_t i = 0; i < 6; ++i) {
        add_rgb_slice(model, imgs[i], species_prefix + "_" + kFeatureNames[i],
                      feat_manifold_ids[i], xs[i], ys[i], z_feat);
    }
}

int main(int argc, char** argv) {
    std::string model_path, cat_img_path, cat_bbox_str, dog_img_path, dog_bbox_str;
    std::map<std::string, std::string> cat_feats, dog_feats;
    uint32_t level = 2u;
    uint32_t feature_max_dim = 44u;
    uint32_t grid_dim = 1241u;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--cat-image" && i + 1 < argc) cat_img_path = argv[++i];
        else if (a == "--cat-bbox" && i + 1 < argc) cat_bbox_str = argv[++i];
        else if (a == "--dog-image" && i + 1 < argc) dog_img_path = argv[++i];
        else if (a == "--dog-bbox" && i + 1 < argc) dog_bbox_str = argv[++i];
        else if (a == "--cat-ear-right" && i + 1 < argc) cat_feats["ear_right"] = argv[++i];
        else if (a == "--cat-ear-left"  && i + 1 < argc) cat_feats["ear_left"]  = argv[++i];
        else if (a == "--cat-eye-right" && i + 1 < argc) cat_feats["eye_right"] = argv[++i];
        else if (a == "--cat-eye-left"  && i + 1 < argc) cat_feats["eye_left"]  = argv[++i];
        else if (a == "--cat-nose"      && i + 1 < argc) cat_feats["nose"]      = argv[++i];
        else if (a == "--cat-mouth"     && i + 1 < argc) cat_feats["mouth"]     = argv[++i];
        else if (a == "--dog-ear-right" && i + 1 < argc) dog_feats["ear_right"] = argv[++i];
        else if (a == "--dog-ear-left"  && i + 1 < argc) dog_feats["ear_left"]  = argv[++i];
        else if (a == "--dog-eye-right" && i + 1 < argc) dog_feats["eye_right"] = argv[++i];
        else if (a == "--dog-eye-left"  && i + 1 < argc) dog_feats["eye_left"]  = argv[++i];
        else if (a == "--dog-nose"      && i + 1 < argc) dog_feats["nose"]      = argv[++i];
        else if (a == "--dog-mouth"     && i + 1 < argc) dog_feats["mouth"]     = argv[++i];
        else if (a == "--level" && i + 1 < argc) level = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--feature-max-dim" && i + 1 < argc) feature_max_dim = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--grid" && i + 1 < argc) grid_dim = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    if (model_path.empty() || cat_img_path.empty() || cat_bbox_str.empty() ||
        dog_img_path.empty() || dog_bbox_str.empty()) {
        print_usage();
        return 1;
    }

    RgbImage cat_full, dog_full;
    if (!load_rgb_image(cat_img_path, cat_full) || !load_rgb_image(dog_img_path, dog_full)) {
        std::cerr << "[ERROR] Failed to decode cat or dog RGB image.\n";
        return 1;
    }
    int cx0, cy0, cx1, cy1, dx0, dy0, dx1, dy1;
    if (!parse_bbox(cat_bbox_str, cx0, cy0, cx1, cy1) || !parse_bbox(dog_bbox_str, dx0, dy0, dx1, dy1)) {
        std::cerr << "[ERROR] Invalid --cat-bbox or --dog-bbox format.\n";
        return 1;
    }

    RgbImage cat_full_l2 = pyramid_octave(cat_full, level);
    RgbImage cat_face_l2 = pyramid_octave(crop_rgb(cat_full, cx0, cy0, cx1, cy1), level);
    RgbImage dog_full_l2 = pyramid_octave(dog_full, level);
    RgbImage dog_face_l2 = pyramid_octave(crop_rgb(dog_full, dx0, dy0, dx1, dy1), level);

    gllm::GeometricModel model;
    model.lattice_grid = grid_dim;
    int64_t mid    = static_cast<int64_t>(grid_dim / 2u);
    int64_t cat_cx = mid - 125;
    int64_t dog_cx = mid + 125;
    int64_t z_full = mid - 55;
    int64_t z_face = mid + 5;
    int64_t z_feat = mid + 62;

    // 1. Create Levels 0, 1, 2 (5 manifolds -> 4 outer-shell concept nodes: #0 animals, #1 cat, #2 dog, #3 face)
    uint32_t m_animals  = model.create_hierarchical_manifold("animals",   "animals", "root_animals_manifold",    gllm::kInvalidId, mid,    mid, mid,    1u);
    uint32_t m_cats     = model.create_hierarchical_manifold("cats",      "cat",     "species_cat_manifold",     m_animals,        cat_cx, mid, mid,    1u);
    uint32_t m_dogs     = model.create_hierarchical_manifold("dogs",      "dog",     "species_dog_manifold",     m_animals,        dog_cx, mid, mid,    1u);
    uint32_t m_cat_face = model.create_hierarchical_manifold("cats/face", "face",    "cat_facial_region_L2_rgb", m_cats,           cat_cx, mid, z_face, 1u);
    uint32_t m_dog_face = model.create_hierarchical_manifold("dogs/face", "face",    "dog_facial_region_L2_rgb", m_dogs,           dog_cx, mid, z_face, 1u);

    // 2. Create Level 3 (12 facial feature manifolds -> 6 SHARED outer-shell concept nodes #4..#9!)
    bool has_features = (cat_feats.size() == 6u && dog_feats.size() == 6u);
    std::array<uint32_t, 6> m_cat_feats{}, m_dog_feats{};
    if (has_features) {
        for (size_t i = 0; i < 6; ++i) {
            const auto& fn = kFeatureNames[i];
            m_cat_feats[i] = model.create_hierarchical_manifold(
                "cats/face/" + fn, fn, "cat_feature_" + fn, m_cat_face, cat_cx, mid, z_feat, 1u);
        }
        for (size_t i = 0; i < 6; ++i) {
            const auto& fn = kFeatureNames[i];
            m_dog_feats[i] = model.create_hierarchical_manifold(
                "dogs/face/" + fn, fn, "dog_feature_" + fn, m_dog_face, dog_cx, mid, z_feat, 1u);
        }
    }

    // Position all 10 outer-shell concept nodes cleanly along the upper-front crown of R = 1.0
    for (auto& m : model.manifolds) {
        if (m.outer_idx != gllm::kInvalidId) {
            place_concept_crown_point(m.outer_idx, m.outer_x, m.outer_y, m.outer_z);
        }
    }

    // 3. Map full-body slices (Level 1), cropped face slices (Level 2), and 6 facial features (Level 3)
    std::cout << "[HIERARCHY] Mapping 4-Level Animals -> Cat/Dog -> Face -> 6 Features (RGB):\n";
    add_rgb_slice(model, cat_full_l2, "cat_full_L2", m_cats,     cat_cx, mid, z_full);
    add_rgb_slice(model, cat_face_l2, "cat_face_L2", m_cat_face, cat_cx, mid, z_face);
    if (has_features) {
        map_six_features_anatomically(model, "cat", cat_full, cat_feats, m_cat_feats, cat_cx, mid, z_feat, feature_max_dim);
    }

    add_rgb_slice(model, dog_full_l2, "dog_full_L2", m_dogs,     dog_cx, mid, z_full);
    add_rgb_slice(model, dog_face_l2, "dog_face_L2", m_dog_face, dog_cx, mid, z_face);
    if (has_features) {
        map_six_features_anatomically(model, "dog", dog_full, dog_feats, m_dog_feats, dog_cx, mid, z_feat, feature_max_dim);
    }

    // 4. Recompute exact bottom-up 3D bounding boxes (features -> face -> species -> animals)
    model.recompute_manifold_envelopes();

    std::cout << "\n[MANIFOLDS] Computed 4-Level Nested 3D Bounding-Box Hierarchy (" << model.manifolds.size() << " manifolds):\n";
    for (const auto& m : model.manifolds) {
        std::string indent(m.depth * 2u, ' ');
        std::printf("  #%-2u %s%-20s | Outer #%-2u ('%-9s') | Box [%u..%u, %u..%u, %u..%u] | Vol %.4f%%\n",
                    m.manifold_id, indent.c_str(), m.name.c_str(),
                    m.outer_idx, m.concept_label.c_str(),
                    m.min_lx, m.max_lx, m.min_ly, m.max_ly, m.min_lz, m.max_lz,
                    m.volume_fraction * 100.0f);
    }

    if (!model.save(model_path)) {
        std::cerr << "[ERROR] Failed to save model: " << model_path << "\n";
        return 1;
    }
    std::cout << "\n[SAVED]  " << model_path << " (.gllm v4 4-level hierarchical 3D bounding-box model)\n";
    return 0;
}
