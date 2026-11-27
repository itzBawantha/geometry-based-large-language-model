// =============================================================================
// gllm_talk_viz — Real-Time CUDA + OpenGL 4.6 Parallel Fluid Sponge Visualizer
//
// Concentric 3-Sphere Architecture (Zero Weights, Zero Backprop):
//   1. Outer Sphere (R = 1.00) : ~1.36M Kaikki English dictionary words (CUDA VBO)
//   2. Middle Sphere (R = 0.72): 12 Part-of-Speech (POS) Cluster Hubs & Valves
//   3. Core Lattice  (R < 0.62): 128^3 (2,097,152-voxel) Carvable Fluid Sponge
//
// Parallel Fluid Splitting & Recombining on CUDA:
//   - Question content words inject independent fluid streams (100 mL each) with
//     distinct source bitmasks into the 128^3 CUDA sponge.
//   - `step_fluid_split_recombine` propagates and splits fluid across all 2.1M
//     voxels in parallel: single-word fluid glows electric cyan (brightness & size
//     = exact voxel weight), while 2+ word collisions erupt in blazing magenta-gold!
//   - Every competing candidate answer is initialized in 3D and on the HUD fluid
//     meters, filling live until the highest-fluid answer wins!
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/talk_hud.hpp"

#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// -----------------------------------------------------------------------------
// Compact 3D Math
// -----------------------------------------------------------------------------
struct Vec3 { float x = 0, y = 0, z = 0; };
static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(const Vec3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline Vec3 normalize(const Vec3& v) {
    float l = length(v);
    return l > 1e-7f ? Vec3{v.x / l, v.y / l, v.z / l} : Vec3{0, 1, 0};
}
static inline Vec3 lerp3(const Vec3& a, const Vec3& b, float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
}

struct Mat4 { float m[16] = {}; };
static Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[col * 4 + k];
            r.m[col * 4 + row] = s;
        }
    return r;
}
static Mat4 mat4_perspective(float fov_rad, float aspect, float znear, float zfar) {
    Mat4 r;
    float f = 1.0f / std::tan(0.5f * fov_rad);
    r.m[0] = f / aspect; r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar); r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}
static Mat4 mat4_look_at(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r;
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8]  = s.z;  r.m[12] = -dot(s, eye);
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9]  = u.z;  r.m[13] = -dot(u, eye);
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z; r.m[14] = dot(f, eye);
    r.m[15] = 1.0f;
    return r;
}

// -----------------------------------------------------------------------------
// GPU Structs (Exact Match to kernels/gllm_talk_kernels.ptx)
// -----------------------------------------------------------------------------
struct alignas(16) Float4 { float x, y, z, w; };
struct alignas(16) CarveSeg {
    Float4 p0_r;     // x0, y0, z0, radius
    Float4 p1_d;     // x1, y1, z1, delta_porosity
    Float4 dir_role; // dx, dy, dz, role_sign (-1=Q, +1=A)
};
static_assert(sizeof(CarveSeg) == 48, "CarveSeg must be 48 bytes");

struct alignas(16) FluidDrop {
    Float4   pos_mass;   // x, y, z, fluid_mass
    uint32_t source_bit;
    uint32_t pad0;
    float    pad1, pad2;
};
static_assert(sizeof(FluidDrop) == 32, "FluidDrop must be 32 bytes");

struct LineVert { float x, y, z, r, g, b, a; };
struct SpriteVert { float x, y, z, sz, r, g, b, a; };

// -----------------------------------------------------------------------------
// GLSL 4.60 Shaders
// -----------------------------------------------------------------------------
static const char* kBgVertSrc = R"GLSL(
#version 460 core
out vec2 vUV;
void main() {
    vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(pos[gl_VertexID], 0.9999, 1.0);
    vUV = pos[gl_VertexID] * 0.5 + 0.5;
}
)GLSL";

static const char* kBgFragSrc = R"GLSL(
#version 460 core
in vec2 vUV;
out vec4 fragColor;
void main() {
    vec2 d = (vUV - vec2(0.5)) * vec2(1.35, 1.0);
    vec3 col = mix(vec3(0.016, 0.026, 0.052), vec3(0.004, 0.007, 0.014), smoothstep(0.0, 1.05, length(d)));
    fragColor = vec4(col, 1.0);
}
)GLSL";

static const char* kWordVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec4 aPosVal;
layout(location = 1) in vec4 aColor;
uniform mat4  uMVP;
uniform vec3  uCamPos;
uniform float uViewportHeight;
uniform float uShellAlpha;
out vec3  vColor;
out float vGlow;
out float vAlpha;
void main() {
    float g = aPosVal.w;
    float ag = abs(g);
    if (ag < 0.002 && uShellAlpha <= 0.001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vColor = vec3(0.0); vGlow = 0.0; vAlpha = 0.0;
        return;
    }
    vec3 pos = aPosVal.xyz;
    gl_Position = uMVP * vec4(pos, 1.0);
    float dist = max(length(uCamPos - pos), 0.12);
    vGlow = ag;
    if (ag >= 0.002) {
        vec3 flashCol = (g > 0.0) ? vec3(0.18, 0.92, 1.0) : vec3(1.0, 0.78, 0.22);
        vColor = mix(aColor.rgb, flashCol, clamp(ag * 1.4, 0.25, 1.0));
        vAlpha = clamp(0.35 + ag * 0.95, 0.0, 1.0);
        gl_PointSize = clamp((3.2 + ag * 11.5) * (uViewportHeight / 900.0) * (2.6 / dist), 3.0, 22.0);
    } else {
        vec3 N = normalize(pos);
        vec3 V = normalize(uCamPos - pos);
        float fres = pow(1.0 - abs(dot(N, V)), 2.0);
        vColor = aColor.rgb * (0.45 + 0.55 * fres);
        vAlpha = clamp(uShellAlpha * (0.28 + 1.10 * fres), 0.0, 0.85);
        gl_PointSize = clamp(1.12 * (uViewportHeight / 900.0) * (2.6 / dist), 1.0, 3.0);
    }
}
)GLSL";

static const char* kWordFragSrc = R"GLSL(
#version 460 core
in vec3  vColor;
in float vGlow;
in float vAlpha;
out vec4 fragColor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    if (vGlow >= 0.002) {
        float core = 1.0 - smoothstep(0.0, 0.35, r2);
        float halo = 1.0 - smoothstep(0.18, 1.0, r2);
        vec3 col = mix(vColor, vec3(1.0), core * 0.75);
        fragColor = vec4(col, halo * vAlpha);
    } else {
        float soft = 1.0 - smoothstep(0.55, 1.0, r2);
        fragColor = vec4(vColor, vAlpha * soft);
    }
}
)GLSL";

static const char* kSpriteVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec4 aPosSz;
layout(location = 1) in vec4 aColor;
uniform mat4  uMVP;
uniform vec3  uCamPos;
uniform float uViewportHeight;
uniform float uScale;
out vec4 vColor;
void main() {
    vec3 pos = aPosSz.xyz;
    gl_Position = uMVP * vec4(pos, 1.0);
    vColor = aColor;
    float dist = max(length(uCamPos - pos), 0.08);
    gl_PointSize = clamp(aPosSz.w * uScale * (uViewportHeight / 900.0) * (2.2 / dist), 1.8, 42.0);
}
)GLSL";

static const char* kSpriteFragSrc = R"GLSL(
#version 460 core
in vec4 vColor;
out vec4 fragColor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    float core = 1.0 - smoothstep(0.0, 0.38, r2);
    float halo = 1.0 - smoothstep(0.14, 1.0, r2);
    vec3 col = mix(vColor.rgb, vec3(1.0), core * 0.45);
    fragColor = vec4(col, vColor.a * halo);
}
)GLSL";

static const char* kLineVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uMVP;
out vec4 vColor;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vColor = aColor; }
)GLSL";

static const char* kLineFragSrc = R"GLSL(
#version 460 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)GLSL";

static const char* kHudVertSrc = R"GLSL(
#version 460 core
uniform vec4 uRect;
out vec2 vUV;
void main() {
    vec2 uvs[6] = vec2[6](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    vec2 uv = uvs[gl_VertexID];
    vUV = vec2(uv.x, 1.0 - uv.y);
    gl_Position = vec4(mix(uRect.xy, uRect.zw, uv), 0.0, 1.0);
}
)GLSL";

static const char* kHudFragSrc = R"GLSL(
#version 460 core
in vec2 vUV;
uniform sampler2D uTex;
out vec4 fragColor;
void main() { fragColor = texture(uTex, vUV); }
)GLSL";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::cerr << "[GL SHADER ERROR]\n" << log << "\n";
        std::exit(1);
    }
    return sh;
}

static GLuint link_program(const char* vs_src, const char* fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << "[GL LINK ERROR]\n" << log << "\n";
        std::exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static CUmodule load_talk_ptx_module() {
    std::string dir = gllm::get_exe_dir();
    std::vector<std::string> cands = {
        dir + "../kernels/gllm_talk_kernels.ptx",
        dir + "kernels/gllm_talk_kernels.ptx",
        "./kernels/gllm_talk_kernels.ptx"
    };
    std::string ptx;
    for (const auto& p : cands) {
        std::ifstream ifs(p);
        if (ifs.good()) { std::ostringstream ss; ss << ifs.rdbuf(); ptx = ss.str(); break; }
    }
    if (ptx.empty()) { std::cerr << "[ERROR] Cannot find kernels/gllm_talk_kernels.ptx\n"; std::exit(1); }
    char info[4096] = {}, err[4096] = {};
    int opts[4] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* vals[4] = {info, reinterpret_cast<void*>(sizeof(info)), err, reinterpret_cast<void*>(sizeof(err))};
    CUmodule mod = nullptr;
    if (cuModuleLoadDataEx(&mod, ptx.c_str(), 4, opts, vals) != CUDA_SUCCESS) {
        std::cerr << "[CUDA JIT ERROR]\n" << err << "\n";
        std::exit(1);
    }
    return mod;
}

static inline Vec3 word_outer_xyz(uint32_t idx, uint32_t num_words) {
    double y = 1.0 - 2.0 * (static_cast<double>(idx) + 0.5) / static_cast<double>(num_words);
    float fy = static_cast<float>(y);
    float fr = std::sqrt(std::max(0.0f, 1.0f - fy * fy));
    double turns = static_cast<double>(idx) * 0.38196601125010515;
    double theta = (turns - std::floor(turns)) * 6.283185307179586;
    return {fr * std::cos(static_cast<float>(theta)), fy, fr * std::sin(static_cast<float>(theta))};
}

static std::array<Vec3, gllm::kNumPosClusters> build_pos_hub_positions(float r_pos = 0.72f) {
    std::array<Vec3, gllm::kNumPosClusters> pts{};
    for (uint32_t i = 0; i < gllm::kNumPosClusters; ++i) {
        double y = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(gllm::kNumPosClusters);
        float fy = static_cast<float>(y);
        float fr = std::sqrt(std::max(0.0f, 1.0f - fy * fy));
        double theta = static_cast<double>(i) * 2.39996322972865332;
        pts[i] = {fr * std::cos(static_cast<float>(theta)) * r_pos, fy * r_pos, fr * std::sin(static_cast<float>(theta)) * r_pos};
    }
    return pts;
}

static void add_wire_sphere(std::vector<LineVert>& out, float radius, float r, float g, float b, float a, int rings = 6, int segs = 48) {
    constexpr float kPi = 3.14159265358979323846f;
    for (int i = 1; i < rings; ++i) {
        float phi = kPi * static_cast<float>(i) / rings;
        float y = std::cos(phi) * radius, rr = std::sin(phi) * radius;
        for (int j = 0; j < segs; ++j) {
            float t0 = 2.0f * kPi * j / segs, t1 = 2.0f * kPi * (j + 1) / segs;
            out.push_back({rr * std::cos(t0), y, rr * std::sin(t0), r, g, b, a});
            out.push_back({rr * std::cos(t1), y, rr * std::sin(t1), r, g, b, a});
        }
    }
    for (int m = 0; m < rings; ++m) {
        float theta = kPi * static_cast<float>(m) / rings;
        float ct = std::cos(theta), st = std::sin(theta);
        for (int j = 0; j < segs; ++j) {
            float p0 = 2.0f * kPi * j / segs, p1 = 2.0f * kPi * (j + 1) / segs;
            out.push_back({radius * std::sin(p0) * ct, radius * std::cos(p0), radius * std::sin(p0) * st, r, g, b, a});
            out.push_back({radius * std::sin(p1) * ct, radius * std::cos(p1), radius * std::sin(p1) * st, r, g, b, a});
        }
    }
}

// -----------------------------------------------------------------------------
// Lexicon, Stopwords & Q&A Loader
// -----------------------------------------------------------------------------
static bool is_stop_word(const std::string& w) {
    static const std::unordered_set<std::string> kStop = {
        "what","who","whom","whose","where","when","why","how","which",
        "is","are","was","were","be","been","being","am","do","does","did",
        "have","has","had","can","could","will","would","should","may","might",
        "the","a","an","this","that","these","those","my","your","his","her","its","our","their",
        "of","in","on","at","to","for","with","by","as","from","into","about","between","during","under","over","after","before",
        "and","or","but","if","than","so","not","no","yes","it","he","she","they","we","you","i","many","much","name"
    };
    return w.size() <= 1 || kStop.count(w) > 0;
}

struct Lexicon {
    uint32_t num_words = 0;
    std::vector<uint8_t>  primary_pos;
    std::vector<uint16_t> pos_mask;
    std::vector<std::string> id_to_word;
    std::unordered_map<std::string, uint32_t> word_to_id;

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good()) return false;
        char magic[4] = {};
        uint32_t ver = 0, nwords = 0, npos = 0;
        ifs.read(magic, 4);
        ifs.read(reinterpret_cast<char*>(&ver), 4);
        ifs.read(reinterpret_cast<char*>(&nwords), 4);
        ifs.read(reinterpret_cast<char*>(&npos), 4);
        if (std::memcmp(magic, "GLXL", 4) != 0 || ver != 1u || nwords == 0u) return false;

        num_words = nwords;
        primary_pos.resize(nwords);
        pos_mask.resize(nwords);
        id_to_word.resize(nwords);
        word_to_id.reserve(static_cast<size_t>(nwords) * 13 / 10);

        char wbuf[256];
        for (uint32_t i = 0; i < nwords; ++i) {
            uint8_t pid = 0, wlen = 0;
            uint16_t mask = 0;
            ifs.read(reinterpret_cast<char*>(&pid), 1);
            ifs.read(reinterpret_cast<char*>(&wlen), 1);
            ifs.read(reinterpret_cast<char*>(&mask), 2);
            ifs.read(wbuf, wlen);
            primary_pos[i] = pid < gllm::kNumPosClusters ? pid : 11u;
            pos_mask[i]    = mask;
            id_to_word[i].assign(wbuf, wlen);
            word_to_id.emplace(id_to_word[i], i);
        }
        return ifs.good();
    }

    struct Tok { std::string text; uint32_t word_id; uint8_t pos; };

    std::vector<Tok> tokenize(const std::string& s) const {
        std::vector<Tok> out;
        std::string cur;
        auto flush = [&]() {
            if (cur.empty()) return;
            auto it = word_to_id.find(cur);
            if (it != word_to_id.end()) {
                uint32_t id = it->second;
                out.push_back({cur, id, primary_pos[id]});
            } else {
                bool all_dig = std::all_of(cur.begin(), cur.end(), [](unsigned char c) { return std::isdigit(c); });
                uint32_t h = 2166136261u;
                for (unsigned char c : cur) { h ^= c; h *= 16777619u; }
                uint32_t id = num_words ? (h % num_words) : 0u;
                uint8_t pid = all_dig ? 9u : 0u;
                out.push_back({cur, id, pid});
            }
            cur.clear();
        };
        for (size_t i = 0; i < s.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (std::isalnum(c) || (c == '\'' && !cur.empty() && i + 1 < s.size() && std::isalpha(static_cast<unsigned char>(s[i + 1])))) {
                cur.push_back(static_cast<char>(std::tolower(c)));
            } else {
                flush();
            }
        }
        flush();
        return out;
    }
};

struct QAPair { std::string q; std::string a; };

static std::vector<QAPair> load_qa_tsv(const std::string& path) {
    std::vector<QAPair> out;
    std::ifstream ifs(path);
    std::string line;
    while (std::getline(ifs, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab);
        std::string a = line.substr(tab + 1);
        if (!q.empty() && !a.empty()) out.push_back({std::move(q), std::move(a)});
    }
    return out;
}

// -----------------------------------------------------------------------------
// Zero-Weight Word->Answer Channel Graph + Parallel CUDA Fluid Wave Planner
// -----------------------------------------------------------------------------
struct AnswerNode {
    std::string text;
    uint32_t anchor_word_id = 0;
    uint8_t  primary_pos    = 0; // 9=num/year, 4=name, 0=noun, etc.
    uint32_t total_hits     = 0;
};

struct FluidBranchVisual {
    std::vector<Vec3> poly;    // Outer Q Word -> POS -> Sponge Entry -> Pool -> Exit -> Answer POS -> Answer Outer
    uint32_t source_idx   = 0; // Which question word bit
    uint32_t cand_idx     = 0; // Index into FluidWaveSim::candidates
    float    branch_ml    = 0.0f;
    bool     recombined   = false;
};

struct FluidWaveSim {
    bool active = false;
    uint32_t step = 0;
    uint32_t max_steps = 72;
    float step_accum = 0.0f;
    std::string question;
    std::string pos_valve_label;
    std::vector<uint32_t> q_content_wids;
    std::vector<uint8_t>  q_content_pos;

    std::vector<gllm::CandidateFluidRow> candidates;
    std::vector<uint32_t> cand_anchor_wids;
    std::vector<Vec3>     cand_outer_pts;
    std::vector<Vec3>     collision_pools;
    std::vector<FluidBranchVisual> branches;

    // GPU payloads for real 128^3 CUDA voxel carving & fluid wave injection!
    std::vector<CarveSeg>  branch_carve_segs;
    std::vector<FluidDrop> inject_drops;
};

struct AnimatedQAPath {
    std::vector<std::vector<Vec3>> q_polylines;
    std::vector<std::vector<Vec3>> a_polylines;
    std::vector<uint32_t> q_word_ids, a_word_ids;
    std::vector<uint8_t>  q_pos_ids,  a_pos_ids;
    std::vector<std::string> q_tok_texts, a_tok_texts;
    std::vector<CarveSeg> q_carve_segs, a_carve_segs;
    Vec3 s_Q;
    bool q_carved_gpu = false, a_carved_gpu = false;
};

class SpongeEngine {
public:
    static constexpr uint32_t kGrid = 128u;
    static constexpr float    kRad  = 0.62f;

    std::vector<float> h_porosity;
    std::vector<Vec3>  h_flow;
    uint64_t merge_events = 0;
    uint64_t split_events = 0;

    // Zero-weight porous channel index: content_word_id -> (answer_id -> carved_count)
    std::vector<AnswerNode> answers;
    std::unordered_map<std::string, uint32_t> ans_to_id;
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> word_to_ans;
    // Also track Q&A count per word for mass conservation
    std::unordered_map<uint32_t, uint32_t> word_out_degree;

    SpongeEngine() : h_porosity(kGrid * kGrid * kGrid, 0.0f), h_flow(kGrid * kGrid * kGrid) {}

    void clear() {
        std::fill(h_porosity.begin(), h_porosity.end(), 0.0f);
        std::fill(h_flow.begin(), h_flow.end(), Vec3{0, 0, 0});
        answers.clear();
        ans_to_id.clear();
        word_to_ans.clear();
        word_out_degree.clear();
        merge_events = 0;
        split_events = 0;
    }

    inline bool world_to_grid(const Vec3& p, int& ix, int& iy, int& iz) const {
        if (dot(p, p) > kRad * kRad) return false;
        float sc = 0.5f / kRad;
        ix = static_cast<int>(std::lround((p.x * sc + 0.5f) * kGrid));
        iy = static_cast<int>(std::lround((p.y * sc + 0.5f) * kGrid));
        iz = static_cast<int>(std::lround((p.z * sc + 0.5f) * kGrid));
        return (ix >= 0 && ix < static_cast<int>(kGrid) &&
                iy >= 0 && iy < static_cast<int>(kGrid) &&
                iz >= 0 && iz < static_cast<int>(kGrid));
    }

    inline Vec3 grid_to_world(int ix, int iy, int iz) const {
        float sc = 2.0f / static_cast<float>(kGrid);
        return {((ix + 0.5f) * sc - 1.0f) * kRad, ((iy + 0.5f) * sc - 1.0f) * kRad, ((iz + 0.5f) * sc - 1.0f) * kRad};
    }

    inline size_t vidx(int ix, int iy, int iz) const {
        return (static_cast<size_t>(iz) * kGrid + iy) * kGrid + ix;
    }

    static uint8_t classify_answer_pos(const std::string& a_text, const std::vector<Lexicon::Tok>& a_toks) {
        if (a_toks.empty()) return 11u;
        bool has_digit = false;
        for (unsigned char c : a_text) if (std::isdigit(c)) { has_digit = true; break; }
        if (has_digit) return 9u; // num / date / year
        // Proper noun heuristic: capitalized first character in raw text -> name (4)
        if (!a_text.empty() && std::isupper(static_cast<unsigned char>(a_text[0]))) return 4u;
        return a_toks[0].pos;
    }

    // Record Q -> A into zero-weight word-to-answer channel index (takes microseconds!)
    void index_qa_channels(const Lexicon& lex, const std::string& q_text, const std::string& a_text, uint32_t boost = 1u) {
        auto q_toks = lex.tokenize(q_text);
        auto a_toks = lex.tokenize(a_text);
        if (q_toks.empty() || a_toks.empty()) return;

        // Canonical lowercase key for answer deduplication
        std::string a_key;
        a_key.reserve(a_text.size());
        for (unsigned char c : a_text) a_key.push_back(static_cast<char>(std::tolower(c)));

        uint32_t aid = 0;
        auto it = ans_to_id.find(a_key);
        if (it == ans_to_id.end()) {
            aid = static_cast<uint32_t>(answers.size());
            ans_to_id.emplace(a_key, aid);
            AnswerNode an;
            an.text = a_text;
            // Pick most salient content word in answer as outer-sphere anchor
            an.anchor_word_id = a_toks[0].word_id;
            for (const auto& t : a_toks) {
                if (!is_stop_word(t.text)) { an.anchor_word_id = t.word_id; break; }
            }
            an.primary_pos = classify_answer_pos(a_text, a_toks);
            an.total_hits  = boost;
            answers.push_back(std::move(an));
        } else {
            aid = it->second;
            answers[aid].total_hits += boost;
        }

        // Deduplicate content words in Q so each distinct word gets one channel stamp
        std::unordered_set<uint32_t> seen_w;
        for (const auto& t : q_toks) {
            if (is_stop_word(t.text)) continue;
            if (seen_w.insert(t.word_id).second) {
                word_to_ans[t.word_id][aid] += boost;
                word_out_degree[t.word_id]  += boost;
            }
        }
    }

    Vec3 bend_toward_pores(const Vec3& p, const Vec3& dir_hint, bool& merged_hit) const {
        int cx, cy, cz;
        if (!world_to_grid(p, cx, cy, cz)) { merged_hit = false; return p; }
        float wsum = 0.0f;
        Vec3 com{0, 0, 0};
        constexpr int kR = 3;
        for (int dz = -kR; dz <= kR; ++dz) {
            int iz = cz + dz; if (iz < 0 || iz >= static_cast<int>(kGrid)) continue;
            for (int dy = -kR; dy <= kR; ++dy) {
                int iy = cy + dy; if (iy < 0 || iy >= static_cast<int>(kGrid)) continue;
                for (int dx = -kR; dx <= kR; ++dx) {
                    int ix = cx + dx; if (ix < 0 || ix >= static_cast<int>(kGrid)) continue;
                    size_t id = vidx(ix, iy, iz);
                    float por = h_porosity[id];
                    if (por < 0.04f) continue;
                    Vec3 fl = h_flow[id];
                    float fl_len = length(fl);
                    float align = fl_len > 1e-5f ? std::max(0.25f, 0.55f + 0.45f * dot(fl * (1.0f / fl_len), dir_hint)) : 0.6f;
                    float w = por * align / (1.0f + 0.35f * (dx * dx + dy * dy + dz * dz));
                    com = com + grid_to_world(ix, iy, iz) * w;
                    wsum += w;
                }
            }
        }
        if (wsum > 0.08f) {
            merged_hit = true;
            Vec3 target = com * (1.0f / wsum);
            float pull = std::min(0.62f, 0.25f + 0.37f * (1.0f - 1.0f / (1.0f + wsum * 0.6f)));
            Vec3 out = lerp3(p, target, pull);
            float l = length(out);
            if (l > kRad * 0.96f) out = out * ((kRad * 0.96f) / l);
            return out;
        }
        merged_hit = false;
        return p;
    }

    void stamp_host_segment(const Vec3& a, const Vec3& b, float delta) {
        Vec3 dir = normalize(b - a);
        for (int s = 0; s <= 10; ++s) {
            Vec3 p = lerp3(a, b, static_cast<float>(s) / 10.0f);
            int cx, cy, cz;
            if (!world_to_grid(p, cx, cy, cz)) continue;
            size_t id = vidx(cx, cy, cz);
            h_porosity[id] += delta;
            h_flow[id] = h_flow[id] + dir * delta;
        }
    }

    void trace_curved_sponge_leg(const Vec3& a, const Vec3& b, float role_sign, float delta,
                                 bool do_stamp_host, std::vector<Vec3>& out_poly, std::vector<CarveSeg>& out_segs) {
        Vec3 dir = normalize(b - a);
        constexpr int kSub = 5;
        Vec3 prev = a;
        if (out_poly.empty()) out_poly.push_back(a);
        for (int k = 1; k <= kSub; ++k) {
            Vec3 raw = lerp3(a, b, static_cast<float>(k) / kSub);
            Vec3 next = raw;
            if (k < kSub) {
                bool merged = false;
                next = bend_toward_pores(raw, dir, merged);
                if (do_stamp_host) { if (merged) merge_events++; else split_events++; }
            }
            out_poly.push_back(next);
            Vec3 seg_d = normalize(next - prev);
            out_segs.push_back({{prev.x, prev.y, prev.z, 1.0f}, {next.x, next.y, next.z, delta}, {seg_d.x, seg_d.y, seg_d.z, role_sign}});
            if (do_stamp_host) stamp_host_segment(prev, next, delta);
            prev = next;
        }
    }

    AnimatedQAPath plan_and_carve_qa(const Lexicon& lex,
                                     const std::array<Vec3, gllm::kNumPosClusters>& pos_hubs,
                                     const std::string& q_text, const std::string& a_text,
                                     bool add_to_graph = true, uint32_t graph_boost = 1u) {
        if (add_to_graph) index_qa_channels(lex, q_text, a_text, graph_boost);

        AnimatedQAPath path;
        auto q_toks = lex.tokenize(q_text);
        auto a_toks = lex.tokenize(a_text);
        if (q_toks.empty() || a_toks.empty()) return path;

        Vec3 cursor{0, 0, 0};
        for (size_t i = 0; i < q_toks.size(); ++i) {
            const auto& tk = q_toks[i];
            Vec3 w_pt   = word_outer_xyz(tk.word_id, lex.num_words);
            Vec3 hub_pt = pos_hubs[tk.pos];
            Vec3 entry  = normalize(hub_pt * 0.86f + w_pt * 0.14f) * (kRad * 0.94f);
            path.q_word_ids.push_back(tk.word_id);
            path.q_pos_ids.push_back(tk.pos);
            path.q_tok_texts.push_back(tk.text);

            std::vector<Vec3> leg = {w_pt, hub_pt, entry};
            float target_r = std::max(0.11f, 0.52f - 0.048f * static_cast<float>(i));
            Vec3 next_cursor;
            if (i == 0) {
                next_cursor = normalize(entry) * target_r;
                trace_curved_sponge_leg(entry, next_cursor, -1.0f, 0.22f, true, leg, path.q_carve_segs);
            } else {
                Vec3 blended = cursor * 0.56f + entry * 0.44f;
                float bl = length(blended);
                next_cursor = (bl > 1e-5f ? (blended * (target_r / bl)) : entry * 0.3f);
                trace_curved_sponge_leg(entry, next_cursor, -1.0f, 0.16f, true, leg, path.q_carve_segs);
                std::vector<Vec3> trunk;
                trace_curved_sponge_leg(cursor, next_cursor, -1.0f, 0.24f, true, trunk, path.q_carve_segs);
                path.q_polylines.push_back(std::move(trunk));
            }
            path.q_polylines.push_back(std::move(leg));
            cursor = next_cursor;
        }
        path.s_Q = cursor;

        Vec3 a_cursor = path.s_Q;
        for (size_t j = 0; j < a_toks.size(); ++j) {
            const auto& tk = a_toks[j];
            Vec3 w_pt    = word_outer_xyz(tk.word_id, lex.num_words);
            Vec3 hub_pt  = pos_hubs[tk.pos];
            Vec3 exit_gw = normalize(hub_pt * 0.84f + w_pt * 0.16f) * (kRad * 0.94f);
            path.a_word_ids.push_back(tk.word_id);
            path.a_pos_ids.push_back(tk.pos);
            path.a_tok_texts.push_back(tk.text);

            float step_r = std::min(kRad * 0.88f, length(a_cursor) + 0.10f);
            Vec3 mid_node = normalize(a_cursor * 0.42f + exit_gw * 0.58f) * step_r;
            std::vector<Vec3> leg;
            trace_curved_sponge_leg(a_cursor, mid_node, +1.0f, 0.26f, true, leg, path.a_carve_segs);
            trace_curved_sponge_leg(mid_node, exit_gw,  +1.0f, 0.22f, true, leg, path.a_carve_segs);
            leg.push_back(hub_pt);
            leg.push_back(w_pt);
            path.a_polylines.push_back(std::move(leg));
            a_cursor = mid_node;
        }
        return path;
    }

    // =========================================================================
    // TRUE PARALLEL FLUID SPLITTING & RECOMBINING WAVE LAUNCHER!
    //   1. Each question content word starts with 100.0 mL of fluid
    //   2. Fluid splits across ALL carved outgoing answer channels proportional
    //      to channel porosity (log(1 + carved_count)) -> mass conservation!
    //   3. Where 2+ question words' split streams collide on the same candidate
    //      answer, fluid RECOMBINES constructively & passes through the Middle
    //      POS Sphere Valve (e.g. What/Who/Where -> name/noun, When -> num)!
    //   4. Builds real 3D chord-midpoint pools + CUDA voxel injection drops so
    //      `step_fluid_split_recombine` physically splits & collides in 128^3 VRAM!
    // =========================================================================
    FluidWaveSim launch_fluid_wave(const Lexicon& lex,
                                   const std::array<Vec3, gllm::kNumPosClusters>& pos_hubs,
                                   const std::string& q_text) {
        FluidWaveSim sim;
        sim.question = q_text;
        auto q_toks = lex.tokenize(q_text);
        if (q_toks.empty() || answers.empty()) return sim;

        // 1. Determine Middle POS Sphere Valve from question opening
        //    valve_mask bit i = 1 means POS hub i is WIDE OPEN
        uint32_t valve_mask = 0xFFFFu;
        sim.pos_valve_label = "VALVE: ALL POS";
        std::string first_w = q_toks[0].text;
        std::string second_w = q_toks.size() > 1 ? q_toks[1].text : "";
        if (first_w == "when" || (first_w == "how" && (second_w == "many" || second_w == "much" || second_w == "long" || second_w == "old"))) {
            valve_mask = (1u << 9); // num/date
            sim.pos_valve_label = "VALVE: NUM/DATE";
        } else if (first_w == "who" || first_w == "whom" || first_w == "where") {
            valve_mask = (1u << 4) | (1u << 0); // name / noun
            sim.pos_valve_label = "VALVE: NAME/PLACE";
        } else if (first_w == "what" || first_w == "which") {
            // Check if explicitly asking "what year / what date"
            bool asks_num = (second_w == "year" || second_w == "date" || second_w == "percentage" || second_w == "number");
            if (asks_num) {
                valve_mask = (1u << 9);
                sim.pos_valve_label = "VALVE: NUM/YEAR";
            } else {
                valve_mask = (1u << 4) | (1u << 0) | (1u << 2); // name / noun / adj
                sim.pos_valve_label = "VALVE: NAME/NOUN";
            }
        }

        // 2. Collect unique content words from the question
        std::unordered_set<uint32_t> seen_cw;
        for (const auto& t : q_toks) {
            if (is_stop_word(t.text)) continue;
            if (seen_cw.insert(t.word_id).second && sim.q_content_wids.size() < 12) {
                sim.q_content_wids.push_back(t.word_id);
                sim.q_content_pos.push_back(t.pos);
            }
        }
        if (sim.q_content_wids.empty()) {
            for (const auto& t : q_toks) {
                if (seen_cw.insert(t.word_id).second && sim.q_content_wids.size() < 6) {
                    sim.q_content_wids.push_back(t.word_id);
                    sim.q_content_pos.push_back(t.pos);
                }
            }
        }
        const uint32_t K = static_cast<uint32_t>(sim.q_content_wids.size());
        if (K == 0) return sim;

        // 3. Split 100.0 mL per content word across its carved outgoing answer channels!
        struct AccCand {
            float raw_ml = 0.0f;
            uint32_t src_mask = 0u;
            std::vector<std::pair<uint32_t, float>> src_contribs; // (src_idx, split_ml)
        };
        std::unordered_map<uint32_t, AccCand> acc;

        for (uint32_t k = 0; k < K; ++k) {
            uint32_t wid = sim.q_content_wids[k];
            auto it = word_to_ans.find(wid);
            if (it == word_to_ans.end() || it->second.empty()) continue;

            // Compute total outgoing conductance for mass-conserving split!
            double total_cond = 0.0;
            for (const auto& kv : it->second) {
                total_cond += std::log(1.0 + static_cast<double>(kv.second));
            }
            // Specific words (small total_cond) pour denser fluid per branch;
            // very broad words split thinner across many branches (natural fluid physics!).
            double denom = std::pow(total_cond + 1.5, 0.72);
            for (const auto& kv : it->second) {
                float cond = static_cast<float>(std::log(1.0 + static_cast<double>(kv.second)));
                float split_ml = static_cast<float>(100.0 * cond / denom);
                auto& entry = acc[kv.first];
                entry.raw_ml += split_ml;
                entry.src_mask |= (1u << k);
                entry.src_contribs.push_back({k, split_ml});
            }
        }

        if (acc.empty()) return sim;

        // 4. Score all competing candidate answers by Recombined Fluid Volume + POS Valve!
        struct RankedCand {
            uint32_t aid;
            float final_ml;
            uint32_t drop_sources;
            uint32_t src_mask;
            bool valve_ok;
            std::vector<std::pair<uint32_t, float>> src_contribs;
        };
        std::vector<RankedCand> ranked;
        ranked.reserve(acc.size());
        for (auto& kv : acc) {
            uint32_t aid = kv.first;
            auto& ac = kv.second;
            uint32_t drops = static_cast<uint32_t>(__builtin_popcount(ac.src_mask));
            bool v_ok = ((valve_mask & (1u << answers[aid].primary_pos)) != 0u);
            // Constructive fluid recombination when 2+ independent word streams collide!
            float recomb_boost = 1.0f + 2.6f * static_cast<float>((drops > 0 ? drops - 1 : 0) * (drops > 0 ? drops - 1 : 0));
            float valve_gain   = v_ok ? 1.45f : 0.42f;
            float final_ml     = ac.raw_ml * recomb_boost * valve_gain;
            ranked.push_back({aid, final_ml, drops, ac.src_mask, v_ok, std::move(ac.src_contribs)});
        }

        size_t keep_n = std::min<size_t>(22, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + keep_n, ranked.end(),
                          [](const RankedCand& a, const RankedCand& b) { return a.final_ml > b.final_ml; });
        ranked.resize(keep_n);

        // Normalize display fluid nicely if huge (so bars & mL read crisply)
        sim.active = true;
        sim.step = 0;
        sim.max_steps = 72;

        // 5. Build 3D Geometry (Outer Q Word -> POS -> Chord Midpoint Collision Pool -> Answer POS -> Outer Answer)
        //    AND populate GPU carve segments + CUDA fluid injection drops!
        std::vector<Vec3> q_outer(K), q_hub(K), q_entry(K);
        for (uint32_t k = 0; k < K; ++k) {
            q_outer[k] = word_outer_xyz(sim.q_content_wids[k], lex.num_words);
            q_hub[k]   = pos_hubs[sim.q_content_pos[k]];
            q_entry[k] = normalize(q_hub[k] * 0.72f + q_outer[k] * 0.28f) * (kRad * 0.90f);
            // Inject primary fluid source at entry voxel with its unique source bitmask!
            sim.inject_drops.push_back({{q_entry[k].x, q_entry[k].y, q_entry[k].z, 1.85f}, (1u << k), 0u, 0.0f, 0.0f});
        }

        for (size_t ci = 0; ci < ranked.size(); ++ci) {
            const auto& rc = ranked[ci];
            const auto& an = answers[rc.aid];

            gllm::CandidateFluidRow row;
            row.answer        = an.text;
            row.current_ml    = 0.0f;
            row.target_ml     = rc.final_ml;
            row.drop_sources  = rc.drop_sources;
            row.total_sources = K;
            row.pos_valve_ok  = rc.valve_ok;
            row.answer_pos    = an.primary_pos;
            sim.candidates.push_back(std::move(row));

            Vec3 a_outer = word_outer_xyz(an.anchor_word_id, lex.num_words);
            Vec3 a_hub   = pos_hubs[an.primary_pos];
            Vec3 a_exit  = normalize(a_hub * 0.72f + a_outer * 0.28f) * (kRad * 0.90f);
            sim.cand_anchor_wids.push_back(an.anchor_word_id);
            sim.cand_outer_pts.push_back(a_outer);

            // Compute 3D Chord Midpoint Pool M for this candidate!
            // If 2+ question words collide here, M is their exact geometric centroid inside the sponge!
            Vec3 pool{0, 0, 0};
            for (const auto& sc : rc.src_contribs) pool = pool + q_entry[sc.first];
            pool = pool * (1.0f / std::max<size_t>(1, rc.src_contribs.size()));
            // Pull slightly toward answer exit so radial depth reflects specificity
            float pool_r = rc.drop_sources >= 2 ? 0.26f : 0.42f;
            Vec3 mix_dir = pool * 0.70f + a_exit * 0.30f;
            float ml = length(mix_dir);
            pool = (ml > 1e-5f) ? (mix_dir * (pool_r / ml)) : (a_exit * 0.45f);
            if (rc.drop_sources >= 2) sim.collision_pools.push_back(pool);

            // For each question word contributing a split stream into this candidate:
            for (const auto& sc : rc.src_contribs) {
                uint32_t k = sc.first;
                float weight = std::clamp(sc.second / 35.0f, 0.15f, 1.5f);
                FluidBranchVisual br;
                br.source_idx = k;
                br.cand_idx   = static_cast<uint32_t>(ci);
                br.branch_ml  = sc.second;
                br.recombined = (rc.drop_sources >= 2);
                br.poly = {q_outer[k], q_hub[k], q_entry[k]};

                // Carve sponge channel: q_entry[k] -> pool -> a_exit
                Vec3 mid1 = lerp3(q_entry[k], pool, 0.5f);
                Vec3 mid2 = lerp3(pool, a_exit, 0.5f);
                br.poly.push_back(mid1);
                br.poly.push_back(pool);
                br.poly.push_back(mid2);
                br.poly.push_back(a_exit);
                br.poly.push_back(a_hub);
                br.poly.push_back(a_outer);

                auto add_seg = [&](const Vec3& p0, const Vec3& p1, float role) {
                    Vec3 d = normalize(p1 - p0);
                    sim.branch_carve_segs.push_back({{p0.x, p0.y, p0.z, 1.0f}, {p1.x, p1.y, p1.z, 0.25f * weight}, {d.x, d.y, d.z, role}});
                };
                add_seg(q_entry[k], mid1, -1.0f);
                add_seg(mid1, pool,       -1.0f);
                add_seg(pool, mid2,       +1.0f);
                add_seg(mid2, a_exit,     +1.0f);

                // Seed fluid drops along incoming leg + near chord pool so CUDA waves physically collide!
                Vec3 seed_p = lerp3(q_entry[k], mid1, 0.40f);
                sim.inject_drops.push_back({{seed_p.x, seed_p.y, seed_p.z, 1.35f * weight}, (1u << k), 0u, 0.0f, 0.0f});
                if (rc.drop_sources >= 2) {
                    Vec3 near_pool = lerp3(mid1, pool, 0.90f);
                    sim.inject_drops.push_back({{near_pool.x, near_pool.y, near_pool.z, 1.65f * weight}, (1u << k), 0u, 0.0f, 0.0f});
                }

                sim.branches.push_back(std::move(br));
            }
        }
        return sim;
    }
};

static void save_ppm_screenshot(const std::string& path, int w, int h) {
    std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    std::ofstream ofs(path, std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (int y = h - 1; y >= 0; --y) {
        ofs.write(reinterpret_cast<const char*>(&pixels[static_cast<size_t>(y) * w * 3]),
                  static_cast<std::streamsize>(w) * 3);
    }
    std::cout << "[INFO] Saved screenshot (" << w << "x" << h << ") to " << path << "\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string lex_path = "/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/kaikki_lexicon.bin";
    std::string qa_path  = "/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/squad_qa.tsv";
    std::string screenshot_path;
    std::string auto_ask = "What is the capital of France?";
    int warmup_qa = 120;
    bool index_all = true;
    float cam_dist = 2.62f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lexicon" && i + 1 < argc) lex_path = argv[++i];
        else if (a == "--qa" && i + 1 < argc) qa_path = argv[++i];
        else if (a == "--screenshot" && i + 1 < argc) screenshot_path = argv[++i];
        else if (a == "--ask" && i + 1 < argc) auto_ask = argv[++i];
        else if (a == "--no-auto-ask") auto_ask.clear();
        else if (a == "--warmup-qa" && i + 1 < argc) warmup_qa = std::atoi(argv[++i]);
        else if (a == "--no-index-all") index_all = false;
        else if (a == "--dist" && i + 1 < argc) cam_dist = static_cast<float>(std::atof(argv[++i]));
    }

    // 1. Load Lexicon & SQuAD Q&A Pairs
    Lexicon lex;
    if (!lex.load(lex_path)) { std::cerr << "[ERROR] Failed to load lexicon: " << lex_path << "\n"; return 1; }
    auto qa_pairs = load_qa_tsv(qa_path);
    std::cout << "[DATA]   Loaded " << lex.num_words << " Kaikki words & " << qa_pairs.size() << " SQuAD Q&A pairs.\n";

    // 2. Initialize SDL2 + OpenGL 4.6
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { std::cerr << "[SDL ERROR] " << SDL_GetError() << "\n"; return 1; }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    int win_w = 1520, win_h = 920;
    SDL_Window* window = SDL_CreateWindow(
        "GLLM Fluid Sponge | Parallel CUDA Voxel Splitting & Recombining",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        win_w, win_h, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );
    if (!window) { std::cerr << "[SDL ERROR] " << SDL_GetError() << "\n"; SDL_Quit(); return 1; }
    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    if (!gl_ctx) { std::cerr << "[SDL ERROR] " << SDL_GetError() << "\n"; SDL_DestroyWindow(window); SDL_Quit(); return 1; }
    SDL_GL_SetSwapInterval(1);
    SDL_StartTextInput();

    std::cout << "[OpenGL] " << glGetString(GL_VENDOR) << " | " << glGetString(GL_RENDERER) << "\n";

    // 3. Initialize CUDA Driver API & Load PTX
    CU_CHECK(cuInit(0));
    CUdevice cu_dev = 0; CU_CHECK(cuDeviceGet(&cu_dev, 0));
    char gpu_name[256] = {}; CU_CHECK(cuDeviceGetName(gpu_name, sizeof(gpu_name), cu_dev));
    CUcontext cu_ctx = nullptr; CU_CHECK(cuCtxCreate_v2(&cu_ctx, 0, cu_dev));
    std::cout << "[CUDA]   Device 0: " << gpu_name << "\n";

    CUmodule cu_mod = load_talk_ptx_module();
    CUfunction fn_init_words = nullptr, fn_decay_words = nullptr, fn_carve = nullptr;
    CUfunction fn_inject = nullptr, fn_step_fluid = nullptr, fn_emit = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_init_words,  cu_mod, "init_word_outer_shell"));
    CU_CHECK(cuModuleGetFunction(&fn_decay_words, cu_mod, "decay_word_glow"));
    CU_CHECK(cuModuleGetFunction(&fn_carve,       cu_mod, "carve_sponge_voxels"));
    CU_CHECK(cuModuleGetFunction(&fn_inject,      cu_mod, "inject_fluid_drops"));
    CU_CHECK(cuModuleGetFunction(&fn_step_fluid,  cu_mod, "step_fluid_split_recombine"));
    CU_CHECK(cuModuleGetFunction(&fn_emit,        cu_mod, "emit_sponge_voxels"));

    CUevent ev_s = nullptr, ev_e = nullptr;
    CU_CHECK(cuEventCreate(&ev_s, 0));
    CU_CHECK(cuEventCreate(&ev_e, 0));

    // 4. Zero-Copy Outer Word Sphere VBO (1.36M words)
    const uint32_t N_WORDS = lex.num_words;
    GLuint word_vao = 0, word_vbo = 0;
    glGenVertexArrays(1, &word_vao); glGenBuffers(1, &word_vbo);
    glBindVertexArray(word_vao);
    glBindBuffer(GL_ARRAY_BUFFER, word_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<size_t>(N_WORDS) * 32), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(16));
    glBindVertexArray(0);

    CUgraphicsResource cu_word_res = nullptr;
    CU_CHECK(cuGraphicsGLRegisterBuffer(&cu_word_res, word_vbo, 0));
    {
        CUdeviceptr d_pids = 0, d_pcols = 0;
        const auto& meta = gllm::pos_cluster_meta();
        Float4 h_pcols[gllm::kNumPosClusters];
        for (uint32_t i = 0; i < gllm::kNumPosClusters; ++i) h_pcols[i] = {meta[i].r, meta[i].g, meta[i].b, 1.0f};
        CU_CHECK(cuMemAlloc_v2(&d_pids, N_WORDS));
        CU_CHECK(cuMemcpyHtoD_v2(d_pids, lex.primary_pos.data(), N_WORDS));
        CU_CHECK(cuMemAlloc_v2(&d_pcols, sizeof(h_pcols)));
        CU_CHECK(cuMemcpyHtoD_v2(d_pcols, h_pcols, sizeof(h_pcols)));

        CU_CHECK(cuGraphicsMapResources(1, &cu_word_res, nullptr));
        CUdeviceptr d_wvbo = 0; size_t wsz = 0;
        CU_CHECK(cuGraphicsResourceGetMappedPointer_v2(&d_wvbo, &wsz, cu_word_res));
        uint32_t nw = N_WORDS;
        void* args[] = {&d_wvbo, &d_pids, &d_pcols, &nw};
        CU_CHECK(cuLaunchKernel(fn_init_words, (N_WORDS + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuGraphicsUnmapResources(1, &cu_word_res, nullptr));
        CU_CHECK(cuMemFree_v2(d_pids)); CU_CHECK(cuMemFree_v2(d_pcols));
    }

    // 5. GPU 128^3 Sponge + Double-Buffered Fluid Wave Buffers + Zero-Copy VBO
    constexpr uint32_t kGrid = SpongeEngine::kGrid;
    constexpr uint32_t kTotalVox = kGrid * kGrid * kGrid;
    constexpr uint32_t kMaxSpongeVerts = 650000u;
    constexpr uint32_t kMaxSegsBatch = 4096u;
    constexpr uint32_t kMaxDropsBatch = 4096u;

    CUdeviceptr d_sponge = 0, d_meta = 0, d_segs = 0, d_drops = 0, d_out_cnt = 0;
    CUdeviceptr d_fluid_a = 0, d_fmask_a = 0, d_fluid_b = 0, d_fmask_b = 0, d_fstats = 0;
    CU_CHECK(cuMemAlloc_v2(&d_sponge,  static_cast<size_t>(kTotalVox) * 16));
    CU_CHECK(cuMemAlloc_v2(&d_meta,    static_cast<size_t>(kTotalVox) * 8));
    CU_CHECK(cuMemAlloc_v2(&d_fluid_a, static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_fmask_a, static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_fluid_b, static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_fmask_b, static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_segs,    static_cast<size_t>(kMaxSegsBatch) * sizeof(CarveSeg)));
    CU_CHECK(cuMemAlloc_v2(&d_drops,   static_cast<size_t>(kMaxDropsBatch) * sizeof(FluidDrop)));
    CU_CHECK(cuMemAlloc_v2(&d_out_cnt, sizeof(uint32_t)));
    CU_CHECK(cuMemAlloc_v2(&d_fstats,  4 * sizeof(uint32_t)));

    CU_CHECK(cuMemsetD32_v2(d_sponge,  0u, static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemsetD32_v2(d_meta,    0u, static_cast<size_t>(kTotalVox) * 2));
    CU_CHECK(cuMemsetD32_v2(d_fluid_a, 0u, kTotalVox));
    CU_CHECK(cuMemsetD32_v2(d_fmask_a, 0u, kTotalVox));
    CU_CHECK(cuMemsetD32_v2(d_fluid_b, 0u, kTotalVox));
    CU_CHECK(cuMemsetD32_v2(d_fmask_b, 0u, kTotalVox));

    GLuint sponge_vao = 0, sponge_vbo = 0;
    glGenVertexArrays(1, &sponge_vao); glGenBuffers(1, &sponge_vbo);
    glBindVertexArray(sponge_vao);
    glBindBuffer(GL_ARRAY_BUFFER, sponge_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(static_cast<size_t>(kMaxSpongeVerts) * 32), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(16));
    glBindVertexArray(0);

    CUgraphicsResource cu_sponge_res = nullptr;
    CU_CHECK(cuGraphicsGLRegisterBuffer(&cu_sponge_res, sponge_vbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    // 6. Dynamic GL Buffers (Wireframes, Rays, Sprites, HUD)
    GLuint bg_vao = 0, hud_vao = 0, wire_vao = 0, wire_vbo = 0, ray_vao = 0, ray_vbo = 0, spr_vao = 0, spr_vbo = 0;
    glGenVertexArrays(1, &bg_vao); glGenVertexArrays(1, &hud_vao);
    glGenVertexArrays(1, &wire_vao); glGenBuffers(1, &wire_vbo);
    glGenVertexArrays(1, &ray_vao);  glGenBuffers(1, &ray_vbo);
    glGenVertexArrays(1, &spr_vao);  glGenBuffers(1, &spr_vbo);

    std::vector<LineVert> wire_outer, wire_pos, wire_sponge, all_wires;
    add_wire_sphere(wire_outer,  1.00f, 0.20f, 0.48f, 0.78f, 0.13f, 6, 64);
    add_wire_sphere(wire_pos,    0.72f, 0.24f, 0.80f, 0.65f, 0.17f, 5, 56);
    add_wire_sphere(wire_sponge, 0.62f, 0.95f, 0.68f, 0.28f, 0.14f, 5, 48);
    uint32_t off_outer = 0, cnt_outer = static_cast<uint32_t>(wire_outer.size());
    all_wires.insert(all_wires.end(), wire_outer.begin(), wire_outer.end());
    uint32_t off_pos = static_cast<uint32_t>(all_wires.size()), cnt_pos = static_cast<uint32_t>(wire_pos.size());
    all_wires.insert(all_wires.end(), wire_pos.begin(), wire_pos.end());
    uint32_t off_sp = static_cast<uint32_t>(all_wires.size()), cnt_sp = static_cast<uint32_t>(wire_sponge.size());
    all_wires.insert(all_wires.end(), wire_sponge.begin(), wire_sponge.end());

    glBindVertexArray(wire_vao);
    glBindBuffer(GL_ARRAY_BUFFER, wire_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(all_wires.size() * sizeof(LineVert)), all_wires.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LineVert), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(LineVert), reinterpret_cast<void*>(12));

    glBindVertexArray(ray_vao);
    glBindBuffer(GL_ARRAY_BUFFER, ray_vbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * 1024 * 1024, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LineVert), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(LineVert), reinterpret_cast<void*>(12));

    glBindVertexArray(spr_vao);
    glBindBuffer(GL_ARRAY_BUFFER, spr_vbo);
    glBufferData(GL_ARRAY_BUFFER, 256 * 1024, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(SpriteVert), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(SpriteVert), reinterpret_cast<void*>(16));
    glBindVertexArray(0);

    GLuint hud_tex = 0;
    glGenTextures(1, &hud_tex);
    glBindTexture(GL_TEXTURE_2D, hud_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, gllm::TalkSidebarHUD::kPanelW, gllm::TalkSidebarHUD::kPanelH,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    GLuint bg_prog   = link_program(kBgVertSrc,     kBgFragSrc);
    GLuint word_prog = link_program(kWordVertSrc,   kWordFragSrc);
    GLuint spr_prog  = link_program(kSpriteVertSrc, kSpriteFragSrc);
    GLuint line_prog = link_program(kLineVertSrc,   kLineFragSrc);
    GLuint hud_prog  = link_program(kHudVertSrc,    kHudFragSrc);

    // 7. Simulation State & Helpers
    auto pos_hubs = build_pos_hub_positions(0.72f);
    SpongeEngine sponge;
    gllm::TalkSidebarHUD hud;
    gllm::TalkHudState hst;
    hst.total_dict_words = N_WORDS;
    hst.total_qa_pairs   = static_cast<uint32_t>(qa_pairs.size());
    hst.total_voxels     = kTotalVox;

    std::deque<AnimatedQAPath> history;
    AnimatedQAPath current_qa;
    bool has_current = false;
    FluidWaveSim fluid_sim;
    size_t qa_cursor = 0;

    std::vector<CarveSeg> pending_gpu_segs;
    std::vector<std::pair<uint32_t, float>> pending_word_flashes;

    auto dispatch_carve_gpu = [&](const std::vector<CarveSeg>& segs) {
        if (segs.empty()) return;
        for (size_t off = 0; off < segs.size(); off += kMaxSegsBatch) {
            uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(kMaxSegsBatch, segs.size() - off));
            CU_CHECK(cuMemcpyHtoD_v2(d_segs, segs.data() + off, cnt * sizeof(CarveSeg)));
            uint32_t g = kGrid; float wr = SpongeEngine::kRad;
            void* args[] = {&d_sponge, &d_meta, &d_segs, &cnt, &g, &wr};
            CU_CHECK(cuLaunchKernel(fn_carve, (cnt * 32u + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        }
    };

    auto dispatch_inject_gpu = [&](const std::vector<FluidDrop>& drops) {
        if (drops.empty()) return;
        for (size_t off = 0; off < drops.size(); off += kMaxDropsBatch) {
            uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(kMaxDropsBatch, drops.size() - off));
            CU_CHECK(cuMemcpyHtoD_v2(d_drops, drops.data() + off, cnt * sizeof(FluidDrop)));
            uint32_t g = kGrid; float wr = SpongeEngine::kRad;
            void* args[] = {&d_fluid_a, &d_fmask_a, &d_drops, &cnt, &g, &wr};
            CU_CHECK(cuLaunchKernel(fn_inject, (cnt + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        }
    };

    auto start_fluid_wave = [&](const std::string& q_txt) {
        fluid_sim = sponge.launch_fluid_wave(lex, pos_hubs, q_txt);
        hst.fluid_wave_active = fluid_sim.active;
        hst.pos_valve_label   = fluid_sim.pos_valve_label;
        hst.wave_step         = 0;
        hst.wave_max_steps    = fluid_sim.max_steps;
        hst.candidates        = fluid_sim.candidates;
        hst.gpu_fluid_voxels  = 0;
        hst.gpu_split_forks   = 0;
        hst.gpu_collision_vox = 0;
        hst.gpu_peak_vox_wt   = 0.0f;

        CU_CHECK(cuMemsetD32_v2(d_fluid_a, 0u, kTotalVox));
        CU_CHECK(cuMemsetD32_v2(d_fmask_a, 0u, kTotalVox));
        CU_CHECK(cuMemsetD32_v2(d_fluid_b, 0u, kTotalVox));
        CU_CHECK(cuMemsetD32_v2(d_fmask_b, 0u, kTotalVox));

        if (!fluid_sim.active) {
            hst.fluid_summary = "No carved channels match those words yet (click INDEX ALL 89K or CARVE Q->A).";
            return;
        }
        hst.streaming = false;
        hst.burst_mode = false;
        // Ensure candidate branch geometry exists in the 128^3 CUDA sponge, then inject initial word drops!
        dispatch_carve_gpu(fluid_sim.branch_carve_segs);
        dispatch_inject_gpu(fluid_sim.inject_drops);
        CU_CHECK(cuCtxSynchronize());

        for (size_t k = 0; k < fluid_sim.q_content_wids.size(); ++k) {
            pending_word_flashes.push_back({fluid_sim.q_content_wids[k], +1.0f});
            hst.pos_activity[fluid_sim.q_content_pos[k]] = 1.0f;
        }
        char sbuf[160];
        std::snprintf(sbuf, sizeof(sbuf), "Injected %zu word drops -> %zu competing answers initialized",
                      fluid_sim.q_content_wids.size(), fluid_sim.candidates.size());
        hst.fluid_summary = sbuf;
    };

    auto trigger_qa = [&](const std::string& q_txt, const std::string& a_txt, bool instant_carve, bool add_graph = true, uint32_t boost = 1u) {
        if (has_current) {
            if (!current_qa.q_carved_gpu) {
                pending_gpu_segs.insert(pending_gpu_segs.end(), current_qa.q_carve_segs.begin(), current_qa.q_carve_segs.end());
                current_qa.q_carved_gpu = true;
            }
            if (!current_qa.a_carved_gpu) {
                pending_gpu_segs.insert(pending_gpu_segs.end(), current_qa.a_carve_segs.begin(), current_qa.a_carve_segs.end());
                current_qa.a_carved_gpu = true;
            }
            history.push_back(std::move(current_qa));
            while (history.size() > 22) history.pop_front();
        }
        current_qa = sponge.plan_and_carve_qa(lex, pos_hubs, q_txt, a_txt, add_graph, boost);
        has_current = !current_qa.q_word_ids.empty();
        hst.qa_processed = std::max(hst.qa_processed + 1u, static_cast<uint32_t>(qa_cursor));
        hst.active_q = q_txt;
        hst.active_a = a_txt;
        hst.active_q_toks.clear(); hst.active_a_toks.clear();
        for (size_t i = 0; i < current_qa.q_word_ids.size(); ++i) {
            hst.active_q_toks.push_back({current_qa.q_tok_texts[i], current_qa.q_pos_ids[i]});
            hst.pos_activity[current_qa.q_pos_ids[i]] = 1.0f;
            pending_word_flashes.push_back({current_qa.q_word_ids[i], +1.0f});
        }
        for (size_t i = 0; i < current_qa.a_word_ids.size(); ++i) {
            hst.active_a_toks.push_back({current_qa.a_tok_texts[i], current_qa.a_pos_ids[i]});
            if (instant_carve) {
                hst.pos_activity[current_qa.a_pos_ids[i]] = 1.0f;
                pending_word_flashes.push_back({current_qa.a_word_ids[i], -1.0f});
            }
        }
        hst.anim_phase = instant_carve ? 1.0f : 0.0f;
        if (instant_carve) {
            pending_gpu_segs.insert(pending_gpu_segs.end(), current_qa.q_carve_segs.begin(), current_qa.q_carve_segs.end());
            pending_gpu_segs.insert(pending_gpu_segs.end(), current_qa.a_carve_segs.begin(), current_qa.a_carve_segs.end());
            current_qa.q_carved_gpu = true;
            current_qa.a_carved_gpu = true;
        }
    };

    auto index_all_squad = [&]() {
        for (const auto& qa : qa_pairs) sponge.index_qa_channels(lex, qa.q, qa.a, 1u);
        hst.qa_processed = static_cast<uint32_t>(qa_pairs.size());
        std::cout << "[SPONGE] Indexed all " << qa_pairs.size() << " SQuAD Q&As ("
                  << sponge.answers.size() << " unique answer basins).\n";
    };

    // Index all 89,003 SQuAD pairs into the zero-weight word-channel graph + carve initial arterial sponge
    if (index_all) index_all_squad();
    for (int w = 0; w < warmup_qa && qa_cursor < qa_pairs.size(); ++w) {
        trigger_qa(qa_pairs[qa_cursor].q, qa_pairs[qa_cursor].a, true, !index_all, 1u);
        qa_cursor++;
    }
    if (!pending_gpu_segs.empty()) {
        dispatch_carve_gpu(pending_gpu_segs);
        pending_gpu_segs.clear();
        CU_CHECK(cuCtxSynchronize());
    }
    if (index_all) hst.qa_processed = static_cast<uint32_t>(qa_pairs.size());

    // Auto-launch the live fluid splitting wave so the user IMMEDIATELY sees voxel splitting & candidate meters!
    if (!auto_ask.empty()) {
        hst.input_q = auto_ask;
        start_fluid_wave(auto_ask);
    } else if (qa_cursor < qa_pairs.size()) {
        trigger_qa(qa_pairs[qa_cursor].q, qa_pairs[qa_cursor].a, false, !index_all, 1u);
        qa_cursor++;
    }

    float cam_yaw = 0.52f, cam_pitch = 0.26f, shell_alpha = 0.22f;
    bool dragging = false, auto_rot = true;
    bool show_words = true, show_pos = true, show_sponge = true, show_rays = true;

    uint64_t prev_ticks = SDL_GetPerformanceCounter();
    const double perf_freq = static_cast<double>(SDL_GetPerformanceFrequency());
    double fps_timer = 0.0;
    int fps_frames = 0, total_frames = 0;
    uint32_t emitted_sponge_verts = 0;

    bool running = true;
    while (running) {
        uint64_t now_ticks = SDL_GetPerformanceCounter();
        double dt = std::min(0.08, static_cast<double>(now_ticks - prev_ticks) / perf_freq);
        prev_ticks = now_ticks;

        int cur_w = 0, cur_h = 0;
        SDL_GetWindowSize(window, &cur_w, &cur_h);
        int panel_px = std::min(gllm::TalkSidebarHUD::kPanelW, cur_w / 2);

        auto to_hud_xy = [&](int mx, int my, int& hx, int& hy) {
            hx = (mx * gllm::TalkSidebarHUD::kPanelW) / std::max(1, panel_px);
            hy = (my * gllm::TalkSidebarHUD::kPanelH) / std::max(1, cur_h);
        };

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_MOUSEMOTION) {
                int hx, hy;
                to_hud_xy(ev.motion.x, ev.motion.y, hx, hy);
                hud.hover_btn = 0;
                if (ev.motion.x < panel_px) {
                    if (hud.btn_ask.contains(hx, hy))         hud.hover_btn = 1;
                    else if (hud.btn_carve.contains(hx, hy))  hud.hover_btn = 2;
                    else if (hud.btn_load89.contains(hx, hy)) hud.hover_btn = 3;
                    else if (hud.btn_stream.contains(hx, hy)) hud.hover_btn = 4;
                    else if (hud.btn_replay.contains(hx, hy)) hud.hover_btn = 5;
                } else if (dragging) {
                    cam_yaw   -= ev.motion.xrel * 0.0065f;
                    cam_pitch  = std::clamp(cam_pitch + ev.motion.yrel * 0.0065f, -1.50f, 1.50f);
                }
            } else if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
                if (ev.button.x < panel_px) {
                    int hx, hy;
                    to_hud_xy(ev.button.x, ev.button.y, hx, hy);
                    if (hud.box_q.contains(hx, hy)) {
                        hst.focus_field = 1;
                    } else if (hud.box_a.contains(hx, hy)) {
                        hst.focus_field = 2;
                    } else if (hud.btn_ask.contains(hx, hy)) {
                        hst.focus_field = 0;
                        start_fluid_wave(hst.input_q);
                    } else if (hud.btn_carve.contains(hx, hy)) {
                        hst.focus_field = 0;
                        fluid_sim.active = false; hst.fluid_wave_active = false;
                        hst.streaming = false; hst.burst_mode = false;
                        if (!hst.input_q.empty() && !hst.input_a.empty()) {
                            // Carve custom Q->A with boost=4 so newly taught facts form crisp arteries immediately!
                            trigger_qa(hst.input_q, hst.input_a, false, true, 4u);
                        }
                    } else if (hud.btn_load89.contains(hx, hy)) {
                        hst.focus_field = 0;
                        index_all_squad();
                        start_fluid_wave(hst.input_q);
                    } else if (hud.btn_stream.contains(hx, hy)) {
                        hst.focus_field = 0;
                        hst.streaming = !hst.streaming;
                        if (hst.streaming) { fluid_sim.active = false; hst.fluid_wave_active = false; }
                    } else if (hud.btn_replay.contains(hx, hy)) {
                        hst.focus_field = 0;
                        start_fluid_wave(hst.input_q);
                    } else {
                        hst.focus_field = 0;
                    }
                } else {
                    hst.focus_field = 0;
                    dragging = true;
                }
            } else if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
            } else if (ev.type == SDL_MOUSEWHEEL) {
                cam_dist = std::clamp(cam_dist * (ev.wheel.y > 0 ? 0.90f : 1.11f), 0.45f, 7.5f);
            } else if (ev.type == SDL_TEXTINPUT && hst.focus_field != 0) {
                std::string& dst = (hst.focus_field == 1) ? hst.input_q : hst.input_a;
                if (dst.size() < 96) dst += ev.text.text;
            } else if (ev.type == SDL_KEYDOWN) {
                SDL_Keycode k = ev.key.keysym.sym;
                if (hst.focus_field != 0) {
                    std::string& dst = (hst.focus_field == 1) ? hst.input_q : hst.input_a;
                    if (k == SDLK_BACKSPACE && !dst.empty()) dst.pop_back();
                    else if (k == SDLK_ESCAPE) hst.focus_field = 0;
                    else if (k == SDLK_TAB) hst.focus_field = (hst.focus_field == 1 ? 2 : 1);
                    else if (k == SDLK_RETURN) {
                        if (hst.focus_field == 1 && !hst.input_q.empty()) start_fluid_wave(hst.input_q);
                        else if (hst.focus_field == 2 && !hst.input_q.empty() && !hst.input_a.empty()) {
                            fluid_sim.active = false; hst.fluid_wave_active = false; hst.streaming = false;
                            trigger_qa(hst.input_q, hst.input_a, false, true, 4u);
                        }
                        hst.focus_field = 0;
                    }
                } else {
                    switch (k) {
                        case SDLK_ESCAPE:
                        case SDLK_q: running = false; break;
                        case SDLK_SPACE:
                            hst.streaming = !hst.streaming;
                            if (hst.streaming) { fluid_sim.active = false; hst.fluid_wave_active = false; }
                            break;
                        case SDLK_t:
                            hst.burst_mode = !hst.burst_mode;
                            if (hst.burst_mode) { hst.streaming = true; fluid_sim.active = false; hst.fluid_wave_active = false; }
                            break;
                        case SDLK_s:
                            hst.streaming = false; hst.burst_mode = false;
                            fluid_sim.active = false; hst.fluid_wave_active = false;
                            if (qa_cursor < qa_pairs.size()) {
                                trigger_qa(qa_pairs[qa_cursor].q, qa_pairs[qa_cursor].a, false, true, 1u);
                                qa_cursor++;
                            }
                            break;
                        case SDLK_EQUALS:
                        case SDLK_PLUS:  hst.stream_rate = std::min(30.0f, hst.stream_rate * 1.45f); break;
                        case SDLK_MINUS: hst.stream_rate = std::max(0.15f, hst.stream_rate / 1.45f); break;
                        case SDLK_1: show_words  = !show_words; break;
                        case SDLK_2: show_pos    = !show_pos; break;
                        case SDLK_3: show_sponge = !show_sponge; break;
                        case SDLK_4: show_rays   = !show_rays; break;
                        case SDLK_a: auto_rot    = !auto_rot; break;
                        case SDLK_UP:   shell_alpha = std::min(0.85f, shell_alpha + 0.08f); break;
                        case SDLK_DOWN: shell_alpha = std::max(0.00f, shell_alpha - 0.08f); break;
                        case SDLK_r: cam_yaw = 0.52f; cam_pitch = 0.26f; cam_dist = 2.62f; break;
                        case SDLK_x:
                            sponge.clear(); history.clear(); has_current = false;
                            fluid_sim.active = false; hst.fluid_wave_active = false;
                            hst.candidates.clear(); hst.fluid_summary.clear();
                            qa_cursor = 0; hst.qa_processed = 0;
                            CU_CHECK(cuMemsetD32_v2(d_sponge,  0u, static_cast<size_t>(kTotalVox) * 4));
                            CU_CHECK(cuMemsetD32_v2(d_meta,    0u, static_cast<size_t>(kTotalVox) * 2));
                            CU_CHECK(cuMemsetD32_v2(d_fluid_a, 0u, kTotalVox));
                            CU_CHECK(cuMemsetD32_v2(d_fmask_a, 0u, kTotalVox));
                            break;
                        default: break;
                    }
                }
            }
        }

        // 8. Advance Simulation (Either Live Parallel Fluid Wave OR Q->A Carving Stream)
        for (float& pa : hst.pos_activity) pa = std::max(0.0f, pa - static_cast<float>(dt) * 0.85f);

        bool do_fluid_step_now = false;
        if (fluid_sim.active) {
            if (!screenshot_path.empty()) {
                // Fast-forward wave to step 54 for a dramatic verification screenshot showing splitting + collisions + filled meters!
                do_fluid_step_now = (fluid_sim.step < 54u);
            } else {
                fluid_sim.step_accum += static_cast<float>(dt) * 22.0f; // ~22 fluid wave steps/sec (~3.2s full propagation!)
                if (fluid_sim.step_accum >= 1.0f && fluid_sim.step < fluid_sim.max_steps) {
                    fluid_sim.step_accum -= 1.0f;
                    do_fluid_step_now = true;
                }
            }
        } else if (hst.burst_mode && hst.streaming && !qa_pairs.empty()) {
            for (int b = 0; b < 20 && qa_cursor < qa_pairs.size(); ++b) {
                trigger_qa(qa_pairs[qa_cursor].q, qa_pairs[qa_cursor].a, true, true, 1u);
                qa_cursor = (qa_cursor + 1) % qa_pairs.size();
            }
        } else if (has_current) {
            float prev_ph = hst.anim_phase;
            float speed = hst.streaming ? hst.stream_rate : std::max(0.65f, hst.stream_rate);
            hst.anim_phase = std::min(1.0f, hst.anim_phase + static_cast<float>(dt) * speed);
            if (!current_qa.q_carved_gpu && hst.anim_phase >= 0.18f) {
                pending_gpu_segs.insert(pending_gpu_segs.end(), current_qa.q_carve_segs.begin(), current_qa.q_carve_segs.end());
                current_qa.q_carved_gpu = true;
            }
            if (!current_qa.a_carved_gpu && hst.anim_phase >= 0.52f) {
                pending_gpu_segs.insert(pending_gpu_segs.end(), current_qa.a_carve_segs.begin(), current_qa.a_carve_segs.end());
                current_qa.a_carved_gpu = true;
                for (size_t i = 0; i < current_qa.a_word_ids.size(); ++i) {
                    hst.pos_activity[current_qa.a_pos_ids[i]] = 1.0f;
                    pending_word_flashes.push_back({current_qa.a_word_ids[i], -1.0f});
                }
            }
            if (hst.streaming && prev_ph >= 0.999f && !qa_pairs.empty()) {
                trigger_qa(qa_pairs[qa_cursor].q, qa_pairs[qa_cursor].a, false, true, 1u);
                qa_cursor = (qa_cursor + 1) % qa_pairs.size();
            }
        }

        if (auto_rot && !dragging && screenshot_path.empty()) {
            cam_yaw += static_cast<float>(dt) * 0.14f;
        }

        // 9. Execute CUDA Kernels!
        CU_CHECK(cuEventRecord(ev_s, nullptr));

        if (!pending_gpu_segs.empty()) {
            dispatch_carve_gpu(pending_gpu_segs);
            pending_gpu_segs.clear();
        }

        if (fluid_sim.active && do_fluid_step_now) {
            int sub_steps = screenshot_path.empty() ? 2 : 9;
            for (int ss = 0; ss < sub_steps && fluid_sim.step < fluid_sim.max_steps; ++ss) {
                // Keep feeding word drops during the first 28 steps so continuous waves split & collide!
                if (fluid_sim.step < 28u && (fluid_sim.step % 4u == 0u)) {
                    dispatch_inject_gpu(fluid_sim.inject_drops);
                }
                CU_CHECK(cuMemsetD32_v2(d_fluid_b, 0u, kTotalVox));
                CU_CHECK(cuMemsetD32_v2(d_fmask_b, 0u, kTotalVox));
                CU_CHECK(cuMemsetD32_v2(d_fstats,  0u, 4u));
                uint32_t g = kGrid;
                void* sargs[] = {&d_sponge, &d_fluid_a, &d_fmask_a, &d_fluid_b, &d_fmask_b, &d_fstats, &g};
                CU_CHECK(cuLaunchKernel(fn_step_fluid, (kTotalVox + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, sargs, nullptr));
                std::swap(d_fluid_a, d_fluid_b);
                std::swap(d_fmask_a, d_fmask_b);
                fluid_sim.step++;
            }
            uint32_t h_fstats[4] = {};
            CU_CHECK(cuMemcpyDtoH_v2(h_fstats, d_fstats, sizeof(h_fstats)));
            hst.wave_step         = fluid_sim.step;
            hst.gpu_fluid_voxels  = h_fstats[0];
            hst.gpu_split_forks   = h_fstats[1];
            hst.gpu_collision_vox = h_fstats[2];
            hst.gpu_peak_vox_wt   = static_cast<float>(h_fstats[3]) * 0.001f;

            // Update progressive fluid filling of EVERY competing candidate answer!
            float wave_t = static_cast<float>(fluid_sim.step) / static_cast<float>(fluid_sim.max_steps);
            for (size_t ci = 0; ci < hst.candidates.size(); ++ci) {
                auto& c = hst.candidates[ci];
                // Recombined candidates surge sharply right after the midpoint collision phase (wave_t > 0.35)!
                float arrival = c.drop_sources >= 2 ? 0.30f : 0.22f;
                float fill = std::clamp((wave_t - arrival) / (1.0f - arrival), 0.0f, 1.0f);
                // Smooth S-curve filling
                fill = fill * fill * (3.0f - 2.0f * fill);
                c.current_ml = c.target_ml * fill;
                if (fill > 0.25f) {
                    hst.pos_activity[c.answer_pos] = std::max(hst.pos_activity[c.answer_pos], 0.45f + 0.55f * fill);
                    pending_word_flashes.push_back({fluid_sim.cand_anchor_wids[ci], -std::clamp(0.35f + 0.65f * fill, 0.0f, 1.0f)});
                }
            }
            if (!hst.candidates.empty() && wave_t > 0.35f) {
                char sbuf[160];
                std::snprintf(sbuf, sizeof(sbuf), "HIGHEST FLUID: '%s' (%.1f mL, %u/%u drops)!",
                              hst.candidates[0].answer.c_str(), hst.candidates[0].current_ml,
                              hst.candidates[0].drop_sources, hst.candidates[0].total_sources);
                hst.fluid_summary = sbuf;
            }
        }

        // Emit 128^3 Sponge + Live Voxel Fluid Weight Map into zero-copy VBO
        CU_CHECK(cuMemsetD32_v2(d_out_cnt, 0u, 1));
        CU_CHECK(cuGraphicsMapResources(1, &cu_sponge_res, nullptr));
        CUdeviceptr d_sp_vbo = 0; size_t sp_sz = 0;
        CU_CHECK(cuGraphicsResourceGetMappedPointer_v2(&d_sp_vbo, &sp_sz, cu_sponge_res));
        {
            uint32_t g = kGrid, maxv = kMaxSpongeVerts, fmode = fluid_sim.active ? 1u : 0u;
            float wr = SpongeEngine::kRad, minp = 0.08f, pdec = 0.92f;
            void* args[] = {&d_sponge, &d_meta, &d_fluid_a, &d_fmask_a, &d_sp_vbo, &d_out_cnt, &g, &maxv, &wr, &minp, &pdec, &fmode};
            CU_CHECK(cuLaunchKernel(fn_emit, (kTotalVox + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        }
        CU_CHECK(cuGraphicsUnmapResources(1, &cu_sponge_res, nullptr));
        CU_CHECK(cuMemcpyDtoH_v2(&emitted_sponge_verts, d_out_cnt, sizeof(uint32_t)));
        uint32_t draw_sponge_verts = std::min(emitted_sponge_verts, kMaxSpongeVerts);

        // Update 1.36M Outer Word VBO
        CU_CHECK(cuGraphicsMapResources(1, &cu_word_res, nullptr));
        CUdeviceptr d_wvbo = 0; size_t wsz = 0;
        CU_CHECK(cuGraphicsResourceGetMappedPointer_v2(&d_wvbo, &wsz, cu_word_res));
        {
            uint32_t nw = N_WORDS; float dec = 0.975f;
            void* args[] = {&d_wvbo, &nw, &dec};
            CU_CHECK(cuLaunchKernel(fn_decay_words, (N_WORDS + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
            for (const auto& wf : pending_word_flashes) {
                if (wf.first < N_WORDS) {
                    float val = wf.second;
                    CU_CHECK(cuMemcpyHtoD_v2(d_wvbo + static_cast<size_t>(wf.first) * 32u + 12u, &val, sizeof(float)));
                }
            }
            pending_word_flashes.clear();
        }
        CU_CHECK(cuGraphicsUnmapResources(1, &cu_word_res, nullptr));

        CU_CHECK(cuEventRecord(ev_e, nullptr));
        CU_CHECK(cuEventSynchronize(ev_e));
        CU_CHECK(cuEventElapsedTime(&hst.cuda_ms, ev_s, ev_e));

        hst.carved_voxels = emitted_sponge_verts;
        hst.merge_events  = sponge.merge_events;
        hst.split_events  = sponge.split_events;
        uint64_t tot_ev   = sponge.merge_events + sponge.split_events;
        hst.merge_ratio_pct = tot_ev ? (100.0f * static_cast<float>(sponge.merge_events) / static_cast<float>(tot_ev)) : 0.0f;

        // 10. Build Live 3D Branching Rays, Splitting Droplets & Candidate Answer Fluid Orbs!
        std::vector<LineVert> ray_lines;
        std::vector<SpriteVert> sprites;

        if (show_pos) {
            const auto& meta = gllm::pos_cluster_meta();
            for (uint32_t i = 0; i < gllm::kNumPosClusters; ++i) {
                float act = hst.pos_activity[i];
                const auto& p = pos_hubs[i];
                const auto& m = meta[i];
                sprites.push_back({p.x, p.y, p.z, 8.5f + act * 14.0f, m.r, m.g, m.b, 0.55f + act * 0.45f});
            }
        }

        auto append_poly = [&](const std::vector<Vec3>& poly, float r, float g, float b, float a, float progress, bool emit_drop, float drop_sz = 9.5f) {
            if (poly.size() < 2) return;
            float total_segs = static_cast<float>(poly.size() - 1);
            float fpos = std::clamp(progress, 0.0f, 1.0f) * total_segs;
            for (size_t i = 0; i + 1 < poly.size(); ++i) {
                float fi = static_cast<float>(i);
                if (fi >= fpos) break;
                float seg_t = std::min(1.0f, fpos - fi);
                Vec3 p0 = poly[i], p1 = lerp3(poly[i], poly[i + 1], seg_t);
                ray_lines.push_back({p0.x, p0.y, p0.z, r, g, b, a});
                ray_lines.push_back({p1.x, p1.y, p1.z, r, g, b, a});
                if (emit_drop && seg_t < 0.999f) {
                    sprites.push_back({p1.x, p1.y, p1.z, drop_sz, std::min(1.0f, r + 0.35f), std::min(1.0f, g + 0.35f), std::min(1.0f, b + 0.35f), 0.98f});
                }
            }
        };

        if (show_rays) {
            if (fluid_sim.active) {
                float wt = static_cast<float>(fluid_sim.step) / static_cast<float>(fluid_sim.max_steps);
                // Draw EVERY splitting branch from question words -> collision pools -> competing candidate answers!
                for (const auto& br : fluid_sim.branches) {
                    bool is_winner = (br.cand_idx == 0);
                    float r = is_winner ? 1.00f : (br.recombined ? 0.96f : 0.14f);
                    float g = is_winner ? 0.84f : (br.recombined ? 0.36f : 0.78f);
                    float b = is_winner ? 0.24f : (br.recombined ? 0.96f : 1.00f);
                    float a = is_winner ? 0.94f : (br.recombined ? 0.72f : 0.28f);
                    append_poly(br.poly, r, g, b, a, std::min(1.0f, wt * 1.35f), wt < 0.82f, is_winner ? 13.0f : 8.5f);
                }
                // Highlight multi-word collision pools inside the sponge!
                if (wt > 0.28f) {
                    for (const auto& cp : fluid_sim.collision_pools) {
                        sprites.push_back({cp.x, cp.y, cp.z, 17.0f, 1.00f, 0.42f, 0.95f, 0.95f});
                    }
                }
                // Draw EVERY competing candidate answer orb on the Outer Sphere scaled by its live fluid mL!
                float top_ml = hst.candidates.empty() ? 1.0f : std::max(1.0f, hst.candidates[0].target_ml);
                for (size_t ci = 0; ci < hst.candidates.size(); ++ci) {
                    const auto& c = hst.candidates[ci];
                    float frac = std::clamp(c.current_ml / top_ml, 0.0f, 1.0f);
                    const auto& pt = fluid_sim.cand_outer_pts[ci];
                    if (ci == 0) {
                        // Highest-Fluid Winning Answer Beacon!
                        sprites.push_back({pt.x, pt.y, pt.z, 14.0f + frac * 20.0f, 1.00f, 0.88f, 0.28f, 0.98f});
                    } else if (c.drop_sources >= 2) {
                        sprites.push_back({pt.x, pt.y, pt.z, 8.0f + frac * 14.0f, 0.96f, 0.44f, 0.98f, 0.88f});
                    } else {
                        sprites.push_back({pt.x, pt.y, pt.z, 5.5f + frac * 9.0f, 0.22f, 0.82f, 1.00f, 0.65f});
                    }
                }
            } else {
                for (size_t hi = 0; hi < history.size(); ++hi) {
                    float fade = 0.08f + 0.22f * (static_cast<float>(hi + 1) / static_cast<float>(history.size() + 1));
                    for (const auto& pl : history[hi].q_polylines) append_poly(pl, 0.10f, 0.78f, 1.00f, fade * 0.85f, 1.0f, false);
                    for (const auto& pl : history[hi].a_polylines) append_poly(pl, 1.00f, 0.62f, 0.16f, fade, 1.0f, false);
                }
                if (has_current) {
                    float q_prog = std::clamp(hst.anim_phase / 0.50f, 0.0f, 1.0f);
                    float a_prog = std::clamp((hst.anim_phase - 0.50f) / 0.50f, 0.0f, 1.0f);
                    for (const auto& pl : current_qa.q_polylines) append_poly(pl, 0.15f, 0.90f, 1.00f, 0.88f, q_prog, q_prog > 0.02f && q_prog < 0.99f);
                    if (hst.anim_phase >= 0.45f) sprites.push_back({current_qa.s_Q.x, current_qa.s_Q.y, current_qa.s_Q.z, 15.0f, 1.0f, 0.95f, 0.75f, 0.96f});
                    if (a_prog > 0.001f) {
                        for (const auto& pl : current_qa.a_polylines) append_poly(pl, 1.00f, 0.72f, 0.18f, 0.94f, a_prog, a_prog < 0.99f);
                    }
                }
            }
        }

        // 11. Render Frame in OpenGL 4.6
        int dw = 0, dh = 0;
        SDL_GL_GetDrawableSize(window, &dw, &dh);
        int left_w = std::min(gllm::TalkSidebarHUD::kPanelW, dw / 2);
        int right_w = std::max(1, dw - left_w);

        glViewport(0, 0, dw, dh);
        glClearColor(0.008f, 0.014f, 0.028f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glViewport(left_w, 0, right_w, dh);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(bg_prog);
        glBindVertexArray(bg_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        Vec3 cam_pos = {
            cam_dist * std::cos(cam_pitch) * std::sin(cam_yaw),
            cam_dist * std::sin(cam_pitch),
            cam_dist * std::cos(cam_pitch) * std::cos(cam_yaw)
        };
        Mat4 proj = mat4_perspective(45.0f * 3.14159265f / 180.0f, static_cast<float>(right_w) / std::max(1, dh), 0.02f, 40.0f);
        Mat4 view = mat4_look_at(cam_pos, {0, 0, 0}, {0, 1, 0});
        Mat4 mvp  = mat4_mul(proj, view);

        glEnable(GL_BLEND);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        // Wireframes
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(line_prog);
        glUniformMatrix4fv(glGetUniformLocation(line_prog, "uMVP"), 1, GL_FALSE, mvp.m);
        glBindVertexArray(wire_vao);
        if (show_words)  glDrawArrays(GL_LINES, static_cast<GLint>(off_outer), static_cast<GLsizei>(cnt_outer));
        if (show_pos)    glDrawArrays(GL_LINES, static_cast<GLint>(off_pos),   static_cast<GLsizei>(cnt_pos));
        if (show_sponge) glDrawArrays(GL_LINES, static_cast<GLint>(off_sp),    static_cast<GLsizei>(cnt_sp));

        // 128^3 CUDA Sponge + Live Voxel Fluid Weights & Recombination Hotspots!
        if (show_sponge && draw_sponge_verts > 0) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(spr_prog);
            glUniformMatrix4fv(glGetUniformLocation(spr_prog, "uMVP"), 1, GL_FALSE, mvp.m);
            glUniform3f(glGetUniformLocation(spr_prog, "uCamPos"), cam_pos.x, cam_pos.y, cam_pos.z);
            glUniform1f(glGetUniformLocation(spr_prog, "uViewportHeight"), static_cast<float>(dh));
            glUniform1f(glGetUniformLocation(spr_prog, "uScale"), fluid_sim.active ? 2.45f : 2.15f);
            glBindVertexArray(sponge_vao);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(draw_sponge_verts));
        }

        // Splitting/Recombining Fluid Rays
        if (show_rays && !ray_lines.empty()) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            glUseProgram(line_prog);
            glUniformMatrix4fv(glGetUniformLocation(line_prog, "uMVP"), 1, GL_FALSE, mvp.m);
            glBindVertexArray(ray_vao);
            glBindBuffer(GL_ARRAY_BUFFER, ray_vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(ray_lines.size() * sizeof(LineVert)), ray_lines.data());
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(ray_lines.size()));
        }

        // POS Hubs, Collision Pools, Splitting Droplets & Candidate Answer Fluid Orbs!
        if (!sprites.empty()) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            glUseProgram(spr_prog);
            glUniformMatrix4fv(glGetUniformLocation(spr_prog, "uMVP"), 1, GL_FALSE, mvp.m);
            glUniform3f(glGetUniformLocation(spr_prog, "uCamPos"), cam_pos.x, cam_pos.y, cam_pos.z);
            glUniform1f(glGetUniformLocation(spr_prog, "uViewportHeight"), static_cast<float>(dh));
            glUniform1f(glGetUniformLocation(spr_prog, "uScale"), 1.35f);
            glBindVertexArray(spr_vao);
            glBindBuffer(GL_ARRAY_BUFFER, spr_vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sprites.size() * sizeof(SpriteVert)), sprites.data());
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(sprites.size()));
        }

        // Outer Word Sphere (1.36M Kaikki words)
        if (show_words) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(word_prog);
            glUniformMatrix4fv(glGetUniformLocation(word_prog, "uMVP"), 1, GL_FALSE, mvp.m);
            glUniform3f(glGetUniformLocation(word_prog, "uCamPos"), cam_pos.x, cam_pos.y, cam_pos.z);
            glUniform1f(glGetUniformLocation(word_prog, "uViewportHeight"), static_cast<float>(dh));
            glUniform1f(glGetUniformLocation(word_prog, "uShellAlpha"), shell_alpha);
            glBindVertexArray(word_vao);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(N_WORDS));
        }

        glDepthMask(GL_TRUE);
        glDisable(GL_DEPTH_TEST);

        // Left Sidebar HUD
        hud.render(hst);
        glViewport(0, 0, dw, dh);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(hud_prog);
        float x1_ndc = -1.0f + 2.0f * (static_cast<float>(left_w) / static_cast<float>(dw));
        glUniform4f(glGetUniformLocation(hud_prog, "uRect"), -1.0f, -1.0f, x1_ndc, 1.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hud_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gllm::TalkSidebarHUD::kPanelW, gllm::TalkSidebarHUD::kPanelH,
                        GL_RGBA, GL_UNSIGNED_BYTE, hud.rgba.data());
        glBindVertexArray(hud_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);

        total_frames++;
        if (!screenshot_path.empty() && total_frames >= 8) {
            glFinish();
            save_ppm_screenshot(screenshot_path, dw, dh);
            if (!hst.candidates.empty()) {
                std::cout << "[FLUID]  Active Voxels: " << hst.gpu_fluid_voxels
                          << " | Forks: " << hst.gpu_split_forks
                          << " | Collisions(2+): " << hst.gpu_collision_vox
                          << " | Peak Wt: " << hst.gpu_peak_vox_wt << "\n";
                for (size_t i = 0; i < std::min<size_t>(6, hst.candidates.size()); ++i) {
                    const auto& c = hst.candidates[i];
                    std::cout << "  #" << (i + 1) << " " << c.answer << " -> " << c.current_ml
                              << " mL [" << c.drop_sources << "/" << c.total_sources << " drops"
                              << (c.pos_valve_ok ? " + POS valve" : "") << "]\n";
                }
            }
            running = false;
        }

        SDL_GL_SwapWindow(window);

        fps_timer += dt;
        fps_frames++;
        if (fps_timer >= 0.45) {
            hst.fps = static_cast<float>(fps_frames / fps_timer);
            char title[256];
            std::snprintf(title, sizeof(title),
                          "GLLM Fluid Sponge | %u Active Fluid Voxels | %u Collisions | %.1f FPS | CUDA %.2f ms",
                          hst.gpu_fluid_voxels, hst.gpu_collision_vox, hst.fps, hst.cuda_ms);
            SDL_SetWindowTitle(window, title);
            fps_timer = 0.0;
            fps_frames = 0;
        }
    }

    SDL_StopTextInput();
    CU_CHECK(cuGraphicsUnregisterResource(cu_word_res));
    CU_CHECK(cuGraphicsUnregisterResource(cu_sponge_res));
    CU_CHECK(cuMemFree_v2(d_sponge)); CU_CHECK(cuMemFree_v2(d_meta));
    CU_CHECK(cuMemFree_v2(d_fluid_a)); CU_CHECK(cuMemFree_v2(d_fmask_a));
    CU_CHECK(cuMemFree_v2(d_fluid_b)); CU_CHECK(cuMemFree_v2(d_fmask_b));
    CU_CHECK(cuMemFree_v2(d_segs)); CU_CHECK(cuMemFree_v2(d_drops));
    CU_CHECK(cuMemFree_v2(d_out_cnt)); CU_CHECK(cuMemFree_v2(d_fstats));
    CU_CHECK(cuEventDestroy_v2(ev_s)); CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(cu_mod)); CU_CHECK(cuCtxDestroy_v2(cu_ctx));
    SDL_GL_DeleteContext(gl_ctx); SDL_DestroyWindow(window); SDL_Quit();
    return 0;
}
