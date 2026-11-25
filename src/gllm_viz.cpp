// =============================================================================
// gllm_viz — Interactive 3D CUDA-OpenGL Geometric Sphere Visualizer + Sketch Inference
//
// Features:
//   - Left Sidebar: Interactive 224x224 mouse sketch canvas, live NIST-normalized
//     28x28 CUDA preview, [CAPTURE] & [CLEAR] buttons, Open-World Unknown check,
//     Stage-1 64-bit bitmask Pruned report, and clickable Top-10 overlap bars.
//   - Right 3D Viewport: 1-billion-point (1241^3) implicit sphere lattice with
//     O(Allocated) CUDA rendering, 1.2M outer-shell VBO, and 1-to-1 binding rays.
// =============================================================================

#include "../include/cuda_driver_min.hpp"
#include "../include/gllm_model.hpp"
#include "../include/hud_canvas.hpp"

#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Vec3 { float x, y, z; };
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return len > 1e-8f ? Vec3{v.x / len, v.y / len, v.z / len} : Vec3{0, 0, 1};
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
    vec2 d = (vUV - vec2(0.5)) * vec2(1.4, 1.0);
    vec3 col = mix(vec3(0.020, 0.032, 0.062), vec3(0.004, 0.007, 0.016), smoothstep(0.0, 1.05, length(d)));
    fragColor = vec4(col, 1.0);
}
)GLSL";

static const char* kShellVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec4 aPosVal;
layout(location = 1) in vec4 aColor;

uniform mat4  uMVP;
uniform vec3  uCamPos;
uniform float uViewportHeight;
uniform float uShellAlpha;
uniform int   uFocusClass;
uniform int   uNumClasses;

out vec3  vColor;
out vec3  vNormal;
out vec3  vWorldPos;
out float vVal;
out float vFocusScale;
out float vGhostAlpha;

void main() {
    float val = aPosVal.w;
    vVal = val;
    vGhostAlpha = uShellAlpha;

    if (val < -0.5 && uShellAlpha <= 0.001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vColor = vec3(0.0);
        vNormal = vec3(0.0, 0.0, 1.0);
        vWorldPos = vec3(0.0);
        vFocusScale = 0.0;
        return;
    }

    vec3 pos = aPosVal.xyz;
    gl_Position = uMVP * vec4(pos, 1.0);
    vWorldPos = pos;
    vNormal   = normalize(pos);
    vColor    = aColor.rgb;

    float dist = max(length(uCamPos - pos), 0.12);
    if (val >= -0.5) {
        int iv = int(val + 0.5);
        bool focused = (uFocusClass >= uNumClasses || iv == uFocusClass);
        vFocusScale = focused ? 1.0 : 0.25;
        float baseSz = (uFocusClass < uNumClasses && iv == uFocusClass) ? 12.0 : 6.8;
        gl_PointSize = clamp(baseSz * (uViewportHeight / 900.0) * (2.6 / dist), 4.0, 20.0);
    } else {
        vFocusScale = 1.0;
        gl_PointSize = clamp(1.20 * (uViewportHeight / 900.0) * (2.6 / dist), 1.0, 4.0);
    }
}
)GLSL";

static const char* kShellFragSrc = R"GLSL(
#version 460 core
in vec3  vColor;
in vec3  vNormal;
in vec3  vWorldPos;
in float vVal;
in float vFocusScale;
in float vGhostAlpha;

uniform vec3 uCamPos;
out vec4 fragColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    if (vVal >= -0.5) {
        float core = 1.0 - smoothstep(0.0, 0.40, r2);
        float halo = 1.0 - smoothstep(0.25, 1.0, r2);
        vec3 col = mix(vColor, vec3(1.0), core * 0.70) * vFocusScale;
        fragColor = vec4(col, halo * (0.30 + 0.70 * vFocusScale));
        return;
    }

    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 L = normalize(vec3(0.65, 0.72, 0.55));
    float fres = pow(1.0 - abs(dot(N, V)), 2.2);
    float diff = 0.35 + 0.65 * max(dot(N, L), 0.0);
    vec3 col = vColor * diff + vec3(0.45, 0.85, 1.0) * (fres * 0.55);
    float glassAlpha = mix(vGhostAlpha * (0.28 + 1.35 * fres), 1.0, smoothstep(0.75, 1.0, vGhostAlpha));
    fragColor = vec4(col, clamp(glassAlpha, 0.0, 1.0) * (1.0 - smoothstep(0.60, 1.0, r2)));
}
)GLSL";

static const char* kLatticeVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec4 aPosVal;
layout(location = 1) in vec4 aColor;

uniform mat4  uMVP;
uniform vec3  uCamPos;
uniform float uPointSize;
uniform float uViewportHeight;

out vec4 vColor;

void main() {
    vec3 pos = aPosVal.xyz;
    gl_Position = uMVP * vec4(pos, 1.0);
    vColor = aColor;
    float dist = max(length(uCamPos - pos), 0.05);
    float d = aPosVal.w;
    float boost = 1.15 + 1.05 * d;
    gl_PointSize = clamp(uPointSize * boost * (uViewportHeight / 900.0) * (1.2 / dist), 1.4, 15.0);
}
)GLSL";

static const char* kLatticeFragSrc = R"GLSL(
#version 460 core
in vec4 vColor;
out vec4 fragColor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    float soft = 1.0 - smoothstep(0.35, 1.0, r2);
    float core = 1.0 - 0.45 * r2;
    fragColor = vec4(vColor.rgb * (0.90 + 0.35 * core), vColor.a * soft);
}
)GLSL";

static const char* kLinkVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aClass;

uniform mat4 uMVP;
uniform int  uFocusClass;
uniform int  uNumClasses;
out vec4 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    int c = int(aClass + 0.5);
    float s = (uFocusClass >= uNumClasses) ? 0.32 : ((c == uFocusClass) ? 1.0 : 0.04);
    vColor = vec4(aColor.rgb, aColor.a * s);
}
)GLSL";

static const char* kLinkFragSrc = R"GLSL(
#version 460 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)GLSL";

static const char* kSidebarVertSrc = R"GLSL(
#version 460 core
uniform vec4 uRect; // x0, y0, x1, y1 in NDC
out vec2 vUV;
void main() {
    vec2 uvs[6] = vec2[6](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,0), vec2(1,1), vec2(0,1));
    vec2 uv = uvs[gl_VertexID];
    vUV = vec2(uv.x, 1.0 - uv.y);
    gl_Position = vec4(mix(uRect.xy, uRect.zw, uv), 0.0, 1.0);
}
)GLSL";

static const char* kSidebarFragSrc = R"GLSL(
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

static void save_ppm(const std::string& path, int w, int h) {
    std::vector<unsigned char> px(static_cast<size_t>(w) * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    std::ofstream ofs(path, std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (int y = h - 1; y >= 0; --y) {
        ofs.write(reinterpret_cast<const char*>(&px[static_cast<size_t>(y) * w * 3]),
                  static_cast<std::streamsize>(w) * 3);
    }
    std::cout << "[INFO]   Saved screenshot (" << w << "x" << h << ") to " << path << "\n";
}

int main(int argc, char** argv) {
    std::string model_path, screenshot_path, test_idx_img;
    int test_idx_pos = -1;
    bool test_transpose = false;
    uint32_t heatmap_mode = 0;
    int forced_focus = -1;
    float init_dist = -1.0f;
    float prune_ratio = 0.68f; // Stage-1 64-bit canonical mask overlap ratio threshold

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--screenshot" && i + 1 < argc) screenshot_path = argv[++i];
        else if (a == "--heatmap") heatmap_mode = 1u;
        else if (a == "--focus" && i + 1 < argc) forced_focus = std::atoi(argv[++i]);
        else if (a == "--dist" && i + 1 < argc) init_dist = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--prune-ratio" && i + 1 < argc) prune_ratio = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--test-img" && i + 2 < argc) { test_idx_img = argv[++i]; test_idx_pos = std::atoi(argv[++i]); }
        else if (a == "--test-transpose") test_transpose = true;
    }

    if (model_path.empty()) {
        std::cerr << "Usage: gllm_viz --model <path.gllm> [--heatmap] [--prune-ratio 0.65] [--screenshot out.ppm]\n";
        return 1;
    }

    gllm::GeometricModel model;
    if (!model.load(model_path)) {
        std::cerr << "[ERROR] Cannot load model: " << model_path << "\n";
        return 1;
    }

    uint32_t num_classes = static_cast<uint32_t>(model.classes.size());
    if (num_classes == 0) {
        std::cerr << "[ERROR] Model has 0 allocated classes.\n";
        return 1;
    }

    // Build GPU tables dynamically from .gllm (zero hardcoding to 62!)
    std::vector<gllm::GpuRegionDesc> h_regions;
    std::vector<gllm::GpuOuterPoint> h_outer;
    std::vector<float>    h_shadow_avg;
    std::vector<uint64_t> h_class_masks(num_classes, 0ULL);
    std::vector<float>    h_log_odds;
    std::vector<float>    h_class_bias(num_classes, 0.0f);

    uint32_t max_reg_pix = 0, total_alloc_pix = 0;
    float max_alloc_radius = 0.05f;
    float step = 2.0f / static_cast<float>(model.lattice_grid);
    uint32_t match_pix = 784u; // 28x28
    constexpr uint32_t kNumShifts = 9u; // dx, dy in {-1, 0, +1}

    h_log_odds.resize(static_cast<size_t>(num_classes) * match_pix, 0.0f);

    for (uint32_t c = 0; c < num_classes; ++c) {
        const auto& rec = model.classes[c];
        uint32_t pix = rec.width * rec.height;
        max_reg_pix = std::max(max_reg_pix, pix);

        h_regions.push_back({rec.ox, rec.oy, rec.oz, rec.width, rec.height, rec.outer_idx, total_alloc_pix, c});
        h_outer.push_back({rec.outer_x, rec.outer_y, rec.outer_z, static_cast<float>(c)});

        float inv = rec.train_count > 0 ? (1.0f / static_cast<float>(rec.train_count)) : 0.0f;
        std::vector<float> avg(pix, 0.0f);
        float raw_peak = 0.0f;
        for (uint32_t p = 0; p < pix; ++p) {
            avg[p] = rec.shadow_sum[p] * inv;
            raw_peak = std::max(raw_peak, avg[p]);
            h_shadow_avg.push_back(avg[p]);
        }

        if (pix == match_pix) {
            // Canonicalize class shadow into unified 22x22 frame (removes MNIST 20px vs EMNIST 24px scale bias)
            float pnorm[784] = {};
            gllm::canonicalize_28(avg.data(), pnorm, 22.0f, 0.22f);

            // Stage-1 64-bit coarse footprint mask in canonical space (core shadow support > 0.22)
            h_class_masks[c] = gllm::compute_coarse_mask64(pnorm, 0.22f);

            // Stage-2 Non-Linear Bernoulli Log-Likelihood tables preserving true shadow probability p:
            //   LL(x, c) = sum[ x*log(p) + (1-x)*log(1-p) ] = bias[c] + sum[ x * log(p/(1-p)) ]
            float* lo_dst = &h_log_odds[static_cast<size_t>(c) * match_pix];
            double bias_sum = 0.0;
            for (uint32_t p = 0; p < match_pix; ++p) {
                double prob = std::clamp(static_cast<double>(pnorm[p]) * static_cast<double>(raw_peak), 0.015, 0.96);
                lo_dst[p] = static_cast<float>(std::log(prob / (1.0 - prob)));
                bias_sum += std::log(1.0 - prob);
            }
            h_class_bias[c] = static_cast<float>(bias_sum);
        }
        total_alloc_pix += pix;

        float cx = (rec.ox + rec.width * 0.5f) * step - 1.0f;
        float cy = (rec.oy + rec.height * 0.5f) * step - 1.0f;
        float cz = (rec.oz + 0.5f) * step - 1.0f;
        max_alloc_radius = std::max(max_alloc_radius, std::sqrt(cx * cx + cy * cy + cz * cz) + rec.width * step);
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    int win_w = 1360, win_h = 840;
    SDL_Window* window = SDL_CreateWindow(
        "GLLM 1B-Lattice Interactive Sketch Inference & 3D Visualizer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        win_w, win_h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );
    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(1);

    CU_CHECK(cuInit(0));
    CUdevice dev = 0;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext cu_ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&cu_ctx, 0, dev));

    CUmodule mod = gllm::load_gllm_ptx_module();
    CUfunction fn_shell = nullptr, fn_emit = nullptr, fn_match = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_shell, mod, "init_outer_shell_dynamic"));
    CU_CHECK(cuModuleGetFunction(&fn_emit,  mod, "emit_allocated_regions"));
    CU_CHECK(cuModuleGetFunction(&fn_match, mod, "match_sketch_pruned"));

    CUevent ev_s = nullptr, ev_e = nullptr;
    CU_CHECK(cuEventCreate(&ev_s, 0));
    CU_CHECK(cuEventCreate(&ev_e, 0));

    // Upload GPU model & matcher buffers
    CUdeviceptr d_regions = 0, d_outer = 0, d_shadow = 0, d_counter = 0;
    CUdeviceptr d_masks = 0, d_log_odds = 0, d_bias = 0, d_sketch_raw = 0, d_sketch_ent = 0;
    CUdeviceptr d_out_ov = 0, d_out_st = 0;
    CU_CHECK(cuMemAlloc_v2(&d_regions, h_regions.size() * sizeof(gllm::GpuRegionDesc)));
    CU_CHECK(cuMemcpyHtoD_v2(d_regions, h_regions.data(), h_regions.size() * sizeof(gllm::GpuRegionDesc)));
    CU_CHECK(cuMemAlloc_v2(&d_outer, h_outer.size() * sizeof(gllm::GpuOuterPoint)));
    CU_CHECK(cuMemcpyHtoD_v2(d_outer, h_outer.data(), h_outer.size() * sizeof(gllm::GpuOuterPoint)));
    CU_CHECK(cuMemAlloc_v2(&d_shadow, h_shadow_avg.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_shadow, h_shadow_avg.data(), h_shadow_avg.size() * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_counter, sizeof(uint32_t)));

    CU_CHECK(cuMemAlloc_v2(&d_masks,    h_class_masks.size() * sizeof(uint64_t)));
    CU_CHECK(cuMemcpyHtoD_v2(d_masks,   h_class_masks.data(), h_class_masks.size() * sizeof(uint64_t)));
    CU_CHECK(cuMemAlloc_v2(&d_log_odds, h_log_odds.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_log_odds, h_log_odds.data(), h_log_odds.size() * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_bias,     h_class_bias.size() * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD_v2(d_bias,    h_class_bias.data(), h_class_bias.size() * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_sketch_raw, static_cast<size_t>(kNumShifts) * match_pix * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_sketch_ent, static_cast<size_t>(kNumShifts) * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_out_ov, num_classes * sizeof(float)));
    CU_CHECK(cuMemAlloc_v2(&d_out_st, num_classes * sizeof(uint32_t)));

    // --- VBO 1: Permanent 1,200,000-point outer sphere ---
    uint32_t total_outer = model.outer_shell_pts;
    GLuint shell_vao = 0, shell_vbo = 0;
    glGenVertexArrays(1, &shell_vao);
    glGenBuffers(1, &shell_vbo);
    glBindVertexArray(shell_vao);
    glBindBuffer(GL_ARRAY_BUFFER, shell_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(size_t(total_outer) * 32), nullptr, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(16));
    glBindVertexArray(0);

    CUgraphicsResource cu_shell_res = nullptr;
    CU_CHECK(cuGraphicsGLRegisterBuffer(&cu_shell_res, shell_vbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    {
        CU_CHECK(cuGraphicsMapResources(1, &cu_shell_res, nullptr));
        CUdeviceptr d_sh = 0;
        size_t sz = 0;
        CU_CHECK(cuGraphicsResourceGetMappedPointer_v2(&d_sh, &sz, cu_shell_res));
        void* p[] = {&d_sh, &total_outer, &d_outer, &num_classes};
        CU_CHECK(cuLaunchKernel(fn_shell, (total_outer + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, p, nullptr));
        CU_CHECK(cuGraphicsUnmapResources(1, &cu_shell_res, nullptr));
    }

    // --- VBO 2: Allocated interior shadow points ---
    uint32_t max_vbo_pts = std::max<uint32_t>(total_alloc_pix, 4096u);
    GLuint lat_vao = 0, lat_vbo = 0;
    glGenVertexArrays(1, &lat_vao);
    glGenBuffers(1, &lat_vbo);
    glBindVertexArray(lat_vao);
    glBindBuffer(GL_ARRAY_BUFFER, lat_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(size_t(max_vbo_pts) * 32), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, reinterpret_cast<void*>(16));
    glBindVertexArray(0);

    CUgraphicsResource cu_lat_res = nullptr;
    CU_CHECK(cuGraphicsGLRegisterBuffer(&cu_lat_res, lat_vbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    // --- VBO 3: 1-to-1 Binding lines ---
    struct LinkVert { float x, y, z, r, g, b, a, cid; };
    std::vector<LinkVert> links;
    for (uint32_t c = 0; c < num_classes; ++c) {
        const auto& rec = model.classes[c];
        float ix = (rec.ox + rec.width * 0.5f) * step - 1.0f;
        float iy = (rec.oy + rec.height * 0.5f) * step - 1.0f;
        float iz = (rec.oz + 0.5f) * step - 1.0f;
        float ph = rec.outer_idx * 2.4f;
        float cr = 0.55f + 0.45f * std::cos(ph);
        float cg = 0.55f + 0.45f * std::cos(ph + 2.0943951f);
        float cb = 0.55f + 0.45f * std::cos(ph + 4.1887902f);
        links.push_back({ix, iy, iz, cr, cg, cb, 0.35f, static_cast<float>(c)});
        links.push_back({rec.outer_x, rec.outer_y, rec.outer_z, cr, cg, cb, 0.85f, static_cast<float>(c)});
    }
    GLuint link_vao = 0, link_vbo = 0;
    glGenVertexArrays(1, &link_vao);
    glGenBuffers(1, &link_vbo);
    glBindVertexArray(link_vao);
    glBindBuffer(GL_ARRAY_BUFFER, link_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(links.size() * sizeof(LinkVert)), links.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LinkVert), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(LinkVert), reinterpret_cast<void*>(12));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(LinkVert), reinterpret_cast<void*>(28));
    glBindVertexArray(0);

    // Sidebar RGBA8 texture
    GLuint sidebar_tex = 0;
    glGenTextures(1, &sidebar_tex);
    glBindTexture(GL_TEXTURE_2D, sidebar_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, gllm::SketchSidebarHUD::kPanelW, gllm::SketchSidebarHUD::kPanelH,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    GLuint empty_vao = 0;
    glGenVertexArrays(1, &empty_vao);

    GLuint bg_prog      = link_program(kBgVertSrc, kBgFragSrc);
    GLuint shell_prog   = link_program(kShellVertSrc, kShellFragSrc);
    GLuint lat_prog     = link_program(kLatticeVertSrc, kLatticeFragSrc);
    GLuint link_prog    = link_program(kLinkVertSrc, kLinkFragSrc);
    GLuint sidebar_prog = link_program(kSidebarVertSrc, kSidebarFragSrc);

    GLint u_sh_mvp   = glGetUniformLocation(shell_prog, "uMVP");
    GLint u_sh_cam   = glGetUniformLocation(shell_prog, "uCamPos");
    GLint u_sh_vh    = glGetUniformLocation(shell_prog, "uViewportHeight");
    GLint u_sh_alpha = glGetUniformLocation(shell_prog, "uShellAlpha");
    GLint u_sh_foc   = glGetUniformLocation(shell_prog, "uFocusClass");
    GLint u_sh_num   = glGetUniformLocation(shell_prog, "uNumClasses");

    GLint u_lt_mvp   = glGetUniformLocation(lat_prog, "uMVP");
    GLint u_lt_cam   = glGetUniformLocation(lat_prog, "uCamPos");
    GLint u_lt_size  = glGetUniformLocation(lat_prog, "uPointSize");
    GLint u_lt_vh    = glGetUniformLocation(lat_prog, "uViewportHeight");

    GLint u_lk_mvp   = glGetUniformLocation(link_prog, "uMVP");
    GLint u_lk_foc   = glGetUniformLocation(link_prog, "uFocusClass");
    GLint u_lk_num   = glGetUniformLocation(link_prog, "uNumClasses");

    GLint u_sb_rect  = glGetUniformLocation(sidebar_prog, "uRect");

    gllm::SketchSidebarHUD hud;
    gllm::MatchSummary match_summary;

    float close_dist  = std::clamp(max_alloc_radius * 2.55f, 0.35f, 2.2f);
    float cam_dist    = (init_dist > 0.0f) ? init_dist : close_dist;
    float cam_yaw     = 0.28f;
    float cam_pitch   = 0.16f;
    float pt_size     = 2.1f;
    float shell_alpha = 0.0f;
    uint32_t focus_cls = (forced_focus >= 0 && forced_focus < static_cast<int>(num_classes))
                             ? static_cast<uint32_t>(forced_focus)
                             : num_classes;
    bool orbit_dragging  = false;
    bool canvas_drawing  = false;
    float prev_cx = 0.0f, prev_cy = 0.0f;
    bool auto_rot   = true;
    bool show_links = true;
    bool dirty_lat  = true;

    // -------------------------------------------------------------------------
    // Run 2-Stage CUDA Non-Linear Bernoulli Log-Likelihood Match on 28x28 Sketch!
    // -------------------------------------------------------------------------
    auto run_cuda_capture = [&]() {
        double base_mean = 0.0;
        for (uint32_t p = 0; p < match_pix; ++p) base_mean += hud.norm28[p];
        base_mean /= match_pix;
        if (base_mean < 1e-4) {
            std::cout << "[CAPTURE] Canvas is empty — draw a digit or letter first.\n";
            return;
        }

        // Build 9 micro-shifted variants (dx, dy in {-1, 0, +1}) + exact self-entropy H(x)
        std::vector<float> h_sk_raw(static_cast<size_t>(kNumShifts) * match_pix, 0.0f);
        std::vector<float> h_sk_ent(kNumShifts, 0.0f);
        uint32_t s_idx = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                float* raw_dst = &h_sk_raw[static_cast<size_t>(s_idx) * match_pix];
                for (int y = 0; y < 28; ++y) {
                    int sy = y - dy;
                    if (sy < 0 || sy >= 28) continue;
                    for (int x = 0; x < 28; ++x) {
                        int sx = x - dx;
                        if (sx >= 0 && sx < 28) raw_dst[y * 28 + x] = hud.norm28[sy * 28 + sx];
                    }
                }
                double ent = 0.0;
                for (uint32_t p = 0; p < match_pix; ++p) {
                    double xc = std::clamp(static_cast<double>(raw_dst[p]), 1e-5, 1.0 - 1e-5);
                    ent += xc * std::log(xc) + (1.0 - xc) * std::log(1.0 - xc);
                }
                h_sk_ent[s_idx] = static_cast<float>(ent);
                s_idx++;
            }
        }
        uint64_t q_mask = gllm::compute_coarse_mask64(hud.norm28.data(), 0.24f);
        uint32_t num_shifts = kNumShifts;
        float inv_t_ln2 = static_cast<float>(1.0 / (200.0 * std::log(2.0)));

        CU_CHECK(cuMemcpyHtoD_v2(d_sketch_raw, h_sk_raw.data(), h_sk_raw.size() * sizeof(float)));
        CU_CHECK(cuMemcpyHtoD_v2(d_sketch_ent, h_sk_ent.data(), h_sk_ent.size() * sizeof(float)));

        void* m_params[] = {
            &d_sketch_raw, &d_sketch_ent, &q_mask, &d_masks, &d_log_odds, &d_bias,
            &d_out_ov, &d_out_st, &match_pix, &num_classes, &num_shifts, &prune_ratio, &inv_t_ln2
        };
        CU_CHECK(cuEventRecord(ev_s, nullptr));
        CU_CHECK(cuLaunchKernel(fn_match, (num_classes + 63u) / 64u, 1, 1, 64, 1, 1, 0, nullptr, m_params, nullptr));
        CU_CHECK(cuEventRecord(ev_e, nullptr));
        CU_CHECK(cuEventSynchronize(ev_e));

        float ms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&ms, ev_s, ev_e));

        std::vector<float>    h_ov(num_classes);
        std::vector<uint32_t> h_st(num_classes);
        CU_CHECK(cuMemcpyDtoH_v2(h_ov.data(), d_out_ov, num_classes * sizeof(float)));
        CU_CHECK(cuMemcpyDtoH_v2(h_st.data(), d_out_st, num_classes * sizeof(uint32_t)));

        match_summary = gllm::MatchSummary{};
        match_summary.has_result    = true;
        match_summary.cuda_ms       = ms;
        match_summary.total_classes = num_classes;

        for (uint32_t c = 0; c < num_classes; ++c) {
            const auto& rec = model.classes[c];
            if (h_st[c] == 0u) {
                match_summary.pruned_count++;
                match_summary.pruned_labels.push_back(rec.label);
            } else {
                match_summary.survived_count++;
                match_summary.ranked.push_back({c, rec.outer_idx, rec.label, h_ov[c] * 100.0f, 0.0f, false});
            }
        }

        std::sort(match_summary.ranked.begin(), match_summary.ranked.end(),
                  [](const gllm::MatchResultRow& a, const gllm::MatchResultRow& b) {
                      return a.overlap_pct > b.overlap_pct;
                  });

        // Compute relative probability % among surviving finalists (temperature = 10.0)
        if (!match_summary.ranked.empty()) {
            float max_ov = match_summary.ranked[0].overlap_pct / 100.0f;
            double sum_exp = 0.0;
            for (const auto& r : match_summary.ranked) {
                sum_exp += std::exp((r.overlap_pct / 100.0f - max_ov) * 10.0);
            }
            for (auto& r : match_summary.ranked) {
                r.rel_prob_pct = static_cast<float>(100.0 * std::exp((r.overlap_pct / 100.0f - max_ov) * 10.0) / sum_exp);
            }
            match_summary.is_unknown = (match_summary.ranked[0].overlap_pct < 35.0f);
            focus_cls = match_summary.is_unknown ? num_classes : match_summary.ranked[0].class_id;
        } else {
            match_summary.is_unknown = true;
            focus_cls = num_classes;
        }
        dirty_lat = true;
        hud.dirty_tex = true;

        // Print crisp diagnostic table to terminal
        std::printf("\n[CAPTURE] CUDA 2-Stage Match: %.3f ms | Survived: %u / %u | Pruned (Stage 1): %u / %u\n",
                    ms, match_summary.survived_count, num_classes, match_summary.pruned_count, num_classes);
        if (match_summary.is_unknown) {
            std::printf("  -> RESULT: UNKNOWN / UNTRAINED SHAPE (top overlap %.1f%% < 35.0%% threshold)\n",
                        match_summary.ranked.empty() ? 0.0f : match_summary.ranked[0].overlap_pct);
        } else {
            const auto& w = match_summary.ranked[0];
            std::printf("  -> WINNER: '%s' (Outer Point #%.1f) | Overlap: %.1f%% | Relative Prob: %.1f%%\n",
                        w.label.c_str(), static_cast<float>(w.outer_idx), w.overlap_pct, w.rel_prob_pct);
        }
        int topk = std::min<int>(10, static_cast<int>(match_summary.ranked.size()));
        for (int i = 0; i < topk; ++i) {
            const auto& r = match_summary.ranked[i];
            std::printf("     #%2d  %-4s (Outer #%-4.1f)  Overlap: %5.1f%%  | RelProb: %5.1f%%\n",
                        i + 1, r.label.c_str(), static_cast<float>(r.outer_idx), r.overlap_pct, r.rel_prob_pct);
        }
        std::cout << "  -> Pruned as improbable (" << match_summary.pruned_count << "): ";
        for (size_t i = 0; i < match_summary.pruned_labels.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << match_summary.pruned_labels[i];
        }
        std::cout << "\n";
    };

    // Optional automated test-image capture for verification screenshots
    if (!test_idx_img.empty() && test_idx_pos >= 0) {
        gzFile f = gzopen(test_idx_img.c_str(), "rb");
        if (f) {
            gllm::read_gz_be32(f);
            uint32_t cnt = gllm::read_gz_be32(f);
            gllm::read_gz_be32(f);
            gllm::read_gz_be32(f);
            if (static_cast<uint32_t>(test_idx_pos) < cnt) {
                std::vector<uint8_t> tmp(784);
                for (int k = 0; k <= test_idx_pos; ++k) gzread(f, tmp.data(), 784);
                if (test_transpose) {
                    std::vector<uint8_t> tr(784);
                    for (int y = 0; y < 28; ++y)
                        for (int x = 0; x < 28; ++x)
                            tr[y * 28 + x] = tmp[x * 28 + y];
                    tmp.swap(tr);
                }
                hud.load_raw_28x28(tmp.data());
                run_cuda_capture();
            }
            gzclose(f);
        }
    }

    std::cout << "[VIZ]    Loaded " << model_path << " (" << num_classes << " classes inside "
              << model.approx_sphere_lattice_points() << "-pt sphere)\n"
              << "[UI]     Draw on the left canvas -> click [CAPTURE] (or Enter) to run 2-stage CUDA match!\n";

    uint64_t prev_ticks = SDL_GetPerformanceCounter();
    const double perf_freq = static_cast<double>(SDL_GetPerformanceFrequency());
    double fps_timer = 0.0;
    int fps_frames = 0, frame_count = 0;
    float last_emit_ms = 0.0f;
    uint32_t emitted_pts = 0;

    auto map_mouse_to_hud = [&](int mx, int my, int& hx, int& hy) {
        int ww = 1, wh = 1;
        SDL_GetWindowSize(window, &ww, &wh);
        float scale_y = static_cast<float>(gllm::SketchSidebarHUD::kPanelH) / std::max(1, wh);
        hx = static_cast<int>(mx * scale_y);
        hy = static_cast<int>(my * scale_y);
    };

    bool running = true;
    while (running) {
        uint64_t now_ticks = SDL_GetPerformanceCounter();
        double dt = std::min(0.1, static_cast<double>(now_ticks - prev_ticks) / perf_freq);
        prev_ticks = now_ticks;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            else if (ev.type == SDL_MOUSEBUTTONDOWN) {
                int hx = 0, hy = 0;
                map_mouse_to_hud(ev.button.x, ev.button.y, hx, hy);
                if (hx < gllm::SketchSidebarHUD::kPanelW) {
                    if (ev.button.button == SDL_BUTTON_RIGHT && hud.canvas_rect.contains(hx, hy)) {
                        hud.clear();
                        match_summary = gllm::MatchSummary{};
                        focus_cls = num_classes;
                        dirty_lat = true;
                    } else if (ev.button.button == SDL_BUTTON_LEFT) {
                        if (hud.canvas_rect.contains(hx, hy)) {
                            canvas_drawing = true;
                            prev_cx = static_cast<float>(hx - hud.canvas_rect.x);
                            prev_cy = static_cast<float>(hy - hud.canvas_rect.y);
                            hud.draw_stroke(prev_cx, prev_cy, prev_cx, prev_cy);
                        } else if (hud.btn_capture.contains(hx, hy)) {
                            run_cuda_capture();
                        } else if (hud.btn_clear.contains(hx, hy)) {
                            hud.clear();
                            match_summary = gllm::MatchSummary{};
                            focus_cls = num_classes;
                            dirty_lat = true;
                        } else if (match_summary.has_result) {
                            for (int r = 0; r < 10 && r < static_cast<int>(match_summary.ranked.size()); ++r) {
                                if (hud.row_rects[r].contains(hx, hy)) {
                                    focus_cls = match_summary.ranked[r].class_id;
                                    dirty_lat = true;
                                    hud.dirty_tex = true;
                                }
                            }
                        }
                    }
                } else if (ev.button.button == SDL_BUTTON_LEFT) {
                    orbit_dragging = true;
                }
            } else if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
                orbit_dragging = false;
                canvas_drawing = false;
            } else if (ev.type == SDL_MOUSEMOTION) {
                int hx = 0, hy = 0;
                map_mouse_to_hud(ev.motion.x, ev.motion.y, hx, hy);
                bool old_hc = hud.hover_capture, old_hcl = hud.hover_clear;
                int  old_hr = hud.hover_row;
                hud.hover_capture = hud.btn_capture.contains(hx, hy);
                hud.hover_clear   = hud.btn_clear.contains(hx, hy);
                hud.hover_row     = -1;
                for (int r = 0; r < 10; ++r) if (hud.row_rects[r].contains(hx, hy)) hud.hover_row = r;
                if (old_hc != hud.hover_capture || old_hcl != hud.hover_clear || old_hr != hud.hover_row)
                    hud.dirty_tex = true;

                if (canvas_drawing) {
                    float cx = std::clamp(static_cast<float>(hx - hud.canvas_rect.x), 0.0f, float(gllm::SketchSidebarHUD::kCanvasS - 1));
                    float cy = std::clamp(static_cast<float>(hy - hud.canvas_rect.y), 0.0f, float(gllm::SketchSidebarHUD::kCanvasS - 1));
                    hud.draw_stroke(prev_cx, prev_cy, cx, cy);
                    prev_cx = cx; prev_cy = cy;
                } else if (orbit_dragging) {
                    cam_yaw   -= ev.motion.xrel * 0.0065f;
                    cam_pitch  = std::clamp(cam_pitch + ev.motion.yrel * 0.0065f, -1.52f, 1.52f);
                }
            } else if (ev.type == SDL_MOUSEWHEEL) {
                cam_dist = std::clamp(cam_dist * ((ev.wheel.y > 0) ? 0.90f : 1.11f), 0.12f, 10.0f);
            } else if (ev.type == SDL_KEYDOWN) {
                SDL_Keycode k = ev.key.keysym.sym;
                if (k == SDLK_ESCAPE || k == SDLK_q) running = false;
                else if (k == SDLK_RETURN || k == SDLK_KP_ENTER) run_cuda_capture();
                else if (k == SDLK_c || k == SDLK_BACKSPACE) {
                    hud.clear(); match_summary = gllm::MatchSummary{}; focus_cls = num_classes; dirty_lat = true;
                } else if (k == SDLK_SPACE) { focus_cls = num_classes; dirty_lat = true; hud.dirty_tex = true; }
                else if (k == SDLK_RIGHT) { focus_cls = (focus_cls >= num_classes) ? 0u : ((focus_cls + 1u) % num_classes); dirty_lat = true; hud.dirty_tex = true; }
                else if (k == SDLK_LEFT)  { focus_cls = (focus_cls >= num_classes) ? (num_classes - 1u) : ((focus_cls + num_classes - 1u) % num_classes); dirty_lat = true; hud.dirty_tex = true; }
                else if (k == SDLK_h) { heatmap_mode = 1u - heatmap_mode; dirty_lat = true; }
                else if (k == SDLK_l) show_links = !show_links;
                else if (k == SDLK_a) auto_rot = !auto_rot;
                else if (k == SDLK_z) cam_dist = (cam_dist > close_dist * 1.4f) ? close_dist : 2.65f;
                else if (k == SDLK_UP)   shell_alpha = std::min(1.0f, shell_alpha + 0.08f);
                else if (k == SDLK_DOWN) shell_alpha = std::max(0.0f, shell_alpha - 0.08f);
                else if (k == SDLK_LEFTBRACKET)  pt_size = std::max(0.8f, pt_size - 0.25f);
                else if (k == SDLK_RIGHTBRACKET) pt_size = std::min(6.0f, pt_size + 0.25f);
                else if (k == SDLK_r) { cam_yaw = 0.28f; cam_pitch = 0.16f; cam_dist = close_dist; shell_alpha = 0.0f; focus_cls = num_classes; dirty_lat = true; hud.dirty_tex = true; }
            }
        }

        if (auto_rot && !orbit_dragging && screenshot_path.empty()) cam_yaw += static_cast<float>(dt) * 0.20f;

        if (dirty_lat) {
            CU_CHECK(cuMemsetD32_v2(d_counter, 0u, 1));
            CU_CHECK(cuGraphicsMapResources(1, &cu_lat_res, nullptr));
            CUdeviceptr d_lat = 0;
            size_t sz = 0;
            CU_CHECK(cuGraphicsResourceGetMappedPointer_v2(&d_lat, &sz, cu_lat_res));

            uint32_t g = model.lattice_grid;
            uint32_t total_work = num_classes * max_reg_pix;
            void* p[] = {
                &d_lat, &d_counter, &d_regions, &d_shadow,
                &num_classes, &max_reg_pix, &g, &max_vbo_pts,
                &focus_cls, &heatmap_mode
            };
            CU_CHECK(cuEventRecord(ev_s, nullptr));
            CU_CHECK(cuLaunchKernel(fn_emit, (total_work + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, p, nullptr));
            CU_CHECK(cuEventRecord(ev_e, nullptr));
            CU_CHECK(cuEventSynchronize(ev_e));
            CU_CHECK(cuEventElapsedTime(&last_emit_ms, ev_s, ev_e));
            CU_CHECK(cuGraphicsUnmapResources(1, &cu_lat_res, nullptr));
            CU_CHECK(cuMemcpyDtoH_v2(&emitted_pts, d_counter, sizeof(uint32_t)));
            if (emitted_pts > max_vbo_pts) emitted_pts = max_vbo_pts;
            dirty_lat = false;
        }

        if (hud.dirty_tex) {
            hud.rebuild_texture(match_summary, focus_cls, num_classes);
            glBindTexture(GL_TEXTURE_2D, sidebar_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gllm::SketchSidebarHUD::kPanelW, gllm::SketchSidebarHUD::kPanelH,
                            GL_RGBA, GL_UNSIGNED_BYTE, hud.rgba.data());
        }

        SDL_GL_GetDrawableSize(window, &win_w, &win_h);
        int panel_px = static_cast<int>(gllm::SketchSidebarHUD::kPanelW * (static_cast<float>(win_h) / gllm::SketchSidebarHUD::kPanelH));
        panel_px = std::min(panel_px, win_w / 2);

        // Background across full window
        glViewport(0, 0, win_w, win_h);
        glClearColor(0.01f, 0.015f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(bg_prog);
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // 3D Sphere Viewport on the right of the Sketch Sidebar
        int vp_x = panel_px, vp_w = std::max(1, win_w - panel_px);
        glViewport(vp_x, 0, vp_w, win_h);

        Vec3 cam_pos = {
            cam_dist * std::cos(cam_pitch) * std::sin(cam_yaw),
            cam_dist * std::sin(cam_pitch),
            cam_dist * std::cos(cam_pitch) * std::cos(cam_yaw)
        };
        float aspect = static_cast<float>(vp_w) / std::max(1, win_h);
        Mat4 proj = mat4_perspective(45.0f * 3.14159265f / 180.0f, aspect, 0.01f, 50.0f);
        Mat4 view = mat4_look_at(cam_pos, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        Mat4 mvp  = mat4_mul(proj, view);

        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (show_links) {
            glUseProgram(link_prog);
            glUniformMatrix4fv(u_lk_mvp, 1, GL_FALSE, mvp.m);
            glUniform1i(u_lk_foc, static_cast<int>(focus_cls));
            glUniform1i(u_lk_num, static_cast<int>(num_classes));
            glBindVertexArray(link_vao);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(links.size()));
        }

        glUseProgram(lat_prog);
        glUniformMatrix4fv(u_lt_mvp, 1, GL_FALSE, mvp.m);
        glUniform3f(u_lt_cam, cam_pos.x, cam_pos.y, cam_pos.z);
        glUniform1f(u_lt_size, pt_size);
        glUniform1f(u_lt_vh, static_cast<float>(win_h));
        glBindVertexArray(lat_vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(emitted_pts));

        glUseProgram(shell_prog);
        glUniformMatrix4fv(u_sh_mvp, 1, GL_FALSE, mvp.m);
        glUniform3f(u_sh_cam, cam_pos.x, cam_pos.y, cam_pos.z);
        glUniform1f(u_sh_vh, static_cast<float>(win_h));
        glUniform1f(u_sh_alpha, shell_alpha);
        glUniform1i(u_sh_foc, static_cast<int>(focus_cls));
        glUniform1i(u_sh_num, static_cast<int>(num_classes));
        glBindVertexArray(shell_vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(total_outer));
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);

        // Draw Interactive Left Sidebar HUD over full window
        glViewport(0, 0, win_w, win_h);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(sidebar_prog);
        float sb_x1 = -1.0f + 2.0f * static_cast<float>(panel_px) / std::max(1, win_w);
        glUniform4f(u_sb_rect, -1.0f, -1.0f, sb_x1, 1.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sidebar_tex);
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);

        frame_count++;
        if (!screenshot_path.empty() && frame_count >= 10) {
            glFinish();
            save_ppm(screenshot_path, win_w, win_h);
            running = false;
        }

        SDL_GL_SwapWindow(window);

        fps_timer += dt;
        fps_frames++;
        if (fps_timer >= 0.40) {
            double fps = fps_frames / fps_timer;
            char title[256];
            std::snprintf(
                title, sizeof(title),
                "GLLM 1B-Sphere Sketch Inference | %u Classes | Outer %u/1.2M | Emit %.3f ms | %.1f FPS",
                num_classes, num_classes, last_emit_ms, fps
            );
            SDL_SetWindowTitle(window, title);
            fps_timer = 0.0;
            fps_frames = 0;
        }
    }

    CU_CHECK(cuGraphicsUnregisterResource(cu_shell_res));
    CU_CHECK(cuGraphicsUnregisterResource(cu_lat_res));
    CU_CHECK(cuMemFree_v2(d_regions));
    CU_CHECK(cuMemFree_v2(d_outer));
    CU_CHECK(cuMemFree_v2(d_shadow));
    CU_CHECK(cuMemFree_v2(d_counter));
    CU_CHECK(cuMemFree_v2(d_masks));
    CU_CHECK(cuMemFree_v2(d_log_odds));
    CU_CHECK(cuMemFree_v2(d_bias));
    CU_CHECK(cuMemFree_v2(d_sketch_raw));
    CU_CHECK(cuMemFree_v2(d_sketch_ent));
    CU_CHECK(cuMemFree_v2(d_out_ov));
    CU_CHECK(cuMemFree_v2(d_out_st));
    CU_CHECK(cuEventDestroy_v2(ev_s));
    CU_CHECK(cuEventDestroy_v2(ev_e));
    CU_CHECK(cuModuleUnload(mod));
    CU_CHECK(cuCtxDestroy_v2(cu_ctx));
    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
