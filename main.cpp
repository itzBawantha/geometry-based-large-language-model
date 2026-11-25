// =============================================================================
// CUDA Outer-Layer Point Sphere Visualizer (CUDA Driver API + OpenGL 4.6 Interop)
//
// Renders >= 1,000,000 (default 1,200,000) points forming the outer spherical
// shell, generated and animated every frame on the NVIDIA GPU via CUDA and
// displayed zero-copy via CUDA-OpenGL VBO interoperability.
// =============================================================================

#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// -----------------------------------------------------------------------------
// Minimal CUDA Driver API declarations (linked directly against libcuda.so)
// -----------------------------------------------------------------------------
extern "C" {
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef struct CUgraphicsResource_st* CUgraphicsResource;
typedef unsigned long long CUdeviceptr;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr unsigned int CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2;
constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSynchronize(void);
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            unsigned int numOptions, int* options,
                            void** optionValues);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra);
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult cuEventDestroy_v2(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource,
                                    unsigned int buffer, unsigned int Flags);
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);
CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources,
                                CUstream hStream);
CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources,
                                  CUstream hStream);
CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize,
                                               CUgraphicsResource resource);
CUresult cuGetErrorString(CUresult error, const char** pStr);
}

#define CU_CHECK(call)                                                         \
    do {                                                                       \
        CUresult _err = (call);                                                \
        if (_err != CUDA_SUCCESS) {                                            \
            const char* _msg = "unknown";                                      \
            cuGetErrorString(_err, &_msg);                                     \
            std::cerr << "[CUDA ERROR] " << #call << " failed (" << _err       \
                      << "): " << (_msg ? _msg : "unknown") << "\n";           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// Compact 3D Math Helpers (column-major OpenGL matrices)
// -----------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;
};

static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return len > 1e-8f ? Vec3{v.x / len, v.y / len, v.z / len} : Vec3{0, 0, 1};
}

struct Mat4 {
    float m[16] = {};
};

static Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[col * 4 + k];
            r.m[col * 4 + row] = s;
        }
    }
    return r;
}

static Mat4 mat4_perspective(float fov_rad, float aspect, float znear, float zfar) {
    Mat4 r;
    float f = 1.0f / std::tan(0.5f * fov_rad);
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
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
    r.m[3] = 0.0f; r.m[7] = 0.0f; r.m[11] = 0.0f; r.m[15] = 1.0f;
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
    float r = length(d);
    vec3 centerCol = vec3(0.032, 0.048, 0.090);
    vec3 edgeCol   = vec3(0.006, 0.009, 0.020);
    vec3 col = mix(centerCol, edgeCol, smoothstep(0.0, 1.05, r));
    fragColor = vec4(col, 1.0);
}
)GLSL";

static const char* kPointVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec4 aPosWave;
layout(location = 1) in vec4 aColor;

uniform mat4  uMVP;
uniform vec3  uCamPos;
uniform float uPointSize;
uniform float uViewportHeight;

out vec3  vColor;
out vec3  vNormal;
out vec3  vWorldPos;
out float vWave;

void main() {
    vec3 pos = aPosWave.xyz;
    vec4 clip = uMVP * vec4(pos, 1.0);
    gl_Position = clip;

    vWorldPos = pos;
    vNormal   = normalize(pos);
    vColor    = aColor.rgb;
    vWave     = aPosWave.w;

    float dist = max(length(uCamPos - pos), 0.15);
    float sz = uPointSize * (uViewportHeight / 900.0) * (2.6 / dist);
    gl_PointSize = clamp(sz, 1.25, 10.0);
}
)GLSL";

static const char* kPointFragSrc = R"GLSL(
#version 460 core
in vec3  vColor;
in vec3  vNormal;
in vec3  vWorldPos;
in float vWave;

uniform vec3 uCamPos;
uniform vec3 uLightDir;

out vec4 fragColor;

void main() {
    // Crisp circular point sprite with soft anti-aliased edge
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(L + V);

    // Subtle two-sided shading so flying inside the hollow point shell still looks 3D
    float ndotl = dot(N, L);
    float diff  = 0.24 + 0.76 * max(ndotl, 0.0) + 0.14 * max(-ndotl, 0.0);
    float spec  = pow(max(dot(N, H), 0.0), 36.0) * 0.45;
    float fres  = pow(1.0 - abs(dot(N, V)), 2.6);

    float coreGlow = 1.0 - 0.35 * r2;
    vec3 col = vColor * diff * coreGlow
             + vec3(0.55, 0.85, 1.0) * (fres * 0.42 + spec)
             + vColor * (vWave * 0.14);

    // Filmic-ish soft tone curve to avoid harsh clipping
    col = col / (vec3(1.0) + col * 0.30);
    float alpha = 1.0 - smoothstep(0.65, 1.0, r2);
    fragColor = vec4(col, alpha);
}
)GLSL";

static const char* kCoreVertSrc = R"GLSL(
#version 460 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
uniform float uRadius;
out vec3 vNormal;
void main() {
    vNormal = normalize(aPos);
    gl_Position = uMVP * vec4(aPos * uRadius, 1.0);
}
)GLSL";

static const char* kCoreFragSrc = R"GLSL(
#version 460 core
in vec3 vNormal;
uniform vec3 uCamPos;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos);
    float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 col = vec3(0.012, 0.018, 0.036) + vec3(0.03, 0.09, 0.18) * fres;
    fragColor = vec4(col, 1.0);
}
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

// Build a smooth UV-sphere mesh for the optional dark inner core
static void build_core_sphere(int stacks, int slices,
                              std::vector<float>& verts,
                              std::vector<uint32_t>& indices) {
    constexpr float kPi = 3.14159265358979323846f;
    for (int i = 0; i <= stacks; ++i) {
        float v = static_cast<float>(i) / stacks;
        float phi = v * kPi;
        for (int j = 0; j <= slices; ++j) {
            float u = static_cast<float>(j) / slices;
            float theta = u * 2.0f * kPi;
            verts.push_back(std::sin(phi) * std::cos(theta));
            verts.push_back(std::cos(phi));
            verts.push_back(std::sin(phi) * std::sin(theta));
        }
    }
    int ring = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = i * ring + j;
            uint32_t b = (i + 1) * ring + j;
            uint32_t c = (i + 1) * ring + (j + 1);
            uint32_t d = i * ring + (j + 1);
            indices.insert(indices.end(), {a, b, d, b, c, d});
        }
    }
}

static std::string load_ptx_file() {
    std::vector<std::string> candidates;
    char exe_buf[4096] = {};
    ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (n > 0) {
        std::string exe_path(exe_buf, static_cast<size_t>(n));
        auto slash = exe_path.find_last_of('/');
        if (slash != std::string::npos) {
            candidates.push_back(exe_path.substr(0, slash + 1) + "sphere_kernel.ptx");
        }
    }
    candidates.push_back("./sphere_kernel.ptx");

    for (const auto& path : candidates) {
        std::ifstream ifs(path);
        if (ifs.good()) {
            std::ostringstream ss;
            ss << ifs.rdbuf();
            return ss.str();
        }
    }
    std::cerr << "[ERROR] Could not find sphere_kernel.ptx\n";
    std::exit(1);
}

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

int main(int argc, char** argv) {
    uint32_t num_points = 1200000; // >= 1,000,000 points on the outer spherical shell
    std::string screenshot_path;
    uint32_t initial_mode = 0;
    uint32_t initial_palette = 0;
    float cam_dist = 2.85f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--points" || arg == "-n") && i + 1 < argc) {
            long long parsed = std::atoll(argv[++i]);
            if (parsed < 1000000) {
                std::cerr << "[WARN] Enforcing minimum 1,000,000 points (requested "
                          << parsed << ").\n";
                parsed = 1000000;
            }
            num_points = static_cast<uint32_t>(parsed);
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            initial_mode = static_cast<uint32_t>(std::atoi(argv[++i])) % 3;
        } else if (arg == "--palette" && i + 1 < argc) {
            initial_palette = static_cast<uint32_t>(std::atoi(argv[++i])) % 4;
        } else if (arg == "--dist" && i + 1 < argc) {
            cam_dist = std::max(0.35f, static_cast<float>(std::atof(argv[++i])));
        }
    }

    // -------------------------------------------------------------------------
    // 1. Initialize SDL2 + OpenGL 4.6 Core Profile Window
    // -------------------------------------------------------------------------
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "[SDL ERROR] " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    int win_w = 1280, win_h = 840;
    SDL_Window* window = SDL_CreateWindow(
        "CUDA Outer-Shell Point Sphere",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        win_w, win_h,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );
    if (!window) {
        std::cerr << "[SDL ERROR] SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    if (!gl_ctx) {
        std::cerr << "[SDL ERROR] SDL_GL_CreateContext failed: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_GL_SetSwapInterval(1);

    const char* gl_vendor   = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* gl_renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* gl_version  = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    std::cout << "[OpenGL] " << (gl_vendor ? gl_vendor : "?") << " | "
              << (gl_renderer ? gl_renderer : "?") << " | "
              << (gl_version ? gl_version : "?") << "\n";

    // -------------------------------------------------------------------------
    // 2. Initialize CUDA Driver API + JIT-compile PTX Kernel on NVIDIA GPU
    // -------------------------------------------------------------------------
    CU_CHECK(cuInit(0));
    int drv_ver = 0;
    CU_CHECK(cuDriverGetVersion(&drv_ver));

    CUdevice cu_dev = 0;
    CU_CHECK(cuDeviceGet(&cu_dev, 0));
    char gpu_name[256] = {};
    CU_CHECK(cuDeviceGetName(gpu_name, sizeof(gpu_name), cu_dev));
    std::cout << "[CUDA]   Device 0: " << gpu_name
              << " (Driver " << (drv_ver / 1000) << "." << ((drv_ver % 1000) / 10) << ")\n";

    CUcontext cu_ctx = nullptr;
    CU_CHECK(cuCtxCreate_v2(&cu_ctx, 0, cu_dev));

    std::string ptx_src = load_ptx_file();
    char jit_info[4096] = {};
    char jit_err[4096] = {};
    int jit_opts[4] = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    void* jit_vals[4] = {
        jit_info,
        reinterpret_cast<void*>(sizeof(jit_info)),
        jit_err,
        reinterpret_cast<void*>(sizeof(jit_err))
    };

    CUmodule cu_mod = nullptr;
    CUresult load_res = cuModuleLoadDataEx(&cu_mod, ptx_src.c_str(), 4, jit_opts, jit_vals);
    if (load_res != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(load_res, &msg);
        std::cerr << "[CUDA JIT ERROR] " << (msg ? msg : "unknown") << "\n" << jit_err << "\n";
        return 1;
    }

    CUfunction cu_kernel = nullptr;
    CU_CHECK(cuModuleGetFunction(&cu_kernel, cu_mod, "update_sphere_points"));

    CUevent ev_start = nullptr, ev_stop = nullptr;
    CU_CHECK(cuEventCreate(&ev_start, 0));
    CU_CHECK(cuEventCreate(&ev_stop, 0));

    // -------------------------------------------------------------------------
    // 3. Create Shared OpenGL VBO (32 bytes/point) & Register with CUDA Interop
    // -------------------------------------------------------------------------
    const size_t vbo_bytes = static_cast<size_t>(num_points) * 8 * sizeof(float);
    GLuint point_vao = 0, point_vbo = 0;
    glGenVertexArrays(1, &point_vao);
    glGenBuffers(1, &point_vbo);

    glBindVertexArray(point_vao);
    glBindBuffer(GL_ARRAY_BUFFER, point_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vbo_bytes), nullptr, GL_DYNAMIC_DRAW);

    // location 0: vec4 pos_wave (offset 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), reinterpret_cast<void*>(0));
    // location 1: vec4 color (offset 16)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), reinterpret_cast<void*>(16));
    glBindVertexArray(0);

    CUgraphicsResource cu_vbo_res = nullptr;
    CU_CHECK(cuGraphicsGLRegisterBuffer(&cu_vbo_res, point_vbo,
                                        CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    std::cout << "[INFO]   Allocated " << num_points << " outer-shell points ("
              << (vbo_bytes / (1024.0 * 1024.0)) << " MiB GPU VBO, zero-copy CUDA-GL interop)\n";

    // Build optional dark inner core sphere mesh
    std::vector<float> core_verts;
    std::vector<uint32_t> core_idx;
    build_core_sphere(48, 72, core_verts, core_idx);

    GLuint core_vao = 0, core_vbo = 0, core_ebo = 0;
    glGenVertexArrays(1, &core_vao);
    glGenBuffers(1, &core_vbo);
    glGenBuffers(1, &core_ebo);
    glBindVertexArray(core_vao);
    glBindBuffer(GL_ARRAY_BUFFER, core_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(core_verts.size() * sizeof(float)),
                 core_verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, core_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(core_idx.size() * sizeof(uint32_t)),
                 core_idx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);

    GLuint bg_vao = 0;
    glGenVertexArrays(1, &bg_vao);

    // Compile shader programs
    GLuint bg_prog    = link_program(kBgVertSrc, kBgFragSrc);
    GLuint point_prog = link_program(kPointVertSrc, kPointFragSrc);
    GLuint core_prog  = link_program(kCoreVertSrc, kCoreFragSrc);

    GLint u_pt_mvp   = glGetUniformLocation(point_prog, "uMVP");
    GLint u_pt_cam   = glGetUniformLocation(point_prog, "uCamPos");
    GLint u_pt_size  = glGetUniformLocation(point_prog, "uPointSize");
    GLint u_pt_vh    = glGetUniformLocation(point_prog, "uViewportHeight");
    GLint u_pt_light = glGetUniformLocation(point_prog, "uLightDir");

    GLint u_core_mvp = glGetUniformLocation(core_prog, "uMVP");
    GLint u_core_rad = glGetUniformLocation(core_prog, "uRadius");
    GLint u_core_cam = glGetUniformLocation(core_prog, "uCamPos");

    // -------------------------------------------------------------------------
    // 4. Interactive Camera & Controls State
    // -------------------------------------------------------------------------
    float cam_yaw   = 0.55f;
    float cam_pitch = 0.28f;
    float point_size = 1.45f;
    bool  dragging   = false;
    bool  paused     = false;
    bool  auto_rot   = true;
    bool  show_core  = false;
    uint32_t mode    = initial_mode;
    uint32_t palette = initial_palette;

    const char* mode_names[3]    = {"Pure Outer Sphere", "Harmonic Ripples", "Outer Shell Crust"};
    const char* palette_names[4] = {"Cosmic", "Solar", "Hologram", "Normal RGB"};

    std::cout << "\nControls:\n"
              << "  Left-drag       : Orbit camera\n"
              << "  Scroll / +/-    : Zoom in/out (zoom close to inspect individual points)\n"
              << "  Space           : Pause/resume CUDA wave animation\n"
              << "  W               : Cycle CUDA outer-shell mode\n"
              << "  M               : Cycle color palette\n"
              << "  A               : Toggle auto-rotation\n"
              << "  C               : Toggle dark inner core occlusion\n"
              << "  [ / ]           : Decrease / increase point size\n"
              << "  R               : Reset camera\n"
              << "  Q / Esc         : Quit\n\n";

    float sim_time = 0.0f;
    uint64_t prev_ticks = SDL_GetPerformanceCounter();
    const double perf_freq = static_cast<double>(SDL_GetPerformanceFrequency());
    double fps_timer = 0.0;
    int fps_frames = 0;
    float last_cuda_ms = 0.0f;
    int frame_count = 0;

    bool running = true;
    while (running) {
        uint64_t now_ticks = SDL_GetPerformanceCounter();
        double dt = static_cast<double>(now_ticks - prev_ticks) / perf_freq;
        prev_ticks = now_ticks;
        if (dt > 0.1) dt = 0.1;

        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
                dragging = true;
            } else if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
            } else if (ev.type == SDL_MOUSEMOTION && dragging) {
                cam_yaw   -= ev.motion.xrel * 0.0065f;
                cam_pitch += ev.motion.yrel * 0.0065f;
                cam_pitch  = std::clamp(cam_pitch, -1.52f, 1.52f);
            } else if (ev.type == SDL_MOUSEWHEEL) {
                cam_dist *= (ev.wheel.y > 0) ? 0.90f : 1.11f;
                cam_dist  = std::clamp(cam_dist, 0.35f, 12.0f);
            } else if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                    case SDLK_ESCAPE:
                    case SDLK_q:
                        running = false;
                        break;
                    case SDLK_SPACE:
                        paused = !paused;
                        break;
                    case SDLK_w:
                        mode = (mode + 1) % 3;
                        break;
                    case SDLK_m:
                        palette = (palette + 1) % 4;
                        break;
                    case SDLK_a:
                        auto_rot = !auto_rot;
                        break;
                    case SDLK_c:
                        show_core = !show_core;
                        break;
                    case SDLK_LEFTBRACKET:
                        point_size = std::max(1.0f, point_size - 0.25f);
                        break;
                    case SDLK_RIGHTBRACKET:
                        point_size = std::min(6.0f, point_size + 0.25f);
                        break;
                    case SDLK_EQUALS:
                    case SDLK_PLUS:
                        cam_dist = std::max(0.35f, cam_dist * 0.88f);
                        break;
                    case SDLK_MINUS:
                        cam_dist = std::min(12.0f, cam_dist * 1.14f);
                        break;
                    case SDLK_r:
                        cam_yaw = 0.55f;
                        cam_pitch = 0.28f;
                        cam_dist = 2.85f;
                        point_size = 1.9f;
                        break;
                    default:
                        break;
                }
            }
        }

        if (!paused) {
            sim_time += static_cast<float>(dt);
        }
        if (auto_rot && !dragging && screenshot_path.empty()) {
            cam_yaw += static_cast<float>(dt) * 0.22f;
        }
        if (!screenshot_path.empty()) {
            sim_time = 1.35f; // Deterministic visually rich wave phase for capture
        }

        // ---------------------------------------------------------------------
        // 5. Execute CUDA Kernel directly on Mapped OpenGL VBO (Zero-Copy)
        // ---------------------------------------------------------------------
        CU_CHECK(cuGraphicsMapResources(1, &cu_vbo_res, nullptr));
        CUdeviceptr d_ptr = 0;
        size_t mapped_size = 0;
        CU_CHECK(cuGraphicsResourceGetMappedPointer_v2(&d_ptr, &mapped_size, cu_vbo_res));

        constexpr uint32_t kBlockSize = 256;
        uint32_t grid_size = (num_points + kBlockSize - 1) / kBlockSize;

        void* kernel_params[] = {
            &d_ptr,
            &num_points,
            &sim_time,
            &mode,
            &palette
        };

        CU_CHECK(cuEventRecord(ev_start, nullptr));
        CU_CHECK(cuLaunchKernel(
            cu_kernel,
            grid_size, 1, 1,
            kBlockSize, 1, 1,
            0, nullptr,
            kernel_params, nullptr
        ));
        CU_CHECK(cuEventRecord(ev_stop, nullptr));
        CU_CHECK(cuEventSynchronize(ev_stop));
        CU_CHECK(cuEventElapsedTime(&last_cuda_ms, ev_start, ev_stop));
        CU_CHECK(cuGraphicsUnmapResources(1, &cu_vbo_res, nullptr));

        // ---------------------------------------------------------------------
        // 6. Render Frame in OpenGL 4.6
        // ---------------------------------------------------------------------
        SDL_GL_GetDrawableSize(window, &win_w, &win_h);
        glViewport(0, 0, win_w, win_h);
        glClearColor(0.01f, 0.015f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Background radial space gradient
        glDisable(GL_DEPTH_TEST);
        glUseProgram(bg_prog);
        glBindVertexArray(bg_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Camera matrices
        Vec3 cam_pos = {
            cam_dist * std::cos(cam_pitch) * std::sin(cam_yaw),
            cam_dist * std::sin(cam_pitch),
            cam_dist * std::cos(cam_pitch) * std::cos(cam_yaw)
        };
        float aspect = static_cast<float>(win_w) / std::max(1, win_h);
        Mat4 proj = mat4_perspective(45.0f * 3.14159265f / 180.0f, aspect, 0.02f, 50.0f);
        Mat4 view = mat4_look_at(cam_pos, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        Mat4 mvp  = mat4_mul(proj, view);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        // Optional dark inner core
        if (show_core) {
            glUseProgram(core_prog);
            glUniformMatrix4fv(u_core_mvp, 1, GL_FALSE, mvp.m);
            glUniform1f(u_core_rad, 0.945f);
            glUniform3f(u_core_cam, cam_pos.x, cam_pos.y, cam_pos.z);
            glBindVertexArray(core_vao);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(core_idx.size()), GL_UNSIGNED_INT, nullptr);
        }

        // Draw all >= 1,000,000 outer-shell points directly from CUDA-updated VBO
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(point_prog);
        glUniformMatrix4fv(u_pt_mvp, 1, GL_FALSE, mvp.m);
        glUniform3f(u_pt_cam, cam_pos.x, cam_pos.y, cam_pos.z);
        glUniform1f(u_pt_size, point_size);
        glUniform1f(u_pt_vh, static_cast<float>(win_h));
        glUniform3f(u_pt_light, 0.65f, 0.72f, 0.55f);

        glBindVertexArray(point_vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(num_points));
        glBindVertexArray(0);
        glDisable(GL_BLEND);

        frame_count++;
        if (!screenshot_path.empty() && frame_count >= 4) {
            glFinish();
            save_ppm_screenshot(screenshot_path, win_w, win_h);
            std::cout << "[BENCH]  CUDA kernel time for " << num_points
                      << " points: " << last_cuda_ms << " ms\n";
            running = false;
        }

        SDL_GL_SwapWindow(window);

        // Update HUD window title twice per second
        fps_timer += dt;
        fps_frames++;
        if (fps_timer >= 0.45) {
            double fps = fps_frames / fps_timer;
            char title[256];
            std::snprintf(
                title, sizeof(title),
                "CUDA Point Sphere | %u pts | %.1f FPS | CUDA %.2f ms | %s | %s | %s",
                num_points, fps, last_cuda_ms,
                mode_names[mode], palette_names[palette], gpu_name
            );
            SDL_SetWindowTitle(window, title);
            fps_timer = 0.0;
            fps_frames = 0;
        }
    }

    // -------------------------------------------------------------------------
    // 7. Clean Shutdown
    // -------------------------------------------------------------------------
    CU_CHECK(cuGraphicsUnregisterResource(cu_vbo_res));
    CU_CHECK(cuEventDestroy_v2(ev_start));
    CU_CHECK(cuEventDestroy_v2(ev_stop));
    CU_CHECK(cuModuleUnload(cu_mod));
    CU_CHECK(cuCtxDestroy_v2(cu_ctx));

    glDeleteProgram(point_prog);
    glDeleteProgram(core_prog);
    glDeleteProgram(bg_prog);
    glDeleteBuffers(1, &point_vbo);
    glDeleteVertexArrays(1, &point_vao);
    glDeleteBuffers(1, &core_vbo);
    glDeleteBuffers(1, &core_ebo);
    glDeleteVertexArrays(1, &core_vao);
    glDeleteVertexArrays(1, &bg_vao);

    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
