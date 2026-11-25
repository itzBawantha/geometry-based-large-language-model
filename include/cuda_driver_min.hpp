#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

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
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
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

namespace gllm {

inline std::string get_exe_dir() {
    char buf[4096] = {};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        std::string s(buf, static_cast<size_t>(n));
        auto pos = s.find_last_of('/');
        if (pos != std::string::npos) return s.substr(0, pos + 1);
    }
    return "./";
}

inline CUmodule load_gllm_ptx_module() {
    std::string dir = get_exe_dir();
    std::vector<std::string> candidates = {
        dir + "../kernels/gllm_kernels.ptx",
        dir + "kernels/gllm_kernels.ptx",
        "./kernels/gllm_kernels.ptx"
    };
    std::string ptx_src;
    for (const auto& p : candidates) {
        std::ifstream ifs(p);
        if (ifs.good()) {
            std::ostringstream ss;
            ss << ifs.rdbuf();
            ptx_src = ss.str();
            break;
        }
    }
    if (ptx_src.empty()) {
        std::cerr << "[ERROR] Could not locate kernels/gllm_kernels.ptx\n";
        std::exit(1);
    }

    char jit_info[4096] = {}, jit_err[4096] = {};
    int jit_opts[4] = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    void* jit_vals[4] = {
        jit_info, reinterpret_cast<void*>(sizeof(jit_info)),
        jit_err, reinterpret_cast<void*>(sizeof(jit_err))
    };

    CUmodule mod = nullptr;
    if (cuModuleLoadDataEx(&mod, ptx_src.c_str(), 4, jit_opts, jit_vals) != CUDA_SUCCESS) {
        std::cerr << "[CUDA JIT ERROR]\n" << jit_err << "\n";
        std::exit(1);
    }
    return mod;
}

} // namespace gllm
