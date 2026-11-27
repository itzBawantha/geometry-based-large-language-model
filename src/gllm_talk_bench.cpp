// =============================================================================
// GLLM 4-Sphere Fluid Sponge Benchmark (100-Query Real GPU + CPU Telemetry)
//   Measures exact model size, CPU RAM, GPU VRAM, CSR fluid latency,
//   RTX A3000 128^3 CUDA voxel wave latency, active voxels, multi-word
//   collisions, and raw Top-1 / Top-5 answers across 100 queries (ZERO levers!).
// =============================================================================

#include "../include/cuda_driver_min.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" CUresult cuMemGetInfo_v2(size_t* free, size_t* total);

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};
static inline float dot3(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float len3(const Vec3& a) { return std::sqrt(dot3(a, a)); }
static inline Vec3 norm3(const Vec3& a) { float l = len3(a); return l > 1e-6f ? a * (1.0f / l) : Vec3{0, 1, 0}; }
static inline Vec3 lerp3(const Vec3& a, const Vec3& b, float t) { return a * (1.0f - t) + b * t; }

struct Float4 { float x, y, z, w; };
struct CarveSeg { Float4 p0, p1, dir_role; };
struct FluidDrop { Float4 pos_amt; uint32_t mask, pad0; float pad1, pad2; };

static inline uint32_t fnv1a_32(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) { h ^= c; h *= 16777619u; }
    return h;
}

static Vec3 word_outer_xyz(uint32_t wid, uint32_t total_words) {
    constexpr float kGolden = 2.39996322972865332f;
    float t = (static_cast<float>(wid) + 0.5f) / static_cast<float>(std::max(1u, total_words));
    float y = 1.0f - 2.0f * t;
    float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
    float th = kGolden * static_cast<float>(wid);
    return {std::cos(th) * r, y, std::sin(th) * r};
}

template <size_t N>
static std::array<Vec3, N> build_sphere_hubs(float radius, float phase) {
    std::array<Vec3, N> out{};
    constexpr float kGolden = 2.39996322972865332f;
    for (size_t i = 0; i < N; ++i) {
        float y = 1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(N);
        float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float th = kGolden * static_cast<float>(i) + phase;
        out[i] = {std::cos(th) * r * radius, y * radius, std::sin(th) * r * radius};
    }
    return out;
}

static size_t read_rss_kb() {
    std::ifstream ifs("/proc/self/status");
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            size_t kb = 0;
            iss >> kb;
            return kb;
        }
    }
    return 0;
}

struct Lexicon {
    uint32_t num_words = 0;
    std::vector<uint8_t> primary_pos;
    std::unordered_map<std::string, uint32_t> word_to_id;

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        char magic[4] = {};
        ifs.read(magic, 4);
        if (std::memcmp(magic, "GLXL", 4) != 0) return false;
        uint32_t ver = 0, n_pos = 0;
        ifs.read(reinterpret_cast<char*>(&ver), 4);
        ifs.read(reinterpret_cast<char*>(&num_words), 4);
        ifs.read(reinterpret_cast<char*>(&n_pos), 4);
        primary_pos.resize(num_words);
        word_to_id.reserve(static_cast<size_t>(num_words) * 1.2);
        std::string w;
        for (uint32_t i = 0; i < num_words; ++i) {
            uint8_t pid = 0, wlen = 0; uint16_t mask = 0;
            ifs.read(reinterpret_cast<char*>(&pid), 1);
            ifs.read(reinterpret_cast<char*>(&wlen), 1);
            ifs.read(reinterpret_cast<char*>(&mask), 2);
            w.resize(wlen);
            ifs.read(w.data(), wlen);
            primary_pos[i] = pid;
            word_to_id.emplace(w, i);
        }
        return true;
    }

    struct Tok { uint32_t word_id; uint8_t pos; };
    std::vector<Tok> tokenize(const std::string& text) const {
        std::vector<Tok> out;
        size_t i = 0, n = text.size();
        while (i < n) {
            while (i < n && !std::isalnum(static_cast<unsigned char>(text[i]))) ++i;
            if (i >= n) break;
            size_t s = i;
            while (i < n && (std::isalnum(static_cast<unsigned char>(text[i])) ||
                             (text[i] == '\'' && i + 1 < n && std::isalpha(static_cast<unsigned char>(text[i + 1]))))) {
                ++i;
            }
            std::string raw = text.substr(s, i - s);
            std::string low = raw;
            for (char& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            auto it = word_to_id.find(low);
            if (it != word_to_id.end()) {
                out.push_back({it->second, primary_pos[it->second]});
            } else {
                uint32_t wid = fnv1a_32(low) % std::max(1u, num_words);
                bool has_dig = false;
                for (unsigned char c : raw) if (std::isdigit(c)) has_dig = true;
                uint8_t pid = has_dig ? 9u : (std::isupper(static_cast<unsigned char>(raw[0])) ? 4u : 0u);
                out.push_back({wid, pid});
            }
        }
        return out;
    }
};

struct AnswerNode {
    std::string text;
    uint32_t anchor_word_id = 0;
    uint8_t  primary_pos    = 0;
    uint32_t total_hits     = 0;
};
struct CsrWordMeta { uint64_t total_count = 0; uint32_t edge_off = 0; uint32_t edge_cnt = 0; };
struct CsrEdge { uint32_t aid; uint32_t count; };

struct SpongeIndex {
    static constexpr uint32_t kNumPos = 12u;
    uint32_t qa_indexed = 0;
    uint64_t total_transitions = 0;
    std::vector<AnswerNode> answers;
    std::vector<CsrWordMeta> csr_words;
    std::vector<CsrEdge> csr_edges;
    std::unordered_map<uint32_t, std::array<uint32_t, kNumPos>> opener_to_ans_pos;
    std::vector<float> log_hits;
    std::vector<float> acc_ml;
    std::vector<uint32_t> acc_mask;
    std::vector<uint32_t> touched_aids;

    bool load(const std::string& path, uint32_t expected_num_words) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        char magic[4] = {}; ifs.read(magic, 4);
        if (std::memcmp(magic, "GSPG", 4) != 0) return false;
        #pragma pack(push, 1)
        struct BinHdr { uint32_t ver, num_words, qa_count, num_answers; uint64_t ans_blob_len; uint32_t num_active_words; uint64_t num_edges; uint32_t num_openers; uint64_t total_transitions; };
        struct BinAnsMeta { uint32_t anchor_wid, hits; uint8_t primary_pos, pad; uint16_t text_len; uint32_t text_off; };
        struct BinWordIdx { uint32_t wid; uint64_t total_count; uint32_t edge_off, edge_cnt; };
        struct BinOpener { uint32_t wid; uint32_t cnts[12]; };
        #pragma pack(pop)
        BinHdr hdr{}; ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!ifs || hdr.ver != 1u) return false;
        qa_indexed = hdr.qa_count;
        total_transitions = hdr.total_transitions;
        ifs.seekg(12 * 8 * 8 + 8 * 8 * 8, std::ios::cur);

        std::vector<BinAnsMeta> ameta(hdr.num_answers);
        ifs.read(reinterpret_cast<char*>(ameta.data()), static_cast<std::streamsize>(ameta.size() * sizeof(BinAnsMeta)));
        std::string ablob(hdr.ans_blob_len, '\0');
        ifs.read(ablob.data(), static_cast<std::streamsize>(hdr.ans_blob_len));
        answers.resize(hdr.num_answers);
        for (uint32_t i = 0; i < hdr.num_answers; ++i) {
            answers[i].text.assign(ablob.data() + ameta[i].text_off, ameta[i].text_len);
            answers[i].anchor_word_id = ameta[i].anchor_wid;
            answers[i].primary_pos = ameta[i].primary_pos;
            answers[i].total_hits = ameta[i].hits;
        }
        uint32_t nw = std::max(expected_num_words, hdr.num_words);
        csr_words.assign(nw, CsrWordMeta{});
        std::vector<BinWordIdx> widx(hdr.num_active_words);
        ifs.read(reinterpret_cast<char*>(widx.data()), static_cast<std::streamsize>(widx.size() * sizeof(BinWordIdx)));
        for (const auto& w : widx) if (w.wid < nw) csr_words[w.wid] = {w.total_count, w.edge_off, w.edge_cnt};
        csr_edges.resize(hdr.num_edges);
        ifs.read(reinterpret_cast<char*>(csr_edges.data()), static_cast<std::streamsize>(csr_edges.size() * sizeof(CsrEdge)));
        std::vector<BinOpener> ops(hdr.num_openers);
        ifs.read(reinterpret_cast<char*>(ops.data()), static_cast<std::streamsize>(ops.size() * sizeof(BinOpener)));
        opener_to_ans_pos.reserve(hdr.num_openers);
        for (const auto& op : ops) {
            std::array<uint32_t, 12> arr{};
            for (int p = 0; p < 12; ++p) arr[p] = op.cnts[p];
            opener_to_ans_pos.emplace(op.wid, arr);
        }
        log_hits.resize(answers.size());
        for (size_t i = 0; i < answers.size(); ++i) {
            log_hits[i] = std::log(static_cast<float>(std::max<uint32_t>(1u, answers[i].total_hits)));
        }
        acc_ml.assign(answers.size(), 0.0f);
        acc_mask.assign(answers.size(), 0u);
        return true;
    }
};

static CUmodule load_ptx(const std::string& ptx_path) {
    std::ifstream ifs(ptx_path);
    if (!ifs) { std::cerr << "[ERROR] Cannot open PTX: " << ptx_path << "\n"; std::exit(1); }
    std::stringstream ss; ss << ifs.rdbuf();
    std::string src = ss.str();
    CUmodule mod = nullptr;
    CU_CHECK(cuModuleLoadDataEx(&mod, src.c_str(), 0, nullptr, nullptr));
    return mod;
}

static std::string resolve_data_file(const std::string& filename) {
    std::string exe_dir = gllm::get_exe_dir();
    std::vector<std::string> cands = {
        "./data/" + filename,
        exe_dir + "../data/" + filename,
        exe_dir + "data/" + filename,
        "/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/" + filename
    };
    for (const auto& p : cands) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) return p;
    }
    return cands[0];
}

int main(int argc, char** argv) {
    std::string lex_path    = resolve_data_file("kaikki_lexicon.bin");
    std::string sponge_path = resolve_data_file("msmarco_sponge.bin");
    std::string ptx_path    = "kernels/gllm_talk_kernels.ptx";
    std::string queries_tsv = "/home/liveuser/.gemini/antigravity-cli/brain/a8eaa546-9a14-44da-badc-ed3f36787d7c/scratch/bench_100_qa.tsv";
    std::string out_tsv     = "/home/liveuser/.gemini/antigravity-cli/brain/a8eaa546-9a14-44da-badc-ed3f36787d7c/scratch/bench_100_results.tsv";
    int gpu_steps = 36; // 36 GPU diffusion waves (matches 72 sub-steps in visualizer)

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--queries" && i + 1 < argc) queries_tsv = argv[++i];
        else if (a == "--sponge" && i + 1 < argc) sponge_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_tsv = argv[++i];
        else if (a == "--gpu-steps" && i + 1 < argc) gpu_steps = std::atoi(argv[++i]);
    }

    auto t_load0 = std::chrono::steady_clock::now();
    Lexicon lex;
    if (!lex.load(lex_path)) { std::cerr << "[ERROR] Lexicon load failed\n"; return 1; }
    SpongeIndex sp;
    if (!sp.load(sponge_path, lex.num_words)) { std::cerr << "[ERROR] Sponge load failed\n"; return 1; }
    auto t_load1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
    size_t rss_mb = read_rss_kb() / 1024;

    // Load 100 benchmark queries
    struct BenchQ { std::string split, q, gt; };
    std::vector<BenchQ> queries;
    {
        std::ifstream ifs(queries_tsv);
        std::string line;
        while (std::getline(ifs, line)) {
            size_t t1 = line.find('\t');
            if (t1 == std::string::npos) continue;
            size_t t2 = line.find('\t', t1 + 1);
            if (t2 == std::string::npos) continue;
            queries.push_back({line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1), line.substr(t2 + 1)});
        }
    }

    // Init CUDA GPU & allocate exact same 128^3 sponge + double-buffered fluid grids
    CU_CHECK(cuInit(0));
    CUdevice dev = 0; CU_CHECK(cuDeviceGet(&dev, 0));
    char gpu_name[256] = {}; CU_CHECK(cuDeviceGetName(gpu_name, sizeof(gpu_name), dev));
    CUcontext ctx = nullptr; CU_CHECK(cuCtxCreate_v2(&ctx, 0, dev));

    size_t vram_free_before = 0, vram_total = 0;
    CU_CHECK(cuMemGetInfo_v2(&vram_free_before, &vram_total));

    CUmodule mod = load_ptx(ptx_path);
    CUfunction fn_carve = nullptr, fn_inject = nullptr, fn_step = nullptr;
    CU_CHECK(cuModuleGetFunction(&fn_carve,  mod, "carve_sponge_voxels"));
    CU_CHECK(cuModuleGetFunction(&fn_inject, mod, "inject_fluid_drops"));
    CU_CHECK(cuModuleGetFunction(&fn_step,   mod, "step_fluid_split_recombine"));

    constexpr uint32_t kGrid = 128u;
    constexpr uint32_t kTotalVox = kGrid * kGrid * kGrid;
    constexpr float    kRad = 0.52f;
    constexpr uint32_t kMaxBatch = 4096u;

    CUdeviceptr d_sponge = 0, d_meta = 0, d_fa = 0, d_ma = 0, d_fb = 0, d_mb = 0, d_segs = 0, d_drops = 0, d_stats = 0;
    CU_CHECK(cuMemAlloc_v2(&d_sponge, static_cast<size_t>(kTotalVox) * 16));
    CU_CHECK(cuMemAlloc_v2(&d_meta,   static_cast<size_t>(kTotalVox) * 8));
    CU_CHECK(cuMemAlloc_v2(&d_fa,     static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_ma,     static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_fb,     static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_mb,     static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemAlloc_v2(&d_segs,   static_cast<size_t>(kMaxBatch) * sizeof(CarveSeg)));
    CU_CHECK(cuMemAlloc_v2(&d_drops,  static_cast<size_t>(kMaxBatch) * sizeof(FluidDrop)));
    CU_CHECK(cuMemAlloc_v2(&d_stats,  4 * sizeof(uint32_t)));

    size_t vram_free_after = 0;
    CU_CHECK(cuMemGetInfo_v2(&vram_free_after, &vram_total));
    double vram_used_mb = static_cast<double>(vram_free_before - vram_free_after) / (1024.0 * 1024.0);

    CUevent ev0 = nullptr, ev1 = nullptr;
    CU_CHECK(cuEventCreate(&ev0, 0));
    CU_CHECK(cuEventCreate(&ev1, 0));

    auto pos_hubs = build_sphere_hubs<12>(0.80f, 0.0f);
    auto syn_hubs = build_sphere_hubs<8>(0.64f, 0.75f);

    std::vector<double> cpu_ms_list, gpu_ms_list, tot_ms_list;
    std::vector<uint32_t> active_vox_list, fork_list, coll_list;
    std::ofstream ofs(out_tsv);
    ofs << "idx\tsplit\tcpu_ms\tgpu_ms\ttot_ms\tactive_vox\tcollisions\tdrops\tquery\tground_truth\ttop1\ttop2\ttop3\ttop4\ttop5\n";

    // Warm up CUDA context once
    CU_CHECK(cuMemsetD32_v2(d_sponge, 0u, static_cast<size_t>(kTotalVox) * 4));
    CU_CHECK(cuMemsetD32_v2(d_meta,   0u, static_cast<size_t>(kTotalVox) * 2));
    CU_CHECK(cuCtxSynchronize());

    auto t_all0 = std::chrono::steady_clock::now();
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& bq = queries[qi];
        auto tc0 = std::chrono::steady_clock::now();

        auto q_toks = lex.tokenize(bq.q);
        // 1. Sphere 2 Pure Learned POS Valve
        std::array<double, 12> vc{}; double vtot = 0.0;
        for (size_t k = 0; k < std::min<size_t>(2, q_toks.size()); ++k) {
            auto it = sp.opener_to_ans_pos.find(q_toks[k].word_id);
            if (it == sp.opener_to_ans_pos.end()) continue;
            double wt = (k == 0) ? 2.0 : 1.0;
            for (int p = 0; p < 12; ++p) { vc[p] += it->second[p] * wt; vtot += it->second[p] * wt; }
        }
        std::array<float, 12> vp{};
        for (int p = 0; p < 12; ++p) vp[p] = static_cast<float>(vtot > 0.0 ? vc[p] / vtot : 1.0 / 12.0);

        // 2. Collect ALL unique question words with carved channels (ZERO stopwords!)
        struct AW { uint32_t wid; uint8_t pos; double tot; };
        std::vector<AW> aws;
        std::unordered_set<uint32_t> seen;
        for (const auto& t : q_toks) {
            if (!seen.insert(t.word_id).second) continue;
            if (t.word_id < sp.csr_words.size() && sp.csr_words[t.word_id].total_count > 0 && aws.size() < 24) {
                aws.push_back({t.word_id, t.pos, static_cast<double>(sp.csr_words[t.word_id].total_count)});
            }
        }
        uint32_t K = static_cast<uint32_t>(aws.size());
        sp.touched_aids.clear();
        const double N_qa = static_cast<double>(std::max<uint32_t>(1u, sp.qa_indexed));
        const float log_N = static_cast<float>(std::log(N_qa));
        const float* lh = sp.log_hits.data();
        for (uint32_t k = 0; k < K; ++k) {
            const auto& m = sp.csr_words[aws[k].wid];
            const float log_inv_pw = static_cast<float>(std::log(N_qa / aws[k].tot));
            const uint32_t bit = 1u << std::min<uint32_t>(k, 31u);
            const CsrEdge* ep = sp.csr_edges.data() + m.edge_off;
            for (uint32_t i = 0; i < m.edge_cnt; ++i) {
                uint32_t aid = ep[i].aid;
                float lc = (ep[i].count == 1u) ? 0.0f : std::log(static_cast<float>(ep[i].count));
                float llr = lc + log_inv_pw - lh[aid];
                if (llr > 0.0f) {
                    if (sp.acc_mask[aid] == 0u) sp.touched_aids.push_back(aid);
                    sp.acc_ml[aid] += llr;
                    sp.acc_mask[aid] |= bit;
                }
            }
        }
        struct RC { uint32_t aid; float fml; uint32_t drops; uint32_t mask; };
        std::vector<RC> ranked;
        ranked.reserve(sp.touched_aids.size());
        for (uint32_t aid : sp.touched_aids) {
            uint32_t mask = sp.acc_mask[aid];
            uint32_t drops = static_cast<uint32_t>(__builtin_popcount(mask));
            uint8_t ap = sp.answers[aid].primary_pos < 12u ? sp.answers[aid].primary_pos : 11u;
            float fml = (lh[aid] - log_N) + sp.acc_ml[aid] + std::log(std::max(1e-6f, vp[ap]));
            ranked.push_back({aid, fml, drops, mask});
            sp.acc_ml[aid] = 0.0f; sp.acc_mask[aid] = 0u;
        }
        size_t keep = std::min<size_t>(22, ranked.size());
        if (keep > 0) {
            std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.end(),
                              [](const RC& a, const RC& b) { return a.fml > b.fml; });
            ranked.resize(keep);
        }
        auto tc1 = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

        // 3. Build 3D sponge carve segments & drops for Top-22 candidates -> run REAL CUDA 128^3 fluid wave!
        std::vector<CarveSeg> segs;
        std::vector<FluidDrop> drops_gpu;
        std::vector<Vec3> q_entry(K);
        for (uint32_t k = 0; k < K; ++k) {
            Vec3 wo = word_outer_xyz(aws[k].wid, lex.num_words);
            q_entry[k] = norm3(syn_hubs[1] * 0.45f + pos_hubs[aws[k].pos] * 0.35f + wo * 0.20f) * (kRad * 0.90f);
            drops_gpu.push_back({{q_entry[k].x, q_entry[k].y, q_entry[k].z, 1.85f}, (1u << std::min<uint32_t>(k, 31u)), 0u, 0, 0});
        }
        for (const auto& rc : ranked) {
            const auto& an = sp.answers[rc.aid];
            Vec3 ao = word_outer_xyz(an.anchor_word_id, lex.num_words);
            Vec3 a_exit = norm3(syn_hubs[an.primary_pos == 9u ? 5 : 4] * 0.45f + pos_hubs[an.primary_pos] * 0.35f + ao * 0.20f) * (kRad * 0.90f);
            Vec3 pool{0, 0, 0}; int pc = 0;
            for (uint32_t k = 0; k < K; ++k) if (rc.mask & (1u << std::min<uint32_t>(k, 31u))) { pool = pool + q_entry[k]; pc++; }
            if (pc > 0) pool = pool * (1.0f / pc);
            float pr = rc.drops >= 2 ? 0.22f : 0.35f;
            pool = norm3(pool * 0.70f + a_exit * 0.30f) * pr;
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t bit = 1u << std::min<uint32_t>(k, 31u);
                if (!(rc.mask & bit)) continue;
                Vec3 m1 = lerp3(q_entry[k], pool, 0.5f), m2 = lerp3(pool, a_exit, 0.5f);
                auto add_s = [&](const Vec3& a, const Vec3& b, float role) {
                    Vec3 d = norm3(b - a);
                    segs.push_back({{a.x, a.y, a.z, 1.0f}, {b.x, b.y, b.z, 0.22f}, {d.x, d.y, d.z, role}});
                };
                add_s(q_entry[k], m1, -1.0f); add_s(m1, pool, -1.0f);
                add_s(pool, m2, +1.0f); add_s(m2, a_exit, +1.0f);
                if (rc.drops >= 2) {
                    Vec3 np = lerp3(m1, pool, 0.90f);
                    drops_gpu.push_back({{np.x, np.y, np.z, 1.5f}, bit, 0u, 0, 0});
                }
            }
        }

        CU_CHECK(cuEventRecord(ev0, nullptr));
        CU_CHECK(cuMemsetD32_v2(d_sponge, 0u, static_cast<size_t>(kTotalVox) * 4));
        CU_CHECK(cuMemsetD32_v2(d_meta,   0u, static_cast<size_t>(kTotalVox) * 2));
        CU_CHECK(cuMemsetD32_v2(d_fa, 0u, kTotalVox)); CU_CHECK(cuMemsetD32_v2(d_ma, 0u, kTotalVox));
        CU_CHECK(cuMemsetD32_v2(d_fb, 0u, kTotalVox)); CU_CHECK(cuMemsetD32_v2(d_mb, 0u, kTotalVox));

        uint32_t g = kGrid; float wr = kRad;
        if (!segs.empty()) {
            uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(kMaxBatch, segs.size()));
            CU_CHECK(cuMemcpyHtoD_v2(d_segs, segs.data(), cnt * sizeof(CarveSeg)));
            void* a[] = {&d_sponge, &d_meta, &d_segs, &cnt, &g, &wr};
            CU_CHECK(cuLaunchKernel(fn_carve, (cnt * 32u + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
        }
        if (!drops_gpu.empty()) {
            uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(kMaxBatch, drops_gpu.size()));
            CU_CHECK(cuMemcpyHtoD_v2(d_drops, drops_gpu.data(), cnt * sizeof(FluidDrop)));
            void* a[] = {&d_fa, &d_ma, &d_drops, &cnt, &g, &wr};
            CU_CHECK(cuLaunchKernel(fn_inject, (cnt + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
        }
        uint32_t h_stats[4] = {};
        for (int st = 0; st < gpu_steps; ++st) {
            CU_CHECK(cuMemsetD32_v2(d_stats, 0u, 4u));
            CU_CHECK(cuMemsetD32_v2(d_fb, 0u, kTotalVox));
            CU_CHECK(cuMemsetD32_v2(d_mb, 0u, kTotalVox));
            void* a[] = {&d_sponge, &d_fa, &d_ma, &d_fb, &d_mb, &d_stats, &g};
            CU_CHECK(cuLaunchKernel(fn_step, (kTotalVox + 255u) / 256u, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
            std::swap(d_fa, d_fb); std::swap(d_ma, d_mb);
        }
        CU_CHECK(cuMemcpyDtoH_v2(h_stats, d_stats, sizeof(h_stats)));
        CU_CHECK(cuEventRecord(ev1, nullptr));
        CU_CHECK(cuEventSynchronize(ev1));
        float gpu_ms_f = 0.0f;
        CU_CHECK(cuEventElapsedTime(&gpu_ms_f, ev0, ev1));
        double gpu_ms = static_cast<double>(gpu_ms_f);

        cpu_ms_list.push_back(cpu_ms);
        gpu_ms_list.push_back(gpu_ms);
        tot_ms_list.push_back(cpu_ms + gpu_ms);
        active_vox_list.push_back(h_stats[0]);
        fork_list.push_back(h_stats[1]);
        coll_list.push_back(h_stats[2]);

        auto clean_t = [](std::string s) {
            for (char& c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
            return s;
        };
        ofs << qi << "\t" << bq.split << "\t" << std::fixed << std::setprecision(3)
            << cpu_ms << "\t" << gpu_ms << "\t" << (cpu_ms + gpu_ms) << "\t"
            << h_stats[0] << "\t" << h_stats[2] << "\t"
            << (ranked.empty() ? 0 : ranked[0].drops) << "/" << K << "\t"
            << clean_t(bq.q) << "\t" << clean_t(bq.gt);
        for (int r = 0; r < 5; ++r) {
            ofs << "\t" << (r < static_cast<int>(ranked.size()) ? clean_t(sp.answers[ranked[r].aid].text) : "");
        }
        ofs << "\n";
    }
    auto t_all1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_all1 - t_all0).count();

    auto stats_of = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / std::max<size_t>(1, v.size());
        double p50 = v[v.size() / 2];
        double p95 = v[std::min(v.size() - 1, (v.size() * 95) / 100)];
        return std::array<double, 3>{mean, p50, p95};
    };
    auto cs = stats_of(cpu_ms_list);
    auto gs = stats_of(gpu_ms_list);
    auto ts = stats_of(tot_ms_list);
    double avg_vox  = std::accumulate(active_vox_list.begin(), active_vox_list.end(), 0.0) / queries.size();
    double avg_coll = std::accumulate(coll_list.begin(), coll_list.end(), 0.0) / queries.size();

    std::cout << "=== GLLM 4-SPHERE MS MARCO 100-QUERY BENCHMARK ===\n";
    std::cout << "GPU:            " << gpu_name << "\n";
    std::cout << "Dataset:        " << sp.qa_indexed << " Q&As | " << sp.answers.size() << " basins | " << sp.csr_edges.size() << " CSR channels\n";
    std::cout << "Load Time:      " << std::fixed << std::setprecision(1) << load_ms << " ms\n";
    std::cout << "CPU RAM (RSS):  " << rss_mb << " MiB\n";
    std::cout << "GPU VRAM Alloc: " << std::setprecision(1) << vram_used_mb << " MiB (out of " << (vram_total / (1024*1024)) << " MiB)\n";
    std::cout << "CPU CSR Fluid:  mean=" << std::setprecision(2) << cs[0] << " ms | p50=" << cs[1] << " ms | p95=" << cs[2] << " ms (" << (1000.0 / cs[0]) << " Q/s)\n";
    std::cout << "GPU 128^3 Wave: mean=" << gs[0] << " ms | p50=" << gs[1] << " ms | p95=" << gs[2] << " ms (" << gpu_steps << " parallel steps)\n";
    std::cout << "End-to-End:     mean=" << ts[0] << " ms | p50=" << ts[1] << " ms | p95=" << ts[2] << " ms (" << (queries.size() / wall_s) << " Q/s)\n";
    std::cout << "Avg Voxels:     active=" << std::setprecision(0) << avg_vox << " | collisions(2+)=" << avg_coll << "\n";
    std::cout << "Results TSV:    " << out_tsv << "\n";
    return 0;
}
