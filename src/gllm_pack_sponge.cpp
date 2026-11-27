// =============================================================================
// gllm_pack_sponge — Ultra-Fast C++17 Streaming Binary CSR Sponge Packer
//   Streams millions of Q&A pairs (MS MARCO + PAQ-L1 / NQ-open / TriviaQA)
//   into a single compact zero-lever binary CSR sponge (`GSPG` v1) in seconds!
// =============================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr uint32_t kNumPos = 12u;
static constexpr uint32_t kNumSyn = 8u;

static inline uint32_t fnv1a_32(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) { h ^= c; h *= 16777619u; }
    return h;
}
static inline uint64_t fnv1a_64_lower(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        unsigned char lc = static_cast<unsigned char>(std::tolower(c));
        h ^= lc;
        h *= 1099511628211ULL;
    }
    return h;
}

#pragma pack(push, 1)
struct BinHdr {
    uint32_t ver, num_words, qa_count, num_answers;
    uint64_t ans_blob_len;
    uint32_t num_active_words;
    uint64_t num_edges;
    uint32_t num_openers;
    uint64_t total_transitions;
};
struct BinAnsMeta {
    uint32_t anchor_wid;
    uint32_t hits;
    uint8_t  primary_pos;
    uint8_t  pad;
    uint16_t text_len;
    uint32_t text_off;
};
struct BinWordIdx {
    uint32_t wid;
    uint64_t total_count;
    uint32_t edge_off;
    uint32_t edge_cnt;
};
struct BinEdge {
    uint32_t aid;
    uint32_t count;
};
struct BinOpener {
    uint32_t wid;
    uint32_t cnts[kNumPos];
};
#pragma pack(pop)

struct Lexicon {
    uint32_t num_words = 0;
    std::vector<uint8_t> primary_pos;
    std::unordered_map<std::string, uint32_t> word_to_id;

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        char magic[4] = {}; ifs.read(magic, 4);
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

    struct Tok { uint32_t wid; uint8_t pos; };
    void tokenize(const std::string& text, std::vector<Tok>& out) const {
        out.clear();
        size_t i = 0, n = text.size();
        std::string low;
        while (i < n) {
            while (i < n && !std::isalnum(static_cast<unsigned char>(text[i]))) ++i;
            if (i >= n) break;
            size_t s = i;
            while (i < n && (std::isalnum(static_cast<unsigned char>(text[i])) ||
                             (text[i] == '\'' && i + 1 < n && std::isalpha(static_cast<unsigned char>(text[i + 1]))))) {
                ++i;
            }
            size_t len = i - s;
            low.resize(len);
            bool has_dig = false;
            for (size_t k = 0; k < len; ++k) {
                unsigned char c = static_cast<unsigned char>(text[s + k]);
                if (std::isdigit(c)) has_dig = true;
                low[k] = static_cast<char>(std::tolower(c));
            }
            auto it = word_to_id.find(low);
            if (it != word_to_id.end()) {
                out.push_back({it->second, primary_pos[it->second]});
            } else {
                uint32_t wid = fnv1a_32(low) % std::max(1u, num_words);
                uint8_t pid = has_dig ? 9u : (std::isupper(static_cast<unsigned char>(text[s])) ? 4u : 0u);
                out.push_back({wid, pid});
            }
        }
    }
};

// Extract first answer from either plain string OR NQ-open / PAQ bracketed list `['Ans1', 'Ans2']` or `["Ans1"]`
static std::string clean_answer_field(const std::string& raw) {
    size_t s = 0, e = raw.size();
    while (s < e && std::isspace(static_cast<unsigned char>(raw[s]))) ++s;
    while (e > s && std::isspace(static_cast<unsigned char>(raw[e - 1]))) --e;
    if (e <= s) return "";
    if (raw[s] == '[' && raw[e - 1] == ']') {
        // Find first quoted element inside list
        size_t i = s + 1;
        while (i < e && std::isspace(static_cast<unsigned char>(raw[i]))) ++i;
        if (i < e && (raw[i] == '\'' || raw[i] == '"')) {
            char q = raw[i++];
            std::string out;
            out.reserve(64);
            while (i < e) {
                if (raw[i] == '\\' && i + 1 < e) {
                    out.push_back(raw[i + 1]);
                    i += 2;
                } else if (raw[i] == q) {
                    break;
                } else {
                    out.push_back(raw[i++]);
                }
            }
            return out;
        }
    }
    return raw.substr(s, e - s);
}

static uint64_t update_syntax(const std::vector<Lexicon::Tok>& toks,
                              uint64_t pos_to_syn[kNumPos][kNumSyn],
                              uint64_t syn_flow[kNumSyn][kNumSyn]) {
    if (toks.empty()) return 0;
    bool seen_verb = false, in_prep = false;
    uint8_t prev_s = 255u;
    uint64_t tr = 0;
    for (size_t i = 0; i < toks.size(); ++i) {
        uint8_t p = toks[i].pos;
        uint8_t s = 1u;
        if (i == 0 && (p == 5u || p == 3u)) { s = 0u; in_prep = false; }
        else if (p == 8u) { s = 7u; in_prep = false; }
        else if (p == 1u) { s = seen_verb ? 3u : 2u; seen_verb = true; in_prep = false; }
        else if (p == 6u) { s = 5u; in_prep = true; }
        else if (in_prep) { s = 5u; }
        else { s = seen_verb ? 4u : 1u; }
        uint8_t pc = p < kNumPos ? p : 11u;
        pos_to_syn[pc][s]++;
        if (prev_s < kNumSyn && prev_s != s) { syn_flow[prev_s][s]++; tr++; }
        prev_s = s;
    }
    if (prev_s < kNumSyn) { syn_flow[prev_s][6u]++; tr++; }
    return tr;
}

// Fast zero-allocation JSON string extractor for PAQ JSONL: {"question":"...","answer":["..."]}
static bool parse_paq_jsonl_line(const std::string& line, std::string& out_q, std::string& out_a) {
    auto extract_quoted = [&](size_t start_pos, std::string& dst, size_t& end_pos) -> bool {
        size_t p = line.find('"', start_pos);
        if (p == std::string::npos) return false;
        ++p;
        dst.clear();
        while (p < line.size()) {
            char c = line[p++];
            if (c == '\\' && p < line.size()) {
                char nxt = line[p++];
                if (nxt == '"' || nxt == '\\' || nxt == '/') dst.push_back(nxt);
                else if (nxt == 'n' || nxt == 't' || nxt == 'r') dst.push_back(' ');
                else if (nxt == 'u' && p + 4 <= line.size()) { p += 4; dst.push_back(' '); }
                else dst.push_back(nxt);
            } else if (c == '"') {
                end_pos = p;
                return true;
            } else {
                dst.push_back(c);
            }
        }
        return false;
    };
    size_t qk = line.find("\"question\"");
    if (qk == std::string::npos) return false;
    size_t qc = line.find(':', qk + 10);
    if (qc == std::string::npos) return false;
    size_t qend = 0;
    if (!extract_quoted(qc + 1, out_q, qend)) return false;

    size_t ak = line.find("\"answer\"", qend);
    if (ak == std::string::npos) return false;
    size_t ab = line.find('[', ak + 8);
    if (ab == std::string::npos) return false;
    size_t aend = 0;
    if (!extract_quoted(ab + 1, out_a, aend)) return false;
    return !out_q.empty() && !out_a.empty();
}

// Unescape \\n -> real newline inside rich multi-line answers
static std::string unescape_newlines(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i + 1] == 'n') { out.push_back('\n'); ++i; continue; }
            if (s[i + 1] == '\\') { out.push_back('\\'); ++i; continue; }
        }
        out.push_back(s[i]);
    }
    return out;
}

int main(int argc, char** argv) {
    std::string lex_path = "/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/kaikki_lexicon.bin";
    std::string out_bin  = "/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/gllm_15m_sponge.bin";
    std::vector<std::pair<std::string, bool>> inputs; // (path, is_jsonl)
    size_t max_ans_len = 2400;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lexicon" && i + 1 < argc) lex_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_bin = argv[++i];
        else if (a == "--max-ans-len" && i + 1 < argc) max_ans_len = static_cast<size_t>(std::atoi(argv[++i]));
        else if (a == "--tsv" && i + 1 < argc) inputs.push_back({argv[++i], false});
        else if (a == "--jsonl" && i + 1 < argc) inputs.push_back({argv[++i], true});
    }
    if (inputs.empty()) {
        std::cerr << "Usage: gllm_pack_sponge --lexicon <lex> --out <bin> --tsv <file> [--jsonl <file>]\n";
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    Lexicon lex;
    if (!lex.load(lex_path)) { std::cerr << "[ERROR] Cannot load lexicon: " << lex_path << "\n"; return 1; }
    std::cout << "[PACK] Loaded " << lex.num_words << " Kaikki words.\n";

    std::unordered_map<uint64_t, uint32_t> ans_hash_to_id;
    ans_hash_to_id.reserve(6000000);
    std::vector<BinAnsMeta> ans_meta;
    ans_meta.reserve(6000000);
    std::string ans_blob;
    ans_blob.reserve(256 * 1024 * 1024);

    std::vector<uint64_t> raw_edges;
    raw_edges.reserve(100000000);

    std::unordered_map<uint32_t, std::array<uint32_t, kNumPos>> opener_pos;
    opener_pos.reserve(300000);
    uint64_t pos_to_syn[kNumPos][kNumSyn] = {};
    uint64_t syn_flow[kNumSyn][kNumSyn] = {};
    uint64_t total_transitions = 0;
    uint32_t qa_count = 0;

    std::vector<Lexicon::Tok> q_toks, a_toks;
    std::vector<uint32_t> unique_qw;
    q_toks.reserve(32); a_toks.reserve(64); unique_qw.reserve(32);

    std::string q, a;
    for (const auto& inp : inputs) {
        const std::string& fpath = inp.first;
        bool is_jsonl = inp.second;
        auto tf0 = std::chrono::steady_clock::now();
        std::ifstream ifs(fpath);
        if (!ifs) { std::cerr << "[WARN] Cannot open: " << fpath << "\n"; continue; }
        std::string line;
        uint32_t file_qa = 0;
        while (std::getline(ifs, line)) {
            if (is_jsonl) {
                if (!parse_paq_jsonl_line(line, q, a)) continue;
            } else {
                size_t tab = line.find('\t');
                if (tab == std::string::npos) continue;
                q = line.substr(0, tab);
                a = unescape_newlines(clean_answer_field(line.substr(tab + 1)));
                if (q.empty() || a.empty()) continue;
            }
            if (a.size() > max_ans_len) a.resize(max_ans_len);

            lex.tokenize(q, q_toks);
            lex.tokenize(a, a_toks);
            if (q_toks.empty() || a_toks.empty()) continue;

            uint8_t ap = a_toks[0].pos;
            uint64_t ah = fnv1a_64_lower(a);
            uint32_t aid = 0;
            auto it = ans_hash_to_id.find(ah);
            if (it == ans_hash_to_id.end()) {
                if (ans_blob.size() + a.size() >= 0xFFFFFF00ULL) continue; // safety within u32 offset
                aid = static_cast<uint32_t>(ans_meta.size());
                ans_hash_to_id.emplace(ah, aid);
                uint32_t off = static_cast<uint32_t>(ans_blob.size());
                ans_blob.append(a);
                ans_meta.push_back({a_toks[0].wid, 1u, ap, 0u, static_cast<uint16_t>(a.size()), off});
            } else {
                aid = it->second;
                ans_meta[aid].hits++;
            }

            // Sphere 2: pure raw opening-word -> answer POS counts
            for (size_t k = 0; k < std::min<size_t>(2, q_toks.size()); ++k) {
                opener_pos[q_toks[k].wid][ap] += (k == 0 ? 2u : 1u);
            }

            // Sphere 3: pure syntactic hub flow learned from Q & A
            total_transitions += update_syntax(q_toks, pos_to_syn, syn_flow);
            total_transitions += update_syntax(a_toks, pos_to_syn, syn_flow);

            // Sphere 4: ALL unique question words carve channels to answer basin (ZERO stopwords!)
            unique_qw.clear();
            for (const auto& t : q_toks) unique_qw.push_back(t.wid);
            std::sort(unique_qw.begin(), unique_qw.end());
            unique_qw.erase(std::unique(unique_qw.begin(), unique_qw.end()), unique_qw.end());
            for (uint32_t wid : unique_qw) {
                raw_edges.push_back((static_cast<uint64_t>(wid) << 32) | static_cast<uint64_t>(aid));
            }

            qa_count++;
            file_qa++;
            if (file_qa % 2000000u == 0u) {
                std::cout << "  ... " << file_qa << " Q&As streamed (" << ans_meta.size()
                          << " basins, " << raw_edges.size() << " raw links)\n" << std::flush;
            }
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
        std::cout << "[FILE] " << fpath << " -> +" << file_qa << " Q&As (cum=" << qa_count
                  << ", basins=" << ans_meta.size() << ", raw_links=" << raw_edges.size()
                  << ") in " << std::fixed << std::setprecision(1) << secs << "s\n" << std::flush;
    }

    // Free hash table before sorting raw_edges to keep peak RAM super lean!
    ans_hash_to_id.clear();
    ans_hash_to_id.rehash(0);

    std::cout << "[SORT] Sorting " << raw_edges.size() << " raw (word, answer) links into CSR...\n" << std::flush;
    auto ts0 = std::chrono::steady_clock::now();
    std::sort(raw_edges.begin(), raw_edges.end());
    double sort_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts0).count();
    std::cout << "[SORT] Done in " << std::fixed << std::setprecision(1) << sort_s << "s. Collapsing CSR...\n" << std::flush;

    std::vector<BinWordIdx> word_idx;
    std::vector<BinEdge> edges;
    word_idx.reserve(300000);
    edges.reserve(raw_edges.size());

    size_t i = 0, n = raw_edges.size();
    while (i < n) {
        uint32_t cur_w = static_cast<uint32_t>(raw_edges[i] >> 32);
        uint32_t e_off = static_cast<uint32_t>(edges.size());
        uint64_t w_tot = 0;
        while (i < n && static_cast<uint32_t>(raw_edges[i] >> 32) == cur_w) {
            uint64_t val = raw_edges[i];
            uint32_t aid = static_cast<uint32_t>(val & 0xFFFFFFFFu);
            uint32_t cnt = 0;
            while (i < n && raw_edges[i] == val) { ++cnt; ++i; }
            edges.push_back({aid, cnt});
            w_tot += cnt;
        }
        uint32_t e_cnt = static_cast<uint32_t>(edges.size()) - e_off;
        word_idx.push_back({cur_w, w_tot, e_off, e_cnt});
    }
    raw_edges.clear();
    raw_edges.shrink_to_fit();

    std::vector<BinOpener> ops;
    ops.reserve(opener_pos.size());
    for (const auto& kv : opener_pos) {
        BinOpener bo{};
        bo.wid = kv.first;
        for (uint32_t p = 0; p < kNumPos; ++p) bo.cnts[p] = kv.second[p];
        ops.push_back(bo);
    }

    std::ofstream ofs(out_bin, std::ios::binary);
    if (!ofs) { std::cerr << "[ERROR] Cannot write: " << out_bin << "\n"; return 1; }
    ofs.write("GSPG", 4);
    BinHdr hdr{};
    hdr.ver               = 1u;
    hdr.num_words         = lex.num_words;
    hdr.qa_count          = qa_count;
    hdr.num_answers       = static_cast<uint32_t>(ans_meta.size());
    hdr.ans_blob_len      = static_cast<uint64_t>(ans_blob.size());
    hdr.num_active_words  = static_cast<uint32_t>(word_idx.size());
    hdr.num_edges         = static_cast<uint64_t>(edges.size());
    hdr.num_openers       = static_cast<uint32_t>(ops.size());
    hdr.total_transitions = total_transitions;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    ofs.write(reinterpret_cast<const char*>(pos_to_syn), sizeof(pos_to_syn));
    ofs.write(reinterpret_cast<const char*>(syn_flow), sizeof(syn_flow));
    ofs.write(reinterpret_cast<const char*>(ans_meta.data()), static_cast<std::streamsize>(ans_meta.size() * sizeof(BinAnsMeta)));
    ofs.write(ans_blob.data(), static_cast<std::streamsize>(ans_blob.size()));
    ofs.write(reinterpret_cast<const char*>(word_idx.data()), static_cast<std::streamsize>(word_idx.size() * sizeof(BinWordIdx)));
    ofs.write(reinterpret_cast<const char*>(edges.data()), static_cast<std::streamsize>(edges.size() * sizeof(BinEdge)));
    ofs.write(reinterpret_cast<const char*>(ops.data()), static_cast<std::streamsize>(ops.size() * sizeof(BinOpener)));
    ofs.close();

    double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ifstream sz_chk(out_bin, std::ios::binary | std::ios::ate);
    double mb = static_cast<double>(sz_chk.tellg()) / (1024.0 * 1024.0);
    std::cout << "[DONE] Wrote " << out_bin << " (" << std::fixed << std::setprecision(1) << mb << " MB) in " << tot_s << "s\n";
    std::cout << "       Q&As: " << qa_count << " | Basins: " << ans_meta.size()
              << " | Active Words: " << word_idx.size() << " | CSR Edges: " << edges.size()
              << " | Syntax Flows: " << total_transitions << "\n";
    return 0;
}
