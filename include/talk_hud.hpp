#pragma once

#include "hud_canvas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gllm {

static constexpr uint32_t kNumPosClusters = 12u;
static constexpr uint32_t kNumSynNodes    = 8u;

struct PosClusterMeta {
    const char* name;
    float r, g, b;
};

inline const std::array<PosClusterMeta, kNumPosClusters>& pos_cluster_meta() {
    static const std::array<PosClusterMeta, kNumPosClusters> kMeta = {{
        {"noun",  0.22f, 0.72f, 1.00f}, // 0
        {"verb",  1.00f, 0.45f, 0.22f}, // 1
        {"adj",   0.25f, 0.90f, 0.52f}, // 2
        {"adv",   1.00f, 0.84f, 0.24f}, // 3
        {"name",  0.20f, 0.95f, 0.92f}, // 4
        {"pron",  0.96f, 0.36f, 0.88f}, // 5
        {"prep",  0.64f, 0.46f, 1.00f}, // 6
        {"det",   0.20f, 0.80f, 0.72f}, // 7
        {"conj",  0.66f, 0.96f, 0.28f}, // 8
        {"num",   1.00f, 0.66f, 0.20f}, // 9
        {"intj",  1.00f, 0.38f, 0.52f}, // 10
        {"other", 0.72f, 0.76f, 0.84f}  // 11
    }};
    return kMeta;
}

// 8 Emergent Syntactic Sphere Hubs (R = 0.64) learned from real English sentences & Q&As
enum SynNodeId : uint32_t {
    kSynClauseWH   = 0, // Question opener / interrogative trigger
    kSynNPSubj     = 1, // Subject Noun Phrase (topic / entity)
    kSynVPCop      = 2, // Copula / linking verb ("is", "was", "are")
    kSynVPAct      = 3, // Action Verb Phrase ("became", "wrote", "ate")
    kSynNPObj      = 4, // Object / Predicate Noun Phrase (answer payload)
    kSynPPMod      = 5, // Prepositional / Time / Place phrase ("in 1596", "in Europe")
    kSynClauseDecl = 6, // Declarative sentence frame (SVO assembly)
    kSynClauseConj = 7  // Compound / clause connector
};

inline const std::array<PosClusterMeta, kNumSynNodes>& syn_node_meta() {
    static const std::array<PosClusterMeta, kNumSynNodes> kMeta = {{
        {"WH_Q",  0.25f, 0.95f, 1.00f}, // 0 bright cyan
        {"NP_S",  0.20f, 0.96f, 0.58f}, // 1 emerald
        {"VP_BE", 1.00f, 0.78f, 0.22f}, // 2 warm gold
        {"VP_AC", 1.00f, 0.46f, 0.28f}, // 3 coral
        {"NP_O",  1.00f, 0.86f, 0.34f}, // 4 bright amber
        {"PP_M",  0.76f, 0.48f, 1.00f}, // 5 violet
        {"DECL",  0.34f, 1.00f, 0.82f}, // 6 mint
        {"CONJ",  0.96f, 0.40f, 0.78f}  // 7 magenta
    }};
    return kMeta;
}

struct CandidateFluidRow {
    std::string answer;
    float       current_ml    = 0.0f;
    float       target_ml     = 0.0f;
    uint32_t    drop_sources  = 0;
    uint32_t    total_sources = 1;
    bool        pos_valve_ok  = false;
    uint8_t     answer_pos    = 0;
};

struct TalkHudState {
    uint32_t total_dict_words  = 0;
    uint32_t total_qa_pairs    = 0;
    uint32_t total_syn_sents   = 0;
    uint32_t qa_processed      = 0;
    uint32_t carved_voxels     = 0;
    uint32_t total_voxels      = 128u * 128u * 128u;
    uint64_t syn_transitions   = 0;
    float    cuda_ms           = 0.0f;
    float    fps               = 0.0f;
    float    stream_rate       = 0.75f;
    bool     streaming         = true;
    bool     burst_mode        = false;
    float    anim_phase        = 0.0f;

    std::string active_q;
    std::string active_a;
    std::vector<std::pair<std::string, uint8_t>> active_q_toks;
    std::vector<std::pair<std::string, uint8_t>> active_a_toks;

    std::string input_q = "What is the capital of France?";
    std::string input_a = "Paris";
    int         focus_field = 0;

    bool     fluid_wave_active  = false;
    uint32_t wave_step          = 0;
    uint32_t wave_max_steps     = 72;
    uint32_t gpu_fluid_voxels   = 0;
    uint32_t gpu_split_forks    = 0;
    uint32_t gpu_collision_vox  = 0;
    float    gpu_peak_vox_wt    = 0.0f;
    std::string pos_valve_label;
    std::string fluid_summary;

    // Syntactic Sphere Natural Sentence Output!
    std::string syn_path_str;       // e.g. "WH_Q -> NP_S -> VP_BE -> NP_O -> DECL"
    std::string spoken_sentence;    // Full raw sponge answer (scrollable when long!)
    uint32_t    spoken_words_shown  = 0;
    uint32_t    spoken_words_total  = 0;
    int         resp_scroll         = 0;
    int         resp_max_scroll     = 0;

    std::vector<CandidateFluidRow> candidates;
    uint32_t selected_cand = 0;
    std::array<float, kNumPosClusters> pos_activity{};
    std::array<float, kNumSynNodes>    syn_activity{};
};

class TalkSidebarHUD {
public:
    static constexpr int kPanelW = 408;
    static constexpr int kPanelH = 940;

    Rect box_q      = {16, 266, 376, 30};
    Rect box_a      = {16, 312, 376, 30};
    Rect btn_ask    = {16, 350, 120, 26};
    Rect btn_carve  = {144, 350, 120, 26};
    Rect btn_load89 = {272, 350, 120, 26};
    Rect btn_stream = {16, 382, 184, 24};
    Rect btn_replay = {208, 382, 184, 24};
    Rect box_resp   = {14, 414, 380, 102};

    static inline Rect cand_row_rect(size_t i) {
        return {20, 606 + static_cast<int>(i) * 25, 368, 23};
    }

    std::vector<uint8_t> rgba;
    int hover_btn  = 0;
    int hover_cand = -1;

    TalkSidebarHUD() : rgba(kPanelW * kPanelH * 4, 0) {}

    void put_px(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        if (x < 0 || x >= kPanelW || y < 0 || y >= kPanelH) return;
        size_t i = (static_cast<size_t>(y) * kPanelW + x) * 4;
        rgba[i + 0] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = a;
    }

    void fill_rect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        int x0 = std::max(0, x), y0 = std::max(0, y);
        int x1 = std::min(kPanelW, x + w), y1 = std::min(kPanelH, y + h);
        for (int yy = y0; yy < y1; ++yy)
            for (int xx = x0; xx < x1; ++xx)
                put_px(xx, yy, r, g, b, a);
    }

    void stroke_rect(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int xx = x; xx < x + w; ++xx) { put_px(xx, y, r, g, b); put_px(xx, y + h - 1, r, g, b); }
        for (int yy = y; yy < y + h; ++yy) { put_px(x, yy, r, g, b); put_px(x + w - 1, yy, r, g, b); }
    }

    void draw_text(int x, int y, const std::string& text, uint8_t r, uint8_t g, uint8_t b, int scale = 1) {
        int cx = x;
        for (char ch : text) {
            uint8_t rows[7];
            glyph_5x7(ch, rows);
            for (int ry = 0; ry < 7; ++ry) {
                for (int rx = 0; rx < 5; ++rx) {
                    if (rows[ry] & (1u << (4 - rx))) {
                        for (int sy = 0; sy < scale; ++sy)
                            for (int sx = 0; sx < scale; ++sx)
                                put_px(cx + rx * scale + sx, y + ry * scale + sy, r, g, b);
                    }
                }
            }
            cx += 6 * scale;
        }
    }

    int draw_wrapped(int x, int y, int max_chars, int max_lines, const std::string& text,
                     uint8_t r, uint8_t g, uint8_t b, int line_h = 11, int scale = 1) {
        size_t pos = 0;
        int lines = 0;
        while (pos < text.size() && lines < max_lines) {
            while (pos < text.size() && text[pos] == ' ') pos++;
            if (pos >= text.size()) break;
            size_t len = std::min<size_t>(max_chars, text.size() - pos);
            if (pos + len < text.size() && text[pos + len] != ' ') {
                size_t sp = text.rfind(' ', pos + len);
                if (sp != std::string::npos && sp > pos) len = sp - pos;
            }
            std::string sub = text.substr(pos, len);
            if (lines == max_lines - 1 && pos + len < text.size() && sub.size() > 3) {
                sub = sub.substr(0, sub.size() - 3) + "...";
            }
            draw_text(x, y + lines * line_h, sub, r, g, b, scale);
            pos += len;
            lines++;
        }
        return std::max(1, lines);
    }

    void render(const TalkHudState& st) {
        fill_rect(0, 0, kPanelW, kPanelH, 9, 14, 26, 255);
        for (int y = 0; y < kPanelH; ++y) put_px(kPanelW - 1, y, 36, 62, 98, 255);

        // Title Banner
        fill_rect(0, 0, kPanelW - 1, 44, 14, 24, 46, 255);
        draw_text(16, 8, "GLLM 4-SPHERE TALK", 110, 228, 255, 2);
        draw_text(16, 28, "WORDS(1.0) -> POS(0.80) -> SYNTAX(0.64) -> SPONGE(0.52)", 142, 198, 228, 1);

        // 4-Sphere Telemetry Card
        fill_rect(14, 50, 380, 68, 13, 21, 38, 255);
        stroke_rect(14, 50, 380, 68, 34, 58, 92);
        char buf[160];
        std::snprintf(buf, sizeof(buf), "1.WORDS: %u | 2.POS: 12 | 3.SYNTAX: 8 Hubs", st.total_dict_words);
        draw_text(22, 57, buf, 165, 224, 255);
        std::snprintf(buf, sizeof(buf), "4.SPONGE: 128^3 | %u carved | CUDA %.2fms | %.0fFPS",
                      st.carved_voxels, st.cuda_ms, st.fps);
        draw_text(22, 71, buf, 255, 204, 115);
        std::snprintf(buf, sizeof(buf), "LEARNED: %u QAs | %llu Syntax Flows (ZERO RULES)",
                      st.qa_processed, static_cast<unsigned long long>(st.syn_transitions));
        draw_text(22, 85, buf, 120, 242, 188);
        std::snprintf(buf, sizeof(buf), "VALVE  : %s",
                      st.pos_valve_label.empty() ? "ALL POS OPEN" : st.pos_valve_label.c_str());
        draw_text(22, 99, buf, 255, 224, 125);

        // Syntactic Sphere (R=0.64) + POS Sphere (R=0.80) Live Hubs
        fill_rect(14, 124, 380, 126, 12, 20, 35, 255);
        stroke_rect(14, 124, 380, 126, 42, 82, 102);
        draw_text(22, 131, "SPHERE 3: SYNTACTIC HIERARCHY HUBS (R=0.64)", 95, 245, 190);
        const auto& smeta = syn_node_meta();
        for (uint32_t i = 0; i < kNumSynNodes; ++i) {
            int col = i / 2;
            int row = i % 2;
            int bx = 22 + col * 92;
            int by = 146 + row * 18;
            const auto& mc = smeta[i];
            uint8_t cr = static_cast<uint8_t>(mc.r * 255);
            uint8_t cg = static_cast<uint8_t>(mc.g * 255);
            uint8_t cb = static_cast<uint8_t>(mc.b * 255);
            fill_rect(bx, by + 1, 7, 7, cr, cg, cb);
            draw_text(bx + 10, by + 1, mc.name, 210, 232, 245);
            fill_rect(bx + 44, by + 2, 42, 6, 22, 32, 48);
            int aw = static_cast<int>(std::clamp(st.syn_activity[i], 0.0f, 1.0f) * 42.0f);
            if (aw > 0) fill_rect(bx + 44, by + 2, aw, 6, cr, cg, cb);
        }

        draw_text(22, 186, "SPHERE 2: PART-OF-SPEECH HUBS (R=0.80)", 150, 215, 250);
        const auto& pmeta = pos_cluster_meta();
        for (uint32_t i = 0; i < kNumPosClusters; ++i) {
            int col = i / 3;
            int row = i % 3;
            int bx = 22 + col * 92;
            int by = 200 + row * 15;
            const auto& mc = pmeta[i];
            uint8_t cr = static_cast<uint8_t>(mc.r * 255);
            uint8_t cg = static_cast<uint8_t>(mc.g * 255);
            uint8_t cb = static_cast<uint8_t>(mc.b * 255);
            fill_rect(bx, by + 1, 6, 6, cr, cg, cb);
            draw_text(bx + 9, by + 1, mc.name, 195, 215, 235);
            fill_rect(bx + 43, by + 2, 43, 5, 22, 32, 48);
            int aw = static_cast<int>(std::clamp(st.pos_activity[i], 0.0f, 1.0f) * 43.0f);
            if (aw > 0) fill_rect(bx + 43, by + 2, aw, 5, cr, cg, cb);
        }

        // Interactive Custom Q / A Controls
        draw_text(16, 255, "ASK QUESTION OR CARVE CUSTOM Q / A (CLICK TO TYPE):", 118, 226, 255);
        fill_rect(box_q.x, box_q.y, box_q.w, box_q.h,
                  st.focus_field == 1 ? 20 : 14, st.focus_field == 1 ? 36 : 22, st.focus_field == 1 ? 62 : 40);
        stroke_rect(box_q.x, box_q.y, box_q.w, box_q.h,
                    st.focus_field == 1 ? 90 : 48, st.focus_field == 1 ? 220 : 80, st.focus_field == 1 ? 255 : 120);
        draw_text(box_q.x + 6, box_q.y + 6, "Q:", 80, 225, 255);
        draw_wrapped(box_q.x + 22, box_q.y + 6, 58, 2, st.input_q + (st.focus_field == 1 ? "_" : ""), 235, 248, 255, 11);

        draw_text(16, 301, "A (FOR CARVING NEW FACT):", 255, 198, 92);
        fill_rect(box_a.x, box_a.y, box_a.w, box_a.h,
                  st.focus_field == 2 ? 44 : 14, st.focus_field == 2 ? 32 : 22, st.focus_field == 2 ? 18 : 40);
        stroke_rect(box_a.x, box_a.y, box_a.w, box_a.h,
                    st.focus_field == 2 ? 255 : 48, st.focus_field == 2 ? 195 : 80, st.focus_field == 2 ? 85 : 120);
        draw_text(box_a.x + 6, box_a.y + 6, "A:", 255, 192, 78);
        draw_wrapped(box_a.x + 22, box_a.y + 6, 58, 2, st.input_a + (st.focus_field == 2 ? "_" : ""), 255, 240, 200, 11);

        auto draw_btn = [&](const Rect& r, const char* lbl, bool hot, uint8_t br, uint8_t bg, uint8_t bb) {
            fill_rect(r.x, r.y, r.w, r.h, hot ? br : br / 2, hot ? bg : bg / 2, hot ? bb : bb / 2);
            stroke_rect(r.x, r.y, r.w, r.h, br, bg, bb);
            int tw = static_cast<int>(std::strlen(lbl)) * 6;
            draw_text(r.x + (r.w - tw) / 2, r.y + (r.h - 7) / 2, lbl, 245, 250, 255);
        };
        draw_btn(btn_ask,    "ASK (4-SPHERE)", hover_btn == 1, 215, 80, 255);
        draw_btn(btn_carve,  "CARVE Q->A",     hover_btn == 2, 240, 165, 45);
        draw_btn(btn_load89, "RELOAD SPONGE",  hover_btn == 3, 45, 205, 145);
        draw_btn(btn_stream, st.streaming ? "PAUSE STREAM" : "RESUME STREAM", hover_btn == 4, 55, 185, 245);
        draw_btn(btn_replay, "REPLAY FLUID WAVE", hover_btn == 5, 85, 220, 175);

        // =====================================================================
        // SYNTACTIC SPHERE RAW SPONGE OUTPUT (SCROLLABLE FOR IN-DEPTH ANSWERS!)
        // =====================================================================
        fill_rect(14, 414, 380, 102, 12, 28, 34, 255);
        stroke_rect(14, 414, 380, 102, 70, 235, 185);
        if (!st.candidates.empty()) {
            std::snprintf(buf, sizeof(buf), "RAW SPONGE OUTPUT (#%u - CLICK #1-#8 / SCROLL):",
                          st.selected_cand + 1u);
            draw_text(22, 422, buf, 95, 248, 195);
        } else {
            draw_text(22, 422, "RAW SPONGE -> SYNTACTIC SPHERE OUTPUT:", 95, 248, 195);
        }
        if (!st.syn_path_str.empty()) {
            draw_text(22, 435, st.syn_path_str.substr(0, 60), 150, 215, 235);
        }
        if (!st.spoken_sentence.empty()) {
            int scale  = (st.spoken_sentence.size() > 70 || st.spoken_sentence.find('\n') != std::string::npos) ? 1 : 2;
            int max_ch = scale == 2 ? 30 : 59;
            int max_ln = scale == 2 ? 3 : 6;
            int lh     = scale == 2 ? 18 : 10;

            // Wrap respecting real newlines so multi-step explanations scroll smoothly!
            std::vector<std::string> wlines;
            size_t pos = 0;
            while (pos < st.spoken_sentence.size()) {
                size_t nl = st.spoken_sentence.find('\n', pos);
                std::string para = (nl == std::string::npos)
                    ? st.spoken_sentence.substr(pos)
                    : st.spoken_sentence.substr(pos, nl - pos);
                pos = (nl == std::string::npos) ? st.spoken_sentence.size() : (nl + 1);
                if (para.empty()) { continue; }
                size_t p = 0;
                while (p < para.size()) {
                    while (p < para.size() && para[p] == ' ') p++;
                    if (p >= para.size()) break;
                    size_t len = std::min<size_t>(max_ch, para.size() - p);
                    if (p + len < para.size() && para[p + len] != ' ') {
                        size_t sp = para.rfind(' ', p + len);
                        if (sp != std::string::npos && sp > p) len = sp - p;
                    }
                    wlines.push_back(para.substr(p, len));
                    p += len;
                }
            }
            int max_sc = std::max(0, static_cast<int>(wlines.size()) - max_ln);
            int sc = std::clamp(st.resp_scroll, 0, max_sc);
            for (int li = 0; li < max_ln && (sc + li) < static_cast<int>(wlines.size()); ++li) {
                draw_text(22, 449 + li * lh, wlines[sc + li], 255, 248, 205, scale);
            }
            if (max_sc > 0) {
                // Draw subtle scrollbar on right side of response box
                int sb_x = 386, sb_y = 448, sb_h = 62;
                fill_rect(sb_x, sb_y, 4, sb_h, 24, 52, 58);
                int thumb_h = std::max(10, (sb_h * max_ln) / static_cast<int>(wlines.size()));
                int thumb_y = sb_y + ((sb_h - thumb_h) * sc) / max_sc;
                fill_rect(sb_x, thumb_y, 4, thumb_h, 95, 248, 195);
            }
        } else {
            draw_wrapped(22, 456, 58, 2, "Ask any question to watch water split in the sponge and emerge raw through the Syntactic Sphere!", 145, 195, 205, 11);
        }

        // =====================================================================
        // LIVE PARALLEL VOXEL FLUID SPLITTING & CANDIDATE ANSWER METERS
        // =====================================================================
        fill_rect(14, 524, 380, 296, 14, 17, 34, 255);
        stroke_rect(14, 524, 380, 296,
                    st.fluid_wave_active ? 225 : 44,
                    st.fluid_wave_active ? 110 : 68,
                    st.fluid_wave_active ? 255 : 108);

        draw_text(22, 532, "SPONGE CANDIDATES (CLICK #1-#8 TO VIEW FULL ANSWER)", 235, 155, 255);

        fill_rect(20, 545, 368, 38, 18, 24, 46, 255);
        stroke_rect(20, 545, 368, 38, 65, 88, 138);
        std::snprintf(buf, sizeof(buf), "WAVE STEP %2u/%2u | ACTIVE VOXELS: %u",
                      st.wave_step, st.wave_max_steps, st.gpu_fluid_voxels);
        draw_text(26, 551, buf, 115, 232, 255);
        std::snprintf(buf, sizeof(buf), "FORKS: %u | COLLISIONS(2+): %u | PEAK WT: %.2f",
                      st.gpu_split_forks, st.gpu_collision_vox, st.gpu_peak_vox_wt);
        draw_text(26, 565, buf, 255, 180, 255);

        if (!st.fluid_summary.empty()) {
            draw_text(22, 589, st.fluid_summary.substr(0, 60), 205, 225, 245);
        }

        float max_ml = 1e-4f;
        for (const auto& c : st.candidates) max_ml = std::max(max_ml, c.target_ml);

        int ry = 606;
        size_t shown = std::min<size_t>(8, st.candidates.size());
        for (size_t i = 0; i < shown; ++i) {
            const auto& c = st.candidates[i];
            bool selected = (i == st.selected_cand && c.current_ml > 0.0f);
            bool hovered  = (static_cast<int>(i) == hover_cand);
            bool top1     = (i == 0 && c.current_ml > 0.0f);
            fill_rect(20, ry, 368, 23,
                      selected ? 24 : (hovered ? 26 : 16),
                      selected ? 46 : (hovered ? 36 : 23),
                      selected ? 38 : (hovered ? 60 : 40));
            stroke_rect(20, ry, 368, 23,
                        selected ? 85  : (hovered ? 120 : (top1 ? 255 : (c.drop_sources >= 2 ? 195 : 42))),
                        selected ? 255 : (hovered ? 230 : (top1 ? 210 : (c.drop_sources >= 2 ? 85  : 64))),
                        selected ? 195 : (hovered ? 255 : (top1 ? 75  : (c.drop_sources >= 2 ? 240 : 96))));

            float frac = std::clamp(c.current_ml / max_ml, 0.0f, 1.0f);
            int bar_w = static_cast<int>(frac * 364.0f);
            if (bar_w > 0) {
                uint8_t br = selected ? 36  : (top1 ? 165 : (c.drop_sources >= 2 ? 125 : 22));
                uint8_t bg = selected ? 145 : (top1 ? 115 : (c.drop_sources >= 2 ? 38  : 92));
                uint8_t bb = selected ? 110 : (top1 ? 24  : (c.drop_sources >= 2 ? 160 : 148));
                fill_rect(22, ry + 2, bar_w, 19, br, bg, bb);
            }

            std::string ans_short = c.answer.substr(0, 24);
            std::snprintf(buf, sizeof(buf), "#%zu %s", i + 1, ans_short.c_str());
            draw_text(26, ry + 7, buf,
                      selected ? 255 : (top1 ? 255 : 230),
                      selected ? 255 : (top1 ? 245 : 240),
                      selected ? 215 : (top1 ? 185 : 255));

            if (c.current_ml < 10.0f) {
                std::snprintf(buf, sizeof(buf), "%5.2fmL [%u/%u%s]",
                              c.current_ml, c.drop_sources, c.total_sources,
                              c.pos_valve_ok ? "V" : "");
            } else {
                std::snprintf(buf, sizeof(buf), "%5.1fmL [%u/%u%s]",
                              c.current_ml, c.drop_sources, c.total_sources,
                              c.pos_valve_ok ? "V" : "");
            }
            int rw = static_cast<int>(std::strlen(buf)) * 6;
            draw_text(384 - rw, ry + 7, buf,
                      selected ? 160 : (top1 ? 255 : (c.drop_sources >= 2 ? 255 : 170)),
                      selected ? 255 : (top1 ? 235 : (c.drop_sources >= 2 ? 190 : 225)),
                      selected ? 225 : (top1 ? 120 : 255));
            ry += 25;
        }

        // Color Legend & Controls Footer
        fill_rect(14, 826, 380, 104, 11, 17, 30, 255);
        stroke_rect(14, 826, 380, 104, 32, 52, 82);
        fill_rect(22, 834, 8, 8, 25, 215, 255);  draw_text(35, 835, "1-Drop Split Fluid (brightness = voxel weight)", 180, 228, 250);
        fill_rect(22, 847, 8, 8, 245, 95, 235);  draw_text(35, 848, "2+ Drops RECOMBINING in Sponge (collision)", 250, 180, 255);
        fill_rect(22, 860, 8, 8, 65, 245, 180);  draw_text(35, 861, "Syntactic Sphere Cascade (natural sentence!)", 120, 248, 200);
        draw_text(22, 878, "1:Words 2:POS 3:Syntax 4:Sponge 5:Rays", 145, 175, 210);
        draw_text(22, 891, "SPACE:Stream  T:Burst  S:Step  +/-:Speed  X:Clear", 145, 175, 210);
        draw_text(22, 904, "Up/Dn:Shell  Drag:Orbit  Scroll:Zoom  R:ResetCam", 125, 165, 205);
    }
};

} // namespace gllm
