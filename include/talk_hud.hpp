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

struct PosClusterMeta {
    const char* name;
    float r, g, b;
};

inline const std::array<PosClusterMeta, kNumPosClusters>& pos_cluster_meta() {
    static const std::array<PosClusterMeta, kNumPosClusters> kMeta = {{
        {"noun",  0.22f, 0.72f, 1.00f}, // 0 sky blue
        {"verb",  1.00f, 0.45f, 0.22f}, // 1 coral orange
        {"adj",   0.25f, 0.90f, 0.52f}, // 2 emerald
        {"adv",   1.00f, 0.84f, 0.24f}, // 3 gold
        {"name",  0.20f, 0.95f, 0.92f}, // 4 cyan
        {"pron",  0.96f, 0.36f, 0.88f}, // 5 magenta
        {"prep",  0.64f, 0.46f, 1.00f}, // 6 violet
        {"det",   0.20f, 0.80f, 0.72f}, // 7 teal
        {"conj",  0.66f, 0.96f, 0.28f}, // 8 lime
        {"num",   1.00f, 0.66f, 0.20f}, // 9 amber
        {"intj",  1.00f, 0.38f, 0.52f}, // 10 rose
        {"other", 0.72f, 0.76f, 0.84f}  // 11 silver
    }};
    return kMeta;
}

struct CandidateFluidRow {
    std::string answer;
    float       current_ml    = 0.0f; // Live accumulated fluid volume (mL)
    float       target_ml     = 0.0f; // Final converged fluid volume (mL)
    uint32_t    drop_sources  = 0;    // Distinct question words recombining here
    uint32_t    total_sources = 1;    // Total content words in question
    bool        pos_valve_ok  = false;
    uint8_t     answer_pos    = 0;
};

struct TalkHudState {
    uint32_t total_dict_words = 0;
    uint32_t total_qa_pairs   = 0;
    uint32_t qa_processed     = 0;
    uint32_t carved_voxels    = 0;
    uint32_t total_voxels     = 128u * 128u * 128u;
    uint64_t merge_events     = 0;
    uint64_t split_events     = 0;
    float    merge_ratio_pct  = 0.0f;
    float    cuda_ms          = 0.0f;
    float    fps              = 0.0f;
    float    stream_rate      = 0.75f;
    bool     streaming        = true;
    bool     burst_mode       = false;
    float    anim_phase       = 0.0f;

    std::string active_q;
    std::string active_a;
    std::vector<std::pair<std::string, uint8_t>> active_q_toks;
    std::vector<std::pair<std::string, uint8_t>> active_a_toks;

    // Interactive Q/A typing & Live Parallel Voxel Fluid Wave
    std::string input_q = "What is the capital of France?";
    std::string input_a = "Paris";
    int         focus_field = 0; // 0 = none, 1 = Q box, 2 = A box

    bool     fluid_wave_active  = false;
    uint32_t wave_step          = 0;
    uint32_t wave_max_steps     = 72;
    uint32_t gpu_fluid_voxels   = 0;
    uint32_t gpu_split_forks    = 0;
    uint32_t gpu_collision_vox  = 0;
    float    gpu_peak_vox_wt    = 0.0f;
    std::string pos_valve_label;
    std::string fluid_summary;
    std::vector<CandidateFluidRow> candidates;

    std::array<float, kNumPosClusters> pos_activity{};
};

class TalkSidebarHUD {
public:
    static constexpr int kPanelW = 392;
    static constexpr int kPanelH = 920;

    Rect box_q      = {16, 360, 360, 32};
    Rect box_a      = {16, 408, 360, 32};
    Rect btn_ask    = {16, 448, 114, 28};
    Rect btn_carve  = {138, 448, 114, 28};
    Rect btn_load89 = {260, 448, 116, 28};
    Rect btn_stream = {16, 482, 174, 24};
    Rect btn_replay = {198, 482, 178, 24};

    std::vector<uint8_t> rgba;
    int hover_btn = 0; // 1=ask, 2=carve, 3=load89, 4=stream, 5=replay

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
                     uint8_t r, uint8_t g, uint8_t b, int line_h = 11) {
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
            draw_text(x, y + lines * line_h, sub, r, g, b, 1);
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
        draw_text(16, 8, "GLLM FLUID SPONGE", 110, 228, 255, 2);
        draw_text(16, 28, "PARALLEL CUDA VOXEL SPLITTING / RECOMBINING", 145, 185, 225, 1);

        // Architecture Telemetry Card
        fill_rect(14, 50, 364, 72, 13, 21, 38, 255);
        stroke_rect(14, 50, 364, 72, 34, 58, 92);
        char buf[160];
        std::snprintf(buf, sizeof(buf), "WORDS: %u Kaikki | POS: 12 Hubs | QA: %u/%u",
                      st.total_dict_words, st.qa_processed, st.total_qa_pairs);
        draw_text(22, 58, buf, 165, 224, 255);
        double pct_carved = 100.0 * static_cast<double>(st.carved_voxels) / std::max(1u, st.total_voxels);
        std::snprintf(buf, sizeof(buf), "SPONGE: 128^3 | %u carved (%.2f%%) | CUDA %.2fms",
                      st.carved_voxels, pct_carved, st.cuda_ms);
        draw_text(22, 72, buf, 255, 204, 115);
        std::snprintf(buf, sizeof(buf), "WATER : Merge %.1f%% (%llu M / %llu S) | %.1f FPS",
                      st.merge_ratio_pct,
                      static_cast<unsigned long long>(st.merge_events),
                      static_cast<unsigned long long>(st.split_events), st.fps);
        draw_text(22, 86, buf, 255, 232, 155);
        std::snprintf(buf, sizeof(buf), "MODE  : %s",
                      st.fluid_wave_active ? "LIVE VOXEL FLUID WAVE (SPLITTING/RECOMBINING)" :
                      (st.burst_mode ? "BURST CARVING STREAM" : (st.streaming ? "REAL-TIME Q->A CARVING" : "PAUSED / INTERACTIVE")));
        draw_text(22, 100, buf, st.fluid_wave_active ? 240 : 140, st.fluid_wave_active ? 140 : 210, 255);

        // Live Q->A Stream Card (compact)
        fill_rect(14, 130, 364, 116, 12, 20, 36, 255);
        stroke_rect(14, 130, 364, 116, 38, 72, 108);
        draw_text(22, 138, "LAST CARVED Q->A:", 100, 220, 255);
        fill_rect(146, 140, 224, 5, 22, 34, 54);
        int fill_w = static_cast<int>(std::clamp(st.anim_phase, 0.0f, 1.0f) * 224.0f);
        fill_rect(146, 140, std::min(112, fill_w), 5, 40, 205, 255);
        if (fill_w > 112) fill_rect(146 + 112, 140, fill_w - 112, 5, 255, 175, 55);

        draw_text(22, 152, "Q:", 72, 225, 255);
        draw_wrapped(38, 152, 54, 2, st.active_q.empty() ? "(waiting...)" : st.active_q, 225, 245, 255, 11);

        const auto& meta = pos_cluster_meta();
        int tx = 22, ty = 178;
        for (size_t i = 0; i < st.active_q_toks.size() && i < 6; ++i) {
            const auto& tk = st.active_q_toks[i];
            std::string chip = tk.first.substr(0, 8) + ":" + meta[tk.second % kNumPosClusters].name;
            int cw = static_cast<int>(chip.size()) * 6 + 6;
            if (tx + cw > 370) break;
            const auto& mc = meta[tk.second % kNumPosClusters];
            fill_rect(tx, ty - 2, cw, 11, 20, 32, 52);
            draw_text(tx + 3, ty, chip, static_cast<uint8_t>(mc.r * 255), static_cast<uint8_t>(mc.g * 255), static_cast<uint8_t>(mc.b * 255));
            tx += cw + 4;
        }
        draw_text(22, 198, "A:", 255, 188, 72);
        draw_wrapped(38, 198, 54, 2, st.active_a.empty() ? "(waiting...)" : st.active_a, 255, 236, 185, 11);
        tx = 22; ty = 224;
        for (size_t i = 0; i < st.active_a_toks.size() && i < 6; ++i) {
            const auto& tk = st.active_a_toks[i];
            std::string chip = tk.first.substr(0, 9) + ":" + meta[tk.second % kNumPosClusters].name;
            int cw = static_cast<int>(chip.size()) * 6 + 6;
            if (tx + cw > 370) break;
            const auto& mc = meta[tk.second % kNumPosClusters];
            fill_rect(tx, ty - 2, cw, 11, 44, 32, 18);
            draw_text(tx + 3, ty, chip, static_cast<uint8_t>(mc.r * 255), static_cast<uint8_t>(mc.g * 255), static_cast<uint8_t>(mc.b * 255));
            tx += cw + 4;
        }

        // 12 POS Cluster Activity Monitor (Middle Sphere R=0.72)
        fill_rect(14, 252, 364, 92, 12, 19, 34, 255);
        stroke_rect(14, 252, 364, 92, 34, 56, 88);
        draw_text(22, 259, "POS SPHERE VALVES (R=0.72):", 150, 220, 195);
        if (!st.pos_valve_label.empty()) {
            draw_text(192, 259, st.pos_valve_label, 255, 220, 120);
        }
        for (uint32_t i = 0; i < kNumPosClusters; ++i) {
            int col = i / 4;
            int row = i % 4;
            int bx = 22 + col * 118;
            int by = 274 + row * 16;
            const auto& mc = meta[i];
            uint8_t cr = static_cast<uint8_t>(mc.r * 255);
            uint8_t cg = static_cast<uint8_t>(mc.g * 255);
            uint8_t cb = static_cast<uint8_t>(mc.b * 255);
            fill_rect(bx, by + 1, 7, 7, cr, cg, cb);
            draw_text(bx + 10, by + 1, mc.name, 200, 218, 238);
            fill_rect(bx + 44, by + 2, 64, 6, 22, 32, 48);
            int aw = static_cast<int>(std::clamp(st.pos_activity[i], 0.0f, 1.0f) * 64.0f);
            if (aw > 0) fill_rect(bx + 44, by + 2, aw, 6, cr, cg, cb);
        }

        // Interactive Custom Q / A Controls
        draw_text(16, 349, "ASK QUESTION OR CARVE CUSTOM Q / A (CLICK TO TYPE):", 118, 226, 255);
        fill_rect(box_q.x, box_q.y, box_q.w, box_q.h,
                  st.focus_field == 1 ? 20 : 14, st.focus_field == 1 ? 36 : 22, st.focus_field == 1 ? 62 : 40);
        stroke_rect(box_q.x, box_q.y, box_q.w, box_q.h,
                    st.focus_field == 1 ? 90 : 48, st.focus_field == 1 ? 220 : 80, st.focus_field == 1 ? 255 : 120);
        draw_text(box_q.x + 6, box_q.y + 6, "Q:", 80, 225, 255);
        draw_wrapped(box_q.x + 22, box_q.y + 6, 55, 2, st.input_q + (st.focus_field == 1 ? "_" : ""), 235, 248, 255, 11);

        draw_text(16, 397, "A (FOR CARVING NEW FACT):", 255, 198, 92);
        fill_rect(box_a.x, box_a.y, box_a.w, box_a.h,
                  st.focus_field == 2 ? 44 : 14, st.focus_field == 2 ? 32 : 22, st.focus_field == 2 ? 18 : 40);
        stroke_rect(box_a.x, box_a.y, box_a.w, box_a.h,
                    st.focus_field == 2 ? 255 : 48, st.focus_field == 2 ? 195 : 80, st.focus_field == 2 ? 85 : 120);
        draw_text(box_a.x + 6, box_a.y + 6, "A:", 255, 192, 78);
        draw_wrapped(box_a.x + 22, box_a.y + 6, 55, 2, st.input_a + (st.focus_field == 2 ? "_" : ""), 255, 240, 200, 11);

        auto draw_btn = [&](const Rect& r, const char* lbl, bool hot, uint8_t br, uint8_t bg, uint8_t bb) {
            fill_rect(r.x, r.y, r.w, r.h, hot ? br : br / 2, hot ? bg : bg / 2, hot ? bb : bb / 2);
            stroke_rect(r.x, r.y, r.w, r.h, br, bg, bb);
            int tw = static_cast<int>(std::strlen(lbl)) * 6;
            draw_text(r.x + (r.w - tw) / 2, r.y + (r.h - 7) / 2, lbl, 245, 250, 255);
        };
        draw_btn(btn_ask,    "ASK (FLUID)",  hover_btn == 1, 215, 80, 255);
        draw_btn(btn_carve,  "CARVE Q->A",   hover_btn == 2, 240, 165, 45);
        draw_btn(btn_load89, "INDEX ALL 89K", hover_btn == 3, 45, 205, 145);
        draw_btn(btn_stream, st.streaming ? "PAUSE STREAM" : "RESUME STREAM", hover_btn == 4, 55, 185, 245);
        draw_btn(btn_replay, "REPLAY FLUID WAVE", hover_btn == 5, 120, 145, 255);

        // =====================================================================
        // LIVE PARALLEL VOXEL FLUID SPLITTING & CANDIDATE ANSWER FLUID METERS!
        // =====================================================================
        fill_rect(14, 514, 364, 294, 14, 17, 34, 255);
        stroke_rect(14, 514, 364, 294,
                    st.fluid_wave_active ? 225 : 44,
                    st.fluid_wave_active ? 110 : 68,
                    st.fluid_wave_active ? 255 : 108);

        draw_text(22, 522, "VOXEL FLUID SPLITTING / CANDIDATE METERS", 235, 155, 255);

        // Live GPU Voxel Fluid Telemetry Strip
        fill_rect(20, 535, 352, 42, 18, 24, 46, 255);
        stroke_rect(20, 535, 352, 42, 65, 88, 138);
        std::snprintf(buf, sizeof(buf), "WAVE STEP %2u/%2u | ACTIVE VOXELS: %u",
                      st.wave_step, st.wave_max_steps, st.gpu_fluid_voxels);
        draw_text(26, 542, buf, 115, 232, 255);
        std::snprintf(buf, sizeof(buf), "FORKS: %u | COLLISIONS(2+): %u | PEAK WT: %.2f",
                      st.gpu_split_forks, st.gpu_collision_vox, st.gpu_peak_vox_wt);
        draw_text(26, 556, buf, 255, 180, 255);

        if (!st.fluid_summary.empty()) {
            draw_text(22, 583, st.fluid_summary.substr(0, 58), 205, 225, 245);
        } else {
            draw_wrapped(22, 583, 56, 2, "Click [ASK (FLUID)] to launch real CUDA voxel splitting and watch all possible answers fill with fluid!", 145, 175, 210, 11);
        }

        // Top-8 Competing Possible Answers Filling Live with Fluid!
        float max_ml = 1.0f;
        for (const auto& c : st.candidates) max_ml = std::max(max_ml, c.target_ml);

        int ry = 602;
        size_t shown = std::min<size_t>(8, st.candidates.size());
        for (size_t i = 0; i < shown; ++i) {
            const auto& c = st.candidates[i];
            bool winner = (i == 0 && c.current_ml > 0.5f);
            fill_rect(20, ry, 352, 23, winner ? 38 : 16, winner ? 32 : 23, winner ? 18 : 40);
            stroke_rect(20, ry, 352, 23,
                        winner ? 255 : (c.drop_sources >= 2 ? 195 : 42),
                        winner ? 210 : (c.drop_sources >= 2 ? 85  : 64),
                        winner ? 75  : (c.drop_sources >= 2 ? 240 : 96));

            // Fluid Fill Bar inside row!
            float frac = std::clamp(c.current_ml / max_ml, 0.0f, 1.0f);
            int bar_w = static_cast<int>(frac * 348.0f);
            if (bar_w > 0) {
                uint8_t br = winner ? 165 : (c.drop_sources >= 2 ? 125 : 22);
                uint8_t bg = winner ? 115 : (c.drop_sources >= 2 ? 38  : 92);
                uint8_t bb = winner ? 24  : (c.drop_sources >= 2 ? 160 : 148);
                fill_rect(22, ry + 2, bar_w, 19, br, bg, bb);
            }

            // Rank + Answer text + Exact Fluid mL + Recombination Drops badge
            std::string ans_short = c.answer.substr(0, 22);
            std::snprintf(buf, sizeof(buf), "#%zu %s", i + 1, ans_short.c_str());
            draw_text(26, ry + 7, buf,
                      winner ? 255 : 230,
                      winner ? 245 : 240,
                      winner ? 185 : 255);

            std::snprintf(buf, sizeof(buf), "%5.1fmL [%u/%u%s]",
                          c.current_ml, c.drop_sources, c.total_sources,
                          c.pos_valve_ok ? "V" : "");
            int rw = static_cast<int>(std::strlen(buf)) * 6;
            draw_text(368 - rw, ry + 7, buf,
                      winner ? 255 : (c.drop_sources >= 2 ? 255 : 170),
                      winner ? 235 : (c.drop_sources >= 2 ? 190 : 225),
                      winner ? 120 : 255);
            ry += 25;
        }

        // Color Legend & Controls Footer
        fill_rect(14, 814, 364, 98, 11, 17, 30, 255);
        stroke_rect(14, 814, 364, 98, 32, 52, 82);
        fill_rect(22, 822, 8, 8, 25, 215, 255);  draw_text(35, 823, "1-Drop Split Fluid (brightness = voxel weight)", 180, 228, 250);
        fill_rect(22, 835, 8, 8, 245, 95, 235);  draw_text(35, 836, "2+ Drops RECOMBINING (collision hotspot!)", 250, 180, 255);
        fill_rect(22, 848, 8, 8, 255, 215, 72);  draw_text(35, 849, "Highest-Fluid Winning Answer Beacon", 255, 235, 165);
        draw_text(22, 867, "SPACE:Stream  T:Burst  S:Step  +/-:Speed  X:Clear", 145, 175, 210);
        draw_text(22, 880, "1:Words 2:POS 3:Sponge 4:Rays Up/Dn:Shell R:Cam", 145, 175, 210);
        draw_text(22, 893, "Enter in Q box: ASK FLUID | Enter in A: CARVE", 125, 165, 205);
    }
};

} // namespace gllm
