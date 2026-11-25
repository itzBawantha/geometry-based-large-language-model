#pragma once

#include "gllm_model.hpp"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gllm {

// =============================================================================
// Generalized Parent-Manifold Inference Dispatcher
//
// Decouples inference eligibility from any hardcoded domain (handwritten vs.
// animals vs. future modalities). Every root Operational Manifold declares:
//   - enabled  (runtime toggleable without recompiling or retraining!)
//   - modality (kSketch28x28, kRgbPyramid, kCustom, ...)
//
// At inference time, the dispatcher generates the exact active 64-bit hardware
// bitmask table for CUDA `match_sketch_pruned` — zeroing masks for any disabled
// or non-matching-modality root manifold so the GPU prunes them in 2 clock cycles!
// =============================================================================

struct RootInferenceSlot {
    uint32_t          manifold_id = 0;
    std::string       name;
    std::string       concept_label;
    InferenceModality modality    = InferenceModality::kNone;
    bool              enabled     = false;
    uint32_t          region_cnt  = 0;
};

class InferenceDispatcher {
public:
    std::vector<RootInferenceSlot> roots;

    void init_from_model(const GeometricModel& model) {
        roots.clear();
        for (const auto& m : model.manifolds) {
            if (m.parent_manifold_id != kInvalidId) continue;
            RootInferenceSlot slot;
            slot.manifold_id   = m.manifold_id;
            slot.name          = m.name;
            slot.concept_label = m.concept_label.empty() ? m.name : m.concept_label;
            slot.modality      = get_infer_modality(m.inference_flags);
            slot.enabled       = is_infer_flag_enabled(m.inference_flags);
            for (const auto& c : model.classes) {
                if (model.get_root_manifold_id(c.manifold_id) == m.manifold_id) slot.region_cnt++;
            }
            roots.push_back(std::move(slot));
        }
    }

    // Apply current root toggle state directly into a GeometricModel (for persistence / GPU descriptor sync)
    void sync_to_model(GeometricModel& model) const {
        for (const auto& slot : roots) {
            for (auto& m : model.manifolds) {
                if (model.get_root_manifold_id(m.manifold_id) == slot.manifold_id) {
                    auto mod = get_infer_modality(m.inference_flags);
                    if (mod == InferenceModality::kNone) mod = slot.modality;
                    m.inference_flags = pack_inference_flags(slot.enabled, mod);
                }
            }
        }
    }

    bool toggle_root_index(size_t idx, GeometricModel& model) {
        if (idx >= roots.size()) return false;
        roots[idx].enabled = !roots[idx].enabled;
        sync_to_model(model);
        std::cout << "[DISPATCH] Root Manifold #" << roots[idx].manifold_id
                  << " '" << roots[idx].name << "' [" << modality_name(roots[idx].modality) << "] -> "
                  << (roots[idx].enabled ? "INFERENCE ENABLED" : "INFERENCE DISABLED (STANDBY)") << "\n";
        return true;
    }

    bool set_root_by_name(const std::string& name, bool enabled, GeometricModel& model) {
        for (auto& slot : roots) {
            if (slot.name == name || slot.concept_label == name) {
                slot.enabled = enabled;
                sync_to_model(model);
                return true;
            }
        }
        return false;
    }

    bool is_region_active_for(const GeometricModel& model, size_t region_idx, InferenceModality query_modality) const {
        if (region_idx >= model.classes.size()) return false;
        const auto& rec = model.classes[region_idx];
        uint32_t root_id = model.get_root_manifold_id(rec.manifold_id);
        for (const auto& slot : roots) {
            if (slot.manifold_id == root_id) {
                return slot.enabled && (slot.modality == query_modality);
            }
        }
        // Fallback for models without explicit roots
        return (query_modality == InferenceModality::kSketch28x28 && rec.width == 28u && rec.height == 28u && rec.channels == 1u);
    }

    uint32_t count_active_regions(const GeometricModel& model, InferenceModality query_modality) const {
        uint32_t cnt = 0;
        for (size_t r = 0; r < model.classes.size(); ++r) {
            if (is_region_active_for(model, r, query_modality)) cnt++;
        }
        return cnt;
    }

    // Build GPU 64-bit hardware bitmask vector where disabled / non-matching-modality regions get 0ULL
    // -> pruned in 2 GPU clock cycles inside `match_sketch_pruned` with zero overhead!
    std::vector<uint64_t> build_active_gpu_masks(const GeometricModel& model,
                                                 const std::vector<uint64_t>& base_masks,
                                                 InferenceModality query_modality) const {
        std::vector<uint64_t> out(base_masks.size(), 0ULL);
        for (size_t r = 0; r < base_masks.size() && r < model.classes.size(); ++r) {
            if (is_region_active_for(model, r, query_modality)) {
                out[r] = base_masks[r];
            }
        }
        return out;
    }

    // Compact status line for HUD / terminal (e.g., "1:handwritten[ON] 2:animals[OFF]")
    std::string hud_status_line() const {
        std::ostringstream ss;
        for (size_t i = 0; i < roots.size(); ++i) {
            if (i > 0) ss << " ";
            ss << (i + 1) << ":" << roots[i].name.substr(0, 11)
               << (roots[i].enabled ? "[ON]" : "[OFF]");
        }
        return ss.str();
    }
};

} // namespace gllm
