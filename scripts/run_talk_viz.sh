#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "${ROOT}/bin/gllm_talk_viz" ]]; then
    "${ROOT}/scripts/build.sh"
fi

# Auto-fetch the 816k MS MARCO + Clean OpenHermes-2.5 sponge (87% Top-1 / 99% Top-5)
# from the official GitHub Release on first run if not already present locally.
SPONGE_816K="${ROOT}/data/msmarco_openhermes_clean_sponge.bin"
RELEASE_URL="https://github.com/itzBawantha/geometry-based-large-language-model/releases/download/v2.0-816k-sponge/msmarco_openhermes_clean_sponge.bin.gz"
if [[ ! -f "${SPONGE_816K}" ]] && [[ ! -f "/run/media/liveuser/GLLM_DATA/gllm_datasets/extracted/talk_index/msmarco_openhermes_clean_sponge.bin" ]]; then
    if command -v curl >/dev/null 2>&1 && command -v gunzip >/dev/null 2>&1; then
        echo "[GLLM] Downloading 816k MS MARCO + OpenHermes-2.5 sponge from GitHub Release v2.0-816k-sponge..."
        mkdir -p "${ROOT}/data"
        if curl -fSL --progress-bar "${RELEASE_URL}" -o "${SPONGE_816K}.gz"; then
            gunzip -f "${SPONGE_816K}.gz"
            echo "[GLLM] Ready: ${SPONGE_816K}"
        else
            rm -f "${SPONGE_816K}.gz"
            echo "[GLLM] Notice: could not download 816k release asset; falling back to bundled data/msmarco_sponge.bin"
        fi
    fi
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
cd "${ROOT}"
exec "${ROOT}/bin/gllm_talk_viz" "$@"
