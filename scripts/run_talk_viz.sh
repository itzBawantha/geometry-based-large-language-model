#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "${ROOT}/bin/gllm_talk_viz" ]]; then
    "${ROOT}/scripts/build.sh"
fi
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
exec "${ROOT}/bin/gllm_talk_viz" "$@"
