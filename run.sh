#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Force NVIDIA PRIME render offload + X11/GLX so CUDA-OpenGL interop binds
# directly to the NVIDIA RTX A3000 GPU instead of integrated Intel Mesa.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export SDL_VIDEODRIVER=x11
exec "${SCRIPT_DIR}/cuda_sphere" "$@"
