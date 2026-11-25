#!/usr/bin/env bash
# =============================================================================
# One-command reproducible GLLM benchmark suite:
#   1. Builds all CLI binaries if needed
#   2. Ensures MNIST + EMNIST ByClass holdout test sets are downloaded
#   3. Trains models/sphere_1b.gllm (K=8 sub-shadows/class) if not yet present
#   4. Runs full 68,405-image holdout + live 9-Shift Soft-IoU matcher + OOD audit
#
# Usage:
#   ./scripts/benchmark.sh                         # Full fast CLI benchmark
#   ./scripts/benchmark.sh --visual                # Real-time visual benchmark in 3D gllm_viz!
#   ./scripts/benchmark.sh models/sphere_1b.gllm 100
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VISUAL_MODE="0"
MODEL="${ROOT}/models/sphere_1b.gllm"
LIVE_PER_CLASS="100"

if [[ "${1:-}" == "--visual" ]]; then
    VISUAL_MODE="1"
    MODEL="${2:-${ROOT}/models/sphere_1b.gllm}"
elif [[ -n "${1:-}" ]]; then
    MODEL="$1"
    LIVE_PER_CLASS="${2:-100}"
fi

if [[ ! -x "${ROOT}/bin/gllm_bench" || ! -x "${ROOT}/bin/gllm_train" || ! -x "${ROOT}/bin/gllm_viz" ]]; then
    "${ROOT}/scripts/build.sh"
fi

"${ROOT}/scripts/download_mnist.sh"
"${ROOT}/scripts/download_emnist_byclass.sh"

if [[ ! -f "${MODEL}" ]]; then
    echo "[BENCH] Model ${MODEL} not found — running full 62-class GPU training first..."
    "${ROOT}/scripts/train_digits_and_letters.sh" "${MODEL}"
fi

if [[ "${VISUAL_MODE}" == "1" ]]; then
    exec "${ROOT}/scripts/run_viz.sh" --model "${MODEL}" --benchmark \
        --bench-mnist "${ROOT}/data/mnist" \
        --bench-emnist "${ROOT}/data/emnist_byclass"
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
exec "${ROOT}/bin/gllm_bench" \
    --model "${MODEL}" \
    --mnist-dir "${ROOT}/data/mnist" \
    --emnist-dir "${ROOT}/data/emnist_byclass" \
    --live-per-class "${LIVE_PER_CLASS}"
