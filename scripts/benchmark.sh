#!/usr/bin/env bash
# =============================================================================
# One-command reproducible GLLM benchmark suite:
#   1. Builds all CLI binaries if needed
#   2. Ensures MNIST + EMNIST ByClass holdout test sets are downloaded
#   3. Trains models/sphere_1b.gllm (K=8 sub-shadows/class) if not yet present
#   4. Runs full 68,405-image holdout + live 9-Shift Soft-IoU matcher + OOD audit
#
# Usage:
#   ./scripts/benchmark.sh
#   ./scripts/benchmark.sh models/sphere_1b.gllm 100
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${ROOT}/models/sphere_1b.gllm}"
LIVE_PER_CLASS="${2:-100}"

if [[ ! -x "${ROOT}/bin/gllm_bench" || ! -x "${ROOT}/bin/gllm_train" ]]; then
    "${ROOT}/scripts/build.sh"
fi

"${ROOT}/scripts/download_mnist.sh"
"${ROOT}/scripts/download_emnist_byclass.sh"

if [[ ! -f "${MODEL}" ]]; then
    echo "[BENCH] Model ${MODEL} not found — running full 62-class GPU training first..."
    "${ROOT}/scripts/train_digits_and_letters.sh" "${MODEL}"
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
exec "${ROOT}/bin/gllm_bench" \
    --model "${MODEL}" \
    --mnist-dir "${ROOT}/data/mnist" \
    --emnist-dir "${ROOT}/data/emnist_byclass" \
    --live-per-class "${LIVE_PER_CLASS}"
