#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/data/emnist_byclass"
mkdir -p "${OUT}"
BASE="https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main"
for f in emnist-byclass-train-images-idx3-ubyte.gz emnist-byclass-train-labels-idx1-ubyte.gz emnist-byclass-test-images-idx3-ubyte.gz emnist-byclass-test-labels-idx1-ubyte.gz emnist-byclass-mapping.txt; do
    if [[ -s "${OUT}/${f}" ]]; then
        continue
    fi
    echo "[DL] ${f}..."
    curl -fSL -o "${OUT}/${f}" "${BASE}/${f}"
done
echo "[OK] EMNIST ByClass ready in ${OUT}"
