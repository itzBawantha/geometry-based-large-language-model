#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/data/mnist"
mkdir -p "${OUT}"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    if [[ -s "${OUT}/${f}" ]]; then
        continue
    fi
    echo "[DL] ${f}..."
    curl -fSL -o "${OUT}/${f}" "${BASE}/${f}"
done
echo "[OK] MNIST ready in ${OUT}"
