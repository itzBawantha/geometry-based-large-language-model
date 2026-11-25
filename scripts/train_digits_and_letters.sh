#!/usr/bin/env bash
# =============================================================================
# Reproduces the full 62-class (0-9, A-Z, a-z) 1-Billion-Point Sphere Model
#   - MNIST digits (0..9):         60,000 shadows
#   - EMNIST ByClass (A..Z, a..z): 352,897 shadows
#   - Total:                       412,897 shadows clustered into 62 * K
#                                  guaranteed non-overlapping subspaces
#
# Usage:
#   ./scripts/train_digits_and_letters.sh [models/sphere_1b.gllm] [clusters=8]
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${ROOT}/models/sphere_1b.gllm}"
CLUSTERS="${2:-8}"

mkdir -p "$(dirname "${MODEL}")"
rm -f "${MODEL}"

"${ROOT}/scripts/build.sh"
"${ROOT}/scripts/download_mnist.sh"
"${ROOT}/scripts/download_emnist_byclass.sh"

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"

echo ""
echo "[STEP 1/3] Training MNIST digits (0-9) into ${MODEL} (K=${CLUSTERS} sub-shadows/class)..."
"${ROOT}/bin/gllm_train" \
    --model "${MODEL}" \
    --idx-images "${ROOT}/data/mnist/train-images-idx3-ubyte.gz" \
    --idx-labels "${ROOT}/data/mnist/train-labels-idx1-ubyte.gz" \
    --grid 1241 \
    --clusters "${CLUSTERS}"

echo ""
echo "[STEP 2/3] Incrementally appending EMNIST ByClass letters (A-Z, a-z) into ${MODEL} (K=${CLUSTERS} sub-shadows/class)..."
"${ROOT}/bin/gllm_train" \
    --model "${MODEL}" \
    --idx-images "${ROOT}/data/emnist_byclass/emnist-byclass-train-images-idx3-ubyte.gz" \
    --idx-labels "${ROOT}/data/emnist_byclass/emnist-byclass-train-labels-idx1-ubyte.gz" \
    --min-raw-label 10 --max-raw-label 61 --label-offset 10 --transpose \
    --clusters "${CLUSTERS}" \
    --label-names A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z

echo ""
echo "[STEP 3/3] Verifying serialized outer-shell lookup table (first 20 rows)..."
"${ROOT}/bin/gllm_inspect" --model "${MODEL}" | head -n 25

echo ""
echo "[DONE]   Next steps:"
echo "         Run full 68,405-image holdout & OOD benchmark: ./scripts/benchmark.sh ${MODEL}"
echo "         Launch interactive 3D sphere & sketch matcher: ./scripts/run_viz.sh --model ${MODEL}"
