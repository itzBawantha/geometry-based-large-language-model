#!/usr/bin/env bash
# =============================================================================
# One-command custom dataset trainer for GLLM!
#
# Usage:
#   ./scripts/train_custom.sh --model models/custom.gllm --input path/to/my_data
#   ./scripts/train_custom.sh --model models/sphere_1b.gllm --input path/to/my_data --auto-invert --center
#   ./scripts/train_custom.sh --model models/demo.gllm --demo
#
# Expected input folder layout:
#   <input_dir>/<class_name>/*.png   (supports .png .jpg .jpeg .bmp .webp .pgm .ppm)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=""
INPUT_DIR=""
SIZE="28"
GRID="1241"
CLUSTERS="1"
EXTRA_PREP_ARGS=()
DEMO_MODE="0"

usage() {
    cat <<EOF
Usage:
  ./scripts/train_custom.sh --model <out.gllm> --input <class_folders_dir> [options]
  ./scripts/train_custom.sh --model <out.gllm> --demo

Options:
  --model <path.gllm>   Target checkpoint (creates new or incrementally appends to existing!)
  --input <dir>         Folder containing one subfolder per class label (<dir>/<label>/*.*)
  --size <N>            Resize images to N x N grayscale (default: 28; 28 enables interactive sketch match)
  --grid <G>            3D sphere lattice grid dimension (default: 1241 -> ~1B implicit lattice points)
  --clusters <K>        GPU spherical K-means sub-shadows per class (default: 1; e.g. 8 isolates variants)
  --invert              Force intensity inversion (for dark drawings on white paper)
  --auto-invert         Automatically detect & invert light-background images
  --center              Tight-bbox center & scale-normalize ink before training
  --demo                Generate & train a zero-dependency 3-class demo (circle, square, triangle)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --input)       INPUT_DIR="$2"; shift 2 ;;
        --size)        SIZE="$2"; shift 2 ;;
        --grid)        GRID="$2"; shift 2 ;;
        --clusters)    CLUSTERS="$2"; shift 2 ;;
        --invert)      EXTRA_PREP_ARGS+=("--invert"); shift ;;
        --auto-invert) EXTRA_PREP_ARGS+=("--auto-invert"); shift ;;
        --center)      EXTRA_PREP_ARGS+=("--center"); shift ;;
        --demo)        DEMO_MODE="1"; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "[ERROR] --model <path.gllm> is required."
    usage
    exit 1
fi

if [[ ! -x "${ROOT}/bin/gllm_train" || ! -x "${ROOT}/bin/gllm_inspect" ]]; then
    echo "[BUILD] Binaries not found — building GLLM CLIs..."
    "${ROOT}/scripts/build.sh"
fi

TMP_PGM="$(mktemp -d "${TMPDIR:-/tmp}/gllm_custom_XXXXXX")"
trap 'rm -rf "${TMP_PGM}"' EXIT

if [[ "${DEMO_MODE}" == "1" ]]; then
    python3 "${ROOT}/scripts/prepare_custom_dataset.py" --generate-demo "${TMP_PGM}" --size "${SIZE}"
    TRAIN_DIR="${TMP_PGM}"
else
    if [[ -z "${INPUT_DIR}" || ! -d "${INPUT_DIR}" ]]; then
        echo "[ERROR] --input <dir> must point to an existing folder of class subdirectories."
        usage
        exit 1
    fi
    # Check if input already contains exclusively .pgm/.ppm and no preprocessing flags were requested
    NON_PNM=$(find "${INPUT_DIR}" -mindepth 2 -maxdepth 2 -type f ! -iname '*.pgm' ! -iname '*.ppm' | head -n 1 || true)
    if [[ -z "${NON_PNM}" && ${#EXTRA_PREP_ARGS[@]} -eq 0 ]]; then
        TRAIN_DIR="${INPUT_DIR}"
        echo "[PREP]   Using existing PGM/PPM folder directly: ${TRAIN_DIR}"
    else
        echo "[PREP]   Converting custom images to ${SIZE}x${SIZE} PGM..."
        python3 "${ROOT}/scripts/prepare_custom_dataset.py" \
            --input "${INPUT_DIR}" \
            --output "${TMP_PGM}" \
            --size "${SIZE}" \
            "${EXTRA_PREP_ARGS[@]}"
        TRAIN_DIR="${TMP_PGM}"
    fi
fi

echo ""
echo "[TRAIN]  Accumulating geometric shadows into ${MODEL} on CUDA GPU (clusters=${CLUSTERS})..."
"${ROOT}/bin/gllm_train" --model "${MODEL}" --folder "${TRAIN_DIR}" --grid "${GRID}" --clusters "${CLUSTERS}"

echo ""
echo "[INSPECT] Serialized Outer-Shell Lookup Table:"
"${ROOT}/bin/gllm_inspect" --model "${MODEL}"

echo ""
echo "[NEXT]   Launch the interactive 3D sphere & sketch matcher:"
echo "         ./scripts/run_viz.sh --model ${MODEL}"
