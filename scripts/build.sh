#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/bin"

echo "[BUILD] Compiling gllm_train..."
g++ -std=c++17 -O3 -Wall -Wextra "${ROOT}/src/gllm_train.cpp" -o "${ROOT}/bin/gllm_train" -lcuda -lz -ldl -lpthread

echo "[BUILD] Compiling gllm_eval..."
g++ -std=c++17 -O3 -Wall -Wextra "${ROOT}/src/gllm_eval.cpp" -o "${ROOT}/bin/gllm_eval" -lcuda -lz -ldl -lpthread

echo "[BUILD] Compiling gllm_inspect..."
g++ -std=c++17 -O3 -Wall -Wextra "${ROOT}/src/gllm_inspect.cpp" -o "${ROOT}/bin/gllm_inspect" -lz

echo "[BUILD] Compiling gllm_bench..."
g++ -std=c++17 -O3 -Wall -Wextra "${ROOT}/src/gllm_bench.cpp" -o "${ROOT}/bin/gllm_bench" -lcuda -lz -ldl -lpthread

echo "[BUILD] Compiling gllm_viz..."
g++ -std=c++17 -O3 -Wall -Wextra "${ROOT}/src/gllm_viz.cpp" -o "${ROOT}/bin/gllm_viz" -lcuda -lSDL2 -lGL -lz -ldl -lpthread

echo "[BUILD] Done: bin/gllm_train, bin/gllm_eval, bin/gllm_inspect, bin/gllm_bench, bin/gllm_viz"
