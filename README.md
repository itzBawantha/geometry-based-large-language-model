# Geometry-Based Large Language Model (GLLM)

> **Intelligence represented purely as geometry** — a zero-weight, zero-backpropagation CUDA Driver API + OpenGL 4.6 geometric intelligence engine designed to run at **~97,000 images/sec** with **sub-millisecond interactive inference** on consumer GPUs (no `nvcc` / CUDA Toolkit required).

---

## Core Architecture

GLLM maps patterns onto a **1-billion-point implicit 3D sphere lattice** (`1241³`, **0 bytes** for unallocated space) bound to a permanent **1.2M-point outer shell** (`R = 1.0`). Every class owns one unique outer-shell node and `K` guaranteed **non-overlapping** `W×H×1` interior subspaces discovered via GPU Spherical K-Means. All five standalone GPU CLI tools share a single self-describing `.gllm` binary checkpoint:

| CLI Tool | Purpose |
|---|---|
| **`bin/gllm_train`** | Allocates guaranteed **0-overlap** `W×H` subspaces inside the 3D sphere, clusters multi-modal shape variants on GPU (`--clusters K`), and accumulates density shadows via `atom.global.add.f32`. Supports incremental continual learning without overwriting prior classes. |
| **`bin/gllm_eval`** | Evaluates any unseen IDX or PGM/PPM folder dataset directly on GPU (`eval_score_shadows`), max-pooling across every class's `K` sub-shadows. |
| **`bin/gllm_bench`** | Reproducible end-to-end audit suite: verifies **0 spatial overlaps**, runs the full **68,405-image holdout**, benchmarks the live **2-stage Soft-IoU CUDA matcher**, and measures **Open-World OOD rejection** (dots & partial strokes). |
| **`bin/gllm_inspect`** | Inspects, queries (`--label` / `--value`), or exports (`--json` / `--csv`) the serialized outer-shell ↔ 3D lattice lookup table inside any `.gllm` file. |
| **`bin/gllm_viz`** | Zero-copy CUDA–OpenGL 4.6 3D sphere visualizer + **Interactive Sketch Inference Canvas** (`224×224` fixed `8:1` raw-proportion downsampling, 64-bit hardware bitmask pruning, 9-shift `Symmetric Bernoulli × Soft-IoU` completeness matcher, `< 28%` `UNKNOWN` gate, and clickable Top-10 spotlights). |

---

## Prerequisites

- **Linux with an NVIDIA GPU** and standard NVIDIA driver (`libcuda.so` — **no CUDA Toolkit / `nvcc` needed**; PTX kernels in `kernels/gllm_kernels.ptx` are JIT-compiled by the driver at runtime).
- `g++` (C++17), `SDL2`, `OpenGL 4.6`, `zlib`, `curl`.
- Optional for custom PNG/JPG/BMP conversion: `python3` + `pillow` (or `ffmpeg` / `ImageMagick`).

```bash
# Ubuntu / Debian
sudo apt install build-essential libsdl2-dev libgl-dev zlib1g-dev curl

# Arch / CachyOS
sudo pacman -S base-devel sdl2 mesa zlib curl

# Fedora
sudo dnf install gcc-c++ SDL2-devel mesa-libGL-devel zlib-devel curl
```

---

## Quick Start (Build → Train → Benchmark → Visualize)

You can use either `make` or `./scripts/*.sh` directly:

### 1. Build All 5 GPU CLIs
```bash
./scripts/build.sh
# or: make
```

### 2. Download & Train the 62-Class 1B-Sphere Model (`0–9`, `A–Z`, `a–z`)
Downloads MNIST (`60,000` digits) + EMNIST ByClass (`352,897` letters) and trains **496 non-overlapping sub-shadows** (`K=8` clusters/class → `62` unique outer-shell nodes) in `< 1 second` of GPU kernel time:
```bash
./scripts/train_digits_and_letters.sh
# or: make train
```

### 3. Run the Full Reproducible Benchmark (68,405 Unseen Images + OOD Audit)
```bash
./scripts/benchmark.sh
# or: make benchmark
```

### 4. Launch the Interactive 3D Sphere & Live Sketch Inference UI
```bash
./scripts/run_viz.sh --model models/sphere_1b.gllm
# or: make viz
```

---

## Verified Benchmark Results (`models/sphere_1b.gllm`, `K=8`, RTX A3000 Laptop GPU)

Zero neural weights, zero gradient steps, **0 spatial overlaps** across `496` subspaces (`62` unique outer-shell nodes):

### Full Unseen Holdout Evaluation (`68,405` images in `0.70 s` total GPU time)
| Unseen Holdout Dataset | Images | 10-Way Digits | Exact 62-Way Top-1 | Visually-Equiv (`case`, `0/O/o`, `1/I/l`) | 62-Way Top-3 | GPU Throughput |
|---|---:|---:|---:|---:|---:|---:|
| **MNIST Test Digits (`0–9`)** | `10,000` | **92.02%** | **83.48%** | **87.76%** | **94.91%** | `87,152 imgs/s` |
| **EMNIST ByClass Letters (`A–Z, a–z`)** | `58,405` | — | **58.99%** | **79.43%** | **87.55%** | `99,304 imgs/s` |
| **Combined All Unseen** | **`68,405`** | — | **62.57%** | **80.65%** | **88.63%** | **`~97k imgs/s`** |

### Live Interactive CUDA Matcher (`9-Shift Symmetric Bernoulli × Continuous Soft-IoU`, `< 28%` `UNKNOWN` Gate)
- **Latency:** `0.61–0.98 ms` per query across all `496` sub-shadows (`× 9` micro-shifts).
- **Valid Unseen Characters (`6,200` stratified):** **90.30%** 10-way digits, **78.60%** visually-equivalent letters (**87.3%** Top-3), **98.7%** accepted above threshold.
- **Open-World Rejection (`50` dots & short strokes):** **100%** of single click-dots (`max ~8.7%`) rejected as **`UNKNOWN`**, **88%** overall partial-stroke rejection.

---

## Train Your Own Custom Dataset

Nothing in GLLM is hardcoded to digits or letters. You can train **any image dataset** into a fresh `.gllm` sphere or append new classes incrementally into an existing checkpoint:

### Option A: Zero-Dependency 30-Second Shape Demo
```bash
./scripts/train_custom.sh --model models/demo_shapes.gllm --demo --clusters 4
./scripts/run_viz.sh --model models/demo_shapes.gllm
# or: make demo
```

### Option B: Custom PNG / JPG / BMP / PGM Class Folders
Organize images as `<dataset>/<class_name>/*.*` and run:
```bash
# Bright strokes on dark background (MNIST/EMNIST style):
./scripts/train_custom.sh --model models/my_model.gllm --input my_dataset --center --clusters 8

# Dark drawings/scans on white paper (auto-inverts background):
./scripts/train_custom.sh --model models/my_model.gllm --input my_dataset --auto-invert --center --clusters 8
```
> **Continual Learning:** Point `--model` at an existing checkpoint (e.g. `models/sphere_1b.gllm`) to append new classes into fresh, guaranteed non-overlapping 3D spaces without touching prior classes.
>
> **Arbitrary Resolutions:** Use `--size 32` or `--size 64` for non-`28×28` grids (`gllm_train`, `gllm_eval`, `gllm_inspect`, and 3D `gllm_viz` support arbitrary `W×H`; `28×28` enables the live interactive sketch matcher).

### Option C: Any Custom IDX `.gz` Dataset (Fashion-MNIST, KMNIST, EMNIST, etc.)
```bash
./bin/gllm_train \
  --model models/fashion.gllm \
  --idx-images path/to/train-images-idx3-ubyte.gz \
  --idx-labels path/to/train-labels-idx1-ubyte.gz \
  --clusters 8 \
  --label-names T-shirt,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,AnkleBoot
```

Key `gllm_train` flags:
- `--clusters <K>` — GPU Spherical K-Means sub-shadows per class (default `1`; e.g., `8` isolates structural variants).
- `--cluster-iters <I>` — Spherical K-Means iterations (default `10`).
- `--min-raw-label <N> --max-raw-label <M>` — Filter raw IDX label range.
- `--label-offset <O>` — Subtract offset before indexing `--label-names`.
- `--transpose` — Transpose row/column layout (required for EMNIST).
- `--grid <G>` — Implicit sphere lattice grid dimension (default `1241` → ~1.0B sphere points).

---

## Inspect & Evaluate Checkpoints

```bash
# Inspect serialized outer-shell <-> 3D lattice lookup table
./bin/gllm_inspect --model models/sphere_1b.gllm
./bin/gllm_inspect --model models/sphere_1b.gllm --label r
./bin/gllm_inspect --model models/sphere_1b.gllm --json > lookup_table.json

# Evaluate any unseen test dataset on GPU (IDX or folder of PGM/PPM)
./bin/gllm_eval \
  --model models/sphere_1b.gllm \
  --idx-images data/mnist/t10k-images-idx3-ubyte.gz \
  --idx-labels data/mnist/t10k-labels-idx1-ubyte.gz
```

---

## Interactive Visualizer Controls (`gllm_viz`)

- **Left Sketch Canvas (`224×224`, fixed 8:1 raw-proportion scale):** `Left-drag` draw inside the subtle `22×22` scale guide • `[CAPTURE]` or `Enter` run 2-stage CUDA match • `[CLEAR]` / `Right-click` / `C` clear • Click any **Top-10 row** to spotlight all `K` sub-shadows of that class in 3D.
- **Right 3D Sphere Viewport:** `Left-drag` / `Scroll` orbit & zoom • `Z` toggle core close-up vs. full 1B sphere • `Left / Right / Space` cycle class focus • `H` thermal density heatmap • `L` toggle `K`-to-1 binding rays • `Up / Down` ghost-reveal unassigned 1.2M outer-shell points • `R` reset view • `Q / Esc` quit.

---

## Repository Layout

```text
Makefile                              # Convenient build / train / benchmark / demo / viz targets
include/
  cuda_driver_min.hpp                 # Minimal CUDA Driver API bindings (libcuda.so, no nvcc)
  gllm_model.hpp                      # .gllm format, 0-overlap 3D allocator, IDX & folder loaders
  hud_canvas.hpp                      # 224x224 raw-proportion sketch canvas & HUD renderer
kernels/
  gllm_kernels.ptx                    # JIT-compiled CUDA PTX kernels (shell, train, eval, emit, match)
  gllm_kernels.cu                     # Readable CUDA C++ reference mirror
src/
  gllm_train.cpp                      # GPU shadow trainer + Spherical K-Means clustering
  gllm_eval.cpp                       # GPU batch holdout evaluator
  gllm_bench.cpp                      # Full holdout + live Soft-IoU matcher + OOD + overlap auditor
  gllm_inspect.cpp                    # .gllm outer-point lookup table inspector & exporter
  gllm_viz.cpp                        # Interactive 3D CUDA-OpenGL visualizer + live sketch inference
scripts/
  build.sh                            # Compiles all 5 CLIs into bin/
  download_mnist.sh                   # Idempotent MNIST downloader
  download_emnist_byclass.sh          # Idempotent EMNIST ByClass downloader
  train_digits_and_letters.sh         # One-command 62-class (0-9, A-Z, a-z) 1B-sphere trainer
  benchmark.sh                        # One-command 68,405-image holdout + OOD benchmark suite
  prepare_custom_dataset.py           # Converts custom PNG/JPG/BMP folders (or demo) to PGM
  train_custom.sh                     # One-command custom dataset converter + GPU trainer
  run_viz.sh                          # NVIDIA PRIME offload launcher for gllm_viz
```
