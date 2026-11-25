# Geometry-Based Large Language Model (GLLM)

Intelligence is not represented by anything else other than a geometric shape, so our project is currently focused on figuring out a way to represent intelligence in a geometric environment and making it so that it can be run on any kind of consumer hardware that is out there.

---

## Modular CUDA Architecture

GLLM separates a **1-billion-point implicit 3D sphere lattice** (`1241³`, **0 bytes** for unallocated coordinates) and a **1.2M-point outer shell** into three standalone GPU CLI tools that share a self-describing binary checkpoint format (`.gllm`):

- **`bin/gllm_train`** — Dynamically allocates guaranteed non-overlapping `W×H` spaces inside the 3D sphere lattice and binds each class to a unique neighboring outer-shell point (`R = 1.0`). Accumulates overlapping image shadows on GPU via CUDA (`atom.global.add.f32`). **Incremental by design:** run once on digits (`0–9`), then run again on English letters (`A–Z`) or custom image folders into the *same* `.gllm` model—prior spaces and shadows remain untouched with zero overlap.
- **`bin/gllm_eval`** — Scores unseen test images against every trained class's geometric shadow map directly on the GPU via cosine-overlap matching.
- **`bin/gllm_inspect`** — Pulls up or exports (`--json` / `--csv` / `--value` / `--label`) the complete lookup table serialized inside any `.gllm` model (outer-shell scalar ID ↔ class label ↔ outer `(x, y, z)` ↔ unique 3D lattice coordinates).
- **`bin/gllm_viz`** — Zero-copy CUDA–OpenGL 4.6 3D visualizer rendering **only** allocated shadow spaces, their assigned outer-shell points, and 1-to-1 binding lines at 60 FPS (even on a 1-billion-point lattice).

### Repository Layout
```text
include/
  cuda_driver_min.hpp     # Minimal CUDA Driver API bindings (libcuda.so, no nvcc required)
  gllm_model.hpp          # .gllm model format, non-overlapping 3D allocator, IDX & folder loaders
kernels/
  gllm_kernels.ptx        # JIT-compiled CUDA PTX kernels (train, eval, outer shell, O(alloc) viz)
  gllm_kernels.cu         # Readable CUDA C++ reference mirror
src/
  gllm_train.cpp          # Standalone CLI CUDA shadow trainer
  gllm_eval.cpp           # Standalone CLI CUDA evaluator
  gllm_viz.cpp            # Standalone 3D CUDA-OpenGL visualizer
scripts/
  build.sh                # Compiles gllm_train, gllm_eval, gllm_viz
  run_viz.sh              # NVIDIA PRIME offload launcher for gllm_viz
  download_mnist.sh       # Downloads MNIST IDX .gz dataset into data/mnist/
```

---

## Quick Start

### 1. Build All CLIs
```bash
./scripts/build.sh
```

### 2. Download MNIST & Train Digits `0–9` into a 1-Billion-Point Sphere
```bash
./scripts/download_mnist.sh

./bin/gllm_train \
  --model models/sphere.gllm \
  --idx-images data/mnist/train-images-idx3-ubyte.gz \
  --idx-labels data/mnist/train-labels-idx1-ubyte.gz
```

### 3. Incrementally Add New Datasets (e.g., EMNIST English Letters `A–Z` or Custom Folders)
Running `gllm_train` against the **same** `--model models/sphere.gllm` preserves all existing classes and allocates brand-new non-overlapping `W×H` regions + outer-shell points:
```bash
# Example: EMNIST Letters (1..26 -> A..Z, transposed IDX)
./bin/gllm_train \
  --model models/sphere.gllm \
  --idx-images data/emnist/emnist-letters-train-images-idx3-ubyte.gz \
  --idx-labels data/emnist/emnist-letters-train-labels-idx1-ubyte.gz \
  --label-offset 1 --transpose \
  --label-names A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z

# Example: Custom folder dataset (data/custom/<class_name>/*.pgm or *.ppm)
./bin/gllm_train --model models/sphere.gllm --folder data/custom
```

### 4. Evaluate on Unseen Test Images
```bash
./bin/gllm_eval \
  --model models/sphere.gllm \
  --idx-images data/mnist/t10k-images-idx3-ubyte.gz \
  --idx-labels data/mnist/t10k-labels-idx1-ubyte.gz
```

### 5. Visualize in 3D (CUDA + OpenGL 4.6)
```bash
./scripts/run_viz.sh --model models/sphere.gllm
```
**Visualizer Controls:** `Left-drag / Scroll` orbit & zoom • `Z` toggle core close-up vs. full 1B sphere • `Left / Right / Space` focus class • `H` thermal shadow heatmap • `L` binding lines • `Up / Down` ghost-reveal unassigned outer-shell points • `Q` quit.
