# Geometry-Based Large Language Model (GLLM)

Intelligence is not represented by anything else other than a geometric shape, so our project is currently focused on figuring out a way to represent intelligence in a geometric environment and making it so that it can be run on any kind of consumer hardware that is out there.

---

## Modular CUDA Architecture

GLLM separates a **1-billion-point implicit 3D sphere lattice** (`1241³`, **0 bytes** for unallocated coordinates) and a **1.2M-point outer shell** into four standalone GPU CLI tools that share a self-describing binary checkpoint format (`.gllm`):

- **`bin/gllm_train`** — Dynamically allocates guaranteed non-overlapping `W×H` spaces inside the 3D sphere lattice and binds each class to a unique neighboring outer-shell point (`R = 1.0`). Accumulates overlapping image shadows on GPU via CUDA (`atom.global.add.f32`). Supports universal **GPU Spherical K-Means Multi-Shadow Clustering (`--clusters K`)** to separate distinct structural variants of any class into crisp, non-overlapping sub-shadow regions that all converge onto the class's single outer-shell point. **Incremental by design:** train on digits (`0–9`), append English letters (`A–Z`, `a–z`), or add your own custom image folders into the *same* `.gllm` model—prior spaces and shadows remain untouched with **0 overlaps**.
- **`bin/gllm_eval`** — Evaluates unseen test datasets against every trained class (max-pooling across all sub-shadow regions per class) directly on the GPU.
- **`bin/gllm_inspect`** — Inspects or exports (`--json` / `--csv` / `--value` / `--label`) the complete lookup table serialized inside any `.gllm` checkpoint (outer-shell scalar ID ↔ class label ↔ outer `(x, y, z)` ↔ unique 3D lattice coordinates).
- **`bin/gllm_viz`** — Zero-copy CUDA–OpenGL 4.6 3D visualizer + **Interactive Sketch Inference Canvas** with:
  - **224×224 live drawing canvas** & canonical **22×22-in-28×28** CUDA preview.
  - **Stage-1 Hardware 64-bit Bitmask Pruning** (`and.b64` + `popc.b64`) to drop structurally impossible sub-shadows in 2 clock cycles.
  - **Stage-2 Multi-Shift Bias-Free Symmetric Bernoulli Log-Likelihood** (`dx, dy ∈ {-1, 0, +1}`) across all surviving sub-shadows, eliminating both sparse-shadow and dense-shadow bias in 1 FMA/pixel and max-pooling per class onto its single outer-shell node.
  - **Open-World Unknown Detection** (`< 35%` overlap), clickable **Top-10 Overlap & Relative Probability bars**, and **Stage-1 Pruned class readout**.

### Prerequisites
- Linux with an **NVIDIA GPU** and standard NVIDIA driver (`libcuda.so` — **no `nvcc` / CUDA Toolkit required**, PTX is JIT-compiled by the driver at runtime)
- `g++` (C++17), `SDL2`, `OpenGL 4.6`, `zlib`
- Optional for custom PNG/JPG conversion: `python3` + `pillow` (or `ImageMagick` / `ffmpeg`)

### Repository Layout
```text
include/
  cuda_driver_min.hpp            # Minimal CUDA Driver API bindings (libcuda.so, no nvcc)
  gllm_model.hpp                 # .gllm format, non-overlapping 3D allocator, IDX & folder loaders
  hud_canvas.hpp                 # Interactive 224x224 sketch canvas, canonicalizer, HUD renderer
kernels/
  gllm_kernels.ptx               # JIT-compiled CUDA PTX kernels (shell, train, eval, emit, matcher)
  gllm_kernels.cu                # Readable CUDA C++ reference mirror
src/
  gllm_train.cpp                 # Standalone CLI CUDA shadow trainer
  gllm_eval.cpp                  # Standalone CLI CUDA batch evaluator
  gllm_inspect.cpp               # Standalone .gllm outer-point lookup table inspector
  gllm_viz.cpp                   # Interactive 3D CUDA-OpenGL visualizer + sketch inference UI
scripts/
  build.sh                       # Compiles all 4 CLIs into bin/
  run_viz.sh                     # NVIDIA PRIME offload launcher for gllm_viz
  download_mnist.sh              # Downloads MNIST (0-9) into data/mnist/
  download_emnist_byclass.sh     # Downloads EMNIST ByClass (A-Z, a-z) into data/emnist_byclass/
  train_digits_and_letters.sh    # One-command 62-class (0-9, A-Z, a-z) 1B-sphere training pipeline
  prepare_custom_dataset.py      # Converts custom PNG/JPG/BMP folders (or generates a demo) to PGM
  train_custom.sh                # One-command custom dataset converter + GPU trainer + inspector
```

---

## Quick Start

### 1. Build All CLIs
```bash
./scripts/build.sh
```

### 2. Train the 62-Class Model (MNIST `0–9` + EMNIST `A–Z`, `a–z`)
Run the full pipeline in one command (~412,897 shadows trained in `< 20 ms` of GPU kernel time):
```bash
./scripts/train_digits_and_letters.sh
```
Then launch the interactive 3D visualizer & sketch canvas:
```bash
./scripts/run_viz.sh --model models/sphere_1b.gllm
```

---

## Train Your Own Custom Dataset

GLLM is completely dynamic—nothing is hardcoded to 10 or 62 classes. You can train **any custom image dataset** (handwritten symbols, custom alphabets, icons, medical tiles, geometric shapes, etc.) either into a brand-new `.gllm` sphere or **appended incrementally** into an existing `.gllm` model.

### Option A: Quick Zero-Dependency Demo (30 Seconds)
Generate a synthetic 3-class geometric dataset (`circle`, `square`, `triangle`), train it on your GPU, inspect the allocated 3D coordinates, and visualize it:
```bash
./scripts/train_custom.sh --model models/demo_shapes.gllm --demo
./scripts/run_viz.sh --model models/demo_shapes.gllm
```

### Option B: Train From Your Own PNG / JPG / BMP / PGM Folders
1. Organize your images into **one subfolder per class label** (the subfolder name becomes the class label stored in `.gllm` and bound to its outer-shell point):
   ```text
   my_dataset/
     star/
       001.png
       002.png
     arrow/
       001.jpg
       002.jpg
     heart/
       001.png
   ```
2. Run the one-command trainer:
   ```bash
   # Standard bright-on-dark images (like MNIST/EMNIST):
   ./scripts/train_custom.sh --model models/my_model.gllm --input my_dataset --center

   # Dark drawings/scans on white paper (automatically inverts white background -> black):
   ./scripts/train_custom.sh --model models/my_model.gllm --input my_dataset --auto-invert --center
   ```
   > **Tip — Continual Learning:** Point `--model` at an *existing* checkpoint (e.g., `--model models/sphere_1b.gllm`) to append your new custom classes right alongside existing digits & letters in brand-new, non-overlapping 3D spaces!
   >
   > **Custom Resolutions:** Use `--size 32` or `--size 64` for non-28×28 image grids (`gllm_train`, `gllm_eval`, `gllm_inspect`, and 3D `gllm_viz` support arbitrary `W×H`; `28×28` is required for the live interactive sketch matcher).

### Option C: Train Any Custom IDX `.gz` Dataset (Fashion-MNIST, Kuzushiji-MNIST, EMNIST, etc.)
```bash
./bin/gllm_train \
  --model models/custom_idx.gllm \
  --idx-images path/to/train-images-idx3-ubyte.gz \
  --idx-labels path/to/train-labels-idx1-ubyte.gz \
  --label-names T-shirt,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,AnkleBoot
```
Optional flags for subsetting/orienting IDX datasets:
- `--min-raw-label <N> --max-raw-label <M>` — Filter to a specific raw label range.
- `--label-offset <K>` — Offset raw label indices before indexing into `--label-names`.
- `--transpose` — Transpose row/column layout (required for EMNIST datasets).
- `--clusters <K>` — GPU Spherical K-Means sub-shadows per class (default `1`; e.g., `--clusters 8` separates multi-modal shape variants into distinct non-overlapping subspaces bound to the class's single outer-shell point).
- `--cluster-iters <I>` — Number of GPU Spherical K-Means iterations (default `10`).
- `--grid <G>` — Implicit sphere lattice grid size (default `1241` → ~1,000,723,196 implicit points).

---

## Inspect & Evaluate Models

```bash
# Print full serialized outer-point <-> 3D lattice lookup table (or export --json / --csv)
./bin/gllm_inspect --model models/sphere_1b.gllm
./bin/gllm_inspect --model models/sphere_1b.gllm --label B
./bin/gllm_inspect --model models/sphere_1b.gllm --json > lookup_table.json

# Evaluate on unseen test images on GPU (IDX or folder of PGM/PPM)
./bin/gllm_eval \
  --model models/sphere_1b.gllm \
  --idx-images data/mnist/t10k-images-idx3-ubyte.gz \
  --idx-labels data/mnist/t10k-labels-idx1-ubyte.gz
```

---

## Interactive Visualizer Controls (`gllm_viz`)
- **Left Sidebar Canvas:** `Left-drag` to draw • `[CAPTURE]` or `Enter` to run 2-stage CUDA match • `[CLEAR]` / `Right-click` / `C` to clear • Click any **Top-10 row** to spotlight that class in 3D.
- **3D Viewport:** `Left-drag / Scroll` orbit & zoom • `Z` toggle core close-up vs. full 1B sphere • `Left / Right / Space` cycle class focus • `H` thermal shadow heatmap • `L` binding lines • `Up / Down` ghost-reveal unassigned outer-shell points • `R` reset view • `Q / Esc` quit.
