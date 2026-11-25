# Geometry-Based Large Language Model

Intelligence is not represented by anything else other than a geometric shape, so our project is currently focused on figuring out a way to represent intelligence in a geometric environment and making it so that it can be run on any kind of consumer hardware that is out there.

---

## Current Prototype: CUDA Outer-Shell Point Sphere Visualizer

Real-time GPU visualization of a 3D spherical outer shell built from **1,200,000+ points**, computed each frame via the NVIDIA CUDA Driver API (`libcuda.so` + JIT-compiled PTX) and rendered zero-copy through CUDA–OpenGL 4.6 VBO interoperability.

### Files
- `main.cpp` — C++17 SDL2 + OpenGL 4.6 + CUDA Driver API visualizer
- `sphere_kernel.ptx` — CUDA PTX kernel JIT-compiled directly by the NVIDIA driver (no `nvcc` required)
- `sphere_kernel.cu` — Equivalent CUDA C++ kernel source
- `Makefile` — Build configuration
- `run.sh` — Launcher configured for NVIDIA PRIME render offload

### Build & Run
```bash
g++ -std=c++17 -O3 -Wall -Wextra main.cpp -o cuda_sphere -lcuda -lSDL2 -lGL -ldl -lpthread
./run.sh --points 1200000
```

### Controls
- **Left-drag** — Orbit camera
- **Scroll / `+` / `-`** — Zoom in/out
- **`W`** — Cycle outer-shell modes (*Pure Outer Sphere*, *Harmonic Ripples*, *Outer Shell Crust*)
- **`M`** — Cycle color palettes (*Cosmic*, *Solar*, *Hologram*, *Normal RGB*)
- **`Space`** — Pause/resume animation
- **`A`** — Toggle auto-rotation
- **`C`** — Toggle dark inner core occlusion
- **`[` / `]`** — Adjust point size
- **`R`** — Reset camera
- **`Q` / `Esc`** — Quit
