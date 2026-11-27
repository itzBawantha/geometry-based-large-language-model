# Geometry-Based Large Language Model (GLLM)
# Builds all 5 modular CUDA Driver API + OpenGL 4.6 CLIs (no nvcc required)

MODEL ?= models/sphere_1b.gllm
CLUSTERS ?= 8

.PHONY: all talk-viz train benchmark benchmark-viz demo viz inspect clean legacy

all:
	@./scripts/build.sh

talk-viz: all
	@./scripts/run_talk_viz.sh --ask "What is the capital of France?"

train: all
	@./scripts/train_digits_and_letters.sh $(MODEL) $(CLUSTERS)

benchmark: all
	@./scripts/benchmark.sh $(MODEL)

benchmark-viz: all
	@./scripts/benchmark.sh --visual $(MODEL)

demo: all
	@./scripts/train_custom.sh --model models/demo_shapes.gllm --demo --clusters 4
	@./scripts/run_viz.sh --model models/demo_shapes.gllm

viz: all
	@./scripts/run_viz.sh --model $(MODEL)

inspect: all
	@./bin/gllm_inspect --model $(MODEL)

clean:
	rm -rf bin cuda_sphere *.o

# Legacy v1 single-binary sphere prototype
legacy: main.cpp sphere_kernel.ptx
	g++ -std=c++17 -O3 -Wall -Wextra main.cpp -o cuda_sphere -lcuda -lSDL2 -lGL -ldl -lpthread
