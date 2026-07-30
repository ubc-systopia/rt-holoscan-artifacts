# RTSS 2026 artifact

This artifact contains two independent reproduction workflows.

## Workflows

- [figures_7_8/README.md](figures_7_8/README.md) reproduces Figures 7 and 8.
- [table_v/README.md](table_v/README.md) reproduces Table V.

## Shared requirements

- Linux container with NVIDIA Container Toolkit GPU access (`--gpus all`).
- CUDA Toolkit 12.4 or newer, including `nvcc`, CUDA Runtime headers, and CUDA Driver headers/library. Green Context support is required for Figure 8.
- NVIDIA driver and GPU that support CUDA Green Contexts. To reproduce every Figure 8 point, the GPU must provide at least 128 SMs (four 32-SM contexts); the paper used an RTX 6000 Ada with 142 SMs.
- Python 3.10+ with the packages in `requirements.txt`.
- `make` and a C++17-capable host compiler.
