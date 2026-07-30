#!/usr/bin/env bash
set -euo pipefail

# Reproduces Figs. 7 and 8.  Results are intentionally kept separate from plots.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
results="${1:-$root/results}"
mkdir -p "$results/baseline" "$results/green"

make -C "$root"
benchmark="$root/rtss2026artifact_benchmark"
export CUDA_DEVICE_MAX_CONNECTIONS=12

# Figure 7: 1,000 repetitions; 100 calls per timed sequence.
for size in 512 1024 2048; do
  "$benchmark" --malloc-only --malloc-size "$size" --calls 100 --repetitions 1000 \
    --output "$results/baseline/malloc_${size}.csv"
  "$benchmark" --matmul-only --matmul-size "$size" --calls 100 --repetitions 1000 \
    --output "$results/baseline/matmul_${size}.csv"
done

# Figure 8: fixed payloads, Green Contexts of 4--32 SMs, at most four threads.
for sm_count in 4 6 8 16 32; do
  "$benchmark" --green-contexts --sm-partition "$sm_count" --malloc-only \
    --malloc-size 1024 --calls 100 --repetitions 1000 \
    --output "$results/green/malloc_sm${sm_count}.csv"
  "$benchmark" --green-contexts --sm-partition "$sm_count" --matmul-only \
    --matmul-size 512 --calls 100 --repetitions 1000 \
    --output "$results/green/matmul_sm${sm_count}.csv"
done

python3 "$root/plot_paper_figures.py" --results "$results" --output "$results/figures"
