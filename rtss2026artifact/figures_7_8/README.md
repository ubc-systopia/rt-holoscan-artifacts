# Figures 7 and 8: CUDA host-side contention

This workflow reproduces the paper's CUDA API contention microbenchmarks.

## Run

From this directory:

```bash
python3 -m pip install -r ../requirements.txt
bash run_paper_experiments.sh
```

The script writes raw CSVs and the Figure 7/8 images to `results/`.

Figure 7 evaluates 1--12 host threads, three malloc sizes (512, 1024, and 2048 bytes), and three square matrix sizes. Figure 8 fixes the malloc size to 1024 bytes and matrix size to 512, evaluates 1--4 threads, and sweeps 4, 6, 8, 16, and 32 SM Green Context partitions. Each configuration has 1,000 repetitions of 100 CUDA calls. Plot whiskers stop at the 95th percentile and hide fliers.
