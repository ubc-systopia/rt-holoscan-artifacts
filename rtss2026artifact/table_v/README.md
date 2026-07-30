# Table V: partition-search evaluation

This directory reproduces the GPU-partition search results. The primary
comparison uses maximum CPU execution times measured under lockstep contention.
It reports the exact MILP optimum, practical and high-depth Algorithm 1
searches, and the effect of Algorithm 1's coarse-to-fine structure.

## Requirements

- Python 3.10+.
- `networkx` (installed by `../requirements.txt`).

No GPU, CUDA installation, or container GPU passthrough is required: the evaluation consumes GPU-time profiles measured by the other artifact workflow.

## Inputs

- `inputs/application.dot` is the raw augmented application graph.
- `inputs/gpu_execution_profiles.csv` contains `Operator,Resources,Time` rows.
- `inputs/cpu_execution_profiles.csv` contains the lockstep maximum and p99 CPU
  measurements. The paper comparison selects `lockstep_max_ms`.

The profile is authoritative: only graph GPU nodes named in this CSV participate in partitioning. Other GPU nodes receive no synthetic profile or allocation, eliminating the manual removal that the old setup required. Replace these two files for a new experiment; do not carry over search logs or caches.

## Run

From the artifact root, reproduce all headline partition-comparison results
with one command:

```bash
python3 -m pip install -r requirements.txt
python3 table_v/run_partition_comparison.py \
  --output results/partition_comparison
```

This writes `partition_comparison.json` (complete machine-readable results) and
`partition_comparison.csv` (the main solver/search comparison). The default
one-million-partition sample uses a fixed seed and estimates the practical
depth-5 result's rank, the median feasible WCRT, and the success probability of
a uniform random search with the same number of unique evaluations. Use
`--write-search-logs` to retain every Algorithm 1 candidate evaluation.

The deterministic WCRT results should be:

| Method | Depth | Start granularity | Unique partitions | WCRT (ms) |
|---|---:|---:|---:|---:|
| Exact MILP | -- | -- | -- | 19.498 |
| Algorithm 1, coarse-to-fine | 5 | 8 | 47 | 19.819 |
| Algorithm 1, direct | 5 | 2 | 9 | 21.491 |
| Algorithm 1, coarse-to-fine | 15,000 | 8 | 56,875 | 19.498 |
| Algorithm 1, direct | 15,000 | 2 | 81,198 | 19.818 |

Runtime is machine-dependent. On our evaluation machine, model construction
plus the exact MILP solve takes about 0.1 seconds, while the high-depth
coarse-to-fine search takes about 2 seconds. The MILP is specialized to the
TG-DFS max-of-affine response-time formulation. Algorithm 1 only requires an
evaluation oracle, so the same search also applies when candidate allocations
must be assessed by profiling rather than this analysis.

Run the regression checks with:

```bash
python3 -m unittest discover -s table_v -p "test_*.py"
```

The older Table V configuration sweep remains available:

```bash
python3 table_v/run_table_v.py --output results/table_v
```

`results/table_v/table_v.csv` is the rendered table data. Each individual candidate search is retained as `search_d*_g*.csv`.

The implementation modules map directly to the paper: `algorithm_1.py`
implements Algorithm 1, `response_time_analysis.py` evaluates candidate WCRTs,
`exact_partition_milp.py` provides the TG-DFS-specific exact formulation, and
`load_gpu_execution_profiles.py` loads the operator execution-time functions.

To use external raw inputs:

```bash
python3 table_v/run_table_v.py --graph /data/application.dot \
  --profile /data/gpu_execution_profiles.csv --output results/table_v
```
