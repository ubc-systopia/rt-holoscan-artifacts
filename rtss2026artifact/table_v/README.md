# Table V: partition-search evaluation

> **Analysis documentation:** Before modifying TG-DFS or its GPU-aware graph
> transformation, read [`ANALYSIS_GUIDE.md`](ANALYSIS_GUIDE.md). It explains
> the 2025 formal basis, the GPU-aware correctness argument, implementation
> invariants, and the relevant paper and code references.

This directory reproduces the GPU-partition search results. The primary
comparison uses maximum CPU execution times measured under lockstep contention.
It reports the exact MILP optimum, practical and high-depth Algorithm 1
searches, the effect of Algorithm 1's coarse-to-fine structure, and a baseline
computed with the GPU-oblivious analysis from the 2025 artifact.

## Requirements

- Python 3.10+.
- `networkx` (installed by `../requirements.txt`).

No GPU, CUDA installation, or container GPU passthrough is required: the evaluation consumes GPU-time profiles measured by the other artifact workflow.

## Inputs

- `inputs/application.dot` is the raw augmented application graph.
- `inputs/gpu_execution_profiles.csv` contains `Operator,Resources,Time` rows.
- `inputs/cpu_execution_profiles.csv` contains the lockstep maximum and p99 CPU
  measurements. The paper comparison selects `lockstep_max_ms`.
- `inputs/gpu_execution_times_2025_baseline.csv` records the 142-SM GPU times
  (in microseconds) used to construct the synchronous, serialized 2025
  baseline. It includes the seven partitioned operators and the separately
  profiled Holoviz GPU work; operators without a measured GPU component are
  explicitly assigned zero.

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
The command also writes `application_2025_serialized.dot`, the exact linear
input graph passed to the unmodified `rtss2025artifact/code/TG_DFS.py`. In
addition to the chain edges, the graph contains a direct source-to-sink edge.
Its downstream condition prevents operators from different input iterations
from being pipelined while each assumes exclusive use of the full GPU.

The deterministic WCRT results should be:

| Method | Depth | Start granularity | Unique partitions | WCRT (ms) |
|---|---:|---:|---:|---:|
| 2025 TG-DFS, serialized full GPU | -- | -- | -- | 32.083 |
| Exact MILP | -- | -- | -- | 19.498 |
| Algorithm 1, coarse-to-fine | 5 | 8 | 47 | 19.819 |
| Algorithm 1, direct | 5 | 2 | 9 | 21.491 |
| Algorithm 1, coarse-to-fine | 15,000 | 8 | 56,875 | 19.498 |
| Algorithm 1, direct | 15,000 | 2 | 81,198 | 19.818 |

For the 2025 baseline, the pipeline collapses each original operator into one
node whose WCET is its lockstep-maximum CPU time plus its execution time with
all 142 SMs. It then places the operators in one topologically valid linear
order, which serializes every GPU user, and adds a direct edge from the source
to the sink to prevent cross-iteration pipelining. It then runs the unmodified
2025 TG-DFS implementation. The resulting bound is 32.083 ms. The new
analysis's exact 19.498 ms bound is 39.23% lower (1.65 times smaller).

The pipeline also checks all 810 linear extensions consistent with the original
application DAG. With the source-to-sink edge, every order produces the same
32.083 ms result. This both removes the previous ordering ambiguity and checks
that the baseline contains no benefit from cross-iteration overlap.

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

## Independent MILP correctness verification

The MILP is also checked by an implementation that does not use the MILP
constraints to search the partition space:

```bash
python3 table_v/verify_milp_correctness.py \
  --output results/milp_verification.json
```

This command performs four checks:

1. It evaluates the returned allocation with the raw TG-DFS expression strings.
2. It evaluates the same allocation with a separately implemented expression
   parser.
3. It enumerates every one of the 131,115,985 feasible granularity-2
   partitions and checks whether any has a WCRT below the MILP result.
4. It compares 100 randomly generated reduced MILPs against pure-Python brute
   force.

For the checked-in inputs, all three WCRT evaluations return 19.498 ms at
`[2, 2, 2, 22, 66, 38, 10]`; exhaustive enumeration finds no lower partition;
all 100 randomized cases match brute force; and HiGHS reports a zero MIP gap.
The complete machine-readable certificate is
[`results/milp_verification.json`](../results/milp_verification.json).

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
