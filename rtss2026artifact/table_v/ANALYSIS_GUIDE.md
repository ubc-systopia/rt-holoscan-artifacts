# Guide to TG-DFS and the GPU-aware response-time analysis

This document is the recommended starting point for anyone modifying or
reviewing the response-time analysis in this artifact. It explains the common
idea behind the original 2025 TG-DFS analysis and the GPU-aware adaptation,
identifies the authoritative formal and code references, and records several
implementation details that are easy to get subtly wrong.

## Authoritative references

Formal definitions and arguments:

- The original analysis is from Schowitz et al., *Faster, Exact, More General
  Response-time Analysis for NVIDIA Holoscan Applications*, IEEE RTSS 2025.
  Its bibliographic entry is in
  [`greencontext/references/biblio.bib`](../../../rt-holoscan/greencontext/references/biblio.bib)
  under `Schowitz2025`.
- The active explanation of both the original analysis and GPU-aware extension
  is
  [`greencontext/sections/rta-new.tex`](../../../rt-holoscan/greencontext/sections/rta-new.tex).
  In particular, see *Background: CPU-Only Response-Time Analysis*,
  *Augmenting the DAG and Trace Graph with GPU Nodes*, and *Equivalence to the
  CPU-Only Setting*.
- [`greencontext/sections/rta.tex`](../../../rt-holoscan/greencontext/sections/rta.tex)
  is an older draft. It contains useful detailed proof sketches, including the
  common-ancestor, upper-bound, and tightness lemmas, but much of that material
  is commented out. Treat `rta-new.tex` as the current paper prose.

Implementations:

- Original 2025 TG-DFS:
  [`rtss2025artifact/code/TG_DFS.py`](../../rtss2025artifact/code/TG_DFS.py)
- Corrected GPU-aware TG-DFS:
  [`response_time_analysis.py`](response_time_analysis.py)
- GPU-aware application input:
  [`inputs/application.dot`](inputs/application.dot)
- Exact partition MILP, which consumes TG-DFS expressions but is not part of
  the response-time proof:
  [`exact_partition_milp.py`](exact_partition_milp.py)
- Coarse-to-fine partition search:
  [`algorithm_1.py`](algorithm_1.py)
- Reproducible comparison pipeline:
  [`run_partition_comparison.py`](run_partition_comparison.py)
- Independent MILP verifier:
  [`verify_milp_correctness.py`](verify_milp_correctness.py)
- Construction of the serialized 2025 baseline:
  [`baseline_2025.py`](baseline_2025.py)

## 1. Original 2025 model

Let the Holoscan application be a DAG `D`. An operator executes repeatedly as
successive input items move through the application. If `s_i(o)` and `f_i(o)`
are the start and finish times of operator `o` in iteration `i`, scheduling is
governed by three constraints:

1. **Data dependency:** `s_i(o) >= f_i(p)` for every predecessor `p`.
2. **Sequential execution:** `s_i(o) >= f_(i-1)(o)`.
3. **Downstream blocking:** for a size-one queue and successor `q`,
   `s_(i+1)(o) >= s_i(q)`.

The third condition is start-to-start rather than finish-to-start. It expresses
backpressure: the upstream operator may advance once its successor has accepted
the preceding item.

The analysis expands `D` conceptually into an infinite **trace graph**. A trace
node `o^i` represents one execution of `o`. Finish-to-start constraints carry
the execution time of their source node; downstream start-to-start constraints
carry zero cost. Consequently, the earliest possible start time of a trace node
is the cost of the longest path leading to it.

For iteration `i`, response time is

```text
L(sink^i) - L(source^i),
```

where `L(v)` is the longest-path cost to `v`. The trace graph is infinite, so it
cannot be constructed and searched directly.

## 2. What TG-DFS computes

TG-DFS traverses the transpose of the trace graph, beginning at a sink
iteration and moving backward through data-dependency, sequential-execution,
and downstream-blocking edges. A path terminates when it reaches a node known
to be a shared ancestor of the target source and sink. The path before that
ancestor is shared by both longest-path calculations and cancels from their
difference.

The traversal produces strings representing the finite set of potentially
critical suffixes. A term such as `A_3` means that the WCET of node `A` in
iteration 3 contributes to that suffix. In the GPU-aware implementation, a
terminal `!` marks the path-specific cancellation case: `compute_max()` does
not count that final term. The WCRT is the maximum evaluated expression.

The algorithm never constructs the infinite trace graph. The DFS tree created
in memory is only the finite backward traversal needed to produce these
expressions.

### Why the traversal terminates

Within an iteration, every node has a path toward the sink. Downstream edges
move opposite the application DAG and forward by one iteration. Repeated use of
these edges therefore connects sufficiently old trace-graph layers to the
target source. The 2025 proof bounds how far backward the traversal must go by
the application DAG's path depth.

### Why the result is sound and exact

**Soundness** follows from the longest-path representation: every scheduling
constraint is an edge, so no valid execution can start a node earlier than the
longest incoming constraint path permits. The maximum suffix expression
therefore upper-bounds response time.

**Exactness under the analysis model** follows from the assumed execution-time
domain. Each execution may take any value from zero through its WCET. Set the
nodes on a maximizing suffix to their WCETs and competing off-path executions
to zero; the maximizing path is then realizable. Thus the computed upper bound
can be attained within the model.

This does **not** mean exact prediction of a physical deployment. The guarantee
is conditional on the graph, scheduling rules, execution-time bounds, queue
size, and resource-isolation assumptions being correct.

## 3. Why the 2025 representation is insufficient for asynchronous GPU work

The 2025 graph gives every Holoscan operator one WCET. If that WCET includes
both CPU and GPU work, the operator is implicitly synchronous: it cannot finish
and be scheduled again until all of the folded-in GPU work completes. Real
Holoscan operators commonly enqueue kernels and return while those kernels are
still running. Folding this behavior into one synchronous WCET loses CPU/GPU
overlap and cannot represent GPU dependencies independently.

Moreover, a WCET measured while an operator owns the entire GPU cannot be used
for several operators running concurrently. A GPU-oblivious comparison must
serialize those operators. The baseline generated by `baseline_2025.py`:

1. Adds each operator's lockstep-maximum CPU time to its measured 142-SM GPU
   time.
2. Collapses every operator to one ordinary 2025 node.
3. Places the operators in a linear chain.
4. Adds a direct edge from the chain's source to its sink.

The fourth step is essential. A chain alone still permits different iterations
to be pipelined through downstream blocking. The direct source-to-sink edge
adds a downstream constraint from the sink of iteration `i` to the source of
iteration `i+1`, eliminating that cross-iteration benefit. With the current
inputs, all 810 topologically valid chain orders then produce the same 32.083 ms
2025 bound.

## 4. GPU-aware DAG augmentation

The new analysis replaces the application DAG `D` with an augmented DAG `D'`
that represents CPU and GPU work separately.

### Asynchronous operator

An asynchronous operator becomes:

```text
async CPU node -> GPU node
```

Application CPU edges leave the async CPU node, because successor operators may
become schedulable once the launch-side CPU work finishes. GPU-to-GPU edges
mirror application dataflow so downstream kernels still wait for upstream GPU
results. Async CPU nodes have sequential edges across iterations. Their GPU
nodes also have sequential edges, representing FIFO execution on the
operator's CUDA stream.

An operator with no material GPU work uses the same structure with a zero-cost
GPU node.

### Synchronous operator

A synchronous operator becomes:

```text
sync-pre CPU node -> GPU node -> sync-post CPU node
```

Incoming application edges enter sync-pre; outgoing edges leave sync-post.
Together the three nodes represent the start, GPU wait, and completion of one
operator invocation. Its cross-iteration serialization is represented by the
sync-post-to-next-sync-pre relationship.

### Which conditions apply

- Data-dependency edges remain finish-to-start weighted edges.
- Downstream blocking attaches only to nodes representing the start of a
  Holoscan operator: async CPU and sync-pre nodes.
- GPU and sync-post nodes are not independently scheduled Holoscan operators,
  so they do not receive downstream-blocking conditions.
- GPU dependencies and per-stream sequential constraints make otherwise
  implicit CUDA ordering explicit.

The GPU nodes have WCETs determined by their fixed SM allocations. Green
Contexts provide the spatial isolation needed to interpret those profiled
times without unmodeled interference from operators in other partitions.

## 5. Why the GPU-aware extension preserves correctness

The proof is a structural reduction to the 2025 setting.

1. Every node in `D'` still represents an execution segment with a bounded
   execution time.
2. Every edge still represents either a finish-to-start constraint weighted by
   its source execution time or a zero-cost start-to-start constraint.
3. Within each iteration, every augmented node still reaches the augmented
   sink.
4. Sequential edges still serialize repeated invocations of each logical
   operator or stream.
5. Downstream edges still move against dataflow by one iteration, preserving
   the bounded-common-ancestor argument used for TG-DFS termination.

Therefore the augmented trace graph retains the start-time-as-longest-path
property. The original termination, soundness, and tightness arguments apply to
the richer graph. CPU and GPU work overlap only when there is no causal path
requiring them to serialize; parallel work is not incorrectly added to one
path.

For a **fixed** GPU partition, TG-DFS computes the exact WCRT under these
assumptions. Algorithm 1 and the MILP solve a separate optimization problem:
they choose which fixed partition to analyze. Algorithm 1 is heuristic over the
partition space, whereas the current MILP proves the optimum of the discrete
profiled partition problem. Neither changes the correctness of TG-DFS for an
individual candidate partition.

The MILP encoding is independently validated rather than trusted only because
it improves upon Algorithm 1. `verify_milp_correctness.py` re-evaluates the
MILP witness with raw expressions and a separate parser, enumerates the entire
131,115,985-partition case-study space, and compares randomized reduced MILPs
with brute force. This distinguishes three possible failure modes: an invalid
returned witness, an expression-compilation error, and an incorrect optimum
claim.

## 6. Implementation map and invariants

The important entry points in `response_time_analysis.py` are:

| Symbol | Role |
|---|---|
| `read_dot_file()` | Loads the augmented application DAG. |
| `logical_depth_dict()` | Computes minimum iteration offsets to the source. |
| `SuccessorChecker` | Answers termination-reachability queries. |
| `dfs_post_order()` | Performs the finite implicit trace-graph traversal. |
| `run_algorithm()` | Builds node metadata and returns path expressions. |
| `compute_max()` | Evaluates expressions for a WCET assignment. |
| `extract_wcet()` | Reads integer WCETs from the DOT input. |

### Logical depth is weighted reachability, not ordinary graph depth

`logical_depth_dict()` constructs a finite reachability graph encoding how
trace-graph movement changes iteration:

- Data-dependency movement has weight 0.
- Sequential-execution movement has weight 1.
- Downstream movement reverses CPU dataflow and has weight 1.
- For a synchronous operator, sync-post-to-sync-pre movement has weight 1.

A shortest-path calculation gives the minimum iteration offset at which each
node can reach the target source. Ordinary edge count is incorrect: combining
data-dependency and downstream movement can reach a node in fewer iterations
than the number of original DAG edges suggests.

### Termination is path-specific

Termination cannot be based only on a node's nominal level. The implementation
also checks whether the node, a transitive successor, or an earlier iteration
of the same logical operator has the required path to the source. In
particular, a sync-post node must check both:

- whether its paired sync-pre is exactly at `iteration - 1`; and
- whether that sync-pre has a strictly lower minimum level.

The second case handles autoconcurrency paths that reach an earlier version of
the sync-pre. Omitting it causes an off-by-one traversal and can count one extra
edge. The same principle explains the specialized GPU termination check for a
GPU node belonging to a synchronous operator.

Do not replace these checks with an unweighted DAG depth or a single global
iteration cutoff. Such shortcuts may still terminate quickly but need not
preserve exactness.

### Other maintained assumptions

- Queue size is fixed to 1 in the GPU-aware artifact.
- The input must have one source and one sink.
- WCETs are integer microseconds in the DOT and evaluation code.
- GPU profile values are truncated to integer microseconds before evaluation.
- Unprofiled GPU nodes used as structural placeholders have zero cost.
- The sink is represented synchronously so completion includes its GPU work.

## 7. Reproduction and regression checks

From `rtss2026artifact`:

```bash
python3 table_v/run_partition_comparison.py \
  --output results/partition_comparison
```

This regenerates the exact MILP result, Algorithm 1 comparisons, sampled
partition statistics, the serialized 2025 graph, and the 2025-versus-GPU-aware
bound comparison.

Independently verify the MILP with:

```bash
python3 table_v/verify_milp_correctness.py \
  --output results/milp_verification.json
```

The full enumerator uses the MILP value only as a challenge threshold. It
visits every feasible partition and reports an error if any partition is
strictly lower. A separately evaluated MILP partition supplies the matching
witness, so the two facts together prove optimality for the finite profiled
instance.

Run the focused regression suite with:

```bash
python3 -m unittest discover -s table_v -p "test_*.py"
```

The tests in [`test_partition_comparison.py`](test_partition_comparison.py)
check the lockstep CPU inputs, exact partition result, low-depth Algorithm 1
result, serialized 2025 construction, source-to-sink serialization behavior,
and partition sampler. Current headline sanity checks are:

- GPU-aware optimum: 19.498 ms at `[2, 2, 2, 22, 66, 38, 10]`.
- Serialized, full-GPU 2025 baseline: 32.083 ms.
- Practical depth-5 Algorithm 1 result: 19.819 ms after 47 unique evaluations.

Runtime values are machine-dependent; WCRTs, partitions, expression counts,
and unique-evaluation counts are deterministic for the checked-in inputs.

## 8. Review checklist for future changes

Before accepting a modification to the analysis, verify:

1. Which real scheduling or CUDA constraint does each new edge represent?
2. Is the edge finish-to-start and WCET-weighted, or start-to-start and zero
   cost?
3. Does it remain within an iteration or change the iteration index?
4. Does the node represent a Holoscan operator start and therefore receive
   downstream blocking?
5. Does every node still reach the sink within an iteration?
6. Can downstream and sequential edges still reach a shared source ancestor in
   boundedly many iterations?
7. Does the weighted logical-depth map represent the new transition?
8. Are both equal-level and strictly-lower-level termination cases handled?
9. Does `compute_max()` count or cancel the terminal expression term correctly?
10. Do the focused regression tests and the single-command pipeline reproduce
    the deterministic headline values?

If any answer is unclear, reason first in terms of the infinite trace graph and
its longest paths. The finite TG-DFS implementation should be treated as an
optimized representation of that model, not as the definition of the model
itself.
