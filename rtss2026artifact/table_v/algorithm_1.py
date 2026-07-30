"""Algorithm 1 from Section V: coarse-to-fine GPU partition search."""

import csv
import heapq
import time
from pathlib import Path

from compiled_wcrt import CompiledWcrtEvaluator
import response_time_analysis
from load_gpu_execution_profiles import (
    Operator,
    gpu_operator_names,
    load_gpu_execution_profiles,
)


def closest_partitions(partition: list[int], granularity: int,
                       depth: int) -> list[tuple[float, list[int]]]:
    """Return up to `depth` positive, granularity-aligned partitions nearest to p."""
    operator_count = len(partition)
    total_resources = sum(partition)
    if total_resources % granularity:
        raise ValueError("Partition sum must be divisible by granularity")

    # Re-express the allocation in units of `granularity`, preserving positivity.
    total_units = total_resources // granularity - operator_count
    scaled = [value / granularity - 1 for value in partition]
    floors = [int(value // 1) for value in scaled]
    remainder = total_units - sum(floors)
    fractions = ((scaled[i] - floors[i], i) for i in range(operator_count))
    for _, index in sorted(fractions, reverse=True)[:remainder]:
        floors[index] += 1
    initial = tuple(value + 1 for value in floors)
    target = [value / granularity for value in partition]

    def squared_distance(candidate: tuple[int, ...]) -> float:
        return sum((target[i] - candidate[i]) ** 2 for i in range(operator_count))

    queue = [(squared_distance(initial), initial)]
    visited = {initial}
    candidates = []
    while queue and len(candidates) < depth:
        distance, candidate = heapq.heappop(queue)
        candidates.append((distance, [units * granularity for units in candidate]))
        for recipient in range(operator_count):
            for donor in range(operator_count):
                if recipient == donor or candidate[donor] == 1:
                    continue
                neighbor = list(candidate)
                neighbor[recipient] += 1
                neighbor[donor] -= 1
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(queue, (squared_distance(neighbor), neighbor))
    return candidates


def run_partition_search(operators: list[Operator], expressions,
                         static_wcets: dict[str, int], p_initial: list[int],
                         g_start: int, g_target: int, depth: int,
                         total_sms: int, csv_filename: Path | None,
                         statistics: dict | None = None):
    """Evaluate Algorithm 1 and return runtime, best partition, and WCRT in ms."""
    start_time = time.perf_counter()
    best_wcrt = float("inf")
    best_partition = list(p_initial)
    granularity = g_start
    partition_cache = {}
    candidate_evaluations = 0
    refinement_rounds = 0
    evaluator = CompiledWcrtEvaluator.compile(
        expressions,
        static_wcets,
        [operator.name for operator in operators],
    )

    log_stream = None
    log_writer = None
    if csv_filename is not None:
        log_stream = csv_filename.open("w", newline="")
        log_writer = csv.writer(log_stream)
        log_writer.writerow(("Iteration-G", "Distance", "Partition", "WCRT (ms)"))

    while granularity >= g_target:
        refinement_rounds += 1
        improved = False
        for distance, partition in closest_partitions(best_partition, granularity, depth):
            candidate_evaluations += 1
            partition_key = tuple(partition)
            if partition_key in partition_cache:
                wcrt = partition_cache[partition_key]
            else:
                execution_times = [
                    operator.get_execution_time(resources)
                    for operator, resources in zip(operators, partition)
                ]
                if any(value == float("inf") for value in execution_times):
                    wcrt = float("inf")
                else:
                    wcrt = evaluator.evaluate_times(execution_times)
                partition_cache[partition_key] = wcrt
            if log_writer is not None:
                log_writer.writerow(
                    (granularity, distance, partition, wcrt / 1000.0)
                )
            # Algorithm 1 updates only for a strictly lower WCRT.
            if wcrt < best_wcrt:
                best_wcrt = wcrt
                best_partition = partition
                improved = True

        if not improved:
            granularity //= 2
            # The paper's 142-SM case is not divisible by every coarse
            # granularity. Preserve the existing implementation's top-up rule.
            if granularity >= g_target:
                usable_total = (total_sms // granularity) * granularity
                extra = usable_total - sum(best_partition)
                if extra > 0:
                    largest = max(range(len(best_partition)), key=best_partition.__getitem__)
                    best_partition[largest] += extra

    if log_stream is not None:
        log_stream.close()
    if statistics is not None:
        statistics.update({
            "candidate_evaluations": candidate_evaluations,
            "unique_partitions": len(partition_cache),
            "feasible_unique_partitions": sum(
                value != float("inf") for value in partition_cache.values()
            ),
            "refinement_rounds": refinement_rounds,
        })
    return time.perf_counter() - start_time, best_partition, best_wcrt / 1000.0


def setup_environment(dot_file: Path, profile_csv: Path):
    """Create exactly the Algorithm 1 inputs from the two raw input files."""
    graph_operators = gpu_operator_names(dot_file)
    operators = load_gpu_execution_profiles(profile_csv, graph_operators)
    expressions = response_time_analysis.run_algorithm(dot_file)
    static_wcets = response_time_analysis.extract_wcet(dot_file)
    partitioned = {operator.name for operator in operators}
    for name in graph_operators:
        if name not in partitioned:
            static_wcets[name] = 0
    return operators, expressions, static_wcets
