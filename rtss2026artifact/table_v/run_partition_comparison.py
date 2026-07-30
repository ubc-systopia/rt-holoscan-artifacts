#!/usr/bin/env python3
"""Reproduce the paper's MILP-versus-Algorithm-1 comparison in one command."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from algorithm_1 import run_partition_search, setup_environment
from compiled_wcrt import CompiledWcrtEvaluator
from exact_partition_milp import solve_exact_partition


TOTAL_SMS = 142
TARGET_GRANULARITY = 2
PRACTICAL_DEPTH = 5
HIGH_DEPTH = 15_000
COARSE_GRANULARITY = 8
DIRECT_GRANULARITY = 2


def balanced_partition(total_sms: int, granularity: int,
                       operator_count: int) -> list[int]:
    """Evenly distribute the SMs usable at the requested granularity."""
    units, remainder = divmod(total_sms // granularity, operator_count)
    return [
        (units + (index < remainder)) * granularity
        for index in range(operator_count)
    ]


def apply_cpu_profile(static_wcets: dict[str, int], profile_csv: Path,
                      column: str) -> dict[str, int]:
    """Replace graph CPU WCETs with one measured CPU-profile column."""
    result = static_wcets.copy()
    with profile_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or {"Node", column} - set(reader.fieldnames):
            raise ValueError(f"{profile_csv} must contain Node and {column}")
        for row in reader:
            node = row["Node"].strip()
            if node not in result:
                raise ValueError(f"CPU profile node is absent from graph: {node}")
            result[node] = int(float(row[column]) * 1000)
    return result


def sample_positive_compositions(rng: np.random.Generator, count: int,
                                 total_units: int,
                                 operator_count: int) -> np.ndarray:
    """Uniformly sample ordered positive compositions using random separators."""
    separator_count = operator_count - 1
    separators = np.empty((count, separator_count), dtype=np.int16)
    filled = 0
    while filled < count:
        remaining = count - filled
        draw_count = max(1024, int(remaining * 1.4))
        draws = rng.integers(
            1, total_units, size=(draw_count, separator_count), dtype=np.int16
        )
        draws.sort(axis=1)
        valid = np.all(np.diff(draws, axis=1) != 0, axis=1)
        accepted = draws[valid][:remaining]
        separators[filled:filled + len(accepted)] = accepted
        filled += len(accepted)

    boundaries = np.empty((count, operator_count + 1), dtype=np.int16)
    boundaries[:, 0] = 0
    boundaries[:, 1:-1] = separators
    boundaries[:, -1] = total_units
    return np.diff(boundaries, axis=1)


def estimate_partition_distribution(operators, expressions, static_wcets,
                                    threshold_us: int, sample_count: int,
                                    seed: int, evaluation_budget: int):
    """Estimate the rank of a result among all granularity-2 partitions."""
    evaluator = CompiledWcrtEvaluator.compile(
        expressions, static_wcets, [operator.name for operator in operators]
    )
    total_units = TOTAL_SMS // TARGET_GRANULARITY
    full_space = math.comb(total_units - 1, len(operators) - 1)
    rng = np.random.default_rng(seed)
    sampled_wcrts = np.empty(sample_count, dtype=np.int64)

    lookups = []
    for operator in operators:
        lookup = np.full(TOTAL_SMS + 1, -1, dtype=np.int64)
        for resources, execution_time in operator.execution_times.items():
            if resources <= TOTAL_SMS:
                lookup[resources] = int(execution_time)
        lookups.append(lookup)

    batch_size = 100_000
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        units = sample_positive_compositions(
            rng, stop - start, total_units, len(operators)
        )
        partitions = units * TARGET_GRANULARITY
        execution_times = np.column_stack([
            lookup[partitions[:, index]]
            for index, lookup in enumerate(lookups)
        ])
        if np.any(execution_times < 0):
            raise ValueError("GPU profile does not cover every sampled allocation")
        path_values = (
            evaluator.static_costs[:, None]
            + evaluator.multiplicities @ execution_times.T
        )
        sampled_wcrts[start:stop] = np.max(path_values, axis=0)

    equal_or_better = int(np.count_nonzero(sampled_wcrts <= threshold_us))
    estimated_fraction = equal_or_better / sample_count
    random_success = 1.0 - (1.0 - estimated_fraction) ** evaluation_budget
    standard_error = math.sqrt(
        estimated_fraction * (1.0 - estimated_fraction) / sample_count
    )
    return {
        "seed": seed,
        "sample_count": sample_count,
        "full_partition_count": full_space,
        "threshold_wcrt_ms": threshold_us / 1000.0,
        "sample_equal_or_better": equal_or_better,
        "estimated_equal_or_better_fraction": estimated_fraction,
        "estimated_best_percent": estimated_fraction * 100.0,
        "estimated_fraction_standard_error": standard_error,
        "sample_median_wcrt_ms": float(np.median(sampled_wcrts)) / 1000.0,
        "random_search_budget": evaluation_budget,
        "estimated_random_success_probability": random_success,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/partition_comparison"))
    parser.add_argument("--graph", type=Path,
                        default=root / "inputs" / "application.dot")
    parser.add_argument("--gpu-profile", type=Path,
                        default=root / "inputs" / "gpu_execution_profiles.csv")
    parser.add_argument("--cpu-profile", type=Path,
                        default=root / "inputs" / "cpu_execution_profiles.csv")
    parser.add_argument("--cpu-column", default="lockstep_max_ms")
    parser.add_argument("--sample-count", type=int, default=1_000_000)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--write-search-logs", action="store_true")
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    setup_started = time.perf_counter()
    operators, expressions, graph_wcets = setup_environment(
        args.graph, args.gpu_profile
    )
    static_wcets = apply_cpu_profile(
        graph_wcets, args.cpu_profile, args.cpu_column
    )
    setup_seconds = time.perf_counter() - setup_started
    operator_names = [operator.name for operator in operators]

    milp_started = time.perf_counter()
    exact = solve_exact_partition(
        expressions, static_wcets, operators, TOTAL_SMS
    )
    milp_model_and_solve_seconds = time.perf_counter() - milp_started

    configurations = (
        ("practical_coarse_to_fine", PRACTICAL_DEPTH, COARSE_GRANULARITY),
        ("practical_direct", PRACTICAL_DEPTH, DIRECT_GRANULARITY),
        ("high_depth_coarse_to_fine", HIGH_DEPTH, COARSE_GRANULARITY),
        ("high_depth_direct", HIGH_DEPTH, DIRECT_GRANULARITY),
    )
    searches = []
    for name, depth, g_start in configurations:
        statistics = {}
        log_path = (
            args.output / f"{name}.csv" if args.write_search_logs else None
        )
        runtime, partition, wcrt_ms = run_partition_search(
            operators=operators,
            expressions=expressions,
            static_wcets=static_wcets,
            p_initial=balanced_partition(
                TOTAL_SMS, g_start, len(operators)
            ),
            g_start=g_start,
            g_target=TARGET_GRANULARITY,
            depth=depth,
            total_sms=TOTAL_SMS,
            csv_filename=log_path,
            statistics=statistics,
        )
        searches.append({
            "name": name,
            "depth": depth,
            "g_start": g_start,
            "runtime_seconds": runtime,
            "partition": partition,
            "wcrt_ms": wcrt_ms,
            "optimum_gap_percent": (
                (wcrt_ms - exact.response_time_us / 1000.0)
                / (exact.response_time_us / 1000.0) * 100.0
            ),
            **statistics,
        })

    practical = searches[0]
    sampling_started = time.perf_counter()
    distribution = estimate_partition_distribution(
        operators=operators,
        expressions=expressions,
        static_wcets=static_wcets,
        threshold_us=round(practical["wcrt_ms"] * 1000),
        sample_count=args.sample_count,
        seed=args.sample_seed,
        evaluation_budget=practical["unique_partitions"],
    )
    distribution["sampling_seconds"] = time.perf_counter() - sampling_started
    distribution["practical_below_median_percent"] = (
        (distribution["sample_median_wcrt_ms"] - practical["wcrt_ms"])
        / distribution["sample_median_wcrt_ms"] * 100.0
    )

    result = {
        "input": {
            "graph": str(args.graph.resolve()),
            "gpu_profile": str(args.gpu_profile.resolve()),
            "cpu_profile": str(args.cpu_profile.resolve()),
            "cpu_column": args.cpu_column,
            "total_sms": TOTAL_SMS,
            "target_granularity": TARGET_GRANULARITY,
            "operator_order": operator_names,
            "tg_dfs_expressions": len(expressions),
            "setup_seconds": setup_seconds,
        },
        "milp": {
            "proven_optimal": True,
            "partition": exact.allocation,
            "wcrt_ms": exact.response_time_us / 1000.0,
            "solver_seconds": exact.solve_seconds,
            "model_and_solve_seconds": milp_model_and_solve_seconds,
            "binary_variables": exact.binary_variables,
            "constraints": exact.constraints,
            "mip_gap": exact.mip_gap,
        },
        "algorithm_1": searches,
        "sampled_partition_distribution": distribution,
    }

    with (args.output / "partition_comparison.json").open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with (args.output / "partition_comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "method", "depth", "g_start", "wcrt_ms", "runtime_seconds",
            "unique_partitions", "optimum_gap_percent", "partition",
        ))
        writer.writerow((
            "MILP", "", "", result["milp"]["wcrt_ms"],
            result["milp"]["model_and_solve_seconds"], "",
            0.0, " ".join(map(str, exact.allocation)),
        ))
        for search in searches:
            writer.writerow((
                search["name"], search["depth"], search["g_start"],
                search["wcrt_ms"], search["runtime_seconds"],
                search["unique_partitions"], search["optimum_gap_percent"],
                " ".join(map(str, search["partition"])),
            ))

    print(f"CPU profile: {args.cpu_column}")
    print(f"Operator order: {operator_names}")
    print(
        f"MILP optimum: {result['milp']['wcrt_ms']:.3f} ms at "
        f"{exact.allocation} "
        f"({milp_model_and_solve_seconds:.3f} s model + solve)"
    )
    for search in searches:
        print(
            f"Algorithm 1 {search['name']}: {search['wcrt_ms']:.3f} ms at "
            f"{search['partition']}; {search['unique_partitions']} unique; "
            f"{search['runtime_seconds']:.3f} s; "
            f"{search['optimum_gap_percent']:.2f}% above optimum"
        )
    print(
        f"Practical result estimated in best "
        f"{distribution['estimated_best_percent']:.4f}% of "
        f"{distribution['full_partition_count']:,} partitions; "
        f"{distribution['practical_below_median_percent']:.2f}% below sampled "
        f"median; random-{distribution['random_search_budget']} success "
        f"{distribution['estimated_random_success_probability'] * 100:.2f}%"
    )
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
