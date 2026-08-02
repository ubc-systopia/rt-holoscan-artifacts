#!/usr/bin/env python3
"""Independently verify the exact partition MILP.

The full-instance check enumerates every positive granularity-2 composition of
142 SMs. It uses an independent TG-DFS-expression parser and never calls the
MILP while enumerating. The randomized checks compare the MILP with a separate
pure-Python exhaustive search on small generated max-of-affine problems.
"""

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

import response_time_analysis
from algorithm_1 import setup_environment
from exact_partition_milp import solve_exact_partition
from load_gpu_execution_profiles import Operator
from run_partition_comparison import apply_cpu_profile


TOTAL_SMS = 142
GRANULARITY = 2


def compile_expressions_independently(expressions: list[str],
                                      static_wcets: dict[str, int],
                                      operator_names: list[str]):
    """Parse TG-DFS strings without using CompiledWcrtEvaluator."""
    partitioned = set(operator_names)
    rows = {}
    for expression in expressions:
        terms = expression.split("+")
        counted_terms = terms[:-1] if terms[-1].endswith("!") else terms
        multiplicities = dict.fromkeys(operator_names, 0)
        static_cost = 0
        for term in counted_terms:
            name = term.split("_", 1)[0]
            if name in partitioned:
                multiplicities[name] += 1
            elif name in static_wcets:
                static_cost += static_wcets[name]
            else:
                raise KeyError(f"Expression references unknown node {name}")
        coefficients = tuple(multiplicities[name] for name in operator_names)
        rows[coefficients] = max(static_cost, rows.get(coefficients, -1))
    if not rows:
        raise ValueError("No TG-DFS expressions to verify")
    return (
        np.asarray(list(rows), dtype=np.int64),
        np.asarray(list(rows.values()), dtype=np.int64),
    )


def evaluate_rows(coefficients: np.ndarray, static_costs: np.ndarray,
                  execution_times: list[int]) -> int:
    times = np.asarray(execution_times, dtype=np.int64)
    return int(np.max(static_costs + coefficients @ times))


def profile_lookup(operators: list[Operator], total_sms: int,
                   granularity: int) -> np.ndarray:
    total_units = total_sms // granularity
    unavailable = np.iinfo(np.int64).max // 1000
    lookup = np.full(
        (len(operators), total_units + 1), unavailable, dtype=np.int64
    )
    for operator_index, operator in enumerate(operators):
        for resources, execution_time in operator.execution_times.items():
            if resources <= total_sms and resources % granularity == 0:
                lookup[operator_index, resources // granularity] = int(
                    execution_time
                )
    return lookup


@njit(parallel=True, cache=True)
def exhaustive_seven_operator_challenge(coefficients, static_costs,
                                        execution_times, total_units,
                                        threshold):
    """Visit every positive seven-part composition and seek value < threshold."""
    local_best = np.full(total_units, threshold, dtype=np.int64)
    local_allocations = np.zeros((total_units, 7), dtype=np.int16)
    local_counts = np.zeros(total_units, dtype=np.int64)

    # The first six parts are enumerated; the seventh is fixed by the sum.
    for u0 in prange(1, total_units - 5):
        best = threshold
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0
        count = 0
        for u1 in range(1, total_units - u0 - 4):
            for u2 in range(1, total_units - u0 - u1 - 3):
                for u3 in range(1, total_units - u0 - u1 - u2 - 2):
                    for u4 in range(1, total_units - u0 - u1 - u2 - u3 - 1):
                        for u5 in range(
                                1, total_units - u0 - u1 - u2 - u3 - u4):
                            u6 = total_units - u0 - u1 - u2 - u3 - u4 - u5
                            count += 1
                            candidate = -1
                            for row in range(coefficients.shape[0]):
                                value = static_costs[row]
                                value += coefficients[row, 0] * execution_times[0, u0]
                                value += coefficients[row, 1] * execution_times[1, u1]
                                value += coefficients[row, 2] * execution_times[2, u2]
                                value += coefficients[row, 3] * execution_times[3, u3]
                                value += coefficients[row, 4] * execution_times[4, u4]
                                value += coefficients[row, 5] * execution_times[5, u5]
                                value += coefficients[row, 6] * execution_times[6, u6]
                                if value > candidate:
                                    candidate = value
                                # This candidate cannot improve the best value
                                # already witnessed, but it has still been visited.
                                if candidate >= best:
                                    break
                            if candidate < best:
                                best = candidate
                                b0, b1, b2, b3 = u0, u1, u2, u3
                                b4, b5, b6 = u4, u5, u6
        local_best[u0] = best
        local_allocations[u0, 0] = b0
        local_allocations[u0, 1] = b1
        local_allocations[u0, 2] = b2
        local_allocations[u0, 3] = b3
        local_allocations[u0, 4] = b4
        local_allocations[u0, 5] = b5
        local_allocations[u0, 6] = b6
        local_counts[u0] = count
    return local_best, local_allocations, local_counts


def execution_times_for_allocation(operators: list[Operator],
                                   allocation: list[int]) -> list[int]:
    return [
        int(operator.get_execution_time(resources))
        for operator, resources in zip(operators, allocation)
    ]


def raw_tg_dfs_value(expressions: list[str], static_wcets: dict[str, int],
                     operators: list[Operator], allocation: list[int]) -> int:
    wcets = static_wcets.copy()
    for operator, execution_time in zip(
            operators, execution_times_for_allocation(operators, allocation)):
        wcets[operator.name] = execution_time
    return response_time_analysis.compute_max(expressions, wcets)


def run_full_instance_verification(graph: Path, gpu_profile: Path,
                                   cpu_profile: Path, cpu_column: str):
    operators, expressions, graph_wcets = setup_environment(graph, gpu_profile)
    static_wcets = apply_cpu_profile(graph_wcets, cpu_profile, cpu_column)
    names = [operator.name for operator in operators]
    coefficients, static_costs = compile_expressions_independently(
        expressions, static_wcets, names
    )

    milp_started = time.perf_counter()
    milp = solve_exact_partition(
        expressions, static_wcets, operators, TOTAL_SMS
    )
    milp_total_seconds = time.perf_counter() - milp_started
    raw_value = raw_tg_dfs_value(
        expressions, static_wcets, operators, milp.allocation
    )
    independent_value = evaluate_rows(
        coefficients, static_costs,
        execution_times_for_allocation(operators, milp.allocation),
    )
    if raw_value != milp.response_time_us or independent_value != raw_value:
        raise AssertionError("MILP witness failed independent WCRT reevaluation")

    lookup = profile_lookup(operators, TOTAL_SMS, GRANULARITY)
    witness_units = np.asarray(milp.allocation, dtype=np.int64) // GRANULARITY
    witness_times = np.asarray([
        lookup[index, units] for index, units in enumerate(witness_units)
    ])
    # Put expressions tight at the witness first. This changes only evaluation
    # order and makes rejection of non-improving partitions faster.
    row_order = np.argsort(
        -(static_costs + coefficients @ witness_times), kind="stable"
    )
    coefficients = np.ascontiguousarray(coefficients[row_order])
    static_costs = np.ascontiguousarray(static_costs[row_order])

    # Compile the Numba kernel on a tiny problem before timing enumeration.
    exhaustive_seven_operator_challenge(
        coefficients, static_costs, lookup, 8, milp.response_time_us
    )
    enumeration_started = time.perf_counter()
    bests, allocations, counts = exhaustive_seven_operator_challenge(
        coefficients, static_costs, lookup,
        TOTAL_SMS // GRANULARITY, milp.response_time_us,
    )
    enumeration_seconds = time.perf_counter() - enumeration_started
    visited = int(np.sum(counts))
    expected = math.comb(TOTAL_SMS // GRANULARITY - 1, len(operators) - 1)
    if visited != expected:
        raise AssertionError(f"Enumerated {visited} partitions, expected {expected}")

    best_index = int(np.argmin(bests))
    lower_value = int(bests[best_index])
    lower_found = lower_value < milp.response_time_us
    lower_allocation = None
    if lower_found:
        lower_allocation = (
            allocations[best_index].astype(np.int64) * GRANULARITY
        ).tolist()
        raise AssertionError(
            f"Exhaustive search found {lower_value} at {lower_allocation}"
        )

    return {
        "operator_order": names,
        "tg_dfs_expression_count": len(expressions),
        "independent_affine_row_count": len(static_costs),
        "milp_partition": milp.allocation,
        "milp_wcrt_us": milp.response_time_us,
        "milp_mip_gap": milp.mip_gap,
        "milp_model_and_solve_seconds": milp_total_seconds,
        "raw_tg_dfs_wcrt_us": raw_value,
        "independent_parser_wcrt_us": independent_value,
        "full_partition_count_expected": expected,
        "full_partition_count_visited": visited,
        "partition_below_milp_found": False,
        "exhaustive_enumeration_seconds": enumeration_seconds,
        "verified_optimal": True,
    }


def positive_compositions(total: int, parts: int):
    for separators in itertools.combinations(range(1, total), parts - 1):
        boundaries = (0,) + separators + (total,)
        yield tuple(
            boundaries[index + 1] - boundaries[index]
            for index in range(parts)
        )


def randomized_reduced_cross_checks(trials: int, seed: int):
    rng = random.Random(seed)
    checked_partitions = 0
    for trial in range(trials):
        operator_count = rng.randint(3, 5)
        total_resources = operator_count + rng.randint(3, 8)
        names = [f"GPU-O{index}" for index in range(operator_count)]
        operators = []
        for name in names:
            base = rng.randint(200, 2000)
            times = {
                resources: max(
                    1, base // resources + rng.randint(0, max(1, base // 20))
                )
                for resources in range(1, total_resources + 1)
            }
            operators.append(Operator(name, times))

        static_wcets = {name: 0 for name in names}
        static_wcets.update({"CPU0": rng.randint(1, 500),
                             "CPU1": rng.randint(1, 500)})
        expressions = []
        for _ in range(rng.randint(4, 12)):
            counted = []
            for name in names:
                counted.extend(
                    [f"{name}_0"] * rng.randint(0, 3)
                )
            counted.extend([f"CPU{rng.randrange(2)}_0"] * rng.randint(0, 2))
            if not counted:
                counted.append(f"{rng.choice(names)}_0")
            rng.shuffle(counted)
            if rng.random() < 0.4:
                counted.append(f"CPU{rng.randrange(2)}_0!")
            expressions.append("+".join(counted))

        milp = solve_exact_partition(
            expressions, static_wcets, operators, total_resources
        )
        exhaustive_best = None
        for allocation in positive_compositions(
                total_resources, operator_count):
            checked_partitions += 1
            value = raw_tg_dfs_value(
                expressions, static_wcets, operators, list(allocation)
            )
            if exhaustive_best is None or value < exhaustive_best:
                exhaustive_best = value
        witness_value = raw_tg_dfs_value(
            expressions, static_wcets, operators, milp.allocation
        )
        if milp.response_time_us != exhaustive_best:
            raise AssertionError(
                f"Trial {trial}: MILP {milp.response_time_us} != "
                f"exhaustive {exhaustive_best}"
            )
        if witness_value != exhaustive_best:
            raise AssertionError(f"Trial {trial}: returned partition is invalid")
    return {
        "seed": seed,
        "trials": trials,
        "partitions_checked": checked_partitions,
        "all_optimum_values_match": True,
        "all_returned_partitions_recheck": True,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/milp_verification.json"),
    )
    parser.add_argument(
        "--graph", type=Path, default=root / "inputs" / "application.dot"
    )
    parser.add_argument(
        "--gpu-profile", type=Path,
        default=root / "inputs" / "gpu_execution_profiles.csv",
    )
    parser.add_argument(
        "--cpu-profile", type=Path,
        default=root / "inputs" / "cpu_execution_profiles.csv",
    )
    parser.add_argument("--cpu-column", default="lockstep_max_ms")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    full = run_full_instance_verification(
        args.graph, args.gpu_profile, args.cpu_profile, args.cpu_column
    )
    randomized = randomized_reduced_cross_checks(
        args.random_trials, args.random_seed
    )
    report = {
        "verification_claim": (
            "MILP witness independently re-evaluated; every full-instance "
            "partition checked; randomized MILPs matched brute force"
        ),
        "full_instance": full,
        "randomized_reduced_instances": randomized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print(
        f"MILP witness: {full['milp_wcrt_us'] / 1000.0:.3f} ms at "
        f"{full['milp_partition']}"
    )
    print(
        f"Raw TG-DFS and independent parser: "
        f"{full['raw_tg_dfs_wcrt_us'] / 1000.0:.3f} ms"
    )
    print(
        f"Exhaustive verification: visited "
        f"{full['full_partition_count_visited']:,} / "
        f"{full['full_partition_count_expected']:,} partitions in "
        f"{full['exhaustive_enumeration_seconds']:.3f} s; no lower value"
    )
    print(
        f"Randomized reduced checks: {randomized['trials']} / "
        f"{randomized['trials']} matched exhaustive search "
        f"({randomized['partitions_checked']:,} partitions)"
    )
    print(f"Verification report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
