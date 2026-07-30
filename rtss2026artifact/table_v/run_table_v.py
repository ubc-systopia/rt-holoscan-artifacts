#!/usr/bin/env python3
"""Reproduce Table V's partition-search configuration sweep."""

import argparse
import csv
from pathlib import Path

from algorithm_1 import run_partition_search, setup_environment


TOTAL_SMS = 142
TARGET_GRANULARITY = 2
DEPTHS = (100, 1000, 10000)
START_GRANULARITIES = (8, 4, 2)


def balanced_partition(total_sms, granularity, operator_count):
    """Evenly distribute usable SMs, assigning the remainder in operator order."""
    units, remainder = divmod(total_sms // granularity, operator_count)
    return [(units + (index < remainder)) * granularity for index in range(operator_count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/table_v"))
    parser.add_argument("--graph", type=Path,
                        help="Raw application DOT file (default: inputs/application.dot)")
    parser.add_argument("--profile", type=Path,
                        help="Raw GPU execution-time profile CSV (default: inputs/gpu_execution_profiles.csv)")
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    parser.add_argument("--start-granularities", type=int, nargs="+", default=START_GRANULARITIES)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    graph = args.graph or root / "inputs" / "application.dot"
    profile = args.profile or root / "inputs" / "gpu_execution_profiles.csv"
    if not graph.is_file() or not profile.is_file():
        raise FileNotFoundError("--graph and --profile must name existing raw input files")
    args.output.mkdir(parents=True, exist_ok=True)
    operators, expressions, static_wcets = setup_environment(
        dot_file=graph,
        profile_csv=profile,
    )

    rows = []
    for depth in args.depths:
        for g_start in args.start_granularities:
            initial = balanced_partition(TOTAL_SMS, g_start, len(operators))
            runtime, partition, wcrt = run_partition_search(
                operators=operators,
                expressions=expressions,
                static_wcets=static_wcets,
                p_initial=initial,
                g_start=g_start,
                g_target=TARGET_GRANULARITY,
                depth=depth,
                total_sms=TOTAL_SMS,
                csv_filename=args.output / f"search_d{depth}_g{g_start}.csv",
            )
            rows.append((depth, g_start, wcrt, runtime, " ".join(map(str, partition))))
            print(f"d={depth:5d}, g_start={g_start}: WCRT={wcrt:.3f} ms, runtime={runtime:.1f} s")

    with (args.output / "table_v.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("depth", "g_start", "wcrt_ms", "runtime_seconds", "best_partition"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
