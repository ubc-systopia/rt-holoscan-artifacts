"""Regression checks for the lockstep-maximum partition comparison."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import response_time_analysis
from algorithm_1 import run_partition_search, setup_environment
from baseline_2025 import (
    build_serialized_operators,
    evaluate_valid_serialization_orders,
    run_2025_tg_dfs,
    write_linear_graph,
)
from exact_partition_milp import solve_exact_partition
from run_partition_comparison import (
    COARSE_GRANULARITY,
    PRACTICAL_DEPTH,
    TARGET_GRANULARITY,
    TOTAL_SMS,
    apply_cpu_profile,
    balanced_partition,
    sample_positive_compositions,
)
from verify_milp_correctness import (
    compile_expressions_independently,
    evaluate_rows,
    execution_times_for_allocation,
    randomized_reduced_cross_checks,
    raw_tg_dfs_value,
)


class PartitionComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.graph = cls.root / "inputs" / "application.dot"
        cls.cpu_profile = cls.root / "inputs" / "cpu_execution_profiles.csv"
        cls.operators, cls.expressions, graph_wcets = setup_environment(
            cls.graph, cls.root / "inputs" / "gpu_execution_profiles.csv"
        )
        cls.static_wcets = apply_cpu_profile(
            graph_wcets, cls.cpu_profile, "lockstep_max_ms"
        )

    def test_graph_embeds_lockstep_maximum_cpu_values(self):
        graph_wcets = response_time_analysis.extract_wcet(self.graph)
        with self.cpu_profile.open(newline="") as stream:
            for row in csv.DictReader(stream):
                expected = int(float(row["lockstep_max_ms"]) * 1000)
                self.assertEqual(graph_wcets[row["Node"]], expected)

    def test_milp_lockstep_optimum(self):
        result = solve_exact_partition(
            self.expressions, self.static_wcets, self.operators, TOTAL_SMS
        )
        self.assertEqual(result.response_time_us, 19_498)
        self.assertEqual(result.allocation, [2, 2, 2, 22, 66, 38, 10])
        self.assertEqual(result.mip_gap, 0.0)

        coefficients, static_costs = compile_expressions_independently(
            self.expressions,
            self.static_wcets,
            [operator.name for operator in self.operators],
        )
        independent_value = evaluate_rows(
            coefficients,
            static_costs,
            execution_times_for_allocation(self.operators, result.allocation),
        )
        raw_value = raw_tg_dfs_value(
            self.expressions,
            self.static_wcets,
            self.operators,
            result.allocation,
        )
        self.assertEqual(independent_value, 19_498)
        self.assertEqual(raw_value, 19_498)

    def test_randomized_reduced_milps_match_brute_force(self):
        result = randomized_reduced_cross_checks(trials=5, seed=2026)
        self.assertTrue(result["all_optimum_values_match"])
        self.assertTrue(result["all_returned_partitions_recheck"])

    def test_practical_algorithm_1_result(self):
        statistics = {}
        _, partition, wcrt_ms = run_partition_search(
            operators=self.operators,
            expressions=self.expressions,
            static_wcets=self.static_wcets,
            p_initial=balanced_partition(
                TOTAL_SMS, COARSE_GRANULARITY, len(self.operators)
            ),
            g_start=COARSE_GRANULARITY,
            g_target=TARGET_GRANULARITY,
            depth=PRACTICAL_DEPTH,
            total_sms=TOTAL_SMS,
            csv_filename=None,
            statistics=statistics,
        )
        self.assertEqual(wcrt_ms, 19.819)
        self.assertEqual(partition, [4, 4, 8, 28, 46, 32, 20])
        self.assertEqual(statistics["unique_partitions"], 47)

    def test_serialized_2025_baseline(self):
        operators = build_serialized_operators(
            application_graph=self.graph,
            cpu_profile=self.cpu_profile,
            cpu_column="lockstep_max_ms",
            full_gpu_profile=(
                self.root / "inputs" /
                "gpu_execution_times_2025_baseline.csv"
            ),
            full_gpu_resources=TOTAL_SMS,
        )
        self.assertEqual(
            [operator.name for operator in operators],
            [
                "VideoSource", "FCResize", "FCPlaxCham", "FCAorticSTE",
                "FCBModePers", "AIPlaxCham", "AIAorticSTE",
                "AIBModePers", "PostProcessor", "Visualizer", "Holoviz",
            ],
        )
        self.assertEqual(sum(
            operator.combined_time_us for operator in operators
        ), 32_083)
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "application_2025_serialized.dot"
            write_linear_graph(operators, graph)
            self.assertIn(
                "VideoSource -> Holoviz;", graph.read_text()
            )
            result = run_2025_tg_dfs(
                graph=graph,
                implementation=(
                    self.root.parents[1] / "rtss2025artifact" /
                    "code" / "TG_DFS.py"
                ),
                operators=operators,
            )
        self.assertEqual(result.response_time_us, 32_083)
        self.assertEqual(result.expression_count, 12)
        serialization_range = evaluate_valid_serialization_orders(
            self.graph, result
        )
        self.assertEqual(serialization_range["count"], 810)
        self.assertEqual(
            serialization_range["minimum_response_time_us"], 32_083
        )
        self.assertEqual(
            serialization_range["maximum_response_time_us"], 32_083
        )

    def test_uniform_sampler_returns_valid_partitions(self):
        samples = sample_positive_compositions(
            np.random.default_rng(2026), 1000, 71, 7
        )
        self.assertEqual(samples.shape, (1000, 7))
        self.assertTrue(np.all(samples > 0))
        self.assertTrue(np.all(np.sum(samples, axis=1) == 71))


if __name__ == "__main__":
    unittest.main()
