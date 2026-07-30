"""Regression checks for the lockstep-maximum partition comparison."""

import csv
import unittest
from pathlib import Path

import numpy as np

import response_time_analysis
from algorithm_1 import run_partition_search, setup_environment
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

    def test_uniform_sampler_returns_valid_partitions(self):
        samples = sample_positive_compositions(
            np.random.default_rng(2026), 1000, 71, 7
        )
        self.assertEqual(samples.shape, (1000, 7))
        self.assertTrue(np.all(samples > 0))
        self.assertTrue(np.all(np.sum(samples, axis=1) == 71))


if __name__ == "__main__":
    unittest.main()
