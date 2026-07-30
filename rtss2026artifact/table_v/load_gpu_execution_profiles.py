"""Load the GPU operator set and execution-time functions used by Algorithm 1."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Operator:
    """One partitioned GPU operator and its SM-count-to-WCET mapping."""

    name: str
    execution_times: dict[int, float]

    def get_execution_time(self, resources: int) -> float:
        return self.execution_times.get(resources, float("inf"))


def gpu_operator_names(application_graph: Path) -> list[str]:
    """Return GPU node names, in graph order, from the raw DOT application graph."""
    names = []
    for line in application_graph.read_text().splitlines():
        if "type=GPU" not in line:
            continue
        match = re.match(r'^\s*"?([\w-]+)"?\s*\[', line)
        if match:
            names.append(match.group(1))
    if not names:
        raise ValueError(f"No GPU nodes found in {application_graph}")
    return names


def load_gpu_execution_profiles(profile_csv: Path,
                                graph_gpu_operators: list[str]) -> list[Operator]:
    """Load `Operator,Resources,Time` rows for GPU nodes present in the graph.

    The CSV is authoritative: graph GPU nodes without rows are intentionally not
    partitioned, and CSV rows for nodes absent from the graph are rejected.
    """
    profiles: dict[str, dict[int, float]] = {}
    with profile_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Operator", "Resources", "Time"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Profile CSV must have Operator,Resources,Time columns")
        for row in reader:
            name = row["Operator"].strip()
            if name not in graph_gpu_operators:
                raise ValueError(f"Profile contains GPU operator absent from graph: {name}")
            resources = int(row["Resources"])
            if resources <= 0:
                raise ValueError(f"Profile has non-positive SM allocation for {name}")
            profiles.setdefault(name, {})[resources] = float(row["Time"])

    if not profiles:
        raise ValueError("Profile CSV contains no GPU execution-time rows")
    return [Operator(name, profiles[name]) for name in graph_gpu_operators if name in profiles]
