"""Construct and evaluate the serialized, GPU-oblivious 2025 baseline."""

import csv
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

import response_time_analysis


@dataclass(frozen=True)
class SerializedOperator:
    name: str
    cpu_time_us: int
    gpu_time_us: int

    @property
    def combined_time_us(self) -> int:
        return self.cpu_time_us + self.gpu_time_us


@dataclass(frozen=True)
class Baseline2025Result:
    response_time_us: int
    expression_count: int
    operators: list[SerializedOperator]
    expressions: list[str]


def _node_type(graph: nx.DiGraph, node: str) -> str:
    return str(graph.nodes[node].get("type", "")).strip('"')


def _syncpre_name(syncpost: str) -> str:
    return re.sub(r"(-?prime)$", "", syncpost)


def load_cpu_times(profile_csv: Path, column: str) -> dict[str, int]:
    times = {}
    with profile_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or {"Node", column} - set(reader.fieldnames):
            raise ValueError(f"{profile_csv} must contain Node and {column}")
        for row in reader:
            times[row["Node"].strip()] = int(float(row[column]) * 1000)
    return times


def load_full_gpu_times(profile_csv: Path,
                        resources: int) -> dict[str, int]:
    times = {}
    with profile_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Operator", "Resources", "Time"}
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise ValueError(
                f"{profile_csv} must contain Operator, Resources, and Time"
            )
        for row in reader:
            if int(row["Resources"]) != resources:
                continue
            name = row["Operator"].strip()
            if name in times:
                raise ValueError(f"Duplicate {resources}-SM GPU time for {name}")
            times[name] = int(float(row["Time"]))
    if not times:
        raise ValueError(f"No {resources}-SM GPU measurements in {profile_csv}")
    return times


def build_original_operator_dag(application_graph: Path) -> nx.DiGraph:
    """Collapse GPU and sync-post helpers back into original operators."""
    graph = response_time_analysis.read_dot_file(application_graph)
    original_nodes = [
        node for node in graph.nodes
        if _node_type(graph, node) in {"async", "syncpre"}
    ]
    original_set = set(original_nodes)
    collapsed = nx.DiGraph()
    collapsed.add_nodes_from(original_nodes)

    def original_operator(node: str) -> str | None:
        node_type = _node_type(graph, node)
        if node_type in {"async", "syncpre"}:
            return node
        if node_type == "syncpost":
            return _syncpre_name(node)
        return None

    for source, target in graph.edges():
        source_operator = original_operator(source)
        target_operator = original_operator(target)
        if (source_operator in original_set and target_operator in original_set
                and source_operator != target_operator):
            collapsed.add_edge(source_operator, target_operator)

    if not nx.is_directed_acyclic_graph(collapsed):
        raise ValueError("Collapsed application graph is not acyclic")
    return collapsed


def build_serialized_operators(application_graph: Path,
                               cpu_profile: Path,
                               cpu_column: str,
                               full_gpu_profile: Path,
                               full_gpu_resources: int) -> list[SerializedOperator]:
    """Collapse the augmented DAG and serialize one topological operator order."""
    graph = response_time_analysis.read_dot_file(application_graph)
    collapsed = build_original_operator_dag(application_graph)
    original_nodes = list(collapsed.nodes)
    cpu_times = load_cpu_times(cpu_profile, cpu_column)
    gpu_times = load_full_gpu_times(full_gpu_profile, full_gpu_resources)
    order = list(nx.lexicographical_topological_sort(
        collapsed, key=original_nodes.index
    ))

    operators = []
    for name in order:
        if name not in cpu_times:
            raise ValueError(f"Missing CPU time for {name}")
        cpu_time = cpu_times[name]
        if _node_type(graph, name) == "syncpre":
            syncpost = name + "-prime"
            if syncpost not in cpu_times:
                raise ValueError(f"Missing sync-post CPU time for {name}")
            cpu_time += cpu_times[syncpost]

        gpu_name = "GPU-" + name
        if gpu_name not in gpu_times:
            raise ValueError(f"Missing full-GPU time for {gpu_name}")
        operators.append(SerializedOperator(
            name=name,
            cpu_time_us=cpu_time,
            gpu_time_us=gpu_times[gpu_name],
        ))
    return operators


def evaluate_valid_serialization_orders(application_graph: Path,
                                        result: Baseline2025Result) -> dict:
    """Find the 2025-bound range over every valid linear extension of the DAG."""
    position_names = [operator.name for operator in result.operators]
    coefficient_rows = set()
    for expression in result.expressions:
        coefficients = dict.fromkeys(position_names, 0)
        terms = expression.split("+")
        for term in terms[:-1]:
            coefficients[term.split("_", 1)[0]] += 1
        if terms[-1].endswith("!"):
            coefficients[terms[-1].split("_", 1)[0]] -= 1
        coefficient_rows.add(tuple(
            coefficients[name] for name in position_names
        ))
    matrix = np.asarray(list(coefficient_rows), dtype=np.int16)

    operators_by_name = {
        operator.name: operator for operator in result.operators
    }
    operator_dag = build_original_operator_dag(application_graph)
    minimum = None
    maximum = None
    minimum_order = None
    maximum_order = None
    count = 0
    for order in nx.all_topological_sorts(operator_dag):
        times = np.asarray([
            operators_by_name[name].combined_time_us for name in order
        ], dtype=np.int64)
        bound = int(np.max(matrix @ times))
        count += 1
        if minimum is None or bound < minimum:
            minimum = bound
            minimum_order = list(order)
        if maximum is None or bound > maximum:
            maximum = bound
            maximum_order = list(order)
    return {
        "count": count,
        "minimum_response_time_us": minimum,
        "minimum_order": minimum_order,
        "maximum_response_time_us": maximum,
        "maximum_order": maximum_order,
    }


def write_linear_graph(operators: list[SerializedOperator], output: Path):
    """Write the serialized graph consumed by 2025 TG-DFS.

    The direct source-to-sink edge induces a downstream condition from the
    sink in iteration i to the source in iteration i+1. Without it, the linear
    chain can still pipeline operators belonging to different iterations.
    """
    if len(operators) < 2:
        raise ValueError("The serialized baseline requires at least two operators")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["digraph G {"]
    lines.extend(
        f"    {operator.name} [WCET={operator.combined_time_us}];"
        for operator in operators
    )
    lines.extend(
        f"    {source.name} -> {target.name};"
        for source, target in zip(operators, operators[1:])
    )
    lines.append(f"    {operators[0].name} -> {operators[-1].name};")
    lines.append("}")
    output.write_text("\n".join(lines) + "\n")


def run_2025_tg_dfs(graph: Path,
                    implementation: Path,
                    operators: list[SerializedOperator]) -> Baseline2025Result:
    """Load and execute the unmodified analysis shipped in the 2025 artifact."""
    spec = importlib.util.spec_from_file_location(
        "artifact_2025_tg_dfs", implementation
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load 2025 TG-DFS from {implementation}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expressions = module.run_algorithm(str(graph), 1)
    wcets = module.extract_wcet(str(graph))
    response_time = module.compute_max(expressions, wcets)
    return Baseline2025Result(
        response_time_us=response_time,
        expression_count=len(expressions),
        operators=operators,
        expressions=expressions,
    )
