"""Exact discrete SM-partition search over the TG-DFS response-time model."""

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from compiled_wcrt import CompiledWcrtEvaluator
from load_gpu_execution_profiles import Operator


@dataclass(frozen=True)
class ExactPartitionResult:
    allocation: list[int]
    response_time_us: int
    solve_seconds: float
    binary_variables: int
    constraints: int
    mip_gap: float | None


def solve_exact_partition(expressions: list[str],
                          static_wcets: dict[str, int],
                          operators: list[Operator],
                          total_sms: int) -> ExactPartitionResult:
    """Minimize the TG-DFS WCRT over all profiled allocations."""
    evaluator = CompiledWcrtEvaluator.compile(
        expressions, static_wcets, [operator.name for operator in operators]
    )

    choices = []
    choices_by_operator = []
    for operator_index, operator in enumerate(operators):
        indices = []
        for resources, execution_time in sorted(operator.execution_times.items()):
            if resources <= total_sms:
                indices.append(len(choices))
                choices.append(
                    (operator_index, resources, int(execution_time))
                )
        if not indices:
            raise ValueError(f"No feasible allocation for {operator.name}")
        choices_by_operator.append(indices)

    binary_count = len(choices)
    wcrt_index = binary_count
    variable_count = binary_count + 1
    constraint_count = len(operators) + 1 + evaluator.row_count

    matrix = lil_matrix((constraint_count, variable_count), dtype=float)
    lower = np.full(constraint_count, -np.inf)
    upper = np.full(constraint_count, np.inf)
    row = 0

    for indices in choices_by_operator:
        matrix[row, indices] = 1.0
        lower[row] = 1.0
        upper[row] = 1.0
        row += 1

    for choice_index, (_, resources, _) in enumerate(choices):
        matrix[row, choice_index] = resources
    lower[row] = total_sms
    upper[row] = total_sms
    row += 1

    for multiplicities, static_cost in zip(
            evaluator.multiplicities, evaluator.static_costs):
        for choice_index, (operator_index, _, execution_time) in enumerate(choices):
            coefficient = multiplicities[operator_index]
            if coefficient:
                matrix[row, choice_index] = coefficient * execution_time
        matrix[row, wcrt_index] = -1.0
        upper[row] = -static_cost
        row += 1

    objective = np.zeros(variable_count)
    objective[wcrt_index] = 1.0
    integrality = np.zeros(variable_count)
    integrality[:binary_count] = 1
    variable_lower = np.zeros(variable_count)
    variable_upper = np.ones(variable_count)
    variable_upper[wcrt_index] = np.inf

    started = time.perf_counter()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    solve_seconds = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"MILP did not prove optimality: {result.message}")

    allocation = []
    execution_times = []
    for operator_index, indices in enumerate(choices_by_operator):
        selected = [index for index in indices if result.x[index] > 0.5]
        if len(selected) != 1:
            raise RuntimeError("MILP did not select exactly one allocation")
        _, resources, execution_time = choices[selected[0]]
        allocation.append(resources)
        execution_times.append(execution_time)

    if sum(allocation) != total_sms:
        raise RuntimeError("MILP allocation does not use all SMs")
    verified_wcrt = evaluator.evaluate_times(execution_times)
    if abs(verified_wcrt - result.fun) > 1e-5:
        raise RuntimeError(
            f"MILP objective {result.fun} != TG-DFS value {verified_wcrt}"
        )

    return ExactPartitionResult(
        allocation=allocation,
        response_time_us=verified_wcrt,
        solve_seconds=solve_seconds,
        binary_variables=binary_count,
        constraints=constraint_count,
        mip_gap=getattr(result, "mip_gap", None),
    )
