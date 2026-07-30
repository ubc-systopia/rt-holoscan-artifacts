"""Compile TG-DFS strings for fast repeated fixed-partition evaluation."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


def expression_multiplicities(expression: str) -> Counter[str]:
    """Return the counted WCET multiplicity of each node in an expression."""
    terms = expression.split("+")
    counted_terms = terms[:-1]
    if not terms[-1].endswith("!"):
        counted_terms.append(terms[-1])
    return Counter(term.split("_", 1)[0] for term in counted_terms)


@dataclass(frozen=True)
class CompiledWcrtEvaluator:
    """Vectorized max-of-affine-functions representation of TG-DFS."""

    operator_names: tuple[str, ...]
    multiplicities: np.ndarray
    static_costs: np.ndarray
    expression_count: int

    @property
    def row_count(self) -> int:
        return len(self.static_costs)

    @classmethod
    def compile(cls, expressions: list[str], static_wcets: dict[str, int],
                operator_names: list[str]) -> "CompiledWcrtEvaluator":
        partitioned = set(operator_names)
        known = set(static_wcets) | partitioned
        compiled: dict[tuple[int, ...], int] = {}
        for expression in expressions:
            counts = expression_multiplicities(expression)
            unknown = set(counts) - known
            if unknown:
                raise KeyError(
                    f"Expression references nodes without WCETs: {sorted(unknown)}"
                )
            coefficients = tuple(counts[name] for name in operator_names)
            static_cost = sum(
                count * static_wcets[name]
                for name, count in counts.items()
                if name not in partitioned
            )
            previous = compiled.get(coefficients)
            if previous is None or static_cost > previous:
                compiled[coefficients] = static_cost
        if not compiled:
            raise ValueError("Cannot compile an empty TG-DFS expression set")
        return cls(
            operator_names=tuple(operator_names),
            multiplicities=np.asarray(list(compiled), dtype=np.int64),
            static_costs=np.asarray(list(compiled.values()), dtype=np.int64),
            expression_count=len(expressions),
        )

    def evaluate_times(self, execution_times) -> int:
        times = np.asarray(execution_times, dtype=np.int64)
        if times.shape != (len(self.operator_names),):
            raise ValueError(
                f"Expected {len(self.operator_names)} execution times, got {times.shape}"
            )
        return int(np.max(self.static_costs + self.multiplicities @ times))
