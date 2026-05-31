"""Tests for scalar operation counts in query complexity analysis."""

import unittest

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.queries.complexity import compute_query_complexity
from src.queries.complexity.elimination import simulate_variable_elimination


def _chain_bn() -> DiscreteBayesianNetwork:
    """Build a small binary chain A -> B -> C."""
    model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
    state_names = {
        "A": ["s0", "s1"],
        "B": ["s0", "s1"],
        "C": ["s0", "s1"],
    }

    cpd_a = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.4], [0.6]],
        state_names={"A": state_names["A"]},
    )
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.7, 0.2], [0.3, 0.8]],
        evidence=["A"],
        evidence_card=[2],
        state_names={k: state_names[k] for k in ("A", "B")},
    )
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[[0.9, 0.1], [0.1, 0.9]],
        evidence=["B"],
        evidence_card=[2],
        state_names={k: state_names[k] for k in ("B", "C")},
    )

    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


class QueryComplexityOpsTests(unittest.TestCase):
    """Validate scalar addition and multiplication accounting."""

    def test_simulate_variable_elimination_counts_scalar_ops(self) -> None:
        """Count scalar operations for a known one-variable elimination order."""
        result = simulate_variable_elimination(
            _chain_bn(),
            elim_order=["B"],
            keep_vars={"C"},
            evidence_vars_set={"A"},
        )

        self.assertEqual(result["factor_sizes"], [4])
        self.assertEqual(result["scalar_multiplications_by_step"], [4])
        self.assertEqual(result["scalar_additions_by_step"], [2])
        self.assertEqual(result["final_join_multiplications"], 2)
        self.assertEqual(result["normalization_additions"], 1)
        self.assertEqual(result["scalar_multiplications"], 6)
        self.assertEqual(result["scalar_additions"], 3)

    def test_compute_query_complexity_exposes_scalar_ops(self) -> None:
        """Expose total scalar operation counts on ComplexityMetrics."""
        metrics = compute_query_complexity(_chain_bn(), ["C"], ["A"])

        self.assertEqual(metrics.scalar_multiplications, 6)
        self.assertEqual(metrics.scalar_additions, 3)
        self.assertEqual(metrics.normalization_additions, 1)


if __name__ == "__main__":
    unittest.main()
