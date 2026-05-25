"""Tests for generated-code static analysis helpers."""

import unittest

from src.utils.code_analysis_utils import (
    analyze_code_arithmetic,
    analyze_response_arithmetic,
    extract_code_text,
)


class CodeAnalysisUtilsTests(unittest.TestCase):
    """Validate arithmetic operator and factor-size metrics."""

    def test_extract_code_text_prefers_tagged_code_blocks(self) -> None:
        """Extract tagged code without surrounding response text."""
        response = "before <code>\nprint(1 + 2)\n</code> after"

        self.assertEqual(extract_code_text(response), "print(1 + 2)")

    def test_extract_code_text_ignores_prior_generic_fences_with_tag(self) -> None:
        """Prefer tagged code over earlier generic markdown fences."""
        response = "```table text```\n<code>\nprint(1 + 2)\n</code>"

        self.assertEqual(extract_code_text(response), "print(1 + 2)")

    def test_counts_ast_arithmetic_operators(self) -> None:
        """Count explicit AST arithmetic operators by symbol."""
        code = "p = (a * b + c * d) / (e - f)\nprint(round(p, 3))"

        metrics = analyze_code_arithmetic(code)

        self.assertIsNone(metrics.parse_error)
        self.assertEqual(metrics.arithmetic_operator_count, 5)
        self.assertEqual(metrics.operator_counts, {"*": 2, "+": 1, "/": 1, "-": 1})
        self.assertEqual(metrics.scalar_operation_count, 5)
        self.assertEqual(metrics.scalar_additions, 1)
        self.assertEqual(metrics.scalar_subtractions, 1)
        self.assertEqual(metrics.scalar_multiplications, 2)
        self.assertEqual(metrics.scalar_divisions, 1)
        self.assertFalse(metrics.uses_vector_operations)
        self.assertEqual(metrics.largest_factor_size, 2)

    def test_counts_augmented_assignment_and_largest_product_chain(self) -> None:
        """Count augmented assignment and detect the largest product chain."""
        code = "x = a * b * c\nx *= d * e"

        metrics = analyze_code_arithmetic(code)

        self.assertEqual(metrics.arithmetic_operator_count, 4)
        self.assertEqual(metrics.operator_counts, {"*": 4})
        self.assertEqual(metrics.scalar_multiplications, 4)
        self.assertEqual(metrics.vector_operation_count, 0)
        self.assertEqual(metrics.largest_factor_size, 3)

    def test_signals_vectorized_binary_operations(self) -> None:
        """Do not fold vectorized array operators into scalar counts."""
        code = "\n".join(
            [
                "import numpy as np",
                "weights = np.array([0.2, 0.8])",
                "values = np.array([0.3, 0.7])",
                "weighted = weights * values",
                "p = weighted.sum() / 2",
            ]
        )

        metrics = analyze_code_arithmetic(code)

        self.assertEqual(metrics.operator_counts, {"*": 1, "/": 1})
        self.assertEqual(metrics.scalar_multiplications, 0)
        self.assertEqual(metrics.scalar_divisions, 1)
        self.assertTrue(metrics.uses_vector_operations)
        self.assertEqual(metrics.vector_operation_count, 2)
        self.assertEqual(metrics.vector_operator_counts, {"*": 1, "sum": 1})

    def test_signals_vectorized_arithmetic_calls(self) -> None:
        """Detect vectorized arithmetic hidden behind numpy calls."""
        code = "\n".join(
            [
                "from numpy import array, divide, sum",
                "values = array([0.1, 0.2, 0.7])",
                "total = sum(values)",
                "normalized = divide(values, total)",
            ]
        )

        metrics = analyze_code_arithmetic(code)

        self.assertEqual(metrics.scalar_operation_count, 0)
        self.assertTrue(metrics.uses_vector_operations)
        self.assertEqual(metrics.vector_operation_count, 2)
        self.assertEqual(
            metrics.vector_operator_counts,
            {"sum": 1, "divide": 1},
        )

    def test_analyze_response_handles_syntax_errors(self) -> None:
        """Return a parse error instead of raising on invalid generated code."""
        metrics = analyze_response_arithmetic("<code>p = (1 +</code>")

        self.assertEqual(metrics.arithmetic_operator_count, 0)
        self.assertEqual(metrics.scalar_operation_count, 0)
        self.assertFalse(metrics.uses_vector_operations)
        self.assertEqual(metrics.largest_factor_size, 0)
        self.assertIsNotNone(metrics.parse_error)


if __name__ == "__main__":
    unittest.main()
