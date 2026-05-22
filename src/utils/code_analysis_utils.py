"""Static analysis helpers for LLM-generated Python code."""

import ast
import re
from dataclasses import dataclass, field

CODE_TAG_PATTERN = re.compile(r"<code>(.*?)</code>", re.IGNORECASE | re.DOTALL)
PYTHON_FENCE_PATTERN = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_FENCE_PATTERN = re.compile(r"```\s*(.*?)```", re.DOTALL)

BINARY_OPERATOR_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.MatMult: "@",
}
UNARY_OPERATOR_SYMBOLS: dict[type[ast.unaryop], str] = {
    ast.UAdd: "unary+",
    ast.USub: "unary-",
}


@dataclass
class ArithmeticCodeMetrics:
    """
    Static arithmetic metrics for a Python code snippet.

    ``largest_factor_size`` is a heuristic for manual probability solutions: it
    counts operands in the largest explicit multiplication chain, e.g.
    ``a * b * c`` has size 3. It does not infer Bayesian factor cardinalities or
    expand loops, helper functions, or library calls.
    """

    arithmetic_operator_count: int
    operator_counts: dict[str, int] = field(default_factory=dict)
    largest_factor_size: int = 0
    parse_error: str | None = None


def extract_code_text(response: str) -> str:
    """Extract code blocks from an LLM response, or return the input as-is."""
    if not isinstance(response, str):
        return ""

    tagged_blocks = _extract_matches(CODE_TAG_PATTERN, response)
    if tagged_blocks:
        return "\n\n".join(tagged_blocks)

    python_fenced_blocks = _extract_matches(PYTHON_FENCE_PATTERN, response)
    if python_fenced_blocks:
        return "\n\n".join(python_fenced_blocks)

    generic_fenced_blocks = _extract_matches(GENERIC_FENCE_PATTERN, response)
    if generic_fenced_blocks:
        return "\n\n".join(generic_fenced_blocks)

    return response


def analyze_code_arithmetic(code: str) -> ArithmeticCodeMetrics:
    """
    Count explicit arithmetic operators and product-term size in Python code.

    This uses Python's AST instead of substring matching, so operators inside
    comments and strings are ignored. Function-call work such as ``sum(values)``
    is not expanded because the number of operations is runtime-dependent.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ArithmeticCodeMetrics(
            arithmetic_operator_count=0,
            operator_counts={},
            largest_factor_size=0,
            parse_error=f"{exc.__class__.__name__}: {exc.msg}",
        )

    visitor = _ArithmeticMetricsVisitor()
    visitor.visit(tree)
    operator_counts = {
        operator: count
        for operator, count in visitor.operator_counts.items()
        if count > 0
    }

    return ArithmeticCodeMetrics(
        arithmetic_operator_count=sum(operator_counts.values()),
        operator_counts=operator_counts,
        largest_factor_size=visitor.largest_factor_size,
        parse_error=None,
    )


def analyze_response_arithmetic(response: str) -> ArithmeticCodeMetrics:
    """Extract generated code from a response and analyze its arithmetic."""
    return analyze_code_arithmetic(extract_code_text(response))


class _ArithmeticMetricsVisitor(ast.NodeVisitor):
    """Collect arithmetic metrics from a parsed Python AST."""

    def __init__(self) -> None:
        self.operator_counts: dict[str, int] = {}
        self.largest_factor_size = 0

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Count binary arithmetic operators and product-chain operands."""
        operator = BINARY_OPERATOR_SYMBOLS.get(type(node.op))
        if operator is not None:
            self._count(operator)

        if isinstance(node.op, ast.Mult):
            self.largest_factor_size = max(
                self.largest_factor_size,
                _multiplication_operand_count(node),
            )

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Count augmented arithmetic assignments such as ``x *= y``."""
        operator = BINARY_OPERATOR_SYMBOLS.get(type(node.op))
        if operator is not None:
            self._count(operator)

        if isinstance(node.op, ast.Mult):
            self.largest_factor_size = max(
                self.largest_factor_size,
                1 + _multiplication_operand_count(node.value),
            )

        self.visit(node.target)
        self.visit(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Count unary plus/minus operators."""
        operator = UNARY_OPERATOR_SYMBOLS.get(type(node.op))
        if operator is not None:
            self._count(operator)

        self.generic_visit(node)

    def _count(self, operator: str) -> None:
        self.operator_counts[operator] = self.operator_counts.get(operator, 0) + 1


def _multiplication_operand_count(node: ast.AST) -> int:
    """Return operand count for an explicit multiplication chain."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _multiplication_operand_count(node.left) + _multiplication_operand_count(
            node.right
        )
    return 1


def _extract_matches(pattern: re.Pattern[str], response: str) -> list[str]:
    """Return non-empty regex matches stripped of surrounding whitespace."""
    return [match.strip() for match in pattern.findall(response) if match.strip()]
