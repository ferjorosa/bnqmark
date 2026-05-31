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
SCALAR_BINARY_OPERATORS: dict[type[ast.operator], str] = {
    ast.Add: "additions",
    ast.Sub: "subtractions",
    ast.Mult: "multiplications",
    ast.Div: "divisions",
    ast.FloorDiv: "divisions",
}
VECTOR_LIBRARY_MODULES = {
    "jax",
    "jax.numpy",
    "jnp",
    "np",
    "numpy",
    "pandas",
    "pd",
    "tensorflow",
    "tf",
    "torch",
}
VECTOR_CONSTRUCTOR_CALLS = {
    "array",
    "asarray",
    "as_tensor",
    "DataFrame",
    "eye",
    "full",
    "linspace",
    "matrix",
    "ones",
    "ones_like",
    "Series",
    "tensor",
    "Tensor",
    "zeros",
    "zeros_like",
}
VECTOR_ARITHMETIC_CALLS = {
    "add",
    "divide",
    "dot",
    "einsum",
    "inner",
    "matmul",
    "multiply",
    "outer",
    "prod",
    "sum",
    "subtract",
}
VECTOR_OUTPUT_ARITHMETIC_CALLS = {
    "add",
    "divide",
    "einsum",
    "matmul",
    "multiply",
    "outer",
    "subtract",
}


@dataclass
class ArithmeticCodeMetrics:
    """
    Static arithmetic metrics for a Python code snippet.

    ``largest_factor_size`` is a heuristic for manual probability solutions: it
    counts operands in the largest explicit multiplication chain, e.g.
    ``a * b * c`` has size 3. It does not infer Bayesian factor cardinalities or
    expand loops, helper functions, or library calls.

    ``scalar_*`` fields count explicit scalar binary operators only. Detected
    vectorized operators and numeric-library calls are signaled separately and
    excluded from the scalar counts because their scalar cost is shape-dependent.
    """

    arithmetic_operator_count: int
    operator_counts: dict[str, int] = field(default_factory=dict)
    scalar_operation_count: int = 0
    scalar_additions: int = 0
    scalar_subtractions: int = 0
    scalar_multiplications: int = 0
    scalar_divisions: int = 0
    vector_operation_count: int = 0
    vector_operator_counts: dict[str, int] = field(default_factory=dict)
    uses_vector_operations: bool = False
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
    Vectorized numeric-library calls are flagged without estimating their
    scalar cost.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ArithmeticCodeMetrics(
            arithmetic_operator_count=0,
            operator_counts={},
            vector_operator_counts={},
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
    scalar_counts = visitor.scalar_counts
    vector_operator_counts = {
        operator: count
        for operator, count in visitor.vector_operator_counts.items()
        if count > 0
    }

    return ArithmeticCodeMetrics(
        arithmetic_operator_count=sum(operator_counts.values()),
        operator_counts=operator_counts,
        scalar_operation_count=sum(scalar_counts.values()),
        scalar_additions=scalar_counts["additions"],
        scalar_subtractions=scalar_counts["subtractions"],
        scalar_multiplications=scalar_counts["multiplications"],
        scalar_divisions=scalar_counts["divisions"],
        vector_operation_count=sum(vector_operator_counts.values()),
        vector_operator_counts=vector_operator_counts,
        uses_vector_operations=bool(vector_operator_counts),
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
        self.scalar_counts = {
            "additions": 0,
            "subtractions": 0,
            "multiplications": 0,
            "divisions": 0,
        }
        self.vector_operator_counts: dict[str, int] = {}
        self.vector_module_aliases = set(VECTOR_LIBRARY_MODULES)
        self.vector_constructor_names = set[str]()
        self.vector_arithmetic_names = set[str]()
        self.vector_output_names = set[str]()
        self.vector_names = set[str]()
        self.largest_factor_size = 0

    def visit_Import(self, node: ast.Import) -> None:
        """Track aliases for common vectorized numeric libraries."""
        for alias in node.names:
            root_name = alias.name.split(".", maxsplit=1)[0]
            if (
                alias.name in VECTOR_LIBRARY_MODULES
                or root_name in VECTOR_LIBRARY_MODULES
            ):
                self.vector_module_aliases.add(alias.asname or root_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track directly imported vector constructors and arithmetic helpers."""
        if node.module is None:
            return

        root_name = node.module.split(".", maxsplit=1)[0]
        if node.module in VECTOR_LIBRARY_MODULES or root_name in VECTOR_LIBRARY_MODULES:
            for alias in node.names:
                imported_name = alias.asname or alias.name
                if alias.name in VECTOR_CONSTRUCTOR_CALLS:
                    self.vector_constructor_names.add(imported_name)
                if alias.name in VECTOR_ARITHMETIC_CALLS:
                    self.vector_arithmetic_names.add(imported_name)
                if alias.name in VECTOR_OUTPUT_ARITHMETIC_CALLS:
                    self.vector_output_names.add(imported_name)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track names that are assigned vector-like values."""
        self.visit(node.value)
        if self._is_vector_expression(node.value):
            for target in node.targets:
                self._record_vector_target(target)
        else:
            for target in node.targets:
                self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments to vector-like values."""
        if node.value is not None:
            self.visit(node.value)
            if self._is_vector_expression(node.value):
                self._record_vector_target(node.target)
            else:
                self.visit(node.target)
        else:
            self.visit(node.target)
        self.visit(node.annotation)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Count binary arithmetic operators and product-chain operands."""
        operator = BINARY_OPERATOR_SYMBOLS.get(type(node.op))
        if operator is not None:
            self._count(operator)
            if self._is_vector_operation(node):
                self._count_vector(operator)
            else:
                scalar_operator = SCALAR_BINARY_OPERATORS.get(type(node.op))
                if scalar_operator is not None:
                    self.scalar_counts[scalar_operator] += 1

        if isinstance(node.op, ast.Mult) and not self._is_vector_operation(node):
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
            if self._is_vector_operation(node):
                self._count_vector(operator)
            else:
                scalar_operator = SCALAR_BINARY_OPERATORS.get(type(node.op))
                if scalar_operator is not None:
                    self.scalar_counts[scalar_operator] += 1

        if isinstance(node.op, ast.Mult) and not self._is_vector_operation(node):
            self.largest_factor_size = max(
                self.largest_factor_size,
                1 + _multiplication_operand_count(node.value),
            )

        self.visit(node.target)
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Signal vectorized arithmetic hidden behind library calls."""
        operator = self._vector_arithmetic_call_name(node)
        if operator is not None:
            self._count_vector(operator)

        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Count unary plus/minus operators."""
        operator = UNARY_OPERATOR_SYMBOLS.get(type(node.op))
        if operator is not None:
            self._count(operator)

        self.generic_visit(node)

    def _count(self, operator: str) -> None:
        self.operator_counts[operator] = self.operator_counts.get(operator, 0) + 1

    def _count_vector(self, operator: str) -> None:
        self.vector_operator_counts[operator] = (
            self.vector_operator_counts.get(operator, 0) + 1
        )

    def _is_vector_operation(self, node: ast.BinOp | ast.AugAssign) -> bool:
        """Return whether an operator acts on a vector-like expression."""
        if isinstance(node.op, ast.MatMult):
            return True

        if isinstance(node, ast.AugAssign):
            return self._is_vector_expression(
                node.target
            ) or self._is_vector_expression(node.value)

        return self._is_vector_expression(node.left) or self._is_vector_expression(
            node.right
        )

    def _is_vector_expression(self, node: ast.AST) -> bool:
        """Return whether an expression is likely a vectorized value."""
        if isinstance(node, ast.Name):
            return node.id in self.vector_names

        if isinstance(node, ast.Subscript):
            return self._is_vector_expression(node.value)

        if isinstance(node, ast.BinOp):
            return self._is_vector_operation(node)

        if isinstance(node, ast.Call):
            return self._is_vector_constructor_call(
                node
            ) or self._is_vector_operation_call(node)

        if isinstance(node, ast.Attribute):
            return self._is_vector_expression(node.value)

        return False

    def _is_vector_constructor_call(self, node: ast.Call) -> bool:
        """Return whether a call constructs an array-like value."""
        name_parts = _call_name_parts(node.func)
        if not name_parts:
            return False

        if len(name_parts) == 1:
            return name_parts[0] in self.vector_constructor_names

        module_name = name_parts[0]
        call_name = name_parts[-1]
        return (
            module_name in self.vector_module_aliases
            and call_name in VECTOR_CONSTRUCTOR_CALLS
        )

    def _is_vector_operation_call(self, node: ast.Call) -> bool:
        """Return whether a call performs vectorized arithmetic."""
        name_parts = _call_name_parts(node.func)
        if not name_parts:
            return False

        if len(name_parts) == 1:
            return (
                name_parts[0] in self.vector_constructor_names
                or name_parts[0] in self.vector_output_names
            )

        module_name = name_parts[0]
        call_name = name_parts[-1]
        return module_name in self.vector_module_aliases and call_name in (
            VECTOR_CONSTRUCTOR_CALLS | VECTOR_OUTPUT_ARITHMETIC_CALLS
        )

    def _vector_arithmetic_call_name(self, node: ast.Call) -> str | None:
        """Return a vectorized arithmetic call name, if present."""
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in VECTOR_ARITHMETIC_CALLS
            and self._is_vector_expression(node.func.value)
        ):
            return node.func.attr

        name_parts = _call_name_parts(node.func)
        if not name_parts:
            return None

        if len(name_parts) == 1:
            if name_parts[0] in self.vector_arithmetic_names:
                return name_parts[0]
            return None

        module_name = name_parts[0]
        call_name = name_parts[-1]
        if (
            module_name in self.vector_module_aliases
            and call_name in VECTOR_ARITHMETIC_CALLS
        ):
            return call_name
        return None

    def _record_vector_target(self, target: ast.AST) -> None:
        """Remember assignment targets that hold vector-like values."""
        if isinstance(target, ast.Name):
            self.vector_names.add(target.id)
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self._record_vector_target(element)
        else:
            self.visit(target)


def _multiplication_operand_count(node: ast.AST) -> int:
    """Return operand count for an explicit multiplication chain."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _multiplication_operand_count(node.left) + _multiplication_operand_count(
            node.right
        )
    return 1


def _call_name_parts(node: ast.AST) -> tuple[str, ...]:
    """Return dotted name parts for a call target."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_call_name_parts(node.value), node.attr)
    return ()


def _extract_matches(pattern: re.Pattern[str], response: str) -> list[str]:
    """Return non-empty regex matches stripped of surrounding whitespace."""
    return [match.strip() for match in pattern.findall(response) if match.strip()]
