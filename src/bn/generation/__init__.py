"""
BN Generation Module.

This module provides functionality for generating discrete Bayesian Networks
with controllable properties for probabilistic reasoning experiments.

Main Functions:
    generate_single_bn() - Generate a single BN from scratch

Example:
    >>> from src.bn_generation import generate_single_bn
    >>> bn, dag, meta = generate_single_bn(
    ...     n_nodes=10,
    ...     target_treewidth=3,
    ...     arity_strategy={"type": "range", "min": 2, "max": 4},
    ...     seed=42,
    ... )
    >>> print(f"Generated BN with {bn.number_of_nodes()} nodes")

Advanced Usage:
    For fine-grained control, import from submodules:
    >>> from src.bn_generation.generation.core import (
    ...     generate_discrete_bn_from_dag,
    ...     ArityStrategy,
    ... )
"""

# Main entry points
# Re-export key types for type hints and advanced usage
from .core.types import (
    ArityStrategy,
    BNGenerationMetadata,
)
from .generator import (
    generate_single_bn,
)

__all__ = [
    # Main API
    "generate_single_bn",
    # Types (for type hints)
    "ArityStrategy",
    "BNGenerationMetadata",
]
