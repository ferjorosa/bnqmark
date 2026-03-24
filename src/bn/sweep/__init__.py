"""
BN Sweep Module.

This module provides high-level functions for generating multiple Bayesian Networks
with systematic parameter sweeps. The key feature is that naming variants are applied
as a post-processing step, enabling clean ablation testing where the same BN structure
and CPTs are used with different naming strategies.

Main Function:
    generate_bayesian_networks_and_metadata() - Generate BNs with parameter sweeps

Example:
    >>> from src.bn_generation import generate_bayesian_networks_and_metadata
    >>> bns, metadata, dags = generate_bayesian_networks_and_metadata(
    ...     ns=[10, 15],
    ...     treewidths=[2, 3],
    ...     arity_specs=[{"type": "fixed", "fixed": 2}],
    ...     dirichlet_alphas=[0.5, 1.0],
    ...     determinism_fracs=[0.0],
    ...     naming_strategies=["simple"],
    ...     variants_per_combo=4,
    ...     base_seed=42,
    ... )
"""

# Re-export key types for type hints
from ..generation.core.types import (
    BaseBNMetadata,
    OutputRowMetadata,
)
from .sweep import (
    generate_bayesian_networks_and_metadata,
)

__all__ = [
    # Main API
    "generate_bayesian_networks_and_metadata",
    # Types (for type hints)
    "BaseBNMetadata",
    "OutputRowMetadata",
]
