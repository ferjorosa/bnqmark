"""
Core Bayesian Network generation implementation.

This package contains the low-level implementation details for generating
Bayesian Networks. Most users should use the high-level API from generator.py
and sweep.py instead of importing directly from this package.

Internal modules:
    - generation: Core BN generation from DAGs
    - types: Type definitions and metadata dataclasses
"""

# Optionally re-export key functions for internal use
from .generation import generate_discrete_bn_from_dag, generate_variants_for_dag
from .types import (
    ArityStrategy,
    BaseBNMetadata,
    BNGenerationMetadata,
    OutputRowMetadata,
)

__all__ = [
    # Generation
    "generate_discrete_bn_from_dag",
    "generate_variants_for_dag",
    # Types
    "ArityStrategy",
    "BNGenerationMetadata",
    "BaseBNMetadata",
    "OutputRowMetadata",
]
