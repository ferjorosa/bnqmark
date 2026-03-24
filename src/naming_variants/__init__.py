"""
Shared naming variant generation for Bayesian Networks and DAGs.

This module provides functionality to create naming variants of both Bayesian Networks
and DAGs, enabling clean ablation testing where structure and CPTs remain identical
but node names vary. This is useful for evaluating how node naming affects LLM
probabilistic reasoning performance.

This is shared code used by both bn_generation and dag modules.
"""

from .naming_variants import (
    create_bn_naming_variant,
    create_dag_naming_variant,
    create_name_mapping_from_strategy,
)

__all__ = [
    "create_name_mapping_from_strategy",
    "create_bn_naming_variant",
    "create_dag_naming_variant",
]
