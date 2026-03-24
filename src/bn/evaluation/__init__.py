"""
BN Evaluation Module.

This module provides functions to verify that naming variants preserve
structure and CPTs correctly, which is essential for ablation testing.

Main Functions:
    verify_naming_variant() - Verify that a naming variant preserves structure and CPTs
    compare_bn_structures() - Compare if two BNs have the same structure
    compare_cpt_values() - Compare if two BNs have the same CPT values

Example:
    >>> from src.bn_generation import verify_naming_variant
    >>> structure_match, cpt_match, error_msg = verify_naming_variant(
    ...     original_bn=bn1, variant_bn=bn2, name_mapping=name_mapping
    ... )
    >>> if structure_match and cpt_match:
    ...     print("Variant is valid!")
"""

from .evaluation import (
    compare_bn_structures,
    compare_cpt_values,
    verify_naming_variant,
)

__all__ = [
    "compare_bn_structures",
    "compare_cpt_values",
    "verify_naming_variant",
]
