"""
Bayesian Network Generation Module.

This module provides functionality for generating discrete Bayesian Networks
with controllable properties for probabilistic reasoning experiments.

The key insight is that different BN properties (treewidth, arity, CPT skewness, etc.)
create different reasoning challenges for LLMs, making systematic BN generation crucial
for comprehensive evaluation.

Main Functions (start here):
    Generation:
    - generate_single_bn() - Generate a single BN from scratch
    - generate_bayesian_networks_and_metadata() - Generate BNs with parameter sweeps

    Evaluation:
    - verify_naming_variant() - Verify naming variants preserve structure and CPTs

    Analysis:
    - draw_bayesian_network() - Visualize BN with hierarchical layout
    - compute_average_markov_blanket_size() - Network-level Markov blanket metrics

Example:
    >>> from src.bn import generate_single_bn, draw_bayesian_network
    >>> # Generate a single Bayesian Network
    >>> bn, dag, meta = generate_single_bn(
    ...     n_nodes=10,
    ...     target_treewidth=3,
    ...     arity_strategy={"type": "range", "min": 2, "max": 4},
    ...     seed=42,
    ... )
    >>> print(f"Generated BN with {bn.number_of_nodes()} nodes")
    >>> # Visualize the network
    >>> draw_bayesian_network(bn, show_treewidth=True)

Advanced Usage:
    For fine-grained control, import from submodules:
    >>> from src.bn.generation.core import generate_discrete_bn_from_dag, ArityStrategy
"""

# Generation API
# Analysis API
from .analysis import (
    compute_average_markov_blanket_size,
    draw_bayesian_network,
    draw_networkx_graph,
    num_edges,
)

# Evaluation API
from .evaluation import (
    compare_bn_structures,
    compare_cpt_values,
    verify_naming_variant,
)
from .generation import (
    ArityStrategy,
    BNGenerationMetadata,
    generate_single_bn,
)

# Sweep API
from .sweep import (
    BaseBNMetadata,
    OutputRowMetadata,
    generate_bayesian_networks_and_metadata,
)

__all__ = [
    # Generation API
    "generate_single_bn",
    "ArityStrategy",
    "BNGenerationMetadata",
    # Sweep API
    "generate_bayesian_networks_and_metadata",
    "BaseBNMetadata",
    "OutputRowMetadata",
    # Evaluation API
    "compare_bn_structures",
    "compare_cpt_values",
    "verify_naming_variant",
    # Analysis API
    "num_edges",
    "compute_average_markov_blanket_size",
    "draw_bayesian_network",
    "draw_networkx_graph",
]
