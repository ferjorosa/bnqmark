"""
Node naming strategies for DAG generation.

This module provides functions for generating node names using different
strategies and relabeling graph nodes with custom names.
"""

import random
import string

import networkx as nx
import numpy as np

from src.dag.generation.core.types import NamingStrategy


def generate_node_names(
    n_nodes: int,
    strategy: NamingStrategy = NamingStrategy.SIMPLE,
    seed: int | None = None,
) -> list[str]:
    """
    Generate node names using different strategies for experiments.

    Different naming strategies can help test how node names affect downstream
    tasks like LLM probabilistic reasoning or human interpretability.

    Args:
        n_nodes: Number of node names to generate
        strategy: Naming strategy (NamingStrategy enum)
        seed: Random seed for reproducible name generation

    Returns:
        List of node names

    Strategies:
        - SIMPLE: V0, V1, V2, ... (clear and systematic)
        - CONFUSING: X_445aFa, S_af3a34, ... (random alphanumeric)
        - SEMANTIC: meaningful names like 'Rain', 'Sprinkler', 'WetGrass'
        - MIXED: combination of different strategies
        - DEFAULT: keeps numeric labels (0, 1, 2, ...)

    Example:
        >>> names = generate_node_names(3, NamingStrategy.SIMPLE)
        >>> print(names)
        ['V0', 'V1', 'V2']

        >>> names = generate_node_names(3, NamingStrategy.CONFUSING, seed=42)
        >>> print(names)
        ['X_7a4f2b', 'Q_9c1e8d', 'Z_3b6a9f']
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if strategy == NamingStrategy.DEFAULT:
        # Return numeric labels as strings
        return [str(i) for i in range(n_nodes)]

    elif strategy == NamingStrategy.SIMPLE:
        return [f"V{i}" for i in range(n_nodes)]

    elif strategy == NamingStrategy.CONFUSING:
        names = []
        prefixes = list(string.ascii_uppercase)
        for _ in range(n_nodes):
            prefix = np.random.choice(prefixes)
            # Generate random alphanumeric suffix
            chars = string.ascii_lowercase + string.digits
            suffix = "".join(np.random.choice(list(chars), size=6))
            names.append(f"{prefix}_{suffix}")
        return names

    elif strategy == NamingStrategy.SEMANTIC:
        # Common semantic names for Bayesian networks
        semantic_names = [
            "Rain",
            "Sprinkler",
            "WetGrass",
            "Cloudy",
            "Season",
            "Temperature",
            "Humidity",
            "Wind",
            "Pressure",
            "Visibility",
            "Traffic",
            "Accident",
            "Weather",
            "Road",
            "Time",
            "Age",
            "Gender",
            "Income",
            "Education",
            "Health",
            "Smoking",
            "Exercise",
            "Diet",
            "Stress",
            "Sleep",
            "Disease",
            "Symptom",
            "Treatment",
            "Recovery",
            "Test",
            "Cause",
            "Effect",
            "Factor",
            "Outcome",
            "Risk",
            "Signal",
            "Noise",
            "Data",
            "Model",
            "Prediction",
        ]

        if n_nodes <= len(semantic_names):
            return list(np.random.choice(semantic_names, size=n_nodes, replace=False))
        else:
            # If we need more names than available, cycle through and add numbers
            base_names = list(
                np.random.choice(
                    semantic_names,
                    size=len(semantic_names),
                    replace=False,
                ),
            )
            names = base_names.copy()
            counter = 1
            while len(names) < n_nodes:
                for base_name in base_names:
                    if len(names) >= n_nodes:
                        break
                    names.append(f"{base_name}{counter}")
                counter += 1
            return names[:n_nodes]

    elif strategy == NamingStrategy.MIXED:
        # Mix of different strategies
        names = []
        strategies = [
            NamingStrategy.SIMPLE,
            NamingStrategy.CONFUSING,
            NamingStrategy.SEMANTIC,
        ]

        for i in range(n_nodes):
            chosen_strategy = np.random.choice(strategies)
            if chosen_strategy == NamingStrategy.SIMPLE:
                names.append(f"V{i}")
            elif chosen_strategy == NamingStrategy.CONFUSING:
                prefix = np.random.choice(list(string.ascii_uppercase))
                chars = string.ascii_lowercase + string.digits
                suffix = "".join(np.random.choice(list(chars), size=4))
                names.append(f"{prefix}_{suffix}")
            else:  # NamingStrategy.SEMANTIC
                semantic_options = [
                    "Factor",
                    "Node",
                    "Variable",
                    "Element",
                    "Component",
                ]
                base = np.random.choice(semantic_options)
                names.append(f"{base}{i}")

        return names

    else:
        # This should never happen due to enum validation above
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Must be one of {[s.value for s in NamingStrategy]}",
        )


def relabel_graph_nodes(graph: nx.Graph, node_names: list[str]) -> nx.Graph:
    """
    Relabel graph nodes with custom names.

    Args:
        graph: NetworkX graph with default node labels (0, 1, 2, ...)
        node_names: List of new node names (must match number of nodes)

    Returns:
        New graph with relabeled nodes

    Raises:
        ValueError: If number of names doesn't match number of nodes

    Example:
        >>> import networkx as nx
        >>> graph = nx.DiGraph([(0, 1), (1, 2)])
        >>> names = ["A", "B", "C"]
        >>> graph_relabeled = relabel_graph_nodes(graph, names)
        >>> print(list(graph_relabeled.edges()))
        [('A', 'B'), ('B', 'C')]
    """
    if len(node_names) != graph.number_of_nodes():
        raise ValueError(
            f"Number of names ({len(node_names)}) must match "
            f"number of nodes ({graph.number_of_nodes()})",
        )

    # Create mapping from old labels to new names
    old_nodes = sorted(graph.nodes())  # Ensure consistent ordering
    mapping = {old_nodes[i]: node_names[i] for i in range(len(old_nodes))}

    return nx.relabel_nodes(graph, mapping)
