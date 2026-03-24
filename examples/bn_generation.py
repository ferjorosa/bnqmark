"""Example script for BN generation with naming variants."""

from __future__ import annotations

from pgmpy.models import DiscreteBayesianNetwork

from src.bn import (
    generate_bayesian_networks_and_metadata,
    verify_naming_variant,
)
from src.bn.generation.core.generation import generate_discrete_bn_from_dag
from src.dag import NamingStrategy, generate_single_dag
from src.naming_variants import (
    create_bn_naming_variant,
)


def test_bn_generation():
    """Test basic BN generation."""
    print("Test 1: Basic BN Generation")

    dag, tw, _ = generate_single_dag(
        n_nodes=5,
        target_treewidth=2,
        node_naming=NamingStrategy.SIMPLE,
        seed=42,
    )
    print(f"Generated DAG with {dag.number_of_nodes()} nodes, treewidth={tw}")

    bn, meta = generate_discrete_bn_from_dag(
        dag,
        arity_strategy={"type": "range", "min": 2, "max": 3},
        dirichlet_alpha=1.0,
        determinism_fraction=0.0,
        seed=123,
    )
    print(f"Generated BN with {bn.number_of_nodes()} nodes")

    return bn, dag, meta


def test_single_naming_variant(original_bn: DiscreteBayesianNetwork):
    """Test that a single naming variant preserves structure and CPTs."""
    print("\nTest 2: Single Naming Variant Verification")

    strategy = NamingStrategy.CONFUSING
    variant_bn = create_bn_naming_variant(original_bn, strategy, seed=42)

    from src.naming_variants import create_name_mapping_from_strategy

    old_node_names = list(original_bn.nodes())
    name_mapping = create_name_mapping_from_strategy(
        old_node_names,
        strategy,
        seed=42,
    )

    structure_match, cpt_match, error_message = verify_naming_variant(
        original_bn,
        variant_bn,
        name_mapping,
    )

    if structure_match and cpt_match:
        print(f"{strategy.value} naming variant preserved structure and CPTs.")
    else:
        print(f"Error: {strategy.value} naming variant failed: {error_message}")
        return False

    return True


def test_sweep_pipeline():
    """Test the full sweep pipeline."""
    print("\nTest 3: Full Sweep Pipeline")

    bn_dag_pairs = generate_bayesian_networks_and_metadata(
        ns=[5],
        treewidths=[2],
        arity_specs=[{"type": "range", "min": 2, "max": 3}],
        dirichlet_alphas=[1.0],
        determinism_fracs=[0.0],
        variants_per_combo=2,
        base_seed=42,
    )

    print(f"Generated {len(bn_dag_pairs)} BN-DAG pairs")

    if len(bn_dag_pairs) == 0:
        print("Error: No BN-DAG pairs generated")
        return False

    return True


def main():
    """Run all tests."""
    print("Bayesian Network Generation Pipeline Test")

    try:
        original_bn, original_dag, original_meta = test_bn_generation()

        if not test_single_naming_variant(original_bn):
            print("Single naming variant test failed")
            return

        if not test_sweep_pipeline():
            print("Sweep pipeline test failed")
            return

        print("\nAll tests passed")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
