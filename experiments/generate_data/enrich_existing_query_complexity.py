#!/usr/bin/env python3
"""
Add scalar VE operation counts to an existing query parquet file.

This script does not regenerate Bayesian networks or queries. It loads existing
``bns.parquet`` and ``queries.parquet`` files, recomputes query complexity for
each stored query, and writes the query rows back with scalar addition and
multiplication counts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.queries.complexity import compute_query_complexity


def _parse_json_mapping(value: Any) -> dict[str, str]:
    """Parse a JSON-encoded mapping stored in the query parquet."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, float) and pd.isna(value):
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    raise TypeError(f"Unsupported mapping value type: {type(value).__name__}")


def _load_bn_lookup(bns_df: pd.DataFrame) -> dict[tuple[str, str], Any]:
    """Load BN pickles keyed by (bn_uuid, naming_strategy)."""
    lookup = {}
    for _, row in bns_df.iterrows():
        key = (row["bn_uuid"], row.get("naming_strategy", "simple"))
        lookup[key] = pickle.loads(row["bn_pickle"])
    return lookup


def enrich_queries(bns_path: Path, queries_path: Path) -> pd.DataFrame:
    """Return queries with scalar VE operation-count columns added."""
    bns_df = pd.read_parquet(bns_path)
    queries_df = pd.read_parquet(queries_path)
    bn_lookup = _load_bn_lookup(bns_df)

    scalar_additions: list[int] = []
    scalar_multiplications: list[int] = []

    total = len(queries_df)
    for _, row in queries_df.iterrows():
        key = (row["bn_uuid"], row.get("naming_strategy", "simple"))
        if key not in bn_lookup:
            raise KeyError(f"No BN found for key {key}")

        target = _parse_json_mapping(row["target"])
        evidence = _parse_json_mapping(row["evidence"])
        metrics = compute_query_complexity(
            bn_lookup[key],
            list(target.keys()),
            list(evidence.keys()),
            verbose=False,
        )
        scalar_additions.append(metrics.scalar_additions)
        scalar_multiplications.append(metrics.scalar_multiplications)

        processed = len(scalar_additions)
        if processed % 50 == 0 or processed == total:
            print(f"  Processed {processed}/{total} queries")

    enriched_df = queries_df.copy()
    enriched_df["scalar_additions"] = scalar_additions
    enriched_df["scalar_multiplications"] = scalar_multiplications
    return enriched_df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_repo_root / "data",
        help="Directory containing bns.parquet and queries.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to overwriting queries.parquet.",
    )
    args = parser.parse_args()

    bns_path = args.data_dir / "bns.parquet"
    queries_path = args.data_dir / "queries.parquet"
    output_path = args.output or queries_path

    if not bns_path.exists():
        raise FileNotFoundError(f"Missing BN parquet: {bns_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing query parquet: {queries_path}")

    print(f"Loading BNs from {bns_path}")
    print(f"Loading queries from {queries_path}")
    enriched_df = enrich_queries(bns_path, queries_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    enriched_df.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.replace(output_path)

    print(f"Saved {len(enriched_df)} enriched query rows to {output_path}")
    print(
        "Scalar additions range: "
        f"{enriched_df['scalar_additions'].min()}-"
        f"{enriched_df['scalar_additions'].max()}",
    )
    print(
        "Scalar multiplications range: "
        f"{enriched_df['scalar_multiplications'].min()}-"
        f"{enriched_df['scalar_multiplications'].max()}",
    )


if __name__ == "__main__":
    main()
