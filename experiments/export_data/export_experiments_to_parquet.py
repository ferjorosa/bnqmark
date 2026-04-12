#!/usr/bin/env python3
"""
Script to export the discrete_experiments table to a Parquet file.

Exports to: data/experiments.parquet
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.database.database import query_db
from src.database.discrete_experiments_db import TABLE_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Export experiments table to Parquet."""
    logger.info("Starting export...")

    # Query all experiments
    query = f"SELECT * FROM {TABLE_NAME}"
    logger.info(f"Querying table: {TABLE_NAME}")
    df = query_db(query)

    logger.info(f"Retrieved {len(df)} rows")

    # Define output path
    data_dir = _repo_root / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "experiments.parquet"

    # Export to Parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Exported to: {output_path}")


if __name__ == "__main__":
    main()
