#!/usr/bin/env python3
"""
Script to re-run code execution for code_generation experiments.

Re-runs code execution and updates llm_probability.
This script iterates through experiments in the database where:
1. experiment_type is 'code_generation'
2. llm_probability is NULL

It executes the code in the 'response' field using the secure execution utility
and updates the 'llm_probability' if execution is successful.
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.database.database import execute_query, query_db
from src.database.discrete_experiments_db import FULL_TABLE_NAME
from src.utils.code_execution_utils import execute_and_extract_probability

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the code execution recovery script."""
    logger.info("Starting code execution recovery script...")

    # 1. Fetch target experiments
    # Only fetch experiments that haven't been processed yet
    # (i.e., NULL probability AND no exception info recorded)
    # Note: We don't filter by code markers here because the execution utility
    # will handle that and record appropriate error messages
    fetch_query = f"""
        SELECT id, response, model_name, run, query_uuid
        FROM {FULL_TABLE_NAME}
        WHERE experiment_type = 'code_generation'
          AND llm_probability IS NULL
          AND code_exception_type IS NULL
    """

    logger.info("Fetching experiments with missing probabilities...")
    experiments_df = query_db(fetch_query)

    if experiments_df.empty:
        logger.info("No experiments found requiring code execution.")
        return

    total_experiments = len(experiments_df)
    logger.info(f"Found {total_experiments} experiments to process.")

    updated_count = 0
    failed_count = 0

    # 2. Iterate and process
    for i, row in experiments_df.iterrows():
        if i % 10 == 0:
            print(f"Processing {i}/{total_experiments}...", end="\r")

        experiment_id = row["id"]
        response = row["response"]

        # Execute code
        try:
            # Using 30s timeout to allow for subprocess + pgmpy import overhead
            probability, pgmpy_fix_applied, exception_type, exception_message = (
                execute_and_extract_probability(response, timeout=30)
            )

            # Check if they followed instructions (used <code> tags)
            followed_instructions = "<code>" in response and "</code>" in response

            # Log execution result
            if probability is not None:
                logger.info(
                    f"✅ Experiment {experiment_id}: probability = {probability:.6f}"
                )
                updated_count += 1
            else:
                error_preview = (
                    exception_message[:100] if exception_message else "Unknown error"
                )
                logger.warning(
                    f"❌ Experiment {experiment_id}: {exception_type} - {error_preview}"
                )
                failed_count += 1

            # Update database regardless of success or failure
            # This ensures we capture exception info even for failed executions
            update_query = f"""
                UPDATE {FULL_TABLE_NAME}
                SET llm_probability = %s,
                    code_followed_formatting_instructions = %s,
                    code_pgmpy_library_fix = %s,
                    code_exception_type = %s,
                    code_exception_message = %s
                WHERE id = %s
            """
            execute_query(
                update_query,
                (
                    probability,
                    followed_instructions,
                    pgmpy_fix_applied,
                    exception_type,
                    exception_message,
                    experiment_id,
                ),
            )

        except Exception as e:
            logger.error(f"Error processing experiment {experiment_id}: {e}")
            failed_count += 1

    logger.info("=" * 40)
    logger.info("Processing complete.")
    logger.info(f"Total processed: {total_experiments}")
    logger.info(f"Successfully updated: {updated_count}")
    logger.info(f"Failed / No result: {failed_count}")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
