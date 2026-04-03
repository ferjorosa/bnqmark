#!/usr/bin/env python3
"""Arithmetic Behavior Trace Analysis Execution Script."""

import logging
import sys
from pathlib import Path

from dotenv import get_key

# Get API key from .env file
openrouter_api_key = get_key(".env", "OPENROUTER_API_KEY")

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.database.discrete_arithmetic_behavior_analysis_db import (
    get_existing_arithmetic_analysis_identifiers,
    initialize_arithmetic_analysis_db,
)
from src.trace_analysis.core import AnalysisType
from src.trace_analysis.fetch_experiments import fetch_experiments
from src.trace_analysis.parallel import run_trace_analysis_parallel
from src.utils.yaml_utils import load_yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the arithmetic behavior trace analysis."""
    # Check API key was loaded
    if not openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not found in .env file")
        return

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "config"
        / "trace_analysis"
        / "raw_reasoning_arithmetic_behaviour.yaml"
    )
    prompts_path = (
        repo_root
        / "prompts"
        / "trace_analysis"
        / "raw_reasoning_arithmetic_behaviour.yaml"
    )

    logger.info(f"Loading config from {config_path}")
    config = load_yaml(config_path)

    logger.info(f"Loading prompt from {prompts_path}")
    prompt_config = load_yaml(prompts_path)

    system_prompt = prompt_config["system_prompt"]
    task_prompt = prompt_config["task_prompt"]

    logger.info("Initializing arithmetic analysis database...")
    initialize_arithmetic_analysis_db()

    analyzer_config = config["analysis_model"]
    model_name = analyzer_config["model_name"]
    reasoning_effort = analyzer_config.get("reasoning_effort")
    reasoning_summary = analyzer_config.get("reasoning_summary")
    max_tokens = analyzer_config.get("max_tokens")

    models_to_analyze = config.get("models_to_analyze", [])
    run = config.get("run", 1)

    logger.info(
        f"Starting Arithmetic Behavior Analysis for models: {models_to_analyze}, "
        f"run: {run}"
    )

    # Get existing analyses to skip
    existing_analyses = get_existing_arithmetic_analysis_identifiers()

    for model_to_analyze in models_to_analyze:
        logger.info(f"Fetching experiments for {model_to_analyze}...")
        experiments = fetch_experiments(model_to_analyze, "raw_reasoning", run)
        logger.info(f"Found {len(experiments)} experiments for {model_to_analyze}")

        run_trace_analysis_parallel(
            experiments_df=experiments,
            existing_analyses=existing_analyses,
            model_name=model_name,
            analysis_type=AnalysisType.ARITHMETIC_BEHAVIOR,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            openrouter_api_key=openrouter_api_key,
            temperature=analyzer_config["temperature"],
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_tokens=max_tokens,
            batch_size=4,
            max_workers=4,
            verbose=True,
        )

    logger.info("Arithmetic Behavior Analysis completed!")


if __name__ == "__main__":
    main()
