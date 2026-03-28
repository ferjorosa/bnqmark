#!/usr/bin/env python3
"""Code Generation Behaviour Trace Analysis Execution Script."""

import logging
from pathlib import Path

from src.database.discrete_code_generation_behaviour_analysis_db import (
    get_existing_code_generation_behaviour_analysis_identifiers,
    initialize_code_generation_behaviour_analysis_db,
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
    """Run the code generation behaviour trace analysis."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = (
        repo_root / "config" / "trace_analysis" / "code_generation_behaviour.yaml"
    )
    prompts_path = (
        repo_root / "prompts" / "trace_analysis" / "code_generation_behaviour.yaml"
    )

    logger.info(f"Loading config from {config_path}")
    config = load_yaml(config_path)

    logger.info(f"Loading prompt from {prompts_path}")
    prompt_config = load_yaml(prompts_path)

    system_prompt = prompt_config["system_prompt"]
    task_prompt = prompt_config["task_prompt"]

    logger.info("Initializing code generation behaviour analysis database...")
    initialize_code_generation_behaviour_analysis_db()

    analyzer_config = config["analysis_model"]
    model_name = analyzer_config["model_name"]
    reasoning_effort = analyzer_config.get("reasoning_effort")
    reasoning_summary = analyzer_config.get("reasoning_summary")
    max_tokens = analyzer_config.get("max_tokens")

    models_to_analyze = config.get("models_to_analyze", [])
    run = config.get("run", 1)

    logger.info(
        f"Starting Code Generation Behaviour Analysis for models: {models_to_analyze}, "
        f"run: {run}"
    )

    # Get existing analyses to skip
    existing_analyses = get_existing_code_generation_behaviour_analysis_identifiers()

    for model_to_analyze in models_to_analyze:
        logger.info(f"Fetching experiments for {model_to_analyze}...")
        experiments = fetch_experiments(model_to_analyze, "code_generation", run)
        logger.info(f"Found {len(experiments)} experiments for {model_to_analyze}")

        run_trace_analysis_parallel(
            experiments_df=experiments,
            existing_analyses=existing_analyses,
            model_name=model_name,
            analysis_type=AnalysisType.CODE_GENERATION_BEHAVIOUR,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            temperature=analyzer_config["temperature"],
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_tokens=max_tokens,
            batch_size=8,
            max_workers=8,
            verbose=True,
        )

    logger.info("Code Generation Behaviour Analysis completed!")


if __name__ == "__main__":
    main()
