#!/usr/bin/env python3
"""
Experiment Execution Script for Discrete Probabilistic Reasoning.

This script executes LLM queries on the discrete probabilistic reasoning dataset.
It loads datasets, configuration, and prompts, then runs experiments for each
configured model and run number.

Steps:
1. Load dataframes from parquet files (bns.parquet, queries.parquet)
2. Load configuration (experiments.yaml)
3. Load prompts (prompts.yaml) - selects prompt_base for raw_reasoning or
   prompt_base_code for code_generation based on experiment_type
4. Execute queries for each model/run combination
"""

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import get_key

# Get API key from .env file
openrouter_api_key = get_key(".env", "OPENROUTER_API_KEY")

# Add project root to Python path for imports
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.database.discrete_experiments_db import initialize_discrete_experiments_db
from src.experiment.core import ExperimentType
from src.experiment.parallel import run_discrete_queries_parallel
from src.utils.yaml_utils import load_yaml


def _load_bn_dataset(parquet_path: Path) -> pd.DataFrame:
    """Load BN dataset from parquet file."""
    print(f"Loading BN dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"✓ Loaded {len(df)} Bayesian networks\n")
    return df


def _load_query_dataset(parquet_path: Path) -> pd.DataFrame:
    """Load query dataset from parquet file."""
    print(f"Loading query dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Deserialize JSON strings back to dictionaries for target and evidence
    df["target"] = df["target"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x,
    )
    df["evidence"] = df["evidence"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if x else {}),
    )

    print(f"✓ Loaded {len(df)} queries\n")
    return df


def _load_prompts(
    prompts_path: Path, experiment_type: str = "raw_reasoning"
) -> tuple[str, str]:
    """
    Load prompts from YAML file based on experiment type.

    Args:
        prompts_path: Path to the prompts YAML file.
        experiment_type: Type of experiment ("raw_reasoning" or "code_generation").
                        Determines which task prompt to use.

    Returns:
        Tuple of (system_prompt, task_prompt) where:
        - system_prompt: System prompt string (same for all experiment types)
        - task_prompt: Task prompt template string (task_prompt_raw for raw_reasoning,
                      task_prompt_code for code_generation)
    """
    print(f"Loading prompts from {prompts_path}...")
    prompts = load_yaml(prompts_path)

    system_prompt = prompts["system_prompt"]

    # Select task prompt based on experiment type
    if experiment_type == "code_generation":
        task_prompt = prompts["task_prompt_code"]
        print("  Using task_prompt_code for code_generation experiment")
    else:
        task_prompt = prompts["task_prompt_raw"]
        print(f"  Using task_prompt_raw for {experiment_type} experiment")

    print("✓ Loaded prompts\n")
    return system_prompt, task_prompt


def main():
    """Main execution function."""
    print("=" * 60)
    print("Discrete Probabilistic Reasoning Experiment Execution")
    print("=" * 60)
    print()

    # Check API key was loaded
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return

    # Initialize database table
    print("Initializing discrete_experiments database table (if needed)...")
    initialize_discrete_experiments_db()
    print()

    # Setup paths
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data" / "discrete"
    config_path = repo_root / "config" / "experiments.yaml"
    prompts_path = repo_root / "prompts" / "experiments" / "prompts.yaml"

    # Step 1: Load dataframes
    bns_df = _load_bn_dataset(data_dir / "bns.parquet")
    queries_df = _load_query_dataset(data_dir / "queries.parquet")

    queries_df = queries_df[queries_df["naming_strategy"] == "simple"]

    # Step 2: Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_yaml(config_path)
    num_runs = config.get("runs", 1)
    llm_models = config.get("llm_models", [])
    experiment_type_str = config.get("experiment_type", "raw_reasoning")
    # Convert string to Enum
    experiment_type = ExperimentType(experiment_type_str)

    print(
        f"✓ Configuration loaded: {len(llm_models)} model(s), "
        f"{num_runs} run(s), experiment_type: {experiment_type.value}\n"
    )

    # Step 3: Load prompts (select appropriate prompt based on experiment_type)
    system_prompt, task_prompt = _load_prompts(
        prompts_path, experiment_type=experiment_type.value
    )

    # Step 4: Execute queries for each model/run combination
    total_experiments = len(llm_models) * num_runs
    experiment_num = 0

    for model_idx, model_cfg in enumerate(llm_models, 1):
        model_name = model_cfg.get("model_name")
        reasoning_model = model_cfg.get("reasoning_model")
        temperature = model_cfg.get("temperature")
        reasoning_effort = model_cfg.get("reasoning_effort")
        reasoning_summary = model_cfg.get("reasoning_summary")
        max_tokens = model_cfg.get("max_tokens")

        print(f"[Model {model_idx}/{len(llm_models)}] {model_name}")
        print(f"  Reasoning model: {reasoning_model}")
        print(f"  Temperature: {temperature}")
        if reasoning_effort:
            print(f"  Reasoning effort: {reasoning_effort}")
        if reasoning_summary:
            print(f"  Reasoning summary: {reasoning_summary}")
        if max_tokens:
            print(f"  Max tokens: {max_tokens}")

        for run in range(1, num_runs + 1):
            experiment_num += 1
            print(
                f"\n  [Run {run}/{num_runs}] "
                f"Experiment {experiment_num}/{total_experiments}",
            )

            # Run queries in parallel batches (langfuse initialization happens
            # inside run_discrete_queries_parallel)
            run_discrete_queries_parallel(
                queries_df=queries_df,
                bns_df=bns_df,
                model_name=model_name,
                run=run,
                experiment_type=experiment_type,
                task_prompt=task_prompt,
                system_prompt=system_prompt,
                reasoning_model=reasoning_model,
                openrouter_api_key=openrouter_api_key,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                max_tokens=max_tokens,
                batch_size=8,
                max_workers=8,
                verbose=True,
            )

            # run_discrete_queries_sequential(
            #     queries_df=queries_df,
            #     bns_df=bns_df,
            #     model_name=model_name,
            #     run=run,
            #     experiment_type=experiment_type,
            #     task_prompt=task_prompt,
            #     system_prompt=system_prompt,
            #     reasoning_model=reasoning_model,
            #     temperature=temperature,
            #     reasoning_effort=reasoning_effort,
            #     reasoning_summary=reasoning_summary,
            #     max_tokens=max_tokens,
            #     verbose=True,
            # )

            print(f"  ✓ Completed run {run}")

        print()

    print("=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
