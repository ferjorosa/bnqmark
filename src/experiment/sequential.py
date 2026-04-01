"""
Sequential execution logic for probabilistic reasoning experiments.

This module provides sequential processing capabilities for running
discrete queries one at a time.
"""

import logging
import pickle

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from tqdm import tqdm

from src.experiment.core import (
    ExperimentType,
    filter_existing_queries,
    run_single_query,
)

logger = logging.getLogger(__name__)


def run_discrete_queries_sequential(
    queries_df: pd.DataFrame,
    bns_df: pd.DataFrame,
    model_name: str,
    run: int,
    experiment_type: ExperimentType,
    task_prompt: str,
    system_prompt: str,
    reasoning_model: bool,
    openrouter_api_key: str,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Run discrete queries sequentially for a given model and run.

    For all queries, perform LLM calls, parse response probability, and
    insert experiment row into the database.

    Args:
        queries_df: DataFrame with queries.
        bns_df: DataFrame with Bayesian networks.
        model_name: Model name (e.g., "openai/gpt-4o", "google/gemini-2.5-flash").
        run: Experiment run number.
        experiment_type: ExperimentType, e.g. RAW_REASONING or CODE_GENERATION.
        task_prompt: Prompt template with {cpts} and {query} placeholders.
            For RAW_REASONING: prompt_base (direct probability answer).
            For CODE_GENERATION: prompt_base_code (asks for Python code).
        system_prompt: LLM system prompt (for all types).
        reasoning_model: If this is a reasoning model.
        openrouter_api_key: OpenRouter API key (required).
        temperature: LLM temperature parameter.
        reasoning_effort: Reasoning effort, one of "low", "medium", "high",
            or None. Defaults to None.
        reasoning_summary: Reasoning summary, one of "auto", "concise",
            "detailed", or None. Defaults to None.
        max_tokens: Max tokens for the response. Defaults to None.
        verbose: Print per-query progress if True.
    """
    # Create mapping from (bn_uuid, naming_strategy) to BN objects
    bn_map: dict[tuple[str, str], DiscreteBayesianNetwork] = {}
    for _, bn_row in bns_df.iterrows():
        bn_uuid = bn_row["bn_uuid"]
        naming_strategy = bn_row["naming_strategy"]
        bn = pickle.loads(bn_row["bn_pickle"])
        bn_map[(bn_uuid, naming_strategy)] = bn

    # Filter out queries that already exist in the database
    queries_df = filter_existing_queries(
        queries_df=queries_df,
        run=run,
        model_name=model_name,
        experiment_type=experiment_type,
        verbose=verbose,
    )

    if len(queries_df) == 0:
        return

    # Process each query
    total_queries = len(queries_df)
    pbar = None
    if verbose:
        pbar = tqdm(total=total_queries, desc="Processing queries", unit="query")

    for _, query_row in queries_df.iterrows():
        query_uuid = query_row["query_uuid"]

        if verbose and pbar:
            pbar.set_postfix(query_id=query_uuid[:8])

        result = run_single_query(
            query_row=query_row,
            bn_map=bn_map,
            model_name=model_name,
            run=run,
            experiment_type=experiment_type,
            task_prompt=task_prompt,
            system_prompt=system_prompt,
            reasoning_model=reasoning_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_tokens=max_tokens,
            openrouter_api_key=openrouter_api_key,
        )

        if result.get("error") is not None:
            logger.error(f"Error processing query {query_uuid[:8]}: {result['error']}")

        if verbose and pbar:
            pbar.update(1)
