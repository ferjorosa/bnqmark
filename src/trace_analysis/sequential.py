"""
Sequential execution logic for trace analysis.

This module provides sequential processing capabilities for analyzing
traces one at a time.
"""

import logging

import pandas as pd
from tqdm import tqdm

from src.trace_analysis.core import (
    AnalysisType,
    filter_existing_analyses,
    run_single_analysis,
)

logger = logging.getLogger(__name__)


def run_trace_analysis_sequential(
    experiments_df: pd.DataFrame,
    existing_analyses: set,
    model_name: str,
    analysis_type: AnalysisType,
    system_prompt: str,
    task_prompt: str,
    temperature: float = 0.0,
    openrouter_api_key: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    max_tokens: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Run trace analysis sequentially for a given set of experiments.

    Args:
        experiments_df: DataFrame containing experiments to analyze.
        existing_analyses: Set of existing analysis identifiers to skip.
        model_name: Model name for the analysis LLM (e.g., "openai/gpt-4o").
        analysis_type: Type of analysis to perform.
        system_prompt: System prompt string.
        task_prompt: Task prompt template string.
        temperature: Temperature setting for the analysis LLM.
        openrouter_api_key: OpenRouter API key. If not provided, will be read
            from OPENROUTER_API_KEY environment variable.
        reasoning_effort: Reasoning effort level. Values: "xhigh", "high",
            "medium", "low", "minimal", "none".
        reasoning_summary: Reasoning summary level. Values: "auto",
            "concise", "detailed".
        max_tokens: Maximum tokens for the response.
        verbose: If True, print progress.
    """
    experiments_df = filter_existing_analyses(
        experiments_df=experiments_df,
        existing_analyses=existing_analyses,
        verbose=verbose,
    )

    if len(experiments_df) == 0:
        if verbose:
            print("  No new traces to analyze")
        return

    total_experiments = len(experiments_df)
    pbar = None
    if verbose:
        pbar = tqdm(total=total_experiments, desc="Analyzing traces", unit="trace")

    for _, row in experiments_df.iterrows():
        query_uuid = row["query_uuid"]

        if verbose and pbar:
            pbar.set_postfix(query_id=query_uuid[:8])

        result, duration, _ = run_single_analysis(
            row=row,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            model_name=model_name,
            analysis_type=analysis_type,
            temperature=temperature,
            openrouter_api_key=openrouter_api_key,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_tokens=max_tokens,
        )

        if result is None:
            pass  # run_single_analysis already logs errors

        if verbose and pbar:
            pbar.update(1)
