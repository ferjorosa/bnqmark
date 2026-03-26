"""Module for fetching experiments for trace analysis."""

import json
from typing import Any

from src.database.database import query_db


def fetch_experiments(model_name: str, experiment_type: str, run: int) -> Any:
    """
    Fetch experiments for a model and experiment type, filtered by run.

    Includes query and CPT details.

    Args:
        model_name: The name of the model to fetch experiments for.
        experiment_type: The type of experiment (e.g., 'code_generation',
            'raw_reasoning').
        run: Run number to fetch.

    Returns:
        DataFrame containing the experiments with joined query and BN information.
    """
    query = """
        SELECT
            e.query_uuid,
            e.naming_strategy,
            e.run,
            e.model_name,
            e.experiment_type,
            e.response,
            e.response_reasoning_summary,
            q.target,
            q.evidence,
            bn.bn_pickle
        FROM research_probabilistic_reasoning.discrete_experiments e
        JOIN research_probabilistic_reasoning.discrete_queries q
            ON e.query_uuid = q.query_uuid AND e.naming_strategy = q.naming_strategy
        JOIN research_probabilistic_reasoning.discrete_bns bn
            ON q.bn_uuid = bn.bn_uuid AND q.naming_strategy = bn.naming_strategy
        WHERE e.model_name = %s
          AND e.experiment_type = %s
          AND e.run = %s
          AND e.llm_probability != -1000
    """
    df = query_db(query, [model_name, experiment_type, run])

    # Deserialize JSONB strings back to dictionaries for target and evidence
    df["target"] = df["target"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x,
    )
    df["evidence"] = df["evidence"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if x else {}),
    )

    return df
