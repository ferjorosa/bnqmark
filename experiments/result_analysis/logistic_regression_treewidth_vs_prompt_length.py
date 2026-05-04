#!/usr/bin/env python3
"""Pooled logistic regressions for treewidth vs prompt length."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial

repo_root = Path(__file__).resolve().parents[2]

# Configuration
NAMING_STRATEGY = "simple"
EXPERIMENT_TYPE = "code_generation"
ACCURACY_THRESHOLD = 0.01


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load experiments, queries, and BN metadata from parquet."""
    data_dir = repo_root / "data"
    experiments_df = pd.read_parquet(data_dir / "experiments.parquet")
    queries_df = pd.read_parquet(data_dir / "queries.parquet")
    bns_df = pd.read_parquet(data_dir / "bns.parquet")
    return experiments_df, queries_df, bns_df


def standardize(series: pd.Series) -> pd.Series:
    """Z-score standardization with population std."""
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def build_analysis_df() -> pd.DataFrame:
    """Create one supported query x model row for raw reasoning."""
    experiments_df, queries_df, bns_df = load_data()

    experiments_df = experiments_df[
        (experiments_df["naming_strategy"] == NAMING_STRATEGY)
        & (experiments_df["experiment_type"] == EXPERIMENT_TYPE)
        & (experiments_df["llm_probability"] != -1000)
    ].copy()

    queries_df = queries_df[queries_df["naming_strategy"] == NAMING_STRATEGY][
        ["query_uuid", "bn_uuid", "probability"]
    ].copy()

    bns_df = bns_df[bns_df["naming_strategy"] == NAMING_STRATEGY][
        ["bn_uuid", "n", "achieved_tw"]
    ].copy()

    df = experiments_df.merge(queries_df, on="query_uuid", how="inner").merge(
        bns_df, on="bn_uuid", how="inner"
    )

    df["answerable"] = df["llm_probability"].notna().astype(int)
    df["abs_error"] = (df["llm_probability"] - df["probability"]).abs()
    df["correct"] = (
        df["llm_probability"].notna() & (df["abs_error"] <= ACCURACY_THRESHOLD)
    ).astype(int)

    df["z_achieved_tw"] = standardize(df["achieved_tw"].astype(float))
    df["z_input_tokens"] = standardize(df["input_tokens"].astype(float))
    df["z_n"] = standardize(df["n"].astype(float))

    return df


def fit_logistic_model(df: pd.DataFrame, outcome: str):
    """Fit a pooled logistic regression with model fixed effects."""
    formula = f"{outcome} ~ z_achieved_tw + z_input_tokens + z_n + C(model_name)"
    model = smf.glm(formula=formula, data=df, family=Binomial())
    return model.fit()


def coefficient_table(result) -> pd.DataFrame:
    """Build a compact coefficient table with odds ratios."""
    table = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "p_value": result.pvalues,
            "odds_ratio": np.exp(result.params),
        }
    )
    return table


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print compact dataset diagnostics."""
    models = sorted(df["model_name"].unique().tolist())
    print("Dataset summary")
    print("---------------")
    print(f"Rows: {len(df)}")
    print(f"Unique queries: {df['query_uuid'].nunique()}")
    print(f"Unique models: {len(models)}")
    print("Models:")
    for model in models:
        print(f"  - {model}")
    print()

    summary = df[["achieved_tw", "input_tokens", "n"]].agg(
        ["mean", "std", "min", "max"]
    )
    print("Predictor summary")
    print("-----------------")
    print(summary.round(3).to_string())
    print()

    print("Predictor correlations")
    print("----------------------")
    print(df[["achieved_tw", "input_tokens", "n"]].corr().round(3).to_string())
    print()


def interpret_predictor(result, predictor: str, label: str) -> str:
    """Return a short plain-English interpretation for one predictor."""
    coef = result.params[predictor]
    p_value = result.pvalues[predictor]
    direction = "negative" if coef < 0 else "positive"
    strength = "statistically credible" if p_value < 0.05 else "not statistically clear"
    return (
        f"{label}: coefficient is {direction} ({coef:.3f}), "
        f"odds ratio = {np.exp(coef):.3f}, p = {p_value:.4g} -> {strength}."
    )


def print_model_results(name: str, result) -> None:
    """Print compact regression output."""
    print(name)
    print("-" * len(name))
    print(
        coefficient_table(result)
        .round({"coef": 3, "std_err": 3, "p_value": 4, "odds_ratio": 3})
        .to_string()
    )
    print()
    print(interpret_predictor(result, "z_achieved_tw", "Treewidth"))
    print(interpret_predictor(result, "z_input_tokens", "Input tokens"))
    print()


def main() -> None:
    """Run the pooled logistic-regression analysis."""
    df = build_analysis_df()
    print_dataset_summary(df)

    answerable_result = fit_logistic_model(df, "answerable")
    correct_result = fit_logistic_model(df, "correct")

    print_model_results("Answerability regression", answerable_result)
    print_model_results("Accuracy regression", correct_result)


if __name__ == "__main__":
    main()
