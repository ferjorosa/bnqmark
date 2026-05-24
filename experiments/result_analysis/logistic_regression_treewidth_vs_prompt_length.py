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
TREEWIDTH_COLUMN = "target_tw"  # Options: "achieved_tw", "target_tw"
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
        ["bn_uuid", "n", TREEWIDTH_COLUMN]
    ].copy()

    df = experiments_df.merge(queries_df, on="query_uuid", how="inner").merge(
        bns_df, on="bn_uuid", how="inner"
    )

    df["treewidth"] = df[TREEWIDTH_COLUMN]

    df["answerable"] = df["llm_probability"].notna().astype(int)
    df["abs_error"] = (df["llm_probability"] - df["probability"]).abs()
    df["correct"] = (
        df["llm_probability"].notna() & (df["abs_error"] <= ACCURACY_THRESHOLD)
    ).astype(int)

    df["z_treewidth"] = standardize(df["treewidth"].astype(float))
    df["z_input_tokens"] = standardize(df["input_tokens"].astype(float))
    df["z_n"] = standardize(df["n"].astype(float))

    return df


def fit_logistic_model(df: pd.DataFrame, outcome: str):
    """Fit a pooled logistic regression with model fixed effects."""
    formula = f"{outcome} ~ z_treewidth + z_input_tokens + z_n + C(model_name)"
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


def compute_mcfadden_r2(result) -> float:
    """Compute McFadden pseudo-R^2 for a fitted logistic model."""
    llf = getattr(result, "llf", np.nan)
    llnull = getattr(result, "llnull", np.nan)
    if pd.notna(llf) and pd.notna(llnull) and llnull != 0:
        return 1 - (llf / llnull)

    null_deviance = getattr(result, "null_deviance", np.nan)
    deviance = getattr(result, "deviance", np.nan)
    if pd.notna(null_deviance) and pd.notna(deviance) and null_deviance != 0:
        return 1 - (deviance / null_deviance)

    return np.nan


def compute_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Compute ROC AUC from binary labels and fitted probabilities."""
    y_true_array = y_true.to_numpy(dtype=int)
    positive_count = int(y_true_array.sum())
    negative_count = len(y_true_array) - positive_count
    if positive_count == 0 or negative_count == 0:
        return np.nan

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positive_rank_sum = ranks[y_true_array == 1].sum()
    u_statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return float(u_statistic / (positive_count * negative_count))


def fit_diagnostics(result, df: pd.DataFrame, outcome: str) -> dict[str, float]:
    """Compute compact in-sample fit diagnostics."""
    y_true = df[outcome].astype(int)
    y_score = result.predict(df)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "mcfadden_r2": compute_mcfadden_r2(result),
        "auc": compute_auc(y_true, y_score),
        "brier": float(np.mean((y_score - y_true) ** 2)),
        "classification_accuracy": float(np.mean(y_pred == y_true)),
    }


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

    summary = df[["treewidth", "input_tokens", "n"]].agg(["mean", "std", "min", "max"])
    print("Predictor summary")
    print("-----------------")
    print(summary.round(3).to_string())
    print()

    print("Predictor correlations")
    print("----------------------")
    print(df[["treewidth", "input_tokens", "n"]].corr().round(3).to_string())
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


def print_model_results(name: str, result, df: pd.DataFrame, outcome: str) -> None:
    """Print compact regression output."""
    print(name)
    print("-" * len(name))
    print(
        coefficient_table(result)
        .round({"coef": 3, "std_err": 3, "p_value": 4, "odds_ratio": 3})
        .to_string()
    )
    print()
    diagnostics = fit_diagnostics(result, df, outcome)
    print("Fit diagnostics")
    print("---------------")
    print(f"McFadden pseudo-R^2: {diagnostics['mcfadden_r2']:.3f}")
    if pd.isna(diagnostics["auc"]):
        print("AUC: n/a (outcome has a single class)")
    else:
        print(f"AUC: {diagnostics['auc']:.3f}")
    print(f"Brier score: {diagnostics['brier']:.3f}")
    print(f"Acc (0.5 threshold): {diagnostics['classification_accuracy']:.3f}")
    print()
    print(interpret_predictor(result, "z_treewidth", "Treewidth"))
    print(interpret_predictor(result, "z_input_tokens", "Input tokens"))
    print(interpret_predictor(result, "z_n", "Network size (n)"))
    print()


def main() -> None:
    """Run the pooled logistic-regression analysis."""
    print(f"Treewidth column: {TREEWIDTH_COLUMN}")
    df = build_analysis_df()
    print_dataset_summary(df)

    answerable_result = fit_logistic_model(df, "answerable")
    correct_result = fit_logistic_model(df, "correct")

    print_model_results("Answerability regression", answerable_result, df, "answerable")
    print_model_results("Accuracy regression", correct_result, df, "correct")


if __name__ == "__main__":
    main()
