#!/usr/bin/env python3
"""Pooled logistic regressions for accuracy and answerability."""

import argparse
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
PREDICTORS = {
    "z_treewidth": "Treewidth",
    "z_input_tokens": "Input tokens",
    "z_n": "Network size (n)",
    "z_total_factor_size": "Total factor size",
}


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load experiments, queries, and BN metadata from parquet."""
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


def build_analysis_df(data_dir: Path = repo_root / "data") -> pd.DataFrame:
    """Create one supported query x model row for the configured experiment."""
    experiments_df, queries_df, bns_df = load_data(data_dir)

    experiments_df = experiments_df[
        (experiments_df["naming_strategy"] == NAMING_STRATEGY)
        & (experiments_df["experiment_type"] == EXPERIMENT_TYPE)
        & (experiments_df["llm_probability"] != -1000)
    ].copy()

    queries_df = queries_df[queries_df["naming_strategy"] == NAMING_STRATEGY].copy()
    if "total_factor_size" not in queries_df.columns:
        raise KeyError(
            "queries.parquet is missing 'total_factor_size'. Run "
            "experiments/generate_data/enrich_existing_query_complexity.py first."
        )
    queries_df = queries_df[
        ["query_uuid", "bn_uuid", "probability", "total_factor_size"]
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
    df["z_total_factor_size"] = standardize(df["total_factor_size"].astype(float))

    return df


def fit_logistic_model(
    df: pd.DataFrame,
    outcome: str,
    *,
    include_model_fixed_effects: bool = True,
):
    """Fit a logistic regression with optional model fixed effects."""
    formula = f"{outcome} ~ {' + '.join(PREDICTORS)}"
    if include_model_fixed_effects:
        formula += " + C(model_name)"
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
    outcome_rate = float(y_true.mean())
    return {
        "outcome_rate": outcome_rate,
        "mcfadden_r2": compute_mcfadden_r2(result),
        "auc": compute_auc(y_true, y_score),
        "brier": float(np.mean((y_score - y_true) ** 2)),
        "classification_accuracy": float(np.mean(y_pred == y_true)),
        "majority_accuracy": max(outcome_rate, 1 - outcome_rate),
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

    predictor_columns = ["treewidth", "input_tokens", "n", "total_factor_size"]
    summary = df[predictor_columns].agg(["mean", "std", "min", "max"])
    print("Predictor summary")
    print("-----------------")
    print(summary.round(3).to_string())
    print()

    print("Predictor correlations")
    print("----------------------")
    print(df[predictor_columns].corr().round(3).to_string())
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
    print(f"Outcome rate: {diagnostics['outcome_rate']:.3f}")
    print(f"Majority-class acc: {diagnostics['majority_accuracy']:.3f}")
    print(f"McFadden pseudo-R^2: {diagnostics['mcfadden_r2']:.3f}")
    if pd.isna(diagnostics["auc"]):
        print("AUC: n/a (outcome has a single class)")
    else:
        print(f"AUC: {diagnostics['auc']:.3f}")
    print(f"Brier score: {diagnostics['brier']:.3f}")
    print(f"Acc (0.5 threshold): {diagnostics['classification_accuracy']:.3f}")
    print()
    for predictor, label in PREDICTORS.items():
        print(interpret_predictor(result, predictor, label))
    print()


def build_per_model_tables(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one logistic model per model_name and return diagnostics and coefficients."""
    diagnostic_rows = []
    coefficient_rows = []

    for model_name in sorted(df["model_name"].unique()):
        model_df = df[df["model_name"] == model_name].copy()
        y_true = model_df[outcome].astype(int)
        if y_true.nunique() < 2:
            outcome_rate = float(y_true.mean())
            diagnostic_rows.append(
                {
                    "model_name": model_name,
                    "rows": len(model_df),
                    "outcome_rate": outcome_rate,
                    "majority_accuracy": max(outcome_rate, 1 - outcome_rate),
                    "mcfadden_r2": np.nan,
                    "auc": np.nan,
                    "brier": np.nan,
                    "classification_accuracy": np.nan,
                    "status": "single outcome class",
                }
            )
            continue

        result = fit_logistic_model(
            model_df,
            outcome,
            include_model_fixed_effects=False,
        )
        diagnostics = fit_diagnostics(result, model_df, outcome)
        diagnostic_rows.append(
            {
                "model_name": model_name,
                "rows": len(model_df),
                "outcome_rate": diagnostics["outcome_rate"],
                "majority_accuracy": diagnostics["majority_accuracy"],
                "mcfadden_r2": diagnostics["mcfadden_r2"],
                "auc": diagnostics["auc"],
                "brier": diagnostics["brier"],
                "classification_accuracy": diagnostics["classification_accuracy"],
                "status": "ok",
            }
        )

        coefficients = coefficient_table(result)
        for predictor, label in PREDICTORS.items():
            coefficient_rows.append(
                {
                    "model_name": model_name,
                    "predictor": label,
                    "coef": coefficients.loc[predictor, "coef"],
                    "odds_ratio": coefficients.loc[predictor, "odds_ratio"],
                    "p_value": coefficients.loc[predictor, "p_value"],
                }
            )

    return pd.DataFrame(diagnostic_rows), pd.DataFrame(coefficient_rows)


def print_per_model_results(title: str, diagnostics: pd.DataFrame, coefs: pd.DataFrame):
    """Print compact per-model logistic regression summaries."""
    print(title)
    print("-" * len(title))
    print("Diagnostics")
    print("-----------")
    diagnostic_columns = [
        "model_name",
        "rows",
        "outcome_rate",
        "majority_accuracy",
        "classification_accuracy",
        "auc",
        "mcfadden_r2",
        "brier",
        "status",
    ]
    print(
        diagnostics[diagnostic_columns]
        .round(
            {
                "outcome_rate": 3,
                "majority_accuracy": 3,
                "classification_accuracy": 3,
                "auc": 3,
                "mcfadden_r2": 3,
                "brier": 3,
            }
        )
        .to_string(index=False)
    )
    print()

    if coefs.empty:
        print(
            "No per-model coefficient tables; "
            "every model had a single outcome class."
        )
        print()
        return

    print("Predictor coefficients")
    print("----------------------")
    print(
        coefs.round({"coef": 3, "odds_ratio": 3, "p_value": 4}).to_string(index=False)
    )
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help=(
            "Directory containing experiments.parquet, queries.parquet, "
            "and bns.parquet."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the pooled logistic-regression analysis."""
    args = parse_args()
    data_dir = args.data_dir

    print(f"Treewidth column: {TREEWIDTH_COLUMN}")
    print(f"Data directory: {data_dir}")
    df = build_analysis_df(data_dir)
    print_dataset_summary(df)

    answerable_result = fit_logistic_model(df, "answerable")
    correct_result = fit_logistic_model(df, "correct")

    print_model_results("Answerability regression", answerable_result, df, "answerable")
    print_model_results("Accuracy regression", correct_result, df, "correct")

    answerable_diagnostics, answerable_coefs = build_per_model_tables(
        df,
        "answerable",
    )
    correct_diagnostics, correct_coefs = build_per_model_tables(df, "correct")

    print_per_model_results(
        "Per-model answerability regressions",
        answerable_diagnostics,
        answerable_coefs,
    )
    print_per_model_results(
        "Per-model accuracy regressions",
        correct_diagnostics,
        correct_coefs,
    )


if __name__ == "__main__":
    main()
