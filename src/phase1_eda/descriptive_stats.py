"""
Phase 1a — Descriptive Statistics & Normality Testing

Computes comprehensive statistical profiling for all dQ/dV peak features
and applies the Shapiro-Wilk test to assess normality.

Reference: Shapiro & Wilk (1965), Biometrika, 52(3-4), 591–611.
"""

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def compute_descriptive_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with cell_id and feature columns.
    features : list[str]
        Column names to analyze.

    Returns
    -------
    pd.DataFrame
        Statistics table indexed by feature name.
    """
    records = []
    for feat in features:
        x = df[feat].dropna()
        record = {
            "feature": feat,
            "n": len(x),
            "mean": x.mean(),
            "median": x.median(),
            "std": x.std(),
            "variance": x.var(),
            "cv_pct": (x.std() / x.mean() * 100) if x.mean() != 0 else np.nan,
            "min": x.min(),
            "Q1": x.quantile(0.25),
            "Q2": x.quantile(0.50),
            "Q3": x.quantile(0.75),
            "max": x.max(),
            "iqr": x.quantile(0.75) - x.quantile(0.25),
            "skewness": x.skew(),
            "kurtosis": x.kurtosis(),
        }
        records.append(record)

    return pd.DataFrame(records).set_index("feature")


def test_normality(df: pd.DataFrame, features: list[str],
                   alpha: float = config.NORMALITY_ALPHA) -> pd.DataFrame:
    """
    Apply Shapiro-Wilk normality test to each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns.
    features : list[str]
        Column names to test.
    alpha : float
        Significance level (default from config).

    Returns
    -------
    pd.DataFrame
        Test results with W-statistic, p-value, and normality flag.
    """
    results = []
    for feat in features:
        x = df[feat].dropna().values
        w_stat, p_value = stats.shapiro(x)
        results.append({
            "feature": feat,
            "W_statistic": w_stat,
            "p_value": p_value,
            "is_normal": p_value >= alpha,
        })

    return pd.DataFrame(results).set_index("feature")


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 1a: Descriptive Statistics & Normality Testing")
    print("=" * 55)

    df = pd.read_csv(config.SAMPLE_DATA)
    features = config.ALL_FEATURES

    # Descriptive statistics
    desc = compute_descriptive_stats(df, features)
    print("\n--- Descriptive Statistics ---")
    print(desc.round(6).to_string())

    # Normality tests
    norm = test_normality(df, features)
    print("\n--- Shapiro-Wilk Normality Test ---")
    print(norm.to_string())

    non_normal = norm[~norm["is_normal"]].index.tolist()
    if non_normal:
        print(f"\nNon-Gaussian features (p < {config.NORMALITY_ALPHA}): {non_normal}")
    else:
        print("\nAll features consistent with normal distribution.")

    # Save results
    desc.to_csv(config.RESULTS_DIR / "descriptive_statistics.csv")
    norm.to_csv(config.RESULTS_DIR / "normality_tests.csv")
    print(f"\nResults saved to {config.RESULTS_DIR}")
