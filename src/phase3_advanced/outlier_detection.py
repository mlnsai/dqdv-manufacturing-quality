"""
Phase 3c — Multi-Method Outlier Detection

Applies three complementary outlier detection approaches and flags cells
with consensus across methods.

Methods:
    1. Isolation Forest (Liu et al., 2008)
    2. IQR fences (Tukey, 1977)
    3. Z-score threshold (Grubbs, 1969)

Cells flagged by ≥ OUTLIER_CONSENSUS methods are marked high-risk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def isolation_forest_outliers(X: np.ndarray,
                               contamination: float = config.IF_CONTAMINATION
                               ) -> np.ndarray:
    """
    Detect outliers using Isolation Forest.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    contamination : float
        Expected proportion of outliers.

    Returns
    -------
    np.ndarray of bool
        True for outliers.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    preds = iso.fit_predict(X)
    return preds == -1  # -1 = outlier


def iqr_outliers(df: pd.DataFrame, features: list[str],
                 multiplier: float = config.IQR_MULTIPLIER) -> np.ndarray:
    """
    Detect outliers using IQR fences on each feature independently.
    A cell is an IQR outlier if ANY feature falls outside fences.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns.
    features : list[str]
        Columns to check.
    multiplier : float
        IQR fence multiplier (default 1.5).

    Returns
    -------
    np.ndarray of bool
        True for outliers.
    """
    is_outlier = np.zeros(len(df), dtype=bool)
    for feat in features:
        x = df[feat]
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        is_outlier |= (x < lower) | (x > upper)
    return is_outlier


def zscore_outliers(df: pd.DataFrame, features: list[str],
                    threshold: float = config.ZSCORE_THRESHOLD) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    A cell is a Z-score outlier if ANY feature has |z| > threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns.
    features : list[str]
        Columns to check.
    threshold : float
        Z-score threshold (default 3.0).

    Returns
    -------
    np.ndarray of bool
        True for outliers.
    """
    is_outlier = np.zeros(len(df), dtype=bool)
    for feat in features:
        x = df[feat]
        z = np.abs((x - x.mean()) / x.std())
        is_outlier |= (z > threshold)
    return is_outlier


def ensemble_outlier_detection(df: pd.DataFrame,
                                features: list[str],
                                consensus: int = config.OUTLIER_CONSENSUS
                                ) -> pd.DataFrame:
    """
    Run all three methods and compute consensus flags.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns.
    features : list[str]
        Columns to analyze.
    consensus : int
        Minimum number of methods agreeing to flag high-risk.

    Returns
    -------
    pd.DataFrame
        Original df augmented with outlier flags and risk level.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    result = df.copy()
    result["outlier_IF"] = isolation_forest_outliers(X)
    result["outlier_IQR"] = iqr_outliers(df, features)
    result["outlier_Zscore"] = zscore_outliers(df, features)

    result["outlier_count"] = (
        result["outlier_IF"].astype(int) +
        result["outlier_IQR"].astype(int) +
        result["outlier_Zscore"].astype(int)
    )
    result["high_risk"] = result["outlier_count"] >= consensus

    return result


def summarize_outliers(result: pd.DataFrame) -> dict:
    """
    Generate summary counts for outlier detection results.
    """
    return {
        "total_cells": len(result),
        "IF_outliers": result["outlier_IF"].sum(),
        "IQR_outliers": result["outlier_IQR"].sum(),
        "Zscore_outliers": result["outlier_Zscore"].sum(),
        "high_risk": result["high_risk"].sum(),
        "high_risk_pct": result["high_risk"].mean() * 100,
    }


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 3c: Outlier Detection")
    print("=" * 35)

    df = pd.read_csv(config.SAMPLE_DATA)
    result = ensemble_outlier_detection(df, config.ALL_FEATURES)
    summary = summarize_outliers(result)

    print("\n--- Outlier Detection Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if result["high_risk"].any():
        print("\n--- High-Risk Cells ---")
        print(result.loc[result["high_risk"], ["cell_id", "outlier_count"]].to_string(index=False))

    outlier_cols = ["cell_id", "outlier_IF", "outlier_IQR", "outlier_Zscore",
                    "outlier_count", "high_risk"]
    result[outlier_cols].to_csv(config.RESULTS_DIR / "outlier_flags.csv", index=False)
