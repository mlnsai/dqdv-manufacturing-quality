"""
Phase 4 — Composite Health Scoring & Quality Grading

Computes a health score based on primary peak intensities (CP3, DP2)
and assigns cells to three quality tiers (A/B/C) using percentile-based
thresholds.

Health Score = (CP3_intensity + DP2_intensity) / 2

Rationale: CP3 and DP2 represent the dominant capacity-delivering processes
(graphite Stage I formation and NMC cathode phase transitions). Averaging
charge and discharge contributions provides a balanced assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def compute_health_score(df: pd.DataFrame,
                          peaks: list[str] = config.HEALTH_PEAKS
                          ) -> pd.Series:
    """
    Compute composite health score as mean of specified peak intensities.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with intensity feature columns.
    peaks : list[str]
        Column names for the peaks to average.

    Returns
    -------
    pd.Series
        Health score per cell.
    """
    return df[peaks].mean(axis=1)


def assign_grades(health_scores: pd.Series,
                  percentiles: dict = config.GRADE_PERCENTILES) -> pd.Series:
    """
    Assign quality grades based on percentile thresholds.

    Grade A: above 66th percentile (top 33%)
    Grade B: 33rd–66th percentile (middle 33%)
    Grade C: below 33rd percentile (bottom 33%)

    Parameters
    ----------
    health_scores : pd.Series
        Health score per cell.
    percentiles : dict
        Grade boundaries {grade: lower_percentile}.

    Returns
    -------
    pd.Series
        Grade label per cell.
    """
    p66 = health_scores.quantile(percentiles["A"] / 100)
    p33 = health_scores.quantile(percentiles["B"] / 100)

    grades = pd.Series("C", index=health_scores.index)
    grades[health_scores >= p33] = "B"
    grades[health_scores >= p66] = "A"
    return grades


def apply_qc_thresholds(df: pd.DataFrame,
                        sigma_reject: int = config.QC_SIGMA_REJECT) -> pd.DataFrame:
    """
    Apply quality control acceptance/rejection thresholds.

    Rejection criteria:
        - health_score < (mean - sigma_reject * std)
        - high_risk flag from outlier detection (if column exists)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'health_score' column and optionally 'high_risk'.
    sigma_reject : int
        Number of standard deviations below mean for rejection.

    Returns
    -------
    pd.DataFrame
        Augmented with 'qc_status' column.
    """
    result = df.copy()
    mu = result["health_score"].mean()
    sigma = result["health_score"].std()
    threshold = mu - sigma_reject * sigma

    result["qc_status"] = "ACCEPT"
    result.loc[result["health_score"] < threshold, "qc_status"] = "REJECT"

    if "high_risk" in result.columns:
        result.loc[result["high_risk"] == True, "qc_status"] = "REJECT"

    return result


def plot_grade_distribution(df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Plot health score histogram colored by grade."""
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)

    colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
    for grade in ["A", "B", "C"]:
        subset = df[df["grade"] == grade]
        ax.hist(subset["health_score"], bins=15, alpha=0.7,
                color=colors[grade], label=f"Grade {grade} (n={len(subset)})",
                edgecolor="black")

    ax.axvline(df["health_score"].mean(), color="black", ls="--",
               label=f"Mean = {df['health_score'].mean():.4f}")
    ax.set_xlabel("Health Score (Ah/V)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Cell Quality Grading Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Grading plot saved → {save_path}")
    return fig


def generate_grade_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-grade summary statistics."""
    return df.groupby("grade")["health_score"].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(6)


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 4: Health Scoring & Quality Grading")
    print("=" * 45)

    df = pd.read_csv(config.SAMPLE_DATA)
    df["health_score"] = compute_health_score(df)
    df["grade"] = assign_grades(df["health_score"])
    df = apply_qc_thresholds(df)

    report = generate_grade_report(df)
    print("\n--- Grade Report ---")
    print(report.to_string())

    print(f"\n--- QC Summary ---")
    print(df["qc_status"].value_counts().to_string())

    df[["cell_id", "health_score", "grade", "qc_status"]].to_csv(
        config.RESULTS_DIR / "health_grades.csv", index=False
    )
    plot_grade_distribution(df, config.FIGURES_DIR / f"grade_distribution.{config.FIGURE_FORMAT}")
    plt.close("all")
