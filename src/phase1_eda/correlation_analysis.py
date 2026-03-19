"""
Phase 1b — Pearson Correlation Analysis & Heatmap

Computes pairwise Pearson correlation coefficients for all dQ/dV features
and generates a publication-quality annotated heatmap.

Reference: Pearson (1895), Proc. Royal Society of London, 58, 240–242.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def compute_correlation_matrix(df: pd.DataFrame,
                               features: list[str]) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns.
    features : list[str]
        Column names to correlate.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    return df[features].corr(method="pearson")


def identify_strong_correlations(corr_matrix: pd.DataFrame,
                                  threshold: float = config.STRONG_CORR_THRESHOLD
                                  ) -> pd.DataFrame:
    """
    Extract feature pairs with |r| above threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    threshold : float
        Minimum |r| to flag as strong correlation.

    Returns
    -------
    pd.DataFrame
        Sorted table of strong correlations.
    """
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "r": r,
                    "abs_r": abs(r),
                })

    result = pd.DataFrame(pairs)
    if not result.empty:
        result = result.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return result


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              save_path=None) -> plt.Figure:
    """
    Generate annotated correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    save_path : Path or str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("dQ/dV Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Heatmap saved → {save_path}")

    return fig


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 1b: Correlation Analysis")
    print("=" * 40)

    df = pd.read_csv(config.SAMPLE_DATA)
    features = config.ALL_FEATURES

    corr = compute_correlation_matrix(df, features)
    strong = identify_strong_correlations(corr)

    print("\n--- Strong Correlations (|r| > {:.1f}) ---".format(
        config.STRONG_CORR_THRESHOLD))
    if strong.empty:
        print("None found.")
    else:
        print(strong.to_string(index=False))

    corr.to_csv(config.RESULTS_DIR / "correlation_matrix.csv")
    plot_correlation_heatmap(
        corr,
        save_path=config.FIGURES_DIR / f"correlation_heatmap.{config.FIGURE_FORMAT}",
    )
    plt.close("all")
    print(f"\nResults saved to {config.RESULTS_DIR}")
