"""
Phase 3a — Principal Component Analysis (PCA)

Reduces the 10-dimensional dQ/dV feature space while preserving
maximum variance. Features are Z-score normalized prior to decomposition.

Reference: Jolliffe (2002), Principal Component Analysis, Springer.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def run_pca(df: pd.DataFrame, features: list[str],
            n_components: int | None = None) -> dict:
    """
    Perform PCA on Z-score normalized features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns.
    features : list[str]
        Column names to include in PCA.
    n_components : int, optional
        Number of components to retain. None = all.

    Returns
    -------
    dict with keys:
        'pca': fitted PCA object
        'scaler': fitted StandardScaler
        'scores': pd.DataFrame of PC scores per cell
        'loadings': pd.DataFrame of feature loadings per PC
        'variance_explained': array of explained variance ratios
        'cumulative_variance': array of cumulative explained variance
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    if n_components is None:
        n_components = len(features)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    pc_names = [f"PC{i+1}" for i in range(pca.n_components_)]

    scores_df = pd.DataFrame(scores, columns=pc_names, index=df.index)
    if "cell_id" in df.columns:
        scores_df.insert(0, "cell_id", df["cell_id"].values)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=pc_names,
    )

    return {
        "pca": pca,
        "scaler": scaler,
        "scores": scores_df,
        "loadings": loadings_df,
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    }


def plot_scree(result: dict, save_path=None) -> plt.Figure:
    """Plot scree plot with cumulative variance line."""
    var = result["variance_explained"]
    cum = result["cumulative_variance"]
    n = len(var)

    fig, ax1 = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    ax1.bar(range(1, n + 1), var * 100, color="steelblue", alpha=0.8,
            label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Variance Explained (%)", fontsize=12)
    ax1.set_xticks(range(1, n + 1))

    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), cum * 100, "ro-", label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12)
    ax2.axhline(90, color="gray", ls="--", alpha=0.5)

    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88))
    plt.title("PCA Scree Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Scree plot saved → {save_path}")
    return fig


def plot_pc_scatter(result: dict, pc_x: int = 1, pc_y: int = 2,
                    labels=None, save_path=None) -> plt.Figure:
    """Plot 2D scatter of PC scores."""
    scores = result["scores"]
    var = result["variance_explained"]

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    scatter = ax.scatter(
        scores[f"PC{pc_x}"], scores[f"PC{pc_y}"],
        c=labels if labels is not None else "steelblue",
        cmap="Set1" if labels is not None else None,
        edgecolors="k", linewidth=0.3, alpha=0.7, s=50,
    )
    ax.set_xlabel(f"PC{pc_x} ({var[pc_x-1]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC{pc_y} ({var[pc_y-1]*100:.1f}%)", fontsize=12)
    ax.set_title("PCA Score Plot", fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.axvline(0, color="gray", ls="--", alpha=0.3)

    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 3a: PCA Analysis")
    print("=" * 30)

    df = pd.read_csv(config.SAMPLE_DATA)
    result = run_pca(df, config.ALL_FEATURES)

    print("\n--- Variance Explained ---")
    for i, (v, c) in enumerate(zip(result["variance_explained"],
                                    result["cumulative_variance"])):
        print(f"  PC{i+1}: {v*100:6.2f}%  (cumulative: {c*100:6.2f}%)")

    print("\n--- Loadings (top 3 PCs) ---")
    print(result["loadings"].iloc[:, :3].round(3).to_string())

    result["scores"].to_csv(config.RESULTS_DIR / "pca_scores.csv", index=False)
    result["loadings"].to_csv(config.RESULTS_DIR / "pca_loadings.csv")
    plot_scree(result, config.FIGURES_DIR / f"pca_scree.{config.FIGURE_FORMAT}")
    plot_pc_scatter(result, save_path=config.FIGURES_DIR / f"pca_scatter.{config.FIGURE_FORMAT}")
    plt.close("all")
