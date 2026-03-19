"""
Phase 3b — K-Means Clustering with Optimal K Selection

Identifies natural cell groupings using K-means with three complementary
metrics for optimal cluster selection: elbow method, silhouette score,
and Davies-Bouldin index.

References:
    - MacQueen (1967), Proc. 5th Berkeley Symposium.
    - Rousseeuw (1987), J. Computational and Applied Mathematics, 20, 53–65.
    - Davies & Bouldin (1979), IEEE TPAMI, 1(2), 224–227.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def evaluate_k_range(X: np.ndarray,
                     k_range=config.K_RANGE) -> pd.DataFrame:
    """
    Evaluate clustering quality across a range of K values.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix (n_samples × n_features).
    k_range : range
        Range of K values to test.

    Returns
    -------
    pd.DataFrame
        Metrics for each K: inertia, silhouette, Davies-Bouldin.
    """
    records = []
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=42,
        )
        labels = km.fit_predict(X)
        records.append({
            "K": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })

    return pd.DataFrame(records)


def fit_optimal_kmeans(X: np.ndarray, k: int) -> dict:
    """
    Fit K-means with the chosen optimal K.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    k : int
        Number of clusters.

    Returns
    -------
    dict with keys: 'model', 'labels', 'centroids', 'silhouette_score'
    """
    km = KMeans(
        n_clusters=k,
        n_init=config.KMEANS_N_INIT,
        max_iter=config.KMEANS_MAX_ITER,
        random_state=42,
    )
    labels = km.fit_predict(X)
    return {
        "model": km,
        "labels": labels,
        "centroids": km.cluster_centers_,
        "silhouette_score": silhouette_score(X, labels),
    }


def plot_k_selection(metrics: pd.DataFrame, save_path=None) -> plt.Figure:
    """Plot elbow, silhouette, and Davies-Bouldin metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(metrics["K"], metrics["inertia"], "bo-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(metrics["K"], metrics["silhouette"], "go-")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")

    best_k = metrics.loc[metrics["silhouette"].idxmax(), "K"]
    axes[1].axvline(best_k, color="red", ls="--", alpha=0.7,
                    label=f"Best K = {best_k}")
    axes[1].legend()

    axes[2].plot(metrics["K"], metrics["davies_bouldin"], "ro-")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].set_title("Davies-Bouldin Index")

    plt.suptitle("Optimal K Selection", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"K-selection plot saved → {save_path}")
    return fig


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 3b: K-Means Clustering")
    print("=" * 35)

    df = pd.read_csv(config.SAMPLE_DATA)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[config.ALL_FEATURES])

    metrics = evaluate_k_range(X)
    print("\n--- K Selection Metrics ---")
    print(metrics.round(4).to_string(index=False))

    optimal_k = int(metrics.loc[metrics["silhouette"].idxmax(), "K"])
    print(f"\nOptimal K (max silhouette): {optimal_k}")

    result = fit_optimal_kmeans(X, optimal_k)
    print(f"Silhouette score: {result['silhouette_score']:.4f}")

    df["cluster"] = result["labels"]
    print("\n--- Cluster Sizes ---")
    print(df["cluster"].value_counts().sort_index().to_string())

    metrics.to_csv(config.RESULTS_DIR / "k_selection_metrics.csv", index=False)
    df[["cell_id", "cluster"]].to_csv(
        config.RESULTS_DIR / "cluster_assignments.csv", index=False
    )
    plot_k_selection(metrics, config.FIGURES_DIR / f"k_selection.{config.FIGURE_FORMAT}")
    plt.close("all")
