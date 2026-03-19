#!/usr/bin/env python3
"""
run_pipeline.py — Execute the full four-phase dQ/dV analysis pipeline.

Usage:
    python run_pipeline.py                      # Run all phases
    python run_pipeline.py --phase 1            # Run only Phase 1
    python run_pipeline.py --data path/to.csv   # Use custom data file

All outputs are saved to data/results/ and figures/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import config

# ──────────────────────────────────────────────────
# Phase imports
# ──────────────────────────────────────────────────
from src.phase1_eda.descriptive_stats import compute_descriptive_stats, test_normality
from src.phase1_eda.correlation_analysis import (
    compute_correlation_matrix, identify_strong_correlations, plot_correlation_heatmap,
)
from src.phase2_characterization.peak_analysis import characterize_peaks, rank_manufacturing_sensitivity
from src.phase2_characterization.asymmetry_metrics import (
    compute_voltage_hysteresis, compute_intensity_ratios, summarize_asymmetry, plot_asymmetry,
)
from src.phase3_advanced.pca_analysis import run_pca, plot_scree, plot_pc_scatter
from src.phase3_advanced.clustering import evaluate_k_range, fit_optimal_kmeans, plot_k_selection
from src.phase3_advanced.outlier_detection import ensemble_outlier_detection, summarize_outliers
from src.phase3_advanced.correlation_network import (
    build_correlation_network, detect_communities, plot_network,
)
from src.phase4_grading.health_grading import (
    compute_health_score, assign_grades, apply_qc_thresholds,
    generate_grade_report, plot_grade_distribution,
)


def banner(text: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_phase1(df: pd.DataFrame):
    banner("PHASE 1: Exploratory Data Analysis")

    # 1a. Descriptive statistics
    desc = compute_descriptive_stats(df, config.ALL_FEATURES)
    desc.to_csv(config.RESULTS_DIR / "descriptive_statistics.csv")
    print("  ✓ Descriptive statistics computed")

    # 1b. Normality tests
    norm = test_normality(df, config.ALL_FEATURES)
    norm.to_csv(config.RESULTS_DIR / "normality_tests.csv")
    non_normal = norm[~norm["is_normal"]].index.tolist()
    print(f"  ✓ Shapiro-Wilk tests: {len(non_normal)} non-Gaussian features")

    # 1c. Correlation analysis
    corr = compute_correlation_matrix(df, config.ALL_FEATURES)
    corr.to_csv(config.RESULTS_DIR / "correlation_matrix.csv")
    strong = identify_strong_correlations(corr)
    print(f"  ✓ Correlation matrix: {len(strong)} strong pairs (|r| > {config.STRONG_CORR_THRESHOLD})")

    plot_correlation_heatmap(corr, config.FIGURES_DIR / f"correlation_heatmap.{config.FIGURE_FORMAT}")
    plt.close("all")

    return corr


def run_phase2(df: pd.DataFrame):
    banner("PHASE 2: Peak Characterization")

    # 2a. Peak statistics
    peak_stats = characterize_peaks(df)
    for group in ["voltage", "intensity"]:
        peak_stats[group].to_csv(config.RESULTS_DIR / f"peak_{group}_stats.csv", index=False)
    sensitivity = rank_manufacturing_sensitivity(peak_stats)
    sensitivity.to_csv(config.RESULTS_DIR / "sensitivity_ranking.csv", index=False)
    print(f"  ✓ Peak stats computed, most sensitive: {sensitivity.iloc[0]['feature']}")

    # 2b. Asymmetry metrics
    df_asym = compute_voltage_hysteresis(df)
    df_asym = compute_intensity_ratios(df_asym)
    summary = summarize_asymmetry(df_asym)
    summary.to_csv(config.RESULTS_DIR / "asymmetry_summary.csv")
    print("  ✓ Asymmetry metrics computed")

    plot_asymmetry(df_asym, save_dir=config.FIGURES_DIR)
    plt.close("all")


def run_phase3(df: pd.DataFrame, corr: pd.DataFrame = None):
    banner("PHASE 3: Advanced Analytics")

    features = config.ALL_FEATURES
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # 3a. PCA
    pca_result = run_pca(df, features)
    pca_result["scores"].to_csv(config.RESULTS_DIR / "pca_scores.csv", index=False)
    pca_result["loadings"].to_csv(config.RESULTS_DIR / "pca_loadings.csv")
    n90 = int(np.searchsorted(pca_result["cumulative_variance"], 0.90)) + 1
    print(f"  ✓ PCA: {n90} PCs explain ≥90% variance")

    plot_scree(pca_result, config.FIGURES_DIR / f"pca_scree.{config.FIGURE_FORMAT}")

    # 3b. Clustering
    metrics = evaluate_k_range(X)
    metrics.to_csv(config.RESULTS_DIR / "k_selection_metrics.csv", index=False)
    optimal_k = int(metrics.loc[metrics["silhouette"].idxmax(), "K"])
    km_result = fit_optimal_kmeans(X, optimal_k)
    df["cluster"] = km_result["labels"]
    print(f"  ✓ K-means: optimal K={optimal_k}, silhouette={km_result['silhouette_score']:.4f}")

    plot_k_selection(metrics, config.FIGURES_DIR / f"k_selection.{config.FIGURE_FORMAT}")
    plot_pc_scatter(pca_result, labels=km_result["labels"],
                    save_path=config.FIGURES_DIR / f"pca_clusters.{config.FIGURE_FORMAT}")

    # 3c. Outlier detection
    df_outliers = ensemble_outlier_detection(df, features)
    outlier_summary = summarize_outliers(df_outliers)
    outlier_cols = ["cell_id", "outlier_IF", "outlier_IQR", "outlier_Zscore",
                    "outlier_count", "high_risk"]
    df_outliers[outlier_cols].to_csv(config.RESULTS_DIR / "outlier_flags.csv", index=False)
    print(f"  ✓ Outliers: {outlier_summary['high_risk']} high-risk cells "
          f"({outlier_summary['high_risk_pct']:.1f}%)")

    # 3d. Correlation network
    if corr is None:
        corr = df[features].corr()
    G = build_correlation_network(corr)
    partition = detect_communities(G)
    print(f"  ✓ Network: {G.number_of_edges()} edges, "
          f"{len(set(partition.values()))} communities")

    plot_network(G, partition, config.FIGURES_DIR / f"correlation_network.{config.FIGURE_FORMAT}")
    plt.close("all")

    return df_outliers


def run_phase4(df: pd.DataFrame):
    banner("PHASE 4: Health Scoring & Quality Grading")

    df["health_score"] = compute_health_score(df)
    df["grade"] = assign_grades(df["health_score"])
    df = apply_qc_thresholds(df)

    report = generate_grade_report(df)
    report.to_csv(config.RESULTS_DIR / "grade_report.csv")

    output_cols = ["cell_id", "health_score", "grade", "qc_status"]
    if "high_risk" in df.columns:
        output_cols.insert(3, "high_risk")
    if "cluster" in df.columns:
        output_cols.insert(2, "cluster")
    df[output_cols].to_csv(config.RESULTS_DIR / "health_grades.csv", index=False)

    n_reject = (df["qc_status"] == "REJECT").sum()
    print(f"  ✓ Grades: A={len(df[df['grade']=='A'])}, "
          f"B={len(df[df['grade']=='B'])}, C={len(df[df['grade']=='C'])}")
    print(f"  ✓ QC: {n_reject} cells rejected, {len(df) - n_reject} accepted")

    plot_grade_distribution(df, config.FIGURES_DIR / f"grade_distribution.{config.FIGURE_FORMAT}")
    plt.close("all")

    return df


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="dQ/dV Manufacturing Quality Analysis Pipeline"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to input CSV (default: sample data)")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Run only this phase (default: all)")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else config.SAMPLE_DATA
    print(f"\n📊 dQ/dV Manufacturing Quality Analysis")
    print(f"   Data: {data_path}")
    print(f"   Output: {config.RESULTS_DIR}")
    print(f"   Figures: {config.FIGURES_DIR}")

    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} cells × {len(config.ALL_FEATURES)} features")

    t0 = time.time()

    if args.phase is None or args.phase == 1:
        corr = run_phase1(df)
    else:
        corr = None

    if args.phase is None or args.phase == 2:
        run_phase2(df)

    if args.phase is None or args.phase == 3:
        df = run_phase3(df, corr)

    if args.phase is None or args.phase == 4:
        df = run_phase4(df)

    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"\n  Results → {config.RESULTS_DIR}/")
    print(f"  Figures → {config.FIGURES_DIR}/")
    print()


if __name__ == "__main__":
    main()
