"""
Phase 2b — Charge–Discharge Asymmetry Metrics

Computes voltage hysteresis, intensity ratios, and reversibility metrics
for corresponding charge–discharge peak pairs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def compute_voltage_hysteresis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate voltage gap between corresponding charge and discharge peaks.

    ΔV_hyst = V_charge − V_discharge

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with voltage feature columns.

    Returns
    -------
    pd.DataFrame
        Original df augmented with hysteresis columns.
    """
    result = df.copy()
    for name, pair in config.PEAK_PAIRS.items():
        cp = pair["charge"]
        dp = pair["discharge"]
        col = f"hysteresis_{name}_mV"
        result[col] = (result[f"{cp}_voltage"] - result[f"{dp}_voltage"]) * 1000
    return result


def compute_intensity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate charge-to-discharge intensity ratio for each peak pair.

    R_intensity = I_charge / I_discharge
    Values near 1.0 indicate good reversibility.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with intensity feature columns.

    Returns
    -------
    pd.DataFrame
        Original df augmented with ratio columns.
    """
    result = df.copy()
    for name, pair in config.PEAK_PAIRS.items():
        cp = pair["charge"]
        dp = pair["discharge"]
        col = f"intensity_ratio_{name}"
        result[col] = result[f"{cp}_intensity"] / result[f"{dp}_intensity"]
    return result


def summarize_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all asymmetry metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hysteresis and ratio columns (from above functions).

    Returns
    -------
    pd.DataFrame
        Summary statistics for asymmetry metrics.
    """
    asym_cols = [c for c in df.columns if "hysteresis" in c or "intensity_ratio" in c]
    summary = df[asym_cols].describe().T
    summary["cv_pct"] = (summary["std"] / summary["mean"].abs() * 100)
    return summary


def plot_asymmetry(df: pd.DataFrame, save_dir=None) -> plt.Figure:
    """
    Plot hysteresis and intensity ratio distributions.
    """
    asym_cols = [c for c in df.columns if "hysteresis" in c or "intensity_ratio" in c]
    n = len(asym_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, asym_cols):
        ax.hist(df[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(df[col].mean(), color="red", ls="--", label=f"mean={df[col].mean():.2f}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("Charge–Discharge Asymmetry Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        path = save_dir / f"asymmetry_distributions.{config.FIGURE_FORMAT}"
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved → {path}")

    return fig


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 2b: Asymmetry Metrics")
    print("=" * 35)

    df = pd.read_csv(config.SAMPLE_DATA)
    df = compute_voltage_hysteresis(df)
    df = compute_intensity_ratios(df)

    summary = summarize_asymmetry(df)
    print("\n--- Asymmetry Summary ---")
    print(summary.round(4).to_string())

    summary.to_csv(config.RESULTS_DIR / "asymmetry_summary.csv")
    plot_asymmetry(df, save_dir=config.FIGURES_DIR)
    plt.close("all")
