"""
Phase 2a — Peak Voltage & Intensity Characterization

Quantifies population statistics for each dQ/dV peak and identifies
features with greatest manufacturing sensitivity (highest CV).
"""

import pandas as pd

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import config


def characterize_peaks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute population statistics for voltage positions and intensities
    of each dQ/dV peak.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with dQ/dV feature columns.

    Returns
    -------
    dict with keys 'voltage' and 'intensity', each a DataFrame of
    per-peak statistics.
    """
    results = {}

    for group_name, features in [
        ("voltage", config.VOLTAGE_FEATURES),
        ("intensity", config.INTENSITY_FEATURES),
    ]:
        records = []
        for feat in features:
            x = df[feat].dropna()
            peak_name = feat.split("_")[0]  # e.g. "CP1"
            records.append({
                "peak": peak_name,
                "feature": feat,
                "mean": x.mean(),
                "std": x.std(),
                "cv_pct": (x.std() / x.mean() * 100) if x.mean() != 0 else float("nan"),
                "min": x.min(),
                "max": x.max(),
                "range": x.max() - x.min(),
                "range_mV": (x.max() - x.min()) * 1000 if group_name == "voltage" else None,
            })
        results[group_name] = pd.DataFrame(records)

    return results


def rank_manufacturing_sensitivity(peak_stats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Rank features by coefficient of variation (CV) — higher CV indicates
    greater manufacturing variability.

    Parameters
    ----------
    peak_stats : dict
        Output of characterize_peaks().

    Returns
    -------
    pd.DataFrame
        All features ranked by CV descending.
    """
    combined = pd.concat(
        [peak_stats["voltage"], peak_stats["intensity"]],
        ignore_index=True,
    )
    return combined.sort_values("cv_pct", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Phase 2a: Peak Characterization")
    print("=" * 40)

    df = pd.read_csv(config.SAMPLE_DATA)
    peak_stats = characterize_peaks(df)

    for group in ["voltage", "intensity"]:
        print(f"\n--- {group.title()} Statistics ---")
        print(peak_stats[group].to_string(index=False))
        peak_stats[group].to_csv(
            config.RESULTS_DIR / f"peak_{group}_stats.csv", index=False
        )

    sensitivity = rank_manufacturing_sensitivity(peak_stats)
    print("\n--- Manufacturing Sensitivity Ranking (by CV%) ---")
    print(sensitivity[["feature", "cv_pct"]].to_string(index=False))
    sensitivity.to_csv(config.RESULTS_DIR / "sensitivity_ranking.csv", index=False)
