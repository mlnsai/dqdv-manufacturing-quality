"""
Generate Synthetic Sample Dataset
==================================

Creates a realistic sample dataset with statistical properties representative
of a graphite||NMC 18650 cell population, for demonstration purposes.

The synthetic data preserves:
  - Realistic voltage positions for graphite staging & NMC phase transitions
  - Typical CV% ranges seen in manufacturing populations
  - Correlation structure between charge/discharge pairs

This script is NOT part of the analysis pipeline — it is provided so that
users can run the pipeline out-of-the-box without needing real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_data(n_cells: int = 134, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dQ/dV peak features for n_cells.

    Voltage positions are based on typical graphite||NMC half-cell
    literature values with realistic manufacturing spread.
    Intensities are generated with correlated noise to mimic
    real cell-to-cell variation.

    Parameters
    ----------
    n_cells : int
        Number of synthetic cells to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with cell_id + 10 feature columns.
    """
    rng = np.random.default_rng(seed)

    # --- Voltage positions (V) ---
    # Based on typical graphite staging potentials and NMC transitions
    # Mean ± std chosen to give realistic CV% (0.1–0.5% for voltages)
    CP1_voltage = rng.normal(loc=3.480, scale=0.005, size=n_cells)   # Low-V charge
    CP2_voltage = rng.normal(loc=3.560, scale=0.004, size=n_cells)   # Mid charge
    CP3_voltage = rng.normal(loc=3.650, scale=0.003, size=n_cells)   # Main charge
    DP1_voltage = rng.normal(loc=3.420, scale=0.005, size=n_cells)   # Main discharge
    DP2_voltage = rng.normal(loc=3.580, scale=0.004, size=n_cells)   # High discharge

    # --- Intensities (Ah/V) ---
    # CP3 and DP2 are the dominant peaks; CP2 is typically smallest
    # Add correlated noise: charge/discharge pairs share a common component
    common_1 = rng.normal(0, 0.02, size=n_cells)  # Shared variation (pair 1)
    common_2 = rng.normal(0, 0.03, size=n_cells)  # Shared variation (pair 2)

    CP1_intensity = rng.normal(loc=1.20, scale=0.08, size=n_cells) + common_1
    CP2_intensity = rng.normal(loc=0.45, scale=0.04, size=n_cells)
    CP3_intensity = rng.normal(loc=2.80, scale=0.15, size=n_cells) + common_2
    DP1_intensity = rng.normal(loc=1.15, scale=0.07, size=n_cells) + common_1
    DP2_intensity = rng.normal(loc=2.70, scale=0.14, size=n_cells) + common_2

    # Inject a few mild outliers (3–4 cells) to make outlier detection meaningful
    outlier_idx = rng.choice(n_cells, size=3, replace=False)
    CP3_intensity[outlier_idx[0]] -= 0.6   # Unusually low main charge peak
    DP2_intensity[outlier_idx[1]] -= 0.55  # Unusually low main discharge peak
    CP1_intensity[outlier_idx[2]] += 0.35  # Unusually high minor peak

    df = pd.DataFrame({
        "cell_id": [f"CELL_{i+1:03d}" for i in range(n_cells)],
        "CP1_voltage": CP1_voltage,
        "CP2_voltage": CP2_voltage,
        "CP3_voltage": CP3_voltage,
        "DP1_voltage": DP1_voltage,
        "DP2_voltage": DP2_voltage,
        "CP1_intensity": CP1_intensity,
        "CP2_intensity": CP2_intensity,
        "CP3_intensity": CP3_intensity,
        "DP1_intensity": DP1_intensity,
        "DP2_intensity": DP2_intensity,
    })

    # Round to realistic measurement precision
    for col in df.columns:
        if "voltage" in col:
            df[col] = df[col].round(6)  # µV-level precision
        elif "intensity" in col:
            df[col] = df[col].round(6)

    return df


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[2] / "data" / "sample" / "dqdv_features_sample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sample_data()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic cells → {output_path}")
    print(f"\nPreview:\n{df.head().to_string()}")
    print(f"\nDescriptive stats:\n{df.describe().round(4).to_string()}")
