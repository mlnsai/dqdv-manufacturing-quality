"""
Global configuration for the dQ/dV Manufacturing Quality Analysis pipeline.

Adjust paths and parameters here to adapt the framework to your dataset.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DATA = DATA_DIR / "sample" / "dqdv_features_sample.csv"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

# Create output directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────
VOLTAGE_FEATURES = [
    "CP1_voltage", "CP2_voltage", "CP3_voltage",
    "DP1_voltage", "DP2_voltage",
]
INTENSITY_FEATURES = [
    "CP1_intensity", "CP2_intensity", "CP3_intensity",
    "DP1_intensity", "DP2_intensity",
]
ALL_FEATURES = VOLTAGE_FEATURES + INTENSITY_FEATURES

# Charge–discharge peak pairs (for asymmetry analysis)
PEAK_PAIRS = {
    "pair_1": {"charge": "CP1", "discharge": "DP1"},  # Low-voltage pair
    "pair_2": {"charge": "CP3", "discharge": "DP2"},  # High-voltage / main pair
}

# ──────────────────────────────────────────────
# Phase 1: EDA parameters
# ──────────────────────────────────────────────
NORMALITY_ALPHA = 0.05          # Shapiro-Wilk significance level
STRONG_CORR_THRESHOLD = 0.7     # |r| above this → strong correlation

# ──────────────────────────────────────────────
# Phase 3: Advanced analytics parameters
# ──────────────────────────────────────────────
K_RANGE = range(2, 9)           # K-means cluster range to evaluate
KMEANS_N_INIT = 10              # Number of random initializations
KMEANS_MAX_ITER = 300           # Max iterations per run
IF_CONTAMINATION = 0.05         # Isolation Forest expected outlier fraction
IQR_MULTIPLIER = 1.5            # IQR fence multiplier
ZSCORE_THRESHOLD = 3.0          # Z-score outlier threshold
OUTLIER_CONSENSUS = 2           # Minimum methods agreeing → high-risk flag
NETWORK_EDGE_THRESHOLD = 0.6    # |r| above this → draw edge

# ──────────────────────────────────────────────
# Phase 4: Grading parameters
# ──────────────────────────────────────────────
HEALTH_PEAKS = ["CP3_intensity", "DP2_intensity"]  # Peaks for health score
GRADE_PERCENTILES = {
    "A": 66,   # Top 33%: above 66th percentile
    "B": 33,   # Middle 33%: 33rd–66th percentile
    "C": 0,    # Bottom 33%: below 33rd percentile
}
QC_SIGMA_REJECT = 2  # Reject if health_score < (mean - QC_SIGMA_REJECT * std)

# ──────────────────────────────────────────────
# Plotting defaults
# ──────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"  # Use "svg" or "pdf" for vector graphics
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
