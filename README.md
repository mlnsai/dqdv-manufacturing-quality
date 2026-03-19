# dQ/dV Manufacturing Quality Assessment Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/XXXX.svg)](https://doi.org/XXXX) -->

**Companion repository for:**
> *dQ/dV-Based Quality Screening Methodology for Fresh 18650 Cell Manufacturing Assessment*, Sai Krishna Mulpuri, Nataly Bañol Arias, Prasanth Venugopal and Thiago Batista Soeiro.  
> DOI: NA

A Python-based analytical framework for assessing lithium-ion cell manufacturing quality using differential voltage (dQ/dV) analysis. The framework processes dQ/dV peak parameters extracted from formation-cycle data and applies a four-phase analytical pipeline — from exploratory statistics through unsupervised machine learning to actionable quality grading.

---

## Overview

This repository provides the **analysis methodology** described in the paper. It includes:

- **Phase 1 — Exploratory Data Analysis**: Statistical profiling, normality testing, and correlation analysis of dQ/dV peak features  
- **Phase 2 — Peak Characterization**: Voltage hysteresis, intensity ratios, and charge–discharge asymmetry metrics  
- **Phase 3 — Advanced Analytics**: PCA dimensionality reduction, K-means clustering, multi-method outlier detection, and correlation network analysis  
- **Phase 4 — Quality Grading**: Composite health scoring and three-tier (A/B/C) cell classification  

A synthetic sample dataset is provided so all scripts can be run out-of-the-box.

> **Note**: Raw experimental data from the Molicel P45B cell population is not included. The sample dataset preserves statistical properties (distributions, correlations) while protecting proprietary measurements.

---

## Repository Structure

```
dqdv-manufacturing-quality/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (alternative)
├── config.py                          # Global paths and parameters
│
├── data/
│   ├── sample/
│   │   └── dqdv_features_sample.csv   # Synthetic sample dataset (N=134)
│   └── results/
│       └── .gitkeep                   # Analysis outputs saved here
│
├── src/
│   ├── __init__.py
│   ├── phase1_eda/
│   │   ├── __init__.py
│   │   ├── descriptive_stats.py       # Statistical profiling & normality tests
│   │   └── correlation_analysis.py    # Pearson correlations & heatmap
│   │
│   ├── phase2_characterization/
│   │   ├── __init__.py
│   │   ├── peak_analysis.py           # Voltage & intensity characterization
│   │   └── asymmetry_metrics.py       # Hysteresis, intensity ratios
│   │
│   ├── phase3_advanced/
│   │   ├── __init__.py
│   │   ├── pca_analysis.py            # PCA decomposition & visualization
│   │   ├── clustering.py              # K-means with optimal K selection
│   │   ├── outlier_detection.py       # IF + IQR + Z-score ensemble
│   │   └── correlation_network.py     # Feature network with Louvain communities
│   │
│   └── phase4_grading/
│       ├── __init__.py
│       └── health_grading.py          # Composite health score & A/B/C grading
│
├── notebooks/
│   └── full_pipeline_demo.ipynb       # End-to-end walkthrough notebook
│
├── figures/                           # Generated publication figures
│   └── .gitkeep
│
├── docs/
│   └── feature_definitions.md         # dQ/dV feature descriptions & units
│
└── run_pipeline.py                    # Single-command full pipeline execution
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/[USERNAME]/dqdv-manufacturing-quality.git
cd dqdv-manufacturing-quality

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate dqdv
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This will process the sample dataset through all four phases and save figures + summary tables to `data/results/` and `figures/`.

### 3. Interactive exploration

```bash
jupyter notebook notebooks/full_pipeline_demo.ipynb
```

---

## Dataset Format

The analysis expects a CSV file with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `cell_id` | Unique cell identifier | — |
| `CP1_voltage` | Charge Peak 1 voltage position | V |
| `CP2_voltage` | Charge Peak 2 voltage position | V |
| `CP3_voltage` | Charge Peak 3 voltage position | V |
| `DP1_voltage` | Discharge Peak 1 voltage position | V |
| `DP2_voltage` | Discharge Peak 2 voltage position | V |
| `CP1_intensity` | Charge Peak 1 intensity | Ah/V |
| `CP2_intensity` | Charge Peak 2 intensity | Ah/V |
| `CP3_intensity` | Charge Peak 3 intensity | Ah/V |
| `DP1_intensity` | Discharge Peak 1 intensity | Ah/V |
| `DP2_intensity` | Discharge Peak 2 intensity | Ah/V |

See [`docs/feature_definitions.md`](docs/feature_definitions.md) for detailed descriptions of each peak and its electrochemical origin.

---

## Adapting to Your Data

1. **Replace the sample CSV** with your own dQ/dV features (same column format)
2. **Update `config.py`** with your file path and any parameter adjustments
3. **Run the pipeline** — all phases adapt automatically to your data

The framework is chemistry-agnostic in its statistical methods, though peak assignments (Phase 4) assume graphite||NMC/NCA chemistry. Adjust the electrochemical interpretation in `docs/feature_definitions.md` for other systems.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{XXXX,
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was conducted as part of the [FreeTwinEV](https://freetwinev.stuba.sk/) project, funded by the European Union.
