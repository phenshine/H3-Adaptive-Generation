# H3-Adaptive-Generation

**CD-MCAR: Content-Driven Multi-Criteria Adaptive Refinement for H3 DGGS**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![IEEE J-STARS](https://img.shields.io/badge/Journal-IEEE%20J--STARS-green.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4609443)

This repository contains the full implementation of the **CD-MCAR (Content-Driven Multi-Criteria Adaptive Refinement)** algorithm and all experimental scripts for the paper:

> Zhang Aiguo, et al., "Content-Driven Multi-Criteria Adaptive Refinement for H3 Discrete Global Grid System: A Remote Sensing Object Detection Case Study", *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2026. (under review)

---

## Overview

CD-MCAR is a novel algorithm for generating spatially adaptive H3 DGGS (Discrete Global Grid System) grids driven by remote sensing object detection results. The algorithm adaptively refines H3 cells based on three criteria:

- **τ_n**: Target count threshold per cell
- **τ_s**: Target density (spatial concentration) threshold
- **τ_d**: Category diversity threshold

### Key Results (Xiamen Haicang Bay Study Area, 233.60 km²)

| Metric | Value |
|---|---|
| Detection targets | 1,438 objects, 12 categories |
| Grid reduction | **83.8%** (17,573 → 2,853 leaf cells) |
| Storage savings | **63%** (0.54 MB → 0.20 MB) |
| Full query time | 2,826 cells / 0.94 ms |
| Spatial coverage | Res7 (31) + Res8 (182) + Res9 (932) + Res10 (1,708) |

---

## Repository Structure

```
H3-Adaptive-Generation/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── src/                               # Core algorithm source code
│   ├── cd_mcar.py                     # CD-MCAR main algorithm
│   ├── slmm.py                        # SLMM (Spatial Load Management Model)
│   └── h3_utils.py                    # H3 utility functions
├── experiments/                       # Experimental scripts (Steps 2–8)
│   ├── h3_steps_2_1_to_2_4.py        # Step 2.1–2.4: Detection ingestion & H0 grid
│   ├── h3_steps_2_5.py               # Step 2.5: Polyfill
│   ├── h3_steps_2_6.py               # Step 2.6: Coverage statistics
│   ├── h3_steps_3_1_to_3_3.py        # Step 3: Target–cell relations
│   ├── h3_steps_4_1_to_4_5.py        # Step 4: Adaptive tree generation
│   ├── h3_steps_5_1_to_5_2.py        # Step 5: Boundary encoding
│   ├── h3_steps_6_1_to_6_4.py        # Step 6: Qualitative & quantitative analysis
│   ├── h3_steps_7_1_to_7_4.py        # Step 7: Paper figures & tables
│   ├── h3_steps_8_1_to_8_4.py        # Step 8: Database storage & indexing
│   ├── cdmcar_batch_experiment.py     # Multi-scene batch experiment (6 scenes)
│   └── quantitative_analysis.py       # Cross-scene quantitative metrics
├── data/                              # Derived datasets (CC BY 4.0)
│   └── sample/
│       ├── detections.geojson         # YOLO11n-OBB detection results (1,438 targets)
│       ├── H0_grid.geojson            # H0-level H3 grid (Res7, 81 cells)
│       ├── h3_adaptive_grid.geojson   # Final adaptive grid (2,853 leaf cells)
│       ├── adaptive_tree_leaves.csv   # Leaf cell attributes
│       ├── boundary_encoding.json     # SLMM boundary encoding
│       └── experiment_manifest.json   # Experiment metadata & reproducibility
└── docs/
    ├── API.md                         # API reference
    └── h3_adaptive_setup.sql          # PostgreSQL + PostGIS schema
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 13+ with PostGIS extension (optional, for database experiments)

### Setup

```bash
# Clone the repository
git clone https://github.com/phenshine/H3-Adaptive-Generation.git
cd H3-Adaptive-Generation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run CD-MCAR on sample data

```bash
# Run the full adaptive grid generation pipeline
python experiments/h3_steps_4_1_to_4_5.py

# Expected output: adaptive_tree.json, adaptive_tree_leaves.csv
```

### 2. Run multi-scene batch experiment

```bash
python experiments/cdmcar_batch_experiment.py
# Processes 6 scenes: coastal, urban, county, farmland, mountain, port
```

### 3. Reproduce paper figures

```bash
python experiments/h3_steps_7_1_to_7_4.py
# Generates Figure1_framework_overview.png, quantitative plots, etc.
```

### 4. Database storage (requires PostgreSQL + PostGIS)

```bash
# Initialize schema
psql -U postgres -d your_db -f docs/h3_adaptive_setup.sql

# Run database experiment
python experiments/h3_steps_8_1_to_8_4.py
```

---

## Data Description

All derived datasets are released under **CC BY 4.0** license.

| File | Description | Format | Size |
|------|-------------|--------|------|
| `detections.geojson` | YOLO11n-OBB detection results, 1,438 targets, 12 categories | GeoJSON | ~2 MB |
| `H0_grid.geojson` | H3 Res7 base grid, 81 cells covering study area | GeoJSON | ~50 KB |
| `h3_adaptive_grid.geojson` | Final adaptive grid, 2,853 leaf cells (Res7–Res10) | GeoJSON | ~0.20 MB |
| `adaptive_tree_leaves.csv` | Leaf cell attributes: resolution, target count, diversity | CSV | ~200 KB |
| `boundary_encoding.json` | SLMM Hilbert-curve boundary encoding | JSON | ~150 KB |
| `experiment_manifest.json` | Full experiment metadata for reproducibility | JSON | ~30 KB |

> **Note**: The original remote sensing imagery was obtained from Gaode Maps (https://www.amap.com) and is subject to their licensing terms. Only derived vector datasets are shared here.

---

## Algorithm Parameters

The CD-MCAR algorithm uses three threshold parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_n` (τ_n) | 5 | Minimum target count to trigger cell splitting |
| `tau_s` (τ_s) | 0.6 | Spatial concentration threshold (Gini-based) |
| `tau_d` (τ_d) | 3 | Minimum category diversity to trigger splitting |
| `max_res` | 10 | Maximum H3 resolution (finest refinement level) |

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{zhang2026cdmcar,
  author  = {Zhang, Aiguo and [Co-authors]},
  title   = {Content-Driven Multi-Criteria Adaptive Refinement for H3 Discrete Global Grid System: A Remote Sensing Object Detection Case Study},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  volume  = {[TBD]},
  number  = {[TBD]},
  pages   = {[TBD]},
  doi     = {[TBD after acceptance]}
}
```

---

## License

- **Code**: MIT License — see [LICENSE](LICENSE)
- **Data**: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Contact

- **Corresponding Author**: Zhang Aiguo, Xiamen University
- **Email**: 623467897@qq.com
- **Institution**: School of Earth Sciences, Xiamen University, Xiamen, China

---

## Acknowledgments

This work was supported by [funding information TBD]. The authors thank Gaode Maps for providing the remote sensing imagery data. The H3 library is developed and maintained by Uber Technologies, Inc.
