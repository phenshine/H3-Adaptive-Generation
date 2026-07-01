"""
quantitative_analysis.py — Cross-scene quantitative metrics for CD-MCAR
========================================================================
Aggregates and reports quantitative performance metrics across all 6
experimental scenes described in the paper (Section 6).

Usage:
    python experiments/quantitative_analysis.py

Outputs:
    cross_scene_summary.csv   — per-scene metrics table
    quantitative_analysis.csv — detailed metrics (matches Table 5 in paper)
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Scene configuration (matches paper Section 6.2)
# ─────────────────────────────────────────────────────────────────────────────

SCENES = {
    "S01": {
        "name": "Coastal Zhoushan",
        "geojson": "01_coastal_Zhoushan_detections.geojson",
        "description": "Coastal port and island region",
    },
    "S02": {
        "name": "Urban Nanjing",
        "geojson": "02_urban_Nanjing_detections.geojson",
        "description": "Dense urban area",
    },
    "S03": {
        "name": "County Suzhou",
        "geojson": "03_county_Suzhou_detections.geojson",
        "description": "County-level mixed land use",
    },
    "S04": {
        "name": "Farmland Jiangxi",
        "geojson": "04_farmland_Jiangxi_detections.geojson",
        "description": "Agricultural plain",
    },
    "S05": {
        "name": "Mountain Wuyi",
        "geojson": "05_mountain_Wuyishan_detections.geojson",
        "description": "Mountainous terrain",
    },
    "S06": {
        "name": "Port Ningbo",
        "geojson": "06_port_Ningbo_detections.geojson",
        "description": "Major seaport and logistics hub",
    },
}


def load_scene_metrics(scene_id: str, data_dir: Path) -> dict:
    """
    Load pre-computed quantitative metrics for a scene.

    Looks for {scene_id}_quantitative_metrics.json in data_dir.
    Falls back to computing from adaptive_tree_leaves.csv if JSON not found.
    """
    json_path = data_dir / f"{scene_id}_quantitative_metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    # Fallback: derive from leaves CSV
    leaves_csv = data_dir / f"{scene_id}_adaptive_tree_leaves.csv"
    if not leaves_csv.exists():
        print(f"  [WARN] No data found for {scene_id}")
        return {}

    df = pd.read_csv(leaves_csv)
    n_leaves = len(df)
    res_dist = df["resolution"].value_counts().to_dict() if "resolution" in df.columns else {}

    return {
        "scene_id": scene_id,
        "n_leaf_cells": n_leaves,
        "resolution_distribution": res_dist,
    }


def compute_grid_reduction(n_leaves: int, study_area_km2: float,
                            res10_density: float = 0.0075) -> float:
    """
    Estimate grid reduction rate vs. a full Res10 coverage.

    res10_density: average number of Res10 cells per km² (default from paper)
    """
    n_res10_full = study_area_km2 / res10_density
    if n_res10_full == 0:
        return 0.0
    return 1.0 - n_leaves / n_res10_full


def summarise_all_scenes(data_dir: Path) -> pd.DataFrame:
    """Load metrics for all 6 scenes and return a summary DataFrame."""
    rows = []
    for scene_id, meta in SCENES.items():
        metrics = load_scene_metrics(scene_id, data_dir)
        if not metrics:
            continue
        rows.append({
            "Scene": scene_id,
            "Name": meta["name"],
            "Leaf Cells": metrics.get("n_leaf_cells", "N/A"),
            "Grid Reduction (%)": metrics.get("grid_reduction_pct", "N/A"),
            "Storage (MB)": metrics.get("storage_mb", "N/A"),
            "Query Time (ms)": metrics.get("query_time_ms", "N/A"),
        })

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).parent.parent
    data_dir = root  # metrics JSON files are at root level

    print("=" * 60)
    print("Cross-scene quantitative analysis")
    print("=" * 60)

    summary_df = summarise_all_scenes(data_dir)

    if summary_df.empty:
        print("[INFO] No pre-computed metrics found.")
        print("       Run cdmcar_batch_experiment.py first.")
        return

    print(summary_df.to_string(index=False))

    out_csv = root / "cross_scene_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
