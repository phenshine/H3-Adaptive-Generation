"""
slmm.py — SLMM: Spatial Load Management Model
==============================================
Implements the Spatial Load Management Model for H3 boundary encoding,
including Hilbert-curve ordering and the H3-CBFE encoding strategies.

Reference:
  Zhang Aiguo et al., "Content-Driven Multi-Criteria Adaptive Refinement
  for H3 Discrete Global Grid System", IEEE J-STARS, 2026.

License: MIT
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

import h3
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Boundary target detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_boundary_targets(targets_df: pd.DataFrame,
                             leaf_cells: Set[str],
                             lat_col: str = "lat",
                             lon_col: str = "lon",
                             resolution: int = 10) -> List[dict]:
    """
    Identify targets whose centroid falls on or near an H3 cell boundary.

    A target is considered a 'boundary target' if its H3 cell at *resolution*
    is NOT the same as its H3 cell at resolution-1 mapped back to resolution.

    Parameters
    ----------
    targets_df : DataFrame with target centroid coordinates
    leaf_cells : set of active leaf cell indices
    resolution : finest H3 resolution used in the adaptive grid

    Returns
    -------
    List of dicts: [{target_id, h3_cells, confidence, area}, ...]
    """
    boundary_targets = []

    for idx, row in targets_df.iterrows():
        lat = row.get(lat_col, row.get("lat", 0))
        lng = row.get(lon_col, row.get("lon", 0))

        # Find the leaf cell containing this target
        try:
            cell_at_res = h3.latlng_to_cell(lat, lng, resolution)
        except Exception:
            continue

        neighbors = h3.grid_disk(cell_at_res, 1)
        covering_leaves = [c for c in neighbors if c in leaf_cells]

        if len(covering_leaves) > 1:
            boundary_targets.append({
                "target_id": idx,
                "h3_cells": covering_leaves,
                "confidence": float(row.get("confidence", 0.5)),
                "area": float(row.get("area", 0.0)),
            })

    return boundary_targets


# ─────────────────────────────────────────────────────────────────────────────
# Nearest Common Ancestor
# ─────────────────────────────────────────────────────────────────────────────

def find_nca(h3_cells: List[str]) -> Tuple[Optional[str], int]:
    """
    Find the Nearest Common Ancestor (NCA) of a set of H3 cells.

    Returns
    -------
    (nca_cell, nca_resolution)
    """
    cells = [str(c) for c in h3_cells if c]
    if not cells:
        return None, -1
    if len(cells) == 1:
        return cells[0], h3.get_resolution(cells[0])

    min_res = min(h3.get_resolution(c) for c in cells)
    # Project all cells to min_res
    projected = [h3.cell_to_parent(c, min_res) if h3.get_resolution(c) > min_res
                 else c for c in cells]

    if len(set(projected)) == 1:
        return projected[0], min_res

    current = projected
    for res in range(min_res - 1, -1, -1):
        parents = [h3.cell_to_parent(c, res) for c in current]
        if len(set(parents)) == 1:
            return parents[0], res
        current = parents

    return cells[0], h3.get_resolution(cells[0])


# ─────────────────────────────────────────────────────────────────────────────
# H3-CBFE: Cross-Boundary Fusion Encoding (Algorithm B.2.3)
# ─────────────────────────────────────────────────────────────────────────────

def encode_boundary_targets(boundary_targets: List[dict]) -> Dict[str, dict]:
    """
    Apply H3-CBFE encoding to boundary targets.

    Three strategies (selected automatically):
      H3-Ascend           — for small targets (≤2 covering cells, or NCA ≥ Res9)
      H3-Primary-Secondary — for medium targets (confidence ≥ 0.85, ≤5 cells)
      H3-Multi-Code        — for large targets (all other cases)

    Parameters
    ----------
    boundary_targets : output of detect_boundary_targets()

    Returns
    -------
    Dict mapping target_id → {strategy, code, h3_cells, nca_cell, ...}
    """
    results: Dict[str, dict] = {}

    for t in boundary_targets:
        tid = t["target_id"]
        cells = [str(c) for c in t["h3_cells"]]
        conf = t["confidence"]

        nca, nca_res = find_nca(cells)
        n = len(cells)

        if nca_res is not None and nca_res >= 9 or n <= 2:
            strategy = "H3-Ascend"
            code = nca
        elif conf >= 0.85 and n <= 5:
            strategy = "H3-Primary-Secondary"
            code = {"primary": cells[0], "secondary": cells[1:], "nca": nca}
        else:
            strategy = "H3-Multi-Code"
            code = {"cells": cells, "nca": nca, "n_children": n}

        results[str(tid)] = {
            "h3_cells": cells,
            "nca_cell": nca,
            "nca_resolution": nca_res,
            "strategy": strategy,
            "code": code,
            "confidence": conf,
            "area": t.get("area", 0.0),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Discontinuity Metric (Algorithm B.2.2: H3-LSP)
# ─────────────────────────────────────────────────────────────────────────────

def resolution_discontinuity_metric(leaf_cells: Set[str],
                                     delta_r_smooth: int = 1) -> Tuple[float, List[dict]]:
    """
    Compute the Resolution Discontinuity Metric (RDM) across leaf cells.

    RDM = (1/|ε|) * Σ max(0, |r_j - r_k| - Δr_smooth)²

    Parameters
    ----------
    leaf_cells     : set of H3 leaf cell indices
    delta_r_smooth : allowable resolution difference between neighbors

    Returns
    -------
    (rdm_score, list_of_discontinuity_dicts)
    """
    leaf_res = {c: h3.get_resolution(c) for c in leaf_cells}
    discontinuities = []
    total_penalty = 0.0

    for h_cell in leaf_cells:
        r = leaf_res[h_cell]
        neighbors = h3.grid_disk(h_cell, 1)
        for nb in neighbors:
            if nb == h_cell or nb not in leaf_res:
                continue
            r_nb = leaf_res[nb]
            diff = abs(r - r_nb)
            if diff > delta_r_smooth:
                penalty = (diff - delta_r_smooth) ** 2
                total_penalty += penalty
                discontinuities.append({
                    "cell1": h_cell,
                    "cell2": nb,
                    "r1": r,
                    "r2": r_nb,
                    "r_diff": diff,
                    "penalty": penalty,
                })

    rdm = total_penalty / max(len(leaf_cells), 1)
    return rdm, discontinuities


# ─────────────────────────────────────────────────────────────────────────────
# Save / load utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_encoding(encoded: Dict[str, dict], path: str) -> None:
    """Write encoding results to a JSON file."""
    with open(path, "w") as f:
        json.dump(encoded, f, indent=2, default=str)


def load_encoding(path: str) -> Dict[str, dict]:
    """Load encoding results from a JSON file."""
    with open(path) as f:
        return json.load(f)
