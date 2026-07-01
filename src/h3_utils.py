"""
h3_utils.py — H3 utility functions for CD-MCAR
================================================
Wraps h3-py v4 API with convenience helpers used throughout the project.

API mapping (h3-py v3 → v4):
  h3.polyfill(geo, res)          → h3.h3shape_to_cells(h3.geo_to_h3shape(geo), res)
  h3.h3_to_geo_boundary(h)       → h3.cell_to_boundary(h)
  h3.cell_area(h, unit='km2')    → h3.cell_area(h, unit='km^2')
  h3.h3_get_resolution(h)        → h3.get_resolution(h)
  h3.h3_to_children(h, r)        → h3.cell_to_children(h, r)
  h3.h3_to_parent(h, r)          → h3.cell_to_parent(h, r)
  h3.compact(cells)              → h3.compact_cells(cells)
  h3.uncompact(cells, r)         → h3.uncompact_cells(cells, r)

License: MIT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import h3
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, mapping


# ─────────────────────────────────────────────────────────────────────────────
# Cell ↔ coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def latlng_to_cell(lat: float, lng: float, resolution: int) -> str:
    """Convert a lat/lng point to the containing H3 cell index."""
    return h3.latlng_to_cell(lat, lng, resolution)


def cell_to_latlng(h_cell: str) -> Tuple[float, float]:
    """Return the (lat, lng) centroid of an H3 cell."""
    return h3.cell_to_latlng(h_cell)


def get_resolution(h_cell: str) -> int:
    """Return the resolution of an H3 cell."""
    return h3.get_resolution(h_cell)


def cell_to_boundary_lonlat(h_cell: str) -> List[Tuple[float, float]]:
    """
    Return the boundary of an H3 cell as a list of (lon, lat) tuples
    (suitable for Shapely / matplotlib plotting).

    h3.cell_to_boundary returns (lat, lon) pairs — we swap here.
    """
    boundary = h3.cell_to_boundary(h_cell)   # list of (lat, lon)
    return [(lng, lat) for lat, lng in boundary]


def cell_to_shapely(h_cell: str) -> Polygon:
    """Return a Shapely Polygon for an H3 cell boundary (lon, lat CRS)."""
    coords = cell_to_boundary_lonlat(h_cell)
    return Polygon(coords)


def cell_area_km2(h_cell: str) -> float:
    """Return the approximate area of an H3 cell in km²."""
    return h3.cell_area(h_cell, unit="km^2")


# ─────────────────────────────────────────────────────────────────────────────
# Region → cells
# ─────────────────────────────────────────────────────────────────────────────

def bbox_to_cells(min_lon: float, min_lat: float,
                  max_lon: float, max_lat: float,
                  resolution: int) -> Set[str]:
    """
    Return all H3 cells at *resolution* that cover a bounding box.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : bounding box in WGS-84
    resolution : H3 resolution (0–15)
    """
    geo = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }
    h3shape = h3.geo_to_h3shape(geo)
    return set(h3.h3shape_to_cells(h3shape, resolution))


def geojson_to_cells(geojson: dict, resolution: int) -> Set[str]:
    """Fill a GeoJSON polygon with H3 cells at the given resolution."""
    h3shape = h3.geo_to_h3shape(geojson)
    return set(h3.h3shape_to_cells(h3shape, resolution))


# ─────────────────────────────────────────────────────────────────────────────
# Tree traversal
# ─────────────────────────────────────────────────────────────────────────────

def get_leaf_descendants(h_cell: str,
                          tree: Dict[str, dict],
                          max_res: int = 10) -> Set[str]:
    """
    Recursively collect all leaf cells that are descendants of *h_cell*
    in the adaptive tree.

    Parameters
    ----------
    h_cell   : starting H3 cell (any resolution)
    tree     : adaptive tree dict {h3_cell: {split, children, ...}}
    max_res  : safety cap on recursion depth

    Returns
    -------
    Set of leaf H3 cell strings
    """
    node = tree.get(h_cell)
    if node is None:
        return {h_cell}   # treat as leaf if not in tree

    if not node.get("split", False):
        return {h_cell}

    leaves: Set[str] = set()
    for child in node.get("children", []):
        leaves |= get_leaf_descendants(child, tree, max_res)
    return leaves


# ─────────────────────────────────────────────────────────────────────────────
# Compact / uncompact
# ─────────────────────────────────────────────────────────────────────────────

def compact(cells: List[str]) -> List[str]:
    """Compact a set of H3 cells to the coarsest possible representation."""
    return list(h3.compact_cells(cells))


def uncompact(cells: List[str], target_resolution: int) -> List[str]:
    """Expand compacted cells to a uniform target resolution."""
    return list(h3.uncompact_cells(cells, target_resolution))


# ─────────────────────────────────────────────────────────────────────────────
# GeoDataFrame export
# ─────────────────────────────────────────────────────────────────────────────

def cells_to_geodataframe(cells: List[str],
                           extra_attrs: Optional[Dict[str, List]] = None,
                           crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Convert a list of H3 cells to a GeoDataFrame of polygon boundaries.

    Parameters
    ----------
    cells       : list of H3 cell index strings
    extra_attrs : dict of additional column_name → list_of_values to attach
    crs         : coordinate reference system (default WGS-84)
    """
    geometries = [cell_to_shapely(c) for c in cells]
    resolutions = [get_resolution(c) for c in cells]

    gdf = gpd.GeoDataFrame(
        {"h3_cell": cells, "resolution": resolutions, "geometry": geometries},
        crs=crs,
    )

    if extra_attrs:
        for col, values in extra_attrs.items():
            gdf[col] = values

    return gdf
