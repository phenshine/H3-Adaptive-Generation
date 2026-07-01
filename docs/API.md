# API Reference — H3-Adaptive-Generation

This document describes the public API of the three core modules in `src/`.

---

## Module: `src/cd_mcar.py`

### Class `CDMCARTree`

The main algorithm class.

```python
from src.cd_mcar import CDMCARTree

tree = CDMCARTree(tau_n=5, tau_s=0.6, tau_d=3, max_res=10)
tree.build(h0_cells, targets_df)
leaves = tree.get_leaves()
```

#### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau_n` | int | 5 | Min target count to trigger splitting |
| `tau_s` | float | 0.6 | Spatial concentration threshold (Gini) |
| `tau_d` | int | 3 | Min category diversity to trigger splitting |
| `max_res` | int | 10 | Maximum H3 resolution (finest level) |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `build(h0_cells, targets_df, ...)` | None | Build the adaptive tree |
| `get_leaves()` | `List[dict]` | Return all leaf nodes |
| `to_json(path=None)` | `dict` | Serialise tree; optionally write file |
| `leaves_to_dataframe()` | `pd.DataFrame` | Leaf nodes as a tidy DataFrame |

---

### Function `run_cdmcar`

Convenience wrapper that builds the tree and optionally writes outputs.

```python
from src.cd_mcar import run_cdmcar

tree = run_cdmcar(
    h0_cells,
    targets_df,
    tau_n=5, tau_s=0.6, tau_d=3, max_res=10,
    output_tree_path="adaptive_tree.json",
    output_leaves_path="adaptive_tree_leaves.csv"
)
```

---

### Function `should_split`

Low-level splitting decision for a single cell.

```python
from src.cd_mcar import should_split

split, metrics = should_split(targets_in_cell, resolution=8)
# metrics = {"n": 12, "s": 0.72, "d": 4, "split": True}
```

---

## Module: `src/h3_utils.py`

Utility wrappers for the h3-py v4 API.

| Function | Signature | Description |
|----------|-----------|-------------|
| `latlng_to_cell` | `(lat, lng, res) → str` | Point → H3 cell |
| `cell_to_latlng` | `(h_cell) → (lat, lng)` | Cell centroid |
| `get_resolution` | `(h_cell) → int` | Cell resolution |
| `cell_to_boundary_lonlat` | `(h_cell) → [(lon,lat),...]` | Boundary coords |
| `cell_to_shapely` | `(h_cell) → Polygon` | Shapely polygon |
| `cell_area_km2` | `(h_cell) → float` | Area in km² |
| `bbox_to_cells` | `(min_lon,min_lat,max_lon,max_lat,res) → Set[str]` | BBox → H3 cells |
| `geojson_to_cells` | `(geojson, res) → Set[str]` | GeoJSON → H3 cells |
| `get_leaf_descendants` | `(h_cell, tree, max_res) → Set[str]` | Recursive leaf query |
| `compact` | `(cells) → List[str]` | H3 compact |
| `uncompact` | `(cells, target_res) → List[str]` | H3 uncompact |
| `cells_to_geodataframe` | `(cells, extra_attrs, crs) → GeoDataFrame` | Export to GDF |

---

## Module: `src/slmm.py`

Spatial Load Management Model — boundary encoding.

### Function `detect_boundary_targets`

```python
from src.slmm import detect_boundary_targets

boundary_targets = detect_boundary_targets(targets_df, leaf_cells, resolution=10)
```

### Function `encode_boundary_targets`

```python
from src.slmm import encode_boundary_targets

encoded = encode_boundary_targets(boundary_targets)
# encoded[target_id] = {"strategy": "H3-Ascend", "code": ..., ...}
```

### Function `resolution_discontinuity_metric`

```python
from src.slmm import resolution_discontinuity_metric

rdm, discontinuities = resolution_discontinuity_metric(leaf_cells, delta_r_smooth=1)
```

---

## Data Schema

### `detections.geojson`

GeoJSON FeatureCollection. Each feature has:

| Property | Type | Description |
|----------|------|-------------|
| `class_name` | string | Object category (e.g., "ship", "aircraft") |
| `confidence` | float | Detection confidence [0, 1] |
| `width_m` | float | Object width in metres |
| `height_m` | float | Object height in metres |

### `adaptive_tree_leaves.csv`

| Column | Type | Description |
|--------|------|-------------|
| `h3_cell` | string | H3 cell index |
| `resolution` | int | H3 resolution (7–10) |
| `n_targets` | int | Number of targets in cell |
| `spatial_conc` | float | Spatial concentration score |
| `diversity` | int | Number of distinct categories |

### `boundary_encoding.json`

JSON object mapping `target_id → {strategy, code, h3_cells, nca_cell, nca_resolution, confidence, area}`.
