"""CD-MCAR package — H3 Adaptive Generation."""
from .cd_mcar import CDMCARTree, run_cdmcar
from .h3_utils import bbox_to_cells, cells_to_geodataframe, get_leaf_descendants
from .slmm import encode_boundary_targets, resolution_discontinuity_metric

__all__ = [
    "CDMCARTree",
    "run_cdmcar",
    "bbox_to_cells",
    "cells_to_geodataframe",
    "get_leaf_descendants",
    "encode_boundary_targets",
    "resolution_discontinuity_metric",
]
