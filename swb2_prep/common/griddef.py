# swb2_prep/common/griddef.py
# -*- coding: utf-8 -*-

"""
Read/write utilities for SWB grid definition TOML files.

The written grid definition includes:
- CRS string
- PROJ4 string
- Resolution
- Raw extents (xmin_raw, ymin_raw, xmax_raw, ymax_raw)
- Final extents (xmin, ymin, xmax, ymax)
- Grid dimensions (nx, ny)
- Snap mode
- Source type ("aoi_polygon" or "bbox")
"""

from __future__ import annotations
from pathlib import Path
import tomllib  # Python 3.11+ TOML reader
import toml     # External, full-feature TOML writer
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS as _CRS


REQUIRED_FIELDS = [
    "crs",
    "proj4",
    "resolution",
    "xmin_raw", "ymin_raw", "xmax_raw", "ymax_raw",
    "xmin", "ymin", "xmax", "ymax",
    "nx", "ny",
    "snap",
    "source",
]


def griddef_to_polygon_gdf(grid: dict) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame containing a single polygon representing
    the bounding box of the SWB grid.

    Parameters
    ----------
    grid : dict
        Parsed grid-definition dictionary, as returned by
        read_grid_definition().

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame containing the bounding-box polygon
        in the correct CRS.
    """
    xmin = grid["xmin"]
    ymin = grid["ymin"]
    xmax = grid["xmax"]
    ymax = grid["ymax"]

    geom = box(xmin, ymin, xmax, ymax)

    # Ensure CRS is a valid CRS object/string
    crs = _CRS.from_user_input(grid["crs"]).to_string()

    return gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)


def write_grid_definition(path: Path, grid: dict) -> None:
    """
    Write the SWB grid definition dictionary to a TOML file.

    Parameters
    ----------
    path : Path
        File path where the TOML file will be written.
    grid : dict
        Dictionary containing SWB grid metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"swb_grid": grid}

    with path.open("w", encoding="utf-8") as f:
        toml.dump(data, f)


def read_grid_definition(path: Path) -> dict:
    """
    Read and validate an SWB grid definition file.

    Parameters
    ----------
    path : Path
        Path to the swb_grid_definition.toml file.

    Returns
    -------
    dict
        Parsed SWB grid definition dictionary.

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid-definition file not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    if "swb_grid" not in data:
        raise ValueError("TOML file missing [swb_grid] section.")

    grid = data["swb_grid"]

    missing = [k for k in REQUIRED_FIELDS if k not in grid]
    if missing:
        raise ValueError(f"Grid-definition TOML missing required fields: {missing}")

    return grid
