# swb_cli/common/grids.py
# -*- coding: utf-8 -*-
"""
Grid operations using xarray/rioxarray.

This module provides routines to:
- Reproject rasters to a target CRS (optionally to a target resolution).
- Resample rasters to a new resolution within the same CRS.
- Reproject polygons (GeoDataFrames) to a target CRS.

All raster functions operate on :class:`xarray.DataArray` with CRS/transform
managed via the rioxarray accessor (``.rio``).
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.warp import Resampling
from shapely.geometry import box
from typing import Tuple
from swb2_prep.common.utils import CRSLike

def reproject_raster_xr(
    data_array: xr.DataArray,
    target_crs: CRSLike,
    *,
    resolution: Optional[float] = None,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """Reproject a raster :class:`xarray.DataArray` to a target CRS using rioxarray.

    Args:
        data_array: Input raster with CRS/transform attached via ``data_array.rio``.
        target_crs: Target CRS (EPSG string, PROJ dict, or :class:`pyproj.CRS`).
        resolution: Optional target pixel size (units of the target CRS). If provided,
            the output is resampled to this resolution during reprojection; otherwise,
            the original resolution is preserved.
        resampling: Resampling algorithm from :mod:`rasterio.warp.Resampling`. Defaults
            to bilinear for continuous data. For categorical rasters, prefer nearest.

    Returns:
        A new :class:`xarray.DataArray` reprojected to ``target_crs`` (and to
        ``resolution`` if specified), with CRS/transform preserved via rioxarray.

    Raises:
        ValueError: If the input does not carry CRS/transform via rioxarray accessors.

    Notes:
        - This XR-first approach is used throughout SWB2-prep for consistent reprojection
          and optional resampling in a single operation. Tests exercise reprojection
          behavior and check CRS and pixel size expectations.
    """
    # rioxarray supports specifying the CRS and (optionally) resolution during reprojection
    # via .rio.reproject.
    if data_array.rio.crs is None:
        raise ValueError("Input DataArray is missing CRS/transform metadata (.rio).")

    kwargs = {}
    if resolution is not None:
        kwargs["resolution"] = resolution

    reprojected = data_array.rio.reproject(
        dst_crs=target_crs,
        resampling=resampling,
        **kwargs,
    )
    return reprojected


def resample_raster_xr(
    data_array: xr.DataArray,
    target_resolution: float,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """Resample a raster :class:`xarray.DataArray` to a new resolution in the **same CRS**.

    Args:
        data_array: Input raster with CRS/transform attached via ``data_array.rio``.
        target_resolution: Desired output pixel size (units of the current CRS).
        resampling: Resampling algorithm from :mod:`rasterio.warp.Resampling`. Defaults
            to bilinear for continuous data. For categorical rasters, prefer nearest.

    Returns:
        A new :class:`xarray.DataArray` with the requested ``target_resolution`` in the
        same CRS as the input, with CRS/transform preserved via rioxarray.

    Raises:
        ValueError: If the input does not carry CRS/transform via rioxarray accessors.

    Notes:
        - This function does **not** change the CRS; it only changes resolution. For CRS
          changes, use :func:`reproject_raster_xr`. Your tests validate that pixel size
          changes while CRS remains the same.
    """
    if data_array.rio.crs is None:
        raise ValueError("Input DataArray is missing CRS/transform metadata (.rio).")

    # rioxarray allows resampling in-place by reprojecting to the same CRS with a new resolution.
    resampled = data_array.rio.reproject(
        dst_crs=data_array.rio.crs,
        resolution=target_resolution,
        resampling=resampling,
    )
    return resampled


def reproject_polygon(
    gdf: gpd.GeoDataFrame,
    target_crs: CRSLike,
) -> gpd.GeoDataFrame:
    """Reproject a polygon :class:`geopandas.GeoDataFrame` to the target CRS.

    Args:
        gdf: Input polygon GeoDataFrame with a defined CRS.
        target_crs: Target CRS (EPSG string, PROJ dict, or :class:`pyproj.CRS`).

    Returns:
        A new GeoDataFrame with geometries transformed to ``target_crs``.

    Raises:
        ValueError: If the input GeoDataFrame has no CRS defined.

    Notes:
        - This function is used across SWB2-prep workflows to ensure AOI polygons
          are aligned to the raster CRS prior to clipping and IO. Tests assert that
          the CRS is updated as expected after reprojection.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS; cannot reproject.")
    return gdf.to_crs(target_crs)


def create_polygon_from_bbox(xmin: float, ymin: float, xmax: float, ymax: float, crs: str):
    """
    Create a GeoDataFrame polygon from a bounding box.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : float
        Bounding-box coordinates representing the area of interest.
        Assumed to be in the PROJECT CRS (Option B1).
    crs : str
        The CRS for the polygon (e.g., "EPSG:5070").

    Returns
    -------
    geopandas.GeoDataFrame
        A single-row GeoDataFrame containing the bounding-box polygon.

    Notes
    -----
    This supports the CLI mode:

        --bbox xmin ymin xmax ymax

    as an alternative to loading a polygon shapefile.

    All coordinates are assumed to already be in the project CRS.
    """
    geom = box(xmin, ymin, xmax, ymax)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)


def snap_extent(
    xmin_raw: float,
    ymin_raw: float,
    xmax_raw: float,
    ymax_raw: float,
    resolution: float,
    mode: str = "outward",
) -> Tuple[float, float, float, float]:
    """Snap raw extent to a resolution grid.

    Args:
        xmin_raw: Raw xmin.
        ymin_raw: Raw ymin.
        xmax_raw: Raw xmax.
        ymax_raw: Raw ymax.
        resolution: Cell size in project CRS units.
        mode: Snapping mode: ``'outward'`` (expand) or ``'inward'`` (shrink).

    Returns:
        Tuple of ``(xmin, ymin, xmax, ymax)`` representing snapped extent.

    Raises:
        ValueError: If ``resolution`` is non-positive or the snapped extent collapses.

    Notes:
        - Outward: floor(min), ceil(max) to fully cover AOI.
        - Inward: ceil(min), floor(max) to ensure all cells lie within AOI.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")
    if mode not in {"outward", "inward"}:
        raise ValueError(f"Unsupported snap mode: {mode!r}")

    if mode == "outward":
        xmin = np.floor(xmin_raw / resolution) * resolution
        ymin = np.floor(ymin_raw / resolution) * resolution
        xmax = np.ceil(xmax_raw / resolution) * resolution
        ymax = np.ceil(ymax_raw / resolution) * resolution
    else:
        xmin = np.ceil(xmin_raw / resolution) * resolution
        ymin = np.ceil(ymin_raw / resolution) * resolution
        xmax = np.floor(xmax_raw / resolution) * resolution
        ymax = np.floor(ymax_raw / resolution) * resolution

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Snapped extent collapsed; check resolution and AOI bounds.")

    return xmin, ymin, xmax, ymax


def compute_grid_dims(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    resolution: float,
) -> Tuple[int, int]:
    """Compute grid dimensions (nx, ny) from snapped extent and resolution.

    Args:
        xmin: Snapped xmin.
        ymin: Snapped ymin.
        xmax: Snapped xmax.
        ymax: Snapped ymax.
        resolution: Cell size (units of the CRS).

    Returns:
        Tuple of ``(nx, ny)`` where ``nx`` is columns and ``ny`` is rows.

    Raises:
        ValueError: If ``resolution`` is non-positive or extent is invalid.

    Notes:
        - Uses width/height divided by resolution, rounded to nearest integer.
          Extents are expected to be exact multiples after snapping.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("Invalid extent; width/height must be positive.")

    nx = int(round(width / resolution))
    ny = int(round(height / resolution))
    if nx <= 0 or ny <= 0:
        raise ValueError("Computed grid dimensions are non-positive.")
    return nx, ny
