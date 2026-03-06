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

def reproject_raster_xr(
    da: xr.DataArray,
    target_crs: Union[str, dict],
    *,
    resolution: Optional[float] = None,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """
    Reproject a raster :class:`xarray.DataArray` to a target CRS using rioxarray.

    Parameters
    ----------
    da : xarray.DataArray
        Input raster with CRS/transform present via ``da.rio``.
    target_crs : str or dict
        Target CRS (e.g., ``"EPSG:5070"``). May also be a rasterio-style dict or WKT.
    resolution : float, optional
        Desired pixel size in the target CRS units. If ``None``, rioxarray determines
        appropriate resolution from the transform.
    resampling : rasterio.warp.Resampling, optional
        Resampling method (default: bilinear). Other common choices include
        ``Resampling.nearest`` and ``Resampling.cubic``.

    Returns
    -------
    xarray.DataArray
        Reprojected raster as a DataArray. CRS and transform are updated.

    Raises
    ------
    ValueError
        If the input DataArray lacks CRS or transform.

    Notes
    -----
    Under the hood, rioxarray uses rasterio/GDAL. This is analogous to calling
    rasterio.warp.reproject but with a cleaner API for xarray objects.

    Examples
    --------
    >>> da_proj = reproject_raster_xr(da, "EPSG:5070", resolution=30.0)
    >>> float(da_proj.rio.transform().a)
    30.0
    """
    if da.rio.crs is None or da.rio.transform() is None:
        raise ValueError("Input DataArray must have CRS and transform (via .rio).")

    return da.rio.reproject(
        target_crs,
        resolution=resolution,
        resampling=resampling,
    )


def resample_raster_xr(
    da: xr.DataArray,
    target_resolution: float,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """
    Resample a raster :class:`xarray.DataArray` to a new resolution in the
    **same CRS** using rioxarray.

    Parameters
    ----------
    da : xarray.DataArray
        Input raster with CRS/transform.
    target_resolution : float
        Desired pixel size in CRS units (e.g., meters for EPSG:5070).
    resampling : rasterio.warp.Resampling, optional
        Resampling method (default: bilinear).
        Other methods include 'nearest', 'bilinear', 'cubic', 'cubic_spline',
           'lanczos', 'average', 'mode'

    Returns
    -------
    xarray.DataArray
        Resampled raster with updated transform and shape.

    Raises
    ------
    ValueError
        If the input DataArray lacks CRS or transform.

    Notes
    -----
    This is implemented as a reprojection **to the same CRS** with a
    different ``resolution``. It’s equivalent to rasterio’s resampling
    but handled via rioxarray’s API.

    Examples
    --------
    >>> da_rs = resample_raster_xr(da, target_resolution=30.0)
    >>> float(da_rs.rio.transform().a)
    30.0
    """
    if da.rio.crs is None or da.rio.transform() is None:
        raise ValueError("Input DataArray must have CRS and transform (via .rio).")

    return da.rio.reproject(
        da.rio.crs,
        resolution=target_resolution,
        resampling=resampling,
    )


def reproject_polygon(
    gdf: gpd.GeoDataFrame,
    target_crs: Union[str, dict],
) -> gpd.GeoDataFrame:
    """
    Reproject a polygon GeoDataFrame to the target CRS.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with a geometry column and a set CRS.
    target_crs : str or dict
        Target CRS (e.g., ``"EPSG:5070"``).

    Returns
    -------
    geopandas.GeoDataFrame
        Reprojected GeoDataFrame with updated ``.crs``.

    Raises
    ------
    ValueError
        If the input GeoDataFrame has no CRS set.

    Examples
    --------
    >>> gdf2 = reproject_polygon(gdf, "EPSG:5070")
    >>> str(gdf2.crs)
    'EPSG:5070'
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame CRS is not set.")

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