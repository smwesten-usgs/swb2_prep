# swb2_prep/common/ops.py
# -*- coding: utf-8 -*-
"""
Common raster/vector operations built on xarray/rioxarray.

This module provides routines for clipping rasters to polygons,
and creating AOI polygons from bounding boxes. Rasters are expected
to be handled as :class:`xarray.DataArray` objects with CRS/transform info
attached via the rioxarray accessor (``.rio``).
"""

from __future__ import annotations

from typing import Iterable
import xarray as xr
import geopandas as gpd
from pyproj import CRS as _CRS


def _crs_equal(crs_a, crs_b) -> bool:
    """
    Robust CRS equivalence using :class:`pyproj.CRS.equals`.

    Parameters
    ----------
    crs_a, crs_b : Any
        CRS in any format accepted by ``pyproj.CRS.from_user_input``
        (e.g., EPSG string, WKT, dict).

    Returns
    -------
    bool
        ``True`` if CRSs are equivalent, ``False`` otherwise.

    Notes
    -----
    String comparisons of CRS (e.g., ``str(crs_a) == str(crs_b)``) are fragile
    because of differences in WKT serialization, axis order, or EPSG variants.
    Prefer a semantic comparison using ``pyproj.CRS.equals``.
    """
    try:
        return _CRS.from_user_input(crs_a).equals(_CRS.from_user_input(crs_b))
    except Exception:
        return False


def clip_raster_to_polygon_xr(
    da: xr.DataArray,
    polygon_gdf: gpd.GeoDataFrame,
    *,
    all_touched: bool = False,
    drop: bool = True,
    invert: bool = False,
    from_disk: bool = False,
    strict: bool = False,
    auto_reproject: bool = True,
) -> xr.DataArray:
    """
    Clip a raster :class:`xarray.DataArray` to the supplied polygon(s) using
    ``rioxarray.DataArray.rio.clip``.

    Parameters
    ----------
    da : xarray.DataArray
        Input raster with CRS and transform available via ``da.rio``.
        Typically produced by an XR-first read (e.g., ``rioxarray.open_rasterio``).
    polygon_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing one or more polygon geometries.
        Its CRS may differ from the raster’s CRS.
    all_touched : bool, optional
        If ``True``, include any pixel touched by the geometry boundary.
        Defaults to ``False`` (include only pixels whose center is within).
    drop : bool, optional
        If ``True`` (default), drop non-overlapping data outside polygon bounds.
    invert : bool, optional
        If ``True``, clip the areas *outside* the geometry (inverse mask).
    from_disk : bool, optional
        If ``True``, geometry will be masked with on-disk reads. Defaults to ``False``.
    strict : bool, optional
        If ``True``, raise on CRS mismatch (no auto reprojection). Default ``False``.
    auto_reproject : bool, optional
        If ``True`` and CRS mismatch is detected, auto-reproject polygons to ``da.rio.crs``.
        Default ``True``.

    Returns
    -------
    xarray.DataArray
        Clipped raster as a DataArray. CRS and transform are preserved.
        Cells outside the polygon are set to NoData (``da.rio.nodata``).

    Raises
    ------
    ValueError
        If the polygon GeoDataFrame has no CRS, the raster has no CRS,
        or if a CRS mismatch occurs with ``strict=True`` and ``auto_reproject=False``.

    Notes
    -----
    This operation uses rioxarray’s clip which is backed by rasterio/GDAL.

    - Robust CRS equivalence is checked via ``pyproj.CRS.equals``.
    - If CRSs differ and ``auto_reproject=True``, polygons are reprojected to
      the raster CRS before clipping.

    Examples
    --------
    >>> clipped = clip_raster_to_polygon_xr(da, aoi_gdf)  # auto-reprojects if needed
    >>> clipped.rio.crs == da.rio.crs
    True
    """
    if polygon_gdf.crs is None:
        raise ValueError("Polygon GeoDataFrame CRS is not set.")
    if da.rio.crs is None:
        raise ValueError("Raster DataArray CRS is not set.")

    # Robust CRS equivalence check
    if not _crs_equal(polygon_gdf.crs, da.rio.crs):
        if strict or not auto_reproject:
            raise ValueError("Raster CRS and polygon CRS must match before clipping.")
        # Auto-reproject polygons to raster CRS
        polygon_gdf = polygon_gdf.to_crs(da.rio.crs)

    geometries: Iterable = polygon_gdf.geometry
    # rioxarray clip expects an iterable of shapely geometries and the CRS
    return da.rio.clip(
        geometries,
        polygon_gdf.crs,
        all_touched=all_touched,
        drop=drop,
        invert=invert,
        from_disk=from_disk,
    )


def create_polygon_from_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs: str,
) -> gpd.GeoDataFrame:
    """
    Create a single-row GeoDataFrame polygon from a bounding box.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : float
        Bounding-box coordinates in the target/project CRS.
    crs : str
        CRS string (e.g., ``"EPSG:5070"``).

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame with the bounding-box polygon in
        the specified CRS.

    Notes
    -----
    This supports the CLI mode that supplies an AOI as a bounding box.

    Examples
    --------
    >>> gdf = create_polygon_from_bbox(0, 0, 1000, 1000, "EPSG:5070")
    >>> gdf.crs.to_string()
    'EPSG:5070'
    """
    from shapely.geometry import box

    geom = box(xmin, ymin, xmax, ymax)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)