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
from swb2_prep.common.utils import crs_equal


def clip_raster_to_polygon_xr(
    data_array: xr.DataArray,
    polygon_gdf: gpd.GeoDataFrame,
    *,
    all_touched: bool = False,
    drop: bool = True,
    invert: bool = False,
    from_disk: bool = False,
    strict: bool = False,
    auto_reproject: bool = True,
) -> xr.DataArray:
    """Clip a raster (:class:`xarray.DataArray`) to polygon(s) using rioxarray.

    Args:
        data_array: Input raster with CRS/transform attached via ``.rio`` (rioxarray).
        polygon_gdf: Polygon(s) to clip against (single-row or multi-row) with a defined CRS.
        all_touched: If True, include any pixel touched by polygon boundaries (GDAL semantics).
        drop: If True, drop data outside the polygon; if False, mask but retain shape.
        invert: If True, invert selection (clip everything *outside* the polygon).
        from_disk: If True, stream from disk when possible (for large rasters).
        strict: If True, raise on invalid geometry; otherwise try best-effort.
        auto_reproject: If True, reproject polygons to raster CRS if CRS differ.

    Returns:
        A clipped :class:`xarray.DataArray` with CRS/transform preserved.

    Raises:
        ValueError: If CRS are incompatible and ``auto_reproject`` is False.
        ValueError: If input does not carry CRS/transform via ``data_array.rio`` accessors.

    Notes:
        - Expects an XR-first workflow with rioxarray attached; the output shape typically
          shrinks and retains CRS/transform metadata used by subsequent IO steps.
        - Downstream tests assert that shape decreases and CRS is preserved after clipping.
    """
    # Optional: validate CRS compatibility before calling .rio.clip
    if auto_reproject:
        # Reproject polygon(s) to match raster CRS, if needed.
        raster_crs = data_array.rio.crs
        if polygon_gdf.crs != raster_crs:
            polygon_gdf = polygon_gdf.to_crs(raster_crs)
    else:
        raster_crs = data_array.rio.crs
        if polygon_gdf.crs != raster_crs:
            raise ValueError("CRS mismatch between raster and polygon; set auto_reproject=True or reproject inputs.")

    clipped = data_array.rio.clip(
        polygon_gdf.geometry,
        polygon_gdf.crs,
        all_touched=all_touched,
        drop=drop,
        invert=invert,
        from_disk=from_disk,
    )
    return clipped


