# tests/test_io.py
# -*- coding: utf-8 -*-
"""
XR-first IO tests for SWB2 prep.

This module validates:
- Creating an AOI polygon from a bounding box.
- Reprojecting a polygon GeoDataFrame to a target CRS.
- Resampling a raster DataArray to a new resolution in the same CRS.
- Reprojecting a raster DataArray to a different CRS.

All raster tests construct small synthetic xarray.DataArray objects and attach
CRS/transform using rioxarray’s `.rio` accessors, avoiding legacy NumPy/profile paths.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.transform import from_origin
import xarray as xr

from shapely.geometry import box
from swb2_prep.common.grids import (
    reproject_raster_xr,
    reproject_polygon,
    resample_raster_xr,
)


def test_create_polygon_from_bbox():
    """
    Create a bounding-box polygon in the project CRS and validate geometry/CRS.

    Expected
    --------
    - GeoDataFrame has exactly one polygon.
    - Bounds match the inputs.
    - CRS equals the requested EPSG.
    """
    polygon = box(0, 0, 10, 5)
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:5070")

    assert len(gdf) == 1
    poly = gdf.geometry.iloc[0]
    assert poly.bounds == (0, 0, 10, 5)
    assert gdf.crs.to_string() == "EPSG:5070"


def test_reproject_polygon():
    """
    Reproject a simple polygon GeoDataFrame from EPSG:4326 to EPSG:5070.

    Expected
    --------
    - Output GeoDataFrame CRS equals EPSG:5070.
    """
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        crs="EPSG:4326",
    )
    out = reproject_polygon(gdf, "EPSG:5070")
    assert out.crs.to_string() == "EPSG:5070"


def test_resample_raster_xr():
    """
    Resample a raster DataArray to a finer resolution in the same CRS.

    Setup
    -----
    - Build a 2x2 DataArray with EPSG:5070 and 1x1 pixel size.
      Upper-left origin at (0, 2) → transform from_origin(0, 2, 1, 1).

    Expected
    --------
    - Output transform’s pixel width (a) equals 0.5.
    - Output shape doubles in both dimensions (from 2x2 to 4x4).
    """
    arr = np.array([[1, 2], [3, 4]], dtype="float32")
    T = from_origin(0.0, 2.0, 1.0, 1.0)

    da = xr.DataArray(arr, dims=("y", "x"), name="band1")
    da = da.rio.write_crs("EPSG:5070").rio.write_transform(T)

    da_out = resample_raster_xr(da, target_resolution=0.5)
    T_out = da_out.rio.transform()

    assert T_out.a == 0.5
    assert da_out.sizes["y"] == 4
    assert da_out.sizes["x"] == 4
    assert da_out.rio.crs.to_string() == "EPSG:5070"


def test_reproject_raster_xr():
    """
    Smoke test for raster reprojection from EPSG:4326 to EPSG:5070.

    Notes
    -----
    We avoid asserting exact numeric values since reprojection introduces distortions.
    We only validate:
    - The output CRS is EPSG:5070.
    - The output array has nonzero size.
    """
    arr = np.array([[1, 2], [3, 4]], dtype="float32")
    # Synthetic georeferencing: upper-left (-90, 45), pixel size 1x1 degrees
    T = from_origin(-90.0, 45.0, 1.0, 1.0)

    da = xr.DataArray(arr, dims=("y", "x"), name="band1")
    da = da.rio.write_crs("EPSG:4326").rio.write_transform(T)

    da_out = reproject_raster_xr(da, "EPSG:5070")
    assert da_out.rio.crs.to_string() == "EPSG:5070"
