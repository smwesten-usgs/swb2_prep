"""
Tests for XR-based grid operations (grids.py).

Validates:
- reproject_raster_xr to a target CRS and resolution.
- resample_raster_xr to a target resolution in same CRS.
- reproject_polygon CRS update.
"""

from pathlib import Path
import numpy as np
import rioxarray as rxr
from rasterio.warp import Resampling
import geopandas as gpd

from swb2_prep.common.grids import (
    reproject_raster_xr,
    resample_raster_xr,
    reproject_polygon,
)


def _affine_close(a1, a2, atol=1e-12):
    return (
        np.isclose(a1.a, a2.a, atol=atol)
        and np.isclose(a1.b, a2.b, atol=atol)
        and np.isclose(a1.c, a2.c, atol=atol)
        and np.isclose(a1.d, a2.d, atol=atol)
        and np.isclose(a1.e, a2.e, atol=atol)
        and np.isclose(a1.f, a2.f, atol=atol)
    )


def test_resample_raster_xr(tmp_path):
    """
    Resample DEM to a coarser resolution in same CRS and confirm pixel size changes.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    dem = data_dir / "hydrosheds_dem__south_manitou.tif"

    da = rxr.open_rasterio(dem, masked=True).squeeze(drop=True)
    original_res = float(da.rio.transform().a)
    target_res = original_res * 2

    da_rs = resample_raster_xr(da, target_resolution=target_res, resampling=Resampling.bilinear)
    assert np.isclose(float(da_rs.rio.transform().a), target_res, atol=1e-12)
    assert da_rs.rio.crs == da.rio.crs


def test_reproject_polygon(tmp_path):
    """
    Reproject AOI polygon and verify CRS update.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    aoi = data_dir / "aoi.shp"
    gdf = gpd.read_file(aoi)

    target_crs = "EPSG:5070"
    gdf2 = reproject_polygon(gdf, target_crs)
    assert str(gdf2.crs) == target_crs
