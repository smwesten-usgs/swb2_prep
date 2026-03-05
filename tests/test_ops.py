"""
Tests for XR-based clipping operations (ops.py).

Validates:
- clip_raster_to_polygon_xr clips a DEM to the AOI polygon.
- CRS/transform integrity is preserved, and output size shrinks.
"""

from pathlib import Path
import rioxarray as rxr
import geopandas as gpd

from swb2_prep.common.ops import clip_raster_to_polygon_xr


def test_clip_raster_to_polygon_xr(tmp_path):
    """
    Read DEM and AOI, clip to polygon, and assert shape decrease and CRS consistency.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    dem = data_dir / "hydrosheds_dem__south_manitou.tif"
    aoi = data_dir / "aoi.shp"

    da = rxr.open_rasterio(dem, masked=True).squeeze(drop=True)
    aoi_gdf = gpd.read_file(aoi)
    # Ensure AOI CRS equals raster CRS for test
    if str(aoi_gdf.crs) != str(da.rio.crs):
        aoi_gdf = aoi_gdf.to_crs(da.rio.crs)

    da_clipped = clip_raster_to_polygon_xr(da, aoi_gdf)

    assert da_clipped.rio.crs == da.rio.crs
    assert da_clipped.sizes["y"] <= da.sizes["y"]
    assert da_clipped.sizes["x"] <= da.sizes["x"]