import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.transform import from_origin

from swb2_prep.common.grids import (
    reproject_raster,
    reproject_polygon,
    resample_raster,
    create_polygon_from_bbox,
)

def test_create_polygon_from_bbox():
    """
    Test creating a bounding-box polygon in project CRS.

    Inputs
    ------
    xmin, ymin, xmax, ymax : floats
        Coordinates in project CRS.
    crs : str
        CRS string.

    Expected Output
    ---------------
    A GeoDataFrame with exactly one polygon whose bounding box
    matches the requested limits.
    """
    gdf = create_polygon_from_bbox(0, 0, 10, 5, "EPSG:5070")

    assert len(gdf) == 1
    poly = gdf.geometry.iloc[0]
#    poly = gdf['geometry']
    assert poly.bounds == (0, 0, 10, 5)
    assert gdf.crs.to_string() == "EPSG:5070"

def test_reproject_polygon():
    """
    Test polygon reprojection with GeoPandas.

    Inputs
    ------
    gdf : GeoDataFrame
        With a simple rectangle and CRS EPSG:4326.
    target_crs : str
        "EPSG:5070"

    Expected Output
    ---------------
    A reprojected GeoDataFrame with CRS EPSG:5070 and a polygon
    in projected coordinates.
    """
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        crs="EPSG:4326",
    )
    out = reproject_polygon(gdf, "EPSG:5070")
    assert out.crs.to_string() == "EPSG:5070"

def test_resample_raster():
    """
    Test raster resampling to different resolution.

    Inputs
    ------
    array : 2x2 numpy array
    profile : rasterio profile with 1x1 pixel size
    target_resolution : float

    Expected Output
    ---------------
    A resampled array with new dimensions determined by scale factor.
    """
    array = np.array([[1, 2], [3, 4]], dtype=float)
    transform = from_origin(0, 2, 1, 1)

    profile = {
        "crs": "EPSG:5070",
        "transform": transform,
        "width": 2,
        "height": 2,
        "count": 1,
    }

    array_out, profile_out = resample_raster(array, profile, target_resolution=0.5)

    assert profile_out["transform"].a == 0.5
    assert array_out.shape == (4, 4)

def test_reproject_raster():
    """
    Smoke test for raster reprojection.

    Notes
    -----
    We do not check exact numeric values because the transformation
    between EPSG:4326 and EPSG:5070 introduces nontrivial distortions.
    Instead, we verify:
    - The output CRS is correct
    - The output array is non-empty
    """
    array = np.array([[1, 2], [3, 4]], dtype=float)
    transform = from_origin(-90, 45, 1, 1)

    profile = {
        "crs": "EPSG:4326",
        "transform": transform,
        "width": 2,
        "height": 2,
        "count": 1,
    }

    array_out, profile_out = reproject_raster(array, profile, "EPSG:5070")
    assert str(profile_out["crs"]) == "EPSG:5070"
    assert array_out.size > 0