# tests/test_ops.py

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_origin

from swb2_prep.common.ops import clip_raster_to_polygon


def test_clip_raster_to_polygon():
    """
    Test clipping a simple 4x4 raster to a rectangular polygon.

    Inputs
    ------
    array : 4x4 numpy array
        Synthetic raster with known values for easy validation.
    profile : dict
        Rasterio profile with:
            - CRS: EPSG:3857
            - transform: from_origin(0, 40, 10, 10)
            - width: 4
            - height: 4
    polygon_gdf : GeoDataFrame
        Simple bounding-box polygon (10,10)-(30,30)
        Also in EPSG:3857, matching raster CRS.

    Expected Output
    ---------------
    clipped_array : 2x2 portion of raster corresponding to the
        central region.
    clipped_profile : updated profile with:
        - height == 2
        - width == 2
        - transform updated to upper-left corner of clipped region
    """
    # Full raster 4x4
    array = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=float,
    )

    # Spatial config: upper-left at (0, 40), cell 10x10
    transform = from_origin(0, 40, 10, 10)
    profile = {
        "crs": "EPSG:3857",
        "transform": transform,
        "width": 4,
        "height": 4,
        "dtype": "float32",
        "count": 1,
    }

    # Clip area: box from (10, 10) to (30, 30)
    # This corresponds to rows 1–2, cols 1–2 in the raster
    poly = box(10, 10, 30, 30)
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:3857")

    clipped_array, clipped_profile = clip_raster_to_polygon(array, profile, gdf)

    # Validate array shape and values
    assert clipped_array.shape == (2, 2)
    print(clipped_array[0, 0])
    print(clipped_array[0, 1])
    print(clipped_array[1, 0])
    print(clipped_array[1, 1])
    # Should be values:
    # [[6, 7],
    #  [10, 11]]
    assert clipped_array[0, 0] == 6
    assert clipped_array[0, 1] == 7
    assert clipped_array[1, 0] == 10
    assert clipped_array[1, 1] == 11

    # Validate profile width/height
    assert clipped_profile["width"] == 2
    assert clipped_profile["height"] == 2