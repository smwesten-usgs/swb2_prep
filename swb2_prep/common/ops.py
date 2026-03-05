# swb_cli/common/ops.py

from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd


def clip_raster_to_polygon(array: np.ndarray, profile: dict, polygon_gdf: gpd.GeoDataFrame):
    """
    Clip a raster to a polygon geometry using rasterio.mask.mask.

    Parameters
    ----------
    array : numpy.ndarray
        2D raster data array (single-band) with shape (rows, cols).
        This array must match the spatial transform and CRS
        described in the `profile` argument.
    profile : dict
        Rasterio profile containing:
        - transform : affine transform for the raster
        - crs       : coordinate reference system
        - width     : number of columns
        - height    : number of rows
        - dtype     : data type
    polygon_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing one or more polygon geometries.
        CRS must match the raster CRS *before* calling this function.
        (Reprojection must be done upstream by reproject_polygon().)

    Returns
    -------
    tuple
        (clipped_array, clipped_profile)
        - clipped_array : numpy.ndarray of the clipped raster values
        - clipped_profile : dict updated with:
            - new width/height
            - updated transform
            - same dtype
            - same CRS
            - driver='GTiff'
            - appropriate nodata if present

    Raises
    ------
    ValueError
        If polygon_gdf has no CRS or does not match raster CRS.

    Notes
    -----
    This is the SWB step:
        "Clip raster → polygon" (Step 7 in NOTES.md)

    This function does NOT reproject. Polygon CRS *must* match
    raster CRS before calling.

    mask() will return the smallest bounding rectangle footprint
    that fully encloses the polygon, with nodata values outside the
    polygon but inside the bounds.
    """

    if polygon_gdf.crs is None:
        raise ValueError("Polygon GeoDataFrame CRS is not set.")

    if str(polygon_gdf.crs) != str(profile["crs"]):
        raise ValueError("Raster CRS and polygon CRS must match before clipping.")

    shapes = [geom.__geo_interface__ for geom in polygon_gdf.geometry]

    from rasterio.io import MemoryFile
    import rasterio

    # Create an in-memory raster and WRITE the array into it
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=profile["height"],
            width=profile["width"],
            count=1,
            dtype=profile["dtype"],
            crs=profile["crs"],
            transform=profile["transform"],
            nodata=profile.get("nodata")
        ) as dataset:

            dataset.write(array, 1)

            clipped_array, clipped_transform = rasterio.mask.mask(
                dataset,
                shapes,
                crop=True,
                filled=True,
                nodata=profile.get("nodata", -9999),
            )

    clipped_array = clipped_array[0]

    new_height, new_width = clipped_array.shape
    clipped_profile = profile.copy()
    clipped_profile.update(
        {
            "height": new_height,
            "width": new_width,
            "transform": clipped_transform,
            "driver": "GTiff",
        }
    )

    return clipped_array, clipped_profile