# swb_cli/common/grids.py
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import box

def reproject_raster(array: np.ndarray, profile: dict, target_crs: str):
    """
    Reproject a raster array + rasterio profile to a new CRS using
    rasterio’s built-in reprojection utilities.

    Parameters
    ----------
    array : numpy.ndarray
        2D raster data array with shape (rows, cols). Single-band only.
    profile : dict
        Rasterio profile for the input raster. Must contain keys:
        - 'crs'
        - 'transform'
        - 'width'
        - 'height'
    target_crs : str
        Target coordinate reference system (e.g., "EPSG:5070").
        Must be compatible with rasterio CRS parsing.

    Returns
    -------
    tuple
        A tuple (array_out, profile_out):
        - array_out : numpy.ndarray
            Reprojected raster values as a 2D float32 array.
        - profile_out : dict
            Updated rasterio profile with:
            - new CRS
            - new transform
            - new width/height
            - dtype=float32
            - driver='GTiff' (safe default)

    Notes
    -----
    This uses rasterio.warp.reproject and is the most transparent
    way to handle CRS transformation without abstractions. Grid
    resolution may change during reprojection depending on the CRS.

    This function does NOT clip or resample to match a target resolution.
    That is handled separately in resample_raster().
    """
    src_crs = profile["crs"]
    transform = profile["transform"]
    height = profile["height"]
    width = profile["width"]

    # Correct way to compute bounds for calculate_default_transform
    left, bottom, right, top = rasterio.transform.array_bounds(
        height, width, transform
    )

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, width, height, left, bottom, right, top
    )

    dst_profile = profile.copy()
    dst_profile.update(
        {
            "crs": target_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "dtype": "float32",
            "driver": "GTiff",
        }
    )

    array_out = np.zeros((dst_height, dst_width), dtype=np.float32)

    reproject(
        source=array,
        destination=array_out,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )

    return array_out, dst_profile

def reproject_polygon(gdf: gpd.GeoDataFrame, target_crs: str):
    """
    Reproject a GeoDataFrame polygon to the target CRS.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain a geometry column. CRS must be set.
    target_crs : str
        Target CRS string (e.g., "EPSG:5070").

    Returns
    -------
    geopandas.GeoDataFrame
        Reprojected GeoDataFrame with updated CRS.

    Raises
    ------
    ValueError
        If the input GeoDataFrame has no CRS set.

    Notes
    -----
    This function performs no clipping or geometry cleaning.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame CRS is not set.")

    return gdf.to_crs(target_crs)


def resample_raster(array: np.ndarray, profile: dict, target_resolution: float):
    """
    Resample a raster to a new resolution in the SAME CRS.
    Resolution refers to the pixel size in CRS units (meters for EPSG:5070).

    Parameters
    ----------
    array : numpy.ndarray
        2D raster data array.
    profile : dict
        Rasterio profile with CRS, transform, width, height.
    target_resolution : float
        Desired pixel size in CRS units.

    Returns
    -------
    tuple
        (array_out, profile_out)

        - array_out : numpy.ndarray
            Resampled array using bilinear interpolation.
        - profile_out : dict
            Updated rasterio profile including:
            - width/height
            - transform with target pixel size

    Notes
    -----
    This function assumes square pixels.

    This step corresponds to Step 6 in the SWB preprocessing workflow:
      - Reproject
      - THEN resample to project resolution
      - THEN clip
    (As documented in NOTES.md.)
    """
    transform = profile["transform"]
    src_res_x = transform.a
    # Compute new shape
    scale_factor = src_res_x / target_resolution

    new_height = int(profile["height"] * scale_factor)
    new_width = int(profile["width"] * scale_factor)

    # New transform
    new_transform = Affine(
        target_resolution, transform.b, transform.c,
        transform.d, -target_resolution, transform.f
    )

    dst_profile = profile.copy()
    dst_profile.update(
        {
            "transform": new_transform,
            "width": new_width,
            "height": new_height,
            "dtype": "float32",
            "driver": "GTiff",
        }
    )

    array_out = np.zeros((new_height, new_width), dtype=np.float32)

    reproject(
        source=array,
        destination=array_out,
        src_transform=transform,
        src_crs=profile["crs"],
        dst_transform=new_transform,
        dst_crs=profile["crs"],
        resampling=Resampling.bilinear,
    )

    return array_out, dst_profile


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