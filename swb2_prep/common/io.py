from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

def read_raster(path: Path):
    """
    Read a raster file (e.g., GeoTIFF) from disk using rasterio.

    Parameters
    ----------
    path : Path
        Path to the raster file. The file must be readable by rasterio
        (e.g., GeoTIFF, ArcASCII converted, etc.).

    Returns
    -------
    tuple
        A tuple (array, profile) where:
        - array : numpy.ndarray
            The raster data read into a NumPy array. This will be
            a 2D array for single-band rasters.
        - profile : dict
            The rasterio profile containing raster metadata such as:
            CRS, transform, width, height, dtype, driver, etc.

    Raises
    ------
    FileNotFoundError
        If the provided raster path does not exist.
    rasterio.errors.RasterioIOError
        If rasterio fails to read the file.

    Notes
    -----
    This function does not modify the raster (no reprojection, no
    resampling). It simply loads data and metadata as-is.

    This is one of the core utilities used by preprocessing CLIs like
    prep_landuse_input.py to bring input data into memory for further
    processing steps such as reprojection, clipping, and writing
    standardized outputs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1)  # First band (landuse and soils are single-band)
        profile = src.profile.copy()

    return array, profile

def write_geotiff(path: Path, array: np.ndarray, profile: dict):
    """
    Write a NumPy array to a GeoTIFF file using rasterio.

    Parameters
    ----------
    path : Path
        Destination file path where the GeoTIFF will be written.
    array : numpy.ndarray
        The raster values to be written. Must be 2D (single-band).
    profile : dict
        A rasterio profile that includes required metadata fields
        such as: driver='GTiff', dtype, width, height, crs, transform.

    Returns
    -------
    None

    Raises
    ------
    rasterio.errors.RasterioIOError
        If the file cannot be written.

    Notes
    -----
    The profile should already be adjusted to match the array,
    including width/height and transform if resampling or clipping
    occurred upstream.
    """
    path = Path(path)
    profile = profile.copy()
    profile.update({"driver": "GTiff"})

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)

def write_arc_ascii(path: Path, array: np.ndarray, profile: dict):
    """
    Write a NumPy array to an ArcASCII grid file.

    Parameters
    ----------
    path : Path
        Destination ASCII grid filename.
    array : numpy.ndarray
        2D raster data array.
    profile : dict
        Raster metadata profile, from which the transform, width,
        height, and nodata are extracted.

    Returns
    -------
    None

    Notes
    -----
    ArcASCII format requires a fixed header structure. This writer
    extracts the necessary components from the rasterio profile.

    Assumptions
    -----------
    - The raster uses an affine transform where:
      transform.a = pixel width
      transform.e = pixel height (negative)
      transform.c = x-origin (upper-left)
      transform.f = y-origin (upper-left)
    - Nodata should be set in the profile for consistent output.
    """
    path = Path(path)
    transform: Affine = profile["transform"]
    ncols = profile["width"]
    nrows = profile["height"]
    xllcorner = transform.c
    yllcorner = transform.f + (transform.e * nrows)  # derive lower-left corner
    cellsize = transform.a
    nodata = profile.get("nodata", -9999)

    with path.open("w") as f:
        f.write(f"NCOLS {ncols}\n")
        f.write(f"NROWS {nrows}\n")
        f.write(f"XLLCORNER {xllcorner}\n")
        f.write(f"YLLCORNER {yllcorner}\n")
        f.write(f"CELLSIZE {cellsize}\n")
        f.write(f"NODATA_VALUE {nodata}\n")

        # Write data rows (ArcASCII expects row-major order, top to bottom)
        for row in array:
            f.write(" ".join(str(float(val)) for val in row))
            f.write("\n")