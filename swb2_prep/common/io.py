# -*- coding: utf-8 -*-
"""
IO utilities for reading/writing rasters in SWB2 preparation workflows.

This module provides both an **xarray/rioxarray-first API** and **NumPy + rasterio**
compatibility wrappers. The XR-first API returns `xarray.DataArray` objects with CRS,
affine transform, and optional NoData masking attached via `rioxarray`. The compatibility
wrappers keep legacy code working where NumPy arrays and rasterio profiles are expected.

Functions included:
- read_raster_xr: Read any raster to an xarray.DataArray (CRS/transform attached).
- read_raster: Legacy: Read a raster to (numpy array, rasterio profile).
- write_geotiff_xr: Write a DataArray to GeoTIFF via rioxarray.
- write_geotiff: Legacy: Write a NumPy array to GeoTIFF via rasterio.
- write_arc_ascii: Write ESRI Arc ASCII Grid (.asc) via rioxarray AAIGrid driver.
- lower_left_to_transform: Helper to derive an Affine from lower-left origin.

Notes
-----
- ESRI ASCII (AAIGrid) is written via GDAL's CreateCopy pathway; precision and formatting
  can be controlled using GDAL creation options (e.g., DECIMAL_PRECISION). See GDAL docs
  and examples.  # See project test suite and references in documentation.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple, Dict

import os

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.transform import Affine, from_origin


__all__ = [
    "read_raster_xr",
    "read_raster",
    "write_geotiff_xr",
    "write_geotiff",
    "write_arc_ascii",
    "lower_left_to_transform",
]


def read_raster_xr(path: Union[str, Path], *, masked: bool = True) -> xr.DataArray:
    """
    Read a raster into an :class:`xarray.DataArray` with CRS/transform via ``rioxarray``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a raster readable by rasterio/GDAL (e.g., GeoTIFF, AAIGrid).
    masked : bool, optional
        If ``True`` (default), pixels with NoData are masked to NaN.

    Returns
    -------
    xarray.DataArray
        For single-band rasters, returns a 2D DataArray with dims ``("y", "x")``.
        For multi-band rasters, the first band is squeezed off if only one band exists;
        otherwise the returned DataArray retains a ``"band"`` dimension.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    rasterio.errors.RasterioIOError
        If GDAL/rasterio cannot open the dataset.

    Notes
    -----
    The returned DataArray has CRS in ``.rio.crs`` and geotransform in ``.rio.transform()``.
    NoData value is available at ``.rio.nodata``. Use ``da.rio.reproject(...)`` and
    ``da.rio.clip(...)`` for further preprocessing.

    Examples
    --------
    >>> da = read_raster_xr("data/hydrosheds_dem__south_manitou.tif")
    >>> da.rio.crs is not None
    True
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    da = rxr.open_rasterio(path, masked=masked)

    # Squeeze band dimension for consistent single-band workflows
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)

    return da


def read_raster(path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Read a raster file using rasterio and return data as a NumPy array plus a metadata profile.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the raster file. Must be readable by rasterio (e.g., GeoTIFF).

    Returns
    -------
    (numpy.ndarray, dict)
        * array : 2D NumPy array from the first band.
        * profile : rasterio profile dict (includes ``crs``, ``transform``, ``width``,
          ``height``, ``dtype``, ``driver``, and related metadata).

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    rasterio.errors.RasterioIOError
        If rasterio fails to read the file.

    Notes
    -----
    This legacy function is kept for compatibility with existing NumPy pipelines. Prefer
    :func:`read_raster_xr` in new code.

    Examples
    --------
    >>> arr, prof = read_raster("data/muraster__south_manitou.tif")
    >>> arr.ndim
    2
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1)  # First band
        profile = src.profile.copy()

    return array, profile


def write_geotiff_xr(
    path: Union[str, Path],
    da: xr.DataArray,
    *,
    dtype: Optional[str] = None,
    **rasterio_kwargs,
) -> Path:
    """
    Write an xarray.DataArray to GeoTIFF using rioxarray.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination .tif path.
    da : xarray.DataArray
        DataArray with CRS and transform attached via ``.rio``.
    dtype : str, optional
        Output dtype, e.g., "float32", "int16". If None, rioxarray/rasterio
        infer dtype from the data.
    **rasterio_kwargs
        Rasterio profile keywords (e.g., ``compress="LZW"``, ``tiled=True``,
        ``blockxsize=256``, ``blockysize=256``).

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError
        If CRS or transform are missing from the DataArray.
    """
    path = Path(path)
    if da.rio.crs is None or da.rio.transform() is None:
        raise ValueError("DataArray must have CRS and transform set via .rio.")

    os.makedirs(path.parent, exist_ok=True)

    if dtype is not None:
        # Ensure DA has requested dtype; NaNs require floating types
        da = da.astype(dtype)

    # Pass standard rasterio keys directly; do not use profile_kwargs for GTiff
    da.rio.to_raster(path, driver="GTiff", **rasterio_kwargs)
    return path

def write_geotiff(path: Union[str, Path], array: np.ndarray, profile: Dict) -> None:
    """
    Write a 2D NumPy array to GeoTIFF via rasterio (legacy path).

    Parameters
    ----------
    path : str or pathlib.Path
        Destination GeoTIFF path.
    array : numpy.ndarray
        2D array (single-band).
    profile : dict
        Rasterio profile including ``driver='GTiff'`` (or will be forced),
        ``dtype``, ``width``, ``height``, ``crs``, and ``transform``.

    Returns
    -------
    None

    Raises
    ------
    rasterio.errors.RasterioIOError
        If the dataset cannot be written.

    Notes
    -----
    This path provides detailed control over compression, tiling, and profile tuning
    when writing GeoTIFFs. For XR-first usage, prefer :func:`write_geotiff_xr`.

    Examples
    --------
    >>> arr, prof = read_raster("data/muraster__south_manitou.tif")
    >>> write_geotiff("out/muraster_copy.tif", arr, prof)
    """
    path = Path(path)
    profile = profile.copy()
    profile.update({"driver": "GTiff"})

    os.makedirs(path.parent, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def lower_left_to_transform(
    xllcorner: float,
    yllcorner: float,
    nrows: int,
    x_res: float,
    y_res: float,
) -> Affine:
    """
    Convert a lower-left origin to a Rasterio/GDAL upper-left Affine transform.

    Parameters
    ----------
    xllcorner : float
        X coordinate of lower-left corner of the raster (Esri ASCII semantics).
    yllcorner : float
        Y coordinate of lower-left corner of the raster.
    nrows : int
        Number of rows in the raster.
    x_res : float
        Pixel width (x resolution).
    y_res : float
        Pixel height (y resolution). For north-up rasters, this should be positive.
    Returns
    -------
    rasterio.transform.Affine
        Affine transform with origin at the upper-left, suitable for rasterio/GDAL.

    Notes
    -----
    Esri ASCII headers store origin at the lower-left; Rasterio/GDAL use upper-left.
    This helper computes ``y_ul = yllcorner + nrows * y_res`` and calls
    :func:`rasterio.transform.from_origin`.

    Examples
    --------
    >>> T = lower_left_to_transform(378923, 4072345, nrows=100, x_res=30, y_res=30)
    >>> isinstance(T, Affine)
    True
    """
    y_ul = yllcorner + nrows * y_res
    return from_origin(xllcorner, y_ul, x_res, y_res)


def write_arc_ascii(
    data: Union[np.ndarray, xr.DataArray],
    out_path: Union[str, Path],
    *,
    transform: Affine,
    crs: Union[str, dict],          # e.g., "EPSG:5070" or rasterio-style dict/WKT
    nodata: Optional[float] = -9999,
    decimal_precision: Optional[int] = None,
    significant_digits: Optional[int] = None,
    force_cellsize: Optional[bool] = None,
) -> Path:
    """
    Write an ESRI Arc ASCII Grid (AAIGrid, ``.asc``) using ``rioxarray``.

    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        2D array (rows, cols). If a DataArray has a ``"band"`` dim, it will be squeezed
        to a single band. If a NumPy array is provided, it will be wrapped as a DataArray
        with dims ``("y", "x")``.
    out_path : str or pathlib.Path
        Destination ASCII grid path (``.asc``).
    transform : rasterio.transform.Affine
        Geotransform with origin at the *upper-left* (Rasterio/GDAL convention).
    crs : str or dict
        CRS as EPSG string (e.g., ``"EPSG:32615"``), WKT, or rasterio-style dict.
    nodata : float or None, optional
        NoData value to encode in the ASCII file; defaults to ``-9999``.
    decimal_precision : int, optional
        GDAL AAIGrid creation option: number of decimal places (e.g., ``3`` makes files smaller).
    significant_digits : int, optional
        GDAL AAIGrid creation option: number of significant digits.
    force_cellsize : bool, optional
        GDAL AAIGrid creation option: ``True`` → ``FORCE_CELLSIZE=YES`` to use the X pixel size
        as ``CELLSIZE`` when pixels are not perfectly square.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError
        If ``data`` is not 2D, or CRS/transform are missing when required.

    Notes
    -----
    This uses the GDAL AAIGrid driver via rioxarray's ``to_raster(..., driver="AAIGrid")``.
    Options like ``DECIMAL_PRECISION`` and ``SIGNIFICANT_DIGITS`` are passed through
    ``profile_kwargs`` to GDAL. For multi-band data, ASCII output is single-band only
    (first band selected).

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.random.rand(50, 60).astype("float32")
    >>> T = lower_left_to_transform(378923, 4072345, 50, 30, 30)
    >>> _ = write_arc_ascii(arr, "out/synthetic.asc", transform=T, crs="EPSG:32616", decimal_precision=2)
    """
    out_path = Path(out_path)

    # --- Normalize to a single-band DataArray ---
    if isinstance(data, xr.DataArray):
        da = data
        if "band" in da.dims:
            da = da.isel(band=0).squeeze(drop=True)
    else:
        # Assume 2D numpy array
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("`data` must be a 2D array for AAIGrid.")
        da = xr.DataArray(arr, dims=("y", "x"), name="band1")

    # --- Attach geospatial metadata via rioxarray ---
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)
    if nodata is not None:
        da = da.rio.write_nodata(nodata)

    # --- Build GDAL AAIGrid creation options ---
    options = []
    if decimal_precision is not None:
        options.append(f"DECIMAL_PRECISION={int(decimal_precision)}")
    if significant_digits is not None:
        options.append(f"SIGNIFICANT_DIGITS={int(significant_digits)}")
    if force_cellsize is not None:
        options.append(f"FORCE_CELLSIZE={'YES' if force_cellsize else 'NO'}")

    profile_kwargs = {}
    if options:
        profile_kwargs["options"] = options

    os.makedirs(out_path.parent, exist_ok=True)

    # --- Write ASCII grid ---
    da.rio.to_raster(
        out_path,
        driver="AAIGrid",
        profile_kwargs=profile_kwargs,
    )

    return out_path