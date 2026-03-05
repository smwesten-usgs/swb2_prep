# -*- coding: utf-8 -*-
"""
IO utilities for reading/writing rasters in SWB2 preparation workflows.

This module provides both an **xarray/rioxarray-first API** and **NumPy + rasterio**
compatibility wrappers. The XR-first API returns `xarray.DataArray` objects with CRS,
affine transform, and optional NoData masking attached via `rioxarray`. The compatibility
wrappers keep legacy code working where NumPy arrays and rasterio profiles are expected.

Functions included
------------------
- read_raster_xr: Read any raster to an xarray.DataArray (CRS/transform attached).
- ensure_dtype_and_nodata: Ensure a DataArray has the requested dtype and a representable nodata.
- write_geotiff_xr: Write a DataArray to GeoTIFF via rioxarray with explicit dtype/nodata.
- write_arc_ascii: Write ESRI Arc ASCII Grid (.asc) via rioxarray AAIGrid driver, typed.
- lower_left_to_transform: Helper to derive an Affine from lower-left origin.

Notes
-----
- ESRI ASCII (AAIGrid) is written via GDAL's CreateCopy pathway; precision and formatting
  can be controlled using GDAL creation options (e.g., DECIMAL_PRECISION). See the project
  test suite and documentation for examples.
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
    "ensure_dtype_and_nodata",
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

def ensure_dtype_and_nodata(
    da: xr.DataArray,
    *,
    dtype: str,
    nodata: Optional[Union[int, float]],
    auto_cast_float32: bool = False,
) -> xr.DataArray:
    """
    Ensure a DataArray has the requested dtype and a representable nodata.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with CRS/transform via ``.rio``.
    dtype : str
        Target dtype (e.g., ``"uint8"``, ``"int16"``, ``"float32"``).
    nodata : int or float, optional
        Desired nodata sentinel. Must be representable in ``dtype``.
    auto_cast_float32 : bool, optional
        If ``True`` and nodata is not representable in an integer dtype, cast the
        DataArray to ``float32`` and set nodata as float. If ``False`` (default),
        raise :class:`ValueError` in that case.

    Returns
    -------
    xarray.DataArray
        DataArray cast to ``dtype`` with nodata written via ``.rio.write_nodata``.

    Raises
    ------
    ValueError
        If the nodata value cannot be represented by the requested dtype
        and ``auto_cast_float32`` is ``False``.

    Notes
    -----
    - For integer dtypes, nodata must fall within the dtype range.
    - For float dtypes, nodata is stored as a float value (e.g., ``-9999.0``).
    - Use ``auto_cast_float32=True`` when you want to keep a standard float nodata
      (e.g., ``-9999``) even if the input is categorical integers.
    """
    target = np.dtype(dtype)
    da_out = da.astype(target)

    if nodata is not None:
        if np.issubdtype(target, np.integer):
            info = np.iinfo(target)
            if not (info.min <= nodata <= info.max):
                if auto_cast_float32:
                    da_out = da.astype("float32")
                    da_out = da_out.rio.write_nodata(float(nodata))
                    return da_out
                raise ValueError(
                    f"nodata={nodata} cannot be represented by dtype={dtype} "
                    f"(range {info.min}..{info.max})"
                )
            da_out = da_out.rio.write_nodata(int(nodata))
        else:
            da_out = da_out.rio.write_nodata(float(nodata))

    return da_out


def write_geotiff_xr(
    path: Union[str, Path],
    da: xr.DataArray,
    *,
    dtype: str,
    nodata: Optional[Union[int, float]],
    auto_cast_float32: bool = False,
    **rasterio_kwargs,
) -> Path:
    """
    Write a typed GeoTIFF using rioxarray.

    Parameters
    ----------
    path : str or pathlib.Path
        Output GeoTIFF path.
    da : xarray.DataArray
        DataArray with CRS/transform via ``.rio``.
    dtype : str
        Target dtype (e.g., ``"uint8"``, ``"int16"``, ``"float32"``).
    nodata : int or float, optional
        Nodata sentinel (must be representable in dtype unless ``auto_cast_float32=True``).
    auto_cast_float32 : bool, optional
        If ``True``, automatically cast to float32 when nodata cannot be represented
        in the requested integer dtype (e.g., ``-9999`` for categorical rasters).
    **rasterio_kwargs
        Rasterio profile options (e.g., ``compress="LZW"``, ``tiled=True``,
        ``blockxsize=256``).

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError
        If CRS or transform are missing from ``da`` or nodata is incompatible and
        ``auto_cast_float32=False``.
    """
    path = Path(path)
    if da.rio.crs is None or da.rio.transform() is None:
        raise ValueError("DataArray must have CRS and transform set via .rio.")

    os.makedirs(path.parent, exist_ok=True)

    da_t = ensure_dtype_and_nodata(
        da, dtype=dtype, nodata=nodata, auto_cast_float32=auto_cast_float32
    )

    # Pass rasterio-style kwargs directly; do not use profile_kwargs for GTiff
    da_t.rio.to_raster(path, driver="GTiff", **rasterio_kwargs)
    return path

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
    out_path: Union[str, Path],
    da: xr.DataArray,
    *,
    dtype: str,
    nodata: Optional[Union[int, float]],
    transform: Affine,
    crs: Union[str, dict],
    decimal_precision: Optional[int] = None,
    significant_digits: Optional[int] = None,
    force_cellsize: Optional[bool] = None,
    auto_cast_float32: bool = False,
) -> Path:
    """
    Write an ESRI Arc ASCII Grid (AAIGrid, ``.asc``) using ``rioxarray`` with explicit dtype/nodata.

    Parameters
    ----------
    da : xarray.DataArray
        2D DataArray (rows, cols). If a ``"band"`` dim exists, the first band is selected.
    out_path : str or pathlib.Path
        Destination ASCII grid path (``.asc``).
    dtype : str
        Target dtype (e.g., ``"uint8"``, ``"int16"``, ``"float32"``).
    nodata : int or float, optional
        Nodata sentinel; must be representable in dtype unless ``auto_cast_float32=True``.
    transform : rasterio.transform.Affine
        Geotransform with origin at the *upper-left* (Rasterio/GDAL convention).
    crs : str or dict
        CRS as EPSG string (e.g., ``"EPSG:32615"``), WKT, or rasterio-style dict.
    decimal_precision : int, optional
        GDAL AAIGrid creation option: number of decimal places (e.g., ``0`` for categorical;
        ``3`` for continuous). Mutually exclusive with ``significant_digits``.
    significant_digits : int, optional
        GDAL AAIGrid creation option: number of significant digits (alternative to decimals).
    force_cellsize : bool, optional
        GDAL AAIGrid creation option: ``True`` → ``FORCE_CELLSIZE=YES`` to use the X pixel size
        as ``CELLSIZE`` when pixels are not perfectly square.
    auto_cast_float32 : bool, optional
        If ``True``, automatically cast to float32 when nodata cannot be represented
        in the requested integer dtype.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError
        If CRS/transform are missing or nodata is incompatible and
        ``auto_cast_float32=False``.

    Notes
    -----
    This uses the GDAL AAIGrid driver via rioxarray's ``to_raster(..., driver="AAIGrid")``.
    Options like ``DECIMAL_PRECISION`` and ``SIGNIFICANT_DIGITS`` are passed through
    ``profile_kwargs`` to GDAL. ASCII output is single-band only (first band selected).

    Examples
    --------
    >>> # Categorical (uint8) with in-range nodata, no decimals:
    >>> da8 = read_raster_xr("landuse.tif")
    >>> T = da8.rio.transform()
    >>> _ = write_arc_ascii(
    ...     "out/landuse.asc", da8, dtype="uint8", nodata=255,
    ...     transform=T, crs=da8.rio.crs, decimal_precision=0, force_cellsize=True
    ... )
    >>> # Continuous (float32) with -9999 nodata, 3 decimals:
    >>> daf = read_raster_xr("awc.tif")
    >>> _ = write_arc_ascii(
    ...     "out/awc.asc", daf, dtype="float32", nodata=-9999.0,
    ...     transform=daf.rio.transform(), crs=daf.rio.crs, decimal_precision=3
    ... )
    """
    out_path = Path(out_path)

    # Normalize to single band
    if "band" in da.dims:
        da = da.isel(band=0).squeeze(drop=True)

    # Attach geospatial metadata
    if crs is None or transform is None:
        raise ValueError("CRS and transform must be provided for ASCII writing.")
    da = da.rio.write_crs(crs).rio.write_transform(transform)

    # Enforce dtype/nodata (with optional auto-cast)
    da_t = ensure_dtype_and_nodata(
        da, dtype=dtype, nodata=nodata, auto_cast_float32=auto_cast_float32
    )

    # Build GDAL AAIGrid creation options
    options = []
    if decimal_precision is not None and significant_digits is not None:
        # Avoid conflicting options; prefer decimal_precision when both given
        options.append(f"DECIMAL_PRECISION={int(decimal_precision)}")
    elif decimal_precision is not None:
        options.append(f"DECIMAL_PRECISION={int(decimal_precision)}")
    elif significant_digits is not None:
        options.append(f"SIGNIFICANT_DIGITS={int(significant_digits)}")

    if force_cellsize is not None:
        options.append(f"FORCE_CELLSIZE={'YES' if force_cellsize else 'NO'}")

    profile_kwargs = {"options": options} if options else {}

    os.makedirs(out_path.parent, exist_ok=True)

    # Write ASCII grid
    da_t.rio.to_raster(out_path, driver="AAIGrid", profile_kwargs=profile_kwargs)
    return out_path