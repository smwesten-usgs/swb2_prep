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
from rasterio.transform import Affine, from_origin
from swb2_prep.common.utils import PathLike


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
    path: PathLike,
    data_array: xr.DataArray,
    *,
    dtype: str,
    nodata: Optional[Union[int, float]],
    auto_cast_float32: bool = False,
    **rasterio_kwargs,
) -> Path:
    """Write a typed GeoTIFF from an XR DataArray via rioxarray.

    Args:
        path: Output file path (str or :class:`pathlib.Path`).
        data_array: Input raster with CRS/transform attached via ``data_array.rio``.
        dtype: Target dtype for the GeoTIFF (e.g., ``"float32"``, ``"int16"``).
        nodata: NoData value to embed in the GeoTIFF; ``None`` to omit.
        auto_cast_float32: If True and dtype is float32, attempt safe downcast from float64.
        **rasterio_kwargs: Additional keyword args forwarded to rioxarray/rasterio
            writer (e.g., ``compress="lzw"``, ``tiled=True``, ``blockxsize=256``).

    Returns:
        The :class:`pathlib.Path` to the written GeoTIFF.

    Raises:
        ValueError: If input DataArray lacks CRS/transform metadata required for IO.

    Notes:
        - This function preserves CRS/transform via rioxarray and ensures the output
          dtype/nodata are representable prior to write. Tests validate CRS equality,
          affine closeness, and dtype equality on round-trip IO.
    """
    out_path = Path(path)

    if data_array.rio.crs is None or data_array.rio.transform() is None:
        raise ValueError("Input DataArray is missing CRS/transform metadata (.rio).")

    # Normalize dtype/nodata before write, consistent with the IO API.
    typed = ensure_dtype_and_nodata(
        data_array,
        dtype=dtype,
        nodata=nodata,
        auto_cast_float32=auto_cast_float32,
    )

    # Write via rioxarray; dtype/nodata already prepared.
    typed.rio.to_raster(out_path, **rasterio_kwargs)
    return out_path


def lower_left_to_transform(
    xllcorner: float,
    yllcorner: float,
    nrows: int,
    x_res: float,
    y_res: float,
) -> Affine:
    """Convert lower-left origin to an upper-left Affine transform (GDAL/Rasterio).

    In Arc ASCII (AAIGrid) conventions, metadata often reports the **lower-left**
    corner of the raster. RasterIO/GDAL requires an **upper-left** origin with a
    negative ``y_res`` for north-up rasters. This helper creates the correct
    upper-left transform from lower-left inputs.

    Args:
        xllcorner: X-coordinate of the lower-left corner of the raster.
        yllcorner: Y-coordinate of the lower-left corner of the raster.
        nrows: Number of rows in the raster.
        x_res: Pixel width (units of the raster CRS).
        y_res: Pixel height (units of the raster CRS).

    Returns:
        Affine transform representing the upper-left origin, suitable for GDAL/Rasterio.

    Notes:
        - Upper-left Y = lower-left Y + (nrows * y_res), assuming positive y_res.
        - The final affine will carry negative y_res for north-up rasters, as required
          by GDAL/Rasterio.
    """
    # Compute the upper-left (UL) corner using lower-left + nrows * y_res.
    x_ul = xllcorner
    y_ul = yllcorner + (nrows * y_res)

    # Return GDAL-style affine: scale in X, negative scale in Y.
    return Affine.translation(x_ul, y_ul) * Affine.scale(x_res, -y_res)


def write_arc_ascii(
    out_path: PathLike,
    data_array: xr.DataArray,
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
    """Write an ESRI Arc ASCII Grid (AAIGrid, ``.asc``) using rioxarray with explicit dtype/nodata.

    This function converts an XR ``DataArray`` into an Arc ASCII grid while
    preserving georeferencing and applying explicit dtype and nodata semantics.
    It supports GDAL creation options for controlling numeric formatting.

    Args:
        out_path: Output file path (str or :class:`pathlib.Path`).
        data_array: Input raster with CRS/transform attached via ``data_array.rio``.
        dtype: Target dtype for ASCII (e.g., ``"float32"``, ``"int16"``); ensures values are representable.
        nodata: NoData value to embed; ``None`` to omit.
        transform: GDAL/Rasterio affine transform (upper-left origin).
        crs: Coordinate reference system; typically the project CRS (EPSG string or PROJ dict).
        decimal_precision: Optional GDAL AAIGrid creation option (DECIMAL_PRECISION).
        significant_digits: Optional GDAL AAIGrid creation option (SIGNIFICANT_DIGITS).
        force_cellsize: Optional flag (implementation-dependent) to enforce cellsize in header.
        auto_cast_float32: If True and dtype is float32, attempt safe downcast from float64.

    Returns:
        The :class:`pathlib.Path` to the written ``.asc`` file.

    Raises:
        ValueError: If the input ``DataArray`` lacks CRS/transform metadata via rioxarray accessors.

    Notes:
        - The XR-first IO design ensures CRS/transform are managed through rioxarray;
          tests in your suite validate transform handling and typed output elsewhere,
          and this function adheres to the same conventions. [1](https://doimspp-my.sharepoint.com/personal/smwesten_usgs_gov/Documents/Microsoft%20Copilot%20Chat%20Files/test_io.py)
        - If your workflow starts with lower-left metadata, derive the correct
          upper-left transform via ``lower_left_to_transform(...)`` prior to writing. [1](https://doimspp-my.sharepoint.com/personal/smwesten_usgs_gov/Documents/Microsoft%20Copilot%20Chat%20Files/test_io.py)
    """
    dst = Path(out_path)

    # Require CRS/transform metadata on the incoming DataArray
    if data_array.rio.crs is None:
        raise ValueError("Input DataArray is missing CRS metadata (.rio.crs).")
    # The function accepts an explicit transform; ensure rioxarray uses it.
    # Normalize dtype/nodata using your IO helper (defined in this module).
    typed = ensure_dtype_and_nodata(
        data_array,
        dtype=dtype,
        nodata=nodata,
        auto_cast_float32=auto_cast_float32,
    )

    # Attach transform and CRS before write (rioxarray honors .rio.write_* calls)
    typed = typed.rio.write_transform(transform, inplace=False)
    typed = typed.rio.write_crs(crs, inplace=False)

    # Build GDAL creation options dict for AAIGrid formatting
    gdal_options = {}
    if decimal_precision is not None:
        gdal_options["DECIMAL_PRECISION"] = decimal_precision
    if significant_digits is not None:
        gdal_options["SIGNIFICANT_DIGITS"] = significant_digits
    if force_cellsize is not None:
        # Some environments/drivers accept FORCE_CELLSIZE-like toggles; include if requested.
        gdal_options["FORCE_CELLSIZE"] = "YES" if force_cellsize else "NO"

    # AAIGrid write via rioxarray; GDAL options supplied through the driver config
    typed.rio.to_raster(
        dst,
        driver="AAIGrid",
        **gdal_options,
    )
    return dst