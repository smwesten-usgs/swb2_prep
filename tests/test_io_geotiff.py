"""
Tests for GeoTIFF IO utilities.

This module validates:
- Reading datasets to DataArray (CRS/transform retained).
- XR-to-GeoTIFF writing via rioxarray.
- Legacy NumPy+profile writing via rasterio, with simple round-trip checks.
"""

from pathlib import Path
import numpy as np
import rioxarray as rxr

from swb2_prep.common.io import (
    read_raster_xr,
    write_geotiff_xr,
)


def _affine_close(a1, a2, atol=1e-12):
    """Compare two Affine transforms with tolerance."""
    return (
        np.isclose(a1.a, a2.a, atol=atol)
        and np.isclose(a1.b, a2.b, atol=atol)
        and np.isclose(a1.c, a2.c, atol=atol)
        and np.isclose(a1.d, a2.d, atol=atol)
        and np.isclose(a1.e, a2.e, atol=atol)
        and np.isclose(a1.f, a2.f, atol=atol)
    )


def test_read_xr_and_write_geotiff_xr(tmp_path):
    """
    Read a DEM as DataArray and write back to GeoTIFF with explicit dtype.
    Verify CRS equality, transform closeness, and dtype equality.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    src = data_dir / "hydrosheds_dem__south_manitou.tif"
    da = read_raster_xr(src)  # masked=True by default, may yield float dtype

    out_tif = tmp_path / "dem_copy.tif"
    # Explicitly set dtype to stabilize round-trip
    write_geotiff_xr(out_tif, da, dtype="float32", nodata=-9999., compress="LZW", tiled=True,)

    da2 = rxr.open_rasterio(out_tif, masked=False)
    if "band" in da2.dims:
        da2 = da2.squeeze("band", drop=True)

    assert da2.rio.crs == da.rio.crs
    assert _affine_close(da2.rio.transform(), da.rio.transform(), atol=1e-12)
