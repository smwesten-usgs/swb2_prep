# -*- coding: utf-8 -*-
"""
Prepare the available water capacity (AWC) raster aligned to the canonical SWB grid template.

Reads a MUKEY raster and a gNATSGO GeoPackage, reclassifies MUKEY values to
AWC (inches per foot), reprojects/aligns to the grid template, and writes
GeoTIFF and Arc ASCII outputs.

Usage:
    python -m swb2_prep.cli.prep_awc_input ^
        --project-options project_options.toml ^
        --input ..\\..\\data\\mukey__south_manitou.tif ^
        --gpkg ..\\..\\data\\gnatsgo__south_manitou.gpkg ^
        --output-dir output ^
        --prefix south_manitou
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import rasterio.fill
import rioxarray as rxr
import xarray as xr
from rasterio.warp import Resampling

from swb2_prep.common.config import load_project_options
from swb2_prep.common.io import ensure_dtype_and_nodata, write_geotiff_xr, write_arc_ascii
from swb2_prep.common.log import setup_logging
from swb2_prep.common.paths import build_output_filename
from swb2_prep.common.soil import read_awc_lookup, reclassify_mukey_to_awc

RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AWC raster preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare AWC raster (inches/foot) aligned to the SWB grid template."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to input MUKEY raster (GeoTIFF).")
    parser.add_argument("--gpkg", type=Path, required=True,
                        help="Path to gNATSGO GeoPackage containing muaggatt table.")
    parser.add_argument("--project-options", dest="project_options", type=Path,
                        default=Path.cwd() / "project_options.toml",
                        help="Path to project_options.toml (default: ./project_options.toml).")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=Path.cwd(),
                        help="Output directory (default: current directory).")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Output dtype (default: float32).")
    parser.add_argument("--nodata", type=float, default=-1.0,
                        help="NoData value (default: -1.0).")
    parser.add_argument("--compress", type=str, default="lzw",
                        help="GeoTIFF compression (default: lzw).")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional filename prefix (e.g., project label).")
    parser.add_argument("--resampling", type=str, choices=list(RESAMPLING_METHODS.keys()),
                        default="bilinear",
                        help="Resampling method (default: bilinear).")
    parser.add_argument("--fill-nodata", dest="fill_nodata", action="store_true",
                        default=False,
                        help="Fill nodata holes in AWC grid before reprojection.")
    parser.add_argument("--fill-max-search-dist", dest="fill_max_search_dist", type=int,
                        default=250,
                        help="Max search distance in pixels for fill (default: 250).")
    parser.add_argument("--awc-floor", dest="awc_floor", type=float, default=None,
                        help="Minimum AWC value; valid pixels below this are clamped upward.")
    parser.add_argument("--awc-ceiling", dest="awc_ceiling", type=float, default=None,
                        help="Maximum AWC value; valid pixels above this are clamped downward.")
    parser.add_argument("--num-digits-precision", dest="num_digits_precision", type=int,
                        default=3,
                        help="Decimal precision for Arc ASCII output (default: 3).")
    parser.add_argument("--no-log", dest="no_log", action="store_true", default=False,
                        help="Suppress logging to file.")
    return parser.parse_args()


def main() -> None:
    """Reclassify MUKEY raster to AWC values and align to the grid template."""
    args = parse_args()

    # Load project options and resolve template path
    opts = load_project_options(args.project_options)
    log = setup_logging("prep_awc_input", args.project_options.parent, no_log=args.no_log)
    log.info(f"CLI arguments: {vars(args)}")
    grid = opts["grid"]
    template_path = Path(grid["template_tif"])
    if not template_path.is_absolute():
        template_path = args.project_options.parent / template_path

    # Open template (authoritative CRS, transform, shape)
    template = rxr.open_rasterio(template_path, masked=False).squeeze("band", drop=True)

    # Read MUKEY raster (unmasked — we need raw integer MUKEY values)
    mukey_da = rxr.open_rasterio(args.input, masked=False).squeeze("band", drop=True)

    # Build MUKEY -> AWC lookup from GeoPackage
    lookup = read_awc_lookup(args.gpkg)

    # Reclassify MUKEY values to AWC (inches/foot)
    mukey_array = mukey_da.values
    awc_array = reclassify_mukey_to_awc(mukey_array, lookup, nodata_value=args.nodata)

    # Optionally fill nodata holes at source resolution
    if args.fill_nodata:
        mask = (awc_array != args.nodata).astype(np.uint8)
        awc_array = rasterio.fill.fillnodata(
            image=awc_array,
            mask=mask,
            max_search_distance=args.fill_max_search_dist,
        ).astype(np.float32)

    # Wrap result as xarray DataArray preserving spatial metadata
    awc_da = xr.DataArray(
        awc_array,
        dims=mukey_da.dims,
        coords=mukey_da.coords,
    )
    awc_da = awc_da.rio.write_crs(mukey_da.rio.crs)
    awc_da = awc_da.rio.write_transform(mukey_da.rio.transform())
    awc_da = awc_da.rio.write_nodata(args.nodata)

    # Align to template grid
    resampling = RESAMPLING_METHODS[args.resampling]
    awc_aligned = awc_da.rio.reproject_match(template, resampling=resampling, nodata=args.nodata)

    # Clamp AWC values to floor/ceiling AFTER reprojection so that any nodata
    # introduced by reproject_match (edge pixels) is also replaced
    if args.awc_floor is not None or args.awc_ceiling is not None:
        arr = awc_aligned.values.copy()
        if args.awc_floor is not None:
            arr = np.where(arr < args.awc_floor, args.awc_floor, arr)
        if args.awc_ceiling is not None:
            arr = np.where(arr > args.awc_ceiling, args.awc_ceiling, arr)
        awc_aligned = awc_aligned.copy(data=arr)

    # Enforce dtype and nodata
    awc_typed = ensure_dtype_and_nodata(awc_aligned, dtype=args.dtype, nodata=args.nodata)

    # Build output filenames
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolution = float(abs(template.rio.transform().a))
    units = opts.get("project", {}).get("units", "m")

    tif_name = build_output_filename("awc", resolution, units, ".tif", args.prefix or None)
    asc_name = build_output_filename("awc", resolution, units, ".asc", args.prefix or None)

    # Write GeoTIFF
    tif_path = args.output_dir / tif_name
    write_geotiff_xr(tif_path, awc_typed, dtype=args.dtype, nodata=args.nodata,
                     compress=args.compress, tiled=True)
    log.info(f"Wrote: {tif_path}")

    # Write Arc ASCII
    asc_path = args.output_dir / asc_name
    write_arc_ascii(asc_path, awc_typed, dtype=args.dtype, nodata=args.nodata,
                    transform=template.rio.transform(), crs=str(template.rio.crs),
                    decimal_precision=args.num_digits_precision)
    log.info(f"Wrote: {asc_path}")

    # Cleanup to avoid shutdown errors
    template.close(); awc_da.close(); awc_aligned.close(); awc_typed.close()
    del template; del awc_da; del awc_aligned; del awc_typed
    gc.collect()


if __name__ == "__main__":
    main()
