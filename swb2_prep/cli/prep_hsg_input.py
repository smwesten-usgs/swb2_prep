# -*- coding: utf-8 -*-
"""
Prepare the hydrologic soil group (HSG) raster aligned to the canonical SWB grid template.

Reads a MUKEY raster and a gNATSGO GeoPackage, reclassifies MUKEY values to
numeric HSG codes, reprojects/aligns to the grid template, and writes GeoTIFF
and Arc ASCII outputs.

Usage:
    python -m swb2_prep.cli.prep_hsg_input ^
        --project-options project_options.toml ^
        --input ..\\..\\data\\mukey__south_manitou.tif ^
        --gpkg ..\\..\\data\\gnatsgo__south_manitou.gpkg ^
        --output-dir output ^
        --prefix south_manitou
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.warp import Resampling
import gc

from swb2_prep.common.config import load_project_options
from swb2_prep.common.io import ensure_dtype_and_nodata, write_geotiff_xr, write_arc_ascii
from swb2_prep.common.log import setup_logging
from swb2_prep.common.paths import build_output_filename
from swb2_prep.common.soil import read_hsg_lookup, reclassify_mukey_to_hsg

RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for HSG raster preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare HSG raster aligned to the SWB grid template."
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
    parser.add_argument("--dtype", type=str, default="int16",
                        help="Output dtype (default: int16).")
    parser.add_argument("--nodata", type=int, default=-1,
                        help="NoData value (default: -1).")
    parser.add_argument("--compress", type=str, default="lzw",
                        help="GeoTIFF compression (default: lzw).")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional filename prefix (e.g., project label).")
    parser.add_argument("--resampling", type=str, choices=list(RESAMPLING_METHODS.keys()),
                        default="nearest",
                        help="Resampling method (default: nearest).")
    parser.add_argument("--no-log", dest="no_log", action="store_true", default=False,
                        help="Suppress logging to file.")
    return parser.parse_args()


def main() -> None:
    """Reclassify MUKEY raster to HSG codes and align to the grid template."""
    args = parse_args()

    # Load project options and resolve template path
    opts = load_project_options(args.project_options)
    log = setup_logging("prep_hsg_input", args.project_options.parent, no_log=args.no_log)
    log.info(f"CLI arguments: {vars(args)}")
    grid = opts["grid"]
    template_path = Path(grid["template_tif"])
    if not template_path.is_absolute():
        template_path = args.project_options.parent / template_path

    # Open template (authoritative CRS, transform, shape)
    template = rxr.open_rasterio(template_path, masked=False).squeeze("band", drop=True)

    # Read MUKEY raster (unmasked — we need raw integer MUKEY values)
    mukey_da = rxr.open_rasterio(args.input, masked=False).squeeze("band", drop=True)

    # Build MUKEY -> HSG lookup from GeoPackage
    lookup = read_hsg_lookup(args.gpkg)

    # Reclassify MUKEY values to HSG codes
    mukey_array = mukey_da.values
    hsg_array = reclassify_mukey_to_hsg(mukey_array, lookup, nodata_value=args.nodata)

    # Wrap result as xarray DataArray preserving spatial metadata
    hsg_da = xr.DataArray(
        hsg_array,
        dims=mukey_da.dims,
        coords=mukey_da.coords,
    )
    hsg_da = hsg_da.rio.write_crs(mukey_da.rio.crs)
    hsg_da = hsg_da.rio.write_transform(mukey_da.rio.transform())
    hsg_da = hsg_da.rio.write_nodata(args.nodata)

    # Align to template grid
    resampling = RESAMPLING_METHODS[args.resampling]
    hsg_aligned = hsg_da.rio.reproject_match(template, resampling=resampling, nodata=args.nodata)

    # Enforce dtype and nodata
    hsg_typed = ensure_dtype_and_nodata(hsg_aligned, dtype=args.dtype, nodata=args.nodata)

    # Build output filenames
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolution = float(abs(template.rio.transform().a))
    units = opts.get("project", {}).get("units", "m")

    tif_name = build_output_filename("hsg", resolution, units, ".tif", args.prefix or None)
    asc_name = build_output_filename("hsg", resolution, units, ".asc", args.prefix or None)

    # Write GeoTIFF
    tif_path = args.output_dir / tif_name
    write_geotiff_xr(tif_path, hsg_typed, dtype=args.dtype, nodata=args.nodata,
                     compress=args.compress, tiled=True)
    log.info(f"Wrote: {tif_path}")

    # Write Arc ASCII
    asc_path = args.output_dir / asc_name
    write_arc_ascii(asc_path, hsg_typed, dtype=args.dtype, nodata=args.nodata,
                    transform=template.rio.transform(), crs=str(template.rio.crs))
    log.info(f"Wrote: {asc_path}")
    template.close(); hsg_da.close(); hsg_aligned.close(); hsg_typed.close()
    del template; del hsg_da; del hsg_aligned; del hsg_typed
    gc.collect()

if __name__ == "__main__":
    main()
