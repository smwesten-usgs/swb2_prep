# -*- coding: utf-8 -*-
"""
Prepare the landuse raster aligned to the canonical SWB grid template.

Assumes `define_swb_grid.py` has already been run and that a grid template
GeoTIFF (`swb_grid_template.tif`) exists. The template carries the authoritative
CRS, transform, and shape; `reproject_match` handles all alignment in one call.

Usage:
    python -m swb2_prep.cli.prep_landuse_input ^
        --project-options project_options.toml ^
        --input ..\\..\\data\\cdl__south_manitou.tif ^
        --output-dir output ^
        --prefix south_manitou
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rioxarray as rxr
from rasterio.warp import Resampling
import gc

from swb2_prep.common.config import load_project_options
from swb2_prep.common.io import ensure_dtype_and_nodata, write_geotiff_xr, write_arc_ascii
from swb2_prep.common.log import setup_logging
from swb2_prep.common.paths import build_output_filename

RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for landuse raster preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare landuse raster aligned to the SWB grid template."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input landuse raster.")
    parser.add_argument("--project-options", dest="project_options", type=Path,
                        default=Path.cwd() / "project_options.toml",
                        help="Path to project_options.toml (default: ./project_options.toml).")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=Path.cwd(),
                        help="Output directory (default: current directory).")
    parser.add_argument("--dtype", type=str, default="int16",
                        help="Output dtype (default: int16).")
    parser.add_argument("--nodata", type=float, default=-1.0,
                        help="NoData value (default: -1).")
    parser.add_argument("--compress", type=str, default="lzw",
                        help="GeoTIFF compression (default: lzw).")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional filename prefix (e.g., project label).")
    parser.add_argument("--resampling", type=str, choices=list(RESAMPLING_METHODS.keys()),
                        default="nearest",
                        help="Resampling method (default: nearest).")
    parser.add_argument("--exclude-codes", dest="exclude_codes", type=int, nargs="+",
                        default=None,
                        help="Landuse codes to replace with nodata (e.g., 111 for open water).")
    parser.add_argument("--no-log", dest="no_log", action="store_true", default=False,
                        help="Suppress logging to file.")
    return parser.parse_args()


def main() -> None:
    """Align landuse raster to the grid template and write GeoTIFF + Arc ASCII."""
    args = parse_args()

    # Load project options and resolve template path
    opts = load_project_options(args.project_options)
    log = setup_logging("prep_landuse_input", args.project_options.parent, no_log=args.no_log)
    log.info(f"CLI arguments: {vars(args)}")
    grid = opts["grid"]
    template_path = Path(grid["template_tif"])
    if not template_path.is_absolute():
        template_path = args.project_options.parent / template_path

    # Open template (carries authoritative CRS, transform, shape)
    template = rxr.open_rasterio(template_path, masked=False).squeeze("band", drop=True)

    # Open input landuse raster
    lu_da = rxr.open_rasterio(args.input, masked=True).squeeze("band", drop=True)

    # Align to template grid: CRS + extent + resolution in one call
    resampling = RESAMPLING_METHODS[args.resampling]
    lu_aligned = lu_da.rio.reproject_match(template, resampling=resampling)

    # Replace excluded landuse codes with nodata (triggers SWB2 cell inactivation)
    if args.exclude_codes:
        arr = lu_aligned.values.copy()
        for code in args.exclude_codes:
            arr[arr == code] = args.nodata
        lu_aligned = lu_aligned.copy(data=arr)

    # Enforce dtype and nodata
    lu_typed = ensure_dtype_and_nodata(lu_aligned, dtype=args.dtype, nodata=args.nodata)

    # Build output filenames
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolution = float(abs(template.rio.transform().a))
    units = opts.get("project", {}).get("units", "m")

    tif_name = build_output_filename("landuse", resolution, units, ".tif", args.prefix or None)
    asc_name = build_output_filename("landuse", resolution, units, ".asc", args.prefix or None)

    # Write GeoTIFF
    tif_path = args.output_dir / tif_name
    write_geotiff_xr(tif_path, lu_typed, dtype=args.dtype, nodata=args.nodata,
                     compress=args.compress, tiled=True)
    log.info(f"Wrote: {tif_path}")

    # Write Arc ASCII
    asc_path = args.output_dir / asc_name
    write_arc_ascii(asc_path, lu_typed, dtype=args.dtype, nodata=args.nodata,
                    transform=template.rio.transform(), crs=str(template.rio.crs))
    log.info(f"Wrote: {asc_path}")
    template.close(); lu_aligned.close(); lu_typed.close()
    del template; del lu_aligned; del lu_typed
    gc.collect()


if __name__ == "__main__":
    main()
