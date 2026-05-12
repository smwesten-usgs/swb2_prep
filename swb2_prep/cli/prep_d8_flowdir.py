# -*- coding: utf-8 -*-
"""
Compute D8 flow direction from a DEM and align to the canonical SWB grid template.

Uses pysheds to resolve flats and compute flow direction with the dirmap
encoding expected by SWB2: (64, 128, 1, 2, 4, 8, 16, 32).

Usage:
    python -m swb2_prep.cli.prep_d8_flowdir ^
        --project-options project_options.toml ^
        --input ..\\..\\data\\hydrosheds_dem__south_manitou.tif ^
        --output-dir output ^
        --prefix south_manitou
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from pysheds.grid import Grid
from rasterio.warp import Resampling

from swb2_prep.common.config import load_project_options
from swb2_prep.common.io import ensure_dtype_and_nodata, write_geotiff_xr, write_arc_ascii
from swb2_prep.common.log import setup_logging
from swb2_prep.common.paths import build_output_filename

# SWB2-compatible D8 dirmap: N, NE, E, SE, S, SW, W, NW
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for D8 flow direction computation."""
    parser = argparse.ArgumentParser(
        description="Compute D8 flow direction aligned to the SWB grid template."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to input DEM raster (GeoTIFF).")
    parser.add_argument("--project-options", dest="project_options", type=Path,
                        default=Path.cwd() / "project_options.toml",
                        help="Path to project_options.toml (default: ./project_options.toml).")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=Path.cwd(),
                        help="Output directory (default: current directory).")
    parser.add_argument("--dtype", type=str, default="uint8",
                        help="Output dtype (default: uint8).")
    parser.add_argument("--nodata", type=int, default=0,
                        help="NoData value (default: 0).")
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
    """Compute D8 flow direction from DEM and align to the grid template."""
    args = parse_args()

    # Load project options and resolve template path
    opts = load_project_options(args.project_options)
    log = setup_logging("prep_d8_flowdir", args.project_options.parent, no_log=args.no_log)
    log.info(f"CLI arguments: {vars(args)}")
    grid_opts = opts["grid"]
    template_path = Path(grid_opts["template_tif"])
    if not template_path.is_absolute():
        template_path = args.project_options.parent / template_path

    # Open template (authoritative CRS, transform, shape)
    template = rxr.open_rasterio(template_path, masked=False).squeeze("band", drop=True)

    # Use pysheds to compute flow direction
    log.info(f"Loading DEM from: {args.input}")
    pysheds_grid = Grid.from_raster(str(args.input))
    dem = pysheds_grid.read_raster(str(args.input))

    # Resolve flats
    log.info("Resolving flats in DEM...")
    inflated_dem = pysheds_grid.resolve_flats(dem)

    # Compute D8 flow direction
    log.info(f"Computing D8 flow direction with dirmap={DIRMAP}...")
    fdir = pysheds_grid.flowdir(inflated_dem, dirmap=DIRMAP)

    # Read the input DEM with rioxarray to get spatial metadata
    dem_da = rxr.open_rasterio(args.input, masked=False).squeeze("band", drop=True)

    # Wrap flow direction result as xarray DataArray
    fdir_array = np.asarray(fdir, dtype=np.uint8)
    fdir_da = xr.DataArray(
        fdir_array,
        dims=dem_da.dims,
        coords=dem_da.coords,
    )
    fdir_da = fdir_da.rio.write_crs(dem_da.rio.crs)
    fdir_da = fdir_da.rio.write_transform(dem_da.rio.transform())
    fdir_da = fdir_da.rio.write_nodata(args.nodata)

    # Align to template grid
    resampling = RESAMPLING_METHODS[args.resampling]
    fdir_aligned = fdir_da.rio.reproject_match(template, resampling=resampling, nodata=args.nodata)

    # Enforce dtype and nodata
    fdir_typed = ensure_dtype_and_nodata(fdir_aligned, dtype=args.dtype, nodata=args.nodata)

    # Build output filenames
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolution = float(abs(template.rio.transform().a))
    units = opts.get("project", {}).get("units", "m")

    tif_name = build_output_filename("d8_flowdir", resolution, units, ".tif", args.prefix or None)
    asc_name = build_output_filename("d8_flowdir", resolution, units, ".asc", args.prefix or None)

    # Write GeoTIFF
    tif_path = args.output_dir / tif_name
    write_geotiff_xr(tif_path, fdir_typed, dtype=args.dtype, nodata=args.nodata,
                     compress=args.compress, tiled=True)
    log.info(f"Wrote: {tif_path}")

    # Write Arc ASCII
    asc_path = args.output_dir / asc_name
    write_arc_ascii(asc_path, fdir_typed, dtype=args.dtype, nodata=args.nodata,
                    transform=template.rio.transform(), crs=str(template.rio.crs))
    log.info(f"Wrote: {asc_path}")

    # Cleanup to avoid shutdown errors
    template.close(); fdir_da.close(); fdir_aligned.close(); fdir_typed.close()
    del template; del fdir_da; del fdir_aligned; del fdir_typed
    gc.collect()


if __name__ == "__main__":
    main()
