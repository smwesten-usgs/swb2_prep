# -*- coding: utf-8 -*-
"""
Compute D8 flow direction from a DEM aligned to the canonical SWB grid template.

Workflow:
1. Reproject/resample the input DEM to match the grid template (CRS, extent,
   resolution) using bilinear interpolation (appropriate for continuous data).
2. Run pysheds resolve_flats + flowdir on the aligned DEM.
3. Write the D8 result directly — no post-resampling of categorical codes.

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
import tempfile
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
    parser.add_argument("--dtype", type=str, default="int16",
                        help="Output dtype (default: int16).")
    parser.add_argument("--nodata", type=int, default=-9999,
                        help="NoData value for D8 output (default: -9999).")
    parser.add_argument("--compress", type=str, default="lzw",
                        help="GeoTIFF compression (default: lzw).")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional filename prefix (e.g., project label).")
    parser.add_argument("--resampling", type=str, choices=list(RESAMPLING_METHODS.keys()),
                        default="bilinear",
                        help="Resampling method for DEM alignment (default: bilinear).")
    parser.add_argument("--no-resolve-flats", dest="resolve_flats", action="store_false",
                        default=True,
                        help="Skip pysheds resolve_flats before computing flow direction. "
                             "Not recommended — resampling typically introduces flats that "
                             "pysheds cannot route without this step.")
    parser.add_argument("--no-log", dest="no_log", action="store_true", default=False,
                        help="Suppress logging to file.")
    return parser.parse_args()


def main() -> None:
    """Resample DEM to grid template, then compute D8 flow direction."""
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

    # Read input DEM and reproject/resample to match template grid
    log.info(f"Loading DEM from: {args.input}")
    dem_da = rxr.open_rasterio(args.input, masked=True).squeeze("band", drop=True)

    resampling = RESAMPLING_METHODS[args.resampling]
    log.info(f"Reprojecting DEM to template grid ({args.resampling} resampling)...")
    dem_aligned = dem_da.rio.reproject_match(template, resampling=resampling)

    # Write aligned DEM to a temporary GeoTIFF for pysheds
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_dem_path = Path(tmp.name)

    dem_aligned.rio.to_raster(tmp_dem_path, driver="GTiff")
    log.info(f"Wrote temporary aligned DEM: {tmp_dem_path}")

    # Run pysheds on the aligned DEM
    log.info("Loading aligned DEM into pysheds...")
    pysheds_grid = Grid.from_raster(str(tmp_dem_path))
    dem_pysheds = pysheds_grid.read_raster(str(tmp_dem_path))

    if args.resolve_flats:
        log.info("Resolving flats in resampled DEM...")
        dem_pysheds = pysheds_grid.resolve_flats(dem_pysheds)
    else:
        log.info("Skipping resolve_flats (--no-resolve-flats specified).")

    log.info(f"Computing D8 flow direction with dirmap={DIRMAP}...")
    fdir = pysheds_grid.flowdir(dem_pysheds, dirmap=DIRMAP)

    # Clean up temporary file
    tmp_dem_path.unlink(missing_ok=True)

    # Wrap flow direction result as xarray DataArray with template spatial metadata
    fdir_array = np.asarray(fdir, dtype=np.int16)

    # Remap pysheds internal values (not valid D8 codes) to nodata
    valid_codes = set(DIRMAP)
    invalid_mask = ~np.isin(fdir_array, list(valid_codes))
    fdir_array[invalid_mask] = args.nodata
    fdir_da = xr.DataArray(
        fdir_array,
        dims=dem_aligned.dims,
        coords=dem_aligned.coords,
    )
    fdir_da = fdir_da.rio.write_crs(dem_aligned.rio.crs)
    fdir_da = fdir_da.rio.write_transform(dem_aligned.rio.transform())
    fdir_da = fdir_da.rio.write_nodata(args.nodata)

    # Enforce dtype and nodata
    fdir_typed = ensure_dtype_and_nodata(fdir_da, dtype=args.dtype, nodata=args.nodata)

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

    # Cleanup
    template.close(); dem_aligned.close(); fdir_da.close(); fdir_typed.close()
    del template; del dem_aligned; del fdir_da; del fdir_typed
    gc.collect()


if __name__ == "__main__":
    main()
