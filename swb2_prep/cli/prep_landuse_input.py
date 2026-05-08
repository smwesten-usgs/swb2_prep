# swb2_prep/cli/prep_landuse_input.py
# -*- coding: utf-8 -*-

"""
Prepare the landuse raster using the canonical grid defined in 'project_options.toml'.

This CLI:
- Loads the single TOML ('--project-options'), reads the canonical [grid] section.
- Reprojects the landuse raster to the grid CRS.
- Resamples and aligns it to the exact grid extent (xmin/ymax origin, resolution, nx/ny).
- Writes a typed GeoTIFF (and optionally ESRI Arc ASCII) to the output directory.

Assumptions (explicit and simple):
- '[grid]' in the TOML is authoritative for CRS, resolution, and extents.
- If '[grid]' is missing, error out.
- Landuse raster is a single-band categorical/int raster; use dtype and nodata via CLI flags.

Examples
--------
Run from your project directory or example directory:

    python -m swb2_prep.cli.prep_landuse_input ^
        --project-options project_options.toml ^
        --input ..\\..\\data\\cdl__south_manitou.tif ^
        --output-dir output ^
        --dtype uint16 ^
        --nodata 0 ^
        --write-asc

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from rasterio.transform import Affine
from pyproj import CRS as _CRS

from swb2_prep.common.cli_args import add_common_io_args
from swb2_prep.common.cli_args import add_common_aoi_args  # for --project-options
from swb2_prep.common.config import load_project_options
from swb2_prep.common.io import (
    read_raster_xr,
    ensure_dtype_and_nodata,
    write_geotiff_xr,
    write_arc_ascii,
)
from swb2_prep.common.paths import build_output_filename
from swb2_prep.common.utils import PathLike


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preparing the landuse raster.

    Returns:
        argparse.Namespace: Parsed arguments containing paths, typing, and output flags.
    """
    parser = argparse.ArgumentParser(
        description="Prepare the landuse raster aligned to the canonical [grid] in project_options.toml."
    )
    # We need --input, --output-dir, and --project-options.
    add_common_io_args(parser)
    add_common_aoi_args(parser)  # includes --project-options

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        help="Output dtype for landuse GeoTIFF (e.g., 'uint16', 'int16', 'float32'). Default: 'uint16'.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=0.0,
        help="NoData value for output rasters (representable in selected dtype). Default: 0.",
    )
    parser.add_argument(
        "--compress",
        type=str,
        default="lzw",
        help="Compression for GeoTIFF (e.g., 'lzw', 'deflate'). Default: 'lzw'.",
    )
    parser.add_argument(
        "--write-asc",
        default=True,
        action="store_true",
        help="Also write ESRI Arc ASCII Grid (.asc) alongside the GeoTIFF.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix (e.g., project or AOI label) for outputs.",
    )
    return parser.parse_args()


def _build_grid_affine_and_shape(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    resolution: float,
) -> Tuple[Affine, Tuple[int, int]]:
    """Construct the upper-left Affine transform and shape from grid bounds and resolution.

    Args:
        xmin: Grid minimum x (snapped).
        ymin: Grid minimum y (snapped).
        xmax: Grid maximum x (snapped).
        ymax: Grid maximum y (snapped).
        resolution: Pixel size in CRS units.

    Returns:
        A tuple of (affine_transform, (ny, nx)) for the target grid.

    Notes:
        - Upper-left origin is (xmin, ymax), y pixel size is negative for north-up rasters.
    """
    width = xmax - xmin
    height = ymax - ymin
    nx = int(round(width / resolution))
    ny = int(round(height / resolution))
    transform = Affine.translation(xmin, ymax) * Affine.scale(resolution, -resolution)
    return transform, (ny, nx)


def _make_grid_template(
    crs: str,
    transform: Affine,
    shape: Tuple[int, int],
) -> xr.DataArray:
    """Create an empty XR DataArray template carrying CRS/transform for reproject_match.

    Args:
        crs: Canonical CRS string (e.g., 'EPSG:5070').
        transform: Upper-left affine transform of the target grid.
        shape: (ny, nx) grid shape.

    Returns:
        xarray.DataArray with 'y' and 'x' dims, CRS/transform written via .rio.
    """
    ny, nx = shape
    template = xr.DataArray(
        np.zeros((ny, nx), dtype="float32"),
        dims=("y", "x"),
        name="template",
    )
    template = template.rio.write_crs(_CRS.from_user_input(crs).to_string(), inplace=False)
    template = template.rio.write_transform(transform, inplace=False)
    return template


def main() -> None:
    """Entry point for preparing the landuse raster aligned to `[grid]` in the TOML.

    Steps:
        1) Load project_options.toml and confirm '[grid]' exists.
        2) Read the input landuse raster with CRS/transform via rioxarray.
        3) Reproject to grid CRS and resample to the grid extent/resolution.
        4) Write GeoTIFF (typed) and optionally ESRI Arc ASCII.

    Raises:
        FileNotFoundError: If '--project-options' file does not exist.
        KeyError: If '[grid]' section is missing from the TOML.
        ValueError: For invalid dtype/nodata semantics or missing CRS/transform in input raster.
    """
    args = parse_args()

    project_path: Path = args.project_options
    if not project_path.exists():
        raise FileNotFoundError(f"Project options TOML not found: {project_path}")

    # Load TOML and extract canonical grid parameters
    opts = load_project_options(project_path)
    if "grid" not in opts:
        raise KeyError("The project_options.toml is missing the [grid] section.")
    grid = opts["grid"]

    grid_crs: str = str(grid["crs"])
    resolution: float = float(grid["resolution"])
    xmin: float = float(grid["xmin"])
    ymin: float = float(grid["ymin"])
    xmax: float = float(grid["xmax"])
    ymax: float = float(grid["ymax"])

    # Build the target grid transform and shape
    transform, (ny, nx) = _build_grid_affine_and_shape(xmin, ymin, xmax, ymax, resolution)
    template = _make_grid_template(grid_crs, transform, (ny, nx))

    # Read input landuse raster
    landuse_da = read_raster_xr(args.input, masked=True)

    # Reproject and align to the canonical grid
    # Using rioxarray's reproject to target CRS first (if needed), then reproject_match to template.
    if str(landuse_da.rio.crs) != _CRS.from_user_input(grid_crs).to_string():
        landuse_da = landuse_da.rio.reproject(grid_crs)

    # Resample/align to the grid template (exact extent, resolution, and shape)
    landuse_da_aligned = landuse_da.rio.reproject_match(template)

    # Ensure dtype and nodata as requested
    dtype: str = args.dtype
    nodata_val: Optional[float] = args.nodata
    landuse_typed = ensure_dtype_and_nodata(landuse_da_aligned, dtype=dtype, nodata=nodata_val)

    # Build output filenames
    units = str(opts.get("project", {}).get("units", "m"))
    base = "landuse"
    prefix = args.prefix if args.prefix else ""
    geotiff_name = build_output_filename(base=base, resolution=resolution, units=units, ext=".tif", prefix=prefix)
    ascii_name = build_output_filename(base=base, resolution=resolution, units=units, ext=".asc", prefix=prefix)

    # Write GeoTIFF
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    geotiff_path = out_dir / geotiff_name
    write_geotiff_xr(
        geotiff_path,
        landuse_typed,
        dtype=dtype,
        nodata=nodata_val,
        compress=args.compress,
        tiled=True,
    )

    print(f"Wrote landuse GeoTIFF: {geotiff_path}")

    # Optionally write ESRI Arc ASCII
    if args.write_asc:
        write_arc_ascii(
            out_dir / ascii_name,
            landuse_typed,
            dtype=dtype,
            nodata=nodata_val,
            transform=transform,
            crs=grid_crs,
            decimal_precision=None,
            significant_digits=None,
            force_cellsize=None,
        )
        print(f"Wrote landuse Arc ASCII: {out_dir / ascii_name}")


if __name__ == "__main__":
    main()