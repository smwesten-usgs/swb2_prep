# swb2_prep/common/cli_args.py
# -*- coding: utf-8 -*-
"""Composable argparse helpers for SWB2-prep CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    """Add common input/output arguments shared by multiple CLIs.

    Args:
        parser: An argparse.ArgumentParser to which flags are added.

    Adds:
        --input (Path): Path to the source raster (or vector, depending on CLI).
        --output-dir (Path): Directory to write outputs (e.g., GeoTIFF/ArcASCII).
    """
    parser.add_argument(
        "--input", 
        type=Path, 
        required=False, 
        help="Path to input dataset."
    )
    parser.add_argument(
        "--output-dir", 
        dest="output_dir",
        default=Path.cwd(),
        type=Path, 
        required=False, 
        help="Directory to write outputs."
    )


def add_common_aoi_args(parser: argparse.ArgumentParser) -> None:
    """Add AOI-related flags shared by multiple CLIs.

    Args:
        parser: An argparse.ArgumentParser to which flags are added.

    Adds:
        --polygon (Path): AOI polygon dataset (e.g., shapefile or GeoPackage).
        --bbox (float x4): xmin ymin xmax ymax in project CRS (alternative AOI path).
        --polygon-name (str): Attribute name for selecting a single AOI feature (requires value).
        --polygon-value (str): Attribute value for selecting a single AOI feature (requires name).
        --config (Path, optional): Path to 'project_options.toml' (if enabled). When omitted,
            CLIs should fall back to reading 'project_options.toml' from the subprocess CWD.
    """
    parser.add_argument(
        "--polygon",
        type=Path, 
        help="AOI polygon dataset (e.g., aoi.shp)."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="AOI bounding box coordinates in project CRS.",
    )
    parser.add_argument(
        "--polygon-name",
        dest="polygon_name",
        type=str,
        help="AOI attribute name used to select a single feature; requires --polygon-value.",
    )
    parser.add_argument(
        "--polygon-value",
        dest="polygon_value",
        type=str,
        help="AOI attribute value used to select a single feature; requires --polygon-name.",
    )
    parser.add_argument(
        "--project-options",
        dest="project_options",
        type=Path,
        default=Path.cwd() / "project_options.toml",
        help="Path to the 'project_options.toml' file. Must NOT exist when used with define_swb_grid.py.",
    )

    
def add_crs_and_resolution_args(parser: argparse.ArgumentParser) -> None:
    """Add CRS and resolution overrides for grid definition.

    Args:
        parser: An argparse.ArgumentParser to which flags are added.

    Adds:
        --resolution (float): Override grid cell size (project CRS units).
        --epsg (str): Override CRS via EPSG code (e.g., 'EPSG:5070').
        --proj4 (str): Override CRS via Proj4 string (mutually exclusive with --epsg).
    """
    parser.add_argument(
        "--resolution",
        type=float,
        help="Grid resolution (cell size) in project CRS units (e.g., 30.0).",
    )
    parser.add_argument(
        "--epsg",
        type=str,
        help="CRS specified as EPSG code (e.g., 'EPSG:5070'). Mutually exclusive with --proj4.",
    )
    parser.add_argument(
        "--proj4",
        type=str,
        help="CRS specified as Proj4 string. Mutually exclusive with --epsg.",
    )


def add_grid_control_args(parser: argparse.ArgumentParser) -> None:
    """Add grid behavior flags (snapping and update semantics).

    Args:
        parser: An argparse.ArgumentParser to which flags are added.

    Adds:
        --snap (str): Snapping mode ('outward' or 'inward').
        --dry-run (flag): Compute and print the grid without writing changes.
    """
    parser.add_argument(
        "--snap",
        choices=["outward", "inward"],
        default="outward",
        help="Snapping mode to apply to raw extents.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Compute and display grid details without writing to project_options.toml.",
    )
    
