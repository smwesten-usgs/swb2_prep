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
    parser.add_argument("--input", type=Path, required=True, help="Path to input dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write outputs.")


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
    # Notes:
    #    The landuse CLI tests depend on default CWD behavior for project_options.toml.
    #    When adopting '--config', introduce it in a separate small step and add one small
    #    test that uses the flag, leaving the original test unchanged.
    parser.add_argument("--polygon", type=Path, help="AOI polygon dataset (e.g., aoi.shp).")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="AOI bounding box coordinates in project CRS.",
    )
    parser.add_argument(
        "--polygon-name",
        type=str,
        help="AOI attribute name used to select a single feature; requires --polygon-value.",
    )
    parser.add_argument(
        "--polygon-value",
        type=str,
        help="AOI attribute value used to select a single feature; requires --polygon-name.",
    )
    parser.add_argument(
        "--config",
        default=Path.cwd()/'project_options.toml',
        type=Path,
        help="Optional path to 'project_options.toml'. If omitted, CLIs should read it from CWD.",
    )