# swb2_prep/cli/define_swb_grid.py
# -*- coding: utf-8 -*-

"""
Create a single 'project_options.toml' with canonical [grid] and minimal stub sections.

Behavior (simple and explicit):
- If the TOML at --project-options exists: error out (no merging or updating).
- If it does not exist: create it, populate [grid], and add stub [project], [paths], [aoi], [provenance].
- CRS must be specified by exactly one of --epsg or --proj4; if both are provided: error.
- Resolution is required as a CLI argument.
- Downstream CLIs should read CRS/resolution from [grid] exclusively.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import toml
from pyproj import CRS as _CRS

from swb2_prep.common.cli_args import (
    add_common_aoi_args,
    add_common_io_args,
    add_crs_and_resolution_args,
    add_grid_control_args,
)
from swb2_prep.common.griddef import griddef_to_polygon_gdf
from swb2_prep.common.grids import reproject_polygon  # assumed present in your repo
from swb2_prep.common.grids import snap_extent, compute_grid_dims
from swb2_prep.common.ops import create_polygon_from_bbox


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for grid definition.

    Returns:
        Parsed argparse namespace containing AOI, CRS/resolution, and grid control flags.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create 'project_options.toml' with canonical [grid] and stub sections. "
            "Errors if the file already exists."
        )
    )
    add_common_aoi_args(parser)            # includes --polygon, --bbox, --polygon-name, --polygon-value, --output-dir, --project-options
    add_common_io_args(parser)             # includes --input and --output-dir
    add_crs_and_resolution_args(parser)    # includes --resolution (required), --epsg, --proj4
    add_grid_control_args(parser)          # includes --snap and --write-grid-shapefile
    parser.add_argument(
        "--output-shapefile",
        dest="output_shapefile",
        type=Path,
        help="Name of the output shapefile to write the generated SWB project boundaries.",
    )
    return parser.parse_args()


def _resolve_crs(epsg: str | None, proj4: str | None) -> Tuple[str, str]:
    """Resolve CRS to a canonical string and proj4.

    Exactly one of ``epsg`` or ``proj4`` must be provided.

    Args:
        epsg: EPSG code string (e.g., ``'EPSG:5070'``), or ``None``.
        proj4: Proj4 string, or ``None``.

    Returns:
        Tuple of ``(canonical_crs, proj4_string)``.

    Raises:
        ValueError: If both ``epsg`` and ``proj4`` are provided, or neither is provided.
    """
    if epsg and proj4:
        raise ValueError("Provide either --epsg or --proj4, not both.")
    if not epsg and not proj4:
        raise ValueError("CRS must be provided via --epsg or --proj4.")

    crs_obj = _CRS.from_user_input(epsg if epsg else proj4)
    canonical_crs = crs_obj.to_string()
    proj4_str = crs_obj.to_proj4()
    return canonical_crs, proj4_str


def load_aoi_polygon(args: argparse.Namespace, project_crs: str) -> Tuple[gpd.GeoDataFrame, str, Dict[str, Any]]:
    """Load the AOI polygon from shapefile or bounding box, reproject to project CRS.

    Args:
        args: Parsed CLI arguments.
        project_crs: Project CRS (EPSG or PROJ string).

    Returns:
        Tuple of ``(polygon_gdf, source, aoi_stub)`` where:
            - ``polygon_gdf`` is the AOI polygon as a GeoDataFrame reprojected to the project CRS.
            - ``source`` is either ``'bbox'`` or ``'aoi_polygon'``.
            - ``aoi_stub`` is a dictionary reflecting AOI inputs for TOML stub writing.

    Raises:
        ValueError: If neither AOI mode is provided or selection is ambiguous.
    """
    if args.bbox:
        xmin, ymin, xmax, ymax = args.bbox
        polygon_gdf = create_polygon_from_bbox(xmin, ymin, xmax, ymax, project_crs)
        aoi_stub = {
            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "polygon": "",
            "polygon_name": "",
            "polygon_value": "",
        }
        return polygon_gdf, "bbox", aoi_stub

    if not args.polygon:
        raise ValueError("Either --polygon or --bbox must be provided.")

    polygon_gdf = gpd.read_file(args.polygon)

    has_name = args.polygon_name is not None
    has_value = args.polygon_value is not None
    if has_name ^ has_value:
        raise ValueError("Both --polygon-name and --polygon-value must be provided or neither.")

    if has_name and has_value:
        field = args.polygon_name
        value = args.polygon_value
        if field not in polygon_gdf.columns:
            raise ValueError(f"Field {field!r} not found in shapefile.")
        selection = polygon_gdf[polygon_gdf[field] == value]
        if len(selection) == 0:
            raise ValueError(f"No polygons where {field} == {value!r}.")
        if len(selection) > 1:
            raise ValueError(f"Multiple polygons match {field} == {value!r}.")
        polygon_gdf = selection
        source = "aoi_polygon"
    else:
        if len(polygon_gdf) == 0:
            raise ValueError("Shapefile contains no polygons.")
        if len(polygon_gdf) > 1:
            raise ValueError(
                "Shapefile contains multiple polygons; must provide --polygon-name and --polygon-value."
            )
        source = "aoi_polygon"

    polygon_gdf = reproject_polygon(polygon_gdf, project_crs)
    aoi_stub = {
        "bbox": [],
        "polygon": str(args.polygon),
        "polygon_name": args.polygon_name or "",
        "polygon_value": args.polygon_value or "",
    }
    return polygon_gdf, source, aoi_stub


def _build_project_options_stub(
    *,
    canonical_crs: str,
    resolution: float,
    aoi_stub: Dict[str, Any],
    grid: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Construct the single TOML dictionary with [grid] and stub sections.

    Args:
        canonical_crs: Canonical CRS string (e.g., ``'EPSG:5070'``).
        resolution: Grid resolution (float).
        aoi_stub: Simple dict reflecting AOI selection arguments.
        grid: Canonical grid dict to store under ``[grid]``.
        output_dir: Output directory for downstream CLIs and optional shapefiles.

    Returns:
        Nested dictionary suitable for ``toml.dump``.
    """
    return {
        "project": {
            # Informational mirrors; downstream code should use [grid]
            "crs": canonical_crs,
            "resolution": float(resolution),
            "units": "m",  # default stub; user may edit later
        },
        "paths": {
            "input_dir": "data",
            "output_dir": str(output_dir),
        },
        "aoi": aoi_stub,
        "grid": grid,
        "provenance": {
            "generated_by": "define_swb_grid.py",
            "generated_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": "0.1.0",
        },
    }


def main() -> None:
    """Entry point for defining and writing the SWB grid into a new TOML file.

    This function enforces create-only semantics:
    - If ``--project-options`` points to an existing file, a ``FileExistsError`` is raised.
    - Otherwise, it computes the grid from the AOI, applies snapping, and writes a new
      ``project_options.toml`` with stub sections and a canonical ``[grid]`` block.

    Raises:
        FileExistsError: If the ``--project-options`` file already exists.
        ValueError: For invalid CRS specification, resolution, or AOI selection.
    """
    args = parse_args()

    # 1) Enforce create-only semantics
    project_path: Path = args.project_options
    if project_path.exists():
        raise FileExistsError(
            f"{project_path} already exists. This tool only creates a new 'project_options.toml'. "
            "Please remove or rename the existing file."
        )

    # 2) Resolve CRS and resolution from CLI (simple and explicit)
    canonical_crs, proj4_str = _resolve_crs(args.epsg, args.proj4)
    resolution: float = float(args.resolution)
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")

    # 3) AOI polygon in project CRS
    polygon_gdf, source, aoi_stub = load_aoi_polygon(args, canonical_crs)

    # 4) Compute extents, snap, grid dimensions
    xmin_raw, ymin_raw, xmax_raw, ymax_raw = map(float, polygon_gdf.total_bounds)
    xmin, ymin, xmax, ymax = snap_extent(
        xmin_raw, ymin_raw, xmax_raw, ymax_raw, resolution, mode=args.snap
    )
    nx, ny = compute_grid_dims(xmin, ymin, xmax, ymax, resolution)

    # 5) Canonical grid dict
    grid: Dict[str, Any] = {
        "crs": canonical_crs,
        "proj4": proj4_str,
        "resolution": float(resolution),
        "snap": str(args.snap),
        "source": source,
        "xmin_raw": float(xmin_raw),
        "ymin_raw": float(ymin_raw),
        "xmax_raw": float(xmax_raw),
        "ymax_raw": float(ymax_raw),
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
        "nx": int(nx),
        "ny": int(ny),
    }

    # 6) Determine output dir (explicit default if not provided)
    output_dir = args.output_dir
    output_shapefile = args.output_shapefile

    # 7) Assemble TOML and write (create-only)
    project_opts = _build_project_options_stub(
        canonical_crs=canonical_crs,
        resolution=resolution,
        aoi_stub=aoi_stub,
        grid=grid,
        output_dir=output_dir,
    )
    with project_path.open("w", encoding="utf-8") as f:
        toml.dump(project_opts, f)

    # 8) Summary
    print(f"Created: {project_path}")
    print(f"CRS:        {canonical_crs}")
    print(f"Proj4:      {proj4_str}")
    print(f"Resolution: {resolution}")
    print(f"Raw extent:  ({xmin_raw}, {ymin_raw}) – ({xmax_raw}, {ymax_raw})")
    print(f"Final extent:({xmin}, {ymin}) – ({xmax}, {ymax})")
    print(f"Dimensions: nx={nx}, ny={ny}")
    print(f"Source:     {source}")
    print(f"Snap mode:  {args.snap}")

    # 9) Optional: write the grid polygon for inspection
    if output_shapefile:
        grid_polygon_gdf = griddef_to_polygon_gdf(grid)
        shapefile_path = output_dir / str(output_shapefile)
        shapefile_path.parent.mkdir(parents=True, exist_ok=True)
        grid_polygon_gdf.to_file(shapefile_path)
        print(f"Wrote grid polygon shapefile: {shapefile_path}")


if __name__ == "__main__":
    main()