# -*- coding: utf-8 -*-
"""
Define the canonical SWB grid and write project_options.toml.

This CLI:
- Takes an AOI (polygon shapefile or bounding box) + CRS + resolution.
- Snaps extents to produce exact integer grid dimensions.
- Writes a grid template GeoTIFF (for use by downstream CLIs via reproject_match).
- Writes project_options.toml with the canonical [grid] section.
- Optionally writes a shapefile of the grid bounding box.

Create-only: errors if project_options.toml already exists.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import toml
from pyproj import CRS as _CRS
from rasterio.transform import Affine

from swb2_prep.common.grids import snap_extent, compute_grid_dims
from swb2_prep.common.log import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for grid definition."""
    p = argparse.ArgumentParser(
        description="Create project_options.toml with canonical [grid]. Errors if file exists."
    )
    # AOI source (mutually exclusive in practice)
    p.add_argument("--polygon", type=Path, help="AOI polygon shapefile.")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="AOI bounding box in project CRS.")
    p.add_argument("--polygon-name", dest="polygon_name", type=str,
                   help="Attribute name to select a single polygon feature.")
    p.add_argument("--polygon-value", dest="polygon_value", type=str,
                   help="Attribute value to select a single polygon feature.")

    # CRS (exactly one required)
    p.add_argument("--epsg", type=str, help="CRS as EPSG code (e.g., 'EPSG:5070').")
    p.add_argument("--proj4", type=str, help="CRS as Proj4 string.")

    # Grid parameters
    p.add_argument("--resolution", type=float, required=True, help="Grid cell size in CRS units.")
    p.add_argument("--snap", choices=["outward", "inward"], default="outward",
                   help="Snap mode (default: outward).")

    # Output
    p.add_argument("--project-options", dest="project_options", type=Path,
                   default=Path.cwd() / "project_options.toml",
                   help="Output TOML path (default: ./project_options.toml).")
    p.add_argument("--output-dir", dest="output_dir", type=Path, default=Path.cwd(),
                   help="Directory for template GeoTIFF and shapefile (default: cwd).")
    p.add_argument("--output-shapefile", dest="output_shapefile", type=Path,
                   help="Optional: write grid bounding box as shapefile.")
    p.add_argument("--no-log", dest="no_log", action="store_true", default=False,
                   help="Suppress logging to file.")
    return p.parse_args()


def load_aoi_polygon(
    polygon_path: Path | None,
    bbox: list[float] | None,
    polygon_name: str | None,
    polygon_value: str | None,
    project_crs: str,
) -> gpd.GeoDataFrame:
    """Load AOI as a single-polygon GeoDataFrame in the project CRS.

    Args:
        polygon_path: Path to AOI shapefile, or None.
        bbox: [xmin, ymin, xmax, ymax] in project CRS, or None.
        polygon_name: Attribute name for feature selection.
        polygon_value: Attribute value for feature selection.
        project_crs: Target CRS string.

    Returns:
        Single-row GeoDataFrame in project_crs.

    Raises:
        ValueError: If inputs are missing, ambiguous, or invalid.
    """
    from shapely.geometry import box

    if bbox:
        geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        return gpd.GeoDataFrame({"geometry": [geom]}, crs=project_crs)

    if not polygon_path:
        raise ValueError("Provide --polygon or --bbox.")

    gdf = gpd.read_file(polygon_path)

    # Optional feature selection
    if (polygon_name is None) != (polygon_value is None):
        raise ValueError("Provide both --polygon-name and --polygon-value, or neither.")

    if polygon_name and polygon_value:
        if polygon_name not in gdf.columns:
            raise ValueError(f"Field {polygon_name!r} not found in shapefile.")
        gdf = gdf[gdf[polygon_name] == polygon_value]
        if len(gdf) == 0:
            raise ValueError(f"No polygons where {polygon_name} == {polygon_value!r}.")
        if len(gdf) > 1:
            raise ValueError(f"Multiple polygons match {polygon_name} == {polygon_value!r}.")
    else:
        if len(gdf) == 0:
            raise ValueError("Shapefile contains no polygons.")
        if len(gdf) > 1:
            raise ValueError("Multiple polygons; use --polygon-name/--polygon-value to select one.")

    return gdf.to_crs(project_crs)


def write_grid_template_tif(out_dir: Path, grid: dict) -> Path:
    """Write a single-band GeoTIFF carrying the canonical grid CRS/transform/shape.

    Args:
        out_dir: Output directory (created if needed).
        grid: Grid dict with keys: crs, xmin, ymax, resolution, nx, ny.

    Returns:
        Path to the written template GeoTIFF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "swb_grid_template.tif"

    nx, ny = int(grid["nx"]), int(grid["ny"])
    resolution = float(grid["resolution"])
    transform = Affine.translation(grid["xmin"], grid["ymax"]) * Affine.scale(resolution, -resolution)

    arr = np.ones((ny, nx), dtype=np.uint16)

    with rio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1,
                  dtype="uint16", crs=str(grid["crs"]), transform=transform,
                  tiled=True, compress="lzw", nodata=0) as dst:
        dst.write(arr, 1)

    return path


def main() -> None:
    """Define the SWB grid and write project_options.toml + template GeoTIFF."""
    args = parse_args()

    # Set up logging (log to project_options parent dir)
    log = setup_logging("define_swb_grid", args.project_options.parent, no_log=args.no_log)
    log.info(f"CLI arguments: {vars(args)}")

    # Enforce create-only
    if args.project_options.exists():
        raise FileExistsError(
            f"{args.project_options} already exists. Remove or rename it first."
        )

    # Resolve CRS
    if args.epsg and args.proj4:
        raise ValueError("Provide --epsg or --proj4, not both.")
    if not args.epsg and not args.proj4:
        raise ValueError("Provide CRS via --epsg or --proj4.")
    crs_obj = _CRS.from_user_input(args.epsg or args.proj4)
    canonical_crs = crs_obj.to_string()
    proj4_str = crs_obj.to_proj4()

    # Validate resolution
    resolution = float(args.resolution)
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")

    # Load AOI polygon in project CRS
    aoi_gdf = load_aoi_polygon(
        args.polygon, args.bbox, args.polygon_name, args.polygon_value, canonical_crs
    )

    # Snap extents and compute grid dimensions
    xmin_raw, ymin_raw, xmax_raw, ymax_raw = map(float, aoi_gdf.total_bounds)
    xmin, ymin, xmax, ymax = [float(v) for v in snap_extent(
        xmin_raw, ymin_raw, xmax_raw, ymax_raw, resolution, mode=args.snap
    )]
    nx, ny = compute_grid_dims(xmin, ymin, xmax, ymax, resolution)

    # Build canonical grid dict
    grid = {
        "crs": canonical_crs,
        "proj4": proj4_str,
        "resolution": resolution,
        "snap": args.snap,
        "source": "bbox" if args.bbox else "aoi_polygon",
        "xmin_raw": xmin_raw, "ymin_raw": ymin_raw, "xmax_raw": xmax_raw, "ymax_raw": ymax_raw,
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "nx": nx, "ny": ny,
    }

    # Write template GeoTIFF
    template_path = write_grid_template_tif(args.output_dir, grid)
    grid["template_tif"] = str(template_path)
    log.info(f"Wrote grid template: {template_path}")

    # Write project_options.toml
    project_opts = {
        "project": {"crs": canonical_crs, "resolution": resolution, "units": "m"},
        "paths": {"input_dir": "data", "output_dir": str(args.output_dir)},
        "aoi": {
            "bbox": list(args.bbox) if args.bbox else [],
            "polygon": str(args.polygon) if args.polygon else "",
            "polygon_name": args.polygon_name or "",
            "polygon_value": args.polygon_value or "",
        },
        "grid": grid,
        "provenance": {
            "generated_by": "define_swb_grid.py",
            "generated_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "version": "0.1.0",
        },
    }
    with args.project_options.open("w", encoding="utf-8") as f:
        toml.dump(project_opts, f)

    # Summary
    log.info(f"Created: {args.project_options}")
    log.info(f"Created swb grid:")
    log.info(f"  CRS:        {canonical_crs}")
    log.info(f"  Resolution: {resolution}")
    log.info(f"  Extent:     ({xmin}, {ymin}) – ({xmax}, {ymax})")
    log.info(f"  Dimensions: {nx} x {ny}")
    log.info(f"  Snap:       {args.snap}")

    # Optional: write grid bounding box shapefile
    if args.output_shapefile:
        from shapely.geometry import box
        gdf = gpd.GeoDataFrame({"geometry": [box(xmin, ymin, xmax, ymax)]}, crs=canonical_crs)
        shp_path = args.output_dir / str(args.output_shapefile)
        gdf.to_file(shp_path)
        log.info(f"Wrote grid shapefile: {shp_path}")


if __name__ == "__main__":
    main()
