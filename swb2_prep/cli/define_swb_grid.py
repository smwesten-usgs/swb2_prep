# swb2_prep/cli/define_swb_grid.py
# -*- coding: utf-8 -*-

"""
Define the SWB grid from an AOI polygon or a bounding box, and
write the grid definition to a TOML file.

This first version does NOT apply snapping yet.
Raw extents == final extents until snapping is integrated.
"""

import argparse
from pathlib import Path

import geopandas as gpd
from pyproj import CRS as _CRS

from swb2_prep.common.config import load_project_options
from swb2_prep.common.griddef import write_grid_definition
from swb2_prep.common.grids import reproject_polygon
from swb2_prep.common.ops import create_polygon_from_bbox


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Define the SWB grid from an AOI polygon or bbox (raw extents only)."
    )

    p.add_argument(
        "--polygon",
        help="AOI shapefile."
    )
    p.add_argument(
        "--polygon-name",
        help="Field name in shapefile used to select AOI polygon (optional)."
    )
    p.add_argument(
        "--polygon-value",
        help="Field value used to select AOI polygon (optional)."
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        help="Bounding box coordinates in project CRS."
    )

    p.add_argument(
        "--config",
        default="project_options.toml",
        help="Path to project_options.toml (default: project_options.toml)"
    )

    p.add_argument(
        "--output",
        required=False,
        help="Output directory for the grid-definition TOML file."
    )

    p.add_argument(
        "--grid-file",
        required=False,
        default="swb_grid_definition.toml",
        help="Optional name for the grid-definition file. Default: swb_grid_definition.toml"
    )

    p.add_argument(
        "--snap",
        choices=["outward", "inward"],
        default="outward",
        help="Snapping mode placeholder (not applied yet)."
    )

    return p.parse_args()


def load_aoi_polygon(args: argparse.Namespace, project_crs: str) -> gpd.GeoDataFrame:
    """
    Load the AOI polygon from shapefile or bounding box.
    Reprojects to project CRS.
    """
    # BBOX MODE
    if args.bbox:
        xmin, ymin, xmax, ymax = args.bbox
        gdf = create_polygon_from_bbox(xmin, ymin, xmax, ymax, project_crs)
        source = "bbox"
        return gdf, source

    # POLYGON MODE
    if not args.polygon:
        raise ValueError("Either --polygon or --bbox must be provided.")

    gdf = gpd.read_file(args.polygon)

    has_name = args.polygon_name is not None
    has_value = args.polygon_value is not None

    # Exactly one of name/value is invalid
    if has_name ^ has_value:
        raise ValueError(
            "Both --polygon-name and --polygon-value must be provided or neither."
        )

    # Attribute selection case
    if has_name and has_value:
        field = args.polygon_name
        value = args.polygon_value
        if field not in gdf.columns:
            raise ValueError(f"Field {field!r} not found in shapefile.")

        sel = gdf[gdf[field] == value]
        if len(sel) == 0:
            raise ValueError(f"No polygons where {field} == {value!r}.")
        if len(sel) > 1:
            raise ValueError(f"Multiple polygons match {field} == {value!r}.")

        gdf = sel
        source = "aoi_polygon"
    else:
        # No name/value provided; require exactly one polygon
        if len(gdf) == 0:
            raise ValueError("Shapefile contains no polygons.")
        if len(gdf) > 1:
            raise ValueError(
                "Shapefile contains multiple polygons; "
                "must provide --polygon-name and --polygon-value."
            )
        source = "aoi_polygon"

    # Reproject to project CRS
    gdf = reproject_polygon(gdf, project_crs)

    return gdf, source


def main() -> None:
    args = parse_args()

    # Load project CRS & resolution
    opts = load_project_options(args.config)
    project_crs = opts["project"]["crs"]
    resolution = float(opts["project"]["resolution"])

    # Determine output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        # Default to project output_dir
        out_dir = Path(opts["paths"]["output_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load AOI in project CRS
    aoi_gdf, source = load_aoi_polygon(args, project_crs)

    # 2. Extract raw extents
    xmin_raw, ymin_raw, xmax_raw, ymax_raw = aoi_gdf.total_bounds

    # 3. Raw extents == final extents for now
    xmin, ymin, xmax, ymax = xmin_raw, ymin_raw, xmax_raw, ymax_raw

    # 4. Compute grid dimensions
    nx = int((xmax - xmin) / resolution)
    ny = int((ymax - ymin) / resolution)

    # 5. Get proj4 string
    try:
        proj4 = _CRS.from_user_input(project_crs).to_proj4()
    except Exception:
        proj4 = str(project_crs)

    # 6. Build grid definition dictionary
    grid = {
        "crs": project_crs,
        "proj4": proj4,
        "resolution": resolution,
        "xmin_raw": xmin_raw,
        "ymin_raw": ymin_raw,
        "xmax_raw": xmax_raw,
        "ymax_raw": ymax_raw,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "nx": nx,
        "ny": ny,
        "snap": args.snap,
        "source": source,
    }

    # 7. Write TOML file
    outfile = out_dir / args.grid_file
    write_grid_definition(outfile, grid)

    # 8. Print summary
    print(f"SWB grid definition written to: {outfile}")
    print(f"CRS:        {project_crs}")
    print(f"Proj4:      {proj4}")
    print(f"Resolution: {resolution}")
    print(f"Raw extent:  ({xmin_raw}, {ymin_raw}) – ({xmax_raw}, {ymax_raw})")
    print(f"Final extent:({xmin}, {ymin}) – ({xmax}, {ymax})")
    print(f"Dimensions: nx={nx}, ny={ny}")
    print(f"Source:     {source}")
    print(f"Snap mode:  {args.snap} (not applied yet)")
    

if __name__ == "__main__":
    main()
