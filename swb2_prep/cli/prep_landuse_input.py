"""
Prepare landuse raster for SWB model input.

This script implements the standard SWB-CLI raster-processing workflow
defined in NOTES.md and bootstrap_prompt.txt:

1. Load project settings
2. Load the area-of-interest (AOI) polygon or bounding box
3. Load the input raster
4. Reproject → project CRS (if needed)
5. Resample → project resolution (if needed)
6. Clip → AOI polygon
7. Write GeoTIFF + ArcASCII outputs
8. Print grid metadata for SWB control-file generation

The design is strictly procedural, uses argparse + pathlib, and relies
only on shared functions in common/.
"""

import argparse
from pathlib import Path
import geopandas as gpd

from swb2_prep.common.config import load_project_options
from swb2_prep.common.grids import (
    reproject_raster,
    reproject_polygon,
    resample_raster,
    create_polygon_from_bbox,
)
from swb2_prep.common.ops import clip_raster_to_polygon
from swb2_prep.common.io import (
    read_raster,
    write_geotiff,
    write_arc_ascii,
)
from swb2_prep.common.paths import (
    ensure_dir,
    build_output_filename,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        An object containing parsed CLI arguments.
    """
    p = argparse.ArgumentParser(
        description="Prepare landuse raster for SWB project."
    )

    p.add_argument(
        "--input",
        required=True,
        help="Input landuse raster (GeoTIFF)."
    )

    p.add_argument(
        "--output-dir",
        required=False,
        help="Directory for output files."
    )

    # AOI modes
    p.add_argument(
        "--polygon",
        help="AOI shapefile."
    )

    p.add_argument(
        "--polygon-name",
        help="Field name in shapefile used to select polygon (optional)."
    )

    p.add_argument(
        "--polygon-value",
        help="Field value used to select polygon (optional)."
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

    return p.parse_args()


def load_aoi(args: argparse.Namespace, project_crs: str) -> gpd.GeoDataFrame:
    """
    Load the AOI polygon from a shapefile or a bounding box.

    The selection rules for shapefile AOIs are:

    1. If both ``--polygon-name`` and ``--polygon-value`` are supplied:
       - Select polygons where the field equals the specified value.
       - If zero matches → error.
       - If one match → OK.
       - If more than one match → error (ambiguous selection).

    2. If *neither* ``--polygon-name`` nor ``--polygon-value`` is supplied:
       - If shapefile contains exactly one polygon → OK.
       - If shapefile contains more than one polygon → error.

    3. If only one of ``--polygon-name`` / ``--polygon-value`` is provided:
       - Error (both must be supplied together).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    project_crs : str
        CRS string of the SWB project.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing exactly one polygon in the project CRS.

    Raises
    ------
    ValueError
        If AOI selection is ambiguous or inconsistent with the options.
    """
    # BOUNDING BOX MODE
    if args.bbox:
        xmin, ymin, xmax, ymax = args.bbox
        poly = create_polygon_from_bbox(xmin, ymin, xmax, ymax)
        return gpd.GeoDataFrame(geometry=[poly], crs=project_crs)

    # POLYGON MODE
    if not args.polygon:
        raise ValueError(
            "Either --polygon or --bbox must be provided."
        )

    gdf = gpd.read_file(args.polygon)

    has_name = args.polygon_name is not None
    has_value = args.polygon_value is not None

    # Case 3: Only one provided → error
    if has_name ^ has_value:
        raise ValueError(
            "Both --polygon-name and --polygon-value must be provided "
            "together, or neither."
        )

    # Case 1: Both provided → attribute selection
    if has_name and has_value:
        field = args.polygon_name
        value = args.polygon_value

        if field not in gdf.columns:
            raise ValueError(
                f"Field {field!r} not found in shapefile."
            )

        sel = gdf[gdf[field] == value]

        if len(sel) == 0:
            raise ValueError(
                f"No polygons found where {field} == {value!r}."
            )
        if len(sel) > 1:
            raise ValueError(
                f"Multiple polygons match {field} == {value!r}. "
                "Selection is ambiguous."
            )

        gdf = sel

    # Case 2: Neither supplied → require 1 polygon
    elif not has_name and not has_value:
        if len(gdf) == 0:
            raise ValueError(
                "Shapefile contains no polygons."
            )
        if len(gdf) > 1:
            raise ValueError(
                "Shapefile contains multiple polygons; "
                "must provide --polygon-name and --polygon-value."
            )

    # Reproject to project CRS
    return reproject_polygon(gdf, project_crs)


def main() -> None:
    """
    Execute the landuse preprocessing workflow.

    Steps
    -----
    1. Read project settings from ``project_options.toml``.
    2. Load AOI using shapefile or bounding box.
    3. Read input raster.
    4. Reproject raster to project CRS (if needed).
    5. Resample raster to project resolution (if needed).
    6. Clip raster to AOI.
    7. Write GeoTIFF and ArcASCII outputs.
    8. Print grid metadata for SWB control-file generation.

    Prints
    ------
    int
        Number of columns in processed raster (nx).
    int
        Number of rows in processed raster (ny).
    float
        Lower-left x-coordinate (llx).
    float
        Lower-left y-coordinate (lly).
    float
        Grid resolution.
    str
        proj4 string of CRS.

    Raises
    ------
    FileNotFoundError
        If input files cannot be located.
    ValueError
        For AOI selection or CRS/resolution inconsistencies.
    """
    args = parse_args()

    # 1. Load project options
    opts = load_project_options(args.config)
    project_crs = opts["project"]["crs"]
    project_resolution = opts["project"]["resolution"]
    project_units = opts["project"]["units"]

    output_dir = Path(args.output_dir or opts["paths"]["output_dir"])
    output_dir = ensure_dir(output_dir)

    # 2. Load AOI
    aoi = load_aoi(args, project_crs)

    # 3. Load input raster
    array, profile = read_raster(args.input)
    input_crs = profile["crs"]
    input_res = profile["transform"].a  # pixel width

    # 4. Reproject → project CRS (if needed)
    if input_crs != project_crs:
        array, profile = reproject_raster(array, profile, project_crs)

    # 5. Resample → project resolution (if needed)
    if abs(input_res - project_resolution) > 1e-6:
        array, profile = resample_raster(
            array,
            profile,
            target_resolution=project_resolution
        )

    # 6. Clip raster → AOI
    array, profile = clip_raster_to_polygon(
        array,
        profile,
        aoi
    )

    # 7. Build output filenames
    label = "bounding_box" if args.bbox else "polygon"

    fname_tif = build_output_filename(
        base="landuse",
        resolution=project_resolution,
        units=project_units,
        ext=".tif",
    )

    fname_asc = build_output_filename(
        base="landuse",
        resolution=project_resolution,
        units=project_units,
        ext=".asc",
    )

    # 8. Write outputs
    write_geotiff(output_dir / fname_tif, array, profile)
    write_arc_ascii(output_dir / fname_asc, array, profile)

    # 9. Print control-file metadata
    nx = array.shape[1]
    ny = array.shape[0]
    llx = profile["transform"].c
    lly = profile["transform"].f
    proj4 = profile["crs"].to_proj4()

    print(nx, ny, llx, lly, project_resolution, proj4)


if __name__ == "__main__":
    main()
