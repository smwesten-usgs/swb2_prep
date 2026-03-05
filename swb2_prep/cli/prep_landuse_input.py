# prep_landuse_input.py
# -*- coding: utf-8 -*-
"""
Prepare landuse raster for SWB model input (XR-first pipeline).

This script implements the SWB2-prep raster-processing workflow:

1. Load project settings
2. Load the area-of-interest (AOI) polygon or bounding box
3. Load the input raster as xarray.DataArray
4. Reproject → project CRS (if needed)
5. Resample → project resolution (if needed)
6. Clip → AOI polygon
7. Write GeoTIFF + ArcASCII outputs
8. Print grid metadata for SWB control-file generation

Design:
- Procedural CLI with argparse + pathlib
- XR-first functions from swb2_prep/common (rioxarray + rasterio under the hood)
"""

import argparse
from pathlib import Path
import geopandas as gpd
from pyproj import CRS as _CRS

from swb2_prep.common.config import load_project_options
from swb2_prep.common.grids import (
    reproject_raster_xr,
    reproject_polygon,
    resample_raster_xr,
)
from swb2_prep.common.ops import (
    clip_raster_to_polygon_xr,
    create_polygon_from_bbox,
)
from swb2_prep.common.io import (
    read_raster_xr,
    write_geotiff_xr,
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
        Parsed CLI arguments:

        --input : str
            Input landuse raster (GeoTIFF).
        --output-dir : str, optional
            Directory for output files (default from project options).
        --polygon : str, optional
            AOI shapefile path.
        --polygon-name : str, optional
            Field name used to select AOI polygon.
        --polygon-value : str, optional
            Field value used to select AOI polygon.
        --bbox : float xmin ymin xmax ymax, optional
            AOI bounding box coordinates in project CRS.
        --config : str, optional
            Path to project_options.toml (default: project_options.toml).
    """
    p = argparse.ArgumentParser(
        description="Prepare landuse raster for SWB project (XR-first)."
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
    Load the AOI polygon from a shapefile or a bounding box, returning a
    single-row GeoDataFrame in the project CRS.

    Selection rules for shapefile AOIs
    ----------------------------------
    1) If both ``--polygon-name`` and ``--polygon-value`` are supplied:
       - Select polygons where the field equals the specified value.
       - If zero matches → error; if one match → OK; if >1 → error (ambiguous).
    2) If neither is supplied:
       - If shapefile contains exactly one polygon → OK; else → error.
    3) If only one of the pair is supplied → error.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    project_crs : str
        Project CRS string (e.g., "EPSG:5070").

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame containing the AOI polygon in ``project_crs``.

    Raises
    ------
    ValueError
        If AOI selection is ambiguous or inconsistent with options.
    """
    # BOUNDING BOX MODE
    if args.bbox:
        xmin, ymin, xmax, ymax = args.bbox
        return create_polygon_from_bbox(xmin, ymin, xmax, ymax, project_crs)

    # POLYGON MODE
    if not args.polygon:
        raise ValueError("Either --polygon or --bbox must be provided.")

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
            raise ValueError(f"Field {field!r} not found in shapefile.")

        sel = gdf[gdf[field] == value]

        if len(sel) == 0:
            raise ValueError(f"No polygons found where {field} == {value!r}.")
        if len(sel) > 1:
            raise ValueError(
                f"Multiple polygons match {field} == {value!r}. "
                "Selection is ambiguous."
            )

        gdf = sel

    # Case 2: Neither supplied → require 1 polygon
    elif not has_name and not has_value:
        if len(gdf) == 0:
            raise ValueError("Shapefile contains no polygons.")
        if len(gdf) > 1:
            raise ValueError(
                "Shapefile contains multiple polygons; "
                "must provide --polygon-name and --polygon-value."
            )

    # Reproject to project CRS
    return reproject_polygon(gdf, project_crs)


def main() -> None:
    """
    Execute the landuse preprocessing workflow (XR-first).

    Steps
    -----
    1. Read project settings from ``project_options.toml``.
    2. Load AOI using shapefile or bounding box (reproject to project CRS).
    3. Read input raster as xarray.DataArray.
    4. Reproject raster to project CRS (if needed).
    5. Resample raster to project resolution (if needed).
    6. Clip raster to AOI polygon.
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

    output_dir = ensure_dir(Path(args.output_dir or opts["paths"]["output_dir"]))

    # 2. Load AOI (GeoDataFrame in project CRS)
    aoi_gdf = load_aoi(args, project_crs)

    # 3. Load input raster as DataArray (CRS/transform via .rio)
    da = read_raster_xr(args.input)  # masked=True by default (NoData -> NaN)

    # 4/5. Reproject and/or resample to project CRS/resolution
    if str(da.rio.crs) != str(project_crs):
        da = reproject_raster_xr(da, project_crs, resolution=project_resolution)
    else:
        xres = float(da.rio.transform().a)
        if abs(xres - project_resolution) > 1e-6:
            da = resample_raster_xr(da, target_resolution=project_resolution)

    # 6. Clip raster to AOI polygon (auto-reprojects AOI if CRSs differ)
    da = clip_raster_to_polygon_xr(da, aoi_gdf)

    # 7. Build output filenames
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

    # 8. Write outputs (explicit dtype keeps GeoTIFF predictable)
    write_geotiff_xr(
        output_dir / fname_tif,
        da,
        dtype="uint8",
        nodata=255,
        compress="LZW",
        tiled=True,
    )

    write_arc_ascii(
        output_dir / fname_asc,
        da,
        dtype="int16",
        transform=da.rio.transform(),
        crs=da.rio.crs,
        nodata=-9999,
        decimal_precision=0,
        force_cellsize=True,
    )

    # 9. Print control-file metadata (lower-left from upper-left transform)
    nx = da.sizes["x"]
    ny = da.sizes["y"]
    T = da.rio.transform()
    llx = T.c
    lly = T.f + T.e * ny  # lower-left y from upper-left transform
    # proj4 string (robust via pyproj)
    try:
        proj4 = _CRS.from_user_input(da.rio.crs).to_proj4()
    except Exception:
        proj4 = str(da.rio.crs)

    print(nx, ny, llx, lly, project_resolution, proj4)


if __name__ == "__main__":
    main()