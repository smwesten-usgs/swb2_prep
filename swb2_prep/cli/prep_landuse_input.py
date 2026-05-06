# prep_landuse_input.py
# -*- coding: utf-8 -*-
"""Prepare landuse raster for SWB model input (XR-first pipeline).

Workflow:
    - Load project settings from a 'project_options.toml'.
    - Load the area-of-interest (AOI) polygon or bounding box.
    - Read the input raster as an xarray.DataArray (CRS/transform via rioxarray).
    - Reproject → project CRS (if needed).
    - Resample → project resolution (if needed).
    - Clip → AOI polygon.
    - Write GeoTIFF + ArcASCII outputs.
    - Print minimal grid metadata useful for SWB control-file generation.

CLI contract (current behavior):
    - Required flags: --input, --output-dir, --polygon
    - Optional flags: --bbox (XMIN YMIN XMAX YMAX), --resampling,
      --polygon-name, --polygon-value
    - Optional: --config PATH to explicitly point at a 'project_options.toml';
      if omitted, the CLI reads 'project_options.toml' from the subprocess
      current working directory (CWD), as validated by tests.

Design:
    - Procedural CLI using argparse + pathlib.
    - XR-first functions from swb2_prep/common (rioxarray + rasterio under the hood).
"""

import gc
import argparse
from pathlib import Path
import geopandas as gpd
from pyproj import CRS as _CRS
from rasterio.enums import Resampling

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
from swb2_prep.common.cli_args import (
    add_common_io_args, 
    add_common_aoi_args
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the landuse preprocessing CLI.

    Returns:
        argparse.Namespace: Parsed arguments including:

        Required:
            --input (Path): Path to the source landuse raster.
            --output-dir (Path): Directory to write outputs (GeoTIFF + ArcASCII).
            --polygon (Path): AOI polygon dataset (e.g., a shapefile or GeoPackage).

        Optional:
            --bbox (float x4): AOI as bounding box coordinates (XMIN YMIN XMAX YMAX) in the
                project CRS.
            --polygon-name (str): Attribute/field name used to select a single AOI feature;
                requires --polygon-value.
            --polygon-value (str): Attribute value used to select a single AOI feature;
                requires --polygon-name.
            --config (Path): Optional path to 'project_options.toml'. If omitted, the CLI
                reads 'project_options.toml' from the subprocess CWD (the test suite depends
                on this default behavior).

    Notes:
        The end-to-end tests assert that the CLI finds 'project_options.toml' in its
        working directory and that AOI ambiguity rules are enforced (supplying only
        --polygon-name or only --polygon-value must raise errors).
    """
    p = argparse.ArgumentParser(
        description="Prepare landuse raster for SWB project (XR-first)."
    )
    add_common_io_args(p)
    add_common_aoi_args(p)
    return p.parse_args()


def load_aoi(args: argparse.Namespace, project_crs: str) -> gpd.GeoDataFrame:
    """Load the AOI polygon from a dataset or a bounding box, then reproject to the project CRS.

    Args:
        args: Parsed CLI arguments containing either --polygon or --bbox, optionally with
            --polygon-name/--polygon-value to select a single feature.
        project_crs: Project CRS string (e.g., "EPSG:5070").

    Returns:
        GeoDataFrame containing a single AOI polygon in the project CRS. If --polygon-name
        and --polygon-value are provided, the AOI is narrowed to a single feature via
        attribute selection and dissolved if multiple matches.

    Raises:
        ValueError:
            - If both or neither of --polygon and --bbox are provided.
            - If only one of --polygon-name or --polygon-value is given.
            - If the specified attribute name does not exist in the polygon dataset.

    Notes:
        The AOI polygon is reprojected to the project CRS prior to clipping. Tests assert
        the AOI selection error rules and CWD-based project options behavior.
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
    """Execute the XR-first landuse preprocessing workflow.

    Steps:
        1) Load project options from a 'project_options.toml':
           - If --config PATH is supplied, load from that path.
           - Otherwise, load 'project_options.toml' from the subprocess CWD (as the tests expect).
        2) Load AOI (polygon or bbox) and reproject to project CRS.
        3) Read input raster via rioxarray (CRS/transform attached).
        4) Reproject → project CRS (if needed).
        5) Resample → project resolution (if needed).
        6) Clip → AOI polygon.
        7) Write GeoTIFF + ArcASCII outputs with standardized filenames.
        8) Print minimal grid metadata for downstream SWB control-file generation.

    Notes:
        The CLI contract and AOI rules are validated by the end-to-end tests; keep behavior
        explicit and avoid implicit changes to arguments or IO semantics.
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
    except:
        proj4 = str(da.rio.crs)
    print(nx, ny, llx, lly, project_resolution, proj4)
    
    del da
    gc.collect()

if __name__ == "__main__":
    main()