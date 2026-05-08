import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def extract_hsg_from_gnatsgo_windowed(
    raster_path: str,
    output_path: str,
    gpkg_path: str,
    lower_left: Tuple[float, float],
    upper_right: Tuple[float, float],
    nodata_mukey: Optional[int] = 0
):
    """
    Build a windowed HSG raster by mapping MUKEY values (from a window of a large MUKEY raster)
    to hydrologic soil group codes stored in muaggatt (GeoPackage).

    Parameters
    ----------
    raster_path : str
        Path to the CONUS MUKEY raster (e.g., 'muraster_30m.tif').
    output_path : str
        Path to write the windowed HSG raster (GeoTIFF).
    gpkg_path : str
        Path to the GeoPackage containing the 'muaggatt' table. Works with either:
          - the original CONUS GPKG, or
          - your subset GPKG rebuilt with FID=mukey.
    lower_left : (x, y)
        Lower-left corner of the area of interest in the raster's CRS.
    upper_right : (x, y)
        Upper-right corner of the area of interest in the raster's CRS.
    nodata_mukey : int, optional
        MUKEY value in the raster that should be treated as NoData (default 0).

    Notes
    -----
    - The function reads muaggatt with fid_as_index=True, so the GeoPackage FID (primary key)
      becomes the GeoDataFrame index. In schemas where FID=mukey, the index equals MUKEY.
    - Hydrologic group column is detected case-insensitively among common names:
      ['hydgrpdcd', 'HYDGRPDCD', 'hydrolgrp', 'HYDROLGRP'].
    """

    # 1) Read muaggatt; use feature id (FID) as index -> MUKEY
    muaggatt = gpd.read_file(
        gpkg_path,
        layer='muaggatt',
        fid_as_index=True,         # make FID the index (often equals MUKEY)
        ignore_geometry=True,
        use_arrow=True             # fast path if pyarrow is available
    )

    # Treat index as MUKEY and cast to uint32 to match raster pixel values
    muaggatt['MUKEY'] = pd.to_numeric(muaggatt.index, errors='coerce')
    muaggatt = muaggatt.dropna(subset=['MUKEY'])
    muaggatt['MUKEY'] = muaggatt['MUKEY'].astype(np.uint32)

    # Find HSG column (case-insensitive)
    def find_hsg_col(cols):
        candidates = ['hydgrpdcd', 'HYDGRPDCD', 'hydrolgrp', 'HYDROLGRP']
        lower = {c.lower(): c for c in cols}
        for name in candidates:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    hsg_col = find_hsg_col(muaggatt.columns)
    if hsg_col is None:
        raise KeyError(
            "Could not find a hydrologic soil group column in muaggatt "
            "(expected one of: hydgrpdcd, HYDGRPDCD, hydrolgrp, HYDROLGRP)."
        )

    # Map HSG strings to integer codes
    hsg_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'A/D': 5, 'B/D': 6, 'C/D': 7, None: 1}
    muaggatt['hsg_numeric'] = muaggatt[hsg_col].map(hsg_mapping).fillna(1).astype(np.int32)

    # 2) Read only the requested window from the big raster
    with rasterio.open(raster_path) as src:
        # Create a window using raster coordinates: (left, bottom, right, top)
        window = src.window(lower_left[0], lower_left[1], upper_right[0], upper_right[1])

        # Read MUKEYs in the window
        data = src.read(1, window=window)

        # Get the affine transform for the windowed subset
        transform = src.window_transform(window)
        crs = src.crs

    # 3) Reduce muaggatt to MUKEYs actually present in the window (speed/memory)
    window_mukeys = np.unique(data)

    # Drop NoData MUKEY if provided
    if nodata_mukey is not None:
        window_mukeys = window_mukeys[window_mukeys != np.uint32(nodata_mukey)]

    muaggatt_sub = muaggatt[muaggatt['MUKEY'].isin(window_mukeys)]

    # Build lookup dict: MUKEY -> HSG code
    mukey_to_hsg = dict(zip(muaggatt_sub['MUKEY'], muaggatt_sub['hsg_numeric']))

    # 4) Vectorized mapping to output raster
    default_val = -1
    output = np.full(data.shape, default_val, dtype=np.int32)

    # Assign by unique MUKEY values
    for mk in window_mukeys:
        val = mukey_to_hsg.get(np.uint32(mk))
        if val is not None:
            output[data == mk] = val

    # 5) Write the windowed HSG raster
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=output.shape[0], width=output.shape[1],
        count=1, dtype=output.dtype,
        crs=crs, transform=transform,
        nodata=default_val
    ) as dst:
        dst.write(output, 1)


if __name__ == "__main__":
    # Operate directly on the large CONUS raster and the original GPKG
    raster_path = r"mukey__south_manitou.tif"
    gpkg_path   = r"gnatsgo__south_manitou.gpkg"

#    raster_path = r"muraster_30m.tif"
#    gpkg_path   = r"gNATSGO_02_03_2025.gpkg"

    # Output windowed HSG raster
    output_path = r"output_hsg_windowed.tif"

    # Area of interest (in the raster's CRS)
    lower_left  = (773040, 2485074)
    upper_right = (779784, 2491710)

    # If your MUKEY raster uses a different NoData than 0, adjust nodata_mukey
    extract_hsg_from_gnatsgo_windowed(
        raster_path=raster_path,
        output_path=output_path,
        gpkg_path=gpkg_path,
        lower_left=lower_left,
        upper_right=upper_right,
        nodata_mukey=0
    )