import geopandas as gpd
import rasterio
import rasterio.fill
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def extract_awc_from_gnatsgo_windowed(
    raster_path: str,
    output_path: str,
    gpkg_path: str,
    lower_left: Tuple[float, float],
    upper_right: Tuple[float, float],
    nodata_mukey: Optional[int] = 0,
    fill_nodata: bool = True,
    fill_max_search_distance: int = 250
):
    """
    Build a windowed AWC raster (inches per foot) by mapping MUKEY values
    from a window of a MUKEY raster to available water capacity attributes
    stored in muaggatt (GeoPackage).

    Parameters
    ----------
    raster_path : str
        Path to the MUKEY raster (e.g., 'muraster_30m.tif' or your subset).
    output_path : str
        Path to write the windowed AWC raster (GeoTIFF).
    gpkg_path : str
        Path to the GeoPackage containing the 'muaggatt' table.
        Assumes FID=MUKEY (as in your subset) or that FID indexes MUKEY.
    lower_left : (x, y)
        Lower-left corner of the area of interest in the raster's CRS.
    upper_right : (x, y)
        Upper-right corner of the area of interest in the raster's CRS.
    nodata_mukey : int, optional
        MUKEY value in the raster that should be treated as NoData (default 0).
    fill_nodata : bool, optional
        If True, fill small holes in the output with rasterio.fill.fillnodata.
    fill_max_search_distance : int, optional
        Max search distance for fillnodata (default 250 pixels).

    Notes
    -----
    - Uses muaggatt with fid_as_index=True so the GeoPackage FID becomes MUKEY.
    - Computes AWC in inches per foot from aws0150wta if available; falls back
      to aws0100wta with appropriate conversion.
    """

    # 1) Read muaggatt; use feature id (FID) as index -> MUKEY
    muaggatt = gpd.read_file(
        gpkg_path,
        layer='muaggatt',
        fid_as_index=True,         # FID is used as MUKEY
        ignore_geometry=True,
        use_arrow=True
    )

    # Treat index as MUKEY and cast to uint32 to match raster pixel values
    muaggatt['MUKEY'] = pd.to_numeric(muaggatt.index, errors='coerce')
    muaggatt = muaggatt.dropna(subset=['MUKEY']).copy()
    muaggatt['MUKEY'] = muaggatt['MUKEY'].astype(np.uint32)

    # 2) Compute AWC (inches per foot)
    # Prefer aws0150wta; fallback to aws0100wta if needed. If neither present, raise.
    if 'aws0150wta' in muaggatt.columns:
        # aws0150wta: total inches over 150 cm
        muaggatt['awc_in_per_ft'] = (muaggatt['aws0150wta'].astype(float) / 150.0) * 12.0
    elif 'aws0100wta' in muaggatt.columns:
        # aws0100wta: total inches over 100 cm
        muaggatt['awc_in_per_ft'] = (muaggatt['aws0100wta'].astype(float) / 100.0) * 12.0
    else:
        raise KeyError(
            "Could not find AWS columns in muaggatt (expected one of: aws0150wta, aws0100wta)."
        )

    # Replace NaNs with 0.0 (or choose another policy)
    muaggatt['awc_in_per_ft'] = muaggatt['awc_in_per_ft'].fillna(0.0).astype(np.float32)

    # 3) Read only the requested window from the MUKEY raster
    with rasterio.open(raster_path) as src:
        window = src.window(lower_left[0], lower_left[1], upper_right[0], upper_right[1])
        data = src.read(1, window=window)           # MUKEY values in window
        transform = src.window_transform(window)
        crs = src.crs

    # 4) Reduce muaggatt to MUKEYs present in window
    window_mukeys = np.unique(data)
    if nodata_mukey is not None:
        window_mukeys = window_mukeys[window_mukeys != np.uint32(nodata_mukey)]

    muaggatt_sub = muaggatt[muaggatt['MUKEY'].isin(window_mukeys)]
    mukey_to_awc = dict(zip(muaggatt_sub['MUKEY'], muaggatt_sub['awc_in_per_ft']))

    # 5) Vectorized mapping to output raster (float32)
    default_val = np.float32(-1.0)
    output = np.full(data.shape, default_val, dtype=np.float32)

    for mk in window_mukeys:
        val = mukey_to_awc.get(np.uint32(mk))
        if val is not None:
            output[data == mk] = val

    # Optional: fill nodata holes to improve continuity
    if fill_nodata:
        # Build mask where we have valid (non-default) values
        mask = (output != default_val).astype(np.uint8)
        output = rasterio.fill.fillnodata(
            image=output,
            mask=mask,
            max_search_distance=fill_max_search_distance
        ).astype(np.float32)

    # 6) Write the windowed AWC raster
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=output.shape[0], width=output.shape[1],
        count=1, dtype=output.dtype,
        crs=crs, transform=transform,
        nodata=default_val
    ) as dst:
        dst.write(output, 1)


if __name__ == "__main__":
    # Mirror the HSG script's main block for consistency
    raster_path = r"mukey__south_manitou.tif"
    gpkg_path   = r"gnatsgo__south_manitou.gpkg"

    # Output windowed AWC raster (in/ft)
    output_path = r"output_awc_windowed.tif"

    # Area of interest (in the raster's CRS)
    lower_left  = (773040, 2485074)
    upper_right = (779784, 2491710)

    extract_awc_from_gnatsgo_windowed(
        raster_path=raster_path,
        output_path=output_path,
        gpkg_path=gpkg_path,
        lower_left=lower_left,
        upper_right=upper_right,
        nodata_mukey=0,
        fill_nodata=True,
        fill_max_search_distance=250
    )