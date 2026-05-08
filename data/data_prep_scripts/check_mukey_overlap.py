import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import pandas as pd

def check_mukey_overlap(raster_path, gpkg_path, ll, ur):
    mu = gpd.read_file(gpkg_path, layer='muaggatt', ignore_geometry=True, use_arrow=True)
    mukey_col = next((c for c in mu.columns if c.lower() == 'mukey'), None)
    if mukey_col is not None:
        mu['MUKEY'] = pd.to_numeric(mu[mukey_col], errors='coerce')
    else:
        mu = gpd.read_file(gpkg_path, layer='muaggatt', fid_as_index=True, ignore_geometry=True, use_arrow=True)
        mu['MUKEY'] = pd.to_numeric(mu.index, errors='coerce')
    mu = mu.dropna(subset=['MUKEY'])
    mu['MUKEY'] = mu['MUKEY'].astype(np.int32)

    with rasterio.open(raster_path) as src:
        window = from_bounds(ll[0], ll[1], ur[0], ur[1], src.transform)
        data = src.read(1, window=window, masked=True)
    window_mukeys = np.unique(data.compressed()).astype(np.int32)

    mukeys_in_gp = set(mu['MUKEY'].tolist())
    overlap = set(window_mukeys) & mukeys_in_gp
    print(f"\n== {raster_path}, {gpkg_path} ==")
    print("Overlap count:", len(overlap))
    print("Example overlap:", list(overlap)[:10])
    print("Raster-only (sample):", list(set(window_mukeys) - mukeys_in_gp)[:10])
    print("GPKG-only (sample):", list(mukeys_in_gp - set(window_mukeys))[:10])
    print("----------------------------------------------------------------\n\n")

if __name__ == "__main__":
    # Operate directly on the large CONUS raster and the original GPKG
    DOS_Batch_SS_raster_path = r"mukey__south_manitou.tif"
    DOS_Batch_SS_gpkg_path   = r"gnatsgo__south_manitou.gpkg"
    #DOS_Batch_SS_gpkg_path   = r"out_with_mukey.gpkg"

    CONUS_raster_path = r"muraster_30m.tif"
    CONUS_gpkg_path   = r"gNATSGO_02_03_2025.gpkg"

    # Area of interest (in the raster's CRS)
    lower_left  = (773040, 2485074)
    upper_right = (779784, 2491710)

    check_mukey_overlap(DOS_Batch_SS_raster_path, DOS_Batch_SS_gpkg_path, lower_left, upper_right)
    check_mukey_overlap(CONUS_raster_path, CONUS_gpkg_path, lower_left, upper_right)
