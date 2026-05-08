import geopandas as gpd
import pandas as pd
import numpy as np

gpkg_paths = [r"gnatsgo__south_manitou.gpkg",r"gNATSGO_02_03_2025.gpkg"]

for gpkg_path in gpkg_paths:
    # Read without assuming FID
    mu = gpd.read_file(gpkg_path, layer='muaggatt', ignore_geometry=True, use_arrow=True)

    print(f"\n== {gpkg_path} ==")

    print("Columns:", list(mu.columns))
    print("\nFirst 10 rows:\n", mu.head(10))

    # Try to find MUKEY column by name
    mukey_col = next((c for c in mu.columns if c.lower() == 'mukey'), None)
    print("\nDetected MUKEY column:", mukey_col)

    if mukey_col is None:
        print("\nNo MUKEY column found. Checking FID as a fallback...")
        mu_fid = gpd.read_file(gpkg_path, layer='muaggatt', fid_as_index=True,
                            ignore_geometry=True, use_arrow=True)
        mu_fid['FID_as_MUKEY'] = pd.to_numeric(mu_fid.index, errors='coerce')
        print(mu_fid[['FID_as_MUKEY']].head(10))
    else:
        # Coerce MUKEY to numeric and summarize
        mu['MUKEY_num'] = pd.to_numeric(mu[mukey_col], errors='coerce')
        non_numeric = mu['MUKEY_num'].isna().sum()
        print(f"\nNon-numeric MUKEY count: {non_numeric} out of {len(mu)}")
        if non_numeric > 0:
            print("Sample non-numeric MUKEYs:", mu.loc[mu['MUKEY_num'].isna(), mukey_col].head(10).tolist())
        else:
            mu['MUKEY_num'] = mu['MUKEY_num'].astype(np.int64)
            print("MUKEY stats:",
                "unique:", mu['MUKEY_num'].nunique(),
                "min:", mu['MUKEY_num'].min(),
                "max:", mu['MUKEY_num'].max())
            print("Sample MUKEYs:", mu['MUKEY_num'].head(10).tolist())

    print("\n-----------------------------------------------------------------\n\n")