import rasterio
import numpy as np
from rasterio.windows import Window

def describe_raster(path, ll, ur):
    with rasterio.open(path) as src:
        print(f"\n== {path} ==")
        print("CRS:", src.crs)
        print("dtype:", src.dtypes[0])
        print("nodata:", src.nodata)
        print("transform:", src.transform)
        print("bounds:", src.bounds)
        print("width x height:", src.width, src.height)

        # Read window and summarize
        window = src.window(ll[0], ll[1], ur[0], ur[1])
        data = src.read(1, window=window)
        print("window:", window)
        print("window shape:", data.shape)
        print("window min/max:", data.min(), data.max())
        # Sample unique values (avoid huge prints)
        uniq = np.unique(data)
        print("unique count in window:", len(uniq))
        print("sample unique:", uniq[:10])
        # Check integer-ness (floats can look like integers in UI)
        if np.issubdtype(data.dtype, np.floating):
            print("Is integer-like? ->", np.all(np.isclose(data, np.round(data))))
        print("--------------------------------------------------------------\n\n")    

if __name__ == "__main__":
    # Operate directly on the large CONUS raster and the original GPKG
    DOS_batch_ss_raster_path = r"mukey__south_manitou.tif"
    CONUS_raster_path = r"muraster_30m.tif"

        # Area of interest (in the raster's CRS)
    lower_left  = (773040, 2485074)
    upper_right = (779784, 2491710)

    describe_raster(DOS_batch_ss_raster_path, lower_left, upper_right)
    describe_raster(CONUS_raster_path, lower_left, upper_right)

