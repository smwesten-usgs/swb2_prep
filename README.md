# swb2_prep
[WIP] Another attempt at a Python package to prepare input grids and a basic swb2 control file at command line.

_The following examples assume that the command-line programs are run from inside the `swb2_prep\examples\south_manitou` directory._


### Step 1. Create the SWB grid.

```batch
swb2-prep-grid ^
  --project-options "project_options.toml" ^
  --polygon "..\..\data\aoi.shp" ^
  --resolution 30 ^
  --epsg EPSG:5070 ^
  --snap outward ^
  --output-shapefile swb_output_grid_extent.shp ^
  --output-dir "output"
```

This first script establishes the coordinate reference system, resolution, and extent for the desired swb2 application files. The grid details are written to a TOML file. The remaining command-line utilites expect this TOML file to be present, and will return an error of the file has not yet been created. In this example the TOML file is called 'project_options.toml' and it looks like this:

```toml
#project_options.toml
[project]
crs = "EPSG:5070"
resolution = 30.0
units = "m"

[paths]
input_dir = "data"
output_dir = "output"

[aoi]
bbox = []
polygon = "..\\..\\data\\aoi.shp"
polygon_name = ""
polygon_value = ""

[grid]
crs = "EPSG:5070"
proj4 = "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +type=crs"
resolution = 30.0
snap = "outward"
source = "aoi_polygon"
xmin_raw = 772994.6515829079
ymin_raw = 2485003.7135628797
xmax_raw = 779770.0854417123
ymax_raw = 2491779.147421684
xmin = 772980.0
ymin = 2484990.0
xmax = 779790.0
ymax = 2491800.0
nx = 227
ny = 227
template_tif = "output\\swb_grid_template.tif"

[provenance]
generated_by = "define_swb_grid.py"
generated_on = "2026-05-22T18:50:53+00:00"
version = "0.1.0"
```

![swb project area](images\swb_project_area__south_manitou.png)

### Step 2. Generate land use grid.

The code was developed with the USGS Cropland Data Layer (CDL) in mind. It may be applicable to other land use data sources.

```batch
swb2-prep-landuse ^
  --project-options "project_options.toml" ^
  --input "..\..\data\cdl__south_manitou.tif" ^
  --output-dir "output"  ^
  --dtype int16 ^
  --nodata -1 ^
  --exclude-codes 111 ^
  --prefix south_manitou
```



![land use grid](images\landuse_grid__south_manitou.png)


### Step 3. Generate hydrologic soil group grid.

```batch
swb2-prep-hsg ^
  --project-options "project_options.toml" ^
  --input "..\..\data\mukey__south_manitou.tif" ^
  --gpkg "..\..\data\gnatsgo__south_manitou.gpkg" ^
  --output-dir "output" ^
  --dtype int16 ^
  --nodata -1  ^
  --prefix south_manitou
```

![alt text](images\hydrologic_soil_group_grid__south_manitou.png)