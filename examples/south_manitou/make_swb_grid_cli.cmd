@echo off
REM Create project_options.toml for the South Manitou example.
REM Run this from examples\south_manitou (current directory).

setlocal

set SCRIPT=swb2-prep-grid
set PROJECT_OPTIONS=project_options.toml
set AOI=..\..\data\aoi.shp
set OUTPUT_DIR=output
set OUTPUT_SHAPEFILE_NAME=swb_output_grid_extent.shp

echo Running %SCRIPT% ...
%SCRIPT% ^
  --project-options "%PROJECT_OPTIONS%" ^
  --polygon "%AOI%" ^
  --resolution 30 ^
  --epsg EPSG:5070 ^
  --snap outward ^
  --output-shapefile %OUTPUT_SHAPEFILE_NAME% ^
  --output-dir "%OUTPUT_DIR%"

if errorlevel 1 (
  echo.
  echo ERROR: swb2-prep-grid failed. See messages above.
  exit /b 1
)

echo.
echo SUCCESS: Wrote %PROJECT_OPTIONS% and grid polygon to "%OUTPUT_DIR%\%OUTPUT_SHAPEFILE_NAME%".
endlocal
