@echo off
REM Prepare landuse raster aligned to the canonical grid from project_options.toml.
setlocal

set SCRIPT=python -m swb2_prep.cli.prep_landuse_input
set PROJECT_OPTIONS=project_options.toml
set INPUT=..\..\data\cdl__south_manitou.tif
set OUTPUT_DIR=output

%SCRIPT% ^
  --project-options "%PROJECT_OPTIONS%" ^
  --input "%INPUT%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --dtype int16 ^
  --nodata -1 ^
  --exclude-codes 111 ^
  --prefix south_manitou

if errorlevel 1 (
  echo ERROR: landuse prep failed.
  exit /b 1
)

echo SUCCESS: Landuse outputs written to "%OUTPUT_DIR%".
endlocal