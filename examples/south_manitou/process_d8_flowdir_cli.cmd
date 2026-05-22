@echo off
REM Compute D8 flow direction from DEM, aligned to the canonical grid from project_options.toml.
setlocal

set SCRIPT=python -m swb2_prep.cli.prep_d8_flowdir
set PROJECT_OPTIONS=project_options.toml
set INPUT=..\..\data\hydrosheds_dem__south_manitou.tif
set OUTPUT_DIR=output

%SCRIPT% ^
  --project-options "%PROJECT_OPTIONS%" ^
  --input "%INPUT%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --dtype int16 ^
  --nodata -9999 ^
  --resampling bilinear ^
  --prefix south_manitou

if errorlevel 1 (
  echo ERROR: D8 flow direction computation failed.
  exit /b 1
)

echo SUCCESS: D8 flow direction outputs written to "%OUTPUT_DIR%".
endlocal
