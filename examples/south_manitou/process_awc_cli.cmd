@echo off
REM Prepare AWC raster (inches/foot) aligned to the canonical grid from project_options.toml.
setlocal

set SCRIPT=python -m swb2_prep.cli.prep_awc_input
set PROJECT_OPTIONS=project_options.toml
set INPUT=..\..\data\mukey__south_manitou.tif
set GPKG=..\..\data\gnatsgo__south_manitou.gpkg
set OUTPUT_DIR=output

%SCRIPT% ^
  --project-options "%PROJECT_OPTIONS%" ^
  --input "%INPUT%" ^
  --gpkg "%GPKG%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --dtype float32 ^
  --fill-nodata ^
  --awc-floor 0.6 ^
  --prefix south_manitou

if errorlevel 1 (
  echo ERROR: AWC prep failed.
  exit /b 1
)

echo SUCCESS: AWC outputs written to "%OUTPUT_DIR%".
endlocal
