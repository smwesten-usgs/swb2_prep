@echo off
REM Prepare HSG raster aligned to the canonical grid from project_options.toml.
setlocal

set SCRIPT=python -m swb2_prep.cli.prep_hsg_input
set PROJECT_OPTIONS=project_options.toml
set INPUT=..\..\data\mukey__south_manitou.tif
set GPKG=..\..\data\gnatsgo__south_manitou.gpkg
set OUTPUT_DIR=output

%SCRIPT% ^
  --project-options "%PROJECT_OPTIONS%" ^
  --input "%INPUT%" ^
  --gpkg "%GPKG%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --dtype int16 ^
  --nodata -1 ^
  --prefix south_manitou

if errorlevel 1 (
  echo ERROR: HSG prep failed.
  exit /b 1
)

echo SUCCESS: HSG outputs written to "%OUTPUT_DIR%".
endlocal
