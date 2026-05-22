::@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ========= USER SETTINGS =========
set "SRC_GPKG=gNATSGO_02_03_2025.gpkg"
set "AOI=aoi.shp"
set "MUKEY_RASTER=muraster_30m.tif"

REM Final outputs
set "OUT_GPKG=gnatsgo__south_manitou.gpkg"
set "OUT_TIF=mukey__south_manitou.tif"

REM Tables to copy (MUKEY-keyed)
set "TBL_MUAGGATT=muaggatt"
set "TBL_MAPUNIT=mapunit"   REM comment out the MAPUNIT section if not needed

REM If your raster uses a different NoData, change this:
set "NODATA=0"
REM ================================

REM Working folder
set "WORK=tmp_south_manitou"

rmdir /S /Q "%WORK%"
mkdir "%WORK%"

REM Clean previous outputs to avoid schema conflicts
::if exist "%OUT_GPKG%" del /f /q "%OUT_GPKG%"
if exist "%OUT_TIF%"  del /f /q "%OUT_TIF%"

echo [STEP 1/5] Clip MUKEY raster to AOI (nearest-neighbor, categorical safe)...
gdalwarp -cutline "%AOI%" -crop_to_cutline -r near -dstnodata %NODATA% -multi -wo NUM_THREADS=ALL_CPUS -of GTiff "%MUKEY_RASTER%" "%WORK%\mukey_clip.tif" || goto :fail_warp

echo [STEP 2/5] Polygonize clipped raster to derive MUKEY classes (DN)...
gdal_polygonize "%WORK%\mukey_clip.tif" "%WORK%\mukey_clip_polys.gpkg" -f GPKG mukey_polys || goto :fail_poly

echo [STEP 3/5] Export DISTINCT MUKEYs (DN) to CSV...
ogr2ogr -f CSV "%WORK%\mukeys.csv" "%WORK%\mukey_clip_polys.gpkg" -sql "SELECT DISTINCT DN AS MUKEY FROM mukey_polys WHERE DN IS NOT NULL" || goto :fail_csv

REM Build comma-separated single-quoted MUKEY list for SQL IN(...)
set "MULIST="
for /f "usebackq skip=1 tokens=1 delims=, " %%A in ("%WORK%\mukeys.csv") do (
  if not defined MULIST (set "MULIST='%%~A'") else (set "MULIST=!MULIST!,'%%~A'")
)

if not defined MULIST (
  echo [WARN] No MUKEYs found in AOI. Check CRS/overlap. Proceeding to raster-only packaging.
  goto :raster_only
)

echo [INFO] MUKEYs: !MULIST!

echo [STEP 4/5] Write the clipped raster externally as "%OUT_TIF%"...
copy /y "%WORK%\mukey_clip.tif" "%OUT_TIF%" >nul || echo [WARN] Could not copy clipped raster; it remains at "%WORK%\mukey_clip.tif".

REM ============================
REM 5) Create muaggatt subset by taking rowid as MUKEY from the ORIGINAL GPKG
REM    and filtering by raster MUKEY list in the same SQL. FID=MUKEY.
REM ============================
echo [STEP 5/5] Subsetting muaggatt (CONUS) with rowid -> MUKEY and MUKEY IN (!MULIST!)...
ogr2ogr -f GPKG "%OUT_GPKG%" "%SRC_GPKG%" -dialect SQLITE -overwrite -nln muaggatt -lco FID=MUKEY -sql "SELECT *, rowid AS MUKEY FROM %TBL_MUAGGATT% WHERE rowid IN (!MULIST!)" || goto :fail_muaggatt

REM ============================
REM 5b) OPTIONAL: subset MAPUNIT (keeps original mukey and musym)
REM ============================
echo [STEP 5b] Subsetting mapunit by MUKEY (optional)...
ogr2ogr -f GPKG "%OUT_GPKG%" "%SRC_GPKG%" -dialect SQLITE -overwrite -nln mapunit -sql "SELECT * FROM %TBL_MAPUNIT% WHERE mukey IN (!MULIST!)" || echo [WARN] MAPUNIT not copied (missing or different name).

:raster_only
echo [STEP] Saving raster only (no MUKEYs found; skipping table filters)...
copy /y "%WORK%\mukey_clip.tif" "%OUT_TIF%" >nul
goto :done

:fail_warp
echo [ERROR] gdalwarp failed. Double-check AOI path and raster CRS.
goto :abort

:fail_poly
echo [ERROR] Polygonization failed. Ensure GDAL Python bindings are available:
echo         python -c "from osgeo import gdal; print(gdal.VersionInfo())"
goto :abort

:fail_csv
echo [ERROR] Failed to export MUKEYs CSV from mukey_polys.
goto :abort

:fail_muaggatt
echo [ERROR] Failed to copy muaggatt from "%SRC_GPKG%". Verify table name and MUKEY field.
goto :abort

:abort
echo [ABORT] Script stopped due to errors.
exit /b 1

:done
echo.
echo [DONE] Outputs:
echo   - GeoPackage: "%OUT_GPKG%" (contains MUKEY-filtered muaggatt [+ mapunit, if present])
echo   - Raster:     "%OUT_TIF%"   (clipped MUKEY GeoTIFF kept external)
echo.
echo [TIP] Load both in QGIS: style the raster by MUKEY; use muaggatt for labels/joins.
echo [TIP] If your raster NoData is not %NODATA%, edit the NODATA variable.
echo.
endlocal