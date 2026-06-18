# SPDX-License-Identifier: CC0-1.0
"""
Tests for swb2_prep/cli/define_swb_grid.py.

Covers all user-facing CLI options:
- AOI via --polygon or --bbox
- CRS via --epsg or --proj4
- --resolution, --snap (outward/inward)
- --output-dir, --output-shapefile, --project-options
- Error cases: file exists, missing CRS, missing AOI, ambiguous polygon selection
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Tuple

import pytest

rasterio = pytest.importorskip("rasterio")
geopandas = pytest.importorskip("geopandas")


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def data_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / "data"
    if not (candidate / "aoi.shp").exists():
        pytest.skip("Missing data/aoi.shp")
    return candidate


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------- Helpers ----------

def _run(args: list[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    cmd = [sys.executable, "-m", "swb2_prep.cli.define_swb_grid", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None)
    return proc.returncode, proc.stdout, proc.stderr


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


# ---------- Success cases ----------

class TestPolygonAOI:
    """Tests using --polygon as the AOI source."""

    def test_polygon_with_epsg(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Provide an AOI polygon shapefile and CRS via --epsg.

        Purpose: Verify the primary workflow — polygon AOI + EPSG CRS — produces
        a valid project_options.toml with all required [grid] fields and a
        matching template GeoTIFF.

        Expected outcome:
        - Exit code 0.
        - TOML contains [grid] with crs='EPSG:5070', resolution=30.0, positive
          nx/ny, snap='outward' (default), source='aoi_polygon'.
        - All extent values are native Python floats (not numpy scalar strings).
        - Template GeoTIFF exists with width==nx, height==ny, CRS==EPSG:5070.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"

        opts = _load_toml(toml_path)
        assert "grid" in opts
        grid = opts["grid"]
        assert grid["crs"] == "EPSG:5070"
        assert grid["resolution"] == 30.0
        assert grid["nx"] > 0 and grid["ny"] > 0
        assert grid["snap"] == "outward"
        assert grid["source"] == "aoi_polygon"

        # All extents are native floats (not strings like "np.float64(...)")
        for key in ("xmin", "ymin", "xmax", "ymax", "xmin_raw", "ymin_raw", "xmax_raw", "ymax_raw"):
            assert isinstance(grid[key], float), f"{key} is not float: {type(grid[key])}"

        # Template GeoTIFF matches grid dimensions and CRS
        template = Path(grid["template_tif"])
        if not template.is_absolute():
            template = tmp_path / template
        assert template.exists()
        with rasterio.open(template) as ds:
            assert ds.width == grid["nx"]
            assert ds.height == grid["ny"]
            assert str(ds.crs) == "EPSG:5070"

    def test_polygon_with_proj4(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Provide CRS via --proj4 instead of --epsg.

        Purpose: Verify that --proj4 is accepted as an alternative CRS
        specification and produces a valid grid definition.

        Expected outcome:
        - Exit code 0.
        - TOML [grid] contains a 'proj4' field and correct resolution.
        """
        toml_path = tmp_path / "project_options.toml"
        proj4 = "+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--proj4", proj4,
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        opts = _load_toml(toml_path)
        assert "proj4" in opts["grid"]
        assert opts["grid"]["resolution"] == 30.0

    def test_snap_inward(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Use --snap inward to contract extents toward the grid center.

        Purpose: Verify that 'inward' snapping produces extents that are
        contracted relative to the raw AOI bounds (ceil for min, floor for max).

        Expected outcome:
        - Exit code 0.
        - TOML [grid].snap == 'inward'.
        - Snapped xmin >= raw xmin, snapped ymin >= raw ymin.
        - Snapped xmax <= raw xmax, snapped ymax <= raw ymax.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--snap", "inward",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        grid = _load_toml(toml_path)["grid"]
        assert grid["snap"] == "inward"
        assert grid["xmin"] >= grid["xmin_raw"]
        assert grid["ymin"] >= grid["ymin_raw"]
        assert grid["xmax"] <= grid["xmax_raw"]
        assert grid["ymax"] <= grid["ymax_raw"]

    def test_output_shapefile(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Use --output-shapefile to write the grid bounding box as a shapefile.

        Purpose: Verify that the optional shapefile output is created and
        contains a single polygon with a valid CRS.

        Expected outcome:
        - Exit code 0.
        - Shapefile exists at the specified path within --output-dir.
        - Shapefile contains exactly 1 polygon feature with a non-null CRS.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
            "--output-shapefile", "grid_extent.shp",
        ])
        assert rc == 0, f"Failed: {err}"
        shp_path = out_dir / "grid_extent.shp"
        assert shp_path.exists()
        gdf = geopandas.read_file(shp_path)
        assert len(gdf) == 1
        assert gdf.crs is not None


class TestBboxAOI:
    """Tests using --bbox as the AOI source."""

    def test_bbox_basic(self, tmp_path: Path, out_dir: Path):
        """Provide AOI as a bounding box already aligned to the resolution.

        Purpose: Verify that --bbox works as an alternative to --polygon,
        and that pre-aligned coordinates produce the expected exact grid
        dimensions (no snapping adjustment needed).

        Expected outcome:
        - Exit code 0.
        - TOML [grid].source == 'bbox'.
        - Grid dimensions are exactly 227 x 227 (matching the known
          South Manitou grid: (779790-772980)/30 = 227).
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--bbox", "772980", "2484990", "779790", "2491800",
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        grid = _load_toml(toml_path)["grid"]
        assert grid["source"] == "bbox"
        assert grid["nx"] == 227
        assert grid["ny"] == 227

    def test_bbox_snap_outward_expands(self, tmp_path: Path, out_dir: Path):
        """Provide a bbox with non-aligned coordinates and outward snapping.

        Purpose: Verify that outward snapping expands the extent so that
        the snapped bounds fully contain the raw bounds, and that the
        resulting dimensions are exact integer multiples of the resolution.

        Expected outcome:
        - Exit code 0.
        - Snapped xmin <= raw xmin (100.5), snapped ymin <= raw ymin (200.5).
        - Snapped xmax >= raw xmax (400.7), snapped ymax >= raw ymax (600.3).
        - (xmax - xmin) / resolution == nx exactly (no fractional pixels).
        - (ymax - ymin) / resolution == ny exactly.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--bbox", "100.5", "200.5", "400.7", "600.3",
            "--epsg", "EPSG:5070",
            "--resolution", "10",
            "--snap", "outward",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        grid = _load_toml(toml_path)["grid"]
        assert grid["xmin"] <= 100.5
        assert grid["ymin"] <= 200.5
        assert grid["xmax"] >= 400.7
        assert grid["ymax"] >= 600.3
        # Dimensions must be exact integers (no fractional pixels)
        assert (grid["xmax"] - grid["xmin"]) / grid["resolution"] == grid["nx"]
        assert (grid["ymax"] - grid["ymin"]) / grid["resolution"] == grid["ny"]


class TestResolution:
    """Tests for different resolution values."""

    def test_non_integer_resolution(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Use a coarser resolution (100m) than the default test case (30m).

        Purpose: Verify that non-default resolution values are accepted and
        produce a valid (smaller) grid with positive dimensions.

        Expected outcome:
        - Exit code 0.
        - TOML [grid].resolution == 100.0.
        - nx > 0 and ny > 0 (grid is not degenerate).
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "100",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        grid = _load_toml(toml_path)["grid"]
        assert grid["resolution"] == 100.0
        assert grid["nx"] > 0 and grid["ny"] > 0


class TestInputDir:
    """Tests for --input-dir argument and absolute path storage."""

    def test_paths_stored_as_absolute(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Verify that input_dir, output_dir, and template_tif are absolute paths.

        Purpose: Ensure the TOML is self-contained and can be read from any
        working directory without path resolution issues.

        Expected outcome:
        - Exit code 0.
        - [paths].input_dir is an absolute path.
        - [paths].output_dir is an absolute path.
        - [grid].template_tif is an absolute path.
        """
        toml_path = tmp_path / "project_options.toml"
        input_dir = tmp_path / "my_data"
        input_dir.mkdir()

        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--input-dir", input_dir.as_posix(),
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        opts = _load_toml(toml_path)
        assert Path(opts["paths"]["input_dir"]).is_absolute()
        assert Path(opts["paths"]["output_dir"]).is_absolute()
        assert Path(opts["grid"]["template_tif"]).is_absolute()

    def test_input_dir_value(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Verify --input-dir value is stored correctly.

        Expected outcome:
        - [paths].input_dir resolves to the directory passed via --input-dir.
        """
        toml_path = tmp_path / "project_options.toml"
        input_dir = tmp_path / "custom_input"
        input_dir.mkdir()

        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--input-dir", input_dir.as_posix(),
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        opts = _load_toml(toml_path)
        assert Path(opts["paths"]["input_dir"]) == input_dir.resolve()


# ---------- Error cases ----------

class TestErrors:
    """Tests for expected error conditions."""

    def test_merge_preserves_existing_sections(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Run swb2-prep-grid when project_options.toml already has other sections.

        Purpose: Verify merge semantics — existing sections not owned by
        define_swb_grid are preserved after the run.

        Expected outcome:
        - Exit code 0.
        - Pre-existing [landuse] section is preserved in the output file.
        - [grid] section is written with correct values.
        """
        toml_path = tmp_path / "project_options.toml"
        toml_path.write_text('[landuse]\ncode = 42\nlabel = "forest"\n')

        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"Failed: {err}"
        opts = _load_toml(toml_path)
        assert opts["landuse"] == {"code": 42, "label": "forest"}
        assert "grid" in opts
        assert opts["grid"]["resolution"] == 30.0

    def test_merge_no_conflict_same_grid(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Run swb2-prep-grid twice with identical parameters.

        Purpose: Verify that re-running with the same grid parameters succeeds
        (no conflict when values are identical).

        Expected outcome:
        - Both runs exit 0.
        - TOML is valid after second run.
        """
        toml_path = tmp_path / "project_options.toml"
        common_args = [
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ]
        rc1, _, err1 = _run(common_args)
        assert rc1 == 0, f"First run failed: {err1}"

        rc2, _, err2 = _run(common_args)
        assert rc2 == 0, f"Second run failed: {err2}"
        opts = _load_toml(toml_path)
        assert opts["grid"]["nx"] > 0

    def test_merge_conflict_different_resolution(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Run swb2-prep-grid when existing [grid] has a conflicting resolution.

        Purpose: Verify conflict detection — the CLI must error if existing
        [grid] values contradict the current computation.

        Expected outcome:
        - Non-zero exit code.
        - stderr contains 'conflict'.
        """
        toml_path = tmp_path / "project_options.toml"
        # First run at 30m
        rc, _, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc == 0, f"First run failed: {err}"

        # Second run at 100m — should conflict
        rc, _, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "100",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "conflict" in err.lower()

    def test_both_epsg_and_proj4(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Provide both --epsg and --proj4 simultaneously.

        Purpose: Verify that the CLI rejects ambiguous CRS specification.
        Exactly one of --epsg or --proj4 must be provided.

        Expected outcome:
        - Non-zero exit code.
        - stderr indicates the conflict.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--proj4", "+proj=aea +lat_0=23 +lon_0=-96",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "not both" in err.lower() or "epsg" in err.lower()

    def test_no_crs(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Omit both --epsg and --proj4.

        Purpose: Verify that the CLI requires a CRS to be specified and
        produces a clear error when neither option is given.

        Expected outcome:
        - Non-zero exit code.
        - stderr mentions CRS, --epsg, or --proj4.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "epsg" in err.lower() or "proj4" in err.lower() or "crs" in err.lower()

    def test_no_aoi(self, tmp_path: Path, out_dir: Path):
        """Omit both --polygon and --bbox.

        Purpose: Verify that the CLI requires an AOI source and produces
        a clear error when neither is provided.

        Expected outcome:
        - Non-zero exit code.
        - stderr mentions --polygon or --bbox.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "polygon" in err.lower() or "bbox" in err.lower()

    def test_polygon_name_without_value(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Provide --polygon-name without --polygon-value.

        Purpose: Verify that partial polygon selection arguments are rejected.
        Both --polygon-name and --polygon-value must be provided together.

        Expected outcome:
        - Non-zero exit code.
        - stderr indicates that both arguments are required.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--polygon-name", "NAME",
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "polygon-name" in err.lower() or "polygon-value" in err.lower() or "both" in err.lower()

    def test_polygon_value_without_name(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Provide --polygon-value without --polygon-name.

        Purpose: Same as above but reversed — verify the check works
        regardless of which argument is missing.

        Expected outcome:
        - Non-zero exit code.
        - stderr indicates that both arguments are required.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--polygon-value", "something",
            "--epsg", "EPSG:5070",
            "--resolution", "30",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "polygon-name" in err.lower() or "polygon-value" in err.lower() or "both" in err.lower()

    def test_negative_resolution(self, data_dir: Path, tmp_path: Path, out_dir: Path):
        """Provide a negative resolution value.

        Purpose: Verify that non-positive resolution is rejected with a
        clear error message.

        Expected outcome:
        - Non-zero exit code.
        - stderr mentions 'resolution' or 'positive'.
        """
        toml_path = tmp_path / "project_options.toml"
        rc, out, err = _run([
            "--polygon", (data_dir / "aoi.shp").as_posix(),
            "--epsg", "EPSG:5070",
            "--resolution", "-10",
            "--project-options", toml_path.as_posix(),
            "--output-dir", out_dir.as_posix(),
        ])
        assert rc != 0
        assert "resolution" in err.lower() or "positive" in err.lower()
