# SPDX-License-Identifier: CC0-1.0
"""
Tests for the swb2_prep/cli/prep_landuse_input.py tool.

Validates end-to-end landuse preprocessing using the template-driven workflow:
- CLI requires 'project_options.toml' with a [grid] section containing template_tif.
- CLI accepts: --input, --output-dir, --project-options
- Outputs: GeoTIFF + ArcASCII
- Raster sanity checks (shape, dtype, CRS, alignment to template)

Directory expectation:
  <project_root>/
    data/                     <-- contains aoi.shp and cdl__south_manitou.tif
    examples/south_manitou/   <-- contains project_options.toml and output/swb_grid_template.tif
    swb2_prep/
    tests/
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import pytest

rasterio = pytest.importorskip("rasterio")
geopandas = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import box as shp_box


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    candidate = project_root / "data"
    aoi = candidate / "aoi.shp"
    cdl = candidate / "cdl__south_manitou.tif"
    if not aoi.exists() or not cdl.exists():
        pytest.skip(f"Missing test inputs in {candidate}")
    return candidate


@pytest.fixture(scope="session")
def example_dir(project_root: Path) -> Path:
    candidate = project_root / "examples" / "south_manitou"
    toml = candidate / "project_options.toml"
    template = candidate / "output" / "swb_grid_template.tif"
    if not toml.exists() or not template.exists():
        pytest.skip(
            f"Missing example files in {candidate}. "
            "Run define_swb_grid first to create project_options.toml and template."
        )
    return candidate


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    p = tmp_path / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Helpers ----------

def _run_cli(args: Iterable[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    cmd = [sys.executable, "-m", "swb2_prep.cli.prep_landuse_input", "--no-log", *list(args)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None)
    return proc.returncode, proc.stdout, proc.stderr


# ---------- Tests ----------

def test_landuse_pipeline_template_driven(tmp_out: Path, data_dir: Path, example_dir: Path):
    """End-to-end landuse preprocessing using the template-driven workflow.

    Validates:
      - CLI completes successfully
      - Outputs exist: GeoTIFF + ArcASCII
      - Output matches template dimensions, CRS, and transform
      - Integer dtype (categorical landuse)
    """
    args = [
        "--input", (data_dir / "cdl__south_manitou.tif").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--prefix", "test",
    ]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    # Find outputs
    tif_list = list(tmp_out.glob("*.tif"))
    asc_list = list(tmp_out.glob("*.asc"))
    assert len(tif_list) == 1, f"Expected 1 GeoTIFF, found {len(tif_list)}"
    assert len(asc_list) == 1, f"Expected 1 ArcASCII, found {len(asc_list)}"

    # Verify GeoTIFF matches template
    template_path = example_dir / "output" / "swb_grid_template.tif"
    with rasterio.open(template_path) as tmpl, rasterio.open(tif_list[0]) as out_ds:
        assert out_ds.width == tmpl.width, "Width mismatch with template"
        assert out_ds.height == tmpl.height, "Height mismatch with template"
        assert out_ds.crs == tmpl.crs, "CRS mismatch with template"
        assert out_ds.transform == tmpl.transform, "Transform mismatch with template"
        assert out_ds.dtypes[0].startswith(("int", "uint")), f"Expected integer dtype, got {out_ds.dtypes[0]}"

    # ArcASCII header sanity
    with asc_list[0].open("r", encoding="utf-8") as f:
        header = "".join(f.readline() for _ in range(6)).lower()
    assert "ncols" in header and "nrows" in header and "cellsize" in header


def test_missing_project_options_errors(tmp_out: Path, data_dir: Path, tmp_path: Path):
    """CLI errors when project_options.toml does not exist."""
    fake_toml = tmp_path / "nonexistent.toml"
    args = [
        "--input", (data_dir / "cdl__south_manitou.tif").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", fake_toml.as_posix(),
    ]

    rc, out, err = _run_cli(args)
    assert rc != 0, "CLI should fail when project_options.toml is missing"


def test_missing_input_raster_errors(tmp_out: Path, example_dir: Path, tmp_path: Path):
    """CLI errors when input raster does not exist."""
    args = [
        "--input", (tmp_path / "nonexistent.tif").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
    ]

    rc, out, err = _run_cli(args)
    assert rc != 0, "CLI should fail when input raster is missing"
