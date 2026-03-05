# SPDX-License-Identifier: CC0-1.0
"""
Tests for the swb2_prep/cli/prep_landuse_input.py tool.

Validates end-to-end landuse preprocessing with a polygon AOI
using the project's 'data' directory and the current CLI contract:
- CLI requires 'project_options.toml' in its process CWD.
- CLI accepts explicit flags: --input, --output-dir, --polygon
- Outputs: GeoTIFF + ArcASCII
- Basic raster sanity checks (shape, dtype) and AOI alignment
- AOI rule errors when only --polygon-name OR only --polygon-value is provided

Directory expectation:
  <project_root>/
    data/                     <-- contains aoi.shp and cdl__south_manitou.tif
    swb2_prep/
    tests/

These tests purposely do NOT depend on any environment variables.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import pytest

# Skip the test file if core geo deps are missing
rasterio = pytest.importorskip("rasterio")
fiona = pytest.importorskip("fiona")
geopandas = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import box as shp_box


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def data_dir() -> Path:
    """
    Always use the project's local 'data' directory, which must contain
    'aoi.shp' and 'cdl__south_manitou.tif'.

    Skip tests only if those files are missing.
    """
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / "data"

    aoi = candidate / "aoi.shp"
    cdl = candidate / "cdl__south_manitou.tif"

    if not aoi.exists() or not cdl.exists():
        pytest.skip(
            f"Missing required test inputs in {candidate}: "
            "expected 'aoi.shp' and 'cdl__south_manitou.tif'."
        )

    return candidate


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    """
    Temporary output directory for each test.
    """
    p = tmp_path / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def tmp_work_dir(tmp_path: Path, data_dir: Path, tmp_out: Path) -> Path:
    """
    A temporary working directory used as the subprocess CWD.
    Writes a minimal 'project_options.toml' that the CLI expects to
    find in its current working directory.
    """
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    toml_text = (
        "[project]\n"
        'crs = "EPSG:5070"\n'
        "resolution = 30.0\n"
        'units = "m"\n'
        "\n"
        "[paths]\n"
        f'input_dir = "{data_dir.as_posix()}"\n'
        f'output_dir = "{tmp_out.as_posix()}"\n'
        "\n"
        "[aoi]\n"
        f'polygon = "{(data_dir / "aoi.shp").as_posix()}"\n'
        "\n"
        "[landuse]\n"
        f'raster = "{(data_dir / "cdl__south_manitou.tif").as_posix()}"\n'
    )
    (work / "project_options.toml").write_text(toml_text, encoding="utf-8")
    return work


# ---------- Helpers ----------

def _run_cli_with_args(args: Iterable[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    """
    Run the CLI via 'python -m', optionally setting the subprocess CWD.

    Parameters
    ----------
    args : Iterable[str]
        CLI arguments to pass after the module name.
    cwd : Path | None
        Working directory for the subprocess. If provided, the CLI
        will look for 'project_options.toml' here.

    Returns
    -------
    (rc, stdout, stderr) : Tuple[int, str, str]
        The process return code and captured text streams.
    """
    module = "swb2_prep.cli.prep_landuse_input"
    cmd = [sys.executable, "-m", module, *list(args)]
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True,
        cwd=str(cwd) if cwd else None
    )
    return proc.returncode, proc.stdout, proc.stderr


def _find_outputs(out_dir: Path) -> Tuple[Path, Path]:
    """
    Find landuse GeoTIFF and ArcASCII outputs without over-constraining
    filename conventions. If you later standardize names, tighten this.

    Returns
    -------
    (tif_path, asc_path)
    """
    tif_list = list(out_dir.glob("*.tif"))
    asc_list = list(out_dir.glob("*.asc"))
    assert len(tif_list) == 1, f"Expected 1 GeoTIFF, found {len(tif_list)}"
    assert len(asc_list) == 1, f"Expected 1 ArcASCII, found {len(asc_list)}"
    return tif_list[0], asc_list[0]


# ---------- Tests ----------

def test_landuse_pipeline_polygon(tmp_out: Path, data_dir: Path, tmp_work_dir: Path):
    """
    End-to-end landuse preprocessing with a polygon AOI using direct CLI flags,
    while ensuring the CLI finds 'project_options.toml' in its CWD.

    Validates:
      - CLI completes successfully
      - Outputs exist: GeoTIFF + ArcASCII
      - Non-zero raster dimensions
      - Integer dtype (categorical landuse)
      - Output extent aligned with AOI polygon (contains/intersects)
    """
    args = [
        "--input", (data_dir / "cdl__south_manitou.tif").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--polygon", (data_dir / "aoi.shp").as_posix(),
    ]

    rc, out, err = _run_cli_with_args(args, cwd=tmp_work_dir)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_path, asc_path = _find_outputs(tmp_out)
    assert tif_path.exists(), "GeoTIFF output missing"
    assert asc_path.exists(), "ArcASCII output missing"

    # Inspect GeoTIFF
    with rasterio.open(tif_path) as ds:
        # Non-zero raster dimensions
        assert ds.width > 0 and ds.height > 0, "Empty raster"

        # Integer dtype (categorical landuse)
        assert ds.dtypes[0].startswith(("int", "uint")), f"dtype not integer: {ds.dtypes[0]}"

        # Clip correctness: bounds should align with AOI polygon
        bounds = ds.bounds
        landuse_poly = shp_box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        aoi_gdf = geopandas.read_file(data_dir / "aoi.shp").to_crs(ds.crs)
        aoi_union = aoi_gdf.geometry.union_all()

        assert aoi_union.contains(landuse_poly) or aoi_union.intersects(landuse_poly), \
            "Output raster extent does not align with AOI"

    # ArcASCII header sanity
    header_lines = []
    with asc_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            header_lines.append(line.strip().lower())
            if i >= 5:
                break
    header = "\n".join(header_lines)
    assert "ncols" in header and "nrows" in header and "cellsize" in header, \
        "ArcASCII header missing required keys"


def test_aoi_ambiguity_error_only_name(tmp_work_dir: Path, data_dir: Path, tmp_out: Path):
    """
    Supplying only --polygon-name (without --polygon-value) must raise an
    AOI selection error per project rules.

    Behavior:
      - Nonzero return code
      - stderr contains an error message
    """
    args = [
        "--input", (data_dir / "cdl__south_manitou.tif").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--polygon", (data_dir / "aoi.shp").as_posix(),
        "--polygon-name", "BASIN_ID",
        # Intentionally omit --polygon-value
    ]

    rc, out, err = _run_cli_with_args(args, cwd=tmp_work_dir)
    assert rc != 0, "CLI should fail when only --polygon-name is provided"
    assert err.strip(), "Expected error message on stderr"


def test_aoi_ambiguity_error_only_value(tmp_work_dir: Path, data_dir: Path, tmp_out: Path):
    """
    Supplying only --polygon-value (without --polygon-name) must raise an
    AOI selection error per project rules.

    Behavior:
      - Nonzero return code
      - stderr contains an error message
    """
    args = [
        "--input", (data_dir / "cdl__south_manitou.tif").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--polygon", (data_dir / "aoi.shp").as_posix(),
        "--polygon-value", "12345",
        # Intentionally omit --polygon-name
    ]

    rc, out, err = _run_cli_with_args(args, cwd=tmp_work_dir)
    assert rc != 0, "CLI should fail when only --polygon-value is provided"
    assert err.strip(), "Expected error message on stderr"