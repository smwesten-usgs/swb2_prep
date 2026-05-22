# SPDX-License-Identifier: CC0-1.0
"""
Tests for the swb2_prep/cli/prep_d8_flowdir.py tool.

Validates end-to-end D8 flow direction computation:
- CLI requires 'project_options.toml' with a [grid] section containing template_tif.
- CLI accepts: --input, --output-dir, --project-options, --prefix, --dtype,
  --nodata, --compress, --resampling
- Outputs: GeoTIFF + ArcASCII
- Raster sanity checks (shape, dtype, CRS, alignment to template, valid pixel values)
- Comparison with expected result (threshold-based, not exact match)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

# Valid D8 direction codes (SWB2 encoding)
VALID_D8_CODES = {1, 2, 4, 8, 16, 32, 64, 128}


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Resolve the project root directory (parent of tests/)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Locate the data directory; skip if required DEM input is missing."""
    candidate = project_root / "data"
    dem = candidate / "hydrosheds_dem__south_manitou.tif"
    if not dem.exists():
        pytest.skip(f"Missing DEM test input: {dem}")
    return candidate


@pytest.fixture(scope="session")
def example_dir(project_root: Path) -> Path:
    """Locate the example directory with project_options.toml and template."""
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
def out_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    d = tmp_path / "out"
    d.mkdir()
    return d


# ---------- Helpers ----------

def _run(args: list[str]) -> Tuple[int, str, str]:
    """Run the D8 CLI as a subprocess."""
    cmd = [sys.executable, "-m", "swb2_prep.cli.prep_d8_flowdir", *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ---------- End-to-end pipeline tests ----------

def test_d8_pipeline_default_options(data_dir: Path, example_dir: Path, out_dir: Path):
    """Run D8 CLI with default options and verify output exists with correct properties.

    Expected outcome:
    - Exit code 0.
    - GeoTIFF output exists with shape 227x227, dtype int16, CRS EPSG:5070.
    - Arc ASCII output exists.
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif = out_dir / "d8_flowdir__30m.tif"
    asc = out_dir / "d8_flowdir__30m.asc"
    assert tif.exists()
    assert asc.exists()

    with rasterio.open(tif) as ds:
        assert ds.width == 227
        assert ds.height == 227
        assert ds.dtypes[0] == "int16"
        assert str(ds.crs) == "EPSG:5070"
        assert ds.nodata == -9999


def test_d8_pipeline_with_prefix(data_dir: Path, example_dir: Path, out_dir: Path):
    """Run D8 CLI with --prefix and verify filename includes prefix.

    Expected outcome:
    - Exit code 0.
    - Output filename is 'south_manitou__d8_flowdir__30m.tif'.
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--prefix", "south_manitou",
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"
    assert (out_dir / "south_manitou__d8_flowdir__30m.tif").exists()
    assert (out_dir / "south_manitou__d8_flowdir__30m.asc").exists()


def test_d8_pipeline_cubic_resampling(data_dir: Path, example_dir: Path, out_dir: Path):
    """Run D8 CLI with --resampling cubic and verify it succeeds.

    Expected outcome:
    - Exit code 0.
    - Output GeoTIFF exists with correct shape.
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--resampling", "cubic",
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif = out_dir / "d8_flowdir__30m.tif"
    with rasterio.open(tif) as ds:
        assert ds.width == 227
        assert ds.height == 227


# ---------- Output value validation ----------

def test_d8_pixel_values_valid(data_dir: Path, example_dir: Path, out_dir: Path):
    """Verify all pixel values are valid D8 codes or nodata.

    Expected outcome:
    - Every pixel is one of {1, 2, 4, 8, 16, 32, 64, 128, -9999}.
    - No pysheds artifact values (0, 254, 255) are present.
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    with rasterio.open(out_dir / "d8_flowdir__30m.tif") as ds:
        arr = ds.read(1)

    unique_values = set(np.unique(arr))
    allowed = VALID_D8_CODES | {-9999}
    unexpected = unique_values - allowed
    assert not unexpected, f"Unexpected pixel values in D8 output: {unexpected}"


def test_d8_has_nodata_pixels(data_dir: Path, example_dir: Path, out_dir: Path):
    """Verify that nodata pixels exist (edges/water areas should be masked).

    Expected outcome:
    - At least some pixels have the nodata value (-9999).
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    with rasterio.open(out_dir / "d8_flowdir__30m.tif") as ds:
        arr = ds.read(1)

    nodata_count = np.sum(arr == -9999)
    assert nodata_count > 0, "Expected some nodata pixels in D8 output (edges/water)"


# ---------- Alignment to template ----------

def test_d8_output_aligned_to_template(data_dir: Path, example_dir: Path, out_dir: Path):
    """Verify output grid is pixel-aligned to the template GeoTIFF.

    Expected outcome:
    - Output shape, transform, and CRS match the template exactly.
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    template_path = example_dir / "output" / "swb_grid_template.tif"
    with rasterio.open(template_path) as tmpl:
        tmpl_width = tmpl.width
        tmpl_height = tmpl.height
        tmpl_transform = tmpl.transform
        tmpl_crs = str(tmpl.crs)

    with rasterio.open(out_dir / "d8_flowdir__30m.tif") as ds:
        assert ds.width == tmpl_width
        assert ds.height == tmpl_height
        assert str(ds.crs) == tmpl_crs
        # Compare transform components
        for i in range(6):
            assert np.isclose(ds.transform[i], tmpl_transform[i], atol=1e-6), \
                f"Transform mismatch at index {i}: {ds.transform[i]} vs {tmpl_transform[i]}"


# ---------- Comparison with expected result ----------

def test_d8_agreement_with_expected(data_dir: Path, example_dir: Path, out_dir: Path):
    """Compare output with expected result; require >= 85% pixel agreement.

    The expected file was produced with bicubic resampling (QGIS) and TauDEM.
    Our pipeline uses bilinear resampling and pysheds with resolve_flats
    (default). Differences arise from interpolation and algorithm details.
    85% is a conservative threshold.

    Expected outcome:
    - Pixel agreement >= 85%.
    """
    expected_path = data_dir / "expected_results" / "d8_flow_direction__south_manitou.tif"
    if not expected_path.exists():
        pytest.skip(f"Expected result not found: {expected_path}")

    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", out_dir.as_posix(),
        "--no-log",
    ])
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    with rasterio.open(out_dir / "d8_flowdir__30m.tif") as ds:
        our = ds.read(1).astype(np.int32)

    with rasterio.open(expected_path) as ds:
        exp = ds.read(1).astype(np.int32)

    assert our.shape == exp.shape, "Shape mismatch with expected result"

    match_count = np.sum(our == exp)
    total = our.size
    agreement_pct = 100.0 * match_count / total
    assert agreement_pct >= 85.0, \
        f"Pixel agreement {agreement_pct:.1f}% is below 85% threshold"


# ---------- Error cases ----------

def test_missing_project_options_errors(tmp_path: Path, data_dir: Path):
    """Provide a non-existent project_options.toml path.

    Expected outcome:
    - Non-zero exit code.
    """
    rc, out, err = _run([
        "--input", (data_dir / "hydrosheds_dem__south_manitou.tif").as_posix(),
        "--project-options", (tmp_path / "nonexistent.toml").as_posix(),
        "--output-dir", tmp_path.as_posix(),
        "--no-log",
    ])
    assert rc != 0


def test_missing_input_raster_errors(example_dir: Path, tmp_path: Path):
    """Provide a non-existent input DEM path.

    Expected outcome:
    - Non-zero exit code.
    """
    rc, out, err = _run([
        "--input", (tmp_path / "nonexistent_dem.tif").as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
        "--output-dir", tmp_path.as_posix(),
        "--no-log",
    ])
    assert rc != 0
