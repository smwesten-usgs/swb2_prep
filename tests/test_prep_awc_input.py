# SPDX-License-Identifier: CC0-1.0
"""
Tests for the swb2_prep/cli/prep_awc_input.py tool.

Validates end-to-end AWC preprocessing using the template-driven workflow:
- CLI requires 'project_options.toml' with a [grid] section containing template_tif.
- CLI accepts: --input, --gpkg, --output-dir, --project-options, --prefix, --dtype,
  --nodata, --compress, --resampling
- Outputs: GeoTIFF + ArcASCII
- Raster sanity checks (shape, dtype, CRS, alignment to template, valid pixel values)

Directory expectation:
  <project_root>/
    data/                     <-- contains mukey__south_manitou.tif, gnatsgo__south_manitou.gpkg
    examples/south_manitou/   <-- contains project_options.toml and output/swb_grid_template.tif
    swb2_prep/
    tests/
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Resolve the project root directory (parent of tests/)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Locate the data directory; skip if required AWC inputs are missing."""
    candidate = project_root / "data"
    mukey = candidate / "mukey__south_manitou.tif"
    gpkg = candidate / "gnatsgo__south_manitou.gpkg"
    if not mukey.exists() or not gpkg.exists():
        pytest.skip(
            f"Missing AWC test inputs in {candidate} "
            "(need mukey__south_manitou.tif and gnatsgo__south_manitou.gpkg)"
        )
    return candidate


@pytest.fixture(scope="session")
def example_dir(project_root: Path) -> Path:
    """Locate the example directory; skip if project_options.toml or template are missing."""
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
    """Create a temporary output directory for each test."""
    p = tmp_path / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Helpers ----------

def _run_cli(args: Iterable[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    """Run the prep_awc_input CLI as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "swb2_prep.cli.prep_awc_input", "--no-log", *list(args)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None)
    return proc.returncode, proc.stdout, proc.stderr


def _base_args(data_dir: Path, example_dir: Path, tmp_out: Path) -> list[str]:
    """Return the minimal required CLI arguments."""
    return [
        "--input", (data_dir / "mukey__south_manitou.tif").as_posix(),
        "--gpkg", (data_dir / "gnatsgo__south_manitou.gpkg").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
    ]


# ---------- Tests ----------

def test_awc_pipeline_default_options(tmp_out: Path, data_dir: Path, example_dir: Path):
    """End-to-end AWC preprocessing with default CLI options.

    Summary:
        Runs the CLI with only the required arguments (--input, --gpkg,
        --project-options, --output-dir), relying on defaults for dtype (float32),
        nodata (-1.0), compression (lzw), and resampling (bilinear).

    Expected outcome:
        - CLI exits with return code 0.
        - One GeoTIFF and one Arc ASCII file are produced.
        - Output GeoTIFF has dtype float32.
    """
    args = _base_args(data_dir, example_dir, tmp_out)

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    asc_list = list(tmp_out.glob("*.asc"))
    assert len(tif_list) == 1, f"Expected 1 GeoTIFF, found {len(tif_list)}"
    assert len(asc_list) == 1, f"Expected 1 ArcASCII, found {len(asc_list)}"

    with rasterio.open(tif_list[0]) as ds:
        assert ds.dtypes[0] == "float32", f"Expected float32, got {ds.dtypes[0]}"


def test_awc_pipeline_with_prefix(tmp_out: Path, data_dir: Path, example_dir: Path):
    """AWC preprocessing with a filename prefix.

    Summary:
        Adds --prefix south_manitou to verify the prefix appears in output filenames.

    Expected outcome:
        - CLI exits with return code 0.
        - Output filenames contain 'south_manitou'.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--prefix", "south_manitou"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    asc_list = list(tmp_out.glob("*.asc"))
    assert len(tif_list) == 1
    assert "south_manitou" in tif_list[0].name
    assert "south_manitou" in asc_list[0].name


def test_awc_pipeline_custom_nodata(tmp_out: Path, data_dir: Path, example_dir: Path):
    """AWC preprocessing with custom nodata value.

    Summary:
        Runs with --nodata -9999.0 to verify non-default nodata is honored.

    Expected outcome:
        - CLI exits with return code 0.
        - Output GeoTIFF has nodata value of -9999.0.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--nodata", "-9999.0"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    with rasterio.open(tif_list[0]) as ds:
        assert ds.nodata == -9999.0, f"Expected nodata=-9999.0, got {ds.nodata}"


def test_awc_pipeline_float64_dtype(tmp_out: Path, data_dir: Path, example_dir: Path):
    """AWC preprocessing with float64 dtype.

    Summary:
        Runs with --dtype float64 to verify the tool handles double-precision output.

    Expected outcome:
        - CLI exits with return code 0.
        - Output GeoTIFF has dtype float64.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--dtype", "float64"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    with rasterio.open(tif_list[0]) as ds:
        assert ds.dtypes[0] == "float64", f"Expected float64, got {ds.dtypes[0]}"


def test_awc_pipeline_nearest_resampling(tmp_out: Path, data_dir: Path, example_dir: Path):
    """AWC preprocessing with nearest-neighbor resampling.

    Summary:
        Runs with --resampling nearest to verify the tool accepts alternate
        resampling methods.

    Expected outcome:
        - CLI exits with return code 0.
        - A valid GeoTIFF is produced.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--resampling", "nearest"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    assert len(tif_list) == 1
    with rasterio.open(tif_list[0]) as ds:
        assert ds.width > 0 and ds.height > 0


def test_awc_pipeline_compress_deflate(tmp_out: Path, data_dir: Path, example_dir: Path):
    """AWC preprocessing with deflate compression.

    Summary:
        Runs with --compress deflate to verify the tool accepts alternate
        compression options without error.

    Expected outcome:
        - CLI exits with return code 0.
        - A valid GeoTIFF is produced.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--compress", "deflate"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    assert len(tif_list) == 1


def test_awc_pixel_values_valid(tmp_out: Path, data_dir: Path, example_dir: Path):
    """Verify all AWC pixel values are non-negative or nodata.

    Summary:
        Runs the default pipeline and reads the output GeoTIFF. Checks that
        all non-nodata pixel values are >= 0 (AWC cannot be negative).

    Expected outcome:
        - No pixel values less than 0 except the nodata sentinel (-1.0).
    """
    args = _base_args(data_dir, example_dir, tmp_out)

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    with rasterio.open(tif_list[0]) as ds:
        data = ds.read(1)
        nodata = ds.nodata

    # Mask out nodata pixels
    valid = data[data != nodata]
    assert np.all(valid >= 0.0), f"Found negative AWC values: min={valid.min()}"


def test_awc_output_aligned_to_template(tmp_out: Path, data_dir: Path, example_dir: Path):
    """Verify output raster is pixel-aligned to the grid template.

    Summary:
        Compares the output GeoTIFF dimensions, CRS, and transform against the
        canonical grid template to confirm exact alignment.

    Expected outcome:
        - Width, height, CRS, and affine transform all match the template.
    """
    args = _base_args(data_dir, example_dir, tmp_out)

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    template_path = example_dir / "output" / "swb_grid_template.tif"
    tif_list = list(tmp_out.glob("*.tif"))

    with rasterio.open(template_path) as tmpl, rasterio.open(tif_list[0]) as out_ds:
        assert out_ds.width == tmpl.width, "Width mismatch with template"
        assert out_ds.height == tmpl.height, "Height mismatch with template"
        assert out_ds.crs == tmpl.crs, "CRS mismatch with template"
        assert out_ds.transform == tmpl.transform, "Transform mismatch with template"


def test_missing_project_options_errors(tmp_out: Path, data_dir: Path, tmp_path: Path):
    """CLI errors when project_options.toml does not exist.

    Summary:
        Points --project-options at a nonexistent file path.

    Expected outcome:
        - CLI exits with a non-zero return code.
    """
    fake_toml = tmp_path / "nonexistent.toml"
    args = [
        "--input", (data_dir / "mukey__south_manitou.tif").as_posix(),
        "--gpkg", (data_dir / "gnatsgo__south_manitou.gpkg").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", fake_toml.as_posix(),
    ]

    rc, out, err = _run_cli(args)
    assert rc != 0, "CLI should fail when project_options.toml is missing"


def test_missing_input_raster_errors(tmp_out: Path, data_dir: Path, example_dir: Path, tmp_path: Path):
    """CLI errors when input MUKEY raster does not exist.

    Summary:
        Points --input at a nonexistent .tif file path.

    Expected outcome:
        - CLI exits with a non-zero return code.
    """
    args = [
        "--input", (tmp_path / "nonexistent.tif").as_posix(),
        "--gpkg", (data_dir / "gnatsgo__south_manitou.gpkg").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
    ]

    rc, out, err = _run_cli(args)
    assert rc != 0, "CLI should fail when input raster is missing"


def test_missing_gpkg_errors(tmp_out: Path, data_dir: Path, example_dir: Path, tmp_path: Path):
    """CLI errors when GeoPackage does not exist.

    Summary:
        Points --gpkg at a nonexistent .gpkg file path.

    Expected outcome:
        - CLI exits with a non-zero return code.
    """
    args = [
        "--input", (data_dir / "mukey__south_manitou.tif").as_posix(),
        "--gpkg", (tmp_path / "nonexistent.gpkg").as_posix(),
        "--output-dir", tmp_out.as_posix(),
        "--project-options", (example_dir / "project_options.toml").as_posix(),
    ]

    rc, out, err = _run_cli(args)
    assert rc != 0, "CLI should fail when GeoPackage is missing"
