# SPDX-License-Identifier: CC0-1.0
"""
Tests for the swb2_prep/cli/prep_hsg_input.py tool.

Validates end-to-end HSG preprocessing using the template-driven workflow:
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
    """Locate the data directory; skip if required HSG inputs are missing."""
    candidate = project_root / "data"
    mukey = candidate / "mukey__south_manitou.tif"
    gpkg = candidate / "gnatsgo__south_manitou.gpkg"
    if not mukey.exists() or not gpkg.exists():
        pytest.skip(
            f"Missing HSG test inputs in {candidate} "
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
    """Run the prep_hsg_input CLI as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "swb2_prep.cli.prep_hsg_input", "--no-log", *list(args)]
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

def test_hsg_pipeline_default_options(tmp_out: Path, data_dir: Path, example_dir: Path):
    """End-to-end HSG preprocessing with default CLI options.

    Summary:
        Runs the CLI with only the required arguments (--input, --gpkg,
        --project-options, --output-dir), relying on defaults for dtype (int16),
        nodata (-1), compression (lzw), and resampling (nearest).

    Expected outcome:
        - CLI exits with return code 0.
        - One GeoTIFF and one Arc ASCII file are produced.
        - Output GeoTIFF has dtype int16.
    """
    args = _base_args(data_dir, example_dir, tmp_out)

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    asc_list = list(tmp_out.glob("*.asc"))
    assert len(tif_list) == 1, f"Expected 1 GeoTIFF, found {len(tif_list)}"
    assert len(asc_list) == 1, f"Expected 1 ArcASCII, found {len(asc_list)}"

    with rasterio.open(tif_list[0]) as ds:
        assert ds.dtypes[0] == "int16", f"Expected int16, got {ds.dtypes[0]}"


def test_hsg_pipeline_with_prefix(tmp_out: Path, data_dir: Path, example_dir: Path):
    """HSG preprocessing with a filename prefix.

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


def test_hsg_pipeline_custom_dtype_nodata(tmp_out: Path, data_dir: Path, example_dir: Path):
    """HSG preprocessing with custom dtype and nodata.

    Summary:
        Runs with --dtype int32 --nodata -9999 to verify non-default dtype/nodata
        are honored in the output.

    Expected outcome:
        - CLI exits with return code 0.
        - Output GeoTIFF has dtype int32 and nodata value of -9999.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--dtype", "int32", "--nodata", "-9999"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    with rasterio.open(tif_list[0]) as ds:
        assert ds.dtypes[0] == "int32", f"Expected int32, got {ds.dtypes[0]}"
        assert ds.nodata == -9999, f"Expected nodata=-9999, got {ds.nodata}"


def test_hsg_pipeline_uint8_dtype(tmp_out: Path, data_dir: Path, example_dir: Path):
    """HSG preprocessing with uint8 dtype and nodata=0.

    Summary:
        Runs with --dtype uint8 --nodata 0 to verify the tool handles unsigned
        integer output with a zero nodata sentinel.

    Expected outcome:
        - CLI exits with return code 0.
        - Output GeoTIFF has dtype uint8 and nodata value of 0.
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--dtype", "uint8", "--nodata", "0"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    with rasterio.open(tif_list[0]) as ds:
        assert ds.dtypes[0] == "uint8", f"Expected uint8, got {ds.dtypes[0]}"
        assert ds.nodata == 0, f"Expected nodata=0, got {ds.nodata}"


def test_hsg_pipeline_compress_deflate(tmp_out: Path, data_dir: Path, example_dir: Path):
    """HSG preprocessing with deflate compression.

    Summary:
        Runs with --compress deflate to verify the tool accepts alternate
        compression options without error.

    Expected outcome:
        - CLI exits with return code 0.
        - A valid GeoTIFF is produced (compression is an internal detail).
    """
    args = _base_args(data_dir, example_dir, tmp_out) + ["--compress", "deflate"]

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    assert len(tif_list) == 1
    with rasterio.open(tif_list[0]) as ds:
        assert ds.width > 0 and ds.height > 0


def test_hsg_pixel_values_valid(tmp_out: Path, data_dir: Path, example_dir: Path):
    """Verify all HSG pixel values are within the valid code range.

    Summary:
        Runs the default pipeline and reads the output GeoTIFF. Checks that
        every pixel is either the nodata value (-1) or a valid HSG code (1–7).

    Expected outcome:
        - No pixel values outside the set {-1, 1, 2, 3, 4, 5, 6, 7}.
    """
    args = _base_args(data_dir, example_dir, tmp_out)

    rc, out, err = _run_cli(args)
    assert rc == 0, f"CLI failed (rc={rc}). stderr:\n{err}"

    tif_list = list(tmp_out.glob("*.tif"))
    with rasterio.open(tif_list[0]) as ds:
        data = ds.read(1)

    valid_values = {-1, 1, 2, 3, 4, 5, 6, 7}
    unique_values = set(np.unique(data))
    invalid = unique_values - valid_values
    assert not invalid, f"Found invalid HSG pixel values: {invalid}"


def test_hsg_output_aligned_to_template(tmp_out: Path, data_dir: Path, example_dir: Path):
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
