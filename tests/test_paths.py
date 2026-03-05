"""
Tests for common.paths module.

These tests validate:
- Directory creation via ensure_dir()
- Filename construction via build_output_filename()

All tests use simple, deterministic values and make no assumptions
about the larger project configuration.
"""

from pathlib import Path
import shutil
import pytest
from swb2_prep.common.paths import ensure_dir, build_output_filename


def test_ensure_dir_creates_directory(tmp_path):
    """
    ensure_dir() should create the directory if it does not exist,
    and return a Path object referencing the created directory.
    """
    target = tmp_path / "output_dir"

    assert not target.exists()
    result = ensure_dir(target)

    assert target.exists()
    assert target.is_dir()
    assert isinstance(result, Path)


def test_ensure_dir_existing_directory(tmp_path):
    """
    ensure_dir() should not raise an error if the directory already exists.
    It should simply return the Path object.
    """
    target = tmp_path / "existing"
    target.mkdir()

    result = ensure_dir(target)

    assert result == target
    assert target.exists()
    assert target.is_dir()


def test_build_output_filename_basic():
    """
    build_output_filename() should embed:
    - base name
    - resolution with units
    - label
    - extension

    and follow the format:
        base__<resolution><units>.ext
    """
    fname = build_output_filename(
        base="AWC",
        resolution=30,
        units="m",
        prefix="south_manitou",
        ext=".asc"
    )

    assert fname == "south_manitou__AWC__30m.asc"


def test_build_output_filename_normalizes_extension():
    """
    build_output_filename() should accept an extension with or without a dot
    and always normalize it to include the leading dot.
    """
    fname = build_output_filename(
        base="HSG",
        resolution=325,
        units="ft",
        prefix="south_manitou",
        ext="tif"   # note: missing dot
    )

    assert fname == "south_manitou__HSG__325ft.tif"
