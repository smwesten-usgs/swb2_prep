from pathlib import Path
import tomllib
import pytest

from swb2_prep.common.config import load_project_options


def test_load_project_options_success(tmp_path):
    # Create a temporary TOML file
    toml_content = b"""
    [project]
    crs = "EPSG:5070"
    resolution = 30

    [paths]
    input_data_dir = "data"
    output_data_dir = "outputs"
    """
    toml_path = tmp_path / "project_options.toml"
    toml_path.write_bytes(toml_content)

    result = load_project_options(toml_path)

    assert "project" in result
    assert result["project"]["crs"] == "EPSG:5070"
    assert result["project"]["resolution"] == 30

    assert "paths" in result
    assert result["paths"]["input_data_dir"] == "data"
    assert result["paths"]["output_data_dir"] == "outputs"


def test_load_project_options_missing_file():
    with pytest.raises(FileNotFoundError):
        load_project_options(Path("does_not_exist.toml"))


def test_load_project_options_bad_toml(tmp_path):
    # Invalid TOML
    toml_path = tmp_path / "bad.toml"
    toml_path.write_text("not = valid = toml")

    with pytest.raises(tomllib.TOMLDecodeError):
        load_project_options(toml_path)