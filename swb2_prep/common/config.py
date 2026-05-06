from __future__ import annotations

from pathlib import Path
import tomllib

def load_project_options(path: Path) -> dict:
    """Load a ``project_options.toml`` file into a nested dictionary.

    This function reads the project's configuration TOML using Python's
    standard ``tomllib`` reader (Python ≥ 3.11) and returns a nested dict
    with sections such as ``[project]`` and ``[paths]``.

    Args:
        path: Path to the ``project_options.toml`` file.

    Returns:
        A nested dictionary representing the TOML contents.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        tomllib.TOMLDecodeError: If the TOML is syntactically invalid.

    Notes:
        - Unit tests assert that required keys within ``[project]`` and ``[paths]``
          are loadable and that missing/invalid files raise the documented exceptions.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data
