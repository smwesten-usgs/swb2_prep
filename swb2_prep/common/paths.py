"""
Utility functions for directory creation and consistent SWB output filenames.

Rules:
- All handled with pathlib.Path
- No object-oriented patterns
- Filenames must embed resolution + units (e.g., "30m", "325ft")
- Output label comes from polygon name or "bounding_box"
"""

from pathlib import Path

def ensure_dir(path):
    """
    Ensure that the given directory exists.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The created/existing directory as a Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_output_filename(base, resolution, units, ext, prefix=None):
    """
    Construct a standardized SWB output filename.

    Examples:
        AWC__30m__basin.asc
        HSG__325ft__bounding_box.tif

    Parameters
    ----------
    base : str
        Base dataset name, e.g., "AWC" or "HSG".
    resolution : int or float
        Resolution in project CRS units.
    units : str
        Units (e.g., "m", "ft").
    ext : str
        Extension including leading dot (".tif" or ".asc").
    prefix : str
        Optional prefix value.

    Returns
    -------
    str
        Constructed filename.
    """
    # Normalize extension
    if not ext.startswith("."):
        ext = "." + ext

    res_str = f"{resolution}{units}"
    if prefix is not None:
        return f"{prefix}__{base}__{res_str}{ext}"
    else:
        return f"{base}__{res_str}{ext}"