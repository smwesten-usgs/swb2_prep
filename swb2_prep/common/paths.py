"""
Utility functions for directory creation and consistent SWB output filenames.

Rules:
- All handled with pathlib.Path
- No object-oriented patterns
- Filenames must embed resolution + units (e.g., "30m", "325ft")
- Output label comes from polygon name or "bounding_box"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def build_output_filename(
    base: str,
    resolution: float,
    units: str,
    ext: str,
    prefix: Optional[str] = None,
) -> str:
    """Construct a standardized SWB output filename.

    The convention is: ``{prefix}__{base}__{resolution}{units}{ext}``.

    Examples:
        - ``south_manitou__landuse__30m.tif``
        - ``HSG__325ft.tif``

    Behavior:
        - **Extension normalization:** Accepts ``ext`` with or without a leading dot; always returns with a dot.
        - **Resolution formatting:** If ``resolution`` is integral (e.g., ``30.0``), it is rendered as an integer (``30``); otherwise, the original float is used.
        - **Prefix joining:** If a ``prefix`` is provided, it becomes the first segment, followed by ``base`` and the ``{resolution}{units}`` token.

    Args:
        base: Base label (e.g., ``'landuse'``, ``'HSG'``).
        resolution: Cell size (e.g., ``30`` or ``30.0``).
        units: Units for resolution (e.g., ``'m'`` or ``'ft'``).
        ext: File extension with or without leading dot (e.g., ``'.tif'`` or ``'tif'``).
        prefix: Optional prefix, commonly an AOI/project label (e.g., ``'south_manitou'``).

    Returns:
        Filename string (no directory component).

    Notes:
        - Your tests validate extension normalization and the presence of base, resolution+units,
          and prefix when supplied, ensuring deterministic naming across outputs. [3](https://doimspp-my.sharepoint.com/personal/smwesten_usgs_gov/Documents/Microsoft%20Copilot%20Chat%20Files/test_ops.py)
    """
    normalized_ext = ext if ext.startswith(".") else f".{ext}"

    if float(resolution).is_integer():
        res_str = f"{int(resolution)}"
    else:
        res_str = f"{resolution}"

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base)
    parts.append(f"{res_str}{units}")

    name = "__".join(parts) + normalized_ext
    return name