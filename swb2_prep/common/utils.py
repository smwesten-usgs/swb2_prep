# -*- coding: utf-8 -*-
"""Common utilities for SWB2-prep."""

from __future__ import annotations

from pathlib import Path
from typing import Union
from pyproj import CRS

# ---- Type aliases (used across modules) ----
CRSLike = Union[str, dict, CRS]
PathLike = Union[str, Path]


def crs_equal(a: CRSLike, b: CRSLike) -> bool:
    """Return True if two CRS definitions are equivalent.

    Args:
        a: First CRS (EPSG string, PROJ dict, or pyproj.CRS).
        b: Second CRS (EPSG string, PROJ dict, or pyproj.CRS).

    Returns:
        True if CRS are considered equal via pyproj.CRS.equals; False otherwise.
    """
    try:
        return CRS.from_user_input(a).equals(CRS.from_user_input(b))
    except Exception:
        return False