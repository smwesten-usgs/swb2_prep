# -*- coding: utf-8 -*-
"""
Soil utilities for SWB2-prep: hydrologic soil group (HSG) and available water
capacity (AWC) lookup and reclassification.

This module provides reusable functions for reading the muaggatt table from a
gNATSGO GeoPackage and mapping MUKEY values to numeric HSG codes or continuous
AWC values (inches per foot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from swb2_prep.common.utils import PathLike

__all__ = [
    "HSG_MAPPING",
    "read_hsg_lookup",
    "reclassify_mukey_to_hsg",
    "read_awc_lookup",
    "reclassify_mukey_to_awc",
]

HSG_MAPPING: Dict[str, int] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "A/D": 5,
    "B/D": 6,
    "C/D": 7,
}
"""Mapping of HSG string codes to integer values used in SWB2 grids."""

_HSG_DEFAULT: int = 1
"""Default HSG code assigned when the lookup value is null or unrecognized."""


def read_hsg_lookup(gpkg_path: PathLike) -> Dict[int, int]:
    """Read the muaggatt table from a gNATSGO GeoPackage and build a MUKEY-to-HSG lookup.

    Args:
        gpkg_path: Path to a GeoPackage containing a ``muaggatt`` layer.
            The GeoPackage FID is expected to equal the MUKEY value.

    Returns:
        Dictionary mapping MUKEY (uint32) to numeric HSG code (int).

    Raises:
        FileNotFoundError: If ``gpkg_path`` does not exist.
        KeyError: If no recognized HSG column is found in muaggatt.
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    muaggatt = gpd.read_file(
        gpkg_path,
        layer="muaggatt",
        fid_as_index=True,
        ignore_geometry=True,
        use_arrow=True,
    )

    # Derive MUKEY from the FID index
    muaggatt["MUKEY"] = pd.to_numeric(muaggatt.index, errors="coerce")
    muaggatt = muaggatt.dropna(subset=["MUKEY"])
    muaggatt["MUKEY"] = muaggatt["MUKEY"].astype(np.uint32)

    # Locate HSG column (case-insensitive)
    hsg_col = _find_hsg_column(muaggatt.columns)
    if hsg_col is None:
        raise KeyError(
            "Could not find a hydrologic soil group column in muaggatt "
            "(expected one of: hydgrpdcd, hydrolgrp)."
        )

    muaggatt["hsg_numeric"] = (
        muaggatt[hsg_col].map(HSG_MAPPING).fillna(_HSG_DEFAULT).astype(int)
    )

    return dict(zip(muaggatt["MUKEY"], muaggatt["hsg_numeric"]))


def reclassify_mukey_to_hsg(
    mukey_array: np.ndarray,
    lookup: Dict[int, int],
    nodata_value: int = -1,
) -> np.ndarray:
    """Reclassify a MUKEY array to HSG integer codes using a lookup dictionary.

    Args:
        mukey_array: 2-D numpy array of MUKEY values (typically uint32).
        lookup: Dictionary mapping MUKEY to HSG code, as returned by
            :func:`read_hsg_lookup`.
        nodata_value: Value assigned to pixels with no matching MUKEY in the
            lookup (default ``-1``).

    Returns:
        2-D numpy array (int16) of HSG codes with the same shape as the input.
    """
    output = np.full(mukey_array.shape, nodata_value, dtype=np.int16)

    for mukey in np.unique(mukey_array):
        code = lookup.get(int(mukey))
        if code is not None:
            output[mukey_array == mukey] = code

    return output


def _find_hsg_column(columns: pd.Index) -> Union[str, None]:
    """Return the first matching HSG column name (case-insensitive).

    Args:
        columns: Column index from a DataFrame.

    Returns:
        The matched column name, or ``None`` if no match is found.
    """
    candidates = {"hydgrpdcd", "hydrolgrp"}
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None



def read_awc_lookup(gpkg_path: PathLike) -> Dict[int, float]:
    """Read the muaggatt table from a gNATSGO GeoPackage and build a MUKEY-to-AWC lookup.

    AWC is computed as inches per foot. The function prefers ``aws0150wta``
    (total inches over 150 cm depth); if unavailable, falls back to
    ``aws0100wta`` (total inches over 100 cm depth).

    Args:
        gpkg_path: Path to a GeoPackage containing a ``muaggatt`` layer.
            The GeoPackage FID is expected to equal the MUKEY value.

    Returns:
        Dictionary mapping MUKEY (uint32) to AWC in inches/foot (float32).

    Raises:
        FileNotFoundError: If ``gpkg_path`` does not exist.
        KeyError: If neither ``aws0150wta`` nor ``aws0100wta`` is found in muaggatt.
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    muaggatt = gpd.read_file(
        gpkg_path,
        layer="muaggatt",
        fid_as_index=True,
        ignore_geometry=True,
        use_arrow=True,
    )

    # Derive MUKEY from the FID index
    muaggatt["MUKEY"] = pd.to_numeric(muaggatt.index, errors="coerce")
    muaggatt = muaggatt.dropna(subset=["MUKEY"]).copy()
    muaggatt["MUKEY"] = muaggatt["MUKEY"].astype(np.uint32)

    # Compute AWC (inches per foot)
    if "aws0150wta" in muaggatt.columns:
        muaggatt["awc_in_per_ft"] = (
            muaggatt["aws0150wta"].astype(float) / 150.0
        ) * 12.0
    elif "aws0100wta" in muaggatt.columns:
        muaggatt["awc_in_per_ft"] = (
            muaggatt["aws0100wta"].astype(float) / 100.0
        ) * 12.0
    else:
        raise KeyError(
            "Could not find AWS columns in muaggatt "
            "(expected one of: aws0150wta, aws0100wta)."
        )

    muaggatt["awc_in_per_ft"] = muaggatt["awc_in_per_ft"].fillna(0.0).astype(np.float32)

    return dict(zip(muaggatt["MUKEY"], muaggatt["awc_in_per_ft"]))


def reclassify_mukey_to_awc(
    mukey_array: np.ndarray,
    lookup: Dict[int, float],
    nodata_value: float = -1.0,
) -> np.ndarray:
    """Reclassify a MUKEY array to AWC values (inches/foot) using a lookup dictionary.

    Args:
        mukey_array: 2-D numpy array of MUKEY values (typically uint32).
        lookup: Dictionary mapping MUKEY to AWC value, as returned by
            :func:`read_awc_lookup`.
        nodata_value: Value assigned to pixels with no matching MUKEY in the
            lookup (default ``-1.0``).

    Returns:
        2-D numpy array (float32) of AWC values with the same shape as the input.
    """
    output = np.full(mukey_array.shape, nodata_value, dtype=np.float32)

    for mukey in np.unique(mukey_array):
        val = lookup.get(int(mukey))
        if val is not None:
            output[mukey_array == mukey] = val

    return output
