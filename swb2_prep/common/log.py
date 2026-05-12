# -*- coding: utf-8 -*-
"""Logging configuration for SWB2-prep CLI tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    script_name: str,
    log_dir: Path,
    *,
    no_log: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a named logger for a CLI tool.

    Logs to console (always) and appends to ``swb2_prep.log`` in *log_dir*
    unless *no_log* is True.

    Args:
        script_name: Logger name (e.g., ``"prep_hsg_input"``).
        log_dir: Directory for the log file (typically the project_options.toml parent).
        no_log: If True, suppress file logging; console output only.
        level: Logging level (default: INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (append mode)
    if not no_log:
        log_path = log_dir / "swb2_prep.log"
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
