from __future__ import annotations

"""Shared command-line and logging helpers for script entrypoints.

These utilities keep the executable scripts small and make the CLI behaviour
consistent across training, evaluation, and risk-model workflows.
"""

import argparse
import logging
import sys
from pathlib import Path


def add_common_config_args(parser: argparse.ArgumentParser, *, default_config: Path) -> argparse.ArgumentParser:
    """Register the standard `--config` and `--device` arguments used by script entrypoints."""
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Override device from config (cuda/cpu)")
    return parser


def setup_basic_logging(log_file: Path, *, logger_name: str | None = None) -> logging.Logger:
    """Configure a console+file logger and return the selected logger instance."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(logger_name) if logger_name else logging.getLogger()
