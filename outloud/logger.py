"""Logging to outloud-logs/ directory."""

import logging
import os
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).parent.parent / "outloud-logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Get a logger that writes to a daily log file."""
    logger = logging.getLogger(f"outloud.{name}")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        log_file = LOG_DIR / f"outloud_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False

    return logger
