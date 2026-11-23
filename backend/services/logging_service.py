"""
Structured Logging Service
============================
Configures a JSON-formatted logger for MDRIS backend.
Use this service to get consistent, parseable logs across all modules.
"""

import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logging to emit structured output.

    Args:
        level: Logging level string (DEBUG / INFO / WARNING / ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Simple format — swap in python-json-logger for production JSON logs
    fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence noisy third-party loggers
    for noisy in ("ultralytics", "transformers", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name:  Logger name (typically `__name__`).
        level: Optional override level for this logger.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
