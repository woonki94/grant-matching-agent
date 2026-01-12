from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings


def setup_logging(log_name : str | None = "app") -> None:
    """
    Call once at process start.
    Configures root logger (console + rotating file).
    """
    root = logging.getLogger()

    # Prevent duplicate handlers if setup_logging() is called twice
    if root.handlers:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    # File
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(
        log_dir / f"{log_name}.log",
        maxBytes=10_000_000,   # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)