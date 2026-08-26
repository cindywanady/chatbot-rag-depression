"""Structured (JSON) logging.

Spec 2.2 makes this a hard requirement for the whole phase: "use the
``logging`` module with structured (JSON) records ... bake it in from the first
script, not as an afterthought." Every stage logs one JSON object per event,
with arbitrary structured fields passed via ``extra=``::

    log = get_logger("segmentation")
    log.info("segment.kept", extra={"segment_id": sid, "content_type": ct})

Records go to stderr and, optionally, to a per-run log file.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT_LOGGER_NAME = "depression_rag"

# Attributes already present on every LogRecord; anything else is a user-supplied
# structured field and gets serialized into the JSON payload.
_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Render each LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, _dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    json_console: bool = True,
) -> logging.Logger:
    """Configure the package root logger. Idempotent (clears prior handlers)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    json_fmt = JsonFormatter()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        json_fmt
        if json_console
        else logging.Formatter("%(levelname)-7s %(name)s | %(message)s")
    )
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(json_fmt)  # file is always JSON
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger ``depression_rag.<name>``."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
