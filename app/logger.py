"""
Structured JSON logger.
Every log line is a JSON object — trivially ingestible by Datadog, Loki, CloudWatch, etc.
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
        }
        if hasattr(record, "event"):
            payload["event"] = record.event

        # Skip internal LogRecord fields — only keep custom kwargs
        _skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName", "event",
        }
        for key, val in record.__dict__.items():
            if key not in _skip:
                payload[key] = val

        payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


class StructuredLogger(logging.Logger):
    """Logger that accepts keyword args and bundles them into the LogRecord."""

    def _log_with_fields(self, level: int, event: str, **kwargs):
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "(unknown)", 0, event, (), None
            )
            record.event = event
            for k, v in kwargs.items():
                setattr(record, k, v)
            self.handle(record)

    def info(self, event: str, **kwargs):       # type: ignore[override]
        self._log_with_fields(logging.INFO, event, **kwargs)

    def error(self, event: str, **kwargs):      # type: ignore[override]
        self._log_with_fields(logging.ERROR, event, **kwargs)

    def warning(self, event: str, **kwargs):    # type: ignore[override]
        self._log_with_fields(logging.WARNING, event, **kwargs)

    def debug(self, event: str, **kwargs):      # type: ignore[override]
        self._log_with_fields(logging.DEBUG, event, **kwargs)


def get_structured_logger(name: str) -> StructuredLogger:
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = JsonFormatter()

        # Console handler — always on
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler — create logs dir if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/service.log",
            maxBytes=1024 * 1024,   # 1 MB per file
            backupCount=5,          # keep 5 rotated files → max 5 MB on disk
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False    # prevent double-logging to root logger

    return logger  # type: ignore[return-value]