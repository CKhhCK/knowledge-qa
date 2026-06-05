"""
Structured logging setup for the QA agent backend.

Supports both JSON (production) and text (development) formats.
Logs to both console (stdout) and rotating files (./logs/ directory).
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Literal


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for attr in ("request_id", "session_id", "duration_ms"):
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable colored log format for development (console only)."""

    COLORS = {"DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m", "ERROR": "\033[31m"}
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.now().strftime("%H:%M:%S")
        base = f"{color}[{ts} {record.levelname:<7}]{self.RESET} {record.name}: {record.getMessage()}"
        if record.exc_info and record.exc_info[0]:
            base += "\n" + self.formatException(record.exc_info)
        return base


class PlainTextFormatter(logging.Formatter):
    """Plain text formatter for file logs (no color codes)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base = f"[{ts} {record.levelname:<7}] {record.name}: {record.getMessage()}"
        if record.exc_info and record.exc_info[0]:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str = "INFO", fmt: Literal["json", "text"] = "text",
                  log_dir: str = "./logs") -> None:
    """
    Configure logging with console + file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        fmt: Console format ("json" for production, "text" for development)
        log_dir: Directory for log files (created if not exists)
    """
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # --- Console handler (colored text or JSON) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(root.level)
    console.setFormatter(JSONFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(console)

    # --- File handler: all logs, daily rotation, 30 days retention ---
    all_log = os.path.join(log_dir, "qa_backend.log")
    file_handler = TimedRotatingFileHandler(
        all_log, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(root.level)
    file_handler.setFormatter(PlainTextFormatter() if fmt == "text" else JSONFormatter())
    root.addHandler(file_handler)

    # --- File handler: error logs only ---
    error_log = os.path.join(log_dir, "qa_backend_error.log")
    error_handler = TimedRotatingFileHandler(
        error_log, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(PlainTextFormatter() if fmt == "text" else JSONFormatter())
    root.addHandler(error_handler)

    # --- File handler: upload operations ---
    upload_log = os.path.join(log_dir, "upload.log")
    upload_handler = TimedRotatingFileHandler(
        upload_log, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    upload_handler.setLevel(logging.DEBUG)
    upload_handler.setFormatter(PlainTextFormatter() if fmt == "text" else JSONFormatter())
    upload_handler.addFilter(lambda record: record.name.startswith("app.api.routes.documents")
                                          or record.name.startswith("app.knowledge"))
    root.addHandler(upload_handler)

    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized | level={level} | dir={os.path.abspath(log_dir)}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)
