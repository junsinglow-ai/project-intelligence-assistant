"""Structured JSON logging. Every record carries the current trace_id."""

import json
import logging
import sys
from datetime import datetime, timezone

from app.observability.context import trace_id_var


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            # Duplicated as `severity` for hosted log collectors, which is the
            # key Google Cloud Logging reads to rank a record; it ignores
            # `level` and files everything as INFO. Without this a failed
            # request is invisible to a severity filter or an alert -- an
            # `agent failed` ERROR only turned up by grepping every entry.
            # `level` is kept so local output and the tests are unchanged.
            "severity": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        # Structured fields: logger.info("...", extra={"fields": {...}})
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
