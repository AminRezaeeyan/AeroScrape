"""JSON logs for the API, one object per line, each carrying the request ID it belongs to.

These are the logs the monitoring phase reads: request ID, model version, latency and
prediction summaries are fields, not text to parse.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else was passed with `extra=` and becomes a field
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id.get()
        if rid:
            entry["request_id"] = rid
        entry |= {k: v for k, v in record.__dict__.items() if k not in _STANDARD and not k.startswith("_")}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send every logger, uvicorn's included, through one JSON handler on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    # Our middleware logs each request with more context than uvicorn's access log
    logging.getLogger("uvicorn.access").disabled = True
