"""Per-request trace ID, available anywhere via contextvars."""

from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")
