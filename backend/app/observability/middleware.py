"""Assigns a trace ID to each request and logs method, path, status, latency."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.observability.context import trace_id_var

logger = logging.getLogger("http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        token = trace_id_var.set(trace_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            logger.info(
                "request",
                extra={"fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                }},
            )
        finally:
            trace_id_var.reset(token)
        response.headers["x-trace-id"] = trace_id
        return response
