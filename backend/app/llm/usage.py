"""Token accounting for every model call.

Each agent has its own model chain (D-001), so which model answered and what it
spent are the numbers that say whether an assignment is paying off -- and
whether a request fell through to a billed fallback. `UsageLogger` is attached
to every model `app/llm/providers.py` builds, so no call site can forget it.

Two ContextVars carry the request's context into the callback, which LangChain
invokes with no idea which agent it is serving: `_site_var` names the call site
(an agent, or `rewrite`), and `_tally_var` accumulates the request's totals for
the one `chat` log line that summarises it.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


@dataclass
class UsageTally:
    """A request's token spend, per model. Thread-safe: tool calls run in parallel."""

    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, input_tokens: int, output_tokens: int, reasoning: int) -> None:
        with self._lock:
            row = self.by_model.setdefault(
                model, {"calls": 0, "input": 0, "output": 0, "reasoning": 0}
            )
            row["calls"] += 1
            row["input"] += input_tokens
            row["output"] += output_tokens
            row["reasoning"] += reasoning

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "input": sum(r["input"] for r in self.by_model.values()),
                "output": sum(r["output"] for r in self.by_model.values()),
                "reasoning": sum(r["reasoning"] for r in self.by_model.values()),
                "by_model": {m: dict(r) for m, r in self.by_model.items()},
            }


_site_var: ContextVar[str | None] = ContextVar("model_call_site", default=None)
_tally_var: ContextVar[UsageTally | None] = ContextVar("usage_tally", default=None)


@contextmanager
def call_site(name: str) -> Iterator[None]:
    """Attribute the model calls made inside this block to `name`."""
    token = _site_var.set(name)
    try:
        yield
    finally:
        _site_var.reset(token)


@contextmanager
def usage_scope() -> Iterator[UsageTally]:
    """Collect every model call made inside this block into one tally."""
    tally = UsageTally()
    token = _tally_var.set(tally)
    try:
        yield tally
    finally:
        _tally_var.reset(token)


class UsageLogger(BaseCallbackHandler):
    """Log one line per model call, and add it to the request's tally.

    `run_inline` because a synchronous handler is otherwise dispatched to an
    executor thread on async calls; running it on the calling task keeps the
    ContextVars -- and so the call site and the tally -- in view.
    """

    run_inline = True

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue
                metadata = message.response_metadata or {}
                model = metadata.get("model_name") or metadata.get("model") or "unknown"
                reasoning = (usage.get("output_token_details") or {}).get("reasoning", 0)
                logger.info("model call", extra={"fields": {
                    "site": _site_var.get(),
                    "model": model,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "reasoning_tokens": reasoning,
                }})
                tally = _tally_var.get()
                if tally is not None:
                    tally.add(model, usage.get("input_tokens", 0),
                              usage.get("output_tokens", 0), reasoning)


USAGE_LOGGER = UsageLogger()
