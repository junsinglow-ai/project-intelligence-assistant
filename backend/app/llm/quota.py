"""Remember which models are out of quota for the duration of one request.

A chain of models exists so an exhausted one is survivable (D-001). What it
does not do on its own is *remember*: `ModelFallbackMiddleware` walks the chain
from the top on every model call, and a skill loop makes three to five of them,
so a model whose daily quota is spent is re-tried from scratch each time.

Measured on the deployed service, one `data_analysis` question produced **30**
429 responses: ten calls each to the two exhausted models, repeated for every
generation, before reaching a model with quota left. The answer that eventually
came back was from the weakest model in the chain, after 143 seconds.

This narrows that to one failure per model per request. The scope is
deliberately the request and not the process: quota windows reopen, and a
process-wide memo would keep using the weakest model long after the strongest
one recovered. It mirrors the evidence collector's ContextVar, which is
scoped the same way and for the same reason.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware

logger = logging.getLogger(__name__)

_exhausted: ContextVar[set[str] | None] = ContextVar("exhausted_models", default=None)


@contextmanager
def quota_scope() -> Iterator[set[str]]:
    """Track exhausted models for one request. Nesting is safe."""
    spent: set[str] = set()
    token = _exhausted.set(spent)
    try:
        yield spent
    finally:
        _exhausted.reset(token)


def spent_models() -> set[str]:
    """Models already known out of quota. Empty outside a scope."""
    return _exhausted.get() or set()


def mark_spent(name: str) -> None:
    spent = _exhausted.get()
    if spent is not None and name not in spent:
        spent.add(name)
        logger.info("model out of quota", extra={"fields": {"model": name}})


def model_name(model: Any) -> str:
    """The provider's own name for a chat model.

    Both supported providers expose `.model`; anything else falls back to the
    class name, which is still stable enough to deduplicate within a request.
    """
    return str(getattr(model, "model", None) or type(model).__name__)


def is_quota_error(exc: BaseException) -> bool:
    """Whether the provider refused for quota rather than for this request.

    Matched on the wire status rather than the exception type: the provider
    SDKs raise the same class for every 4xx, and a 400 must not mark a model
    spent -- it would be a property of the request, identical on every model.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


class QuotaAwareFallback(ModelFallbackMiddleware):
    """`ModelFallbackMiddleware` that skips models already spent this request.

    Overrides only the async hook, because every agent runs through
    `loop.ainvoke`; the synchronous path keeps the parent's behaviour so this
    cannot change how anything else in LangChain uses the class.

    The parent sanitises a request before handing it to a fallback, which
    strips Anthropic cache-control markers. Neither supported provider sets
    them and neither accepts them, so the request is passed through unchanged
    rather than reaching into the parent's private helpers.
    """

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        from langgraph.errors import GraphBubbleUp

        spent = spent_models()
        candidates = [request.model, *self.models]
        viable = [m for m in candidates if model_name(m) not in spent]
        # Everything is marked spent: fall back to the full chain rather than
        # failing without a call. A quota window can reopen mid-request, and a
        # real provider error is a better outcome than a synthetic one.
        if not viable:
            viable = candidates

        last_exception: BaseException | None = None
        for model in viable:
            try:
                return await handler(request.override(model=model))
            except GraphBubbleUp:
                raise
            except Exception as exc:
                last_exception = exc
                if is_quota_error(exc):
                    mark_spent(model_name(model))
                continue

        assert last_exception is not None  # `viable` is never empty
        raise last_exception
