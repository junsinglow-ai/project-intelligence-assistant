"""The per-request memo of models that are out of quota.

A chain survives an exhausted model; what it does not do on its own is
remember. `ModelFallbackMiddleware` walks from the top of the chain on every
model call, and a skill loop makes several, so a spent model is re-tried once
per generation. Measured on the deployed service: 30 `429`s in one question.
"""

import asyncio
from dataclasses import dataclass, replace

import pytest

from app.llm.quota import (
    QuotaAwareFallback,
    is_quota_error,
    mark_spent,
    model_name,
    quota_scope,
    spent_models,
)


class Quota(Exception):
    """Stands in for a provider's 429; matched on the wire status, not the type."""

    def __init__(self) -> None:
        super().__init__("429 RESOURCE_EXHAUSTED")


class Broken(Exception):
    def __init__(self) -> None:
        super().__init__("400 INVALID_ARGUMENT")


@dataclass
class FakeModel:
    model: str


@dataclass
class FakeRequest:
    model: FakeModel

    def override(self, model):
        return replace(self, model=model)


def _handler(calls, failing):
    async def handler(request):
        calls.append(request.model.model)
        if request.model.model in failing:
            raise failing[request.model.model]()
        return f"answer from {request.model.model}"

    return handler


def _run(middleware, request, handler, turns):
    async def main():
        return [await middleware.awrap_model_call(request, handler) for _ in range(turns)]

    return asyncio.run(main())


def test_a_spent_model_is_tried_once_not_once_per_generation():
    calls: list[str] = []
    request = FakeRequest(FakeModel("strong"))
    middleware = QuotaAwareFallback(FakeModel("weak"))

    async def main():
        with quota_scope():
            handler = _handler(calls, {"strong": Quota})
            for _ in range(4):
                await middleware.awrap_model_call(request, handler)

    asyncio.run(main())

    assert calls.count("strong") == 1, "the exhausted model was re-walked"
    assert calls.count("weak") == 4
    assert calls[:2] == ["strong", "weak"]


def test_without_a_scope_nothing_is_remembered():
    """Outside a request the memo must not leak between callers."""
    calls: list[str] = []
    request = FakeRequest(FakeModel("strong"))
    middleware = QuotaAwareFallback(FakeModel("weak"))

    _run(middleware, request, _handler(calls, {"strong": Quota}), turns=3)

    assert calls.count("strong") == 3


def test_a_non_quota_failure_does_not_retire_a_model():
    """A 400 is a property of the request, identical on every model."""
    calls: list[str] = []
    request = FakeRequest(FakeModel("strong"))
    middleware = QuotaAwareFallback(FakeModel("weak"))

    async def main():
        with quota_scope():
            handler = _handler(calls, {"strong": Broken})
            for _ in range(3):
                await middleware.awrap_model_call(request, handler)

    asyncio.run(main())

    assert calls.count("strong") == 3, "a 400 must not mark the model spent"


def test_everything_spent_still_makes_a_call():
    """A quota window can reopen; a real provider error beats a synthetic one.

    `base.py` only builds the middleware when the chain has fallbacks, so the
    parent's requirement of at least one is the shape real usage takes.
    """
    calls: list[str] = []
    request = FakeRequest(FakeModel("strong"))
    middleware = QuotaAwareFallback(FakeModel("weak"))

    async def main():
        with quota_scope():
            mark_spent("strong")
            mark_spent("weak")
            handler = _handler(calls, {"strong": Quota, "weak": Quota})
            with pytest.raises(Quota):
                await middleware.awrap_model_call(request, handler)

    asyncio.run(main())

    assert calls == ["strong", "weak"], "with all spent, the full chain is still tried"


def test_the_scope_is_the_request_not_the_process():
    with quota_scope():
        mark_spent("gemini-3.6-flash")
        assert spent_models() == {"gemini-3.6-flash"}
    assert spent_models() == set()


def test_quota_errors_are_matched_on_the_wire_status():
    assert is_quota_error(Quota())
    assert not is_quota_error(Broken())
    assert not is_quota_error(ValueError("something local"))


def test_model_name_prefers_the_providers_own_name():
    assert model_name(FakeModel("gemini-3.6-flash")) == "gemini-3.6-flash"
    assert model_name(object()) == "object"
