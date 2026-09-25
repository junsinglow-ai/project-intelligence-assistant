"""Where the chat graph checkpoints, and what happens when Redis is not there.

`REDIS_URL` selects the store (DECISIONS.md D-007). Everything here but the
`redis`-marked test runs without a Redis: what matters is which saver gets
built, that an unreachable one degrades instead of failing the request, and
that eviction still reaches the store it chose.
"""

import asyncio
import logging
import os
import uuid

import pytest

from app.agents.base import AgentResult
from app.agents.document_qa import DocumentQAAgent
from app.agents.router import RouteDecision, RouterAgent
from app.config import get_settings
from app.graph import checkpointer as checkpointer_module
from app.graph.build import run_chat_graph
from app.graph.checkpointer import (
    ensure_checkpointer_ready,
    forget_session,
    get_checkpointer,
)

REDIS_URL = os.environ.get("REDIS_TEST_URL", "redis://localhost:6379/0")


@pytest.fixture
def redis_configured(monkeypatch):
    """Point the settings at a Redis without requiring one to be running."""
    def install(url=REDIS_URL, ttl_minutes=None):
        monkeypatch.setenv("REDIS_URL", url)
        if ttl_minutes is not None:
            monkeypatch.setenv("CHECKPOINT_TTL_MINUTES", str(ttl_minutes))
        get_settings.cache_clear()
        get_checkpointer.cache_clear()
        return get_settings()

    return install


@pytest.fixture
def answering(monkeypatch):
    """Route and answer without a model call, so only the graph is exercised."""
    async def route(self, inp):
        return RouteDecision(agent="document_qa", confidence=0.9, reason="stubbed")

    async def run(self, inp):
        return AgentResult(answer=f"answered: {inp.question}", agent=self.name)

    monkeypatch.setattr(RouterAgent, "route", route)
    monkeypatch.setattr(DocumentQAAgent, "run", run)


# -- which saver gets built ------------------------------------------------


async def test_a_blank_redis_url_keeps_the_checkpointer_in_process():
    """The default, and what the test suite and a single-process run want."""
    from langgraph.checkpoint.memory import InMemorySaver

    assert get_settings().redis_url == ""
    assert isinstance(await ensure_checkpointer_ready(), InMemorySaver)


async def test_a_configured_redis_url_builds_the_redis_saver(redis_configured):
    """Construction must not connect: the connection is the graph run's."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    redis_configured(url="redis://127.0.0.1:1/0")

    assert isinstance(get_checkpointer(), AsyncRedisSaver)


async def test_the_ttl_is_the_configured_one_and_refreshes_on_read(redis_configured):
    """An active conversation must not expire underneath itself."""
    redis_configured(url="redis://127.0.0.1:1/0", ttl_minutes=42)

    assert get_checkpointer().ttl_config == {"default_ttl": 42, "refresh_on_read": True}


# -- an unreachable Redis --------------------------------------------------


async def test_an_unreachable_redis_degrades_to_the_in_process_saver(
        redis_configured, answering):
    """Nothing resumes mid-graph today, so a checkpoint is not worth a 503."""
    from langgraph.checkpoint.memory import InMemorySaver

    # Port 1 is closed rather than merely absent, so this fails fast.
    redis_configured(url="redis://127.0.0.1:1/0")

    state = await run_chat_graph("What is the budget?", "s1")

    assert state["result"].answer.endswith("What is the budget?")
    assert isinstance(get_checkpointer(), InMemorySaver)
    assert checkpointer_module._degraded is True


async def test_the_degraded_saver_is_not_retried_for_every_request(
        redis_configured, answering, monkeypatch):
    """Reattaching mid-process would split one conversation across two stores."""
    redis_configured(url="redis://127.0.0.1:1/0")
    attempts = []

    original = checkpointer_module.get_checkpointer

    def counting():
        saver = original()
        attempts.append(type(saver).__name__)
        return saver

    await run_chat_graph("First?", "s1")
    monkeypatch.setattr(checkpointer_module, "get_checkpointer", counting)
    await run_chat_graph("Second?", "s1")

    assert "AsyncRedisSaver" not in attempts


# -- eviction reaches the store --------------------------------------------


async def test_eviction_deletes_the_redis_thread_without_blocking_the_response(
        redis_configured, monkeypatch):
    """`AsyncRedisSaver.delete_thread` raises; only the async half exists."""
    redis_configured(url="redis://127.0.0.1:1/0")
    deleted = []

    async def adelete_thread(thread_id):
        deleted.append(thread_id)

    monkeypatch.setattr(get_checkpointer(), "adelete_thread", adelete_thread)

    forget_session("old-session")

    assert deleted == []          # detached, not awaited on the response path
    await asyncio.sleep(0)
    assert deleted == ["old-session"]


async def test_a_failed_delete_is_logged_and_never_raised(
        redis_configured, monkeypatch, caplog):
    """Eviction is housekeeping on a path that is already returning an answer."""
    redis_configured(url="redis://127.0.0.1:1/0")
    caplog.set_level(logging.WARNING)

    async def adelete_thread(thread_id):
        raise ConnectionError("redis went away")

    monkeypatch.setattr(get_checkpointer(), "adelete_thread", adelete_thread)

    forget_session("old-session")
    await asyncio.sleep(0)

    assert "could not drop checkpoint thread" in caplog.text


# -- against a real Redis --------------------------------------------------


def _redis_reachable() -> bool:
    try:
        from redis import Redis

        with Redis.from_url(REDIS_URL, socket_connect_timeout=1) as client:
            client.ping()
        return True
    except Exception:
        return False


@pytest.mark.redis
@pytest.mark.skipif(not _redis_reachable(), reason=f"no Redis at {REDIS_URL}")
async def test_a_session_survives_in_redis_and_eviction_removes_it(
        redis_configured, answering):
    """The wiring end to end: a thread is written, read back, then reclaimed."""
    redis_configured()
    session = f"test-{uuid.uuid4().hex[:8]}"

    await run_chat_graph("What is the budget?", session)

    saver = get_checkpointer()
    config = {"configurable": {"thread_id": session}}
    assert await saver.aget_tuple(config) is not None

    forget_session(session)
    await asyncio.sleep(0.2)
    assert await saver.aget_tuple(config) is None
