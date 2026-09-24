"""Graph checkpointing: Redis when configured, in-process otherwise.

`REDIS_URL` selects between them. Blank keeps the `InMemorySaver` the graph
started with, which is what the test suite and a single-process run want; the
Compose stack points it at the redis service. See DECISIONS.md D-017.

What the checkpoint holds is graph state, not conversation context. Follow-ups
are still rewritten into standalone questions (D-013), so the prompt an agent
sees is one question and its tool results however long the conversation runs.
That is also why an unreachable Redis degrades to the in-process saver rather
than failing the request: nothing resumes mid-graph today, so a checkpoint the
user never reads is not worth a 503.

Two bounds, because neither alone is enough once the keys outlive the process.
`SessionStore` evicts its least recently used session and that eviction reaches
here, which is what stops a client inventing a session ID per request from
growing the store; but that store is process-local, so a restart orphans every
thread it would have evicted. `CHECKPOINT_TTL_MINUTES` expires them instead, and
is refreshed on read so an active conversation is never expired underneath it.
"""

import asyncio
import logging
import weakref
from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import get_settings

logger = logging.getLogger(__name__)

# Set once Redis has proved unreachable, and never cleared: a checkpointer that
# reattached itself mid-process would leave one conversation's threads split
# across two stores. Recovery is a restart, which is visible in the logs and in
# `/v1/health/dependencies`.
_degraded = False

# `asetup()` creates the RediSearch indices the saver queries through, so it has
# to run before the first checkpoint is read. A lock per event loop rather than
# one module-level lock: a lock first awaited on a loop that has since closed --
# which is the shape of a test suite -- cannot be awaited on the next one.
_prepared: weakref.WeakSet = weakref.WeakSet()
_setup_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Detached deletes, held only so the event loop does not collect them mid-flight.
_deletions: set[asyncio.Task] = set()


@lru_cache(maxsize=1)
def get_checkpointer() -> BaseCheckpointSaver:
    settings = get_settings()
    if _degraded or not settings.redis_url:
        return InMemorySaver()

    # Imported here rather than at module scope so a deployment that does not
    # use Redis does not pay for the client on the import path.
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    return AsyncRedisSaver(
        settings.redis_url,
        # Seconds. The default is short enough that a managed Redis in another
        # region fails a request outright, and a read timeout happens well
        # after `ensure_checkpointer_ready` could have degraded to the
        # in-process saver, so it surfaces as a failed answer (D-017).
        connection_args={
            "socket_timeout": settings.redis_timeout_s,
            "socket_connect_timeout": settings.redis_timeout_s,
        },
        # Minutes, and refreshed whenever the thread is read, so the TTL expires
        # abandoned conversations rather than long ones.
        ttl={"default_ttl": settings.checkpoint_ttl_minutes, "refresh_on_read": True},
    )


async def ensure_checkpointer_ready() -> BaseCheckpointSaver:
    """Prepare the saver before the graph runs against it.

    Called from `run_chat_graph` rather than from a startup hook so that every
    entry point -- the API, the evaluation harness, a script -- gets a prepared
    saver. It is a set membership test after the first call.
    """
    saver = get_checkpointer()
    if saver in _prepared:
        return saver
    if isinstance(saver, InMemorySaver):
        _prepared.add(saver)
        return saver

    async with _setup_lock():
        if saver in _prepared:
            return saver
        try:
            await saver.asetup()
        except Exception as exc:
            return _degrade(exc)
        _prepared.add(saver)
    return saver


def forget_session(session_id: str) -> None:
    """Drop a session's checkpoints, called when the session store evicts it."""
    saver = get_checkpointer()
    if isinstance(saver, InMemorySaver):
        try:
            saver.delete_thread(session_id)
        except Exception as exc:
            _could_not_drop(exc)
        return

    # `AsyncRedisSaver` implements only the async half of the interface -- its
    # `delete_thread` raises -- while eviction happens on the synchronous side
    # of a request that is already returning an answer. So the delete is
    # detached onto the running loop rather than awaited: reclaiming a thread
    # must not add a round trip to the response.
    _detach(saver.adelete_thread(session_id))


def _setup_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _setup_locks.get(loop)
    if lock is None:
        lock = _setup_locks[loop] = asyncio.Lock()
    return lock


def _degrade(exc: Exception) -> BaseCheckpointSaver:
    """Fall back to the in-process saver for the life of the process."""
    global _degraded

    from app.graph.build import reset_chat_graph

    _degraded = True
    logger.warning("checkpointer degraded to in-process", extra={"fields": {
        "target": get_settings().checkpoint_target,
        "error": f"{type(exc).__name__}: {exc}"}})
    get_checkpointer.cache_clear()
    # The graph was compiled against the saver that just failed.
    reset_chat_graph()
    saver = get_checkpointer()
    _prepared.add(saver)
    return saver


def _detach(coro) -> None:
    reported = _reporting(coro)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running: a script, or a synchronous test.
        asyncio.run(reported)
        return
    task = loop.create_task(reported)
    _deletions.add(task)
    task.add_done_callback(_deletions.discard)


async def _reporting(coro) -> None:
    try:
        await coro
    except Exception as exc:
        _could_not_drop(exc)


def _could_not_drop(exc: Exception) -> None:
    # Eviction is housekeeping on a path that is already returning an answer;
    # failing to reclaim a thread must not fail the request. The TTL collects
    # whatever this misses.
    logger.warning("could not drop checkpoint thread", extra={"fields": {
        "error": f"{type(exc).__name__}: {exc}"}})
