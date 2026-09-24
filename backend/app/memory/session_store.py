"""Conversation session storage and follow-up question rewriting.

Two responsibilities, both deliberately kept out of the agents: an agent
receives a question that already stands on its own, which is what lets the
`AgentInput` -> `AgentResult` contract stay stateless (DECISIONS.md D-013).

Storage is in-process. History does not survive a restart and is not shared
between replicas, so conversational continuity assumes a single backend
instance. It is bounded on both axes -- turns within a session, and sessions
within the process -- because a client is free to invent a new session ID on
every request, and an unbounded store would make that a memory leak.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

MAX_TURNS = 10
MAX_SESSIONS = 500

# Session IDs are echoed into logs and used as dictionary keys, so they are
# constrained to what this service mints rather than accepted as free text.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# A rewrite should be about as long as the question it replaces. Far longer
# means the model started explaining instead, which is a fallback, not an answer.
_MIN_REWRITE_CHARS = 400


def valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID_RE.match(session_id))


@dataclass(slots=True)
class Turn:
    question: str
    answer: str
    agent: str
    at: str


class SessionStore:
    """Bounded, thread-safe conversation history.

    FastAPI serves handlers concurrently and agent work runs on worker threads
    via `asyncio.to_thread`, so the LRU eviction here -- a read, a `move_to_end`
    and a `popitem` -- is a compound operation that needs a lock. `bm25.py`
    guards its module state the same way.
    """

    def __init__(self, max_turns: int = MAX_TURNS, max_sessions: int = MAX_SESSIONS) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, deque[Turn]] = OrderedDict()

    def new_session(self) -> str:
        return uuid.uuid4().hex

    def append(self, session_id: str, question: str, answer: str, agent: str) -> None:
        turn = Turn(question=question, answer=answer, agent=agent,
                    at=datetime.now(timezone.utc).isoformat())
        with self._lock:
            turns = self._sessions.get(session_id)
            if turns is None:
                turns = deque(maxlen=self._max_turns)
                self._sessions[session_id] = turns
            turns.append(turn)
            self._sessions.move_to_end(session_id)
            dropped = []
            while len(self._sessions) > self._max_sessions:
                evicted, _ = self._sessions.popitem(last=False)
                dropped.append(evicted)
                logger.info("session evicted", extra={"fields": {"session_id": evicted}})

        # Outside the lock: the graph's checkpointer is keyed by the same session
        # IDs, so an evicted conversation has to drop its checkpoint thread too
        # or the bound here would be meaningless (D-015). It is no longer the
        # only bound -- checkpoints in Redis outlive this process and carry
        # their own TTL (D-017) -- but it is still the one that reclaims a
        # thread the moment its conversation is gone.
        for session in dropped:
            self._forget(session)

    @staticmethod
    def _forget(session_id: str) -> None:
        from app.graph.checkpointer import forget_session

        forget_session(session_id)

    def transcript(self, session_id: str) -> list[Turn] | None:
        """Full turns, or None if the session is unknown."""
        with self._lock:
            turns = self._sessions.get(session_id)
            if turns is None:
                return None
            self._sessions.move_to_end(session_id)
            return list(turns)

    def history(self, session_id: str) -> list[dict[str, str]]:
        """History in the shape `AgentInput.history` accepts: str keys and values."""
        turns = self.transcript(session_id) or []
        messages: list[dict[str, str]] = []
        for turn in turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    @property
    def max_turns(self) -> int:
        return self._max_turns


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()


class StandaloneQuestion(BaseModel):
    """Structured output, because a small model asked to rewrite a question
    readily answers "Sure! Here is the standalone question: ..." instead."""

    question: str


def render_history(history: list[dict[str, str]]) -> str:
    return "\n".join(f"{turn.get('role', '?')}: {turn.get('content', '')}" for turn in history)


async def rewrite_follow_up(question: str, history: list[dict[str, str]],
                            settings: Settings | None = None) -> str:
    """Rewrite a follow-up into a standalone question.

    Rewriting rather than growing the prompt keeps every downstream prompt the
    size of one question plus its context, whatever the conversation length --
    which matters because prompt processing, not generation, dominates on-prem
    latency (ARCHITECTURE.md section 7.1).

    Never fails a request: the raw question is returned whenever the rewrite
    errors or comes back implausible.
    """
    if not history:
        # The overwhelmingly common case, and a whole model call saved.
        return question

    from app.llm.prompts import REWRITE_PROMPT
    from app.llm.providers import get_structured_llm

    settings = settings or get_settings()
    try:
        chain = REWRITE_PROMPT | get_structured_llm(
            StandaloneQuestion, settings.router_models, settings
        )
        rewritten = (await chain.ainvoke({
            "history": render_history(history),
            "question": question,
        })).question.strip()
    except Exception as exc:
        logger.warning("rewrite failed", extra={"fields": {
            "rewritten": False, "error": f"{type(exc).__name__}: {exc}"}})
        return question

    if not _plausible(rewritten, question):
        logger.warning("rewrite discarded", extra={"fields": {
            "rewritten": False, "reason": "implausible", "length": len(rewritten)}})
        return question

    logger.info("question rewritten", extra={"fields": {"rewritten": rewritten != question}})
    return rewritten


def _plausible(rewritten: str, question: str) -> bool:
    if not rewritten or not any(character.isalnum() for character in rewritten):
        return False
    return len(rewritten) <= max(4 * len(question), _MIN_REWRITE_CHARS)
