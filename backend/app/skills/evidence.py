"""What a skill found, and how it becomes a citation.

A tool loop breaks the assumption the fixed pipeline relied on: the model may
search several times, and only ever sees a tool's *output text*, so there is no
single retrieval result list to map `[n]` markers back onto. Skills therefore
record what they returned into a per-invocation collector, which assigns the
index the model is shown. `[3]` means the same passage whichever tool call
produced it, and `attribute()` can resolve markers afterwards exactly as before.

The collector travels in a ContextVar, mirroring `trace_id_var` in
`app/observability/context.py`. `asyncio.to_thread` copies the context, so a
skill doing its blocking work on a worker thread mutates the same object; note
that this works because the object is mutated in place -- a `set()` inside the
worker would not propagate back to the caller.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.agents.base import Citation

_MARKER_RE = re.compile(r"\[(\d+)\]")
_SNIPPET_CHARS = 200


BUDGET_SPENT = (
    "You have used the tool budget for this question. Do not call any more "
    "tools; answer now from what you already have, and say plainly if it is "
    "not enough to answer."
)


@dataclass
class Evidence:
    """Everything the skills surfaced during one agent invocation.

    This also holds the tool budget, because it is already the one object that
    counts calls. Enforcing it here rather than with `ToolCallLimitMiddleware`
    is deliberate: that middleware blocks the call but does not end the graph,
    so a model that keeps asking spins until the recursion limit and loses the
    answer it could have given. A skill that declines and says why leaves the
    model able to answer from what it has.
    """

    items: list[Any] = field(default_factory=list)
    tool_calls: int = 0
    budget: int = 0                 # 0 means unbounded
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start_call(self) -> bool:
        """Claim a slot in the tool budget; False once it is spent."""
        with self._lock:
            if self.budget and self.tool_calls >= self.budget:
                return False
            self.tool_calls += 1
            return True

    def record(self, items: list[Any]) -> int:
        """Append `items` and return the 1-based index the first one was given.

        Indices are handed out under the lock because a model turn may call
        several tools at once, and two skills numbering their output
        independently would produce two `[1]`s meaning different things.
        """
        with self._lock:
            start = len(self.items) + 1
            self.items.extend(items)
            return start


_evidence_var: ContextVar[Evidence | None] = ContextVar("evidence", default=None)


@contextmanager
def evidence_scope(budget: int = 0) -> Iterator[Evidence]:
    """Collect skill output for the duration of one agent run."""
    evidence = Evidence(budget=budget)
    token = _evidence_var.set(evidence)
    try:
        yield evidence
    finally:
        _evidence_var.reset(token)


def current_evidence() -> Evidence | None:
    """The collector for the run in progress, or None outside an agent."""
    return _evidence_var.get()


def snippet(text: str) -> str:
    from app.ingestion.cleaning import collapse_ws

    collapsed = collapse_ws(text)
    return collapsed[:_SNIPPET_CHARS].rstrip() + ("..." if len(collapsed) > _SNIPPET_CHARS else "")


def citation_for(result) -> Citation:
    return Citation(source=result.source, location=result.citation or None,
                    snippet=snippet(result.text))


def deduplicated(results: list) -> list:
    seen: set[tuple[str, str]] = set()
    unique = []
    for result in results:
        key = (result.source, result.citation)
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique


def attribute(answer: str, results: list) -> tuple[str, list[Citation], str]:
    """Map the answer's [n] markers onto citations, renumbering as it goes.

    Cites only what the answer actually referenced: citing every retrieved chunk
    overstates the evidence and depresses retrieval precision in the evaluation.
    Markers are renumbered to 1..k so the brackets in the prose match the list
    returned to the caller, and a marker pointing at nothing is dropped rather
    than left dangling.

    A small model sometimes omits markers entirely. Rather than return an
    uncited answer, fall back to citing everything retrieved and record that in
    `citation_mode` so the logs and the evaluation can tell the two apart.
    """
    used: list[int] = []
    for token in _MARKER_RE.findall(answer):
        index = int(token)
        if 1 <= index <= len(results) and index not in used:
            used.append(index)

    if not used:
        return answer, [citation_for(result) for result in deduplicated(results)], "inferred"

    renumbering: dict[int, int] = {}
    seen: dict[tuple[str, str], int] = {}
    citations: list[Citation] = []
    for index in used:
        result = results[index - 1]
        key = (result.source, result.citation)
        if key not in seen:
            citations.append(citation_for(result))
            seen[key] = len(citations)
        renumbering[index] = seen[key]

    renumbered = _MARKER_RE.sub(
        lambda match: (f"[{renumbering[int(match.group(1))]}]"
                       if int(match.group(1)) in renumbering else ""),
        answer,
    )
    return renumbered.strip(), citations, "marked"
