"""State carried through the chat graph.

`AgentInput`/`AgentResult` stay the contract an agent sees; this is the envelope
around them, holding what the graph needs between nodes -- the question before
and after rewriting, the routing decision, and the result.
"""

from typing import Any, TypedDict

from app.agents.base import AgentResult


class ChatState(TypedDict, total=False):
    question: str                    # standalone; what the agent is given
    original_question: str           # as asked, before any rewrite
    session_id: str
    history: list[dict[str, str]]
    needs_rewrite: bool              # False when the caller already rewrote
    rewritten: bool                  # whether a rewrite actually changed it
    routing: dict[str, Any]          # RouteDecision.model_dump()
    result: AgentResult | None
