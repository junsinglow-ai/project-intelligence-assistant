"""Assembling the chat graph from the agent registry.

    START -> rewrite -> route -> <one agent node> -> END

The node set is built from `list_agents()`, so registering an agent adds a node
with no edit here -- the registry stays the extension point it always was
(DECISIONS.md D-005). The failure rules ARCHITECTURE.md section 4.4 sets out are
enforced in these nodes rather than in any agent: routing that fails falls back
to `small_talk`, an agent that raises becomes a degraded result, and
`NotImplementedError` is re-raised so a half-built agent surfaces as a 501
instead of being reported as an answer.
"""

import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.base import AgentInput, AgentResult
from app.agents.registry import get_agent, list_agents
from app.agents.router import AGENT_FAILED_MESSAGE, FALLBACK_AGENT, RouterAgent
from app.graph.state import ChatState

logger = logging.getLogger(__name__)


async def _rewrite(state: ChatState) -> dict[str, Any]:
    """Resolve a follow-up into a standalone question before anything routes it.

    Kept rather than replaced by the checkpointer (D-007, D-005): the checkpoint
    holds graph state, but what reaches an agent is still one question, so the
    prompt does not grow with the conversation.
    """
    from app.config import get_settings
    from app.memory.session_store import rewrite_follow_up

    question = state["original_question"]
    if not state.get("needs_rewrite", True) or not state.get("history"):
        return {"question": question, "rewritten": False}

    rewritten = await rewrite_follow_up(question, state["history"], get_settings())
    return {"question": rewritten, "rewritten": rewritten != question}


async def _route(state: ChatState) -> dict[str, Any]:
    decision = await RouterAgent().route(_agent_input(state))
    return {"routing": decision.model_dump()}


def _agent_node(name: str):
    """One graph node per registered agent, all sharing the same failure rules."""

    async def node(state: ChatState) -> dict[str, Any]:
        try:
            result = await get_agent(name).run(_agent_input(state))
        except NotImplementedError:
            # Deliberate: a registered but unimplemented agent must surface
            # loudly rather than be reported as a degraded answer.
            raise
        except Exception as exc:
            # The agent failed, not the routing. Degrade rather than 500: the
            # request still reports which agent handled it.
            logger.exception("agent failed", extra={"fields": {
                "agent": name, "error": f"{type(exc).__name__}: {exc}"}})
            result = AgentResult(answer=AGENT_FAILED_MESSAGE, agent=name,
                                 metadata={"error": f"{type(exc).__name__}: {exc}"})

        result.metadata["routing"] = state["routing"]
        return {"result": result}

    return node


def _agent_input(state: ChatState) -> AgentInput:
    return AgentInput(question=state["question"], session_id=state["session_id"],
                      history=state.get("history", []))


def _chosen(names: set[str]):
    def choose(state: ChatState) -> str:
        agent = state["routing"]["agent"]
        # `route` already validated the name against the registry; this is the
        # narrower guarantee that it names a node in *this* compiled graph, which
        # can differ if an agent was registered after the graph was built.
        return agent if agent in names else FALLBACK_AGENT

    return choose


@lru_cache(maxsize=1)
def get_chat_graph():
    """Compile the graph once per process. `reset_chat_graph()` after registering."""
    names = [agent.name for agent in list_agents()]

    builder = StateGraph(ChatState)
    builder.add_node("rewrite", _rewrite)
    builder.add_node("route", _route)
    for name in names:
        builder.add_node(name, _agent_node(name))

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "route")
    builder.add_conditional_edges("route", _chosen(set(names)), names)
    for name in names:
        builder.add_edge(name, END)

    return builder.compile(checkpointer=_checkpointer())


def reset_chat_graph() -> None:
    """Drop the compiled graph so a newly registered agent gets a node."""
    get_chat_graph.cache_clear()


def _checkpointer():
    from app.graph.checkpointer import get_checkpointer

    return get_checkpointer()


async def _ready_checkpointer():
    """Prepare the store before compiling against it.

    Redis needs its indices created, and a Redis that cannot be reached degrades
    to the in-process saver -- which recompiles the graph -- so this runs before
    `get_chat_graph()`, not after.
    """
    from app.graph.checkpointer import ensure_checkpointer_ready

    await ensure_checkpointer_ready()


async def run_chat_graph(question: str, session_id: str,
                         history: list[dict[str, str]] | None = None,
                         needs_rewrite: bool = True) -> ChatState:
    """Run one question through the graph and return the final state.

    The state rather than just the result, because `/v1/chat` logs whether the
    question was rewritten, and that belongs to the graph rather than to the
    agent that answered.
    """
    await _ready_checkpointer()
    return await get_chat_graph().ainvoke(
        {
            "original_question": question,
            "question": question,
            "session_id": session_id,
            "history": history or [],
            "needs_rewrite": needs_rewrite,
        },
        # The thread is the conversation, so the checkpoint of one session's
        # graph runs never mixes with another's.
        config={"configurable": {"thread_id": session_id}},
    )
