"""The chat graph: rewrite, routing and the failure rules of section 5.4.

These drive the graph directly rather than through `/v1/chat`, so each node's
behaviour is asserted on its own. Routing and the agents are stubbed, because
what is under test is the wiring between them.
"""

import pytest

from app.agents.base import AgentResult
from app.agents.document_qa import DocumentQAAgent
from app.agents.data_analysis import DataAnalysisAgent
from app.agents.router import (
    AGENT_FAILED_MESSAGE,
    FALLBACK_AGENT,
    RouteDecision,
    RouterAgent,
)
from app.agents.small_talk import SmallTalkAgent
from app.graph.build import run_chat_graph

HISTORY = [{"role": "user", "content": "What is the budget?"},
           {"role": "assistant", "content": "6,315,000 USD."}]


@pytest.fixture
def routed(monkeypatch):
    """Route to a named agent without a model call."""
    def to(agent="document_qa", confidence=0.9):
        async def route(self, inp):
            return RouteDecision(agent=agent, confidence=confidence, reason="stubbed")

        monkeypatch.setattr(RouterAgent, "route", route)

    return to


@pytest.fixture
def answers(monkeypatch):
    """Make an agent answer, or fail, without a model call."""
    def install(cls=DocumentQAAgent, error=None, text="answered"):
        async def run(self, inp):
            if error:
                raise error
            return AgentResult(answer=f"{text}: {inp.question}", agent=self.name)

        monkeypatch.setattr(cls, "run", run)

    return install


# -- rewriting -------------------------------------------------------------


async def test_the_first_turn_of_a_session_is_not_rewritten(routed, answers, monkeypatch):
    """No history means nothing to resolve, so the call is skipped entirely."""
    called = []
    monkeypatch.setattr("app.memory.session_store.rewrite_follow_up",
                        lambda *a, **k: called.append(1))
    routed()
    answers()

    state = await run_chat_graph("What is the budget?", "s1")

    assert called == []
    assert state["rewritten"] is False
    assert state["result"].answer.endswith("What is the budget?")


async def test_a_follow_up_is_rewritten_before_it_is_routed(routed, answers, monkeypatch):
    async def rewrite(question, history, settings=None):
        return "What is the spend against the budget?"

    monkeypatch.setattr("app.memory.session_store.rewrite_follow_up", rewrite)
    routed()
    answers()

    state = await run_chat_graph("And the spend?", "s1", history=HISTORY)

    assert state["rewritten"] is True
    assert state["question"] == "What is the spend against the budget?"
    # The agent sees the standalone question, never the follow-up as typed.
    assert state["result"].answer.endswith("What is the spend against the budget?")
    assert state["original_question"] == "And the spend?"


async def test_a_caller_that_already_rewrote_is_not_charged_a_second_call(
        routed, answers, monkeypatch):
    """`RouterAgent.run` enters here; rewriting twice would be a wasted call."""
    called = []
    monkeypatch.setattr("app.memory.session_store.rewrite_follow_up",
                        lambda *a, **k: called.append(1))
    routed()
    answers()

    state = await run_chat_graph("Standalone?", "s1", history=HISTORY, needs_rewrite=False)

    assert called == []
    assert state["rewritten"] is False


# -- routing ---------------------------------------------------------------


async def test_the_decision_selects_the_agent_node(routed, answers):
    routed("data_analysis")
    answers(DataAnalysisAgent, text="from sql")

    state = await run_chat_graph("What is the total?", "s1")

    assert state["result"].agent == "data_analysis"
    assert state["result"].answer.startswith("from sql")


async def test_an_agent_with_no_node_falls_back_to_the_fallback_agent(routed, answers):
    """The graph's own guard: a name the compiled graph has no node for."""
    routed("ghost_agent")
    answers(SmallTalkAgent)

    state = await run_chat_graph("What happened?", "s1")

    assert state["result"].agent == FALLBACK_AGENT


async def test_the_routing_decision_is_recorded_on_the_result(routed, answers):
    routed("document_qa", confidence=0.77)
    answers()

    state = await run_chat_graph("What happened?", "s1")

    assert state["result"].metadata["routing"] == {
        "agent": "document_qa", "confidence": 0.77, "reason": "stubbed"}


# -- failure rules ---------------------------------------------------------


async def test_an_agent_that_raises_degrades_rather_than_failing_the_request(
        routed, answers):
    routed("document_qa")
    answers(DocumentQAAgent, error=RuntimeError("qdrant unreachable"))

    state = await run_chat_graph("What happened?", "s1")
    result = state["result"]

    assert result.answer == AGENT_FAILED_MESSAGE
    assert result.agent == "document_qa"            # still says who handled it
    assert result.metadata["error"] == "RuntimeError: qdrant unreachable"
    assert result.metadata["routing"]["agent"] == "document_qa"


async def test_an_unimplemented_agent_surfaces_loudly(routed, answers):
    """Deliberately not swallowed: it becomes a 501, not a degraded answer."""
    routed("document_qa")
    answers(DocumentQAAgent, error=NotImplementedError("half-built"))

    with pytest.raises(NotImplementedError):
        await run_chat_graph("What happened?", "s1")


# -- checkpointing ---------------------------------------------------------


async def test_each_session_checkpoints_under_its_own_thread(routed, answers):
    from app.graph.checkpointer import get_checkpointer

    routed()
    answers()
    await run_chat_graph("First?", "session-a")
    await run_chat_graph("Second?", "session-b")

    saver = get_checkpointer()
    assert saver.get_tuple({"configurable": {"thread_id": "session-a"}}) is not None
    assert saver.get_tuple({"configurable": {"thread_id": "session-b"}}) is not None


async def test_an_evicted_session_drops_its_checkpoint_thread(routed, answers):
    """The store is bounded because a client can invent IDs; the graph must match."""
    from app.graph.checkpointer import get_checkpointer
    from app.memory.session_store import SessionStore

    routed()
    answers()
    await run_chat_graph("First?", "old-session")
    saver = get_checkpointer()
    assert saver.get_tuple({"configurable": {"thread_id": "old-session"}}) is not None

    store = SessionStore(max_sessions=1)
    store.append("old-session", "q", "a", "document_qa")
    store.append("new-session", "q", "a", "document_qa")     # evicts old-session

    assert saver.get_tuple({"configurable": {"thread_id": "old-session"}}) is None
