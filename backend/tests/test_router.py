"""Routing behaviour, including every way it is allowed to degrade.

The router must never take a request down, so most of these drive a failure
and assert that an answer still comes back.
"""

import pytest

from app.agents.base import AgentInput, AgentResult
from app.agents.data_analysis import DataAnalysisAgent
from app.agents.document_qa import DocumentQAAgent
from app.agents.router import FALLBACK_AGENT, MIN_CONFIDENCE, RouteDecision, RouterAgent
from app.agents.small_talk import SmallTalkAgent


@pytest.fixture
def stub_agents(monkeypatch):
    """Make every routable agent answer instantly, so only routing is under test.

    `small_talk` belongs here as much as the specialists do: it is the fallback,
    so it is what a degraded route lands on, and an unstubbed one would run its
    own model call and report the agent node's failure message instead.
    """
    async def answer(self, inp):
        return AgentResult(answer=f"handled by {self.name}", agent=self.name)

    monkeypatch.setattr(DocumentQAAgent, "run", answer)
    monkeypatch.setattr(DataAnalysisAgent, "run", answer)
    monkeypatch.setattr(SmallTalkAgent, "run", answer)


def ask(question="What is the approved budget?"):
    return AgentInput(question=question, session_id="s1")


async def test_decide_returns_the_agent_the_model_chose(scripted_llm):
    scripted_llm(RouteDecision(agent="data_analysis", confidence=0.9, reason="a total"))

    decision = await RouterAgent().decide(ask())

    assert decision.agent == "data_analysis"
    assert decision.confidence == 0.9


async def test_the_routing_prompt_lists_every_registered_agent(scripted_llm):
    model = scripted_llm(RouteDecision(agent="document_qa", confidence=0.8, reason="narrative"))

    await RouterAgent().decide(ask())

    prompt = model.calls[0].text
    for agent in (DocumentQAAgent, DataAnalysisAgent, SmallTalkAgent):
        assert agent.name in prompt
        assert agent.description in prompt


async def test_the_routing_prompt_omits_the_router_itself(scripted_llm):
    model = scripted_llm(RouteDecision(agent="document_qa", confidence=0.8, reason="r"))

    await RouterAgent().decide(ask())

    assert RouterAgent.description not in model.calls[0].text


async def test_a_confidence_below_the_floor_falls_back_to_small_talk(scripted_llm, stub_agents):
    scripted_llm(RouteDecision(agent="data_analysis",
                               confidence=MIN_CONFIDENCE - 0.01, reason="unsure"))

    result = await RouterAgent().run(ask())

    assert result.agent == FALLBACK_AGENT
    assert result.metadata["routing"]["confidence"] == 0.0
    assert "fallback" in result.metadata["routing"]["reason"]


async def test_an_unknown_agent_name_falls_back(scripted_llm, stub_agents):
    scripted_llm(RouteDecision(agent="sql_agent", confidence=0.99, reason="invented"))

    result = await RouterAgent().run(ask())

    assert result.agent == FALLBACK_AGENT


async def test_a_percentage_confidence_is_normalised(scripted_llm, stub_agents):
    """A small model readily answers 95 for 95%, which must not be discarded."""
    scripted_llm(RouteDecision(agent="data_analysis", confidence=95, reason="a total"))

    result = await RouterAgent().run(ask())

    assert result.agent == "data_analysis"
    assert result.metadata["routing"]["confidence"] == 0.95


async def test_an_llm_failure_routes_to_the_fallback_without_raising(scripted_llm, stub_agents):
    scripted_llm(RuntimeError("provider unreachable"))

    result = await RouterAgent().run(ask())

    assert result.agent == FALLBACK_AGENT
    assert result.metadata["routing"]["reason"].startswith("fallback:")
    assert "provider unreachable" in result.metadata["routing"]["reason"]


async def test_the_routing_decision_is_recorded_on_the_result(scripted_llm, stub_agents):
    scripted_llm(RouteDecision(agent="data_analysis", confidence=0.9, reason="a total"))

    routing = (await RouterAgent().run(ask())).metadata["routing"]

    assert routing == {"agent": "data_analysis", "confidence": 0.9, "reason": "a total"}


async def test_a_failing_agent_does_not_take_the_request_down(scripted_llm, monkeypatch):
    scripted_llm(RouteDecision(agent="document_qa", confidence=0.9, reason="narrative"))

    async def explode(self, inp):
        raise RuntimeError("duckdb is busy")

    monkeypatch.setattr(DocumentQAAgent, "run", explode)

    result = await RouterAgent().run(ask())

    assert result.agent == "document_qa"          # still reports who handled it
    assert "duckdb is busy" in result.metadata["error"]
    assert result.metadata["routing"]["agent"] == "document_qa"


async def test_the_router_is_never_a_routing_target(scripted_llm, stub_agents):
    """`list_agents()` excludes the router, so choosing it is an unknown name."""
    scripted_llm(RouteDecision(agent="router", confidence=0.99, reason="self"))

    result = await RouterAgent().run(ask())

    assert result.agent == FALLBACK_AGENT
