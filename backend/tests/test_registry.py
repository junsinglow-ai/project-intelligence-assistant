import pytest

from app.agents.base import AgentInput, AgentResult, BaseAgent
from app.agents.registry import _REGISTRY, list_agents, register_agent


def test_core_agents_are_discovered():
    names = {a.name for a in list_agents(include_router=True)}
    assert {"router", "document_qa", "data_analysis", "small_talk"} <= names


def test_router_excluded_from_routable_agents():
    assert "router" not in {a.name for a in list_agents()}


def test_new_agent_registers_without_router_changes():
    @register_agent
    class DummyAgent(BaseAgent):
        name = "dummy_test_agent"
        description = "Test-only agent."

        async def run(self, inp: AgentInput) -> AgentResult:
            return AgentResult(answer="ok", agent=self.name)

    try:
        assert "dummy_test_agent" in {a.name for a in list_agents()}
    finally:
        _REGISTRY.pop("dummy_test_agent", None)


def test_agent_without_description_is_rejected():
    with pytest.raises(ValueError):
        @register_agent
        class Bad(BaseAgent):
            name = "bad"

            async def run(self, inp):  # pragma: no cover
                ...
