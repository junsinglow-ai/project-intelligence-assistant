"""Small talk: the agent that answers without reading anything.

Every other agent is tested on whether its answer is grounded. This one has no
skills, so what is under test is the opposite -- that it never produces a
citation, never calls a tool, and carries the instruction not to answer a
project question from memory.
"""

from app.agents.base import AgentInput
from app.agents.small_talk import DEFAULT_GREETING, SmallTalkAgent


def ask(question="Hi there!"):
    return AgentInput(question=question, session_id="s1")


async def test_a_greeting_is_answered_without_calling_a_tool(scripted_llm):
    model = scripted_llm("Hello. Ask me about the project's reports or financials.")

    result = await SmallTalkAgent().run(ask())

    assert result.answer.startswith("Hello.")
    assert result.agent == "small_talk"
    assert model.bound_tools == []
    assert result.metadata["tool_calls"] == 0


async def test_the_answer_carries_no_citations(scripted_llm):
    """Nothing was read, so there is nothing to cite -- and the log says so."""
    scripted_llm("Happy to help.")

    result = await SmallTalkAgent().run(ask("thanks!"))

    assert result.citations == []
    assert result.metadata["grounded"] is False
    # `citation_mode` is the base class's marker-attribution verdict. This agent
    # never attributes, so reporting one would read as a model that forgot its
    # markers rather than an answer with nothing behind it.
    assert "citation_mode" not in result.metadata


async def test_a_stray_marker_does_not_become_a_phantom_citation(scripted_llm):
    """`[1]` here refers to nothing: no skill ran, so no passage was numbered."""
    scripted_llm("Hello [1].")

    result = await SmallTalkAgent().run(ask())

    assert result.citations == []
    assert result.answer == "Hello [1]."


async def test_an_empty_turn_falls_back_to_the_standing_greeting(scripted_llm):
    """An agent whose whole job is the first thing a user reads cannot say nothing."""
    scripted_llm("   ")

    result = await SmallTalkAgent().run(ask())

    assert result.answer == DEFAULT_GREETING


async def test_the_prompt_forbids_answering_project_questions_from_memory(scripted_llm):
    """The containment: with no skills, the system prompt is the whole guard."""
    model = scripted_llm("I cannot answer that here -- ask it as its own question.")

    await SmallTalkAgent().run(ask("What is the budget?"))

    prompt = model.calls[0].text
    assert "no documents in front of you" in prompt
    assert "Never state a fact about the project" in prompt


async def test_the_router_can_see_it():
    """The registry is the extension point: registering it is what routes to it."""
    from app.agents.registry import list_agents

    catalogue = {a.name: a.description for a in list_agents()}

    assert "small_talk" in catalogue
    assert "greetings" in catalogue["small_talk"].lower()


async def test_the_result_names_the_model_that_answered(scripted_llm, monkeypatch):
    """Read from the final message, because a fallback may have answered it."""
    from langchain_core.messages import AIMessage

    async def reply(*a, **k):
        return {"messages": [AIMessage(content="Hi.", response_metadata={
            "model_name": "models/gemini-3-flash-lite"})]}

    scripted_llm()
    monkeypatch.setattr("langchain.agents.create_agent",
                        lambda **k: type("Loop", (), {"ainvoke": staticmethod(reply)})())

    result = await SmallTalkAgent().run(ask())

    assert result.model == "gemini-3-flash-lite"
