"""Shared agent contract.

Every agent receives an AgentInput and returns an AgentResult, so the router,
API and logging treat all agents the same way.

An agent is a set of *skills*: tools its model calls in a loop until it can
answer (DECISIONS.md D-005). `run()` below is that loop, so a new agent is
usually just `name`, `description`, `skills` and a `system_prompt` -- no dispatch
code anywhere else changes, because `discover_agents()` finds it and the router
builds its catalogue from the descriptions.
"""

from abc import ABC
from typing import Any, ClassVar

from pydantic import BaseModel


class Citation(BaseModel):
    source: str                    # file name
    location: str | None = None    # page, section, sheet/row range
    snippet: str | None = None


class AgentInput(BaseModel):
    question: str                  # standalone question (follow-ups already rewritten)
    session_id: str
    history: list[dict[str, str]] = []


class AgentResult(BaseModel):
    answer: str
    agent: str
    citations: list[Citation] = []
    model: str | None = None       # the model that wrote the final turn
    metadata: dict = {}


class BaseAgent(ABC):
    """Subclass this and decorate with @register_agent to add a new agent."""

    name: ClassVar[str]
    description: ClassVar[str]     # used by the router to pick this agent
    skills: ClassVar[list] = []    # LangChain tools this agent's model may call
    system_prompt: ClassVar[str] = ""

    async def run(self, inp: AgentInput) -> AgentResult:
        """Run this agent's skills until its model stops calling them.

        The model and the prompt template are imported inside the body rather
        than at module scope, matching the rest of the package: the endpoints
        that never answer a question should not pull the retrieval and model
        stack onto their import path, and the test fixtures replace
        `app.llm.providers.get_llm` by patching that module attribute.
        """
        from langchain.agents import create_agent
        from app.config import get_settings
        from app.llm.providers import get_llms
        from app.llm.quota import QuotaAwareFallback, quota_scope
        from app.llm.usage import call_site
        from app.skills.evidence import EnforceBudget, evidence_scope

        settings = get_settings()
        # Each agent has its own chain (`AGENT_MODELS__<NAME>`, else LLM_MODEL),
        # preferred model first. The tool loop needs a
        # real chat model to bind its skills to, so the fallbacks go in as
        # middleware rather than as `RunnableWithFallbacks`, which has no
        # `bind_tools`. The middleware retries the *same* turn on the next
        # model, so a mid-loop quota failure resumes with the evidence already
        # collected rather than restarting the question. `QuotaAwareFallback`
        # additionally remembers, for this request only, which models answered
        # 429, so the loop stops re-walking an exhausted model on every one of
        # its generations.
        primary, *fallbacks = get_llms(settings.models_for(self.name), settings)
        loop = create_agent(
            model=primary,
            tools=list(self.skills),
            system_prompt=self.system_prompt or None,
            name=self.name,
            # EnforceBudget first: it is outermost, so a fallback model gets the
            # same tool-less request as the primary would have.
            middleware=[EnforceBudget(),
                        *([QuotaAwareFallback(*fallbacks)] if fallbacks else [])],
        )

        # The budget lives in the collector rather than in
        # `ToolCallLimitMiddleware`, which blocks the call without ending the
        # graph -- see the note on `Evidence` -- and `EnforceBudget` withdraws
        # the skills once it is spent, so the next turn has to answer.
        # `recursion_limit` is the backstop behind that; it raises, which the
        # graph turns into a degraded result rather than a 500.
        with (quota_scope(), call_site(self.name),
              evidence_scope(budget=settings.agent_max_tool_calls) as evidence):
            state = await loop.ainvoke(
                {"messages": [{"role": "user", "content": inp.question}]},
                config={"recursion_limit": 2 * settings.agent_max_tool_calls + 6},
            )
        result = await self.finalise(inp, state, evidence)
        # Set here rather than in `finalise`, so an override cannot forget it.
        # The chain means the configured primary is not necessarily the model
        # that answered; the final message says which one did.
        result.model = result.model or answering_model(state)
        return result

    async def finalise(self, inp: AgentInput, state: dict, evidence: Any) -> AgentResult:
        """Turn the finished loop into an AgentResult. Override to add metadata.

        Async because an agent may need one more piece of blocking work to build
        its citations -- `data_analysis` reads the source files behind the tables
        its queries touched -- and that belongs on a worker thread, not the loop.
        """
        from app.skills.evidence import attribute

        answer, citations, mode = attribute(last_message(state), evidence.items)
        return AgentResult(answer=answer, agent=self.name, citations=citations,
                           metadata={"evidence": len(evidence.items),
                                     "tool_calls": evidence.tool_calls,
                                     "citation_mode": mode})


def last_message(state: dict) -> str:
    """The text of the model's final turn, which is the answer.

    `.text` rather than `str(.content)`: a message's content is a list of typed
    blocks on models that return one -- Gemini 3.x returns text blocks carrying a
    reasoning signature -- and stringifying that list puts `[{'type': 'text', ...}]`
    in front of the user. `.text` concatenates the text blocks and passes a plain
    string through unchanged.
    """
    messages = state.get("messages") or []
    return messages[-1].text if messages else ""


def answering_model(state: dict) -> str | None:
    """The name of the model that produced the final turn, if it reported one.

    Read from the message rather than from settings, because with a model chain
    a quota failure hands the turn to a fallback. Both supported providers set
    `response_metadata["model_name"]`; Google may prefix it with `models/`.
    """
    messages = state.get("messages") or []
    if not messages:
        return None
    metadata = getattr(messages[-1], "response_metadata", None) or {}
    name = metadata.get("model_name") or metadata.get("model")
    return str(name).removeprefix("models/") if name else None
