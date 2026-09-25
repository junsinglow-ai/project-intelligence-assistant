"""Router Agent: picks the specialised agent best suited to a question.

Routing options are built from the registry at runtime, so new agents are
picked up without editing this file.
"""

import logging

from pydantic import BaseModel

from app.agents.base import AgentInput, AgentResult, BaseAgent
from app.agents.registry import get_agent, list_agents, register_agent

logger = logging.getLogger(__name__)

# Where a route goes when the model cannot be trusted with the choice: an
# unknown agent name, confidence below the floor, or the routing call failing
# outright. `small_talk` declines cleanly and says what the assistant covers,
# rather than searching a corpus the question may have nothing to do with.
FALLBACK_AGENT = "small_talk"
MIN_CONFIDENCE = 0.5

# Shown to the user when the chosen agent raised. The request still reports
# which agent handled it, so the answer and the logs stay consistent.
AGENT_FAILED_MESSAGE = (
    "I could not complete that request. The details have been recorded against "
    "this request's trace ID."
)


class RouteDecision(BaseModel):
    agent: str
    confidence: float
    reason: str


def _normalised(decision: RouteDecision) -> RouteDecision:
    """Repair the shapes a small model produces without changing its choice.

    Structured output guarantees the JSON *shape* -- on Ollama the schema is
    enforced by constrained decoding -- but not its meaning. A small model asked
    for a 0-1 confidence readily answers `95`, meaning 95%, which would sail
    past `MIN_CONFIDENCE` and be recorded as nonsense. Repairing it here rather
    than constraining the field keeps a correct route that was merely reported
    on the wrong scale, where a validator would discard it.
    """
    confidence = decision.confidence
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    return RouteDecision(
        agent=decision.agent.strip().lower(),
        confidence=min(max(confidence, 0.0), 1.0),
        reason=decision.reason.strip()[:200],
    )


@register_agent
class RouterAgent(BaseAgent):
    name = "router"
    description = "Routes a question to the most suitable specialised agent."

    def agent_catalogue(self) -> str:
        """Agent names and descriptions, injected into the routing prompt."""
        return "\n".join(f"- {a.name}: {a.description}" for a in list_agents())

    async def decide(self, inp: AgentInput) -> RouteDecision:
        """Classify the question against the registry's catalogue."""
        from app.config import get_settings
        from app.llm.prompts import ROUTER_PROMPT
        from app.llm.providers import get_structured_llm
        from app.llm.usage import call_site

        settings = get_settings()
        routable = list_agents()
        chain = ROUTER_PROMPT | get_structured_llm(
            RouteDecision, settings.models_for(self.name), settings
        )
        # The question is already standalone, so the conversation adds nothing
        # here but tokens -- and prompt length is the on-prem latency lever.
        with call_site(self.name):
            decision = await chain.ainvoke({
                "catalogue": self.agent_catalogue(),
                "names": ", ".join(a.name for a in routable),
                "fallback": FALLBACK_AGENT,
                "question": inp.question,
            })
        return _normalised(decision)

    async def route(self, inp: AgentInput) -> RouteDecision:
        """`decide`, with the guarantee that it always returns a usable agent.

        This is the `route` node of the chat graph, and the one place the
        fallback rules live -- unknown name, confidence below the floor, or the
        routing call failing outright.
        """
        try:
            decision = await self.decide(inp)
            valid = {a.name for a in list_agents()}
            if decision.agent not in valid or decision.confidence < MIN_CONFIDENCE:
                decision = RouteDecision(
                    agent=FALLBACK_AGENT,
                    confidence=0.0,
                    reason="fallback: low confidence or unknown agent",
                )
        except NotImplementedError:
            raise
        except Exception as exc:  # routing must never take the request down
            decision = RouteDecision(agent=FALLBACK_AGENT, confidence=0.0, reason=f"fallback: {exc}")

        if decision.confidence == 0.0:
            # Louder than it looks: the fallback answers nothing from the corpus,
            # so a router that has quietly started falling back on everything --
            # a spent quota, an unreachable endpoint -- turns the whole system
            # into a polite deflection while every request still returns 200.
            logger.warning("routing fell back", extra={"fields": {
                "agent": decision.agent, "reason": decision.reason}})
        return decision

    async def run(self, inp: AgentInput) -> AgentResult:
        """Route and answer one already-standalone question.

        The dispatch this used to do is the chat graph's now, so this runs the
        graph from its `route` node -- `needs_rewrite=False`, because a caller
        holding an `AgentInput` has a standalone question already and rewriting
        it twice would be a second model call for nothing.
        """
        from app.graph.build import run_chat_graph

        state = await run_chat_graph(
            question=inp.question,
            session_id=inp.session_id,
            history=inp.history,
            needs_rewrite=False,
        )
        return state["result"]
