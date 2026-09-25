"""Small Talk Agent: greetings, thanks, and questions about the assistant.

The only agent with no skills, and the only one whose answer is not grounded in
a document -- which is exactly why its remit is drawn as narrowly as it is. It
exists so that "hi" does not become a search of the corpus that retrieves
nothing and is refused; it is not a general-purpose chat model wearing a badge.

It is also `router.FALLBACK_AGENT`, which is the heavier half of the job: an
unroutable question lands here, so the prompt has to decline one it cannot
answer without pretending the decline was the user's fault.
"""

import logging

from app.agents.base import AgentInput, AgentResult, BaseAgent, last_message
from app.agents.registry import register_agent
from app.llm.prompts import SMALL_TALK_SYSTEM

logger = logging.getLogger(__name__)

# Used when the model returns nothing usable. Every other agent has a standing
# message for its empty case; this is the one that fits an agent whose whole job
# is the first thing a user reads.
DEFAULT_GREETING = (
    "Hello. I answer questions about this project's status reports, financials "
    "and risk register -- ask me anything from those and I will cite what I used."
)


@register_agent
class SmallTalkAgent(BaseAgent):
    name = "small_talk"
    # Routing logic, not a comment: this is what the router sees. It names the
    # negative case too, because the expensive mistake is a project question
    # landing here rather than a greeting landing on document_qa.
    description = (
        "Handles greetings, thanks, sign-offs and questions about the assistant itself "
        "-- what it can do, which documents it covers, how to ask. Answers nothing "
        "about the project's content, figures, risks or history."
    )
    skills = []
    system_prompt = SMALL_TALK_SYSTEM

    async def finalise(self, inp: AgentInput, state: dict, evidence) -> AgentResult:
        # No skills, so there is no evidence and there are no citations. Saying
        # that explicitly matters: `BaseAgent.finalise` would run the answer
        # through `attribute()` and record `citation_mode: "inferred"`, which
        # reads in the logs as "the model forgot its markers" rather than "there
        # was nothing to cite".
        answer = last_message(state).strip() or DEFAULT_GREETING

        logger.info("small_talk answered", extra={"fields": {
            "agent": self.name, "grounded": False, "defaulted": not last_message(state).strip()}})
        return AgentResult(answer=answer, agent=self.name, citations=[],
                           metadata={"grounded": False, "tool_calls": evidence.tool_calls})
