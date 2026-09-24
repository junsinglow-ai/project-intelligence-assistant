"""Document Q&A Agent: answers from unstructured text with citations."""

import logging

from app.agents.base import AgentInput, AgentResult, BaseAgent, last_message
from app.agents.registry import register_agent
from app.llm.prompts import DOCUMENT_QA_SYSTEM
from app.skills.documents import search_documents
from app.skills.evidence import attribute

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = (
    "I could not find anything in the indexed documents that bears on that "
    "question. If the document should be there, upload it and ask again."
)


@register_agent
class DocumentQAAgent(BaseAgent):
    name = "document_qa"
    description = (
        "Answers questions from project status reports and other narrative documents: "
        "progress, milestones, issues, decisions, and qualitative risk descriptions."
    )
    skills = [search_documents]
    system_prompt = DOCUMENT_QA_SYSTEM

    async def finalise(self, inp: AgentInput, state: dict, evidence) -> AgentResult:
        # The fixed pipeline refused before spending a model call when retrieval
        # came back empty. A loop cannot do that -- the model is what decides to
        # search -- so the refusal moves here: whatever the model wrote, an
        # answer with no passages behind it is ungrounded and is replaced.
        if not evidence.items:
            logger.info("no context retrieved", extra={"fields": {
                "agent": self.name, "tool_calls": evidence.tool_calls}})
            return AgentResult(answer=NO_CONTEXT_MESSAGE, agent=self.name,
                               metadata={"retrieved": 0, "tool_calls": evidence.tool_calls})

        answer, citations, mode = attribute(last_message(state), evidence.items)

        logger.info("document_qa answered", extra={"fields": {
            "retrieved": len(evidence.items), "cited": len(citations),
            "citation_mode": mode, "tool_calls": evidence.tool_calls}})
        return AgentResult(answer=answer, agent=self.name, citations=citations,
                           metadata={"retrieved": len(evidence.items),
                                     "citation_mode": mode,
                                     "tool_calls": evidence.tool_calls})
