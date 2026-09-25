"""Data Analysis Agent: answers numeric/tabular questions over cleaned tables.

The SQL this agent generates is validated and executed by
`app.ingestion.tabular_store`, which holds all three layers of the containment
described in ARCHITECTURE.md §6.1. Nothing here executes model output
directly; the `run_sql` skill is the only path to the engine, and it goes through
`validate_select()` and a read-only connection.
"""

import asyncio
import logging

from langchain_core.messages import ToolMessage

from app.agents.base import AgentInput, AgentResult, BaseAgent, last_message
from app.agents.registry import register_agent
from app.llm.prompts import DATA_ANALYSIS_SYSTEM
from app.skills.tabular import (
    NO_TABLES_MESSAGE,
    STORE_BUSY_MESSAGE,
    SqlResult,
    citations_for,
    describe_tables,
    run_sql,
)

logger = logging.getLogger(__name__)

NO_RESULT_MESSAGE = (
    "I could not get an answer out of the project data for that question. "
    "Try naming the table or the figure you are after."
)

# A refusal the tools reported is more useful than a generic one, so it is
# passed through verbatim rather than flattened into NO_RESULT_MESSAGE.
_TOOL_REFUSALS = (NO_TABLES_MESSAGE, STORE_BUSY_MESSAGE)


@register_agent
class DataAnalysisAgent(BaseAgent):
    name = "data_analysis"
    description = (
        "Answers quantitative questions over financial summaries and risk registers: "
        "budgets, spend, variances, totals, comparisons, counts and filters."
    )
    skills = [describe_tables, run_sql]
    system_prompt = DATA_ANALYSIS_SYSTEM

    async def finalise(self, inp: AgentInput, state: dict, evidence) -> AgentResult:
        from app.config import get_settings

        queries = [item for item in evidence.items if isinstance(item, SqlResult)]

        if not queries:
            # No query succeeded, so any figure in the answer is invented. The
            # tools' own refusal is more informative than a generic one.
            answer = _reported_refusal(state) or NO_RESULT_MESSAGE
            logger.info("data_analysis had no result", extra={"fields": {
                "tool_calls": evidence.tool_calls}})
            return AgentResult(answer=answer, agent=self.name,
                               metadata={"queries": 0, "tool_calls": evidence.tool_calls})

        settings = get_settings()
        citations = await asyncio.to_thread(citations_for, queries, settings)

        last = queries[-1]
        logger.info("data_analysis answered", extra={"fields": {
            "sql": last.sql, "row_count": len(last.rows),
            "queries": len(queries), "tool_calls": evidence.tool_calls}})
        return AgentResult(answer=last_message(state), agent=self.name, citations=citations,
                           metadata={"sql": last.sql, "row_count": len(last.rows),
                                     "queries": len(queries),
                                     "tool_calls": evidence.tool_calls})


def _reported_refusal(state: dict) -> str | None:
    """The last thing a skill said, when it was one of its standing refusals."""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, ToolMessage) and str(message.content) in _TOOL_REFUSALS:
            return str(message.content)
    return None
