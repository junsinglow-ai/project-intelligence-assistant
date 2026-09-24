"""Document search, as a skill.

`search_documents` is `app.retrieval.hybrid.retrieve` with two additions: the
passages are recorded into the run's evidence collector so `[n]` markers stay
resolvable, and the numbering it shows the model is the collector's, not a fresh
1..k per call. Without that, a second search would restart at `[1]` and the two
would be indistinguishable in the answer.
"""

import asyncio
import logging

from langchain.tools import tool

from app.skills.evidence import BUDGET_SPENT, current_evidence

logger = logging.getLogger(__name__)

NO_MATCHES = (
    "No passages matched that query. Try different wording or a broader query "
    "before concluding the documents do not cover it."
)


@tool
async def search_documents(query: str) -> str:
    """Search the indexed project documents for passages relevant to a query.

    Use this for anything in the narrative documents -- progress, milestones,
    issues, decisions, qualitative risk descriptions. Search again with
    different wording if the first results do not cover the question. Each
    passage is returned with a number; cite the numbers you used as [n].
    """
    from app.config import get_settings
    from app.retrieval import hybrid

    evidence = current_evidence()
    if evidence and not evidence.start_call():
        return BUDGET_SPENT

    settings = get_settings()
    # Retrieval is synchronous and CPU-bound -- ONNX embedding, BM25 scoring and
    # a cross-encoder pass -- so running it inline would block the event loop,
    # including /health. `to_thread` copies the context, so both the trace ID and
    # the evidence collector resolve inside the worker.
    results = await asyncio.to_thread(hybrid.retrieve, query, settings)

    if not results:
        logger.info("no context retrieved", extra={"fields": {"query": query}})
        return NO_MATCHES

    start = evidence.record(results) if evidence else 1
    logger.info("searched documents", extra={"fields": {
        "query": query, "retrieved": len(results), "first_index": start}})
    return render_passages(results, start)


def render_passages(results: list, start: int) -> str:
    """Number each chunk and label it with the citation built at ingest.

    `source` in every header is what makes the document-scope rule in the prompt
    enforceable, and the number is what lets the answer point back at a passage
    without a second model call.
    """
    return "\n\n".join(
        f"[{index}] {result.source} - {result.citation}\n{result.text}"
        for index, result in enumerate(results, start=start)
    )
