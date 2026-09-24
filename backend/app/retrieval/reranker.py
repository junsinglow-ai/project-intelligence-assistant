"""Re-rank retrieved chunks before they reach the LLM.

A local ONNX cross-encoder via FlashRank. Hybrid retrieval is tuned for recall:
it returns candidates that mention the right terms, in an order that reflects
two crude signals. A cross-encoder reads the question and the chunk together and
reorders them, which matters most on-prem, where the context window is small and
every extra token of prompt is measurable latency (DECISIONS.md D-005).

Kept local deliberately: a hosted reranker would be a second external call for
on-prem mode to disable, which would break the single-switch guarantee (D-010).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.retrieval.hybrid import Retrieved

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_ranker(model: str, cache_dir: str = "") -> Any:
    """Load and cache the cross-encoder; loading dominates per-call cost.

    `cache_dir` is part of the cache key so two caches cannot collide on one
    memoised ranker. Blank keeps FlashRank's own default of `/tmp`; the image
    overrides it, since weights baked into `/tmp` would sit in memory (D-009).
    """
    from flashrank import Ranker

    logger.info(
        "loading reranker",
        extra={"fields": {"rerank_model": model, "cache_dir": cache_dir or "default"}},
    )
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    return Ranker(model_name=model, **kwargs)


def rerank(question: str, results: list["Retrieved"], settings: Settings | None = None) -> list["Retrieved"]:
    """Reorder results by cross-encoder relevance and cut to `rerank_top_n`.

    Reranking never fails a query: if the model cannot be loaded the fused order
    is returned unchanged, trimmed, rather than the request erroring.
    """
    settings = settings or get_settings()
    if not results:
        return results

    try:
        from flashrank import RerankRequest

        ranker = get_ranker(settings.rerank_model, settings.rerank_cache_dir)
        passages = [{"id": index, "text": result.text} for index, result in enumerate(results)]
        ranked = ranker.rerank(RerankRequest(query=question, passages=passages))
    except Exception as exc:
        logger.warning("rerank skipped", extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}})
        return results[: settings.rerank_top_n]

    reordered: list["Retrieved"] = []
    for entry in ranked[: settings.rerank_top_n]:
        result = results[int(entry["id"])]
        result.score = float(entry.get("score", result.score))
        reordered.append(result)
    return reordered
