"""Hybrid retrieval: keyword (BM25) + vector search, fused.

RETRIEVAL_MODE=naive falls back to vector-only for the RAGAS baseline.

Fusion is Reciprocal Rank Fusion: each result scores `1 / (rrf_k + rank)` in
every list it appears in, and the scores add up. It uses only the rank, never
the underlying score, so a cosine similarity and a BM25 score never have to be
made comparable and there is no weight to tune. Implementing it here rather than
relying on a store's built-in hybrid keeps the naive baseline honest -- the two
modes differ only in whether the keyword list participates (DECISIONS.md D-004).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.llm.providers import get_embeddings
from app.retrieval import bm25, reranker, vectorstore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Retrieved:
    """A chunk returned by retrieval, with how it was found."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def citation(self) -> str:
        return str(self.metadata.get("citation", ""))


def retrieve(question: str, settings: Settings | None = None,
             limit: int | None = None) -> list[Retrieved]:
    """Retrieve chunks for a question, honouring the configured retrieval mode."""
    settings = settings or get_settings()
    limit = limit or settings.top_k

    dense = vectorstore.search(
        get_embeddings(settings).embed_query(question), limit=limit, settings=settings
    )

    if settings.retrieval_mode == "naive":
        results = [_to_retrieved(hit, hit["score"]) for hit in dense]
    else:
        keyword = bm25.search(question, limit=limit, settings=settings)
        results = _reciprocal_rank_fusion([dense, keyword], settings.rrf_k)[:limit]
        logger.info(
            "hybrid retrieval",
            extra={"fields": {"dense": len(dense), "keyword": len(keyword), "fused": len(results)}},
        )

    if settings.rerank_enabled and results:
        results = reranker.rerank(question, results, settings)

    return results


def _reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], k: int) -> list[Retrieved]:
    """Combine ranked lists by rank alone, best first."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            identifier = str(hit["id"])
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank + 1)
            payloads.setdefault(identifier, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [_to_retrieved(payloads[identifier], score) for identifier, score in ordered]


def _to_retrieved(payload: dict[str, Any], score: float) -> Retrieved:
    metadata = {k: v for k, v in payload.items() if k not in {"id", "text", "score"}}
    return Retrieved(id=str(payload["id"]), text=str(payload.get("text", "")),
                     score=float(score), metadata=metadata)
