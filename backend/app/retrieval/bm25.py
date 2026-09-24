"""In-memory BM25 keyword index, rebuilt from the vector store.

`rank_bm25` has no persistence of its own. Rather than keep a sidecar file that
can drift out of step with the index, the corpus is read back from the Qdrant
payloads, which already hold every chunk's text. Qdrant stays the single source
of truth and there is nothing to reconcile after a restart -- without this, a
restarted container would silently serve vector-only results while still
reporting `RETRIEVAL_MODE=hybrid` (DECISIONS.md D-004).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.retrieval import vectorstore

logger = logging.getLogger(__name__)

# Digit-group separators are removed before tokenising so that a question asking
# for "6,315,000" matches a chunk that writes it as "6315000".
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
# Keeps identifiers such as `R-003`, `ENG-01` and `/v1/chat` in one piece.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_/\.]*")

_lock = threading.Lock()
_index: BM25Okapi | None = None
_documents: list[dict[str, Any]] = []


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(_THOUSANDS_RE.sub("", text.casefold()))


def invalidate() -> None:
    """Force a rebuild on the next search, after ingestion changes the corpus."""
    global _index, _documents
    with _lock:
        _index, _documents = None, []


def ensure_index(settings: Settings | None = None) -> None:
    """Build the index from stored payloads if it is not already built."""
    global _index, _documents
    with _lock:
        if _index is not None:
            return
        settings = settings or get_settings()
        _documents = vectorstore.scroll_all(settings)
        corpus = [tokenize(str(document.get("text", ""))) for document in _documents]
        _index = BM25Okapi(corpus) if corpus else None
        logger.info("built keyword index", extra={"fields": {"documents": len(_documents)}})


def search(query: str, limit: int, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Top-scoring chunks for the query terms, best first."""
    ensure_index(settings)
    if _index is None or not _documents:
        return []

    scores = _index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
    return [{**_documents[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]
