"""Qdrant access: collection lifecycle, upsert and vector search.

Runs embedded by default (a local directory, no server) and reaches a Qdrant
server when `QDRANT_URL` is set -- the same client either way, which is what
makes scaling out a configuration change rather than a rewrite (DECISIONS.md
D-003).

Embedded mode takes a lock on its directory, so only one process can hold the
index at a time. That is why `make ingest` expects the API not to be running,
and why the readiness probe checks the path rather than opening a second client.
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
# Embedded mode takes an exclusive lock on its directory, so two threads that
# both find `_client` unset and both construct a client leave one of them
# holding the lock and the rest raising. Concurrent *search* on one client is
# fine -- only the lazy construction has to be serialised.
_client_lock = threading.Lock()


def get_client(settings: Settings | None = None) -> QdrantClient:
    """Return the process-wide client, embedded or server-backed."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:          # re-check: another thread may have built it
                settings = settings or get_settings()
                if settings.qdrant_url:
                    _client = QdrantClient(url=settings.qdrant_url,
                                           api_key=settings.qdrant_api_key or None)
                else:
                    _client = QdrantClient(path=settings.qdrant_path)
    return _client


def close_client() -> None:
    """Release the embedded-mode directory lock."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def ensure_collection(settings: Settings | None = None, vector_size: int | None = None) -> str:
    """Create the collection if it is missing, sized from the embedding model.

    The size is probed from the configured embedder rather than hardcoded, so a
    model change cannot quietly produce a collection of the wrong width. The
    model name is already part of the collection name, so a change lands in a
    new, empty collection instead of corrupting the existing one.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    name = settings.collection_name

    if client.collection_exists(name):
        return name

    if vector_size is None:
        from app.llm.providers import get_embeddings

        vector_size = len(get_embeddings(settings).embed_query("dimension probe"))

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    # Filtering and per-source replacement both key off `source`. Embedded mode
    # scans payloads directly and warns if asked to build an index, so this is
    # only worth doing against a server.
    if settings.qdrant_url:
        client.create_payload_index(
            name, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
        )
    logger.info(
        "created collection",
        extra={"fields": {"collection": name, "vector_size": vector_size}},
    )
    return name


def replace_source(source: str, settings: Settings | None = None) -> None:
    """Drop every chunk from one file before re-indexing it.

    Deterministic IDs alone would overwrite chunks that still exist, but leave
    orphans behind when a file shrinks -- a deleted section's chunks would
    linger and keep being retrieved.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    if not client.collection_exists(settings.collection_name):
        return
    client.delete(
        collection_name=settings.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        ),
    )


def upsert(chunks: list[Chunk], vectors: list[list[float]], settings: Settings | None = None) -> int:
    """Write chunks and their vectors, keyed by deterministic chunk ID."""
    settings = settings or get_settings()
    if not chunks:
        return 0
    get_client(settings).upsert(
        collection_name=settings.collection_name,
        points=[
            models.PointStruct(id=chunk.id, vector=vector, payload={"text": chunk.text, **chunk.metadata})
            for chunk, vector in zip(chunks, vectors)
        ],
    )
    return len(chunks)


def scroll_all(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Every stored payload, used to rebuild the keyword index.

    Qdrant holds the chunk text, so BM25 can be reconstructed from it and there
    is no second copy of the corpus to keep in sync.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    if not client.collection_exists(settings.collection_name):
        return []

    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend({"id": str(point.id), **(point.payload or {})} for point in points)
        if offset is None:
            return payloads


def search(vector: list[float], limit: int, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Dense nearest-neighbour search, returned as plain payload dicts."""
    settings = settings or get_settings()
    client = get_client(settings)
    if not client.collection_exists(settings.collection_name):
        return []
    points = client.query_points(
        collection_name=settings.collection_name, query=vector, limit=limit, with_payload=True
    ).points
    return [{"id": str(p.id), "score": p.score, **(p.payload or {})} for p in points]


def count(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    client = get_client(settings)
    if not client.collection_exists(settings.collection_name):
        return 0
    return client.count(settings.collection_name).count


def drop_collection(settings: Settings | None = None) -> None:
    """Delete the collection outright; used by `make reindex`."""
    settings = settings or get_settings()
    client = get_client(settings)
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)


def iter_texts(payloads: Iterable[dict[str, Any]]) -> list[str]:
    return [str(p.get("text", "")) for p in payloads]


# Closing at exit avoids the client's finaliser running during interpreter
# shutdown, when the imports it needs are already gone.
atexit.register(close_client)
