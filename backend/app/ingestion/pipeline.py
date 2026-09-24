"""Ingestion orchestrator: files in, index and structured store out.

The one place that sequences load -> clean -> chunk -> embed -> upsert, and the
only caller of both stores. Everything upstream of it is pure: the loaders and
chunkers do not know where their output goes, which is what makes them testable
against the real corpus without a database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.chunking import Chunk, chunk_pdf, chunk_table
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.tabular_loader import CleanTable, load_tabular
from app.ingestion.tabular_store import write_tables
from app.llm.providers import get_embeddings
from app.retrieval import bm25, vectorstore

logger = logging.getLogger(__name__)

PDF_SUFFIXES = {".pdf"}
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xlsm"}


@dataclass(slots=True)
class IngestReport:
    """What one ingest run did, for the API response and the logs."""

    files: list[str] = field(default_factory=list)
    chunks_indexed: int = 0
    rows_loaded: dict[str, int] = field(default_factory=dict)
    flaws: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "chunks_indexed": self.chunks_indexed,
            "rows_loaded": self.rows_loaded,
            "flaws_handled": sorted(self.flaws),
            "skipped": self.skipped,
        }


def ingest_file(path: str | Path, settings: Settings | None = None,
                report: IngestReport | None = None) -> IngestReport:
    """Ingest one document into the index, and the structured store if tabular."""
    settings = settings or get_settings()
    report = report or IngestReport()
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in PDF_SUFFIXES:
        document = load_pdf(path)
        chunks = chunk_pdf(document, settings)
        report.flaws |= document.flaws
    elif suffix in TABULAR_SUFFIXES:
        tables: list[CleanTable] = load_tabular(path)
        chunks = [chunk for table in tables for chunk in chunk_table(table, settings)]
        for table in tables:
            report.flaws |= table.flaws
        # write_tables returns totals after the upsert, not rows written, so
        # assigning avoids double-counting the two financial exports.
        report.rows_loaded.update(write_tables(tables, settings))
    else:
        report.skipped.append(path.name)
        return report

    report.chunks_indexed += _index(chunks, path.name, settings)
    report.files.append(path.name)
    return report


def ingest_directory(directory: str | Path | None = None,
                     settings: Settings | None = None) -> IngestReport:
    """Ingest every supported document in a directory."""
    settings = settings or get_settings()
    directory = Path(directory or settings.data_raw_dir)
    report = IngestReport()

    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            ingest_file(path, settings, report)

    logger.info("ingest complete", extra={"fields": report.as_dict()})
    return report


def _index(chunks: list[Chunk], source: str, settings: Settings) -> int:
    """Embed and upsert, replacing whatever this source had indexed before."""
    if not chunks:
        return 0

    vectors = get_embeddings(settings).embed_documents([chunk.text for chunk in chunks])
    vectorstore.ensure_collection(settings, vector_size=len(vectors[0]))
    vectorstore.replace_source(source, settings)
    written = vectorstore.upsert(chunks, vectors, settings)

    # The keyword index is derived from the stored payloads, so it is stale the
    # moment they change.
    bm25.invalidate()
    logger.info("indexed", extra={"fields": {"source": source, "chunks": written}})
    return written
