"""Chunking strategies and metadata attachment. Rationale in ARCHITECTURE.md.

Chunks follow the structure of the sources rather than a fixed character window:

- **PDF narrative** is split at the numbered section headings the reports
  already use, so a chunk is a whole thought ("4. Financial position") instead
  of an arbitrary window that starts mid-sentence. A section longer than
  `chunk_size` is split with overlap, which in this corpus never happens.
- **PDF tables** are one chunk each, rendered as a pipe table, including a table
  that was stitched back together across a page break.
- **CSV/Excel tables** become row groups, bounded by both a row count and a
  character budget so a wide register and a narrow summary both chunk sensibly.

Every chunk carries a pre-rendered `citation`, because the format differs by
source type and the agents should not each reinvent it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.config import Settings, get_settings
from app.ingestion.pdf_loader import Block, PdfDocument
from app.ingestion.tabular_loader import CleanTable

# Stable namespace: chunk IDs must survive re-ingestion unchanged, so that
# re-running the pipeline replaces points rather than duplicating them.
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "project-intelligence-assistant/chunks")

# Above this many columns a pipe table stops being readable and the header stops
# being usefully close to its value, so rows are rendered as labelled fields.
_WIDE_TABLE_COLUMNS = 6


@dataclass(slots=True)
class Chunk:
    """One indexed unit: the text that gets embedded, plus how to cite it."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def citation(self) -> str:
        return self.metadata.get("citation", "")


def make_id(source: str, *parts: Any) -> str:
    """Deterministic chunk ID, so re-ingesting a file overwrites its chunks."""
    return str(uuid.uuid5(_NAMESPACE, "|".join([source, *(str(p) for p in parts)])))


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def chunk_pdf(document: PdfDocument, settings: Settings | None = None) -> list[Chunk]:
    """Turn a parsed status report into chunks."""
    settings = settings or get_settings()
    chunks: list[Chunk] = []

    if header := _render_document_header(document):
        chunks.append(
            Chunk(
                id=make_id(document.source, "header"),
                text=header,
                metadata=_pdf_metadata(document, None, "header", "1", "document header"),
            )
        )

    for ordinal, block in enumerate(document.blocks):
        if block.kind == "table":
            chunks.append(
                Chunk(
                    id=make_id(document.source, block.section, block.pages, ordinal),
                    text=_render_pdf_table(block),
                    metadata=_pdf_metadata(document, block.section, "table", block.pages,
                                           _pdf_citation(block)),
                )
            )
            continue

        for part, text in enumerate(_split(block.text, settings.chunk_size, settings.chunk_overlap)):
            chunks.append(
                Chunk(
                    id=make_id(document.source, block.section, block.pages, ordinal, part),
                    text=_with_heading(block.section, text),
                    metadata=_pdf_metadata(document, block.section, "narrative", block.pages,
                                           _pdf_citation(block)),
                )
            )

    return chunks


def _pdf_metadata(document: PdfDocument, section: str | None, kind: str, pages: str,
                  citation: str) -> dict[str, Any]:
    meta = {
        "source": document.source,
        "doc_type": "status_report",
        "kind": kind,
        "section": section,
        "pages": pages,
        "citation": citation,
    }
    # Period and status travel with every chunk so a question scoped to a
    # reporting period can be filtered or disambiguated without a second lookup.
    for key in ("period", "report_id", "report_date", "status_normalised", "title"):
        if value := document.meta.get(key):
            meta[key] = value
    if document.flaws:
        meta["flaws"] = sorted(document.flaws)
    return meta


def _pdf_citation(block: Block) -> str:
    """`§5 Key risks (p. 2)` / `§5 Key risks (pp. 1-2)`."""
    pages = f"p. {block.page_start}" if block.page_start == block.page_end else f"pp. {block.pages}"
    return f"§{block.section} ({pages})" if block.section else f"({pages})"


def _render_document_header(document: PdfDocument) -> str:
    """A small chunk of the report's own banner.

    The overall status and reporting period live in the header table, not in the
    prose, so without this chunk "what was the status in each period" has
    nothing to retrieve.
    """
    meta = document.meta
    if not meta:
        return ""
    lines = [meta.get("title", document.source)]
    for label, key in (("Reporting period", "period"), ("Report ID", "report_id"),
                       ("Report date", "report_date"), ("Prepared by", "prepared_by"),
                       ("Sponsor", "sponsor")):
        if value := meta.get(key):
            lines.append(f"{label}: {value}")
    if status := meta.get("status_normalised"):
        raw = meta.get("status_raw")
        lines.append(f"Overall status: {status}" + (f" (recorded as '{raw}')" if raw else ""))
    return "\n".join(lines)


def _render_pdf_table(block: Block) -> str:
    """Render as a pipe table under its section heading.

    The cell text is what the document says, uncleaned: this chunk is the
    evidence a citation points at, so it should read as the report reads. The
    normalised values live in the structured store instead.
    """
    lines = [f"{block.section} (table)"] if block.section else []
    lines.append(" | ".join(block.header))
    lines.extend(" | ".join(row) for row in block.rows)
    return "\n".join(lines)


def _with_heading(section: str | None, text: str) -> str:
    """Prefix the section heading so a chunk is self-describing out of context."""
    return f"{section}\n{text}" if section else text


def _split(text: str, size: int, overlap: int) -> Iterator[str]:
    """Split oversized text on sentence boundaries, with overlap."""
    if len(text) <= size:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text) and (boundary := text.rfind(". ", start, end)) > start:
            end = boundary + 1
        yield text[start:end].strip()
        if end >= len(text):
            return
        start = max(start + 1, end - overlap)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def chunk_table(table: CleanTable, settings: Settings | None = None) -> list[Chunk]:
    """Turn a cleaned table into row-group chunks."""
    settings = settings or get_settings()
    wide = len(table.columns) > _WIDE_TABLE_COLUMNS
    chunks: list[Chunk] = []

    for group, (rows, numbers) in enumerate(_row_groups(table, settings)):
        body = "\n".join(_render_row(row, table, wide) for row in rows)
        heading = _table_heading(table)
        text = f"{heading}\n{body}" if wide else f"{heading}\n{' | '.join(table.columns)}\n{body}"
        chunks.append(
            Chunk(
                id=make_id(table.source, table.sheet, table.name, group),
                text=text,
                metadata=_table_metadata(table, numbers),
            )
        )

    if table.total_row:
        chunks.append(
            Chunk(
                id=make_id(table.source, table.sheet, table.name, "total"),
                text=f"{_table_heading(table)}\nReported total row (excluded from the data rows to "
                     f"avoid double counting):\n{_render_row(table.total_row, table, wide)}",
                metadata=_table_metadata(table, []) | {"kind": "table_total"},
            )
        )

    return chunks


def _row_groups(table: CleanTable, settings: Settings) -> Iterator[tuple[list[dict], list[int]]]:
    """Group rows by row count and character budget, whichever binds first.

    A risk register row runs to several hundred characters while a by-period row
    is a handful, so a fixed row count alone would produce chunks of wildly
    different sizes.
    """
    rows: list[dict] = []
    numbers: list[int] = []
    budget = 0
    for row, number in zip(table.rows, table.row_numbers):
        rendered = len(_render_row(row, table, True))
        if rows and (len(rows) >= settings.table_rows_per_chunk or budget + rendered > settings.chunk_size):
            yield rows, numbers
            rows, numbers, budget = [], [], 0
        rows.append(row)
        numbers.append(number)
        budget += rendered
    if rows:
        yield rows, numbers


def _render_row(row: dict[str, Any], table: CleanTable, wide: bool) -> str:
    """Render one row, keeping the original notation alongside the parsed value.

    Retrieval has to match both: someone may ask using the figure as it appears
    in the source ("$6.3M") while the stored value is normalised, and BM25 in
    particular can only match text that is present.
    """
    values = []
    for column in table.columns:
        value = row.get(column)
        text = "not recorded" if value is None else str(value)
        raw = row.get(f"{column}_raw")
        if raw and raw != text:
            text = f"{text} (as written: {raw})"
        values.append((column, text))

    if not wide:
        return " | ".join(text for _, text in values)

    label = _row_label(row, table)
    fields = " | ".join(f"{column}: {text}" for column, text in values if column != "description")
    parts = [f"{label}: {fields}" if label else fields]
    if description := row.get("description"):
        parts.append(f"description: {description}")
    return "\n".join(parts)


def _row_label(row: dict[str, Any], table: CleanTable) -> str:
    for key in ("risk_id", "cost_code", "category", "date"):
        if value := row.get(key):
            return str(value)
    return ""


def _table_heading(table: CleanTable) -> str:
    sheet = f" [{table.sheet}]" if table.sheet else ""
    return f"{table.name.replace('_', ' ')} from {table.source}{sheet}"


def _table_metadata(table: CleanTable, numbers: list[int]) -> dict[str, Any]:
    # Location only, matching the PDF chunks: `Citation` in app/agents/base.py
    # keeps `source` and `location` as separate fields, and every chunk already
    # carries `source` in its metadata.
    sheet = f"{table.sheet}!" if table.sheet else ""
    if numbers:
        span = f"row {numbers[0]}" if len(numbers) == 1 else f"rows {numbers[0]}-{numbers[-1]}"
    else:
        span = "total row"
    meta: dict[str, Any] = {
        "source": table.source,
        "doc_type": table.name,
        "kind": "table",
        "section": table.name,
        "sheet": table.sheet,
        "row_numbers": numbers,
        "citation": f"{sheet}{span}",
    }
    if table.flaws:
        meta["flaws"] = sorted(table.flaws)
    return meta
