"""Parse PDF status reports into text blocks with page/section metadata.

The reports are laid out as numbered sections (`1. Executive summary` through
`7. Focus for next period`) with tables inside them, so extraction is structural
rather than positional: sections become the chunk boundary and tables are pulled
out as tables.

Two properties of the corpus drive the design:

- **`extract_text()` mangles table rows.** A milestone row flattens to
  `Usage metering pipeline Hannah 03/20/2026 In-progress Lands with a 36h lag;
  batching` / `Okonkwo work to follow.`, splitting the owner across lines and
  interleaving it with the comment. `find_tables()` returns the same row intact,
  so tables are read structurally and their region is excluded from the
  narrative text -- otherwise the same content is indexed twice, once badly.
- **Tables continue across pages.** The Key Risks table starts on page 1 and
  carries on to page 2 under a repeated header, so a continuation is stitched
  back onto the block it belongs to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import pdfplumber

from app.ingestion.cleaning import Parsed, collapse_ws, detect_date_style, normalise_rag, parse_date

BlockKind = Literal["narrative", "table"]

_HEADING_RE = re.compile(r"^(\d{1,2})\.\s+(.{2,80})$")
_DIGITS_RE = re.compile(r"\d+")
# `Project status report | 2026-Q1 | 2026-03-31`
_SUBTITLE_RE = re.compile(r"^(.*?)\s*\|\s*(\S+)\s*\|\s*(.+)$")


@dataclass(slots=True)
class Block:
    """One section-worth of narrative, or one table."""

    kind: BlockKind
    section: str | None
    page_start: int
    page_end: int
    text: str = ""
    lines: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    @property
    def pages(self) -> str:
        return str(self.page_start) if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"


@dataclass(slots=True)
class PdfDocument:
    source: str
    blocks: list[Block]
    meta: dict[str, Any] = field(default_factory=dict)
    flaws: set[str] = field(default_factory=set)


def load_pdf(path: str | Path) -> PdfDocument:
    """Extract a status report into ordered narrative and table blocks."""
    path = Path(path)
    with pdfplumber.open(path) as pdf:
        pages = [_read_page(page, number) for number, page in enumerate(pdf.pages, 1)]

    furniture = _find_furniture([p["lines"] for p in pages])
    blocks: list[Block] = []
    section: str | None = None
    pending: list[str] = []
    pending_start = 1

    def flush() -> None:
        # Lines are kept rather than joined, because the metadata that identifies
        # a sign-off line is not known until the header has been read.
        nonlocal pending, pending_start
        if pending:
            blocks.append(Block("narrative", section, pending_start, page_number, lines=list(pending)))
        pending = []

    for page in pages:
        page_number = page["number"]
        for element in page["elements"]:
            if element["type"] == "line":
                text = element["text"]
                if _normalise_for_furniture(text) in furniture:
                    continue
                if match := _HEADING_RE.match(text):
                    flush()
                    section = f"{match.group(1)}. {collapse_ws(match.group(2))}"
                    pending_start = page_number
                    continue
                if not pending:
                    pending_start = page_number
                pending.append(text)
            else:
                flush()
                _append_table(blocks, element, section, page_number)
        flush()

    meta, blocks = _extract_document_meta(blocks, pages, furniture)
    blocks = _finalise_narrative(blocks, meta)
    document = PdfDocument(source=path.name, blocks=blocks, meta=meta)
    if furniture:
        document.flaws.add("F10")
    return document


def _read_page(page: Any, number: int) -> dict[str, Any]:
    """Split a page into table objects and the text lines that sit outside them."""
    tables = page.find_tables()
    boxes = [t.bbox for t in tables]

    elements: list[dict[str, Any]] = []
    lines: list[str] = []
    for line in page.extract_text_lines():
        text = collapse_ws(line["text"])
        if not text or _inside_any(line, boxes):
            continue
        lines.append(text)
        elements.append({"type": "line", "top": line["top"], "text": text})

    for table in tables:
        rows = [[collapse_ws(cell) for cell in row] for row in table.extract()]
        rows = [r for r in rows if any(r)]
        if rows:
            elements.append({"type": "table", "top": table.bbox[1], "header": rows[0], "rows": rows[1:]})

    elements.sort(key=lambda e: e["top"])
    return {"number": number, "elements": elements, "lines": lines}


def _inside_any(line: dict[str, Any], boxes: Iterable[tuple[float, float, float, float]]) -> bool:
    """True when a text line falls within a table's bounding box.

    This is what stops table content being emitted twice: once as clean rows and
    once as the flattened text that `extract_text` would produce.
    """
    middle = (line["top"] + line["bottom"]) / 2
    return any(top <= middle <= bottom for _, top, _, bottom in boxes)


def _normalise_for_furniture(text: str) -> str:
    """Blank out digits so `Page 1 of 2` and `Page 2 of 2` compare equal."""
    return _DIGITS_RE.sub("#", text.casefold())


def _find_furniture(pages_lines: list[list[str]]) -> set[str]:
    """Lines that recur on most pages are headers and footers, not content (F10).

    Detected rather than hardcoded, so the confidentiality banner, the project
    code line and the `Page N of M` footer all go without naming any of them --
    and a report with different furniture is handled the same way.
    """
    if len(pages_lines) < 2:
        return set()
    counts: dict[str, int] = {}
    for lines in pages_lines:
        for text in set(_normalise_for_furniture(line) for line in lines):
            counts[text] = counts.get(text, 0) + 1
    threshold = max(2, (len(pages_lines) + 1) // 2)
    return {text for text, count in counts.items() if count >= threshold}


def _append_table(blocks: list[Block], element: dict[str, Any], section: str | None, page: int) -> None:
    """Add a table, stitching it onto the previous one if it is a continuation.

    A table split by a page break repeats its header row. Without this, the Key
    Risks table becomes two chunks: R-001 on page 1, and an orphaned R-002/R-009
    on page 2 with no indication of what the columns mean.
    """
    header, rows = element["header"], element["rows"]
    if blocks and blocks[-1].kind == "table" and blocks[-1].header == header:
        blocks[-1].rows.extend(rows)
        blocks[-1].page_end = page
        return
    blocks.append(Block("table", section, page, page, header=header, rows=rows))


def _extract_document_meta(
    blocks: list[Block], pages: list[dict[str, Any]], furniture: set[str]
) -> tuple[dict[str, Any], list[Block]]:
    """Pull the report's own header into metadata.

    Tables before the first numbered heading are the report's key/value banner
    (`Report ID | PSR-2026-03 | Status | G`), not content, so they are lifted
    into metadata and dropped from the blocks.

    The three reports use three different templates and therefore three
    different key names for the same things -- `Status`, `Overall health`, and
    the memo which records it only inside `Re: ... 2026-Q2 status (Yellow)`. They
    are canonicalised here so that "the overall status in each reporting period"
    is answerable across all three without the caller knowing which template it
    is looking at.
    """
    meta: dict[str, Any] = {}
    kept: list[Block] = []
    seen_section = False

    for block in blocks:
        if block.section is not None:
            seen_section = True
        if not seen_section and block.kind == "table":
            for row in [block.header, *block.rows]:
                for index in range(0, len(row) - 1, 2):
                    key, value = row[index], row[index + 1]
                    if key and value:
                        meta[_meta_key(key)] = value
            continue
        kept.append(block)

    body = [line for line in pages[0]["lines"] if _normalise_for_furniture(line) not in furniture]
    for line in body[:6]:
        if match := _SUBTITLE_RE.match(line):
            _, period, reported = match.groups()
            meta.setdefault("period", period)
            meta.setdefault("report_date", reported)
            break
    for line in body[:4]:
        if not _SUBTITLE_RE.match(line) and not _HEADING_RE.match(line):
            meta.setdefault("title", line)
            break

    return _canonicalise_meta(meta), kept


def _finalise_narrative(blocks: list[Block], meta: dict[str, Any]) -> list[Block]:
    """Join narrative lines, dropping the ones that are just report identity.

    Every report ends with a sign-off (`Prepared by Arjun Mehta | PSR-2026-03 |
    issued 2026-03-31`) and opens with a title banner. Neither is content, and
    both are already held structurally in `meta`, so indexing them adds noise to
    the chunk that happens to sit next to them. They appear once rather than on
    every page, so furniture detection cannot see them; what identifies them is
    that they are built out of the document's own metadata values.
    """
    values = {str(v) for k, v in meta.items() if k != "title" and len(str(v)) >= 4}
    kept: list[Block] = []

    for block in blocks:
        if block.kind != "narrative":
            kept.append(block)
            continue

        lines = [line for line in block.lines if sum(v in line for v in values) < 2]
        text = collapse_ws(" ".join(lines))
        if not text:
            continue
        # A short preamble above the first heading is the title banner.
        if block.section is None and len(text) < 200 and any(
            str(v) in text for v in [*values, meta.get("title", "")] if v
        ):
            continue
        block.text = text
        kept.append(block)

    return kept


_META_SYNONYMS = {
    "reporting_period": "period",
    "overall_health": "status",
    "overall_status": "status",
    "ref": "report_id",
    "date": "report_date",
    "issued": "report_date",
    "author": "prepared_by",
    "from": "prepared_by",
}

_PERIOD_RE = re.compile(r"\b(\d{4})[-\s]?Q([1-4])\b|\bQ([1-4])\s?(\d{4})\b", re.I)
_PARENTHESISED_RE = re.compile(r"\(([^)]{2,12})\)")


def _meta_key(key: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return _META_SYNONYMS.get(key, key)


def _canonicalise_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Reconcile the three report templates onto one set of keys."""
    if "period" not in meta:
        for value in meta.values():
            if match := _PERIOD_RE.search(str(value)):
                year, quarter = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
                meta["period"] = f"{year}-Q{quarter}"
                break

    # The memo carries its status only as a parenthesised word inside `Re:`.
    if "status" not in meta:
        for value in meta.values():
            for candidate in _PARENTHESISED_RE.findall(str(value)):
                if normalise_rag(candidate):
                    meta["status"] = candidate
                    break
            if "status" in meta:
                break

    if raw_status := meta.get("status"):
        rag: Parsed = normalise_rag(raw_status)
        meta["status_raw"] = raw_status
        meta["status_normalised"] = rag.value
        if rag.issue:
            meta["status_issue"] = rag.issue

    if raw_date := meta.get("report_date"):
        parsed = parse_date(raw_date, detect_date_style([raw_date]))
        meta["report_date_raw"] = raw_date
        meta["report_date"] = parsed.value or raw_date

    return meta
