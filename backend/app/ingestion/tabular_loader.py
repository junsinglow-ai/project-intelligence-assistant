"""Load CSV/Excel files and normalise messy values (currency, dates, severity).

Cleaned tables go to the structured store for the Data Analysis Agent;
row-group summaries are also chunked for retrieval.

Everything here is driven by two tables rather than per-file special cases: a
synonym map from the spellings the sources use to canonical column names (F04),
and a map from canonical name to the parser that column needs. Adding a source
with different headers is then a data change, not a code change.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from app.ingestion.cleaning import (
    Parsed,
    collapse_ws,
    detect_date_style,
    detect_percent_scale,
    is_placeholder,
    normalise_status,
    null_or,
    parse_date,
    parse_money,
    parse_number,
    parse_percent,
)

# F04: the same field is spelled differently in the workbook and the CSV export.
_COLUMN_SYNONYMS = {
    "cost_code": "cost_code", "costcode": "cost_code",
    "cost_category": "category", "category": "category",
    "supplier": "vendor", "vendor": "vendor",
    "approved_budget": "budget", "budget": "budget",
    "actual_to_date": "actual", "actual": "actual",
    "variance": "variance",
    "pct_spent": "pct_spent", "spent": "pct_spent",
    "comments": "notes", "notes": "notes", "note": "note",
    "risk_id": "risk_id", "id": "risk_id",
    "risk_title": "title", "title": "title",
    "description": "description",
    "likelihood": "likelihood", "impact": "impact",
    "risk_score": "score", "score": "score",
    "risk_owner": "owner", "owner": "owner",
    "status": "status",
    "mitigation_action": "mitigation", "mitigation": "mitigation",
    "raised": "raised", "review_date": "review_date",
    "exposure": "exposure",
    "date": "date", "author": "author",
    "total": "total",
}

# Which parser each canonical column needs. Columns absent from this map are
# kept as cleaned text.
_COLUMN_PARSERS: dict[str, Callable[[Any], Parsed]] = {
    "budget": parse_money, "actual": parse_money, "variance": parse_money,
    "exposure": parse_money, "total": parse_money,
    "pct_spent": parse_percent,
    "score": parse_number,
    "status": normalise_status,
}
_DATE_COLUMNS = {"raised", "review_date", "date"}

_PERIOD_HEADER_RE = re.compile(r"^(?:(\d{4})[-\s]?Q([1-4])|Q(?:tr)?\s?([1-4])\s?(\d{2,4})|([1-4])Q(\d{2}))$", re.I)
_TOTAL_ROW_RE = re.compile(r"^(grand\s+)?(total|subtotal)$", re.I)


@dataclass(slots=True)
class CleanTable:
    """A normalised table plus enough provenance to cite any row."""

    name: str
    source: str
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    raw_rows: list[dict[str, str]] = field(default_factory=list)
    row_numbers: list[int] = field(default_factory=list)
    sheet: str | None = None
    header_row: int | None = None
    total_row: dict[str, Any] | None = None
    flaws: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)


def load_tabular(path: str | Path) -> list[CleanTable]:
    """Load every table in a CSV or Excel file."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return [_load_csv(path)]
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return _load_xlsx(path)
    raise ValueError(f"Unsupported tabular file: {path.name}")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def _load_csv(path: Path) -> CleanTable:
    """Read a CSV, tolerating a BOM, CRLF endings and rows of the wrong width.

    F09 covers the encoding and line endings; the width repair handles a row
    broken by an unquoted comma inside a free-text field.
    """
    # utf-8-sig strips the BOM; newline="" lets csv handle CRLF itself (F09).
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(c.strip() for c in row)]
    if not rows:
        return CleanTable(name="empty", source=path.name, columns=[])

    header, *data = rows
    columns = _canonical_columns(header)
    table = _build_table(columns, data, source=path.name, first_data_row=2)
    if path.read_bytes()[:3] == b"\xef\xbb\xbf":
        table.flaws.add("F09")
    return table


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------


def _load_xlsx(path: Path) -> list[CleanTable]:
    """Read every sheet, finding the real header row behind title banners (M04)."""
    import openpyxl

    workbook = openpyxl.load_workbook(path, data_only=True)
    tables: list[CleanTable] = []

    for sheet in workbook.worksheets:
        grid = [list(row) for row in sheet.iter_rows(values_only=True)]
        grid = _fill_merged(sheet, grid)
        # M09: trailing blank rows and the empty column. Original row numbers are
        # carried through, so a citation points at the row a reader would find in
        # the spreadsheet rather than at a post-trim index.
        grid, source_rows = _trim_empty(grid)
        if not grid:
            continue

        header_index = _find_header_row(grid)
        header = [("" if c is None else str(c)) for c in grid[header_index]]
        columns = _canonical_columns(header)
        table = _build_table(
            columns, grid[header_index + 1 :], source=path.name,
            first_data_row=source_rows[header_index + 1] if header_index + 1 < len(source_rows) else 1,
            sheet=sheet.title, source_rows=source_rows[header_index + 1 :],
        )
        table.header_row = source_rows[header_index]
        if header_index > 0:
            table.flaws.add("M04")
        if sheet.merged_cells.ranges:
            table.flaws.add("M04")
        tables.append(table)

    return tables


def _fill_merged(sheet: Any, grid: list[list[Any]]) -> list[list[Any]]:
    """Forward-fill merged ranges so a merged label is present on every cell (M04)."""
    for merged in sheet.merged_cells.ranges:
        top_left = grid[merged.min_row - 1][merged.min_col - 1]
        for row in range(merged.min_row - 1, merged.max_row):
            for column in range(merged.min_col - 1, merged.max_col):
                if row < len(grid) and column < len(grid[row]):
                    grid[row][column] = top_left
    return grid


def _trim_empty(grid: list[list[Any]]) -> tuple[list[list[Any]], list[int]]:
    """Drop wholly empty rows and columns, keeping original row numbers (M09).

    The used range runs past the data, so row counts and aggregates are wrong
    unless the padding is removed first. The returned row numbers are 1-based
    positions in the original sheet.
    """
    def blank(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    numbered = [(index, row) for index, row in enumerate(grid, 1) if not all(blank(c) for c in row)]
    if not numbered:
        return [], []
    rows = [row for _, row in numbered]
    width = max(len(row) for row in rows)
    keep = [i for i in range(width) if not all(blank(row[i]) if i < len(row) else True for row in rows)]
    return ([[row[i] if i < len(row) else None for i in keep] for row in rows],
            [index for index, _ in numbered])


def _find_header_row(grid: list[list[Any]]) -> int:
    """Pick the row that looks most like field names, not the title banner.

    Scored rather than assumed: the financial workbook puts a merged title in
    rows 1-2 and an export stamp below it, so the real header is row 4, while
    the other sheets start at row 1.
    """
    best_index, best_score = 0, -1.0
    for index, row in enumerate(grid[:10]):
        cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
        if len(cells) < 2:
            continue
        known = sum(1 for c in cells if _canonical_name(c) in _COLUMN_SYNONYMS.values())
        periods = sum(1 for c in cells if _PERIOD_HEADER_RE.match(c.strip()))
        score = known + periods + 0.1 * len(cells)
        if score > best_score:
            best_index, best_score = index, score
    return best_index


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------


_CURRENCY_SUFFIX_RE = re.compile(r"_(usd|gbp|eur)$")


def _canonical_name(header: str) -> str:
    """Normalise a header, dropping a trailing currency qualifier.

    `budget_usd` and `Approved Budget (USD)` are the same field as `budget`; the
    currency belongs to the value, not the column name (F04).
    """
    name = re.sub(r"[^a-z0-9]+", "_", collapse_ws(header).casefold()).strip("_")
    return _CURRENCY_SUFFIX_RE.sub("", name)


def _canonical_columns(header: Iterable[str]) -> list[str]:
    """Resolve header spellings to canonical names (F04), normalising periods (F01)."""
    columns: list[str] = []
    for index, raw in enumerate(header):
        name = _canonical_name(raw)
        if period := _normalise_period_header(collapse_ws(raw)):
            columns.append(period)
        elif name in _COLUMN_SYNONYMS:
            columns.append(_COLUMN_SYNONYMS[name])
        else:
            columns.append(name or f"column_{index + 1}")
    return columns


def _normalise_period_header(raw: str) -> str | None:
    """`2025-Q4`, `Q1 2026`, `2Q26` and `Qtr 3 2026` all name a quarter (F01)."""
    if not (match := _PERIOD_HEADER_RE.match(raw.strip())):
        return None
    if match.group(1):
        year, quarter = match.group(1), match.group(2)
    elif match.group(4):
        year, quarter = match.group(4), match.group(3)
    else:
        year, quarter = match.group(6), match.group(5)
    if len(year) == 2:
        year = f"20{year}"
    return f"{year}-Q{quarter}"


def _build_table(columns: list[str], data: list[list[Any]], source: str,
                 first_data_row: int, sheet: str | None = None,
                 source_rows: list[int] | None = None) -> CleanTable:
    """Coerce rows per cell, separating out any embedded total row (M10)."""
    table = CleanTable(name=_infer_table_name(columns, sheet), source=source,
                       columns=columns, sheet=sheet)

    present = [["" if c is None else c for c in row] for row in data]
    present = [row for row in present if any(str(c).strip() for c in row)]
    # Well-formed rows describe what each column normally holds, which is what
    # lets a broken row be put back together the right way.
    profile = _column_lengths([row for row in present if len(row) == len(columns)], columns)

    repaired: list[tuple[int, list[Any]]] = []
    for offset, row in enumerate(present):
        if len(row) != len(columns):
            row = _repair_width(row, columns, profile)
            table.flaws.add("F09")
        number = source_rows[offset] if source_rows and offset < len(source_rows) else first_data_row + offset
        repaired.append((number, row))

    date_style = _detect_style(columns, repaired)
    percent_scales = _detect_percent_scales(columns, repaired)

    for row_number, row in repaired:
        raw = {column: collapse_ws(value) for column, value in zip(columns, row)}
        if _is_total_row(raw):
            table.total_row = _coerce(raw, columns, date_style, percent_scales, table)
            table.flaws.add("M10")
            continue
        table.rows.append(_coerce(raw, columns, date_style, percent_scales, table))
        table.raw_rows.append(raw)
        table.row_numbers.append(row_number)

    return table


def _detect_style(columns: list[str], rows: list[tuple[int, list[Any]]]) -> str:
    """Resolve dd/mm vs mm/dd once per file from all its date columns (F01)."""
    samples: list[Any] = []
    for _, row in rows:
        for column, value in zip(columns, row):
            if column in _DATE_COLUMNS:
                samples.append(value)
    return detect_date_style(samples)


def _detect_percent_scales(columns: list[str], rows: list[tuple[int, list[Any]]]) -> dict[str, str]:
    """Decide the scale of each percentage column from the whole column (F06)."""
    scales: dict[str, str] = {}
    for index, column in enumerate(columns):
        if _COLUMN_PARSERS.get(column) is parse_percent:
            values = [row[index] if index < len(row) else "" for _, row in rows]
            scales[column] = detect_percent_scale(values)
    return scales


def _coerce(raw: dict[str, str], columns: list[str], date_style: str,
            percent_scales: dict[str, str], table: CleanTable) -> dict[str, Any]:
    """Parse each cell on its own merits, never by trusting a column type (M06)."""
    parsed: dict[str, Any] = {}
    for column in columns:
        value = raw.get(column, "")
        if column in _DATE_COLUMNS:
            result = parse_date(value, date_style)
        elif parser := _COLUMN_PARSERS.get(column):
            result = (parse_percent(value, percent_scales.get(column, "auto"))
                      if parser is parse_percent else parser(value))
        elif _PERIOD_HEADER_RE.match(column) or column == "total":
            result = parse_money(value)
        else:
            result = null_or(value)
            if result.value and is_placeholder(result.value):
                table.flaws.add("M07")
                parsed[f"{column}_is_placeholder"] = True

        parsed[column] = result.value
        if result.issue:
            table.flaws.add(result.issue)
            parsed[f"{column}_issue"] = result.issue
        # Keep the source notation whenever normalising changed it: retrieval
        # has to match a question that quotes the document ("$6.3M"), and BM25
        # can only match text that is actually present in the chunk.
        if value and str(result.value) != value:
            parsed[f"{column}_raw"] = value
    return parsed


def _is_total_row(raw: dict[str, str]) -> bool:
    """A TOTAL row has the shape of a data row but must not be aggregated (M10)."""
    return any(_TOTAL_ROW_RE.match(value.strip()) for value in raw.values() if value)


def _column_lengths(rows: list[list[Any]], columns: list[str]) -> dict[str, int]:
    """Longest value each column holds in the rows that are well-formed."""
    lengths: dict[str, int] = {}
    for row in rows:
        for column, value in zip(columns, row):
            lengths[column] = max(lengths.get(column, 0), len(collapse_ws(value)))
    return lengths


def _repair_width(row: list[Any], columns: list[str], profile: dict[str, int]) -> list[Any]:
    """Realign a row of the wrong width.

    A surplus field means an unquoted separator inside free text, so each
    adjacent pair is tried as a rejoin and the best-fitting result wins. Two
    signals decide it, because neither is sufficient alone: how many typed
    columns parse, and whether each cell is a plausible length for its column.
    Parsing alone ties -- rejoining the ID with the title shifts every later
    field by one and still leaves the dates and the amount parseable -- and the
    tie is broken by noticing that a 60-character risk ID is not credible when
    every other ID is five characters.
    """
    row = [("" if c is None else str(c)) for c in row]
    while len(row) > len(columns):
        candidates = [row[:i] + [f"{row[i]}, {row[i + 1]}".strip(", ")] + row[i + 2:]
                      for i in range(len(row) - 1)]
        row = max(candidates, key=lambda candidate: _alignment_score(candidate, columns, profile))
    return row + [""] * (len(columns) - len(row))


def _alignment_score(row: list[Any], columns: list[str], profile: dict[str, int]) -> int:
    """Score a candidate reading: parseable typed columns, less length anomalies."""
    parsed = 0
    anomalies = 0
    for column, value in zip(columns, row):
        value = collapse_ws(value)
        if not value:
            continue
        if column in _DATE_COLUMNS:
            parsed += bool(parse_date(value))
        elif parser := _COLUMN_PARSERS.get(column):
            parsed += bool(parser(value))
        if (typical := profile.get(column)) and len(value) > max(2 * typical, typical + 20):
            anomalies += 1
    return 10 * parsed - anomalies


def _infer_table_name(columns: list[str], sheet: str | None) -> str:
    """Name the table by what it contains, so both sources land in one place."""
    names = set(columns)
    if "risk_id" in names:
        return "risk_register"
    if "cost_code" in names or {"budget", "actual"} <= names:
        return "financials"
    if any(_PERIOD_HEADER_RE.match(c) for c in columns):
        return "financials_by_period"
    if {"date", "author"} <= names:
        return "notes"
    return _canonical_name(sheet or "table")
