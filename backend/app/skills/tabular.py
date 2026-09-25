"""Schema inspection and read-only SQL, as skills.

These two tools are the only place model output becomes executable, which is why
the containment in ARCHITECTURE.md §6.1 sits in `app.ingestion.tabular_store`
rather than here: `validate_select()` parses the statement and requires exactly
one `SELECT`, and `read_only_connection()` opens the connection read-only with
external access disabled and the configuration locked.

The tool loop changes one thing about that arrangement. A rejected statement is
no longer a caught exception feeding a hand-rolled retry -- it is returned to the
model as the tool's result, so it can correct itself. The budget in
`BaseAgent.run` is what stops that becoming an open-ended loop.
"""

import logging
import re
from dataclasses import dataclass, field

import duckdb
from langchain.tools import tool

from app.agents.base import Citation
from app.skills.evidence import BUDGET_SPENT, current_evidence

logger = logging.getLogger(__name__)

# The corpus is 31 rows, so this is a guard against a runaway join rather than a
# paging strategy. Capping here rather than trusting a generated LIMIT means the
# cap holds even when the model omits one.
MAX_ROWS = 200

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

NO_TABLES_MESSAGE = (
    "There are no structured tables loaded, so I cannot answer a numeric "
    "question yet. Upload a CSV or Excel export and ask again."
)

STORE_BUSY_MESSAGE = (
    "The data store is being updated right now, so I cannot query it. "
    "Try again in a moment."
)


@dataclass(slots=True)
class SqlResult:
    """One successful query, kept so the answer can cite the SQL behind it."""

    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)


@tool
async def describe_tables() -> str:
    """List the tables of project data available to query, and their columns.

    Call this before writing SQL. Column names are shown quoted exactly as they
    must be written.
    """
    import asyncio

    from app.config import get_settings
    from app.ingestion.tabular_store import list_tables

    evidence = current_evidence()
    if evidence and not evidence.start_call():
        return BUDGET_SPENT

    settings = get_settings()
    try:
        tables = await asyncio.to_thread(list_tables, settings)
    except duckdb.Error as exc:
        # An upload holds the store read-write; DuckDB allows one configuration
        # per path per process, so a query arriving mid-ingest cannot open it.
        logger.warning("tabular store unavailable", extra={"fields": {
            "error": f"{type(exc).__name__}: {exc}"}})
        return STORE_BUSY_MESSAGE

    if not tables:
        return NO_TABLES_MESSAGE
    return render_schema(tables)


@tool
async def run_sql(sql: str) -> str:
    """Run one read-only SELECT against the project data tables and return the rows.

    Exactly one SELECT statement, no trailing semicolon needed. If the statement
    is rejected or fails, the reason comes back here -- read it, correct the SQL
    and try once more rather than guessing.
    """
    import asyncio

    from app.config import get_settings
    from app.ingestion.tabular_store import SqlValidationError, validate_select

    evidence = current_evidence()
    if evidence and not evidence.start_call():
        return BUDGET_SPENT

    settings = get_settings()
    try:
        validated = validate_select(sql)
    except SqlValidationError as exc:
        logger.info("sql rejected", extra={"fields": {"sql": sql, "error": str(exc)}})
        return f"Rejected: {exc}"

    try:
        columns, rows = await asyncio.to_thread(_execute, validated, settings)
    except duckdb.Error as exc:
        logger.info("sql failed", extra={"fields": {
            "sql": validated, "error": f"{type(exc).__name__}: {exc}"}})
        return f"Query failed: {type(exc).__name__}: {exc}"

    if evidence:
        evidence.record([SqlResult(sql=validated, columns=columns, rows=rows)])
    logger.info("sql executed", extra={"fields": {
        "sql": validated, "row_count": len(rows)}})
    return render_rows(columns, rows)


def _execute(sql: str, settings) -> tuple[list[str], list[tuple]]:
    from app.ingestion.tabular_store import read_only_connection

    with read_only_connection(settings) as connection:
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        return columns, cursor.fetchmany(MAX_ROWS)


def render_schema(tables: dict[str, list[str]]) -> str:
    """`table(col, "odd-col", ...)`, quoting anything that is not an identifier.

    Showing the quotes is what stops the model writing `2026-Q3` unquoted, which
    DuckDB reads as an arithmetic expression rather than a column.
    """
    return "\n".join(
        f"{name}(" + ", ".join(
            column if _IDENTIFIER_RE.match(column) else f'"{column}"' for column in columns
        ) + ")"
        for name, columns in tables.items()
    )


def render_rows(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "(no rows matched)"
    header = " | ".join(columns)
    body = "\n".join(" | ".join(_cell(value) for value in row) for row in rows)
    capped = f"\n... capped at {MAX_ROWS} rows" if len(rows) == MAX_ROWS else ""
    return f"{header}\n{'-' * len(header)}\n{body}{capped}"


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return str(value)


def citations_for(results: list[SqlResult], settings) -> list[Citation]:
    """Cite the files behind every table the queries touched, plus the query.

    Generated SQL is inspectable and therefore good citation material, so it
    travels as the snippet: a reader can check the figure by reading the
    statement that produced it.
    """
    from app.ingestion.tabular_store import read_only_connection

    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        # The table list is read from the connection already open here rather
        # than through `list_tables`, which would open a second one: DuckDB
        # caches one instance per path per process, so a nested open would close
        # the shared instance out from under this block when it exited.
        with read_only_connection(settings) as connection:
            tables = [row[0] for row in connection.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()]
            for result in results:
                referenced = [name for name in tables
                              if re.search(rf"\b{re.escape(name)}\b", result.sql, re.I)]
                cited_rows = _rows_by_table(result.columns, result.rows)
                for name in referenced:
                    files = connection.execute(
                        f'SELECT DISTINCT "source_file" FROM "{name}"').fetchall()
                    location = f"{name}{cited_rows}" if cited_rows else name
                    for (file,) in files:
                        if not file:
                            continue
                        key = (str(file), location, result.sql)
                        if key in seen:
                            continue
                        seen.add(key)
                        citations.append(
                            Citation(source=str(file), location=location, snippet=result.sql))
    except duckdb.Error as exc:
        logger.warning("citation lookup failed", extra={"fields": {
            "error": f"{type(exc).__name__}: {exc}"}})
    return citations


def _rows_by_table(columns: list[str], rows: list[tuple]) -> str:
    """` row 7` / ` rows 7, 9` when the result carried its provenance."""
    if "source_row" not in columns:
        return ""
    position = columns.index("source_row")
    numbers = sorted({int(row[position]) for row in rows if row[position] is not None})
    if not numbers:
        return ""
    return f" row {numbers[0]}" if len(numbers) == 1 else \
        f" rows {', '.join(str(n) for n in numbers[:5])}"
