"""DuckDB store for cleaned tables, and the guard on the query path.

Ingestion writes here; the Data Analysis Agent reads through
`read_only_connection()` and may only run SQL that `validate_select()` has
accepted. Both halves live in this module because they are one defence, and
because the third layer below can only be configured where the connection is
opened (DECISIONS.md D-007).

The defence is three layers, not two:

1. `validate_select()` parses the statement with DuckDB's own parser and
   requires exactly one `SELECT`.
2. The connection is opened `read_only=True`, so nothing can modify the
   database even if the first layer were wrong.
3. The connection is opened with `enable_external_access=False`. Layers 1 and 2
   do **not** cover the filesystem: `read_only=True` will happily run
   `SELECT * FROM read_csv_auto(...)` against any readable path, and DuckDB
   types that statement as a `SELECT`, so the validator passes it too. Since
   the SQL is written by a model reading untrusted document content, that is a
   real path and this is the layer that closes it. `lock_configuration=True`
   stops generated SQL from turning it back on.

DuckDB caches a database instance per path per process, so every connection to
the store must share these settings -- opening one with a different config
raises `ConnectionException`. That is why this is set here and not in the agent.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from app.config import Settings, get_settings
from app.ingestion.tabular_loader import CleanTable

logger = logging.getLogger(__name__)

# Columns carrying parser diagnostics rather than data; useful in the ingest
# report, but noise in a table an LLM is asked to write SQL against.
_DIAGNOSTIC_SUFFIXES = ("_issue", "_raw", "_is_placeholder")

# The key that identifies the same real-world row across exports. The workbook
# and the CSV describe the same eight cost lines, so without a key the second
# file to be ingested would either duplicate them or silently replace the first.
_NATURAL_KEYS = {
    "financials": ("cost_code",),
    "financials_by_period": ("category",),
    "risk_register": ("risk_id",),
    "notes": ("date", "author"),
}


def write_tables(tables: list[CleanTable], settings: Settings | None = None) -> dict[str, int]:
    """Upsert cleaned rows, returning each table's total row count afterwards.

    Tables are keyed by content-derived name, so the financials workbook sheet
    and the CSV export -- which describe the same eight cost lines -- land in
    one table rather than two near-duplicates. Within a table, rows are matched
    on a natural key so re-ingesting either export updates the same eight rows
    and `sum(budget)` stays right.
    """
    settings = settings or get_settings()
    path = Path(settings.tabular_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    with duckdb.connect(str(path)) as connection:
        for table in tables:
            if not table.rows:
                continue
            columns = [c for c in table.columns if not c.endswith(_DIAGNOSTIC_SUFFIXES)]
            declarations = ", ".join(
                f'"{c}" DOUBLE' if _is_numeric(table, c) else f'"{c}" VARCHAR' for c in columns
            )
            connection.execute(
                f'CREATE TABLE IF NOT EXISTS "{table.name}" '
                f'({declarations}, "source_file" VARCHAR, "source_row" INTEGER)'
            )

            # Replace only the rows this file is responsible for: the same key
            # from a later export updates the row rather than adding a second.
            keys = [k for k in _NATURAL_KEYS.get(table.name, ()) if k in columns]
            if keys:
                predicate = " AND ".join(f'"{k}" = ?' for k in keys)
                connection.executemany(
                    f'DELETE FROM "{table.name}" WHERE {predicate}',
                    [[_to_sql(row.get(k)) for k in keys] for row in table.rows],
                )
            else:
                connection.execute(f'DELETE FROM "{table.name}" WHERE "source_file" = ?', [table.source])

            placeholders = ", ".join("?" * (len(columns) + 2))
            connection.executemany(
                f'INSERT INTO "{table.name}" VALUES ({placeholders})',
                [
                    [*[_to_sql(row.get(c)) for c in columns], table.source, number]
                    for row, number in zip(table.rows, table.row_numbers)
                ],
            )
            totals[table.name] = connection.execute(
                f'SELECT count(*) FROM "{table.name}"'
            ).fetchone()[0]
            logger.info(
                "wrote table",
                extra={"fields": {"table": table.name, "written": len(table.rows),
                                  "total": totals[table.name], "source": table.source}},
            )

    return totals


def _is_numeric(table: CleanTable, column: str) -> bool:
    """Type a column from the values actually parsed, not from its name."""
    values = [row.get(column) for row in table.rows if row.get(column) is not None]
    return bool(values) and all(isinstance(v, (int, float, Decimal)) for v in values)


def _to_sql(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


class SqlValidationError(ValueError):
    """Generated SQL that is not a single, self-contained read-only SELECT."""


# A model that has read the documents writes this SQL, so the ceiling is a
# sanity bound on generated text rather than a limit anyone should reach.
_MAX_SQL_CHARS = 2000

# Table functions that read outside the database. `enable_external_access`
# already blocks these at the engine; naming them here turns a permission error
# deep in execution into a clear rejection the retry can act on.
_FORBIDDEN_CALLS = re.compile(
    r"\b(read_csv|read_csv_auto|read_parquet|parquet_scan|read_json|read_json_auto"
    r"|read_ndjson|read_text|read_blob|glob|sniff_csv|load_extension|getenv)\s*\(",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^\s*```(?:sql)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fences(sql: str) -> str:
    """Unwrap a ```sql block, which a chatty model emits despite instructions."""
    match = _FENCE_RE.match(sql)
    return match.group(1) if match else sql


def validate_select(sql: str) -> str:
    """Return `sql` if it is exactly one read-only SELECT, else raise.

    Parsing is delegated to DuckDB itself rather than a third-party SQL parser,
    so there is no dialect gap between what is validated and what is executed,
    and `StatementType` does the classification. See the module docstring for
    why this is not the whole defence.
    """
    text = _strip_fences(sql).strip()
    if not text:
        raise SqlValidationError("no statement was generated")
    if len(text) > _MAX_SQL_CHARS:
        raise SqlValidationError(f"statement exceeds {_MAX_SQL_CHARS} characters")

    try:
        statements = duckdb.extract_statements(text)
    except duckdb.Error as exc:
        # Also the path for a write inside a CTE: DuckDB's parser rejects
        # `WITH t AS (DELETE ... RETURNING *)` outright with "A CTE needs a SELECT".
        raise SqlValidationError(f"could not be parsed: {exc}") from exc

    if len(statements) != 1:
        # Catches a second statement hidden behind a comment: DuckDB strips
        # comments while parsing, so `SELECT 1 -- x\n; DROP TABLE t` is two.
        raise SqlValidationError(f"expected one statement, found {len(statements)}")
    # `==`, not `is`: DuckDB's StatementType is a pybind11 enum that hands back a
    # fresh object each access, so identity comparison is always False.
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SqlValidationError(
            f"only SELECT is allowed, found {statements[0].type.name}"
        )
    if match := _FORBIDDEN_CALLS.search(text):
        raise SqlValidationError(f"{match.group(1)}() reads outside the database")
    return text


def read_only_connection(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    """Open the store read-only and sealed off from the filesystem.

    Layers 2 and 3 of the D-007 defence. Every caller must use this, including
    `list_tables()`: DuckDB caches one instance per path per process, so a
    second connection opened with a different config raises.
    """
    settings = settings or get_settings()
    return duckdb.connect(
        settings.tabular_db_path,
        read_only=True,
        config={"enable_external_access": False, "lock_configuration": True},
    )


def list_tables(settings: Settings | None = None) -> dict[str, list[str]]:
    """Table names and their columns, for a text-to-SQL prompt to describe."""
    settings = settings or get_settings()
    if not Path(settings.tabular_db_path).exists():
        return {}
    with read_only_connection(settings) as connection:
        names = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
        return {
            name: [row[0] for row in connection.execute(f'DESCRIBE "{name}"').fetchall()]
            for name in names
        }
