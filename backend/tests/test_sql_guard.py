"""The containment around model-generated SQL (ARCHITECTURE.md §6.1).

Layer 1 is `validate_select`. Layers 2 and 3 are the connection itself, and the
last test here is the one that matters: `read_only=True` alone does **not** stop
a SELECT reading the filesystem, and DuckDB types such a statement as a SELECT,
so neither of the first two layers catches it.
"""

import duckdb
import pytest

from app.ingestion.tabular_store import SqlValidationError, read_only_connection, validate_select


@pytest.mark.parametrize("sql", [
    "SELECT sum(budget) FROM financials",
    "select count(*) from risk_register where lower(status) = 'open'",
    "WITH totals AS (SELECT sum(budget) b FROM financials) SELECT b FROM totals",
    'SELECT category, "2026-Q3" FROM financials_by_period',
    "SELECT r.risk_id, f.category FROM risk_register r JOIN financials f ON true",
    "SELECT risk_id FROM risk_register ORDER BY exposure DESC LIMIT 1",
    "SELECT 1 /* a comment */",
])
def test_a_single_read_only_select_is_accepted(sql):
    assert validate_select(sql)


@pytest.mark.parametrize("sql, reason", [
    ("SELECT 1; DROP TABLE financials", "found 2"),
    ("SELECT 1 -- hide\n; DROP TABLE financials", "found 2"),
    ("DELETE FROM financials", "DELETE"),
    ("UPDATE financials SET budget = 0", "UPDATE"),
    ("INSERT INTO financials VALUES (1)", "INSERT"),
    ("CREATE TABLE t (a int)", "CREATE"),
    ("DROP TABLE financials", "DROP"),
    ("COPY (SELECT 1) TO '/tmp/exfiltrated.csv'", "COPY"),
    ("ATTACH '/tmp/other.db' AS other", "ATTACH"),
    ("INSTALL httpfs", "LOAD"),
    ("SET memory_limit = '1GB'", "SET"),
    ("CALL pragma_version()", "CALL"),
    ("SELECT * FROM read_csv_auto('/etc/passwd')", "outside the database"),
    ("SELECT * FROM read_text('/etc/passwd')", "outside the database"),
    ("", "no statement"),
    ("   ", "no statement"),
    ("this is not sql", "could not be parsed"),
])
def test_anything_that_is_not_one_read_only_select_is_rejected(sql, reason):
    with pytest.raises(SqlValidationError, match=reason):
        validate_select(sql)


def test_a_write_hidden_in_a_cte_is_refused_by_the_parser():
    """DuckDB rejects this outright, so no explicit rule is needed for it."""
    with pytest.raises(SqlValidationError, match="A CTE needs a SELECT"):
        validate_select("WITH t AS (DELETE FROM financials RETURNING *) SELECT * FROM t")


def test_a_statement_that_is_merely_long_is_rejected():
    with pytest.raises(SqlValidationError, match="exceeds"):
        validate_select("SELECT " + ", ".join(["1"] * 900) + " FROM financials")


@pytest.mark.parametrize("fenced", [
    "```sql\nSELECT sum(budget) FROM financials\n```",
    "```\nSELECT sum(budget) FROM financials\n```",
])
def test_a_markdown_fence_from_a_chatty_model_is_unwrapped(fenced):
    assert validate_select(fenced) == "SELECT sum(budget) FROM financials"


def test_a_pragma_is_rewritten_into_a_select_and_therefore_accepted():
    """Documents a known limit: PRAGMA cannot be rejected by statement type.

    DuckDB rewrites `PRAGMA database_list` into a SELECT over a system function,
    so a rule keyed on StatementType would silently do nothing. It is read-only,
    and the connection is sealed off from the filesystem, so it is accepted.
    """
    assert validate_select("PRAGMA database_list")


def test_the_read_only_connection_cannot_reach_the_filesystem(isolated_stores, tmp_path):
    """The regression test for the gap `read_only=True` leaves open.

    Without `enable_external_access=False` this SELECT returns the file's
    contents: it is a genuine SELECT, so the validator passes it, and it reads
    the filesystem rather than the database, so read-only does not stop it.
    """
    secret = tmp_path / "secret.csv"
    secret.write_text("column\nclassified\n")

    with duckdb.connect(isolated_stores.tabular_db_path) as setup:
        setup.execute("CREATE TABLE financials (budget DOUBLE)")
        setup.execute("INSERT INTO financials VALUES (100)")

    with read_only_connection(isolated_stores) as connection:
        assert connection.execute("SELECT sum(budget) FROM financials").fetchone() == (100,)

        with pytest.raises(duckdb.Error):
            connection.execute(f"SELECT * FROM read_csv_auto('{secret}')").fetchall()

        with pytest.raises(duckdb.Error):
            connection.execute("SET enable_external_access = true")
