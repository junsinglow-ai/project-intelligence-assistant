"""Skills: evidence numbering, and the SQL guard reached through the tool.

These exercise the tools directly, without an agent around them, because the
properties that matter -- that two searches do not both start at [1], and that a
rejected statement comes back as text rather than an exception -- are properties
of the tools rather than of any model's behaviour.
"""

import duckdb
import pytest

from app.skills.documents import NO_MATCHES, search_documents
from app.skills.evidence import current_evidence, evidence_scope
from app.skills.tabular import (
    NO_TABLES_MESSAGE,
    SqlResult,
    citations_for,
    describe_tables,
    run_sql,
)

ROWS = [
    ("ENG-01", "Platform engineering", 2_400_000.0, 2_100_000.0, "financials.csv", 5),
    ("SEC-02", "Security", 900_000.0, 700_000.0, "financials.csv", 6),
]


@pytest.fixture
def tiny_store(isolated_stores):
    with duckdb.connect(isolated_stores.tabular_db_path) as connection:
        connection.execute(
            'CREATE TABLE financials ("cost_code" VARCHAR, "category" VARCHAR, '
            '"budget" DOUBLE, "actual" DOUBLE, "source_file" VARCHAR, "source_row" INTEGER)'
        )
        connection.executemany("INSERT INTO financials VALUES (?, ?, ?, ?, ?, ?)", ROWS)
    return isolated_stores


# -- evidence numbering ----------------------------------------------------


async def test_a_search_numbers_passages_from_one_and_records_them(stub_retrieval):
    stub_retrieval(("Vendor lock-in is the top risk.", "q1.pdf", "S5 Key risks p.2"))

    with evidence_scope() as evidence:
        rendered = await search_documents.ainvoke({"query": "risks"})

    assert rendered.startswith("[1] q1.pdf - S5 Key risks p.2")
    assert len(evidence.items) == 1
    assert evidence.tool_calls == 1


async def test_a_second_search_continues_the_numbering(stub_retrieval):
    """Restarting at [1] would make two passages indistinguishable in one answer."""
    stub_retrieval(("first passage", "a.pdf", "S1 p.1"), ("second passage", "b.pdf", "S2 p.3"))

    with evidence_scope() as evidence:
        await search_documents.ainvoke({"query": "one"})
        rendered = await search_documents.ainvoke({"query": "two"})

    assert rendered.startswith("[3] a.pdf")
    assert "[4] b.pdf" in rendered
    assert len(evidence.items) == 4
    assert evidence.tool_calls == 2


async def test_an_empty_search_invites_another_query_rather_than_refusing(monkeypatch):
    monkeypatch.setattr("app.retrieval.hybrid.retrieve", lambda *a, **k: [], raising=True)

    with evidence_scope() as evidence:
        assert await search_documents.ainvoke({"query": "nothing"}) == NO_MATCHES

    assert evidence.items == []


async def test_a_skill_outside_an_agent_run_still_works(stub_retrieval):
    """No collector in scope is not an error -- `make query` calls these directly."""
    stub_retrieval(("a passage", "a.pdf", "S1 p.1"))

    assert current_evidence() is None
    assert (await search_documents.ainvoke({"query": "x"})).startswith("[1] a.pdf")


# -- the SQL guard, reached through the tool -------------------------------


@pytest.mark.parametrize("statement", [
    "DROP TABLE financials",
    "SELECT * FROM financials; DROP TABLE financials",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
])
async def test_a_rejected_statement_comes_back_as_text_not_an_exception(tiny_store, statement):
    """The SQL containment is unchanged; what changes is that the model sees why."""
    with evidence_scope() as evidence:
        answer = await run_sql.ainvoke({"sql": statement})

    assert answer.startswith("Rejected:")
    assert evidence.items == []


async def test_a_failing_query_reports_the_error_for_the_model_to_correct(tiny_store):
    with evidence_scope() as evidence:
        answer = await run_sql.ainvoke({"sql": "SELECT nonexistent FROM financials"})

    assert answer.startswith("Query failed:")
    assert evidence.items == []


async def test_a_successful_query_returns_rows_and_records_the_sql(tiny_store):
    with evidence_scope() as evidence:
        answer = await run_sql.ainvoke({"sql": "SELECT sum(budget) AS total FROM financials"})

    assert "3,300,000" in answer.replace(".0", "")
    assert len(evidence.items) == 1
    assert isinstance(evidence.items[0], SqlResult)
    assert evidence.items[0].sql.startswith("SELECT sum(budget)")


async def test_describe_tables_names_every_column(tiny_store):
    rendered = await describe_tables.ainvoke({})

    assert "financials(" in rendered
    for column in ("cost_code", "category", "budget", "source_row"):
        assert column in rendered


async def test_describe_tables_on_an_empty_store_says_so(isolated_stores):
    assert await describe_tables.ainvoke({}) == NO_TABLES_MESSAGE


async def test_the_sql_travels_as_the_citation_snippet(tiny_store):
    sql = "SELECT cost_code, source_row FROM financials WHERE cost_code = 'ENG-01'"
    with evidence_scope() as evidence:
        await run_sql.ainvoke({"sql": sql})

    citations = citations_for(evidence.items, tiny_store)

    assert [c.source for c in citations] == ["financials.csv"]
    assert citations[0].location == "financials row 5"
    assert citations[0].snippet == sql
