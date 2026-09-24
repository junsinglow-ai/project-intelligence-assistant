"""Data analysis: schema prompting, the query budget, and citations.

SQL is scripted rather than generated, so these assert the agent's behaviour
around the model, not the model's SQL ability. The agent is a tool loop now, so
a script is `describe_tables and/or run_sql, then answer`; `model.calls[n]` is
the turn *before* the nth tool result, and `model.calls[n + 1]` is the turn that
saw it. The final test runs that scripted SQL against the real corpus, so the
figures are the manifest's ground truth.
"""

import duckdb
import pytest

from app.agents.base import AgentInput
from app.agents.data_analysis import NO_RESULT_MESSAGE, DataAnalysisAgent
from app.skills.tabular import NO_TABLES_MESSAGE
from tests.fakes import tool_call

ROWS = [
    ("ENG-01", "Platform engineering", 2_400_000.0, 2_100_000.0, "financials.csv", 5),
    ("SEC-02", "Security", 900_000.0, 700_000.0, "financials.csv", 6),
]

TOTAL_SQL = "SELECT sum(budget) FROM financials"


@pytest.fixture
def tiny_store(isolated_stores):
    """A minimal real DuckDB store: fast, but exercises the true query path."""
    with duckdb.connect(isolated_stores.tabular_db_path) as connection:
        connection.execute(
            'CREATE TABLE financials ("cost_code" VARCHAR, "category" VARCHAR, '
            '"budget" DOUBLE, "actual" DOUBLE, "source_file" VARCHAR, "source_row" INTEGER)'
        )
        connection.executemany("INSERT INTO financials VALUES (?, ?, ?, ?, ?, ?)", ROWS)
    return isolated_stores


def ask(question="What is the total budget?"):
    return AgentInput(question=question, session_id="s1")


def query(sql=TOTAL_SQL):
    return tool_call("run_sql", sql=sql)


SCHEMA = tool_call("describe_tables")


async def test_an_empty_store_reports_that_rather_than_guessing(
        isolated_stores, scripted_llm):
    scripted_llm(SCHEMA, "There is no data, so I cannot say.")

    result = await DataAnalysisAgent().run(ask())

    assert result.answer == NO_TABLES_MESSAGE
    assert result.metadata["queries"] == 0
    assert result.citations == []


async def test_an_answer_with_no_successful_query_behind_it_is_replaced(
        tiny_store, scripted_llm):
    """The model must not state a figure it never queried for."""
    scripted_llm("The total budget is 9,999,999 USD.")

    result = await DataAnalysisAgent().run(ask())

    assert result.answer == NO_RESULT_MESSAGE
    assert result.metadata["queries"] == 0


async def test_the_schema_names_every_table_and_column(tiny_store, scripted_llm):
    model = scripted_llm(SCHEMA, query(), "The total budget is 3,300,000 USD.")

    await DataAnalysisAgent().run(ask())

    schema = model.calls[1].text                    # the turn that saw describe_tables
    assert "financials(" in schema
    for column in ("cost_code", "category", "budget", "source_row"):
        assert column in schema


async def test_awkward_column_names_are_shown_quoted(tiny_store, scripted_llm, monkeypatch):
    """Unquoted `2026-Q3` parses as arithmetic, so the rule has to be explicit."""
    monkeypatch.setattr("app.ingestion.tabular_store.list_tables",
                        lambda *a, **k: {"financials_by_period": ["category", "2026-Q3"]})
    model = scripted_llm(SCHEMA, query(), "one")

    await DataAnalysisAgent().run(ask())

    assert '"2026-Q3"' in model.calls[1].text                     # quoted in the schema
    assert "must be wrapped in double quotes" in model.calls[0].text   # and in the rules


async def test_rejected_sql_comes_back_for_the_model_to_correct(tiny_store, scripted_llm):
    """The retry is the model's now, driven by the tool's own rejection text."""
    model = scripted_llm(
        query("DROP TABLE financials"),
        query(TOTAL_SQL),
        "The total budget is 3,300,000 USD.",
    )

    result = await DataAnalysisAgent().run(ask())

    assert result.metadata["queries"] == 1          # only the good one produced rows
    assert result.metadata["sql"] == TOTAL_SQL
    retry_prompt = model.calls[1].text
    assert "Rejected:" in retry_prompt
    assert "only SELECT is allowed" in retry_prompt


async def test_a_query_that_fails_at_execution_is_also_reported_back(tiny_store, scripted_llm):
    model = scripted_llm(
        query("SELECT no_such_column FROM financials"),
        query(TOTAL_SQL),
        "3,300,000 USD.",
    )

    result = await DataAnalysisAgent().run(ask())

    assert result.metadata["sql"] == TOTAL_SQL
    assert "no_such_column" in model.calls[1].text


async def test_the_query_budget_is_bounded(tiny_store, scripted_llm, monkeypatch):
    """A third attempt would cost minutes on-prem, so the budget declines it."""
    from app.config import get_settings

    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "2")
    get_settings.cache_clear()
    model = scripted_llm(
        query("DELETE FROM financials"),
        query("DROP TABLE financials"),
        query(TOTAL_SQL),                           # over budget, never runs
        "I could not query the data.",
    )

    result = await DataAnalysisAgent().run(ask())

    assert result.metadata["queries"] == 0
    assert result.answer == NO_RESULT_MESSAGE
    assert "used the tool budget" in model.calls[3].text
    get_settings.cache_clear()


async def test_the_generated_sql_travels_as_the_citation(tiny_store, scripted_llm):
    sql = "SELECT category, budget, source_row FROM financials"
    scripted_llm(query(sql), "Platform engineering leads.")

    result = await DataAnalysisAgent().run(ask())

    assert result.citations
    assert result.citations[0].source == "financials.csv"
    assert result.citations[0].snippet == sql
    assert result.citations[0].location == "financials rows 5, 6"


async def test_the_result_rows_reach_the_model(tiny_store, scripted_llm):
    model = scripted_llm(query("SELECT category, budget FROM financials"),
                         "Platform engineering is the largest at 2,400,000.")

    await DataAnalysisAgent().run(ask())

    rendered = model.calls[1].text
    assert "Platform engineering" in rendered
    assert "2,400,000" in rendered                  # rendered with thousands separators


async def test_a_query_matching_nothing_says_so(tiny_store, scripted_llm):
    model = scripted_llm(query("SELECT * FROM financials WHERE budget < 0"),
                         "Nothing in the data matched.")

    result = await DataAnalysisAgent().run(ask())

    assert result.metadata["row_count"] == 0
    assert result.answer == "Nothing in the data matched."
    assert "(no rows matched)" in model.calls[1].text


@pytest.mark.slow
@pytest.mark.parametrize("question, sql, expected", [
    ("What is the approved budget?",
     "SELECT sum(budget) FROM financials", "6,315,000"),
    ("How much has been spent?",
     "SELECT sum(actual) FROM financials", "5,183,000"),
    ("Are we over or under budget?",
     "SELECT sum(budget) - sum(actual) FROM financials", "1,132,000"),
    ("Which cost category is largest?",
     "SELECT category, budget FROM financials ORDER BY budget DESC LIMIT 1",
     "Platform engineering"),
    ("How many risks are open?",
     "SELECT count(*) FROM risk_register WHERE lower(status) = 'open'", "9"),
    ("Which risk has the highest exposure?",
     "SELECT risk_id, owner, exposure FROM risk_register ORDER BY exposure DESC LIMIT 1",
     "Yusuf Karim"),
])
async def test_the_manifest_figures_come_back_through_the_agent(
        isolated_stores, scripted_llm, question, sql, expected):
    """Scripted SQL, real corpus: deterministic, and still end-to-end."""
    from app.ingestion.pipeline import ingest_directory

    ingest_directory(settings=isolated_stores)
    model = scripted_llm(query(sql), "Answered.")

    result = await DataAnalysisAgent().run(ask(question))

    assert result.metadata["sql"] == sql
    assert expected in model.calls[1].text, "the figure must reach the model"
    assert result.citations
