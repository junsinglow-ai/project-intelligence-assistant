"""End-to-end ingestion into a real index.

These load the ONNX embedding model, so they are marked `slow`.
"""

import pytest

from app.ingestion.pipeline import ingest_directory, ingest_file
from app.ingestion.tabular_store import read_only_connection
from app.retrieval import bm25, vectorstore
from tests.conftest import DATA_RAW, RISK_REGISTER_CSV, STATUS_REPORT_Q1

pytestmark = pytest.mark.slow


def test_ingesting_the_corpus_indexes_every_file(isolated_stores):
    report = ingest_directory(DATA_RAW, isolated_stores)

    assert len(report.files) == 6
    assert report.chunks_indexed == vectorstore.count(isolated_stores) > 0
    assert report.skipped == []


def test_reingesting_replaces_chunks_rather_than_duplicating_them(isolated_stores):
    """Deterministic chunk IDs are what make the pipeline safe to re-run."""
    first = ingest_file(STATUS_REPORT_Q1, isolated_stores)
    after_first = vectorstore.count(isolated_stores)

    second = ingest_file(STATUS_REPORT_Q1, isolated_stores)

    assert second.chunks_indexed == first.chunks_indexed
    assert vectorstore.count(isolated_stores) == after_first


def test_the_ingest_report_names_the_flaws_it_handled(isolated_stores):
    """The report is what `data/MESSINESS.md` is checked against."""
    report = ingest_directory(DATA_RAW, isolated_stores)

    assert {"F01", "F06", "F09", "F10", "M02", "M04", "M06", "M07", "M10"} <= report.flaws


def test_both_financial_exports_land_in_one_table(isolated_stores):
    """Without a natural key the second export would duplicate or silently replace."""
    ingest_directory(DATA_RAW, isolated_stores)

    with read_only_connection(isolated_stores) as connection:
        rows, budget = connection.execute(
            "select count(*), sum(budget) from financials"
        ).fetchone()

    assert rows == 8
    assert budget == pytest.approx(6_315_000)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("select count(*) from risk_register where status = 'open'", 9),
        ("select sum(actual) from financials", 5_183_000),
        ("select sum(budget) - sum(actual) from financials", 1_132_000),
    ],
)
def test_aggregates_match_the_manifest_ground_truth(isolated_stores, query, expected):
    """These are only right because the TOTAL row was excluded (M10) and status
    matching is case-insensitive (F03)."""
    ingest_directory(DATA_RAW, isolated_stores)

    with read_only_connection(isolated_stores) as connection:
        assert connection.execute(query).fetchone()[0] == pytest.approx(expected)


def test_the_keyword_index_rebuilds_itself_from_the_vector_store(isolated_stores):
    """BM25 is in-memory, so after a restart it must reconstruct from payloads.

    Without this, a restarted process would serve vector-only results while
    still reporting RETRIEVAL_MODE=hybrid.
    """
    ingest_file(RISK_REGISTER_CSV, isolated_stores)
    bm25.invalidate()  # stands in for a process restart

    results = bm25.search("prompt logging data residency", limit=3, settings=isolated_stores)

    assert results
    assert any("R-003" in result["text"] for result in results)


def test_retrieval_finds_the_status_of_each_reporting_period(isolated_stores):
    """The status lives in each report's header table, not in its prose."""
    from app.retrieval.hybrid import retrieve

    ingest_directory(DATA_RAW, isolated_stores)
    results = retrieve("What was the overall status in each reporting period?", isolated_stores)

    text = " ".join(result.text for result in results)
    assert "2026-Q1" in text or "2026-Q2" in text or "2026-Q3" in text
    assert all(result.citation for result in results)


def test_an_unsupported_file_is_skipped_not_fatal(isolated_stores, tmp_path):
    stray = tmp_path / "notes.txt"
    stray.write_text("not a supported document")

    report = ingest_file(stray, isolated_stores)

    assert report.skipped == ["notes.txt"]
    assert report.chunks_indexed == 0


# --------------------------------------------------------------------------
# Upload endpoint
# --------------------------------------------------------------------------


def test_uploading_a_document_indexes_it(isolated_stores):
    from fastapi.testclient import TestClient

    from app.main import app

    with open(RISK_REGISTER_CSV, "rb") as handle:
        response = TestClient(app).post(
            "/v1/upload", files={"file": (RISK_REGISTER_CSV.name, handle, "text/csv")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["doc_type"] == "risk_register"
    assert body["chunks_indexed"] > 0
    assert vectorstore.count(isolated_stores) == body["chunks_indexed"]


def test_an_unsupported_upload_is_rejected_with_a_useful_message(isolated_stores):
    from fastapi.testclient import TestClient

    from app.main import app

    response = TestClient(app).post(
        "/v1/upload", files={"file": ("notes.txt", b"not a document", "text/plain")}
    )

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


def test_an_oversized_upload_is_rejected(isolated_stores, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    from app.main import app

    response = TestClient(app).post(
        "/v1/upload", files={"file": ("big.csv", b"a,b\n1,2\n", "text/csv")}
    )

    assert response.status_code == 413
