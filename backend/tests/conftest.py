"""Shared fixtures.

Tests that touch the corpus run against the real files in `data/raw/` rather
than miniature fixtures: the point of this pipeline is that it copes with the
specific mess in those documents, and a tidied-up fixture would test the
opposite of what matters.
"""

import os
from pathlib import Path

import pytest

# Blanked at import time, before pytest collects the test modules, because
# several of them build `Settings` at module scope -- too early for a fixture
# to help. `Settings` reads the repository's `.env` as well as the
# environment, so without this a developer running on-prem locally sees every
# cloud-default assertion fail: an explicit `LLM_PROVIDER=ollama` in `.env`
# wins over the mode default the test is checking. Blank rather than deleted,
# since a deleted variable just falls through to the file; `_blank_means_unset`
# then drops it so the declared default applies. Keep in step with `_MODE_ENV`
# in test_config_modes.py.
for _name in (
    "DEPLOYMENT_MODE",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ROUTER_MODEL",
    "LLM_API_KEY",
    "LLM_TIMEOUT_S",
    "LLM_MAX_RETRIES",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "QDRANT_COLLECTION",
):
    os.environ[_name] = ""

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

STATUS_REPORT_Q1 = DATA_RAW / "Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf"
STATUS_MEMO_Q2 = DATA_RAW / "conduit_unified_llm_gateway-status-memo-jun2026.pdf"
STATUS_REPORT_Q3 = DATA_RAW / "CND-26_PSR_2026-09-18.pdf"
FINANCIALS_XLSX = DATA_RAW / "Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx"
FINANCIALS_CSV = DATA_RAW / "conduit_unified_llm_gateway_financials_export.csv"
RISK_REGISTER_CSV = DATA_RAW / "conduit_unified_llm_gateway_risk_register_v4.csv"


@pytest.fixture(scope="session", autouse=True)
def _require_corpus():
    if not DATA_RAW.exists() or not any(DATA_RAW.glob("*.pdf")):
        pytest.skip("data/raw corpus is absent; run `make data`", allow_module_level=True)


@pytest.fixture(autouse=True)
def _fresh_graph(monkeypatch):
    """A graph compiled in one test must not serve the next.

    `get_chat_graph` caches the compiled graph and the checkpointer is process
    wide, so a test that registers an agent or leaves checkpoint state behind
    would otherwise change what a later test runs.

    `REDIS_URL` is blanked for the same reason `isolated_stores` blanks
    `QDRANT_URL`: `.env` points it at the redis container so the stack and
    host-side tooling share one store, and the suite must neither depend on that
    container being up nor write into it. Blanking rather than deleting is what
    works -- `Settings` reads `.env` too, so an unset variable falls back to the
    file while an empty one is dropped by `_blank_means_unset`. Tests always
    checkpoint in-process; `test_checkpointer.py` covers the Redis wiring.
    """
    from app.config import get_settings
    from app.graph import checkpointer as checkpointer_module
    from app.graph.build import reset_chat_graph
    from app.graph.checkpointer import get_checkpointer

    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    reset_chat_graph()
    get_checkpointer.cache_clear()
    checkpointer_module._degraded = False
    yield
    reset_chat_graph()
    get_checkpointer.cache_clear()
    checkpointer_module._degraded = False
    get_settings.cache_clear()


@pytest.fixture
def isolated_stores(tmp_path, monkeypatch):
    """Point the vector store and DuckDB at a temporary location.

    `get_settings` is cached and the Qdrant client is a module singleton, so
    both are cleared around the test; otherwise one test's index leaks into the
    next and the embedded-mode directory lock is held against the wrong path.

    `QDRANT_URL` is blanked as well: `.env` points it at the Qdrant container so
    that host-side tooling shares the running stack's index, and a test suite
    that wrote into that shared server would both pollute it and depend on it
    being up. Blanking rather than deleting is what works here -- `Settings`
    reads `.env` too, so an unset variable would simply fall back to the file,
    while an empty one is dropped by `_blank_means_unset`. Tests always run
    embedded, against `tmp_path`.
    """
    from app.config import get_settings
    from app.retrieval import bm25, vectorstore

    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("TABULAR_DB_PATH", str(tmp_path / "tables.duckdb"))
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    get_settings.cache_clear()
    vectorstore.close_client()
    bm25.invalidate()

    yield get_settings()

    vectorstore.close_client()
    bm25.invalidate()
    get_settings.cache_clear()


@pytest.fixture
def scripted_llm(monkeypatch):
    """Replace the model factory with a scripted stand-in.

    Returns a factory: call it with the responses to queue, and it hands back
    the model so the test can assert on `.calls`. Agents import `get_llm`
    inside the method body, so patching the factory here reaches all of them.
    """
    from tests.fakes import ScriptedChatModel

    created: list[ScriptedChatModel] = []

    def install(*responses):
        model = ScriptedChatModel(responses=list(responses))
        created.append(model)
        monkeypatch.setattr("app.llm.providers.get_llm",
                            lambda *a, **k: model, raising=True)
        return model

    return install


@pytest.fixture
def stub_retrieval(monkeypatch):
    """Replace hybrid retrieval with a fixed result list.

    Returns a factory taking `(text, source, citation)` triples. The defaults
    mirror the two citation shapes the pipeline actually produces: a PDF
    section/page and a workbook sheet/row span.
    """
    from app.retrieval.hybrid import Retrieved

    def install(*chunks):
        results = [
            Retrieved(id=f"c{index}", text=text, score=1.0 - index / 100,
                      metadata={"source": source, "citation": citation})
            for index, (text, source, citation) in enumerate(chunks)
        ]
        monkeypatch.setattr("app.retrieval.hybrid.retrieve",
                            lambda *a, **k: list(results), raising=True)
        return results

    return install


@pytest.fixture
def session_store():
    """A store isolated from the process-wide singleton."""
    from app.memory.session_store import SessionStore, get_session_store

    get_session_store.cache_clear()
    yield SessionStore()
    get_session_store.cache_clear()
