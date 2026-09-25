"""The /v1/chat and /v1/sessions contract, with the agents stubbed out."""

import pytest
from fastapi.testclient import TestClient

from app.agents.base import AgentResult, Citation
from app.agents.data_analysis import DataAnalysisAgent
from app.agents.router import RouteDecision, RouterAgent
from app.main import app

CITATION = Citation(source="q1.pdf", location="§4. Financial position (p. 1)", snippet="Budget...")


@pytest.fixture
def client(session_store, monkeypatch):
    """A client whose routing and agent answer instantly.

    The graph itself is left real -- rewrite, route and the agent node all run --
    because it is now the thing standing between the endpoint and an answer. Only
    the two steps that would reach a provider are stubbed, which is what keeps
    these hermetic; `test_graph.py` covers the graph's own behaviour.
    """
    async def route(self, inp):
        return RouteDecision(agent="data_analysis", confidence=0.9, reason="a total")

    async def answer(self, inp):
        return AgentResult(answer=f"Answered: {inp.question}", agent="data_analysis",
                           citations=[CITATION], model="gemini-3-flash")

    async def passthrough(question, history, settings=None):
        return question

    monkeypatch.setattr(RouterAgent, "route", route)
    monkeypatch.setattr(DataAnalysisAgent, "run", answer)
    # Rewriting has its own tests; stubbing it keeps these hermetic and stops
    # the second request in a conversation reaching for a real provider.
    monkeypatch.setattr("app.memory.session_store.rewrite_follow_up", passthrough)
    return TestClient(app)


def post(client, question="What is the approved budget?", session_id=None):
    payload = {"question": question}
    if session_id is not None:
        payload["session_id"] = session_id
    return client.post("/v1/chat", json=payload)


def test_an_answer_carries_the_agent_the_citations_and_a_trace_id(client):
    response = post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "data_analysis"
    assert body["model"] == "gemini-3-flash"
    assert body["citations"][0]["source"] == "q1.pdf"
    assert body["citations"][0]["location"] == "§4. Financial position (p. 1)"
    # The body's trace ID is the one a user can quote from the response header.
    assert body["trace_id"] == response.headers["x-trace-id"]


def test_a_session_id_is_minted_when_the_client_has_none(client):
    body = post(client).json()

    assert body["session_id"]
    assert post(client, session_id=body["session_id"]).json()["session_id"] == body["session_id"]


def test_the_conversation_accumulates_across_requests(client):
    session_id = post(client, "What is the approved budget?").json()["session_id"]
    post(client, "And the spend?", session_id=session_id)

    turns = client.get(f"/v1/sessions/{session_id}").json()["turns"]

    assert [turn["question"] for turn in turns] == ["What is the approved budget?",
                                                    "And the spend?"]
    assert turns[0]["agent"] == "data_analysis"


def test_an_unknown_but_well_formed_session_starts_a_new_conversation(client):
    """A client that kept an ID across a restart should keep working."""
    response = post(client, session_id="a1b2c3")

    assert response.status_code == 200
    assert response.json()["session_id"] == "a1b2c3"


def test_a_malformed_session_id_is_rejected(client):
    assert post(client, session_id="../../etc/passwd").status_code == 422
    assert client.get("/v1/sessions/not a session").status_code == 422


def test_an_empty_question_is_rejected_by_the_schema(client):
    assert post(client, "").status_code == 422
    assert post(client, "x" * 4001).status_code == 422


def test_an_unknown_session_transcript_is_a_404(client):
    assert client.get("/v1/sessions/deadbeef").status_code == 404


def test_a_trimmed_transcript_says_that_it_was_trimmed(client, monkeypatch):
    from app.memory.session_store import SessionStore

    small = SessionStore(max_turns=2)
    monkeypatch.setattr("app.memory.session_store.get_session_store", lambda: small)

    session_id = post(client).json()["session_id"]
    for _ in range(3):
        post(client, session_id=session_id)

    body = client.get(f"/v1/sessions/{session_id}").json()
    assert body["truncated"] is True
    assert len(body["turns"]) == 2


def test_a_downstream_failure_does_not_leak_its_detail(client, monkeypatch):
    """An agent failure degrades (see test_graph); this is the graph itself failing."""
    async def explode(self, inp):
        raise RuntimeError("qdrant at /srv/secret/path is unreachable")

    monkeypatch.setattr(RouterAgent, "route", explode)

    response = post(client)

    assert response.status_code == 503
    assert response.json()["detail"] == "The assistant is unavailable"
    assert "/srv/secret/path" not in response.text


def test_the_routing_decision_reaches_the_log(client, caplog):
    with caplog.at_level("INFO", logger="app.api.routes"):
        post(client)

    records = [r for r in caplog.records if r.msg == "chat"]
    assert records, "the chat handler should log one structured line"
    fields = records[0].fields
    assert fields["routing"] == {"agent": "data_analysis", "confidence": 0.9,
                                 "reason": "a total"}
    assert fields["agent"] == "data_analysis"
    assert fields["citations"] == 1
    assert fields["rewritten"] is False
