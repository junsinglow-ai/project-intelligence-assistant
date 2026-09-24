from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert "x-trace-id" in r.headers


def test_agents_endpoint_lists_routable_agents():
    names = {a["name"] for a in client.get("/v1/agents").json()}
    assert names == {"document_qa", "data_analysis", "small_talk"}


def test_agents_endpoint_reports_each_agent_s_skills():
    """An agent is defined by the tools it can call, so the API says which."""
    skills = {a["name"]: a["skills"] for a in client.get("/v1/agents").json()}

    assert skills["document_qa"] == ["search_documents"]
    assert skills["data_analysis"] == ["describe_tables", "run_sql"]
    # An agent may legitimately hold none: small_talk answers from its prompt.
    assert skills["small_talk"] == []


def test_a_record_carries_severity_for_hosted_log_collectors():
    """Google Cloud Logging ranks a record by `severity`, not `level`.

    Without it every line files as INFO, so a failed request does not show up
    under a severity filter or an alert -- the `agent failed` ERROR from the
    deployed service only surfaced by scanning every entry by hand.
    """
    import json
    import logging

    from app.observability.logging import JsonFormatter

    record = logging.LogRecord("app.test", logging.ERROR, __file__, 1, "boom", None, None)
    payload = json.loads(JsonFormatter().format(record))

    assert payload["severity"] == "ERROR"
    assert payload["level"] == "ERROR"   # kept, so local output is unchanged
