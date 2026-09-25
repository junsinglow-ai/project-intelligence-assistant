"""API keys: issued into their own DuckDB file, checked on every /v1 endpoint but readiness."""

import duckdb
import pytest
from fastapi.testclient import TestClient

from app.api import auth
from app.config import get_settings
from app.main import app


@pytest.fixture
def keys(tmp_path, monkeypatch):
    """Auth switched on, against an empty key store under `tmp_path`."""
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS_DB_PATH", str(tmp_path / "api_keys.duckdb"))
    monkeypatch.setenv("TABULAR_DB_PATH", str(tmp_path / "tables.duckdb"))
    get_settings.cache_clear()
    auth._invalidate()
    yield get_settings()
    auth._invalidate()
    get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_disabled_by_default_needs_no_key(client, monkeypatch):
    monkeypatch.setenv("API_AUTH_ENABLED", "")
    get_settings.cache_clear()
    assert client.get("/v1/agents").status_code == 200


def test_enabled_with_no_store_refuses_everyone(keys, client):
    """A missing key file must fail closed, not open."""
    assert client.get("/v1/agents").status_code == 401
    assert client.get("/v1/agents", headers={"X-API-Key": "pia_guess"}).status_code == 401


def test_issued_key_is_accepted_and_others_are_not(keys, client):
    key = auth.create_key("frontend")
    assert key.startswith(auth.KEY_PREFIX)
    assert client.get("/v1/agents", headers={"X-API-Key": key}).status_code == 200
    assert client.get("/v1/agents", headers={"X-API-Key": key + "x"}).status_code == 401
    response = client.get("/v1/agents")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


def test_every_data_endpoint_is_keyed(keys, client):
    assert client.post("/v1/chat", json={"question": "hi"}).status_code == 401
    assert client.get("/v1/sessions/abc").status_code == 401
    assert client.post("/v1/upload", files={"file": ("a.csv", b"x")}).status_code == 401


def test_health_stays_open(keys, client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health/dependencies").status_code == 200


def test_revoked_key_is_refused(keys, client):
    key = auth.create_key("frontend")
    assert auth.revoke_key("frontend")
    assert client.get("/v1/agents", headers={"X-API-Key": key}).status_code == 401
    assert not auth.revoke_key("frontend")
    assert [k.name for k in auth.list_keys()] == ["frontend"]
    assert auth.list_keys()[0].revoked_at is not None


def test_a_revoked_name_can_be_reissued(keys, client):
    old = auth.create_key("frontend")
    auth.revoke_key("frontend")
    new = auth.create_key("frontend")
    assert client.get("/v1/agents", headers={"X-API-Key": old}).status_code == 401
    assert client.get("/v1/agents", headers={"X-API-Key": new}).status_code == 200
    assert [k.revoked_at is None for k in auth.list_keys()] == [False, True]


def test_names_are_unique(keys):
    auth.create_key("frontend")
    with pytest.raises(ValueError, match="is live"):
        auth.create_key("frontend")
    with pytest.raises(ValueError):
        auth.create_key("  ")


def test_only_a_digest_is_stored(keys):
    key = auth.create_key("frontend")
    with duckdb.connect(keys.api_keys_db_path, read_only=True) as connection:
        stored = connection.execute("SELECT * FROM api_keys").fetchall()
    assert key not in {str(v) for row in stored for v in row}
    assert stored[0][2] == auth._digest(key)


def test_keys_are_out_of_reach_of_generated_sql(keys):
    """The model's SQL runs against the tables store; the keys must not be in it."""
    from app.ingestion.tabular_store import list_tables

    auth.create_key("frontend")
    assert keys.api_keys_db_path != keys.tabular_db_path
    assert "api_keys" not in list_tables(keys)
