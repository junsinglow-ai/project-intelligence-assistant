"""API keys for the REST API: issued by `make api-key`, checked per request.

Keys live in their own DuckDB file (`API_KEYS_DB_PATH`), never in the tables
store. That separation is the point: `run_sql` executes model-written SQL
against `tables.duckdb`, and a model steered by an uploaded document could
otherwise `SELECT * FROM api_keys`. Nothing on the query path opens this file.

Only a SHA-256 digest of each key is stored, and the key itself is printed once
when it is created. A plain hash rather than a slow KDF is deliberate: the keys
are 256 bits from `secrets`, so there is no dictionary to brute-force, and a
KDF would add its cost to every request.

Checking is off unless `API_AUTH_ENABLED=true`, so the local stack and the test
suite run as before. When it is on and no key store exists, every protected
request is refused: a missing file must not mean "open".
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "pia_"
HEADER = "X-API-Key"

# How long a loaded set of digests is trusted. A key issued or revoked with
# `make` reaches a running backend within this window, without a restart and
# without opening the file on every request.
_CACHE_TTL_S = 30.0

# Timestamps are naive UTC: DuckDB returns TIMESTAMPTZ through pytz, which is
# not a dependency, and nothing here needs a zone.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    name        VARCHAR NOT NULL,   -- unique among live keys; revoked rows stay as history
    key_prefix  VARCHAR NOT NULL,
    key_sha256  VARCHAR NOT NULL UNIQUE,
    created_at  TIMESTAMP NOT NULL,
    revoked_at  TIMESTAMP
)
"""


@dataclass(frozen=True, slots=True)
class KeyRecord:
    name: str
    key_prefix: str
    created_at: datetime
    revoked_at: datetime | None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _writable(settings: Settings) -> duckdb.DuckDBPyConnection:
    path = Path(settings.api_keys_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(_SCHEMA)
    return connection


def create_key(name: str, settings: Settings | None = None) -> str:
    """Issue a key named `name` and return it. It is not recoverable afterwards."""
    settings = settings or get_settings()
    name = name.strip()
    if not name:
        raise ValueError("a key needs a name, e.g. NAME=frontend")
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    with _writable(settings) as connection:
        if connection.execute("SELECT 1 FROM api_keys WHERE name = ? AND revoked_at IS NULL", [name]).fetchone():
            raise ValueError(f"a key named {name!r} is live; revoke it first or pick another name")
        connection.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, NULL)",
            [name, key[: len(KEY_PREFIX) + 6], _digest(key), _utcnow()],
        )
    _invalidate()
    return key


def revoke_key(name: str, settings: Settings | None = None) -> bool:
    """Revoke the key named `name`. False if there is no live key by that name."""
    settings = settings or get_settings()
    with _writable(settings) as connection:
        revoked = connection.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE name = ? AND revoked_at IS NULL "
            "RETURNING name",
            [_utcnow(), name],
        ).fetchall()
    _invalidate()
    return bool(revoked)


def list_keys(settings: Settings | None = None) -> list[KeyRecord]:
    settings = settings or get_settings()
    if not Path(settings.api_keys_db_path).exists():
        return []
    with duckdb.connect(settings.api_keys_db_path, read_only=True) as connection:
        rows = connection.execute(
            "SELECT name, key_prefix, created_at, revoked_at FROM api_keys ORDER BY created_at"
        ).fetchall()
    return [KeyRecord(*row) for row in rows]


# --- Verification ---------------------------------------------------------

_lock = threading.Lock()
_cache: tuple[float, str, dict[str, str]] | None = None   # (loaded_at, path, digest -> name)


def _invalidate() -> None:
    global _cache
    with _lock:
        _cache = None


def _live_digests(settings: Settings) -> dict[str, str]:
    """Digests of unrevoked keys, cached for `_CACHE_TTL_S`.

    The file is opened read-only and closed straight away, so `make api-key`
    can write to it while the backend runs. If the open collides with that
    write, the previous set is kept rather than locking everyone out.
    """
    global _cache
    path = settings.api_keys_db_path
    with _lock:
        if _cache and _cache[1] == path and time.monotonic() - _cache[0] < _CACHE_TTL_S:
            return _cache[2]
        if not Path(path).exists():
            digests: dict[str, str] = {}
        else:
            try:
                with duckdb.connect(path, read_only=True) as connection:
                    digests = dict(connection.execute(
                        "SELECT key_sha256, name FROM api_keys WHERE revoked_at IS NULL"
                    ).fetchall())
            except duckdb.Error as exc:
                logger.warning("api key store unreadable", extra={"fields": {
                    "error": f"{type(exc).__name__}: {exc}"}})
                return _cache[2] if _cache and _cache[1] == path else {}
        _cache = (time.monotonic(), path, digests)
        return digests


_header = APIKeyHeader(name=HEADER, auto_error=False,
                       description="Issued with `make api-key NAME=<client>`.")


def require_api_key(key: str | None = Security(_header)) -> str | None:
    """FastAPI dependency: the calling key's name, or 401.

    Returns None when checking is disabled. The key is never logged; a
    rejection records only why, so a probe cannot fill the log with guesses.
    """
    settings = get_settings()
    if not settings.api_auth_enabled:
        return None
    if not key:
        logger.warning("api key rejected", extra={"fields": {"reason": "missing"}})
        raise HTTPException(status_code=401, detail=f"Missing {HEADER} header",
                            headers={"WWW-Authenticate": "ApiKey"})
    name = _live_digests(settings).get(_digest(key))
    if name is None:
        logger.warning("api key rejected", extra={"fields": {"reason": "unknown or revoked"}})
        raise HTTPException(status_code=401, detail="Invalid API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return name
