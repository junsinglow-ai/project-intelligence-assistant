"""Application settings, loaded from environment variables / .env.

The stack ships in two deployment modes, selected by a single knob,
``DEPLOYMENT_MODE``:

- ``cloud`` (default): generation runs against a hosted LLM API.
- ``onprem``: generation runs against an Ollama endpoint on the internal
  network. No component makes an external call.

Everything other than the LLM -- embeddings, keyword search, re-ranking, the
vector store, the tabular engine and the graph checkpointer -- stays inside the
deployment and is identical in both modes: in-process, except for Qdrant and
Redis, which are containers on the same network and call nothing out. That is
deliberate: it is what makes the
switch one variable instead of a migration, and it means an index built in one
mode is valid in the other. See DECISIONS.md D-010.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeploymentMode = Literal["cloud", "onprem"]

# Data paths are written relative to the repository, but the backend is run from
# several working directories (`backend/` under uv and pytest, `/app` in Docker),
# so a relative path is anchored here rather than to the current directory.
# Absolute paths are left alone, which is what the container passes in.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATH_SETTINGS = ("data_raw_dir", "upload_dir", "qdrant_path", "tabular_db_path")


def _anchor(path: str) -> str:
    return path if Path(path).is_absolute() else str(_REPO_ROOT / path)


def redact_url(url: str) -> str:
    """Strip credentials from a URL so it can be logged or served.

    `REDIS_URL` carries its password in the userinfo, and both the startup log
    and `/v1/health/dependencies` report what the checkpointer is pointed at.
    That endpoint is public on the hosted demo -- the frontend polls it
    unauthenticated -- so the raw URL would publish the password to anyone who
    asked. The host is kept, because "which Redis" is the whole point of
    reporting it.
    """
    if "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if "@" not in rest:
        return url
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


def _model_chain(value: str) -> list[str]:
    """Split a comma-separated model chain, strongest first.

    Kept as a string field with a list property rather than a `list[str]`
    field, for the same reason `cors_origins` is: pydantic-settings parses a
    complex-typed field from the environment as JSON, so `LLM_MODEL=a,b` would
    fail to load and `LLM_MODEL=["a","b"]` would be the only accepted spelling.
    """
    return [m.strip() for m in value.split(",") if m.strip()]

# Per-mode defaults for knobs left blank. An explicit value always wins, so
# these are defaults, not overrides.
_MODE_DEFAULTS: dict[str, dict[str, str | int]] = {
    "cloud": {
        "llm_provider": "google",
        # Chains, not single models, because the free tier rate-limits by model
        # and a demo that answers "quota exceeded" is a failed demo. Measured
        # against a free-tier key on 2026-09-24: the full `flash` models are
        # the ones that run out -- `gemini-3.8-flash`, `gemini-3.5-flash` and
        # the `gemini-flash-latest` alias all returned 429 -- while the `lite`
        # models answered every time in about a second. So each chain leads
        # with the strongest model that actually answered and *ends* with a
        # lite model, which is what makes the fallback worth having: the last
        # entry has to be something that reliably works.
        #
        # `gemini-2.5-flash` and `gemini-2.5-flash-lite`, the previous
        # defaults, now return 404 NOT_FOUND. They are still in the provider's
        # ListModels response, so nothing catches that by listing alone -- see
        # the note on `_any_present` in `app/llm/providers.py`.
        "llm_model": "gemini-3.6-flash,gemini-3.5-flash,gemini-flash-lite-latest",
        "router_model": "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest",
        "llm_timeout_s": 60,
    },
    "onprem": {
        "llm_provider": "ollama",
        "llm_model": "qwen2.5:7b",
        "router_model": "llama3.2:3b",
        # CPU inference is slow enough to trip ordinary timeouts. Measured on a
        # loaded 12-core CPU-only laptop, llama3.2:3b took 154s to answer over
        # 2.1k tokens of context, and qwen2.5:7b exceeded 300s because the model
        # did not fit in free RAM and swapped. This is a safety net, not a
        # latency target: a timeout that fires just before the answer arrives is
        # worse than one that waits. Hardware guidance is in README.md.
        "llm_timeout_s": 600,
    },
}


class Settings(BaseSettings):
    # Both locations are read, later winning: the backend is usually run from
    # `backend/` (uv, pytest) while `.env` lives at the repository root.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    # --- Deployment mode: the cloud <-> on-prem switch ---
    deployment_mode: DeploymentMode = "cloud"

    # --- LLM (the only mode-dependent component) ---
    # Blank means "use the default for the active mode".
    llm_provider: str = ""
    llm_model: str = ""
    router_model: str = ""
    llm_api_key: str = ""                          # cloud mode only
    ollama_base_url: str = "http://ollama:11434"   # on-prem; any internal host
    llm_timeout_s: int = 0                         # 0 means "use the mode default"
    llm_num_ctx: int = 8192
    # Retries on a failed generation, which in cloud mode is mostly quota
    # rejection. Mode-invariant because only the hosted provider honours it --
    # ChatOllama has no equivalent, and an on-prem endpoint has no quota to
    # trip. Raising it only helps against a per-minute limit: once a daily quota
    # is spent every retry fails too, and each one still burns LLM_TIMEOUT_S.
    #
    # Lowered from 6 on 2026-09-24, measured against the deployed service. The
    # retries and the model chain solve different problems, and 6 made them
    # fight: the provider backs off exponentially, so a model whose daily quota
    # was gone cost ~33s of 429s (16.6 + 8.9 + 4.9 + 2.0 + 1.4) before the chain
    # moved on. Three models, times the three to five generations a skill loop
    # makes, put a single question past 180s. Retries are for a transient
    # per-minute limit; an exhausted daily quota is what the chain is for, and
    # it cannot do its job until the retries give up.
    llm_max_retries: int = 2

    # --- Embeddings (mode-invariant) ---
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where fastembed keeps its ONNX weights. Blank leaves the library's own
    # default, which is what local development and the test suite use. The
    # image sets it so the weights are baked into a build layer: on a
    # scale-to-zero host an unset cache re-downloads ~150MB on every cold
    # start, and it does so inside the first request (D-009). Deliberately not
    # in `_PATH_SETTINGS` -- anchoring would turn blank into the repo root and
    # lose the "use the library default" meaning.
    embedding_cache_dir: str = ""

    # --- Vector store (mode-invariant) ---
    vector_store: str = "qdrant"
    qdrant_path: str = "./data/qdrant"   # embedded fallback, used when qdrant_url is blank
    qdrant_url: str = ""                 # the Compose stack points this at the qdrant service
    qdrant_api_key: str = ""
    qdrant_collection: str = "project_docs"

    # --- Ingestion and chunking (mode-invariant) ---
    data_raw_dir: str = "./data/raw"
    # Uploads land separately from the curated corpus, which `make data`
    # regenerates wholesale. Both are indexed into the same collection.
    upload_dir: str = "./data/uploads"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    table_rows_per_chunk: int = 4

    # --- Retrieval (mode-invariant) ---
    retrieval_mode: Literal["naive", "hybrid"] = "hybrid"
    rerank_enabled: bool = True
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    # As `embedding_cache_dir`, for FlashRank, whose own default is `/tmp`.
    # Worth overriding in a container, where /tmp is memory rather than disk.
    rerank_cache_dir: str = ""
    rerank_top_n: int = 4
    rrf_k: int = 60
    top_k: int = 8

    # --- Agent tool loops (mode-invariant, deliberately) ---
    # A skill loop multiplies generation cost by its iterations, and
    # ARCHITECTURE.md section 7.1 measured 154s per on-prem generation, so an
    # unbounded loop would exceed even the 600s on-prem timeout. This stays
    # mode-invariant because tests/test_config_modes.py asserts that exactly
    # the five LLM settings differ between modes (D-010); a constrained host
    # lowers it explicitly rather than by switching mode.
    agent_max_tool_calls: int = 3

    # --- Graph checkpointing (mode-invariant) ---
    # Where the chat graph checkpoints its state. Blank keeps the checkpointer
    # in-process, which is what the test suite and a single-replica deployment
    # want; the Compose stack points it at the redis service. Mode-invariant for
    # the same reason as everything else here -- Redis is a container inside the
    # deployment boundary, so on-prem runs the identical configuration (D-017).
    redis_url: str = ""
    # Checkpoints expire on their own rather than only when the session store
    # evicts them: that store is process-local, so its eviction cannot be the
    # only thing reclaiming keys from a Redis that outlives the process.
    checkpoint_ttl_minutes: int = 120
    # Socket timeout for checkpoint reads and writes. Generous because the
    # store is frequently not next to the application: a managed Redis in
    # another region turns every checkpoint operation into an inter-continental
    # round trip, and the graph makes several per request. Measured 2026-09-24
    # with the service in us-central1 and the database in europe-west3, a
    # document question died on `TimeoutError: Timeout reading from ...` rather
    # than degrading -- a runtime read is past the point where
    # `ensure_checkpointer_ready` can fall back to the in-process saver.
    # Colocating the two is the real fix; this stops the latency being fatal.
    redis_timeout_s: int = 30

    # --- Tabular analysis (mode-invariant) ---
    tabular_db_path: str = "./data/processed/tables.duckdb"

    max_upload_mb: int = 20

    @model_validator(mode="before")
    @classmethod
    def _blank_means_unset(cls, data: Any) -> Any:
        """Treat an empty environment value as absent.

        A `.env` file -- and Docker Compose's `env_file` -- passes `KEY=` through
        as an empty string, which fails validation for any non-string field.
        Blank means "use the default" throughout this configuration, so drop
        empty values and let the declared defaults (and then the mode defaults)
        apply.
        """
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v != ""}
        return data

    @model_validator(mode="after")
    def _resolve_mode_defaults(self) -> "Settings":
        for field, default in _MODE_DEFAULTS[self.deployment_mode].items():
            if not getattr(self, field):
                object.__setattr__(self, field, default)
        for field in _PATH_SETTINGS:
            object.__setattr__(self, field, _anchor(getattr(self, field)))
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_models(self) -> list[str]:
        """The answering chain, strongest first. Never empty."""
        return _model_chain(self.llm_model)

    @property
    def router_models(self) -> list[str]:
        """The routing chain, strongest first. Never empty."""
        return _model_chain(self.router_model)

    @property
    def collection_name(self) -> str:
        """Collection name with the embedding model stamped into it.

        Embeddings are the one choice that cannot be changed by configuration
        alone: vector dimensions differ between models, so a new embedder needs
        a re-index. Encoding the model in the name means a change targets a
        different (empty) collection and fails loudly, rather than silently
        querying an index built with incompatible vectors.
        """
        slug = self.embedding_model.split("/")[-1].replace(".", "-").replace("_", "-").lower()
        return f"{self.qdrant_collection}__{slug}"

    @property
    def checkpoint_target(self) -> str:
        """What the graph checkpoints to, for the startup log and the probe.

        Redacted: both consumers are places the value is published, and the
        probe is reachable unauthenticated on the hosted demo.
        """
        return redact_url(self.redis_url) if self.redis_url else "in-process"

    def missing_requirements(self) -> list[str]:
        """Config required by the active mode but not supplied."""
        if self.deployment_mode == "cloud" and not self.llm_api_key:
            return ["LLM_API_KEY is required when DEPLOYMENT_MODE=cloud"]
        if self.deployment_mode == "onprem" and not self.ollama_base_url:
            return ["OLLAMA_BASE_URL is required when DEPLOYMENT_MODE=onprem"]
        return []

    def resolved_summary(self) -> dict[str, Any]:
        """What the mode resolved to, for the startup log and readiness probe."""
        return {
            "deployment_mode": self.deployment_mode,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_models[0],
            "router_model": self.router_models[0],
            "llm_fallbacks": self.llm_models[1:],
            "router_fallbacks": self.router_models[1:],
            "llm_timeout_s": self.llm_timeout_s,
            "embedding_model": self.embedding_model,
            "collection": self.collection_name,
            "retrieval_mode": self.retrieval_mode,
            "checkpointer": self.checkpoint_target,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
