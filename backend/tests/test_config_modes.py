"""The cloud <-> on-prem switch.

The promise in DECISIONS.md D-010 is that moving to a fully on-premises
deployment is one configuration change with no re-index. These tests are what
stop that promise rotting: if a future change makes any part of the retrieval
stack mode-dependent, the parity test fails.
"""

import pytest

from app.config import Settings

# Anything that could leak in from the developer's shell and mask a default.
_MODE_ENV = (
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
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Blank each variable rather than deleting it.

    `Settings` reads `.env` as well as the environment, so a *deleted* variable
    simply falls through to the file -- and a developer running on-prem locally
    would see every cloud-default assertion here fail. Setting it empty wins
    over the file, and `_blank_means_unset` then drops it so the declared
    default applies. Same reasoning as the fixtures in `conftest.py`.
    """
    for name in _MODE_ENV:
        monkeypatch.setenv(name, "")


def test_cloud_is_the_default_mode():
    assert Settings().deployment_mode == "cloud"


def test_switching_to_onprem_needs_only_one_variable():
    """DEPLOYMENT_MODE alone must be enough -- no second knob to remember."""
    onprem = Settings(deployment_mode="onprem")

    assert onprem.llm_provider == "ollama"
    assert onprem.llm_model == "qwen2.5:7b"
    assert onprem.router_model == "llama3.2:3b"
    assert onprem.ollama_base_url  # has a usable default


def test_cloud_mode_resolves_the_hosted_provider():
    cloud = Settings(deployment_mode="cloud")

    assert cloud.llm_provider == "google"
    assert cloud.llm_model.startswith("gemini")
    assert cloud.router_model.startswith("gemini")


def test_retrieval_stack_is_identical_in_both_modes():
    """The parity guarantee: only generation differs between modes.

    Everything asserted here feeds the index or the retrieval path. If any of it
    diverged, switching modes would require a re-index and the single-config
    claim in ARCHITECTURE.md section 9 would be false.
    """
    cloud = Settings(deployment_mode="cloud")
    onprem = Settings(deployment_mode="onprem")

    shared = (
        "data_raw_dir",
        "upload_dir",
        "chunk_size",
        "chunk_overlap",
        "table_rows_per_chunk",
        "embedding_provider",
        "embedding_model",
        "vector_store",
        "qdrant_collection",
        "retrieval_mode",
        "rerank_enabled",
        "rerank_model",
        "rerank_top_n",
        "rrf_k",
        "top_k",
        "tabular_db_path",
    )
    for field in shared:
        assert getattr(cloud, field) == getattr(onprem, field), field

    # The index itself must be reusable across modes.
    assert cloud.collection_name == onprem.collection_name


def test_only_the_llm_differs_between_modes():
    """Catch a future knob that silently becomes mode-dependent."""
    cloud = Settings(deployment_mode="cloud").model_dump()
    onprem = Settings(deployment_mode="onprem").model_dump()

    differing = {k for k in cloud if cloud[k] != onprem[k]}
    assert differing == {
        "deployment_mode",
        "llm_provider",
        "llm_model",
        "router_model",
        "llm_timeout_s",
    }


def test_explicit_value_survives_mode_resolution():
    """Mode defaults fill blanks; they never override an explicit choice."""
    settings = Settings(deployment_mode="onprem", llm_model="mistral:7b", llm_timeout_s=42)

    assert settings.llm_model == "mistral:7b"
    assert settings.llm_timeout_s == 42
    assert settings.router_model == "llama3.2:3b"  # still defaulted


def test_onprem_allows_a_remote_ollama_host():
    """On-prem does not mean same-box: the endpoint is operator-supplied."""
    settings = Settings(deployment_mode="onprem", ollama_base_url="http://gpu-box.internal:11434")

    assert settings.ollama_base_url == "http://gpu-box.internal:11434"
    assert settings.missing_requirements() == []


def test_collection_name_encodes_the_embedding_model():
    """A changed embedder must target a different collection, not corrupt one."""
    default = Settings()
    other = Settings(embedding_model="BAAI/bge-base-en-v1.5")

    assert default.collection_name == "project_docs__bge-small-en-v1-5"
    assert default.collection_name != other.collection_name


def test_each_mode_reports_its_own_missing_requirements():
    assert "LLM_API_KEY" in Settings(deployment_mode="cloud").missing_requirements()[0]
    assert Settings(deployment_mode="cloud", llm_api_key="k").missing_requirements() == []

    # model_copy skips validators, which is the only way to reach this state:
    # a blank env value is treated as unset and falls back to the default.
    onprem = Settings(deployment_mode="onprem").model_copy(update={"ollama_base_url": ""})
    assert "OLLAMA_BASE_URL" in onprem.missing_requirements()[0]


def test_blank_env_values_fall_back_to_defaults():
    """`KEY=` in .env means "unset", not "empty".

    Docker Compose passes blank env_file entries through as empty strings, which
    would otherwise fail validation on every non-string field.
    """
    settings = Settings(llm_timeout_s="", llm_model="", top_k="", llm_api_key="k")
    unset = Settings(llm_api_key="k")

    assert settings.llm_timeout_s == 60      # cloud mode default
    # Compared against an unset instance rather than a literal model name: the
    # claim is that blank and absent resolve identically, and pinning the name
    # here made this fail for the unrelated reason that a default moved.
    assert settings.llm_model == unset.llm_model
    assert settings.top_k == 8               # declared default


def test_onprem_needs_no_api_key():
    """The security story in section 8.2: on-prem holds no secrets at all."""
    assert Settings(deployment_mode="onprem", llm_api_key="").missing_requirements() == []


# --- Credentials must not reach a log or a public endpoint -----------------
# REDIS_URL carries its password in the userinfo, and both the startup log and
# /v1/health/dependencies report the checkpoint target. On the hosted demo that
# endpoint is reachable unauthenticated, so the raw URL would publish it.


def test_the_checkpoint_target_hides_the_password():
    url = "redis://default:hunter2@redis.example.com:15685"

    target = Settings(redis_url=url, llm_api_key="k").checkpoint_target

    assert "hunter2" not in target
    assert "redis.example.com:15685" in target   # which Redis is still the point


def test_a_url_without_credentials_is_untouched():
    url = "redis://redis:6379/0"

    assert Settings(redis_url=url, llm_api_key="k").checkpoint_target == url


def test_the_resolved_summary_is_safe_to_log():
    """`resolved_summary()` goes to the startup log and the readiness body."""
    summary = Settings(
        redis_url="rediss://user:s3cret@host:1234", llm_api_key="k"
    ).resolved_summary()

    assert "s3cret" not in repr(summary)


def test_in_process_is_reported_when_there_is_no_redis():
    assert Settings(llm_api_key="k").checkpoint_target == "in-process"
