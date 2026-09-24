"""The provider factory: the single point of model access.

No test here reaches the network. The embedding test loads a local ONNX model,
which is downloaded once and cached; it is marked `slow` so the rest of the
suite stays usable offline.
"""

import pytest

from pydantic import BaseModel

from app.config import Settings
from app.llm.providers import (
    check_llm_ready,
    get_embeddings,
    get_llm,
    get_llms,
    get_structured_llm,
)


class RouteShape(BaseModel):
    """A stand-in for the router's schema; the shape is irrelevant here."""

    agent: str

CLOUD = Settings(deployment_mode="cloud", llm_api_key="test-key")
ONPREM = Settings(deployment_mode="onprem", ollama_base_url="http://127.0.0.1:1")


@pytest.mark.slow
def test_embeddings_have_the_dimension_the_index_assumes():
    """384 is baked into the collection naming and the D-002 rationale."""
    embeddings = get_embeddings(CLOUD)

    assert len(embeddings.embed_query("total Q3 budget variance")) == 384
    assert [len(v) for v in embeddings.embed_documents(["a", "b"])] == [384, 384]


@pytest.mark.slow
def test_embeddings_are_identical_across_modes():
    """Embedding is in-process and mode-invariant, so vectors must match exactly."""
    assert get_embeddings(CLOUD).embed_query("risk register") == pytest.approx(
        get_embeddings(ONPREM).embed_query("risk register")
    )


def test_unknown_embedding_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embeddings(Settings(embedding_provider="openai", llm_api_key="k"))


def test_unknown_llm_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm(settings=Settings(llm_provider="openai", llm_api_key="k"))


def test_each_mode_builds_its_own_chat_model():
    assert type(get_llm(settings=CLOUD)).__name__ == "ChatGoogleGenerativeAI"
    assert type(get_llm(settings=ONPREM)).__name__ == "ChatOllama"


def test_model_argument_overrides_the_configured_model():
    """How the router asks for the cheaper model without a second factory."""
    assert get_llm(ONPREM.router_model, settings=ONPREM).model == "llama3.2:3b"
    assert get_llm(settings=ONPREM).model == "qwen2.5:7b"


def test_construction_does_not_require_a_reachable_endpoint():
    """Failure must surface on invoke, not at import or startup."""
    llm = get_llm(settings=ONPREM)

    assert llm.base_url == "http://127.0.0.1:1"


def test_readiness_reports_an_unreachable_endpoint_without_raising():
    status = check_llm_ready(ONPREM)

    assert status["reachable"] is False
    assert "error" in status
    assert status["endpoint"] == "http://127.0.0.1:1"


def test_readiness_reports_missing_configuration_before_trying_to_connect():
    status = check_llm_ready(Settings(deployment_mode="cloud", llm_api_key=""))

    assert status["reachable"] is False
    assert "LLM_API_KEY" in status["error"]


# --------------------------------------------------------------------------
# Model chains
#
# Every Settings here passes the values it asserts on explicitly: the repo-root
# `.env` is on the settings search path, so a bare `Settings()` inherits the
# developer's own provider and models.
# --------------------------------------------------------------------------


def _onprem(**kwargs):
    return Settings(deployment_mode="onprem", llm_provider="ollama",
                    ollama_base_url="http://127.0.0.1:1", **kwargs)


def test_a_single_model_is_a_chain_of_one():
    """No comma means no fallback, which is the pre-chain behaviour."""
    settings = _onprem(llm_model="qwen2.5:7b")

    assert settings.llm_models == ["qwen2.5:7b"]


def test_a_chain_splits_on_commas_and_trims():
    settings = _onprem(llm_model="qwen2.5:7b, llama3.2:3b ,, phi3:mini")

    assert settings.llm_models == ["qwen2.5:7b", "llama3.2:3b", "phi3:mini"]


def test_both_chains_are_independent():
    settings = _onprem(llm_model="a:1,b:2", router_model="c:3")

    assert settings.llm_models == ["a:1", "b:2"]
    assert settings.router_models == ["c:3"]


def test_get_llm_returns_the_strongest_model_of_a_chain():
    """The chain is ordered strongest first, and `get_llm` still means "one model"."""
    settings = _onprem(llm_model="qwen2.5:7b,llama3.2:3b")

    assert get_llm(settings=settings).model == "qwen2.5:7b"


def test_get_llms_builds_every_model_in_order():
    settings = _onprem(llm_model="qwen2.5:7b,llama3.2:3b,phi3:mini")

    assert [m.model for m in get_llms(settings=settings)] == [
        "qwen2.5:7b", "llama3.2:3b", "phi3:mini",
    ]


def test_get_llms_accepts_an_explicit_chain():
    """How the router asks for the routing chain rather than the answering one."""
    settings = _onprem(llm_model="qwen2.5:7b", router_model="llama3.2:3b,phi3:mini")

    assert [m.model for m in get_llms(settings.router_models, settings)] == [
        "llama3.2:3b", "phi3:mini",
    ]


def test_one_model_is_not_wrapped_in_fallback_machinery():
    """A single model must stay a plain runnable -- no cost for not opting in."""
    from langchain_core.runnables import RunnableWithFallbacks

    structured = get_structured_llm(RouteShape, "qwen2.5:7b", _onprem())

    assert not isinstance(structured, RunnableWithFallbacks)


def test_a_chain_produces_a_runnable_with_fallbacks():
    from langchain_core.runnables import RunnableWithFallbacks

    structured = get_structured_llm(RouteShape, "qwen2.5:7b,llama3.2:3b", _onprem())

    assert isinstance(structured, RunnableWithFallbacks)


# --------------------------------------------------------------------------
# Falling through the chain
# --------------------------------------------------------------------------


def _quota_error():
    """The 429 a free tier returns once its quota is spent."""
    from google.genai.errors import ClientError

    return ClientError(429, {"error": {"message": "quota exhausted",
                                       "status": "RESOURCE_EXHAUSTED"}})


@pytest.fixture
def chain_of(monkeypatch):
    """Install one scripted model per name, so a test can fail just the first.

    Patches the factory rather than the chain builder: `get_llms` goes through
    `get_llm`, which is the seam the rest of the suite already uses.
    """
    from tests.fakes import ScriptedChatModel

    def install(**by_name):
        models = {name: ScriptedChatModel(responses=list(responses))
                  for name, responses in by_name.items()}
        monkeypatch.setattr("app.llm.providers.get_llm",
                            lambda model=None, settings=None: models[model],
                            raising=True)
        return models

    return install


def test_the_next_model_answers_when_the_strongest_is_rate_limited(chain_of):
    """The whole point: a 429 on the primary must not fail the request."""
    models = chain_of(strong=[_quota_error()], weak=[RouteShape(agent="document_qa")])
    settings = Settings(llm_api_key="k", llm_provider="google", llm_model="strong,weak")

    result = get_structured_llm(RouteShape, settings.llm_models, settings).invoke("q")

    assert result.agent == "document_qa"
    assert len(models["strong"].calls) == 1  # tried first, and only once
    assert len(models["weak"].calls) == 1


def test_the_chain_is_walked_in_order_until_one_succeeds(chain_of):
    chain_of(a=[_quota_error()], b=[_quota_error()], c=[RouteShape(agent="data_analysis")])
    settings = Settings(llm_api_key="k", llm_provider="google", llm_model="a,b,c")

    result = get_structured_llm(RouteShape, settings.llm_models, settings).invoke("q")

    assert result.agent == "data_analysis"


def test_the_strongest_model_is_preferred_when_it_works(chain_of):
    """No fallback unless the primary actually fails -- quality is not traded away."""
    models = chain_of(strong=[RouteShape(agent="document_qa")], weak=[_quota_error()])
    settings = Settings(llm_api_key="k", llm_provider="google", llm_model="strong,weak")

    get_structured_llm(RouteShape, settings.llm_models, settings).invoke("q")

    assert len(models["weak"].calls) == 0  # never reached


def test_a_local_error_is_raised_rather_than_walked_down_the_chain(chain_of):
    """A bug is not transient: every model would fail it identically.

    NotImplementedError specifically must surface loudly (D-015), not be
    swallowed by trying two more models that fail the same way.
    """
    models = chain_of(strong=[NotImplementedError("half-built")],
                      weak=[RouteShape(agent="document_qa")])
    settings = Settings(llm_api_key="k", llm_provider="google", llm_model="strong,weak")

    with pytest.raises(NotImplementedError):
        get_structured_llm(RouteShape, settings.llm_models, settings).invoke("q")

    assert len(models["weak"].calls) == 0  # never reached


def test_the_last_error_surfaces_when_every_model_is_exhausted(chain_of):
    """A fully spent quota still has to fail, not hang or return nothing."""
    from google.genai.errors import ClientError

    chain_of(a=[_quota_error()], b=[_quota_error()])
    settings = Settings(llm_api_key="k", llm_provider="google", llm_model="a,b")

    with pytest.raises(ClientError):
        get_structured_llm(RouteShape, settings.llm_models, settings).invoke("q")
