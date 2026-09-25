"""LLM and embedding factories.

All model access goes through here, so switching provider (e.g. to a local
model for on-prem deployment) is a configuration change, not a code change.

This is the only module that knows deployment modes exist. Agents, retrieval and
ingestion call ``get_llm()`` / ``get_embeddings()`` and stay mode-agnostic.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import Settings, _model_chain, get_settings

logger = logging.getLogger(__name__)

_SUPPORTED_LLM_PROVIDERS = ("google", "ollama")
_SUPPORTED_EMBEDDING_PROVIDERS = ("fastembed",)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings over a local fastembed ONNX model.

    Runs in-process in both deployment modes, so embedding never becomes an
    external call and the index is identical cloud and on-prem. Hand-rolled
    rather than pulled from langchain-community, which is legacy in LangChain 1.x
    and would be a large dependency for one small adapter.
    """

    def __init__(self, model_name: str, cache_dir: str = "") -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    @property
    def _model(self) -> Any:
        return _load_text_embedding(self.model_name, self.cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=4)
def _load_text_embedding(model_name: str, cache_dir: str = "") -> Any:
    """Load and cache the ONNX model; loading dominates per-call latency.

    `cache_dir` is part of the cache key rather than read from settings here,
    so that two different caches cannot collide on one memoised model.
    """
    from fastembed import TextEmbedding

    logger.info(
        "loading embedding model",
        extra={"fields": {"embedding_model": model_name, "cache_dir": cache_dir or "default"}},
    )
    # Blank means "use the library default"; passing None is how fastembed
    # spells that, so an empty setting must not reach it as an empty path.
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir or None)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    provider = settings.embedding_provider
    if provider != "fastembed":
        raise ValueError(
            f"Unsupported embedding provider {provider!r}; "
            f"supported: {', '.join(_SUPPORTED_EMBEDDING_PROVIDERS)}"
        )
    return FastEmbedEmbeddings(settings.embedding_model, settings.embedding_cache_dir)


# --------------------------------------------------------------------------
# Chat models
# --------------------------------------------------------------------------


def get_llm(model: str | None = None, settings: Settings | None = None) -> BaseChatModel:
    """Return one chat model for the active deployment mode.

    ``model`` overrides the configured answering model, which is how the router
    asks for its own chain (``Settings.models_for``) without a second factory.

    This returns the *primary* model when a chain is configured. Callers that
    want the fallbacks want ``get_llms()`` or ``get_structured_llm()``; the
    contract here stays "one BaseChatModel" because `create_agent` and
    `with_structured_output` both require a real chat model, which
    ``RunnableWithFallbacks`` is not.
    """
    from app.llm.usage import USAGE_LOGGER

    settings = settings or get_settings()
    provider = settings.llm_provider
    model = model or settings.llm_models[0]

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.llm_api_key or None,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            callbacks=[USAGE_LOGGER],
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            num_ctx=settings.llm_num_ctx,
            client_kwargs={"timeout": settings.llm_timeout_s},
            callbacks=[USAGE_LOGGER],
        )

    raise ValueError(
        f"Unsupported LLM provider {provider!r}; "
        f"supported: {', '.join(_SUPPORTED_LLM_PROVIDERS)}"
    )


# --------------------------------------------------------------------------
# Model chains
#
# LLM_MODEL, ROUTER_MODEL and each AGENT_MODELS__<NAME> are comma-separated
# chains, preferred first. When
# the strongest model fails with a provider-side error -- on the free tier,
# usually quota -- the next one answers instead. Retries come first: the
# per-model `max_retries` is exhausted before the chain moves on, so a
# per-minute limit is waited out rather than immediately downgrading quality.
# --------------------------------------------------------------------------


def _chain_names(models: str | Sequence[str] | None, settings: Settings) -> list[str]:
    if models is None:
        return settings.llm_models
    if isinstance(models, str):
        return _model_chain(models)
    return [m for m in models if m]


def get_llms(
    models: str | Sequence[str] | None = None, settings: Settings | None = None
) -> list[BaseChatModel]:
    """Every model in a chain, strongest first. Never empty.

    Built through ``get_llm`` rather than inline so that patching
    ``app.llm.providers.get_llm`` -- which is how the test suite installs a
    scripted model -- reaches the chain too.
    """
    settings = settings or get_settings()
    return [get_llm(name, settings) for name in _chain_names(models, settings)]


def _fallback_errors(settings: Settings) -> tuple[type[BaseException], ...]:
    """Errors that justify trying the next model in the chain.

    The line is provider-side versus local. Anything the provider returned over
    the wire -- 429 quota, 5xx overload, a timeout -- may well succeed on a
    different model, so it falls through. A local failure (a bad schema, a
    missing attribute, and `NotImplementedError` above all, which D-005 requires
    to surface loudly) is not transient: every model in the chain would fail it
    identically, so it is raised rather than silently retried N more times.

    Google's `APIError` covers both its 4xx `ClientError` and its 5xx
    `ServerError`. That does include a genuine 400, which will then be attempted
    once per model -- accepted, because a 400 fails fast and the chain re-raises
    it rather than hiding it.
    """
    if settings.llm_provider == "google":
        from google.genai.errors import APIError

        return (APIError, TimeoutError)
    return (TimeoutError,)


def get_structured_llm(
    schema: Any,
    models: str | Sequence[str] | None = None,
    settings: Settings | None = None,
) -> Runnable:
    """A structured-output runnable that falls through the model chain.

    ``with_structured_output`` is applied per model and the fallbacks wrap the
    *result*, because ``RunnableWithFallbacks`` has no
    ``with_structured_output`` of its own -- wrapping the other way round gives
    a runnable the router cannot use.
    """
    settings = settings or get_settings()
    chain = [
        llm.with_structured_output(schema) for llm in get_llms(models, settings)
    ]
    if len(chain) == 1:
        return chain[0]
    return chain[0].with_fallbacks(
        chain[1:], exceptions_to_handle=_fallback_errors(settings)
    )


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------


def check_llm_ready(settings: Settings | None = None) -> dict[str, Any]:
    """Report whether the configured LLM backend is usable. Never raises.

    On-prem the endpoint is operator-supplied and may point anywhere on the
    internal network, so "reachable, and are the models actually pulled?" is the
    first question worth answering after a mode flip.
    """
    settings = settings or get_settings()
    status: dict[str, Any] = {
        "deployment_mode": settings.deployment_mode,
        "provider": settings.llm_provider,
        "models": _chains(settings),
        "reachable": False,
    }
    unknown = unknown_agent_model_keys(settings)
    if unknown:
        status["unknown_agent_models"] = unknown

    missing = settings.missing_requirements()
    if missing:
        status["error"] = missing[0]
        return status

    try:
        if settings.llm_provider == "ollama":
            status["endpoint"] = settings.ollama_base_url
            available = _ollama_models(settings)
            status["reachable"] = True
            status["models_present"] = _chain_present(settings, available)
            if not _all_present(status["models_present"]):
                status["hint"] = "run `make models` to pull the configured models"
        elif settings.llm_provider == "google":
            available = _google_models(settings)
            status["reachable"] = True
            status["models_present"] = _chain_present(settings, available)
        else:
            status["error"] = f"Unsupported LLM provider {settings.llm_provider!r}"
    except Exception as exc:  # readiness reports failure, it does not raise
        status["error"] = f"{type(exc).__name__}: {exc}"

    return status


def _chain_present(settings: Settings, available: list[str]) -> dict[str, dict[str, bool]]:
    """Per-model presence for both chains.

    Every model is reported, not just the primary: a fallback that is missing is
    exactly the thing that will not be noticed until the primary is rate-limited
    and the chain has nowhere to go.
    """
    return {
        site: {m: _model_present(m, available) for m in chain}
        for site, chain in _chains(settings).items()
    }


def _chains(settings: Settings) -> dict[str, list[str]]:
    """The two defaults, then each call site that has a chain of its own."""
    chains = {"answering": settings.llm_models, "routing": settings.router_models}
    for site in settings.agent_models:
        chains[f"agent:{site}"] = settings.models_for(site)
    return chains


def unknown_agent_model_keys(settings: Settings) -> list[str]:
    """`AGENT_MODELS__<NAME>` keys that name no registered agent.

    Such a key is not an error to pydantic, so a typo would otherwise leave its
    agent silently running on LLM_MODEL -- which, for a model chosen to save
    cost or to hold SQL quality, is exactly the failure nobody notices.
    """
    from app.agents.registry import list_agents
    from app.config import ROUTER_CALL_SITES

    known = {a.name for a in list_agents(include_router=True)} | set(ROUTER_CALL_SITES)
    return sorted(set(settings.agent_models) - known)


def _all_present(present: dict[str, dict[str, bool]]) -> bool:
    return all(found for chain in present.values() for found in chain.values())


def any_model_present(status: dict[str, Any]) -> bool:
    """Whether each chain has at least one model the provider lists.

    Not `_all_present`: a chain exists precisely so a missing or exhausted
    fallback is survivable, so requiring every entry would report a healthy
    deployment as degraded. One usable model per chain is the real bar.

    A caveat worth knowing, because it bounds what this can promise: presence
    means "the provider lists it", not "it will generate". Google's ListModels
    still returns `gemini-2.5-flash` and `gemini-2.5-flash-lite`, both of which
    now answer generateContent with 404 NOT_FOUND -- which is how they survived
    as the cloud defaults. Listing is the cheapest check that catches a typo or
    a model pulled from the catalogue entirely; only a real generation catches
    a listed-but-retired one.
    """
    present = status.get("models_present")
    if not present:
        return True  # nothing to judge: the provider reported no listing
    return all(any(chain.values()) for chain in present.values() if chain)


def _ollama_models(settings: Settings) -> list[str]:
    from ollama import Client

    listing = Client(host=settings.ollama_base_url, timeout=10).list()
    return [m.model for m in listing.models if m.model]


def _google_models(settings: Settings) -> list[str]:
    from google import genai

    client = genai.Client(api_key=settings.llm_api_key)
    return [m.name.removeprefix("models/") for m in client.models.list() if m.name]


def _model_present(model: str, available: list[str]) -> bool:
    """Match a configured model against a provider listing.

    Tolerates the tag and prefix conventions each provider uses: Ollama reports
    ``qwen2.5:7b`` where ``qwen2.5`` was configured, Google reports
    ``models/gemini-2.5-flash``.
    """
    if not model:
        return False
    candidates = {model, f"{model}:latest"}
    return any(a in candidates or a.split(":")[0] == model.split(":")[0] for a in available)
