"""Token accounting: every model call is logged and tallied per request."""

import logging

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.config import Settings
from app.llm.providers import get_llm
from app.llm.usage import USAGE_LOGGER, call_site, usage_scope


def _result(model: str, inp: int, out: int, reasoning: int = 0) -> LLMResult:
    message = AIMessage(
        content="x",
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
            "output_token_details": {"reasoning": reasoning},
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_every_model_the_factory_builds_carries_the_logger():
    """Attached at construction, so no call site can forget it."""
    for mode in ("cloud", "onprem"):
        llm = get_llm(settings=Settings(deployment_mode=mode, llm_api_key="k"))
        assert USAGE_LOGGER in llm.callbacks


def test_a_call_is_logged_against_its_call_site(caplog):
    with caplog.at_level(logging.INFO, logger="app.llm.usage"), call_site("data_analysis"):
        USAGE_LOGGER.on_llm_end(_result("gemma-4-26b-a4b-it", 100, 60, reasoning=50))

    fields = caplog.records[-1].fields
    assert fields == {
        "site": "data_analysis", "model": "gemma-4-26b-a4b-it",
        "input_tokens": 100, "output_tokens": 60, "reasoning_tokens": 50,
    }


def test_a_request_tallies_its_calls_per_model():
    """What shows a request that fell through to a billed fallback."""
    with usage_scope() as tally:
        USAGE_LOGGER.on_llm_end(_result("gemma-4-26b-a4b-it", 100, 10))
        USAGE_LOGGER.on_llm_end(_result("gemma-4-26b-a4b-it", 200, 20, reasoning=15))
        USAGE_LOGGER.on_llm_end(_result("gemini-3.5-flash", 50, 5))

    summary = tally.summary()
    assert (summary["input"], summary["output"], summary["reasoning"]) == (350, 35, 15)
    assert summary["by_model"]["gemma-4-26b-a4b-it"]["calls"] == 2
    assert summary["by_model"]["gemini-3.5-flash"]["input"] == 50


def test_a_call_outside_a_request_is_still_logged_but_not_tallied(caplog):
    with caplog.at_level(logging.INFO, logger="app.llm.usage"):
        USAGE_LOGGER.on_llm_end(_result("m", 1, 1))
    assert caplog.records[-1].fields["site"] is None
