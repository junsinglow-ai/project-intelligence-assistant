"""Session storage bounds, and follow-up rewriting with its fallbacks."""

import concurrent.futures

from app.agents.base import AgentInput
from app.memory.session_store import (
    SessionStore,
    StandaloneQuestion,
    rewrite_follow_up,
    valid_session_id,
)

HISTORY = [
    {"role": "user", "content": "What is the approved budget?"},
    {"role": "assistant", "content": "6,315,000 USD."},
]


def test_a_minted_session_id_is_one_the_api_will_accept():
    assert valid_session_id(SessionStore().new_session())


def test_a_session_id_from_a_client_is_constrained(session_store):
    assert not valid_session_id("has spaces")
    assert not valid_session_id("../../etc/passwd")
    assert not valid_session_id("x" * 65)
    assert not valid_session_id("")


def test_history_is_bounded_to_the_most_recent_turns():
    store = SessionStore(max_turns=3)
    for index in range(6):
        store.append("s", f"q{index}", f"a{index}", "document_qa")

    assert [turn.question for turn in store.transcript("s")] == ["q3", "q4", "q5"]


def test_the_least_recently_used_session_is_evicted():
    """Without this bound, a client inventing an ID per request is a leak."""
    store = SessionStore(max_sessions=2)
    for name in ("a", "b", "c"):
        store.append(name, "q", "a", "document_qa")

    assert store.transcript("a") is None
    assert store.transcript("c") is not None


def test_reading_a_session_keeps_it_alive():
    store = SessionStore(max_sessions=2)
    store.append("a", "q", "a", "document_qa")
    store.append("b", "q", "a", "document_qa")
    store.transcript("a")                       # touch it, so "b" is now oldest
    store.append("c", "q", "a", "document_qa")

    assert store.transcript("a") is not None
    assert store.transcript("b") is None


def test_history_is_the_shape_an_agent_accepts(session_store):
    session_store.append("s", "What is the budget?", "6,315,000 USD.", "data_analysis")

    history = session_store.history("s")

    assert history == [{"role": "user", "content": "What is the budget?"},
                       {"role": "assistant", "content": "6,315,000 USD."}]
    AgentInput(question="And the spend?", session_id="s", history=history)


def test_an_unknown_session_has_no_history_but_is_not_an_error(session_store):
    assert session_store.history("unknown") == []
    assert session_store.transcript("unknown") is None


def test_concurrent_appends_do_not_lose_turns():
    """FastAPI serves handlers concurrently and agents run on worker threads."""
    store = SessionStore(max_turns=100)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda i: store.append("s", f"q{i}", "a", "document_qa"), range(50)))

    assert len(store.transcript("s")) == 50


async def test_the_first_question_is_not_sent_to_the_rewriter(scripted_llm):
    """The common case, and a whole model call saved on every new conversation."""
    model = scripted_llm()

    assert await rewrite_follow_up("What is the budget?", []) == "What is the budget?"
    assert model.calls == []


async def test_a_follow_up_is_rewritten_into_a_standalone_question(scripted_llm):
    model = scripted_llm(StandaloneQuestion(question="How much of the approved budget is spent?"))

    rewritten = await rewrite_follow_up("And how much of it is spent?", HISTORY)

    assert rewritten == "How much of the approved budget is spent?"
    assert "6,315,000 USD." in model.calls[0].text      # the history reached the model


async def test_a_rewriter_failure_falls_back_to_the_question_as_asked(scripted_llm):
    scripted_llm(RuntimeError("provider unreachable"))

    assert await rewrite_follow_up("And the spend?", HISTORY) == "And the spend?"


async def test_an_empty_rewrite_is_discarded(scripted_llm):
    scripted_llm(StandaloneQuestion(question="   "))

    assert await rewrite_follow_up("And the spend?", HISTORY) == "And the spend?"


async def test_a_rewrite_with_no_words_is_discarded(scripted_llm):
    scripted_llm(StandaloneQuestion(question="???"))

    assert await rewrite_follow_up("And the spend?", HISTORY) == "And the spend?"


async def test_a_rewriter_that_started_explaining_is_discarded(scripted_llm):
    """A rewrite far longer than the question means the model answered instead."""
    scripted_llm(StandaloneQuestion(question="Certainly! " + "context " * 200))

    assert await rewrite_follow_up("And the spend?", HISTORY) == "And the spend?"
