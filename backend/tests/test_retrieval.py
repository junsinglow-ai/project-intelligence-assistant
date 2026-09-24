"""Rank fusion and retrieval mode.

Fusion is tested on synthetic rankings so the arithmetic is checked
independently of embeddings or a live index.
"""

import pytest

from app.config import Settings
from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import _reciprocal_rank_fusion


def hit(identifier: str, text: str = "") -> dict:
    return {"id": identifier, "text": text or identifier, "score": 1.0}


def test_fusion_rewards_agreement_between_the_two_rankings():
    """A chunk both retrievers like should beat one that only tops a single list."""
    dense = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("c"), hit("b"), hit("z")]

    order = [r.id for r in _reciprocal_rank_fusion([dense, keyword], k=60)]

    assert order[0] in {"b", "c"}
    assert order.index("b") < order.index("a")


def test_fusion_uses_rank_only_so_incomparable_scores_never_meet():
    """Cosine similarity and BM25 are on different scales; RRF never compares them."""
    dense = [{"id": "a", "text": "a", "score": 0.99}]
    keyword = [{"id": "b", "text": "b", "score": 42.0}]

    fused = _reciprocal_rank_fusion([dense, keyword], k=60)

    assert {r.id for r in fused} == {"a", "b"}
    assert fused[0].score == pytest.approx(fused[1].score)


def test_fusion_keeps_every_candidate_from_both_lists():
    fused = _reciprocal_rank_fusion([[hit("a")], [hit("b")]], k=60)

    assert {r.id for r in fused} == {"a", "b"}


def test_a_larger_k_flattens_the_ranking():
    """rrf_k controls how much being top of a list is worth.

    The two lists have to agree on the order for the gap to be visible: with
    opposed rankings both candidates hold rank 0 and rank 1, so they tie at
    every k by construction.
    """
    lists = [[hit("a"), hit("b")], [hit("a"), hit("b")]]

    def spread(k: int) -> float:
        fused = _reciprocal_rank_fusion(lists, k)
        return fused[0].score - fused[1].score

    assert spread(10) > spread(200) > 0


def test_naive_mode_is_available_as_the_evaluation_baseline():
    """RETRIEVAL_MODE=naive must keep working: hybrid is measured against it."""
    assert Settings(retrieval_mode="naive", llm_api_key="k").retrieval_mode == "naive"
    assert Settings(llm_api_key="k").retrieval_mode == "hybrid"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("R-003", ["r-003"]),                      # risk IDs survive tokenising
        ("ENG-01", ["eng-01"]),
        ("$6,315,000", ["6315000"]),               # so a quoted figure still matches
        ("/v1/chat", ["v1/chat"]),
    ],
)
def test_tokeniser_keeps_identifiers_and_figures_matchable(text, expected):
    assert tokenize(text) == expected


def test_the_client_survives_being_opened_from_several_threads_at_once(isolated_stores):
    """Regression: embedded Qdrant locks its directory, so the lazy singleton
    has to be built under a lock.

    Nothing called retrieval concurrently until `/v1/chat` existed; FastAPI
    serves handlers concurrently and agents run on worker threads, so two
    requests arriving together both found the client unset. Without the lock,
    one thread wins the directory lock and the rest raise.
    """
    import concurrent.futures

    from app.retrieval import vectorstore

    vectorstore.close_client()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            clients = list(pool.map(lambda _: vectorstore.get_client(isolated_stores), range(8)))
    finally:
        vectorstore.close_client()

    assert len(clients) == 8
    assert all(client is clients[0] for client in clients), "one shared client, not eight"
