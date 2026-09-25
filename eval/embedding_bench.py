"""Benchmark embedding models on retrieval quality, without an LLM.

Answers the question behind DECISIONS.md D-002: is a stronger
embedder worth a re-index? It sweeps several fastembed models over the same
corpus and the same labelled queries, and scores what retrieval returns --
no generation, no judge, no quota.

Why not RAGAS: RAGAS metrics are LLM-judged, so they are noisy, rate-limited and
comparable only within a deployment mode (see eval/README.md). They also measure
the whole pipeline, which dilutes the embedding's contribution behind RRF, the
cross-encoder and the model's own wording. These metrics are deterministic and
isolate the retriever.

Two configurations are scored for every model:

    naive  + no rerank   the embedding in isolation -- the primary instrument
    hybrid + rerank      the shipping configuration -- does any gain survive?

Each model gets its own Qdrant collection for free, because `Settings.collection_name`
stamps the model into the name, so arms coexist and nothing is torn down between them.

Each model runs in its own subprocess. That keeps peak RSS attributable to one
model (`ru_maxrss` is a high-water mark that never falls) and stops four sets of
ONNX weights -- around 2.1GB -- piling up in one interpreter.

Usage:
    make bench-embeddings                     # full sweep
    python eval/embedding_bench.py --models BAAI/bge-small-en-v1.5
    python eval/embedding_bench.py --worker BAAI/bge-base-en-v1.5   # internal
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

# The backend is `package = false` with pythonpath=["."] relative to backend/, so
# `app` is importable from there rather than installed. This script is run from
# the repository root (and from backend/ by the Makefile), so anchor explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

QUERY_FILE = _REPO_ROOT / "eval" / "retrieval_queries.yaml"
RESULTS_DIR = _REPO_ROOT / "eval" / "results"

# Strongest-first is not the point here; this is the order the table is read in,
# from the current default upwards by size.
DEFAULT_MODELS = [
    "BAAI/bge-small-en-v1.5",             # current default, 384-dim, 0.067GB
    "BAAI/bge-base-en-v1.5",              # 768-dim, 0.21GB
    "mixedbread-ai/mxbai-embed-large-v1",  # 1024-dim, 0.64GB
    "BAAI/bge-large-en-v1.5",             # 1024-dim, 1.2GB
]

# Held fixed across every arm, so the embedding is the only variable. Changing
# any of these invalidates comparison with an earlier table, which is why they
# are stamped into the report.
CONFIGS = {
    "naive": {"retrieval_mode": "naive", "rerank_enabled": False},
    "hybrid+rerank": {"retrieval_mode": "hybrid", "rerank_enabled": True},
}

# `reranker.rerank` cuts its output to `rerank_top_n`, so with the shipping value
# of 4 every metric above k=4 would silently equal the k=4 one for that arm. The
# benchmark raises it to the evaluation depth so both arms return a full ranking
# and the columns mean the same thing. Production still ships RERANK_TOP_N=4, so
# k<=4 are the production-relevant cutoffs -- hence 4 is one of them.
CUTOFFS = (1, 3, 4, 8)
EVAL_DEPTH = max(CUTOFFS)

# Pinned rather than inherited. `Settings()` still reads the repo .env even when
# constructed directly (only get_settings()'s lru_cache is bypassed), so anything
# left unpinned would make a run depend on the machine it was run on.
FIXED = {
    "embedding_provider": "fastembed",
    "top_k": EVAL_DEPTH,
    "rerank_top_n": EVAL_DEPTH,
    "rrf_k": 60,
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "table_rows_per_chunk": 4,
    # The benchmark indexes into its own collections, so a sweep never creates or
    # orphans a `project_docs__*` collection that the running application uses.
    "qdrant_collection": "eval_docs",
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def score_query(ranked: list[tuple[str, str]], gold: set[tuple[str, str]],
                cutoffs=CUTOFFS) -> dict:
    """Score one ranked list of (source, citation) against a gold set.

    Gold is a set because several facts are answerable from more than one place
    -- the budget total appears in the xlsx, the CSV export and the Q3 report --
    so retrieving any of them is a hit, and `recall` reports how many were found.
    """
    relevance = [1 if chunk in gold else 0 for chunk in ranked]
    first = next((i + 1 for i, rel in enumerate(relevance) if rel), 0)

    out: dict[str, float] = {"mrr": 1.0 / first if first else 0.0}
    for k in cutoffs:
        found = sum(relevance[:k])
        out[f"hit@{k}"] = 1.0 if found else 0.0
        out[f"recall@{k}"] = found / len(gold)

    ideal = _dcg([1] * min(len(gold), max(cutoffs)))
    out["ndcg@8"] = _dcg(relevance[: max(cutoffs)]) / ideal if ideal else 0.0
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_delta(baseline: list[float], candidate: list[float],
                    draws: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI on the mean difference candidate - baseline.

    The query set is small, so a difference of one or two queries moves a mean
    metric by several points and looks like a result. Resampling the *pairs*
    (each query contributes both scores, so the models see identical queries)
    gives the interval that says whether a gap is a gap. An interval spanning
    zero means the two models are indistinguishable on this evidence, which is
    the honest reading however tempting the point estimate looks.
    """
    import random

    deltas = [c - b for b, c in zip(baseline, candidate)]
    point = _mean(deltas)
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(_mean([deltas[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return point, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


# --------------------------------------------------------------------------
# Worker: one model, both configurations
# --------------------------------------------------------------------------

def run_model(model: str, queries: list[dict], corpus: str | None) -> dict:
    """Ingest the corpus under one embedding model and score every configuration."""
    import resource
    import tempfile

    from app.config import Settings
    from app.ingestion.pipeline import ingest_directory
    from app.retrieval import bm25, hybrid

    # ingest_file writes the cleaned CSV/XLSX rows into DuckDB as well as Qdrant,
    # upserting on natural keys (cost_code, risk_id). Left at the default that
    # would be data/processed/tables.duckdb -- the store the Data Analysis agent
    # answers from -- so a benchmark over a second corpus would quietly merge
    # another project's risks and cost lines into the live one.
    scratch = tempfile.mkdtemp(prefix="embedding-bench-")
    tabular_db = str(Path(scratch) / "tables.duckdb")

    def settings_for(**overrides) -> Settings:
        # Settings(), not get_settings(): the latter is lru_cached, so every arm
        # would silently share the first arm's embedding model.
        kwargs = {"embedding_model": model, "tabular_db_path": tabular_db, **FIXED}
        if corpus:
            kwargs["data_raw_dir"] = corpus
        return Settings(**(kwargs | overrides))

    base = settings_for()
    if not base.qdrant_url:
        raise SystemExit(
            "QDRANT_URL is empty, so Qdrant would run embedded and take an exclusive "
            "lock on its directory -- which the running API already holds. Point "
            "QDRANT_URL at the server (make up) before benchmarking."
        )

    started = time.monotonic()
    report = ingest_directory(settings=base)
    ingest_s = time.monotonic() - started

    scored: dict[str, dict] = {}
    for name, overrides in CONFIGS.items():
        settings = settings_for(**overrides)
        # bm25 caches its index in module globals that are NOT keyed by collection,
        # so without this the next arm scores the previous arm's keyword index.
        bm25.invalidate()

        per_query, negatives = [], []
        started = time.monotonic()
        for query in queries:
            hits = hybrid.retrieve(query["question"], settings=settings,
                                   limit=EVAL_DEPTH)
            if query["tier"] == "negative":
                # Nothing is retrievable, so recall is undefined. What is worth
                # watching is how confident the top hit is -- a model that returns
                # a strong score for an unanswerable question is the failure mode.
                negatives.append(hits[0].score if hits else 0.0)
                continue
            ranked = [(h.source, h.citation) for h in hits]
            gold = {(g["source"], g["citation"]) for g in query["gold"]}
            per_query.append({"id": query["id"], "tier": query["tier"],
                              **score_query(ranked, gold)})
        latency_ms = (time.monotonic() - started) * 1000 / max(len(queries), 1)

        metrics = [k for k in per_query[0] if k not in {"id", "tier"}]
        tiers = sorted({row["tier"] for row in per_query})
        scored[name] = {
            "overall": {m: _mean([r[m] for r in per_query]) for m in metrics},
            "by_tier": {
                t: {m: _mean([r[m] for r in per_query if r["tier"] == t]) for m in metrics}
                for t in tiers
            },
            "negative_top1_score": _mean(negatives),
            "ms_per_query": latency_ms,
            "per_query": per_query,
        }

    # BM25 alone, as a model-independent floor. Its purpose is to show that the
    # paraphrase and semantic tiers really do defeat lexical matching -- if BM25
    # scores as well there as on the lexical tier, those queries are not stressing
    # the embedder and the comparison is not measuring what it claims to. It reads
    # the same chunk text in every arm, so it should come out identical for every
    # model; that it does is a self-check on the sweep.
    bm25.invalidate()
    bm25.ensure_index(base)
    reference, negatives = [], []
    for query in queries:
        raw = bm25.search(query["question"], limit=EVAL_DEPTH, settings=base)
        if query["tier"] == "negative":
            negatives.append(raw[0]["score"] if raw else 0.0)
            continue
        ranked = [(d.get("source", ""), d.get("citation", "")) for d in raw]
        gold = {(g["source"], g["citation"]) for g in query["gold"]}
        reference.append({"id": query["id"], "tier": query["tier"],
                          **score_query(ranked, gold)})
    metrics = [k for k in reference[0] if k not in {"id", "tier"}]
    scored["bm25-only"] = {
        "overall": {m: _mean([r[m] for r in reference]) for m in metrics},
        "by_tier": {
            t: {m: _mean([r[m] for r in reference if r["tier"] == t]) for m in metrics}
            for t in sorted({r["tier"] for r in reference})
        },
        "negative_top1_score": _mean(negatives),
        "ms_per_query": 0.0,
        "per_query": reference,
    }

    shutil.rmtree(scratch, ignore_errors=True)

    return {
        "model": model,
        "dimension": len(__import__("app.llm.providers", fromlist=["x"])
                         .get_embeddings(base).embed_query("dimension probe")),
        "collection": base.collection_name,
        "chunks_indexed": report.chunks_indexed,
        "ingest_seconds": ingest_s,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "configs": scored,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render(results: list[dict], queries: list[dict], corpus: str) -> str:
    baseline = results[0]
    lines = [
        f"# Embedding model comparison — {date.today().isoformat()}",
        "",
        "Retrieval-only scores: no LLM, no judge, deterministic. Unlike the RAGAS",
        "tables described in `eval/README.md`, these numbers are comparable across",
        "deployment modes, because nothing here depends on the generation model.",
        "",
        "## Setup",
        "",
        f"- Corpus: `{corpus}` — {baseline['chunks_indexed']} chunks",
        f"- Queries: {len(queries)} in `eval/retrieval_queries.yaml` "
        f"({', '.join(f'{t}={sum(1 for q in queries if q['tier'] == t)}' for t in sorted({q['tier'] for q in queries}))})",
        "- Gold labels are chunk-level, keyed by `(source, citation)`",
        f"- Baseline: `{baseline['model']}` ({baseline['dimension']}-dim)",
        "- Held fixed across all arms: chunking, `TOP_K=8`, `RRF_K=60`",
        "- `RERANK_TOP_N` is raised to 8 for scoring, because `reranker.rerank` truncates",
        "  its output; production ships 4, so the `hit@4` column is the deployed depth",
        "",
        "## Headline",
        "",
        "`naive` is vector-only and is the embedding in isolation — the number to read",
        "when asking whether a model embeds this corpus better. `hybrid+rerank` is the",
        "shipping configuration, and shows whether the gain survives RRF and the",
        "cross-encoder.",
        "",
    ]

    for config in CONFIGS:
        lines += [
            f"### {config}",
            "",
            "| model | dim | hit@1 | hit@3 | hit@4 | recall@8 | MRR | nDCG@8 | ms/query |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for result in results:
            scores = result["configs"][config]["overall"]
            marker = " *(baseline)*" if result is baseline else ""
            lines.append(
                f"| `{result['model']}`{marker} | {result['dimension']} | "
                f"{scores['hit@1']:.3f} | {scores['hit@3']:.3f} | {scores['hit@4']:.3f} | "
                f"{scores['recall@8']:.3f} | {scores['mrr']:.3f} | {scores['ndcg@8']:.3f} | "
                f"{result['configs'][config]['ms_per_query']:.0f} |"
            )
        reference = baseline["configs"]["bm25-only"]["overall"]
        lines.append(
            f"| *BM25 only (no embedding)* | — | {reference['hit@1']:.3f} | "
            f"{reference['hit@3']:.3f} | {reference['hit@4']:.3f} | "
            f"{reference['recall@8']:.3f} | {reference['mrr']:.3f} | "
            f"{reference['ndcg@8']:.3f} | — |"
        )
        lines.append("")

    lines += [
        "## Is any difference real?",
        "",
        "Paired bootstrap (5000 draws) on the MRR difference from the baseline, over the",
        f"{len([q for q in queries if q['tier'] != 'negative'])} scored queries. **An interval that spans zero means the two models are",
        "indistinguishable on this evidence** — the point estimate alone is not a result at",
        "this sample size, where a single query is worth roughly 0.02 MRR.",
        "",
    ]
    for config in CONFIGS:
        lines += [f"### {config}", "",
                  "| model | ΔMRR vs baseline | 95% CI | distinguishable? |",
                  "|---|---|---|---|"]
        base_scores = {r["id"]: r["mrr"] for r in baseline["configs"][config]["per_query"]}
        for result in results[1:]:
            rows = result["configs"][config]["per_query"]
            ids = [r["id"] for r in rows]
            point, lo, hi = bootstrap_delta([base_scores[i] for i in ids],
                                            [r["mrr"] for r in rows])
            verdict = "no" if lo <= 0 <= hi else ("**better**" if lo > 0 else "**worse**")
            lines.append(f"| `{result['model']}` | {point:+.3f} | "
                         f"[{lo:+.3f}, {hi:+.3f}] | {verdict} |")
        lines.append("")

    lines += ["## By tier (MRR)", "",
              "`lexical` reuses the documents' own wording, so BM25 already does well there.",
              "`paraphrase` and `semantic` shift the vocabulary off the source text, which is",
              "where a stronger embedder should show up if it is worth anything.",
              "",
              "The BM25-only row is the control: it should score well on `lexical` and drop on",
              "`paraphrase` and `semantic`. If it does not drop, those tiers are not actually",
              "defeating lexical matching and the comparison is not measuring the embedder.",
              ""]
    tiers = sorted({q["tier"] for q in queries} - {"negative"})
    for config in CONFIGS:
        lines += [f"### {config}", "",
                  "| model | " + " | ".join(tiers) + " |",
                  "|---" * (len(tiers) + 1) + "|"]
        for result in results:
            by_tier = result["configs"][config]["by_tier"]
            cells = " | ".join(f"{by_tier.get(t, {}).get('mrr', 0):.3f}" for t in tiers)
            lines.append(f"| `{result['model']}` | {cells} |")
        reference = baseline["configs"]["bm25-only"]["by_tier"]
        cells = " | ".join(f"{reference.get(t, {}).get('mrr', 0):.3f}" for t in tiers)
        lines.append(f"| *BM25 only* | {cells} |")
        lines.append("")

    lines += [
        "## Unanswerable questions",
        "",
        "Mean top-1 similarity for the `negative` tier, where nothing in the corpus",
        "answers the question. Lower is better: a model that returns a confident score",
        "for an unanswerable question gives the agent more to hallucinate from. Scores",
        "are not comparable across configurations — `hybrid+rerank` reports a",
        "cross-encoder score, `naive` a cosine similarity.",
        "",
        "| model | " + " | ".join(CONFIGS) + " |",
        "|---" * (len(CONFIGS) + 1) + "|",
    ]
    for result in results:
        cells = " | ".join(f"{result['configs'][c]['negative_top1_score']:.3f}" for c in CONFIGS)
        lines.append(f"| `{result['model']}` | {cells} |")

    lines += ["", "## Cost", "",
              "| model | dim | ingest | peak RSS | vs baseline |", "|---|---|---|---|---|"]
    for result in results:
        delta = result["peak_rss_mb"] - baseline["peak_rss_mb"]
        lines.append(
            f"| `{result['model']}` | {result['dimension']} | "
            f"{result['ingest_seconds']:.1f}s | {result['peak_rss_mb']:.0f}MB | "
            f"{delta:+.0f}MB |"
        )
    lines += ["",
              "Peak RSS is this benchmark process at its high-water mark, so it also carries",
              "the ingestion libraries (pandas, pdfplumber, openpyxl) and the FlashRank",
              "cross-encoder. It is therefore **higher than the serving footprint** measured in",
              "ARCHITECTURE.md §6 and is not a substitute for it — read the *delta* between",
              "models, which is the part attributable to the embedding weights. Cloud Run is",
              "currently deployed at `--memory 1Gi` against a measured 348MB serving peak, so a",
              "model whose delta eats that headroom needs the host resized before promotion.",
              ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--corpus", default=None,
                        help="override DATA_RAW_DIR, e.g. data/samples/commercial-building")
    parser.add_argument("--out", default=None, help="output path for the report")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--report-only", default=None,
                        help="re-render the report from a saved results JSON, without re-running")
    args = parser.parse_args()

    import yaml
    spec = yaml.safe_load(QUERY_FILE.read_text())
    queries = spec["queries"]

    if args.worker:
        json.dump(run_model(args.worker, queries, args.corpus), sys.stdout)
        return

    corpus = args.corpus or "data/raw"

    if args.report_only:
        results = json.loads(Path(args.report_only).read_text())
        out = args.out or str(args.report_only).replace(".json", ".md")
        Path(out).write_text(render(results, queries, corpus))
        print(f"wrote {out}")
        return

    results = []
    for model in args.models:
        print(f"==> {model}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", model]
        if args.corpus:
            command += ["--corpus", args.corpus]
        # Ingestion and retrieval log to stderr; only stdout carries the payload.
        finished = subprocess.run(command, capture_output=True, text=True,
                                  cwd=_REPO_ROOT, env=os.environ)
        if finished.returncode != 0:
            sys.stderr.write(finished.stderr)
            raise SystemExit(f"{model} failed with exit code {finished.returncode}")
        result = json.loads(finished.stdout)
        results.append(result)
        naive = result["configs"]["naive"]["overall"]
        print(f"    dim={result['dimension']} chunks={result['chunks_indexed']} "
              f"naive MRR={naive['mrr']:.3f} hit@1={naive['hit@1']:.3f} "
              f"peak={result['peak_rss_mb']:.0f}MB", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out or RESULTS_DIR / f"embeddings-{date.today().isoformat()}.md"
    Path(stem).write_text(render(results, queries, corpus))
    Path(str(stem).replace(".md", ".json")).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {stem}")


if __name__ == "__main__":
    main()
