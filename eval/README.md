# Evaluation

RAGAS evaluation of the assistant across four difficulty levels.

- `test_queries.yaml`: queries with ground-truth answers
- `run_ragas.py`: runs the queries through the system and scores them
- `results/`: saved score tables per run (e.g. baseline vs improved)

Metrics: Faithfulness and Answer Relevancy (required); Context Precision, Context Recall and Answer Correctness (bonus).

`RETRIEVAL_MODE=naive` (vector-only) is the baseline that hybrid retrieval is measured against, so
both configurations are scored on the same query set.

## Reading the scores

RAGAS metrics are themselves LLM-judged, so the judge is whatever `DEPLOYMENT_MODE` resolves to:

- **Cloud mode** judges with Gemini. Free-tier rate limits are the constraint — a full run is several
  hundred judge calls, so expect throttling and allow retries.
- **On-prem mode** judges with the local model. A local judge is noisier than a hosted one — measured
  with a 7B model; the 35B default has not been compared — and on CPU a run takes tens of minutes.

Absolute scores are therefore only comparable within a mode. The claim these results support is the
**relative** naive-vs-hybrid delta on the same judge, not the absolute number — quote the delta, and
record which mode and model produced a table alongside it.

## Retrieval quality (LLM-free)

`embedding_bench.py` is a second, separate harness, and its numbers are **not**
comparable to the RAGAS scores above: there is no judge, so they are deterministic,
they are identical in both deployment modes, and they score retrieval alone rather
than the answer. It exists to settle embedding questions, which RAGAS is a poor
instrument for — an LLM-judged, quota-limited score measures the whole pipeline,
and the embedding's contribution arrives diluted through RRF, the cross-encoder and
the model's own choice of search terms.

- `retrieval_queries.yaml`: 45 queries in four tiers, with chunk-level gold labels
- `embedding_bench.py`: sweeps `EMBEDDING_MODEL`, scores hit@k / recall@k / MRR / nDCG
- `make bench-embeddings` — each model indexes into its own `eval_docs__*` collection

Labels are chunk-level rather than file-level because file-level saturates: the
corpus is 38 chunks across 6 files and `TOP_K=8` retrieves 21% of it, so file-level
recall@5 is 0.89 for every configuration and cannot separate two embedders.

### Results — 2026-09-24

Full table: [`results/embeddings-2026-09-24.md`](results/embeddings-2026-09-24.md).
`BAAI/bge-small-en-v1.5` (384-dim, the default) is the baseline; MRR on the `naive`
(vector-only) arm is the instrument, because it is the only one that isolates the
embedding.

| model | dim | naive MRR | ΔMRR, 95% CI | shipping MRR | ΔMRR, 95% CI | ms/query | RSS vs base |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | **0.481** | — | 0.581 | — | 8 | — |
| `BAAI/bge-base-en-v1.5` | 768 | 0.438 | [-0.144, +0.060] | 0.568 | [-0.032, +0.004] | 23 | +336MB |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 0.456 | [-0.121, +0.069] | 0.572 | [-0.080, +0.062] | 70 | +1906MB |
| `BAAI/bge-large-en-v1.5` | 1024 | 0.461 | [-0.110, +0.070] | **0.606** | [-0.021, +0.087] | 67 | +2013MB |

**Every confidence interval spans zero.** No larger model is distinguishable from
the 384-dim baseline on this evidence, in either configuration, while costing 3-9x
the query latency and up to 2GB more resident memory. The decision is recorded in
DECISIONS.md D-002: keep `bge-small-en-v1.5`.

Read this as *no detectable difference*, not as proven equivalence. 41 scored
queries is a small sample and the intervals are correspondingly wide — the
benchmark can rule out a large win, not a marginal one.

The control line is BM25 with no embedding at all (naive MRR 0.448). That every
embedder lands within a few points of it, on a 38-chunk corpus where `TOP_K=8`
returns a fifth of everything, is the more useful finding: **retrieval quality here
is bounded by corpus size and chunking, not by the embedding model.**

Config for the table: `TOP_K=8`, `RRF_K=60`, chunk size 1200/overlap 150,
`RERANK_TOP_N` raised to 8 for scoring (production ships 4 — see the `hit@4`
column), corpus `data/raw` at manifest seed 11.

## Running the RAGAS evaluation

```bash
make eval                                                   # both configs, all five metrics
make eval ARGS="--judge-model gemma-4-31b-it --judge-thinking minimal"
make eval ARGS="--configs hybrid+rerank --only E1,A1"       # a subset
make eval ARGS="--no-judge"                                 # answers + behaviour checks, no scoring
```

It needs Qdrant up and indexed (`make up`, `make ingest`). It does not need Redis: the runner
checkpoints in-process, so evaluation sessions stay out of the app's store. DuckDB takes an exclusive
lock, so close any open `duckdb` shell on `data/processed/tables.duckdb` first. Otherwise every
`data_analysis` query reports the store as busy, and the run scores that refusal.

**How the context reaches the judge.** Faithfulness is judged against what the model actually read.
`/v1/chat` doesn't return that: its citations are 200-character snippets of only the cited passages,
and for SQL answers the snippet is the query, not its rows. Scoring against the citations would mark
supported claims as hallucinated. So the runner calls `run_chat_graph` in-process inside
`evidence_tap()`, which returns every collector the agents opened, in full: each retrieved passage
with its source header, and each SQL query with the rows it returned.

**Two configurations, same queries.** `naive` (vector-only, no re-rank) is the baseline;
`hybrid+rerank` is what ships. Retrieval mode only changes `search_documents`, so a real difference
should show on the `document_qa` rows. Movement on a `data_analysis` row is model and judge noise.
The table keeps those rows in, so you can see how large that noise is.

**`n/a` is not 0.** Faithfulness and the two context metrics are undefined when nothing was
retrieved (the prompt-only `small_talk` agent). Faithfulness is also undefined for an answer that
makes no claims, such as a refusal. Those rows are left out of the means instead of scored 0, and the
report says why for each one.

**The behaviour column.** Each query also carries deterministic checks: the expected agent, regexes
the answer must contain, and "tripwire" regexes it must not. RAGAS scores a correct refusal badly
(Answer Relevancy penalises non-committal answers by design), so for the adversarial tier this column
shows whether the system behaved correctly.

**Choosing the judge.** The judge is built by `get_llm`, so it follows `DEPLOYMENT_MODE` and its
tokens are logged under the `ragas_judge` call site. On the cloud free tier the judge's quota is the
binding constraint. One sample needs roughly 15–25 judge calls, because Context Precision makes one
call per retrieved chunk and Answer Relevancy makes three.

| judge | free-tier limit | observed per call | note |
|---|---|---|---|
| `gemini-3.5-flash` | 5 requests/min | 5–70 s | exhausts its quota within one sample |
| `gemma-4-31b-it` | 30 requests/min | 20–90 s | answers for no agent here, so the judge is independent |
| `gemma-4-26b-a4b-it` | 30 requests/min | ~11 s with `--judge-thinking minimal` | answers for every agent, so it would grade its own work |

The report flags a run as self-judged when the judge appears in any agent's model chain.
`--judge-thinking minimal` caps a Google model's reasoning before each verdict, which is most of
each call's latency. `--concurrency` (default 3) limits judge calls in flight per sample.

## Results

_TBD — pending a full run._
