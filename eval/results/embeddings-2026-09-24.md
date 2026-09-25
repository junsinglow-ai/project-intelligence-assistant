# Embedding model comparison — 2026-09-24

Retrieval-only scores: no LLM, no judge, deterministic. Unlike the RAGAS
tables described in `eval/README.md`, these numbers are comparable across
deployment modes, because nothing here depends on the generation model.

## Setup

- Corpus: `data/raw` — 38 chunks
- Queries: 45 in `eval/retrieval_queries.yaml` (lexical=8, negative=4, paraphrase=15, semantic=18)
- Gold labels are chunk-level, keyed by `(source, citation)`
- Baseline: `BAAI/bge-small-en-v1.5` (384-dim)
- Held fixed across all arms: chunking, `TOP_K=8`, `RRF_K=60`
- `RERANK_TOP_N` is raised to 8 for scoring, because `reranker.rerank` truncates
  its output; production ships 4, so the `hit@4` column is the deployed depth

## Headline

`naive` is vector-only and is the embedding in isolation — the number to read
when asking whether a model embeds this corpus better. `hybrid+rerank` is the
shipping configuration, and shows whether the gain survives RRF and the
cross-encoder.

### naive

| model | dim | hit@1 | hit@3 | hit@4 | recall@8 | MRR | nDCG@8 | ms/query |
|---|---|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` *(baseline)* | 384 | 0.317 | 0.585 | 0.634 | 0.602 | 0.481 | 0.442 | 8 |
| `BAAI/bge-base-en-v1.5` | 768 | 0.293 | 0.585 | 0.610 | 0.561 | 0.438 | 0.414 | 26 |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 0.317 | 0.488 | 0.537 | 0.642 | 0.456 | 0.425 | 65 |
| `BAAI/bge-large-en-v1.5` | 1024 | 0.268 | 0.561 | 0.585 | 0.642 | 0.461 | 0.427 | 64 |
| *BM25 only (no embedding)* | — | 0.317 | 0.512 | 0.585 | 0.484 | 0.448 | 0.377 | — |

### hybrid+rerank

| model | dim | hit@1 | hit@3 | hit@4 | recall@8 | MRR | nDCG@8 | ms/query |
|---|---|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` *(baseline)* | 384 | 0.488 | 0.610 | 0.634 | 0.618 | 0.581 | 0.515 | 757 |
| `BAAI/bge-base-en-v1.5` | 768 | 0.488 | 0.610 | 0.659 | 0.553 | 0.568 | 0.485 | 805 |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 0.488 | 0.585 | 0.683 | 0.589 | 0.572 | 0.499 | 855 |
| `BAAI/bge-large-en-v1.5` | 1024 | 0.537 | 0.634 | 0.659 | 0.598 | 0.606 | 0.522 | 864 |
| *BM25 only (no embedding)* | — | 0.317 | 0.512 | 0.585 | 0.484 | 0.448 | 0.377 | — |

## Is any difference real?

Paired bootstrap (5000 draws) on the MRR difference from the baseline, over the
41 scored queries. **An interval that spans zero means the two models are
indistinguishable on this evidence** — the point estimate alone is not a result at
this sample size, where a single query is worth roughly 0.02 MRR.

### naive

| model | ΔMRR vs baseline | 95% CI | distinguishable? |
|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | -0.043 | [-0.144, +0.060] | no |
| `mixedbread-ai/mxbai-embed-large-v1` | -0.025 | [-0.121, +0.069] | no |
| `BAAI/bge-large-en-v1.5` | -0.021 | [-0.110, +0.070] | no |

### hybrid+rerank

| model | ΔMRR vs baseline | 95% CI | distinguishable? |
|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | -0.013 | [-0.032, +0.004] | no |
| `mixedbread-ai/mxbai-embed-large-v1` | -0.009 | [-0.080, +0.062] | no |
| `BAAI/bge-large-en-v1.5` | +0.025 | [-0.021, +0.087] | no |

## By tier (MRR)

`lexical` reuses the documents' own wording, so BM25 already does well there.
`paraphrase` and `semantic` shift the vocabulary off the source text, which is
where a stronger embedder should show up if it is worth anything.

The BM25-only row is the control: it should score well on `lexical` and drop on
`paraphrase` and `semantic`. If it does not drop, those tiers are not actually
defeating lexical matching and the comparison is not measuring the embedder.

### naive

| model | lexical | paraphrase | semantic |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 0.442 | 0.513 | 0.472 |
| `BAAI/bge-base-en-v1.5` | 0.406 | 0.392 | 0.491 |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.453 | 0.516 | 0.408 |
| `BAAI/bge-large-en-v1.5` | 0.478 | 0.468 | 0.447 |
| *BM25 only* | 0.483 | 0.408 | 0.466 |

### hybrid+rerank

| model | lexical | paraphrase | semantic |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 0.516 | 0.497 | 0.680 |
| `BAAI/bge-base-en-v1.5` | 0.479 | 0.489 | 0.674 |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.510 | 0.511 | 0.650 |
| `BAAI/bge-large-en-v1.5` | 0.521 | 0.520 | 0.715 |
| *BM25 only* | 0.483 | 0.408 | 0.466 |

## Unanswerable questions

Mean top-1 similarity for the `negative` tier, where nothing in the corpus
answers the question. Lower is better: a model that returns a confident score
for an unanswerable question gives the agent more to hallucinate from. Scores
are not comparable across configurations — `hybrid+rerank` reports a
cross-encoder score, `naive` a cosine similarity.

| model | naive | hybrid+rerank |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 0.611 | 0.007 |
| `BAAI/bge-base-en-v1.5` | 0.582 | 0.010 |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.529 | 0.008 |
| `BAAI/bge-large-en-v1.5` | 0.559 | 0.008 |

## Cost

| model | dim | ingest | peak RSS | vs baseline |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 2.5s | 801MB | +0MB |
| `BAAI/bge-base-en-v1.5` | 768 | 9.1s | 1099MB | +297MB |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 27.5s | 2180MB | +1379MB |
| `BAAI/bge-large-en-v1.5` | 1024 | 28.0s | 2196MB | +1395MB |

Peak RSS is this benchmark process at its high-water mark, so it also carries
the ingestion libraries (pandas, pdfplumber, openpyxl) and the FlashRank
cross-encoder. It is therefore **higher than the serving footprint** measured in
ARCHITECTURE.md §6 and is not a substitute for it — read the *delta* between
models, which is the part attributable to the embedding weights. Cloud Run is
currently deployed at `--memory 1Gi` against a measured 348MB serving peak, so a
model whose delta eats that headroom needs the host resized before promotion.
