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
- **On-prem mode** judges with the local model. A 7B judge is measurably noisier than a hosted one,
  and on CPU a run takes tens of minutes.

Absolute scores are therefore only comparable within a mode. The claim these results support is the
**relative** naive-vs-hybrid delta on the same judge, not the absolute number — quote the delta, and
record which mode and model produced a table alongside it.

## Results

_TBD_
