"""Run the RAGAS evaluation.

Loads eval/test_queries.yaml, sends each query through the pipeline,
collects answers and retrieved contexts, scores them with RAGAS and writes
a results table to eval/results/.

Why in-process rather than over HTTP: Faithfulness judges the answer against
the context the model actually read, and `/v1/chat` does not return it. The
citations it does return are 200-character snippets of only the passages the
answer cited -- and, for a SQL answer, the query rather than its rows -- so
scoring against them would mark supported claims as hallucinated. The runner
therefore calls the same graph the endpoint calls (`run_chat_graph`) inside an
`evidence_tap()`, which hands back every collector the agents opened, in full.

Two configurations are scored on the same queries, matching the embedding
benchmark's arms:

    naive          RETRIEVAL_MODE=naive, no re-rank -- the baseline
    hybrid+rerank  RETRIEVAL_MODE=hybrid, re-rank on -- what ships

Retrieval mode only changes `search_documents`, so the delta is expected on the
document_qa rows; a data_analysis row moving between arms is judge noise and
model variance, which is worth seeing in the table rather than hiding.

The judge is built by `app.llm.providers.get_llm`, so it follows
DEPLOYMENT_MODE like everything else (a local model on-prem, no external call),
and its tokens are logged under the `ragas_judge` call site. Answer Relevancy
and Answer Correctness also need embeddings; they use the project's own
FastEmbed model, which runs locally in both modes. RAGAS telemetry is switched
off before RAGAS is imported.

Usage:
    make eval                                        # both configs, all metrics
    make eval ARGS="--configs hybrid+rerank --only E1,A1"
    make eval ARGS="--no-judge"                      # answers + behaviour checks only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path

# `app` is importable from backend/ (the project is `package = false`), and the
# Makefile runs this from there -- but anchor explicitly so it also runs from
# the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

# Before anything imports RAGAS: it otherwise reports usage to its vendor, which
# the on-prem mode's "no external calls" requirement rules out.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
# Checkpoint in-process. Evaluation sessions do not belong in the Redis the
# running app uses, and the run should not depend on it being up.
os.environ["REDIS_URL"] = ""

QUERY_FILE = _REPO_ROOT / "eval" / "test_queries.yaml"
RESULTS_DIR = _REPO_ROOT / "eval" / "results"

CONFIGS = {
    "naive": {"RETRIEVAL_MODE": "naive", "RERANK_ENABLED": "false"},
    "hybrid+rerank": {"RETRIEVAL_MODE": "hybrid", "RERANK_ENABLED": "true"},
}
BASELINE, CANDIDATE = "naive", "hybrid+rerank"

METRICS = ("faithfulness", "answer_relevancy", "context_precision",
           "context_recall", "answer_correctness")
SHORT = {"faithfulness": "Faith", "answer_relevancy": "AnsRel",
         "context_precision": "CtxPrec", "context_recall": "CtxRec",
         "answer_correctness": "AnsCorr"}
# Undefined without retrieved context: faithfulness has nothing to be faithful
# to, and the two context metrics have no context to score. Reported as n/a
# rather than 0, because a prompt-only agent declining is not a retrieval miss.
NEEDS_CONTEXT = {"faithfulness", "context_precision", "context_recall"}
LEVELS = ("easy", "medium", "hard", "adversarial")


# --------------------------------------------------------------------------
# Queries and behaviour checks
# --------------------------------------------------------------------------

def load_queries(path: Path, only: set[str] | None) -> list[dict]:
    import yaml

    queries = yaml.safe_load(path.read_text())["queries"]
    for q in queries:
        if not q.get("question") or not q.get("ground_truth"):
            raise SystemExit(f"{q.get('id')}: question and ground_truth are both required")
        agents = q.get("expected_agent") or []
        q["expected_agent"] = [agents] if isinstance(agents, str) else list(agents)
    if only:
        queries = [q for q in queries if q["id"] in only]
    return queries


def check_behaviour(query: dict, agent: str, answer: str) -> tuple[bool, list[str]]:
    """Deterministic pass/fail alongside the judge: right agent, right content.

    This is what scores the adversarial tier fairly. RAGAS rates a correct
    refusal as non-committal (Answer Relevancy 0) and may find no claims in it
    to check (Faithfulness undefined), so the scores alone cannot say whether
    declining was the right call.
    """
    failures = []
    if query["expected_agent"] and agent not in query["expected_agent"]:
        failures.append(f"routed to {agent}, expected {'/'.join(query['expected_agent'])}")
    for pattern in query.get("must_include") or []:
        if not re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"missing /{pattern}/")
    for pattern in query.get("must_not_include") or []:
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"contains /{pattern}/")
    return not failures, failures


# --------------------------------------------------------------------------
# Running the pipeline
# --------------------------------------------------------------------------

def apply_config(name: str) -> None:
    """Switch retrieval configuration for everything that reads get_settings()."""
    from app.config import get_settings

    os.environ.update(CONFIGS[name])
    get_settings.cache_clear()


def render_context(item) -> str:
    """One evidence item as the text the judge checks the answer against.

    The same content the model was shown: a passage with its source header
    (as `render_passages` formats it), or a query with the rows it returned.
    """
    from app.skills.tabular import SqlResult, render_rows

    if isinstance(item, SqlResult):
        return f"SQL query: {item.sql}\nResult:\n{render_rows(item.columns, item.rows)}"
    if hasattr(item, "text") and hasattr(item, "source"):
        return f"{item.source} - {item.citation}\n{item.text}"
    return str(item)


async def answer_one(query: dict, config: str) -> dict:
    from app.graph.build import run_chat_graph
    from app.skills.evidence import evidence_tap

    # A fresh session per query: no history, so nothing is rewritten and each
    # query is scored on its own.
    session = f"eval-{config}-{query['id']}-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()
    with evidence_tap() as collectors:
        state = await run_chat_graph(query["question"], session, needs_rewrite=False)
    elapsed = time.monotonic() - started

    result = state["result"]
    contexts: list[str] = []
    for evidence in collectors:
        for item in evidence.items:
            text = render_context(item)
            if text not in contexts:
                contexts.append(text)

    ok, failures = check_behaviour(query, result.agent, result.answer)
    return {
        "id": query["id"],
        "difficulty": query["difficulty"],
        "question": query["question"],
        "ground_truth": query["ground_truth"],
        "agent": result.agent,
        "model": result.model,
        "answer": result.answer,
        "contexts": contexts,
        "citations": len(result.citations),
        "tool_calls": result.metadata.get("tool_calls"),
        "citation_mode": result.metadata.get("citation_mode"),
        "routing": result.metadata.get("routing"),
        "error": result.metadata.get("error"),
        "latency_s": round(elapsed, 2),
        "behaviour_ok": ok,
        "behaviour_failures": failures,
        "scores": {},
        "score_notes": {},
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def build_metrics(names: list[str], judge_model: str, timeout: int, thinking: str | None):
    """RAGAS metrics wired to the project's own model and embeddings.

    The LangChain-wrapper metrics rather than `ragas.metrics.collections`: the
    collections take an Instructor client, which would mean building the judge
    outside `app.llm.providers` -- and so outside the usage logging and the
    on-prem switch. The wrappers take the `BaseChatModel` get_llm returns.
    """
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ragas\..*")
    warnings.filterwarnings("ignore", message=r".*ragas\.metrics.*deprecated.*")

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig

    from app.llm.providers import get_embeddings, get_llm

    # bypass_n: Answer Relevancy asks for `strictness` candidates in one call,
    # which Gemini rejects ("Multiple candidates is not enabled") -- make them
    # separate calls instead, which every provider accepts.
    llm = get_llm(judge_model)
    if thinking:
        # Gemini/Gemma reason before every judgement by default, which is most of
        # a judge call's latency; the judgements are extraction and yes/no
        # verdicts that do not need it. Google-only knob, so copy rather than
        # widen get_llm's signature for one caller.
        llm = llm.model_copy(update={"thinking_level": thinking})
    judge = LangchainLLMWrapper(llm, bypass_n=True)
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    factories = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "answer_correctness": AnswerCorrectness,
    }
    run_config = RunConfig(timeout=timeout, max_retries=6, max_wait=60)
    metrics = {}
    for name in names:
        metric = factories[name]()
        metric.llm = judge
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
        metric.init(run_config)
        metrics[name] = metric
    return metrics


async def score_one(record: dict, metrics: dict, timeout: int, concurrency: int) -> None:
    """Score one answer on every metric, `concurrency` judgements at a time.

    Each metric is several judge calls (Context Precision is one per retrieved
    chunk), so scoring them one after another dominates the run. The cap is
    what keeps a per-minute quota from turning the parallelism into 429s.
    """
    from ragas import SingleTurnSample

    from app.llm.usage import call_site

    sample = SingleTurnSample(
        user_input=record["question"],
        response=record["answer"],
        retrieved_contexts=record["contexts"],
        reference=record["ground_truth"],
    )
    gate = asyncio.Semaphore(concurrency)

    async def judge(name: str, metric) -> None:
        if name in NEEDS_CONTEXT and not record["contexts"]:
            record["scores"][name] = None
            record["score_notes"][name] = "no retrieved context"
            return
        try:
            async with gate:
                value = await metric.single_turn_ascore(sample, timeout=timeout)
        except Exception as exc:  # one failed judgement must not end the run
            record["scores"][name] = None
            record["score_notes"][name] = f"judge failed: {type(exc).__name__}: {exc}"[:300]
            return
        if value is None or (isinstance(value, float) and math.isnan(value)):
            record["scores"][name] = None
            # Faithfulness divides by the number of claims; a refusal has none.
            record["score_notes"][name] = "undefined (no statements to judge)"
        else:
            record["scores"][name] = round(float(value), 3)

    with call_site("ragas_judge"):
        await asyncio.gather(*(judge(name, metric) for name, metric in metrics.items()))
    # Report in the declared order, whatever order the judgements finished in.
    record["scores"] = {name: record["scores"][name] for name in metrics}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def mean(values) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def fmt_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def per_query_table(records: list[dict], metric_names: list[str]) -> list[str]:
    head = ["ID", "Level", "Agent", *(SHORT[m] for m in metric_names), "Behaviour"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in records:
        behaviour = "pass" if r["behaviour_ok"] else "FAIL: " + "; ".join(r["behaviour_failures"])
        cells = [r["id"], r["difficulty"], r["agent"],
                 *(fmt(r["scores"].get(m)) for m in metric_names), behaviour]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    return lines


def summary_rows(records: list[dict], metric_names: list[str]) -> list[list[str]]:
    rows = []
    for level in (*LEVELS, "all"):
        subset = [r for r in records if level == "all" or r["difficulty"] == level]
        if not subset:
            continue
        passed = sum(r["behaviour_ok"] for r in subset)
        rows.append([level, str(len(subset)),
                     *(fmt(mean(r["scores"].get(m) for r in subset)) for m in metric_names),
                     f"{passed}/{len(subset)}"])
    return rows


def write_report(runs: dict[str, list[dict]], meta: dict, metric_names: list[str],
                 stem: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"
    json_path.write_text(json.dumps({"meta": meta, "runs": runs}, indent=2, default=str))

    out = [f"# RAGAS evaluation - {meta['date']}", "",
           f"- Deployment mode: `{meta['deployment_mode']}`",
           f"- Answering models: `{meta['llm_model']}` (router: `{meta['router_model']}`)",
           f"- Judge: `{meta['judge_model']}` (thinking: {meta['judge_thinking'] or 'default'}); embeddings: `{meta['embedding_model']}`",
           f"- Queries: {meta['queries']} from `eval/test_queries.yaml`",
           "- `n/a`: the metric is undefined for that row (no retrieved context, or an answer "
           "with no claims to check) and is left out of the means.", ""]
    if meta.get("self_judged"):
        out += ["> The judge is the same model that wrote the answers, so it may favour them. "
                "Compare configurations on the same judge; do not read the absolute numbers "
                "as independent.", ""]

    questions = next(iter(runs.values()))
    out += ["## Queries", "", "| ID | Level | Question |", "|---|---|---|"]
    out += [f"| {r['id']} | {r['difficulty']} | {r['question']} |" for r in questions]

    if BASELINE in runs and CANDIDATE in runs and metric_names:
        base = {r["id"]: r for r in runs[BASELINE]}
        cand = {r["id"]: r for r in runs[CANDIDATE]}
        ids = [r["id"] for r in runs[CANDIDATE] if r["id"] in base]
        out += ["", f"## {CANDIDATE} vs {BASELINE}", "",
                "Mean over the rows where both configurations have a score.", "",
                f"| Metric | {BASELINE} | {CANDIDATE} | delta | n |", "|---|---|---|---|---|"]
        for m in metric_names:
            pairs = [(base[i]["scores"].get(m), cand[i]["scores"].get(m)) for i in ids]
            pairs = [(b, c) for b, c in pairs if b is not None and c is not None]
            b_mean = mean(b for b, _ in pairs)
            c_mean = mean(c for _, c in pairs)
            delta = None if b_mean is None else c_mean - b_mean
            out.append(f"| {m} | {fmt(b_mean)} | {fmt(c_mean)} | {fmt_delta(delta)} | {len(pairs)} |")
        b_ok = sum(base[i]["behaviour_ok"] for i in ids)
        c_ok = sum(cand[i]["behaviour_ok"] for i in ids)
        out.append(f"| behaviour checks passed | {b_ok}/{len(ids)} | {c_ok}/{len(ids)} "
                   f"| {c_ok - b_ok:+d} | {len(ids)} |")

    for config, records in runs.items():
        out += ["", f"## {config}", "", *per_query_table(records, metric_names), "",
                f"### {config} - by difficulty", ""]
        head = ["Level", "n", *(SHORT[m] for m in metric_names), "Behaviour"]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        out += ["| " + " | ".join(row) + " |" for row in summary_rows(records, metric_names)]

    notes = [(config, r["id"], m, note) for config, records in runs.items()
             for r in records for m, note in r["score_notes"].items()]
    errors = [(config, r["id"], r["error"]) for config, records in runs.items()
              for r in records if r["error"]]
    if notes or errors:
        out += ["", "## Notes", ""]
        out += [f"- `{c}` {i}: agent failed - {e}" for c, i, e in errors]
        out += [f"- `{c}` {i} {SHORT[m]}: {n}" for c, i, m, n in notes]

    out += ["", "Answers, full contexts and routing decisions for every row are in "
            f"[`{json_path.name}`]({json_path.name}).", ""]
    md_path.write_text("\n".join(out))
    return json_path, md_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def preflight() -> None:
    from app.config import get_settings
    from app.retrieval import vectorstore

    settings = get_settings()
    try:
        chunks = vectorstore.count(settings)
    except Exception as exc:
        raise SystemExit(f"Vector store unreachable ({type(exc).__name__}: {exc}). "
                         "Start it with `make up` and index with `make ingest`.") from exc
    if not chunks:
        raise SystemExit("The vector store is empty - run `make ingest` first.")
    if not Path(settings.tabular_db_path).exists():
        raise SystemExit(f"No tabular store at {settings.tabular_db_path} - run `make ingest`.")


def answering_models(settings) -> set[str]:
    """Every model any agent (or the router) can answer with, fallbacks included.

    Read per agent rather than from LLM_MODEL: AGENT_MODELS__<NAME> can put a
    different model behind each agent, so the judge may coincide with one of
    them even when it differs from the default.
    """
    from app.agents.registry import list_agents

    return {model for agent in list_agents(include_router=True)
            for model in settings.models_for(agent.name)}


async def run(args) -> None:
    from app.config import get_settings
    from app.observability.logging import configure_logging

    configure_logging(args.log_level)
    only = set(args.only.split(",")) if args.only else None
    queries = load_queries(args.queries, only)
    if not queries:
        raise SystemExit("No queries selected.")
    metric_names = [] if args.no_judge else args.metrics

    settings = get_settings()
    judge_model = args.judge_model or settings.llm_models[0]
    preflight()

    runs: dict[str, list[dict]] = {}
    for config in args.configs:
        apply_config(config)
        metrics = build_metrics(metric_names, judge_model, args.timeout,
                                args.judge_thinking) if metric_names else {}
        records = []
        for i, query in enumerate(queries, 1):
            record = await answer_one(query, config)
            if metrics:
                await score_one(record, metrics, args.timeout, args.concurrency)
            records.append(record)
            scores = " ".join(f"{SHORT[m]}={fmt(record['scores'].get(m))}" for m in metric_names)
            status = "ok" if record["behaviour_ok"] else "FAIL"
            print(f"[{config}] {i}/{len(queries)} {query['id']} -> {record['agent']} "
                  f"{status} {scores} ({record['latency_s']}s)", flush=True)
            if args.pause:
                await asyncio.sleep(args.pause)
        runs[config] = records

    settings = get_settings()
    meta = {
        "date": date.today().isoformat(),
        "deployment_mode": settings.deployment_mode,
        "llm_model": ",".join(settings.llm_models),
        "router_model": ",".join(settings.router_models),
        "judge_model": judge_model,
        "judge_thinking": args.judge_thinking,
        "self_judged": judge_model in answering_models(settings),
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "rerank_top_n": settings.rerank_top_n,
        "queries": len(queries),
        "configs": {name: CONFIGS[name] for name in args.configs},
        "metrics": metric_names,
    }
    stem = args.name or f"ragas-{meta['date']}"
    json_path, md_path = write_report(runs, meta, metric_names, stem)
    print(f"\nwrote {md_path.relative_to(_REPO_ROOT)} and {json_path.relative_to(_REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--only", help="comma-separated query IDs, e.g. E1,A1")
    parser.add_argument("--queries", type=Path, default=QUERY_FILE)
    parser.add_argument("--judge-model", help="defaults to the first LLM_MODEL")
    parser.add_argument("--judge-thinking", choices=["minimal", "low", "medium", "high"],
                        help="Google models only: cap the judge's reasoning to speed it up")
    parser.add_argument("--no-judge", action="store_true",
                        help="run the queries and behaviour checks without scoring")
    parser.add_argument("--timeout", type=int, default=240, help="seconds per metric judgement")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="judge calls in flight per answer; lower it on a tight quota")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds between queries, to stay under a per-minute quota")
    parser.add_argument("--name", help="results file stem (default ragas-<date>)")
    parser.add_argument("--log-level", default="WARNING")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
