# Decision Log

Each entry records what was decided, what else was considered, why, what trade-offs were accepted, and when the decision should be revisited.

---

## D-001: LLM provider selected by deployment mode

- **Date:** 2026-09-22
- **Decision:** Generation runs against Google Gemini (free tier) in the default cloud mode and
  against Ollama (`qwen2.5:7b`) in on-prem mode. Routing and follow-up rewriting use a cheaper model
  in each mode, since classifying a question into one of a handful of agents does not need the
  answering model. Both integrations ship in every image; `DEPLOYMENT_MODE` chooses between them at
  startup.
- **Amended 2026-09-24.** The cloud defaults are now *chains* rather than single models —
  `gemini-3.6-flash,gemini-3.5-flash,gemini-flash-lite-latest` for answering and
  `gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest` for routing. Two things
  forced this. `gemini-2.5-flash` and `gemini-2.5-flash-lite`, the original choices, now answer
  generateContent with **404 NOT_FOUND** while still appearing in the provider's ListModels
  response — so a single-model default left the deployment with nothing to fall through to, and the
  readiness probe reported it healthy (see the amendment note in D-009). And on the free tier the
  full `flash` models are the ones that exhaust: measured on 2026-09-24, `gemini-3.8-flash`,
  `gemini-3.5-flash` and the `gemini-flash-latest` alias all returned 429 within a few minutes of
  light use, while every `lite` model answered in about a second. Each chain therefore ends in a
  lite model, because the last entry is the one that has to work.
- **Amended 2026-09-24, quota handling.** Two changes came out of running the chain against a real
  free-tier key on the deployed service, both measured rather than reasoned:
  - `LLM_MAX_RETRIES` dropped from 6 to 2. Retries and the chain solve different problems and at 6
    they fought: the provider backs off exponentially, so an exhausted model cost ~33s of 429s
    before the chain could move on, and one question ran past 180s.
  - `QuotaAwareFallback` (`app/llm/quota.py`) remembers, for one request, which models answered 429.
    Without it `ModelFallbackMiddleware` walks from the top of the chain on every model call, and a
    skill loop makes several: one question produced **30** 429s. With it, 6-9, and the same question
    went from 143s to 11-23s. The memo is scoped to the request rather than the process because
    quota windows reopen, and a process-wide one would keep using the weakest model long after the
    strongest recovered.
  - *Trade-off accepted:* neither change creates quota. Once every model in the chain is exhausted
    the request fails, quickly and with the provider's own error. On a free tier that is a question
    of how many questions have been asked that day, not of configuration -- so a demo session and an
    evaluation run compete for the same daily allowance.
- **Alternatives considered:** Groq free tier (fastest hosted inference and a viable cloud swap, but
  serves no embedding models); OpenRouter free models (widest selection behind one key, rejected for
  unpredictable availability — a demo failure would look like a defect in this system); vLLM or TGI
  for the on-prem half (better throughput under concurrency, rejected as heavier operations for a
  single-box deployment); a single provider for both modes (impossible — the constraint is that
  on-prem makes no external calls).
- **Rationale:** Gemini's free tier covers the cloud case without a billing relationship, and has
  reliable structured-output support, which the router depends on. Ollama is the lowest-friction way
  to satisfy the on-prem constraint and is reachable at any URL, so the LLM can run on a more
  powerful machine on the internal network rather than the application host.
- **Trade-offs accepted:** Free-tier request limits throttle RAGAS runs in cloud mode. On-prem
  latency is hardware-dependent and far worse than estimated. Measured on a loaded 12-core CPU-only
  laptop (15GB RAM, ~4GB free) against 2.1k tokens of context: `llama3.2:3b` answered correctly in
  **154s**, and `qwen2.5:7b` **exceeded a 300s timeout entirely** because the 4.8GB model did not fit
  in free memory and the host swapped. The on-prem default therefore assumes a host with roughly 8GB
  of free RAM, and constrained hosts should run the smaller model for both roles. Timeouts are
  mode-dependent as a result (60s cloud, 600s on-prem) — a safety net rather than a latency target,
  since a timeout firing just before the answer arrives is worse than one that waits.
- **Verified:** `llama3.2:3b` holds the `RouteDecision` schema reliably through
  `with_structured_output`, returning a correctly typed object on all four probe questions including
  an adversarial one, at roughly 6s per routing call. The concern that a 3B model would be too weak
  for structured routing did not materialise, so the cheaper router model stands.
- **Revisit when:** Free-tier rate limits block evaluation; a GPU host becomes available for the
  on-prem deployment; or concurrent usage makes per-request Ollama inference the bottleneck, at
  which point vLLM becomes the better on-prem server.

---

## D-002: Embeddings run locally via fastembed, in both modes

- **Date:** 2026-09-22
- **Decision:** `BAAI/bge-small-en-v1.5` (384-dim) executed in-process through `fastembed`, which
  runs the model on ONNX Runtime. Identical in cloud and on-prem mode. Wrapped in a small
  `Embeddings` adapter in `app/llm/providers.py`.
- **Alternatives considered:** A hosted embedding API such as Gemini's (smallest image and no model
  download, **rejected** — see rationale); Ollama `nomic-embed-text` (equally on-premises and good
  quality, but 768-dim and requiring a running Ollama for work that fits comfortably in-process, so
  it would make ingestion depend on the LLM host); `sentence-transformers` (widest model choice, but
  pulls PyTorch and roughly 2GB into the image).
- **Rationale:** This is the decision that makes D-010 possible, and it was chosen for that reason
  rather than on embedding quality. If embeddings were provider-switched alongside the LLM, moving
  to on-prem would change the vector dimension and invalidate the entire index — two changes plus a
  data migration, not one configuration change. Keeping embeddings in-process and identical in both
  modes means the switch requires no re-index and an index built in cloud mode can be copied to an
  on-prem deployment unchanged. It also removes embedding rate limits from ingestion entirely and
  makes the RAGAS baseline deterministic across runs.
- **Trade-offs accepted:** Around 150MB of ONNX model weights are downloaded on first run and held
  resident. bge-small is a small model: retrieval quality is lower than a larger embedder would give.
  Changing `EMBEDDING_MODEL` remains the one change that is *not* configuration-only, which is why
  the model name is stamped into the collection name and `make reindex` exists.
- **Revisit when:** Retrieval quality measured by RAGAS context recall plateaus below target, or a
  multilingual corpus arrives — `bge-m3` would be the upgrade, at a re-index.

---

## D-003: Qdrant as the vector store, run as a container

- **Date:** 2026-09-22, amended 2026-09-23 (embedded mode -> a Qdrant service)
- **Decision:** Qdrant via `qdrant-client`. The Compose stack runs it as its own service
  (`qdrant/qdrant`, pinned to the client's minor version, storage in a named volume) and the backend
  reaches it over `QDRANT_URL`. Blanking `QDRANT_URL` runs the same client embedded from
  `QDRANT_PATH`, which stays supported as the no-container fallback.
- **Alternatives considered:** ChromaDB (the most conventional choice and simplest integration, but
  a weaker path to a hosted deployment); FAISS (fastest to stand up, but no real metadata filtering
  and no server story); Pinecone free tier (rejected — a hosted store is an external call, which
  on-prem mode forbids). For the topology: staying embedded (one container fewer, but keeps the
  lock); a sidecar embedded index per process (diverging copies of the corpus).
- **Rationale:** Self-hostable stores were the only candidates that can satisfy on-prem mode at all,
  and a Qdrant container is still entirely inside the deployment boundary — it calls nothing out, so
  the single-switch guarantee in D-010 is untouched. Running it as a service removes the embedded
  directory lock, which was the first scalability bottleneck in ARCHITECTURE.md §7.2: ingestion and
  the API can now hold the index at once, `make ingest` no longer requires stopping the stack, and
  the backend is no longer pinned to one replica. It cost no application code — the switch is the
  variable the embedded decision already provided for.
- **Trade-offs accepted:** A third container in the default deployment, so a single-container host
  has to fall back to embedded mode (D-009). The index now lives in a Docker volume rather than
  under `data/`, so it is backed up by a Qdrant snapshot or a volume copy rather than a file copy,
  and `make clean` discards it. Host-side tooling depends on the container being up, since `.env`
  points `QDRANT_URL` at `localhost:6333`. The embedded fallback keeps the directory lock, which is
  why the readiness probe checks path writability there and opens a client only against a server.
- **Revisit when:** One Qdrant node stops being enough. The same client addresses a cluster, so that
  is a deployment change rather than a change of store.

---

## D-004: Hybrid retrieval as BM25 + dense fused with Reciprocal Rank Fusion

- **Date:** 2026-09-22
- **Decision:** Keyword retrieval with `rank_bm25` and dense retrieval from Qdrant, combined with
  Reciprocal Rank Fusion implemented in `app/retrieval/hybrid.py`. `RETRIEVAL_MODE=naive` bypasses
  fusion and returns vector-only results.
- **Alternatives considered:** A vector store's built-in hybrid search using sparse vectors (less
  code, but the fusion becomes a black box); dense-only retrieval (simpler, but loses exact matches
  on the identifiers, codes and figures that project documents are full of).
- **Rationale:** Project documents mix prose with identifiers such as risk IDs and line-item names,
  where lexical matching outperforms embeddings; narrative status questions are the reverse. RRF
  needs no score normalisation between the two systems and no tuning weight. Implementing it in-repo
  keeps it independent of the store and makes the naive-vs-hybrid comparison that the evaluation
  reports a switch in one place.
- **Trade-offs accepted:** More code than delegating to a store feature, and an in-memory BM25 index
  that is rebuilt on ingest and must be kept consistent with the vector store.
- **Revisit when:** The BM25 index no longer fits comfortably in memory, at which point Qdrant's
  native sparse vectors move it into the store.

---

## D-005: Re-ranking with FlashRank

- **Date:** 2026-09-22
- **Decision:** A local ONNX cross-encoder (`ms-marco-MiniLM-L-12-v2`) via FlashRank, re-ranking the
  fused candidate set down to `RERANK_TOP_N` before the LLM sees it. Controlled by `RERANK_ENABLED`.
- **Alternatives considered:** Cohere or Jina hosted rerankers (better quality, **rejected** — each
  adds an external call that on-prem mode would have to disable, which would turn the single-switch
  guarantee into three changes; Cohere's free tier is also a trial key that can expire mid-project);
  LLM-as-reranker (no extra dependency, rejected — an additional generation per query, which is
  punitive at on-prem latencies); no re-ranking (rejected — recall-oriented hybrid retrieval returns
  candidates whose order needs correcting before a small context window).
- **Rationale:** Keeping the reranker local preserves the property that the LLM is the only
  component that ever leaves the process, which is what D-010 rests on. The model is a few MB and
  ONNX-based, so it costs no meaningful image size and adds no PyTorch dependency.
- **Trade-offs accepted:** A small cross-encoder is weaker than a hosted reranker, and re-ranking
  adds CPU latency to every query on the same host that may already be running inference.
- **Revisit when:** Context precision remains low with re-ranking enabled, indicating the reranker
  rather than retrieval is the limit.

---

## D-006: Orchestration with LangChain primitives and the in-repo registry

> **Amended by D-015 (2026-09-23).** The revisit condition below -- "an agent needs a tool
> loop" -- was met. Orchestration is now a LangGraph state graph and agents have skills.
> The registry survives unchanged and is still the extension point.

- **Date:** 2026-09-22
- **Decision:** Use LangChain for model interfaces, prompt composition and structured output, and
  keep routing in the existing `app/agents/registry.py` plus `RouterAgent`. No LangGraph.
- **Alternatives considered:** LangGraph (state machines, checkpointing, cyclic agent graphs);
  CrewAI or AutoGen (multi-agent frameworks); no framework at all, calling provider SDKs directly.
- **Rationale:** The orchestration this system needs is one classification step followed by one
  agent call. LangGraph's value is in cyclic and stateful flows that do not exist here, and it would
  add a second mental model on top of the registry that already makes agents pluggable. Calling SDKs
  directly would mean writing the provider abstraction that makes D-001 a config switch.
- **Trade-offs accepted:** If agents later need tool-calling loops, this hand-rolled orchestration
  will need replacing rather than extending.
- **Revisit when:** An agent needs a tool loop, conversation state needs checkpointing, or several
  agents must run in parallel and have their results merged.

---

## D-007: Tabular questions answered through read-only DuckDB

> **Amended by D-014 (2026-09-23).** The claim below that "the connection could not write
> even if that validation were wrong" was verified false: read-only protects the database,
> not the filesystem. The defence is now three layers.

- **Date:** 2026-09-22
- **Decision:** Cleaned CSV/Excel tables are loaded into DuckDB. The Data Analysis agent generates
  SQL, which is validated as a single read-only `SELECT` before execution against a connection
  opened read-only.
- **Alternatives considered:** LangChain's pandas DataFrame agent (**rejected** — it executes
  model-generated Python via `exec()`, which is arbitrary code execution driven by untrusted
  document content and user input); precomputed aggregations only (safe but cannot answer unanticipated
  questions); pandas `query()` (narrow expression support, awkward for joins and grouping).
- **Rationale:** SQL against a read-only connection is a far smaller attack surface than Python
  execution, and the defence is layered: statement validation, then a connection that cannot write
  regardless. DuckDB reads CSV and Excel directly, handles the messy synthetic data well, and needs
  no server. Generated SQL is also inspectable, which makes it good citation material.
- **Trade-offs accepted:** Text-to-SQL fails on genuinely ambiguous questions in ways a code agent
  might muddle through; a small on-prem model is weaker at SQL than a hosted one. Validation must be
  conservative, which will reject some legitimate queries.
- **Revisit when:** Questions arrive that legitimately need multi-step computation rather than a
  single query.

---

## D-008: Observability as structured JSON logs with trace IDs

- **Date:** 2026-09-22
- **Decision:** Keep the existing `RequestContextMiddleware` and `JsonFormatter`: a trace ID per
  request in a contextvar, stamped onto every log record, with structured fields emitted via
  `extra={"fields": {...}}`. No external tracing service.
- **Alternatives considered:** LangSmith free tier (**rejected** — an external call, which on-prem
  mode would have to disable, breaking the single-switch guarantee, and it would send document
  content off-box); Arize Phoenix (self-hosted and genuinely on-prem-compatible, deferred rather
  than rejected — it costs a service and OpenTelemetry instrumentation, and effort is better spent
  on retrieval and agents first).
- **Rationale:** Structured logs correlated by trace ID answer the operational questions — which
  agent ran, what was retrieved, how long each stage took — and behave identically in both modes,
  with no data leaving the deployment.
- **Trade-offs accepted:** No span waterfall or token accounting UI; debugging a slow query means
  reading logs rather than clicking a trace.
- **Revisit when:** Per-span retrieval debugging becomes routine, at which point Phoenix is the
  drop-in that preserves the on-prem property.

---

## D-009: Deployment host — Cloud Run for the backend, Vercel for the frontend

- **Date:** 2026-09-24 (supersedes the deferral recorded 2026-09-22)
- **Decision:** The backend runs on **Google Cloud Run** (`us-central1`, 1 GiB / 1 vCPU,
  scale-to-zero), the frontend is a static Vite build on **Vercel**, and the two stateful
  dependencies stay external: **Qdrant Cloud** (free tier) for the vector index and **Redis Cloud**
  (free tier) for the graph checkpoints. `make deploy-backend` builds and ships the image;
  `render.yaml` keeps a fallback deployment described but not primary.
- **Alternatives considered:**
  - *Hugging Face Spaces* — **rejected, no longer free for this app.** The Hub docs now read
    "Gradio and Docker Spaces run on compute and require a paid plan to create"; only Static Spaces
    remain free. The 2026-09-22 note calling it "free, no card, generous RAM" is out of date.
  - *Railway* — **rejected.** No free tier since July 2023: a one-time $5 credit expiring in 30 days,
    then a $1/month plan capped at 0.5GB, below the measured footprint.
  - *Vercel for the backend* — **rejected.** Serverless Python caps far below a 1.4GB image and there
    is no long-running process. Kept for the frontend, which is what it suits.
  - *Render free tier* — **kept as a documented fallback** (`render.yaml`), not chosen. It is the only
    option needing no billing account, but 512MB against a 348MB warm footprint is thin, and 0.1 vCPU
    is the sharper limit: every request runs an ONNX embedding *and* a cross-encoder re-rank.
  - *A single container with embedded Qdrant and the in-process saver* — **rejected for the primary
    deployment**, though the image supports it and the readiness probe confirms it works. It would
    give up the topology D-003 and D-017 chose, in a deployment meant to demonstrate the design.
- **Rationale:** 1 GiB leaves roughly 3x headroom over the measured footprint, which is what the
  earlier deferral was waiting for. Scale-to-zero means an idle demo consumes no quota, and the
  always-free allowance — 180,000 GiB-seconds a month, which at 1 GiB is about 50 instance-hours —
  is far more than a demo uses. Keeping Qdrant and Redis as managed services means the deployed
  system is the one the architecture describes rather than a reduced variant of it.
- **Measured:** backend image 1.4GB (1.17GB before the ONNX weights were baked in), frontend 63.2MB.
  Peak resident memory with both ONNX models warm and a real embed and re-rank: **348MB** on the
  host, **340MB** inside the container with networking disabled. The container idles at ~59MB before
  the models load, so the first query is what sets the requirement.
- **Trade-offs accepted:**
  - **A billing account is required.** Cloud Run has required one since 3 February 2026. The card is
    not charged inside the always-free quota, but it must be on file — which is why `render.yaml`
    exists.
  - **Cold starts.** `--min-instances 0` is what keeps the demo free; `--min-instances 1` would burn
    720 instance-hours a month against a 50-hour allowance. `--cpu-boost` shortens the start, and
    baking the ONNX weights into the image removes a ~150MB download from the first request.
  - **Uploads do not survive a scale-to-zero.** `POST /v1/upload` writes to the container filesystem,
    which Cloud Run discards with the instance. The curated corpus is baked into the image and the
    vector index is in Qdrant Cloud, so the demo always has its documents; an uploaded file is
    indexed into Qdrant and stays searchable, but the file itself does not persist.
  - **The free Qdrant cluster suspends after a week idle** and is deleted after four. A demo left
    untouched for a month needs the cluster recreated and `make ingest` re-run.
  - **The region is fixed** to `us-central1`, `us-east1` or `us-west1`; the free tier covers no
    others. `make deploy-backend` refuses anything else rather than deploying a billable service.
- **Consequences for the image:** the build context moved to the repository root so `./data` could be
  baked in, the ONNX weights are downloaded at build time into `EMBEDDING_CACHE_DIR` /
  `RERANK_CACHE_DIR`, the data paths are set as absolute `ENV` defaults (a relative path anchors to
  `/` in the image, not `/app`), and `CMD` honours `$PORT`.
- **Amended 2026-09-24, readiness.** `/v1/health/dependencies` computed `status` from the LLM
  check's `reachable` flag alone, which only means the provider answered a ListModels call. A chain
  of models that no longer exist reported `ok`. It now also requires at least one model per chain to
  be present (`any_model_present`). The bound worth knowing: presence means "listed", not "will
  generate" — the retired 2.5 models are still listed — so only a real generation catches a
  listed-but-retired model. Verified instead by running the cloud path end to end before deploying.
- **Revisit when:** the demo needs to stay warm, uploads need to persist, or a second replica is
  wanted — at which point the session store's process-locality (ARCHITECTURE.md §7.2) binds before
  the host does.

---

## D-010: Cloud and on-premises parity behind a single configuration switch

- **Date:** 2026-09-22
- **Decision:** `DEPLOYMENT_MODE=cloud|onprem` is the only variable that needs to change to move
  between a hosted LLM and a fully on-premises deployment. It resolves the LLM provider, both model
  names and the request timeout; every other knob keeps its value. Resolution fills only blank
  settings, so an explicit override always wins, and the resolved configuration is logged at startup
  and served from `/v1/health/dependencies`.
- **Alternatives considered:** Separate configuration files or Compose overlays per environment
  (duplicates every unrelated setting and lets the two drift); documenting a migration procedure
  instead of implementing a switch (what the requirement explicitly asks not to do); separate
  container images per mode (makes the mode a build-time property, so it cannot be changed by an
  operator).
- **Rationale:** The requirement is that on-prem adaptation be one configuration change, so the
  design question is what must be true for that to be honest. The answer drove D-002, D-005 and
  D-008: every component except generation has to run inside the deployment, because any hosted
  embedder, reranker or tracer would be a second thing to switch off. What is guaranteed is therefore narrow
  and checkable — one variable, no re-index, an identical retrieval path, and an index that is valid
  in either mode. `tests/test_config_modes.py` asserts exactly that, including a test that the set
  of settings differing between modes is precisely the five LLM-related ones, so a future knob that
  quietly becomes mode-dependent fails the build.
- **Trade-offs accepted:** Both provider integrations ship in every image, so a cloud deployment
  carries the Ollama client it never uses (small — they are API clients, not weights). Parity is
  structural, not behavioural: answer latency and RAGAS judge quality differ substantially between
  modes, and the tests cannot assert otherwise. Committing to in-process embedding and re-ranking
  also means forgoing better hosted models for those stages.
- **Revisit when:** A component genuinely cannot run inside the deployment and must be hosted, at which point
  the single-switch guarantee needs restating as something weaker and the tests updated to match.

---

## D-011: Structure-aware chunking rather than a fixed character window

- **Date:** 2026-09-23
- **Decision:** Chunk on the structure of each source: PDF narrative splits at the reports' numbered
  section headings, PDF tables are extracted structurally and become one chunk each (stitched back
  together when a page break splits them), and CSV/Excel tables become row groups bounded by both a
  row count and a character budget. Each report also gets a synthetic header chunk carrying its
  period, status and identifiers. Chunk IDs are `uuid5` over (source, section, page/row range,
  ordinal), making re-ingestion idempotent.
- **Alternatives considered:** Recursive character splitting (the conventional default, and by far
  the least code); page-based chunks (trivial, and citations are exactly page numbers); one chunk
  per table row for tables.
- **Rationale:** Both alternatives fail on specifics of this corpus rather than in principle.
  `pdfplumber.extract_text()` flattens a milestone row into `Usage metering pipeline Hannah
  03/20/2026 In-progress Lands with a 36h lag; batching` / `Okonkwo work to follow.`, splitting the
  owner across lines and interleaving it with the comment; character splitting inherits that and
  additionally cuts tables mid-row. Page chunking splits the Key Risks table across the page 1/page 2
  boundary, orphaning R-002 and R-009 under a bare header. The reports' numbered headings make the
  structural approach cheap to implement and give citations (`§5 Key risks (pp. 1-2)`) that a
  reviewer can verify by hand.
- **Trade-offs accepted:** Considerably more code than a text splitter, and it leans on a document
  convention — numbered headings — that a differently formatted report would not follow, falling
  back to one narrative block per document. Table bounding-box exclusion depends on pdfplumber
  finding the table at all; a borderless table would be read as narrative.
- **Revisit when:** A source arrives whose headings are not numbered, or whose tables are not ruled,
  at which point heading detection needs to fall back to font-size or layout heuristics.

---

## D-012: Ingestion is idempotent and replaces per source

- **Date:** 2026-09-23
- **Decision:** Re-ingesting a file deletes every chunk carrying that `source` before upserting the
  new ones, and structured rows are upserted on a natural key (`cost_code`, `risk_id`) rather than
  appended. The ingest run returns a report of files, chunks, rows per table and the flaw classes
  handled.
- **Alternatives considered:** Append-only ingestion with a run ID (keeps history, but retrieval
  then has to filter to the current run); dropping and rebuilding the whole index on every change
  (simple, but re-embeds the entire corpus for one changed file, and is minutes of CPU on-prem).
- **Rationale:** Deterministic IDs alone overwrite chunks that still exist but leave orphans when a
  file shrinks — a deleted section's chunks linger and keep being retrieved. Per-source replacement
  handles that while keeping the work proportional to what changed. The natural key matters because
  the financial workbook and the CSV export describe the same eight cost lines: without it the
  second file ingested would either duplicate them, making `sum(budget)` 12,630,000, or silently
  discard the first file's contribution.
- **Trade-offs accepted:** No history of what a document said previously. The natural keys are a
  small piece of corpus knowledge held in `tabular_store.py`, which a genuinely new table shape
  would need extending.
- **Revisit when:** Document versions need to be retained and compared over time.

---

## D-013: Conversation memory is in-process and bounded, and follow-ups are rewritten

- **Date:** 2026-09-23
- **Decision:** Sessions live in a process-local, lock-guarded store bounded on two axes: the last
  `MAX_TURNS` turns of a conversation, and the `MAX_SESSIONS` most recently used sessions. A
  follow-up is rewritten into a standalone question by the cheap router model before routing, so
  agents never receive conversation state. The first turn of a session makes no rewrite call.
- **Alternatives considered:** Concatenating history into every agent prompt (no extra call, but
  every downstream prompt grows with the conversation); LangChain's memory classes (legacy in
  1.x, and they would re-introduce the abstraction the registry already provides); Redis or a
  database (survives restarts and scales across replicas, but a second service for a single-box
  deployment, and on-prem would have to host it); no memory at all (the brief asks for follow-ups).
- **Rationale:** ARCHITECTURE.md section 7.1 measured that prompt processing, not generation,
  dominates on-prem latency -- 154s over 2,145 input tokens against 74 output. Rewriting keeps every
  downstream prompt the size of one question plus its retrieved context, whatever the conversation
  length, where concatenation would grow it without bound. It also keeps agents stateless, which is
  what lets `AgentInput -> AgentResult` stay a contract the router, API and logs treat uniformly.
  Structured output is used for the rewrite because a small model asked to rephrase readily answers
  "Sure! Here is the standalone question: ..." instead.
- **Trade-offs accepted:** History is lost on restart and is not shared across replicas, so the
  deployment is effectively single-instance for conversational continuity. A follow-up costs one
  extra model call. A bad rewrite silently changes the question -- mitigated by falling back to the
  question as asked whenever the rewrite errors, comes back empty, contains no words, or runs far
  longer than the question it replaces, and by logging whether a rewrite happened.
- **Revisit when:** A second backend replica is needed, or transcripts must survive a restart. The
  store then becomes Redis behind the same interface, not a different design -- and D-017 has since
  put that Redis in the stack for graph state, so the remaining work is the store, not the service.

---

## D-014: Model-generated SQL is contained in three layers, not two

- **Date:** 2026-09-23
- **Decision:** Amends D-007. `read_only_connection()` also passes
  `enable_external_access=False` and `lock_configuration=True`, and `validate_select()` additionally
  rejects table functions that read outside the database.
- **Alternatives considered:** Relying on `read_only=True` alone, as D-007 originally stated
  (**rejected** -- verified insufficient, see below); blocking the functions by name only (a
  denylist is only as good as its list); running DuckDB as a separate least-privilege process
  (stronger isolation, but a second service and a large change for a single-box deployment).
- **Rationale:** D-007 claimed the read-only connection meant "the connection could not write even
  if that validation were wrong". That is false as written, and was verified so against duckdb
  1.5.5: on a `read_only=True` connection, `SELECT * FROM read_csv_auto('/etc/passwd')` returns the
  file, `COPY (SELECT 1) TO '/tmp/x.csv'` writes one, and `INSTALL httpfs` succeeds. Read-only
  protects the *database*, not the *filesystem*, and DuckDB types such a statement as a `SELECT`, so
  the first layer passes it too. Since the SQL is written by a model reading untrusted document
  content, that is a reachable path and not a theoretical one. `enable_external_access=False` blocks
  all of it at the engine; `lock_configuration=True` stops generated SQL re-enabling it.
- **Trade-offs accepted:** The setting must be applied inside `read_only_connection()` and shared by
  every reader, because DuckDB caches one instance per database path per process and a second
  connection with a different configuration raises. That couples `list_tables()` to the same
  configuration, which is the reason both live in one module. The agent also loses any legitimate
  use of external files -- there is none today, and re-enabling it would mean re-opening this path.
- **Revisit when:** The agent legitimately needs to read a file DuckDB does not already hold, at
  which point the answer is a separate least-privilege process rather than relaxing this flag.

---

## D-015: Orchestration moves to LangGraph, and agents have skills

- **Date:** 2026-09-23
- **Decision:** Amends D-006. A request is one LangGraph `StateGraph` --
  `rewrite -> route -> <agent> -> END` -- built from the agent registry in `app/graph/build.py`,
  and each agent is a `create_agent` tool loop over the skills it declares. Skills are LangChain
  tools in `app/skills/`: `search_documents` for narrative retrieval, `describe_tables` and
  `run_sql` for the structured store. `BaseAgent.run()` is now concrete, so a new agent is
  `name`, `description`, `skills` and a system prompt. A LangGraph checkpointer holds graph state
  per session -- `InMemorySaver` here, Redis since D-017 -- while follow-ups are still rewritten
  (D-013) so agent prompts do not grow with the conversation.
- **Alternatives considered:** Keeping the hand-rolled dispatch and giving agents tools inside it
  (rejected -- it is the tool loop that has to be written, and `create_agent` is that loop);
  CrewAI or AutoGen (rejected again, for the reasons in D-006); LangGraph for agent internals only,
  leaving `RouterAgent.run` to dispatch (rejected -- it keeps two orchestration models, which is
  what D-006 objected to in the first place); replacing session memory with the checkpointer
  (rejected -- see the trade-off on prompt length in D-013, which has not changed).
- **Rationale:** D-006 named the condition precisely: *"If agents later need tool-calling loops,
  this orchestration is replaced rather than extended."* Needing one is what changed. The fixed
  pipeline could not search twice, consult a second table, or recover from a rejected statement
  except through one hard-coded retry, and each of those is an ordinary question about this corpus.
  Building the graph from the registry keeps the property D-006 was protecting -- an agent is one
  module and nothing else changes -- while the failure rules of ARCHITECTURE.md section 5.4 move
  from `RouterAgent.run` into graph nodes, where they apply to every agent uniformly.
- **Trade-offs accepted:**
  - **A prompt-injection control is gone.** ARCHITECTURE.md section 8.1 previously rested partly on
    "the answering agents have no tools". They have tools now. What replaces it is that every skill
    is read-only and in-process -- retrieval, a schema listing, and a `SELECT` behind the three
    layers of D-007/D-014 -- so a successful injection still reaches nothing outside the corpus.
    The containment on generated SQL is now load-bearing rather than defence in depth, because the
    model can call `run_sql` repeatedly with text it read from an uploaded document.
  - **Cost and latency are multiplied by the loop.** Each tool call is another generation. On the
    CPU-only host measured in section 7.1 that is 154s per iteration, so the budget
    (`AGENT_MAX_TOOL_CALLS`, default 3) is a cost control, not a safety valve. It is enforced in the
    evidence collector rather than with `ToolCallLimitMiddleware`, which blocks the call without
    ending the graph and so spends the recursion limit instead of returning an answer.
  - **Retrieval is no longer deterministic per question**, since the model writes its own queries.
    RAGAS runs are correspondingly noisier, and the naive-vs-hybrid comparison now varies with what
    the model chose to search for as well as with the retrieval mode.
  - **The refusal moved.** An empty index used to be refused before any model call. The model is
    what searches now, so the refusal happens after it: `finalise()` discards an answer with no
    evidence behind it rather than preventing one.
- **Revisit when:** Agents need to run in parallel and have their results merged, a conversation
  needs to resume mid-graph after an interrupt, or the tool budget stops being the right shape --
  at which point a per-skill budget, rather than a per-run count, is the next step.

---

## D-016: Model names are fallback chains, tried strongest first

- **Date:** 2026-09-23
- **Decision:** `LLM_MODEL` and `ROUTER_MODEL` are comma-separated chains ordered strongest first,
  exposed as `Settings.llm_models` / `Settings.router_models`. When a model fails with a
  provider-side error the next one in the chain answers instead. `LLM_MAX_RETRIES` (default 6, the
  hosted provider's own) is spent on each model before the chain advances, so a per-minute limit is
  waited out and only a durable failure — a spent daily quota — costs quality. A name with no comma
  disables fallback and is the pre-chain behaviour exactly.
- **Alternatives considered:** A single model plus more retries (does nothing once a daily quota is
  spent); returning `RunnableWithFallbacks` from `get_llm()` (rejected — it is not a `BaseChatModel`,
  so it has neither `with_structured_output` nor `bind_tools`, and both the router and every agent
  tool loop need one of those); a `list[str]` settings field (rejected — pydantic-settings parses a
  complex-typed field from the environment as JSON, so `LLM_MODEL=a,b` would fail to load and only
  `LLM_MODEL=["a","b"]` would work, which is a worse thing to ask of a `.env`); degrading to on-prem
  Ollama on quota failure (a mode flip mid-request, and 154s per generation — D-010 keeps the modes
  as a deployment choice, not a runtime one).
- **Rationale:** Free-tier quota is the failure this project actually hits, and it is per-model: the
  answering model is exhausted long before the whole account is. Ordering the chain strongest first
  means the best model is always preferred and a weaker one is reached only when the alternative is
  no answer at all. Keeping the fallback inside `app/llm/providers.py` preserves the D-010 promise
  that provider choice is configuration — the chain is a `.env` edit, not a code change.
- **Trade-offs accepted:**
  - **Two mechanisms, because the call sites differ.** The router and the follow-up rewrite get
    `get_structured_llm()`, which applies `with_structured_output` per model and wraps the *results*
    in `with_fallbacks`. Agents get `ModelFallbackMiddleware`, because `create_agent` must bind tools
    to a real chat model. `get_llm()` keeps its "one `BaseChatModel`" contract and returns the
    primary. One concept, two wirings, which is a thing to know before editing either.
  - **The fallback trigger is coarse.** `with_fallbacks` takes exception *types*, not a predicate,
    so the line drawn is provider-side (Google's `APIError`, covering 4xx and 5xx, plus
    `TimeoutError`) versus local. A genuine 400 is therefore attempted once per model before the
    chain re-raises it. Accepted: a 400 fails fast and is not hidden. `ModelFallbackMiddleware`
    is coarser still — it catches every exception except `GraphBubbleUp` — so an agent-loop bug
    costs one attempt per model rather than surfacing on the first.
  - **Answer quality becomes non-deterministic across runs.** A RAGAS run that falls back partway
    is scoring two different models and does not say so. An eval that must be comparable should
    pin a single model.
  - **A missing fallback is invisible until it is needed**, which is why `check_llm_ready()` now
    reports presence per model across both chains rather than for the primary alone.
- **Revisit when:** The chain needs per-model settings rather than one shared `LLM_TIMEOUT_S` and
  `LLM_MAX_RETRIES`, or fallback needs to cross providers (a Groq or on-prem tail behind the hosted
  head), at which point the chain entry becomes a `provider:model` pair rather than a bare name.

---

## D-017: Graph state is checkpointed to Redis

- **Date:** 2026-09-23
- **Decision:** The LangGraph checkpointer is `AsyncRedisSaver`
  (`langgraph-checkpoint-redis`) pointed at a Redis container on the Compose network, selected by
  `REDIS_URL`. Blank keeps the `InMemorySaver` D-015 introduced, which is what the test suite and a
  single-process run use, so Redis is a deployment choice rather than a build one — the client ships
  in every image, as both LLM providers do (D-010). Threads are keyed by session ID, as before, and
  bounded twice: session eviction still drops a thread the moment its conversation is gone, and
  `CHECKPOINT_TTL_MINUTES` (default 120, refreshed on read) expires the threads a restart would
  otherwise orphan. A Redis that cannot be reached degrades to the in-process saver with a WARNING;
  it does not fail the request.
- **Alternatives considered:** Keeping `InMemorySaver` (rejected — it is the one part of the request
  path that cannot survive a restart *or* be shared, and D-013 already names a second replica as the
  thing this deployment will need next); `langgraph-checkpoint-postgres`, the implementation
  LangSmith itself runs (rejected — a relational service and a migration step for data whose whole
  lifetime is a TTL, where this project has no other Postgres to put it beside); `AsyncSqliteSaver`
  (rejected for the reason embedded Qdrant is only a fallback in D-003 — a file is one process's
  lock, so it fixes restart survival and nothing about replicas); the sync `RedisSaver` (rejected —
  the graph is driven with `ainvoke`, and its `delete_thread` and async methods raise on the
  sync class); failing the request when Redis is down (rejected — see the trade-off below).
- **Rationale:** This is the objection in D-013 being answered in the order the system can actually
  take it. That entry rejected Redis for session memory as "a second service for a single-box
  deployment", and for the transcript that is still true — it is small, bounded and cheap to lose.
  Graph state is the half where the argument reverses: it is written by the framework on every
  superstep whether or not anyone reads it, it is what an interrupt or a mid-graph resume would need
  (the case D-015 names as its own revisit trigger), and it is the only store in the request path
  with no persistence at all. Moving it first also puts the connection in place that `SessionStore`
  will reuse when the second replica arrives, which is why §7.2's third bottleneck is now half
  closed rather than restated.
  Redis over the alternatives because it is the one of them this deployment can run as a single
  container with no schema, no migration and a TTL as its eviction policy — and because on-prem
  changes nothing, which is the standing constraint: it sits on the Compose network beside Qdrant
  and calls nothing out (D-010).
- **Trade-offs accepted:**
  - **A fourth service to run.** The default stack is now backend, frontend, Qdrant and Redis, and
    `make dev-backend` wants `make redis` beside `make qdrant`. Blanking `REDIS_URL` is the way back
    to no container, and the suite runs that way by default.
  - **The image carries a Redis client either way**, for the same reason it carries both LLM SDKs:
    the switch has to be configuration. `redisvl` and its dependencies are the cost.
  - **A degraded checkpointer is silent in the answers.** An unreachable Redis falls back to the
    in-process saver rather than returning 503, because nothing resumes mid-graph today, so a
    checkpoint no user ever reads is not worth failing a request over. It is deliberately sticky —
    reattaching mid-process would split one conversation's threads across two stores — so recovery
    is a restart. It is reported at WARNING and by `/v1/health/dependencies`, and that reporting is
    the whole mitigation. **This inverts once anything resumes from a checkpoint**, at which point
    the honest behaviour is to fail loudly instead.
  - **A Redis that dies mid-request still fails that request.** The degradation is decided at setup;
    a write that fails after it surfaces as the generic 503 from `/v1/chat`. The Compose
    `depends_on` gates the backend on a healthy Redis, which makes the common case the setup case.
  - **Checkpoints are now data at rest.** Question text and tool results persist outside the process
    where conversation history never did, which is why §8.3 gained a paragraph and why the TTL is a
    retention period rather than only a memory bound.
  - **The Redis 8 requirement is not obvious from the URL.** The saver queries through RediSearch
    indices, so an older server without the query engine fails at `asetup()` — which, given the
    degradation above, looks like a WARNING and a silently in-process checkpointer rather than a
    startup error. The Compose image is pinned to Redis 8 for that reason.
- **Revisit when:** `SessionStore` moves behind the same Redis (the remaining half of §7.2's third
  bottleneck), an interrupt or human-in-the-loop step makes a checkpoint something a user reads —
  at which point the degradation above becomes the wrong default — or the deployment already runs a
  Postgres, which would make one service do both jobs.

---

## D-018: A small-talk agent, with no skills, and it is the routing fallback

- **Date:** 2026-09-24
- **Decision:** A third routable agent, `small_talk` (`app/agents/small_talk.py`), handles greetings,
  thanks, sign-offs and questions about the assistant itself. It declares `skills = []`, so
  `BaseAgent.run()` builds a `create_agent` loop with no tools and the loop ends on the model's
  first turn; its behaviour is entirely its system prompt, which forbids stating any fact about the
  project and declines a project question rather than attempting it. `finalise()` is overridden to
  return zero citations and skip marker attribution. It is also `RouterAgent.FALLBACK_AGENT`,
  replacing `document_qa`: an unknown agent name, a confidence below `MIN_CONFIDENCE`, or a routing
  call that fails outright all land here.
- **Alternatives considered:** Handling greetings in the router — a fourth branch beside its three
  fallback rules (rejected: the router classifies and answers nothing, and an answering router is
  the dispatch tangle D-006 and D-015 both removed); a keyword or regex intercept in `/v1/chat`
  before the graph (rejected: cheaper by one model call, but it is a second routing mechanism with
  its own vocabulary, and "thanks, that's all I needed" is not a regex — the router already does
  classification and does it with a model); giving the agent a `list_documents`-style skill so it
  could answer "what do you cover?" from the index (rejected for now: it turns a prompt-only agent
  into a retrieval one for a question whose answer is the corpus description, not its contents —
  revisit if users start asking which documents are loaded); keeping `document_qa` as the fallback
  (the previous arrangement — rejected, see the rationale); splitting the two fallbacks so that a
  low-confidence route declines via `small_talk` while a routing *exception* still attempts the
  question via `document_qa` (a better answer to the outage case and worth returning to, but it
  makes "the fallback" two constants and two behaviours to reason about, for a failure that the
  WARNING log already surfaces); letting `document_qa` keep answering greetings (rejected: it searches, retrieves nothing
  and returns `NO_CONTEXT_MESSAGE`, so "hi" reads as a broken system and costs a retrieval pass).
- **Rationale:** The registry made this the cheap change it should be — one module, one
  `description`, and the router, graph, API and frontend pick it up untouched (D-015). What it buys
  is that the assistant's first exchange behaves like an assistant. The narrow remit is the point:
  this is the only agent whose answer nothing grounds, so the prompt does the containment that
  retrieval does elsewhere, and the `description` names the negative case explicitly because the
  expensive misroute is a project question landing here, not a greeting landing on `document_qa`.

  Making it the fallback follows from what a fallback is *for*. `document_qa` held the role because
  it degrades to the general case — it can attempt any question against the corpus. But the route it
  inherits is precisely the one the model could not make confidently, and that is as likely to be a
  message this system has no business answering as a question it does; attempting it means
  retrieving whatever is nearest and presenting it as an answer. Declining and saying what the
  assistant covers is the more honest degradation, and it is the only degradation that is true
  whatever the question was. The rule in ARCHITECTURE.md §5.4 is that a request returns an answer or
  an honest refusal — this makes the routing fallback the refusal rather than a guess.
- **Trade-offs accepted:**
  - **It can be wrong in a way no other agent can.** Every other answer is refused when the evidence
    is missing; this one has no evidence by construction, so the only guard is the prompt and a weak
    on-prem model may still volunteer something. The mitigation is remit, not verification: it is
    asked for one or two sentences about the assistant, and `metadata["grounded"] = False` marks
    every answer it produces so logs and evaluation can separate them.
  - **A misroute costs a whole answer, and the fallback role makes that the common case.** A real
    question the router merely failed to place is declined rather than attempted, where
    `document_qa` would at least have searched. Accepted on the reasoning above, but it moves the
    burden onto the routing prompt: routing accuracy is now the difference between an answer and a
    deflection, which is why `eval/test_queries.yaml` carrying `expected_agent` per query stops
    being a nice-to-have.
  - **A router outage now silences the system quietly.** Every route failure returns 200 with a
    polite decline, so a spent quota or an unreachable endpoint looks like a working assistant that
    has become unhelpful rather than like an incident. The WARNING on every fallback is the whole
    mitigation, which means it has to be alerted on rather than merely logged. Splitting the
    exception path back to `document_qa` is the fix if this bites.
  - **The decline has to be written for a question, not a greeting.** The prompt's rule 2 is reached
    far more often than the greeting the agent is named for, so it asks for the document, period or
    figure the user meant rather than telling them to rephrase — the user has usually phrased it
    perfectly well, and the failure was the router's.
  - **The router prompt grew by two worked examples**, which hardcode an agent name in a file whose
    catalogue is otherwise built from the registry. Prompt length is the on-prem latency lever
    (§7.1), so this is two lines, not a section; the `description` is still doing the work.
  - **Another `AGENT_MAX_TOOL_CALLS`-free path through the graph.** Nothing here is bounded by the
    budget because nothing calls a tool, so the only cost control on this agent is the answer length
    its prompt asks for.
- **Revisit when:** Users ask which documents are loaded often enough to want a real answer, at
  which point the agent needs a read-only corpus-listing skill and stops being prompt-only; routing
  accuracy measured against `eval/test_queries.yaml` turns out to be the binding constraint on
  answer rate, at which point the split fallback above is the next move; or a
  second non-answering agent appears (a help or feedback agent), at which point the shared "answers
  nothing from the corpus" behaviour is worth a base class rather than a second prompt.
