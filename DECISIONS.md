# Decision Log

---

## D-001: LLM provider selected by deployment mode, with a model chain per call site
- **Decision:** Google's hosted models in cloud mode, Ollama in on-prem mode. Each place that calls
  a model (the router, the follow-up rewrite and each agent) has its own short list of models,
  tried in order: the first is the cheapest one that does that job well, and the rest are there in
  case it is unavailable or out of quota. In cloud mode every list starts with Gemma 4, which is not
  charged for, and ends with a Gemini model; On-prem uses one model for everything, because loading
  a second model costs memory on the Ollama host.
- **Alternatives considered:** Groq free tier (fast, but limited models to fall back between);
  OpenRouter free models (wide selection, but availability is unpredictable, and a demo failure
  would look like a bug in this system); vLLM (better throughput, but too heavier to run); one provider for both modes (not possible, because on-prem must make no external calls);
- **Rationale:** One Google API key offers everything from free open models to frontier ones, so
  a harder agent just needs a different model name, not a new provider. Structured output and tool
  calling both work reliably, which the router and agents need. Routing, quoting a passage and
  writing valid SQL are different jobs, and a stronger model is only worth paying for where a
  weaker one fails, so the model is chosen per agent. The failure we actually hit is a quota limit
  on one model rather than the whole account, so stepping to the next model in the list recovers
  faster than waiting and retrying; Ollama is the simplest way to run fully on-prem, and it can sit
  on a stronger machine on the internal network rather than on the app host.
- **Trade-offs accepted:** Free tiers have low per-minute limits, so a burst of questions reaches the
  billed fallback, or with no billing, no answer at all. When a request falls back, the answer can
  come from a different model than usual, so evaluation runs that must be comparable should use
  one model; Ollama's speed depends on the hardware it runs on.
- **Revisit when:** An agent's answers get worse on Gemma in evaluation; Concurrency makes
  Ollama the bottleneck, at which point vLLM is the better on-prem server.

---

## D-002: Embeddings run locally via fastembed, in both modes
- **Decision:** `BAAI/bge-small-en-v1.5`, run in-process through `fastembed` on ONNX Runtime, the
  same in cloud and on-prem mode.
- **Alternatives considered:** A hosted embedding API such as Gemini's (no model to ship, but an
  external call that on-prem mode would have to replace); Ollama `nomic-embed-text` (also local and
  good, but it would make ingestion depend on the LLM host for work that fits in-process);
  `sentence-transformers` (widest choice of models, but pulls in PyTorch and about 2GB of image);
  a larger local model such as `bge-large` or `mxbai-embed-large`.
- **Rationale:** On-prem is a requirement, so the embedding model has to run locally either way.
  Using the same model in cloud mode means one code path to test, and switching modes needs no
  re-index. bge-small is small, runs on CPU and needs no GPU. I didn't want to pick it on speed
  alone, so I measured it against the three larger models on a retrieval-only benchmark
  (`eval/embedding_bench.py`). None of them did measurably better, and they cost 3-9x the query
  latency and up to 2GB more memory. 
- **Trade-offs accepted:** The benchmark is small, so it shows no large difference rather than
  proving the models equal. A marginally better model could still exist. Changing the model means
  a re-index.
- **Revisit when:** The corpus grows by an order of magnitude, since the result depends on its
  size; answers get worse on harder questions; or documents in other languages need support.

---

## D-003: Qdrant as the vector store, run as a container
- **Decision:** Qdrant, run as its own service in the Compose stack and reached over `QDRANT_URL`.
  Leaving `QDRANT_URL` blank runs the same client embedded on local disk, as a no-container fallback.
  The cloud demo uses a free Qdrant Cloud cluster (D-010).
- **Alternatives considered:** ChromaDB (the most conventional choice, but a weaker path to a hosted
  deployment); Pinecone (a hosted store is an external call, which on-prem mode forbids); pgvector (a good fit where
  Postgres already exists, but this project has none, so it would add a database just for vectors);
- **Rationale:** The same client runs Qdrant two ways: as a local container, or as a
  managed cluster. That covers on-prem and cloud with one code path, since switching is only a
  change of URL. Running it as a service also lets ingestion and the API use the index at the same
  time, and lets the backend run more than one replica.
- **Trade-offs accepted:** One more container to run
- **Revisit when:** The index outgrows one node or the free cluster. Moving to a larger cluster is a
  change of URL, not of store.

---

## D-004: Hybrid retrieval with local re-ranking
- **Decision:** Keyword search (BM25) and vector search from Qdrant, merged with Reciprocal Rank
  Fusion that is implemented in the repo, then re-ranked by a small local cross-encoder (FlashRank)
  before the model sees the passages. `RETRIEVAL_MODE=naive` switches to vector-only search as the
  evaluation baseline.
- **Alternatives considered:** Qdrant's built-in hybrid search with sparse vectors (less code, but
  the merging becomes a black box); vector-only search (simpler, but misses exact matches on risk
  IDs, cost codes and figures); hosted rerankers such as Cohere or Jina (better quality, but an
  external call that on-prem would have to disable); using the LLM as the reranker (an extra
  generation per question); 
- **Rationale:** Project documents mix prose with identifiers. Keyword matching wins on the
  identifiers and embeddings win on narrative status questions, so using both covers each one's weak
  spot. Rank fusion needs no score normalisation or tuned weight, and keeping it in the repo makes
  the naive-versus-hybrid comparison a single switch. Hybrid search favours recall, so the order of
  the candidates needs fixing before a small context window, and a local reranker does that without
  adding another component that leaves the process.
- **Trade-offs accepted:** More code than relying on a store feature, and an in-memory keyword index
  that has to be rebuilt on ingest and kept in step with the vector store. A small cross-encoder is
  weaker than a hosted one and adds CPU time to every query.
- **Revisit when:** The keyword index no longer fits comfortably in memory (move to Qdrant's sparse
  vectors), or answers stay poorly grounded with re-ranking on, which could mean the reranker is the
  limit.

---

## D-005: Orchestration with LangGraph, and agents have skills
- **Decision:** Each request runs through one LangGraph graph (rewrite the question, route it, run
  the chosen agent), built from the agent registry. Each agent is a tool loop over its declared
  skills: document search, and a table listing plus read-only SQL for the structured data. Adding
  an agent is still one module declaring a name, description, skills and prompt.
- **Alternatives considered:** Building the agents from scratch (no familarity and could be taking too much time);
  CrewAI or AutoGen (no familarity and could be taking too much time); 
- **Rationale:** Ordinary questions about this corpus need more than one step: a second search, a
  look at another table, a retry after a bad query. Writing that loop by hand would be rebuilding
  what LangGraph already provides. Building the graph from the registry keeps the property worth
  protecting: a new agent is one file, and the router, API and frontend pick it up. The failure
  rules also sit in graph nodes, so they apply to every agent in the same way.
- **Trade-offs accepted:** Agents with tools are a bigger prompt-injection target. This is accepted
  because every skill is read-only and in-process, so the SQL containment in D-006 now carries real
  weight. 
- **Revisit when:** Agents need to run in parallel and merge results, a conversation must resume
  mid-graph after a human step.

---

## D-006: Structure-aware chunking rather than a fixed character window
- **Decision:** Chunks follow the structure of each source. PDF narrative splits at the reports'
  numbered section headings, PDF tables become one chunk each and are stitched back together across
  page breaks, and spreadsheet tables become groups of rows bounded by size. Each report also gets a
  short header chunk with its period, status and identifiers.
- **Alternatives considered:** Recursive character splitting (the usual default and the least code);
  one chunk per page (trivial, and citations are just page numbers); one chunk per table row.
- **Rationale:** Both simple options break on this corpus in practice. Plain text extraction
  scrambles table rows, and character splitting then cuts tables mid-row. Page chunks split the key
  risks table across two pages, leaving risks under a header with no context. The numbered headings
  make the structural approach cheap, and they give citations like "§5 Key risks (pp. 1-2)" that a
  reviewer can check by hand.
- **Trade-offs accepted:** Much more code than a text splitter. It depends on reports having
  numbered headings and ruled tables. A report without them falls back to one block of narrative,
  and a table without borders is read as prose.
- **Revisit when:** A source arrives without numbered headings or ruled tables, and heading
  detection has to fall back to font size or layout; A source that is a scanned PDF that requires an OCR scanner.

---

## D-007: Follow-ups are rewritten, and graph state is checkpointed to Redis
- **Decision:** Before routing, a follow-up question is rewritten into a standalone one by the cheap
  router model, so agents never see the conversation. The transcript lives in a bounded in-process
  store. The graph's own state is checkpointed per session to Redis when `REDIS_URL` is set, with a
  TTL, and in memory when it is not. If Redis cannot be reached, it falls back to memory with a
  warning rather than failing the request.
- **Alternatives considered:** Pasting the history into every agent prompt (no extra call, but the
  prompts grow without limit); 
- **Rationale:** On-prem, reading the prompt is what dominates latency, not writing the answer.
  Rewriting keeps every agent prompt to one question plus its context, however long the
  conversation runs, and keeps agents stateless. Graph state is different from the transcript: the
  framework writes it on every step, a mid-graph resume would need it, and in memory it survives
  nothing. Redis is one container with no schema, and a TTL handles expiry. It also runs on-prem
  without calling out.
- **Trade-offs accepted:** A follow-up costs one extra model call, and a bad rewrite quietly changes
  the question. 
- **Revisit when:** A second backend replica is needed, at which point the transcript moves into the
  same Redis; or a human-in-the-loop step makes checkpoints something a user relies on.

---

## D-008: Deployment host
- **Decision:** The backend runs as a container on Google Cloud Run: 1 vCPU and 1 GiB, scaling from
  zero to two instances, and each instance handles up to four requests at a time. The frontend is
  the static Vite build, served by Vercel, with the backend's URL baked in at build time.
- **Alternatives considered:** Vercel for the backend as well (one platform, but serverless Python
  cannot hold a 1.4GB image or keep the ONNX models warm between requests); Hugging Face Spaces
  (Docker Spaces now need a paid plan); Railway (its cheapest plan has less memory than the app
  needs); Render's free tier (kept as a documented fallback, but 0.1 vCPU is too little for local
  embedding and re-ranking on every request); serving the frontend from the backend container (one
  deploy, but every UI change would rebuild and redeploy the heavy image).
- **Rationale:** The backend is a long-running Python process that loads two ONNX models into memory
  and embeds and re-ranks on the CPU for every question. It needs a real container with a full CPU,
  not a function. Cloud Run runs the same image as the Compose stack, so what I test locally is what
  gets deployed. At 1 GiB it leaves about 3x headroom over the measured peak of roughly 350MB, and
  it scales to zero, so an idle demo costs nothing. Building the image with the model weights baked
  in and turning on CPU boost keeps the cold start short. The frontend is the opposite case: once
  built, it is only static files, and Vercel serves them from a CDN, deploys each push and gives a
  preview URL per branch. Splitting the two means the UI can ship in seconds without touching the
  backend, and the only link between them is one URL and the backend's CORS setting.
- **Trade-offs accepted:** The first request after an idle period waits for a cold start, which is
  the price of scaling to zero. A single instance's filesystem is thrown away when it shuts down,
  so an uploaded file is not kept, only its indexed chunks. Two platforms means two deploys, and
  the backend URL is fixed into the frontend build, so changing it means rebuilding the frontend.
- **Revisit when:** The demo needs to stay warm, uploads need to persist, or more than one replica
  must hold a conversation, at which point the in-process transcript store (D-007) becomes the limit
  before the host does.
