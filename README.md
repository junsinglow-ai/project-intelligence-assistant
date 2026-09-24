# Project Intelligence Assistant

An AI assistant that ingests project documents (PDF status reports, CSV/Excel financials and risk registers) and answers questions about project status, risks and budgets. Queries are routed to specialised agents on top of a retrieval-augmented generation (RAG) pipeline, and every answer shows which agent handled it and the sources it used.

> **Live demo:** https://frontend-bay-nu-35.vercel.app
> **API:** https://project-intelligence-backend-215402207736.us-central1.run.app
> ([`/docs`](https://project-intelligence-backend-215402207736.us-central1.run.app/docs) ·
> [readiness](https://project-intelligence-backend-215402207736.us-central1.run.app/v1/health/dependencies))
> **Walkthrough video (3–5 min):** _TBD_
>
> Frontend on Vercel, backend on Cloud Run with Qdrant Cloud and Redis Cloud behind it — all inside
> free tiers (DECISIONS.md D-009). The backend scales to zero, so the first request after an idle
> period waits for a cold start. Generation runs on Gemini's free tier, which allows 15 requests
> per minute per model; asking questions in quick succession trips it, the model chain falls through
> to the next model, and the limit clears within seconds.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design, technology choices, data pipeline, agent orchestration, cost and scaling
- [DECISIONS.md](DECISIONS.md): log of key technical decisions and trade-offs
- [eval/](eval/): RAGAS evaluation set and results
- [data/](data/): synthetic sample data and the intentional messiness it contains
- [docs/synthetic-data-skill.md](docs/synthetic-data-skill.md): how to change or regenerate that data with your own coding agent

## Repository layout

```
Makefile            Docker and local dev entry points (run `make` for the list)
backend/            FastAPI + LangChain/LangGraph service (uv-managed)
  app/
    api/            REST routes and request/response schemas
    agents/         Router + specialised agents (plug-in registry)
    skills/         The tools agents call, and the evidence/citation collector
    graph/          The LangGraph chat graph and its checkpointer
    ingestion/      PDF and CSV/Excel loaders, cleaning, chunking
    retrieval/      Hybrid search and re-ranking
    llm/            LLM and embedding provider factory
    memory/         Conversation session store
    observability/  Structured logging and tracing
  tests/
frontend/           React (Vite) UI: upload, chat, agent badge, citations
data/               Synthetic sample documents + generation script
eval/               RAGAS test queries, runner and results
docs/               Diagrams and screenshots
```

## Quick start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (for the containerised setup)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Node.js 20+ (for local development)
- **Cloud mode:** a Google Gemini API key (free tier is sufficient)
- **On-prem mode:** a reachable Ollama endpoint — the bundled container, or any host on your network

The first run downloads roughly 150MB of ONNX model weights for the local embedder and re-ranker,
and caches them. `make` with no target lists every available command.

### 1. Configure
```bash
make env       # copies .env.example to .env
```
Cloud mode (the default) needs one value in `.env`:
```bash
LLM_API_KEY=your-gemini-key
```
On-prem mode needs no key at all — see [Deployment modes](#deployment-modes) below.

### 2. Run with Docker
```bash
make up        # build + start; make down to stop, make logs to follow
make ready     # check the LLM, vector store, DuckDB and Redis, and see the resolved config
```
Four services come up: the FastAPI backend, the React frontend, **Qdrant**, which holds the vector
index, and **Redis**, which holds the chat graph's checkpoints. Both keep their data in named
volumes (`qdrant-storage`, `redis-data`) that survive `make down`; `make clean` removes them.

Backend: http://localhost:8000 (API docs at `/docs`) · Frontend: http://localhost:5173 ·
Qdrant: http://localhost:6333/dashboard · Redis: localhost:6379

### 3. Run locally without Docker
```bash
make install        # uv sync for the backend, npm install for the frontend
make qdrant         # start just the Qdrant container (the index lives there)
make redis          # start just the Redis container (graph checkpoints live there)
make dev-backend    # uvicorn with autoreload on :8000
make dev-frontend   # Vite dev server on :5173 (new terminal)
```
To run with no container at all, blank `QDRANT_URL` and `REDIS_URL` in `.env`: Qdrant then runs
embedded, in-process, from `QDRANT_PATH`, and the graph checkpoints in-process too. The Qdrant
trade-off is a directory lock that only one process can hold — see below; the checkpoint trade-off
is that graph state is lost on restart, which is what Redis is there to fix
([DECISIONS.md](DECISIONS.md) D-017).

### 4. Index the documents
```bash
make ingest    # loads data/raw into the vector index and the DuckDB store
```
This cleans the source documents, chunks them, embeds each chunk locally and writes both the vector
index and the structured tables. It prints what it indexed and which data-quality flaws it handled.
Re-running it is safe: chunks are replaced rather than duplicated. `make reindex` drops the index
first, which is required after changing `EMBEDDING_MODEL`.

`make ingest` writes to whatever `QDRANT_URL` points at, which by default is the Qdrant container on
`localhost:6333` — the same index the running backend reads, so **the stack can stay up**. Start it
first with `make up` or `make qdrant`. Individual documents can also be added at runtime through
`POST /v1/upload` or the paperclip in the UI.

In embedded mode (blank `QDRANT_URL`) Qdrant locks its directory instead, so only one process can
hold the index and the stack has to be stopped before ingesting.

### Using the UI
Open http://localhost:5173. The layout follows Open WebUI: conversations in the left sidebar, the
chat in the centre, a composer at the bottom.

- **Ask** in the composer (Enter sends, Shift+Enter adds a line). Follow-ups in the same chat reuse
  the session, so "who prepared it?" resolves against the previous answer.
- **Upload** PDF, CSV or Excel files with the paperclip or by dropping them onto the chat. Each file
  is indexed into the shared corpus; the chat shows how many chunks were indexed.
- **Agent badge**: every answer is labelled with the agent that handled it (hover for its
  description). The badges come from `GET /v1/agents`, so a new agent appears without frontend edits.
- **Citations**: `[n]` markers in an answer are clickable and open the matching source, with the
  passage it came from. Data Analysis citations show the SQL that produced the figure.

Conversations are kept in the browser's local storage, with their citations, so a reload restores
them; the trace ID under each answer matches the backend logs.

### 5. Run tests
```bash
make test                                          # full suite
uv run --directory backend pytest -m "not slow"    # skip tests that load local model weights
```
The suite needs no containers: it runs Qdrant embedded against a temporary directory and
checkpoints in-process. The one test that needs a real Redis is marked `redis` and skips itself
when nothing answers at `REDIS_TEST_URL` (default `redis://localhost:6379/0`); run `make redis`
first to include it.

## Deployment modes

The system runs either against a hosted LLM or fully on-premises with no external API calls, and
moving between them is **one variable**:

```bash
DEPLOYMENT_MODE=cloud     # default: Gemini for generation
DEPLOYMENT_MODE=onprem    # Ollama for generation; nothing leaves your network
```

To run on-premises with the bundled Ollama container:

```bash
# set DEPLOYMENT_MODE=onprem in .env, then
make up-onprem   # starts the stack plus an Ollama container
make models      # pulls the configured models (~6GB on first run)
make ready       # confirms the endpoint is reachable and the models are present
```

To use an existing Ollama host instead — typically a machine with a GPU — point at it and skip the
bundled container:

```bash
OLLAMA_BASE_URL=http://gpu-box.internal:11434
```

If that host is the machine running Docker, note that Ollama listens on `127.0.0.1` by default and is
therefore unreachable from inside a container. Start it bound to all interfaces and point at the
Docker gateway:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
OLLAMA_BASE_URL=http://172.17.0.1:11434     # `docker network inspect bridge` shows the gateway
```

`make ready` reports exactly this failure when it happens, rather than leaving it to surface as a
timeout on the first question.

**Hardware note.** The on-prem default (`qwen2.5:7b`) needs roughly 8GB of free RAM on the Ollama
host. Below that the host swaps and requests effectively stall — measured on a 12-core CPU-only
laptop with ~4GB free, the 7B model failed to answer within 300s while `llama3.2:3b` answered the
same question in 154s. On a constrained host, set `LLM_MODEL=llama3.2:3b`.

Nothing else changes between modes. Embeddings, keyword search, re-ranking, the vector store, the
tabular engine and the graph checkpointer all stay inside the deployment — in-process, apart from the
Qdrant and Redis containers on the same network — and are identical either way, so **no re-index is needed** and an index built in
cloud mode works on-prem unchanged. The trade-off is speed: on CPU-only hardware an
on-prem answer takes 60–120s against a few seconds in cloud mode, which is why the request timeout
also follows the mode. See [ARCHITECTURE.md](ARCHITECTURE.md) §9 and
[DECISIONS.md](DECISIONS.md) D-010.

The one setting that is *not* config-only is `EMBEDDING_MODEL`: vector dimensions differ between
models, so changing it requires `make reindex`. The model name is stamped into the vector collection
name so a mismatch fails loudly instead of returning nonsense.

## Deploying

The hosted demo runs the backend on **Google Cloud Run**, the frontend on **Vercel**, and keeps the
vector index and the graph checkpoints in **Qdrant Cloud** and **Redis Cloud** — so the deployed
system has the same shape as the Compose stack rather than a reduced variant of it. All four sit
inside free tiers. [DECISIONS.md](DECISIONS.md) D-009 records why, and what was rejected.

**Redis Cloud, not Upstash.** The checkpointer creates search indices, so it needs a Redis with the
RediSearch module. Upstash has no RediSearch and the failure is quiet: `app/graph/checkpointer.py`
degrades to the in-process saver with a warning and answers anyway. `make deploy-check` is what
catches it — see step 5.

### Local and hosted side by side

The two configurations live in separate files, so neither has to be edited back
and forth:

| | `.env` | `.env.cloud` |
|---|---|---|
| Qdrant | the `qdrant` container | Qdrant Cloud cluster |
| Redis | the `redis` container | Redis Cloud database |
| Used by | `make up`, `make ingest`, `make ready`, the test suite | `make ingest-cloud`, `make ready-cloud`, `make deploy-*` |
| Committed | no (`.env.example` is) | no (`.env.cloud.example` is) |

`make env` creates both from their templates. The cloud targets source
`.env.cloud` into the environment on top of `.env`, and real environment
variables beat the dotenv file, so the managed endpoints win while everything
not named there — `LLM_API_KEY`, the chunking knobs — still comes from `.env`.
Running the local stack therefore needs no change at all: `make up` and
`make ingest` keep using the containers.

```bash
make ready         # resolve against the local containers
make ready-cloud   # resolve against the managed services, as the deployment will
```

`make ready-cloud` blanks the variables the deployment will not inherit from
`.env`, so it reports the provider and model chain Cloud Run will actually
resolve rather than whatever your local `.env` happens to say.

### 1. Provision

| Service | Tier | What to keep |
|---|---|---|
| [Qdrant Cloud](https://cloud.qdrant.io) | free (1GB RAM / 4GB disk) | cluster URL, API key |
| [Redis Cloud](https://redis.io/try-free/) | free (30MB, no card) | the `rediss://` URL |
| [Google AI Studio](https://aistudio.google.com/apikey) | free | Gemini API key |
| [Google Cloud](https://console.cloud.google.com) | always-free | project ID, **billing enabled** |

Free Qdrant clusters suspend after a week idle and are deleted after four, so a demo left alone for
a month needs the cluster recreated and `make ingest` re-run.

### 2. Build the index and the tables

Fill in `.env.cloud` (created by `make env`) with the cluster URL and key, then:

```bash
make ingest-cloud
```

This fills the cloud index **and** rebuilds `data/processed/tables.duckdb`, which the image bakes in
so the Data Analysis agent has its tables with no volume to mount. `make ingest` still targets the
local container, so the two indexes stay independent.

### 3. Deploy the backend

`GCP_PROJECT`, `GCP_REGION` and the service URLs all come from `.env.cloud`, so there is nothing to
pass on the command line:

```bash
make deploy-setup    # one-time: enables the APIs, creates Artifact Registry and both secrets
make deploy-backend
make deploy-url      # the backend URL, needed by the next step
```

The region must be `us-central1`, `us-east1` or `us-west1` — the Cloud Run free tier covers no
others, and `make deploy-backend` refuses the rest rather than quietly deploying a billable service.

### 4. Deploy the frontend, then close the CORS loop

On Vercel: **Root Directory `frontend`**, framework Vite, and `VITE_API_BASE_URL` set to the backend
URL. It is read at build time (`src/api/client.js` goes through Vite's `import.meta.env`), so
changing it later needs a rebuild, not a restart.

The two URLs depend on each other, so the last step is a second backend deploy that tells it about
the frontend:

Set `CORS_ORIGINS` in `.env.cloud` to the Vercel origin, then redeploy:

```bash
make deploy-backend
```

Get this wrong and the UI shows a permanently red health badge rather than an error — the browser
calls the backend cross-origin and there is no proxy in front of it.

### 5. Check it

```bash
make deploy-check
```

Expect `status: ok`, `vector_store.mode: server`, and `checkpointer.mode: redis`. A `checkpointer`
reading `in-process` means Redis is unreachable or has no RediSearch.

### Without a GCP billing account

Cloud Run has required billing since February 2026. `render.yaml` deploys the same image to Render's
free tier instead, keeping Qdrant Cloud and Redis Cloud behind it. The cost is 512MB against a
measured 348MB warm footprint, 0.1 vCPU for a request that runs an ONNX embedding and a
cross-encoder re-rank, and a spin-down after 15 minutes idle.

## Dependencies

Backend dependencies are declared in `backend/pyproject.toml` and pinned in
`backend/uv.lock`, which is committed; the Docker image installs from the lock
file with `uv sync --locked`, so containers and local checkouts get identical
versions. Add a package with `uv add <pkg> --directory backend` (or
`--dev` for tooling), then commit the updated lock file. `make lock` re-resolves
and `make upgrade` moves dependencies up within their declared bounds.

## Adding a new agent

1. Create `backend/app/agents/<your_agent>.py` with a class that subclasses `BaseAgent`.
2. Give it a unique `name` and a clear `description` (the router uses the description to decide when to call it).
3. List its `skills` — the tools its model may call — and give it a `system_prompt`. Existing skills are in `backend/app/skills/`; a new one is a `@tool`-decorated function there.
4. Decorate it with `@register_agent`.

```python
@register_agent
class RiskAgent(BaseAgent):
    name = "risk"
    description = "Answers questions about the risk register: owners, scores, mitigations."
    skills = [describe_tables, run_sql]
    system_prompt = RISK_SYSTEM
```

`run()` is inherited: it runs the tool loop, applies the budget and collects the evidence citations are built from. Agent modules are discovered automatically and the graph is compiled from the registry, so no router, graph or API changes are needed. See `ARCHITECTURE.md` §5.5 for details.

## Example questions

_TBD once sample data is in place._
