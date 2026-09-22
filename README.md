# Project Intelligence Assistant

An AI assistant that ingests project documents (PDF status reports, CSV/Excel financials and risk registers) and answers questions about project status, risks and budgets. Queries are routed to specialised agents on top of a retrieval-augmented generation (RAG) pipeline, and every answer shows which agent handled it and the sources it used.

> **Live demo:** _TBD_
> **Walkthrough video (3–5 min):** _TBD_

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design, technology choices, data pipeline, agent orchestration, cost and scaling
- [DECISIONS.md](DECISIONS.md): log of key technical decisions and trade-offs
- [eval/](eval/): RAGAS evaluation set and results
- [data/](data/): synthetic sample data and the intentional messiness it contains

## Repository layout

```
Makefile            Docker and local dev entry points (run `make` for the list)
backend/            FastAPI + LangChain service (uv-managed)
  app/
    api/            REST routes and request/response schemas
    agents/         Router + specialised agents (plug-in registry)
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

`make` with no target lists everything available.

### 1. Configure
```bash
make env       # copies .env.example to .env
# fill in LLM / embedding provider settings
```

### 2. Run with Docker
```bash
make up        # build + start; make down to stop, make logs to follow
```
Backend: http://localhost:8000 (API docs at `/docs`) · Frontend: http://localhost:5173

### 3. Run locally without Docker
```bash
make install        # uv sync for the backend, npm install for the frontend
make dev-backend    # uvicorn with autoreload on :8000
make dev-frontend   # Vite dev server on :5173 (new terminal)
```

### 4. Run tests
```bash
make test
```

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
3. Decorate it with `@register_agent`.

Agent modules are discovered automatically, so no router or API changes are needed. See `ARCHITECTURE.md` for details.

## Example questions

_TBD once sample data is in place._
