# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-agent RAG assistant over project documents (PDF status reports, CSV/Excel financials and risk registers). A router agent picks a specialised agent per question; every answer reports which agent handled it and the sources it used.

**Most of the backend is implemented.** Ingestion, cleaning, chunking, retrieval, the vector and tabular stores, configuration, the model providers, all three agents, session memory, the observability layer and every REST endpoint are real and test-covered. Before changing a module, check whether it is a stub to be filled in or working code — the remaining stubs are `eval/run_ragas.py` and `eval/test_queries.yaml` (a skeleton). The React chat UI is implemented.

## Commands

`make` with no target lists everything. Key ones:

```bash
make up                 # docker compose up -d --build (creates .env from .env.example first)
make down / make logs   # stop / follow logs
make test               # backend test suite
make install            # uv sync (backend) + npm install (frontend)
make dev-backend        # uvicorn --reload on :8000
make dev-frontend       # Vite dev server on :5173
make data / make eval   # run data/scripts/generate_synthetic_data.py / eval/run_ragas.py
```

Single test or subset (there is no make target for this):

```bash
uv run --directory backend pytest tests/test_registry.py::test_router_excluded_from_routable_agents
uv run --directory backend pytest -k registry -q
```

## Dependencies

uv only — there is no `requirements.txt`. Add packages with `uv add <pkg> --directory backend` (`--dev` for tooling) and commit the updated `backend/uv.lock`; the image builds with `uv sync --locked --no-dev`, so an unlocked dependency breaks the Docker build. Backend config (including pytest's `testpaths`/`pythonpath`) lives in `backend/pyproject.toml`; the project is `package = false`, so `app` is imported from the `backend/` working directory rather than installed.

## Architecture

Request flow: React → `POST /v1/chat` → **one LangGraph state graph** (`app/graph/build.py`): `rewrite` (session memory turns a follow-up into a standalone question) → `route` → the chosen agent's node. Each agent is a `create_agent` tool loop over its **skills** (`app/skills/`) — `search_documents` for hybrid retrieval + re-rank, `describe_tables`/`run_sql` for the cleaned tables — and the loop ends in a `ChatResponse` with `agent`, `citations` and `trace_id`. See DECISIONS.md D-015, which amends D-006.

**Agent registry (`app/agents/registry.py`) is the central extension point**, and the graph is compiled *from* it. Adding an agent means adding one module under `app/agents/` with a `BaseAgent` subclass decorated `@register_agent`, declaring `name`, `description`, `skills` and `system_prompt`; `run()` is inherited from `BaseAgent` (it builds the tool loop, applies the budget and collects evidence), so override `finalise()` only to shape the `AgentResult`. `discover_agents()` imports every module in the package, so the router, API and frontend pick it up with no edits. Consequences worth knowing:
- A new *non-agent* module in `app/agents/` must be added to `_NON_AGENT_MODULES` or discovery will import it looking for agents. Nothing fails if you forget — discovery just imports it on every `list_agents()` call and finds nothing — so by convention shared helpers live outside the package instead: prompt text in `app/llm/prompts.py`, skills in `app/skills/`, the graph in `app/graph/`, the SQL guard in `app/ingestion/tabular_store.py` beside the connection it protects.
- `register_agent` rejects a skill that is not a LangChain tool, because the failure would otherwise surface deep inside a model call in a traceback naming neither the agent nor the skill.
- `get_chat_graph()` caches the compiled graph, so a test registering an agent must call `reset_chat_graph()` — `tests/conftest.py` does this automatically between tests.
- The router builds its prompt catalogue from each agent's `description` (`RouterAgent.agent_catalogue`), so a `description` is routing logic, not a comment. Registration rejects a class without `name` and `description`.
- `list_agents()` hides the router itself; `list_agents(include_router=True)` includes it.
- `skills = []` is legal: `small_talk` is prompt-only, so its `create_agent` loop ends on the first
  turn and it overrides `finalise()` to return no citations rather than an "inferred" attribution.
  It is also `FALLBACK_AGENT`, so an unroutable question lands there and is declined rather than
  attempted — which makes its prompt's decline path, not its greeting, the part that matters (D-018).

All agents share the `AgentInput` → `AgentResult` contract in `app/agents/base.py`, which is what lets the router, API and logs treat them uniformly. Agents receive an already-rewritten standalone question — follow-up resolution belongs in `app/memory/`, not in an agent.

Routing must never fail a request. `RouterAgent.route()` holds the fallback rules (unknown agent or confidence below `MIN_CONFIDENCE` falls back to `FALLBACK_AGENT`, which is `small_talk`; a routing failure becomes a fallback route) and the graph's agent node degrades a failing agent into an `AgentResult` carrying `metadata["error"]`. `NotImplementedError` is deliberately re-raised in both, so a half-built future agent surfaces loudly rather than being swallowed. `RouterAgent.run` is now a thin shim that runs the graph with `needs_rewrite=False`. The decision is recorded in `result.metadata["routing"]`, which the API logs but does not return — `ChatResponse` has no metadata field.

**Skills are read-only and in-process, deliberately.** ARCHITECTURE.md §8.1 used to rest on "the answering agents have no tools"; D-015 ended that, and what replaces it is that no skill performs network or filesystem access. `run_sql` is the one place model output becomes executable, so the D-007/D-014 containment in `app/ingestion/tabular_store.py` is load-bearing — don't bypass `validate_select()` or `read_only_connection()`. The tool budget (`AGENT_MAX_TOOL_CALLS`) is enforced in `app/skills/evidence.py` rather than with `ToolCallLimitMiddleware`, which blocks a call without ending the graph and so burns the recursion limit instead of returning an answer.

**The graph checkpoints per session**, keyed by session ID, into Redis when `REDIS_URL` is set and
in-process otherwise (D-017). `app/graph/checkpointer.py` owns that choice: `run_chat_graph` awaits
`ensure_checkpointer_ready()` before compiling, because Redis needs its indices created and an
unreachable one degrades to `InMemorySaver` — which recompiles the graph. Don't make the checkpoint
carry conversation context; follow-ups are still rewritten (D-013), so what reaches an agent is one
question however long the conversation runs. `AsyncRedisSaver` implements only the async half of the
interface (its `delete_thread` raises), so eviction detaches `adelete_thread` onto the running loop
rather than awaiting it. The test suite blanks `REDIS_URL` in `tests/conftest.py`; the one test that
needs a real Redis is marked `redis` and skips itself when none is running.

**Citations come from the evidence collector** (`app/skills/evidence.py`), a ContextVar mirroring `trace_id_var`. Skills record what they returned and it assigns the `[n]` numbering globally across calls, so a second search continues rather than restarting at `[1]`. A skill that returns passages must record them, or they cannot be cited.

**All model access goes through `app/llm/providers.py`.** Provider and model come from settings, so the on-prem requirement (swap to a local model with no external calls) stays a configuration change. Don't call an LLM or embedding SDK directly from an agent or the ingestion pipeline.

**Configuration** is a single `Settings` object (`app/config.py`, cached via `get_settings()`) fed by env vars / `.env`; add new knobs there and to `.env.example` rather than reading `os.environ` elsewhere.

**Observability**: `RequestContextMiddleware` assigns an `x-trace-id` per request into a contextvar, and `JsonFormatter` stamps it onto every log record. Emit structured fields as `logger.info("msg", extra={"fields": {...}})` — a plain dict argument will not appear in the JSON output.

**Retrieval** has two modes by design: `RETRIEVAL_MODE=naive` (vector-only) is the RAGAS baseline that the hybrid mode is measured against, so keep it working when changing `app/retrieval/`.

Frontend is a thin Vite/React client (plain CSS in `src/styles.css`, no UI framework); all backend calls go through `src/api/client.js`, which prefixes `/v1` (the backend mounts its router there; `/health` is unprefixed), and the base URL comes from `VITE_API_BASE_URL` (baked in at build time in Docker, read from `.env` in dev). Conversations are stored client-side in `localStorage` because `GET /v1/sessions/{id}` does not return citations. The UI maps an answer's `[n]` markers to `citations[n-1]`, relying on the renumbering in `app/skills/evidence.py`. Agent badges are driven by `GET /v1/agents`, so there is no frontend list of agents to update.

## Documentation expectations

`ARCHITECTURE.md` and `DECISIONS.md` are deliverables with fixed skeletons, not optional notes. Technology choices (LLM, embeddings, vector store, orchestration) get a `D-00N` entry recording decision, alternatives, rationale, trade-offs and when to revisit; `data/README.md` tracks each intentional flaw in the synthetic data and how the pipeline handles it.
