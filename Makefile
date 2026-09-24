# Project Intelligence Assistant
#
# Docker is the primary way to run the stack; the local targets use uv
# (backend) and npm (frontend) for day-to-day development.

COMPOSE := docker compose
BACKEND := backend
FRONTEND := frontend

# The managed-services overrides used by the `*-cloud` and `deploy-*` targets.
# Declared here rather than beside them because it names a target below, and
# make expands a target name as it reads the rule.
CLOUD_ENV ?= .env.cloud

.DEFAULT_GOAL := help
.PHONY: help env up up-onprem down restart build rebuild logs logs-backend logs-frontend ps \
        qdrant redis ready models ingest reindex query sql shell-backend shell-frontend clean install dev-backend dev-frontend test \
        data eval lock upgrade ingest-cloud reindex-cloud ready-cloud \
        deploy-setup deploy-backend deploy-url deploy-check \
        logs-cloud logs-cloud-errors logs-cloud-trace

help: ## Show available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Docker ---------------------------------------------------------------

.env:
	@cp .env.example .env
	@echo "Created .env from .env.example - fill in the LLM / embedding settings."

# Separate from .env on purpose: .env points at the local Qdrant and Redis
# containers, and this holds the managed endpoints that replace them for the
# hosted demo. Keeping them apart is what lets `make up` and `make deploy-*`
# coexist without editing one file back and forth.
$(CLOUD_ENV):
	@cp .env.cloud.example $(CLOUD_ENV)
	@echo "Created $(CLOUD_ENV) - fill in the Qdrant Cloud, Redis Cloud and GCP values."

env: .env $(CLOUD_ENV) ## Create .env and .env.cloud from their templates if missing

up: env ## Build if needed and start the stack in the background
	$(COMPOSE) up -d --build
	@echo "Backend:  http://localhost:8000  (API docs at /docs)"
	@echo "Frontend: http://localhost:5173"
	@echo "Qdrant:   http://localhost:6333/dashboard"
	@echo "Redis:    localhost:6379  (graph checkpoints)"

up-onprem: env ## Start the stack in on-premises mode, with a local Ollama container
	DEPLOYMENT_MODE=onprem $(COMPOSE) --profile onprem up -d --build
	@echo "Ollama:   http://localhost:11434  (run 'make models' to pull the models)"
	@echo "Backend:  http://localhost:8000  (readiness at /v1/health/dependencies)"

down: ## Stop the stack
	$(COMPOSE) --profile onprem down

restart: down up ## Restart the stack

build: env ## Build the images
	$(COMPOSE) build

rebuild: env ## Build the images from scratch, ignoring the cache
	$(COMPOSE) build --no-cache

logs: ## Follow logs from all services
	$(COMPOSE) logs -f

logs-backend: ## Follow backend logs
	$(COMPOSE) logs -f backend

logs-frontend: ## Follow frontend logs
	$(COMPOSE) logs -f frontend

ps: ## Show container status
	$(COMPOSE) ps

# The vector store is a container even when the rest of the stack runs from the
# host, so `make dev-backend` and `make ingest` have an index to talk to.
qdrant: ## Start only the Qdrant service (for local development)
	$(COMPOSE) up -d qdrant
	@echo "Qdrant:   http://localhost:6333/dashboard"

# Same reason as `make qdrant`: `make dev-backend` checkpoints into this. Blank
# REDIS_URL in .env to run with no container at all.
redis: ## Start only the Redis service (for local development)
	$(COMPOSE) up -d redis
	@echo "Redis:    localhost:6379  (graph checkpoints)"

ready: ## Report LLM, vector store, DuckDB and checkpointer readiness, and the resolved mode
	@curl -fsS http://localhost:8000/v1/health/dependencies \
		| python3 -m json.tool || echo "Backend not reachable on :8000"

models: ## Pull the on-prem models onto the configured Ollama host
	@uv run --directory $(BACKEND) python -c "$$PULL_MODELS"

# These write to whatever QDRANT_URL points at -- by default the Qdrant
# container, which `make up` or `make qdrant` starts -- so the stack can stay
# running. In embedded mode (blank QDRANT_URL) the directory lock applies
# instead and the API must be stopped first.
ingest: ## Index every document in data/raw into the running Qdrant
	@uv run --directory $(BACKEND) python -c "$$INGEST"

reindex: ## Drop the index and rebuild it (required after EMBEDDING_MODEL changes)
	@uv run --directory $(BACKEND) python -c "$$REINDEX"

query: ## Retrieve chunks for a question: make query Q="which risk is largest?"
	@Q="$(Q)" uv run --directory $(BACKEND) python -c "$$QUERY"

sql: ## Query the structured store read-only: make sql S="select * from risk_register"
	@S="$(S)" uv run --directory $(BACKEND) python -c "$$SQL"

shell-backend: ## Open a shell in the backend container
	$(COMPOSE) exec backend bash

shell-frontend: ## Open a shell in the frontend container
	$(COMPOSE) exec frontend sh

# -v removes the Qdrant storage volume, so this discards the index; `make
# ingest` rebuilds it. It also discards the Redis checkpoints, which nothing
# needs to rebuild.
clean: ## Stop the stack and remove its volumes (including the index) and local images
	$(COMPOSE) down -v --rmi local --remove-orphans

# --- Local development ----------------------------------------------------

install: ## Install backend (uv) and frontend (npm) dependencies
	uv sync --directory $(BACKEND)
	npm --prefix $(FRONTEND) install

dev-backend: ## Run the backend with autoreload on :8000
	uv run --directory $(BACKEND) uvicorn app.main:app --reload

dev-frontend: ## Run the Vite dev server on :5173
	npm --prefix $(FRONTEND) run dev

test: ## Run the backend test suite
	uv run --directory $(BACKEND) pytest

# Scripts live at the repo root but run against the backend environment; the
# rendering libraries are in the backend's optional `data` dependency group.
data: ## Regenerate the synthetic sample documents in data/raw
	uv run --project $(BACKEND) --group data python data/scripts/generate_synthetic_data.py

eval: ## Run the RAGAS evaluation into eval/results
	uv run --project $(BACKEND) python eval/run_ragas.py

# --- Deployment (DECISIONS.md D-009) --------------------------------------
# Backend on Cloud Run, frontend on Vercel, with Qdrant Cloud and Redis Cloud
# behind them.
#
# Everything here reads `.env.cloud` rather than `.env`, which is what keeps
# the local stack and the hosted one from fighting over the same variables:
# `.env` points QDRANT_URL and REDIS_URL at the containers `make up` starts,
# and `.env.cloud` overrides them with the managed endpoints. It is loaded into
# the environment, so it wins over `.env` (pydantic-settings reads real
# environment variables ahead of the dotenv file) while everything not named in
# it -- LLM_API_KEY, the chunking knobs -- still comes from `.env`.
CLOUD_RUN_SERVICE ?= project-intelligence-backend

# `.env` is sourced first and `.env.cloud` layered on top, which is what makes
# the split work: LLM_API_KEY and the chunking knobs come from `.env` while the
# service endpoints come from `.env.cloud`. `set -a` exports every assignment so
# the values reach both the Python process and gcloud.
#
# The mode-sensitive LLM knobs are then dropped unless `.env.cloud` names them.
# Without that, a developer running on-prem locally would deploy with
# LLM_PROVIDER=ollama exported into the build -- `.env` describes the local
# stack, and none of it should follow a cloud target. Anything `.env.cloud`
# sets explicitly is kept, so overriding a model for the deployment still works.
define load_cloud_env
env_file="$(CLOUD_ENV)"; \
case "$$env_file" in /*) ;; *) env_file="./$$env_file" ;; esac; \
test -f "$$env_file" || { echo "$(CLOUD_ENV) is missing - run 'make env' and fill it in"; exit 1; }; \
set -a; if [ -f ./.env ]; then . ./.env; fi; . "$$env_file"; set +a; \
for v in DEPLOYMENT_MODE LLM_PROVIDER LLM_MODEL ROUTER_MODEL LLM_TIMEOUT_S OLLAMA_BASE_URL; do \
	grep -qE "^[[:space:]]*$$v=" "$$env_file" || unset "$$v"; \
done; \
: $${GCP_REGION:=us-central1}
endef

# Guards shared by the deploy targets. Shell rather than make conditionals
# because the values arrive from the sourced file, after make has parsed this.
define require_gcp
test -n "$$GCP_PROJECT" || { echo "GCP_PROJECT is not set in $(CLOUD_ENV)"; exit 1; }; \
case "$$GCP_REGION" in us-central1|us-east1|us-west1) ;; \
  *) echo "GCP_REGION=$$GCP_REGION is outside the Cloud Run free tier (us-central1|us-east1|us-west1)"; exit 1 ;; esac
endef

ingest-cloud: ## Index data/raw into the managed Qdrant Cloud cluster
	@$(load_cloud_env); \
	echo "indexing into $$QDRANT_URL"; \
	uv run --directory $(BACKEND) python -c "$$INGEST"

reindex-cloud: ## Drop and rebuild the managed index (after EMBEDDING_MODEL changes)
	@$(load_cloud_env); \
	echo "reindexing $$QDRANT_URL"; \
	uv run --directory $(BACKEND) python -c "$$REINDEX"

ready-cloud: ## Resolve config against the managed services, without deploying
	@$(load_cloud_env); \
	uv run --directory $(BACKEND) python -c "$$READY_CLOUD"

# Ordering matters and is easy to get backwards: creating the repository before
# the Artifact Registry API is enabled fails with SERVICE_DISABLED, and an
# `|| echo "already exists"` fallback then reports that failure as success --
# which is exactly how the first run left no repository behind while printing
# "ready". `describe || create` distinguishes the two cases properly, and
# `set -e` stops the target at the first real failure.
#
# The IAM grant is not optional on a fresh project. Google no longer creates the
# legacy `cloudbuild.gserviceaccount.com` account, so builds run as the Compute
# Engine default service account, which starts with no build roles -- and
# `gcloud builds submit` then fails reading its own staging bucket with a 403 on
# storage.objects.get.
#
# The same account is Cloud Run's default runtime identity, and it needs
# secretmanager.secretAccessor as well or the revision never becomes ready:
# Cloud Run resolves --set-secrets when it starts the container, so a missing
# grant fails at revision creation rather than at deploy time. Bound per
# secret rather than project-wide, which is the narrower grant.
deploy-setup: ## One-time: enable the APIs, create the registry and both secrets
	@$(load_cloud_env); $(require_gcp); \
	set -e; \
	echo "enabling APIs (first run takes a couple of minutes)..."; \
	gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
		artifactregistry.googleapis.com secretmanager.googleapis.com \
		--project "$$GCP_PROJECT"; \
	gcloud artifacts repositories describe $(CLOUD_RUN_SERVICE) \
		--project "$$GCP_PROJECT" --location "$$GCP_REGION" >/dev/null 2>&1 \
		|| gcloud artifacts repositories create $(CLOUD_RUN_SERVICE) \
			--project "$$GCP_PROJECT" --location "$$GCP_REGION" --repository-format docker; \
	num=$$(gcloud projects describe "$$GCP_PROJECT" --format='value(projectNumber)'); \
	builder="$$num-compute@developer.gserviceaccount.com"; \
	echo "granting build roles to $$builder"; \
	gcloud projects add-iam-policy-binding "$$GCP_PROJECT" \
		--member="serviceAccount:$$builder" \
		--role=roles/cloudbuild.builds.builder --condition=None >/dev/null; \
	for pair in "llm-api-key:$$LLM_API_KEY" "qdrant-api-key:$$QDRANT_API_KEY"; do \
		name="$${pair%%:*}"; value="$${pair#*:}"; \
		test -n "$$value" || { echo "$$name has no value in $(CLOUD_ENV)"; exit 1; }; \
		gcloud secrets describe "$$name" --project "$$GCP_PROJECT" >/dev/null 2>&1 \
			|| gcloud secrets create "$$name" --project "$$GCP_PROJECT" --replication-policy=automatic; \
		printf %s "$$value" | gcloud secrets versions add "$$name" \
			--project "$$GCP_PROJECT" --data-file=- >/dev/null; \
		gcloud secrets add-iam-policy-binding "$$name" --project "$$GCP_PROJECT" \
			--member="serviceAccount:$$builder" \
			--role=roles/secretmanager.secretAccessor --condition=None >/dev/null; \
		echo "  secret $$name updated and readable by the runtime"; \
	done; \
	echo "APIs, Artifact Registry and secrets ready in $$GCP_PROJECT"

# AGENT_MAX_TOOL_CALLS is pinned rather than left to the code default of 3.
# Every extra tool call is another generation, and on a free tier each
# generation may pay a 429 fallthrough, so the budget is the latency knob in
# cloud mode exactly as it is on-prem (ARCHITECTURE.md section 7.1).
#
# `&&` between build and deploy, not `;`: a failed build otherwise rolls straight
# on to deploying an image that was never pushed, and the useful error scrolls
# past the useless one.
deploy-backend: ## Build and deploy the backend image to Cloud Run
	@$(load_cloud_env); $(require_gcp); set -e; \
	test -n "$$QDRANT_URL" || { echo "QDRANT_URL is not set in $(CLOUD_ENV)"; exit 1; }; \
	test -f data/processed/tables.duckdb || { echo "data/processed/tables.duckdb is missing - run 'make ingest-cloud' first"; exit 1; }; \
	image="$$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$(CLOUD_RUN_SERVICE)/backend:latest"; \
	gcloud builds submit --project "$$GCP_PROJECT" --config cloudbuild.yaml \
		--substitutions _IMAGE="$$image" . && \
	gcloud run deploy $(CLOUD_RUN_SERVICE) \
		--project "$$GCP_PROJECT" --region "$$GCP_REGION" \
		--image "$$image" --port 8080 \
		--memory 1Gi --cpu 1 --min-instances 0 --max-instances 2 \
		--cpu-boost --timeout 300 --concurrency 4 \
		--allow-unauthenticated \
		--set-env-vars "^##^DEPLOYMENT_MODE=cloud##QDRANT_URL=$$QDRANT_URL##REDIS_URL=$$REDIS_URL##CORS_ORIGINS=$$CORS_ORIGINS##AGENT_MAX_TOOL_CALLS=$${AGENT_MAX_TOOL_CALLS:-2}" \
		--set-secrets "LLM_API_KEY=llm-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest"
	@# `^##^` is gcloud's alternate delimiter: CORS_ORIGINS is itself a
	@# comma-separated list, so the default comma separator would split it into
	@# several malformed variables.

deploy-url: ## Print the deployed backend URL
	@$(load_cloud_env); $(require_gcp); \
	gcloud run services describe $(CLOUD_RUN_SERVICE) \
		--project "$$GCP_PROJECT" --region "$$GCP_REGION" --format 'value(status.url)'

# `gcloud logging tail` needs grpc extras that the standard SDK install does not
# ship, so these read rather than stream. LOG_FRESHNESS bounds the window;
# without it a query scans far further back than it needs to.
LOG_FRESHNESS ?= 1h
LOG_LIMIT ?= 50

logs-cloud: ## Recent request logs from the deployed backend
	@$(load_cloud_env); $(require_gcp); \
	gcloud run services logs read $(CLOUD_RUN_SERVICE) \
		--project "$$GCP_PROJECT" --region "$$GCP_REGION" --limit $(LOG_LIMIT)

logs-cloud-errors: ## Only failures from the deployed backend
	@$(load_cloud_env); $(require_gcp); \
	gcloud logging read \
		'resource.type=cloud_run_revision AND resource.labels.service_name=$(CLOUD_RUN_SERVICE) AND severity>=ERROR' \
		--project "$$GCP_PROJECT" --limit $(LOG_LIMIT) --freshness=$(LOG_FRESHNESS) \
		--format='table(timestamp, jsonPayload.msg, jsonPayload.agent, jsonPayload.trace_id, jsonPayload.error)'

logs-cloud-trace: ## Every line for one request: make logs-cloud-trace T=<trace_id>
	@test -n "$(T)" || { echo 'usage: make logs-cloud-trace T=<trace_id>'; exit 1; }
	@$(load_cloud_env); $(require_gcp); \
	gcloud logging read \
		'resource.type=cloud_run_revision AND jsonPayload.trace_id="$(T)"' \
		--project "$$GCP_PROJECT" --limit 200 --freshness=$(LOG_FRESHNESS) --order=asc \
		--format='table(timestamp, jsonPayload.level, jsonPayload.msg)'

deploy-check: ## Probe the deployed backend's readiness endpoint
	@url=$$($(MAKE) -s deploy-url); echo "$$url"; \
		curl -fsS "$$url/v1/health/dependencies" | python3 -m json.tool

lock: ## Re-resolve backend dependencies into uv.lock
	uv lock --directory $(BACKEND)

upgrade: ## Upgrade backend dependencies within their declared bounds
	uv lock --directory $(BACKEND) --upgrade

# --- On-prem model provisioning -------------------------------------------
# Resolves the models from Settings rather than duplicating them here, so the
# on-prem defaults live in exactly one place (backend/app/config.py). Uses the
# ollama Python client, already a backend dependency, so this works against any
# host without the ollama CLI installed.
define PULL_MODELS
import sys
from ollama import Client
from app.config import Settings

settings = Settings(deployment_mode="onprem")
client = Client(host=settings.ollama_base_url, timeout=600)
print(f"Ollama host: {settings.ollama_base_url}")
for model in (settings.llm_model, settings.router_model):
    print(f"  pulling {model} ...", flush=True)
    try:
        client.pull(model)
    except Exception as exc:
        sys.exit(f"  failed: {type(exc).__name__}: {exc}")
print("Done. Check with: make ready")
endef
export PULL_MODELS

# --- Ingestion ------------------------------------------------------------
define INGEST
import json
from app.ingestion.pipeline import ingest_directory
print(json.dumps(ingest_directory().as_dict(), indent=2))
endef
export INGEST

# Vector dimensions differ between embedding models, so an index built with one
# cannot be reused by another; this is the documented recovery.
define REINDEX
import json
from app.config import get_settings
from app.ingestion.pipeline import ingest_directory
from app.retrieval import vectorstore
settings = get_settings()
print(f"dropping collection {settings.collection_name}")
vectorstore.drop_collection(settings)
print(json.dumps(ingest_directory(settings=settings).as_dict(), indent=2))
endef
export REINDEX

# --- Inspection -----------------------------------------------------------
# Exercise retrieval and the structured store directly, without going through
# `/v1/chat` -- useful for checking what an agent was given, not just what it said.
define QUERY
import os, sys
from app.config import get_settings
from app.retrieval.hybrid import retrieve
question = os.environ.get("Q") or sys.exit('usage: make query Q="your question"')
settings = get_settings()
print(f'mode={settings.retrieval_mode} rerank={settings.rerank_enabled} top_k={settings.top_k}\n')
for hit in retrieve(question, settings):
    print(f"{hit.score:7.3f}  {hit.source}")
    print(f"         {hit.citation}")
    for line in hit.text.splitlines()[:4]:
        print(f"         | {line[:100]}")
    print()
endef
export QUERY

define SQL
import os, sys
from app.ingestion.tabular_store import list_tables, read_only_connection
statement = os.environ.get("S")
if statement:
    with read_only_connection() as connection:
        connection.sql(statement).show()
else:
    tables = list_tables()
    if not tables:
        sys.exit("No tables yet - run `make ingest` first.")
    for name, columns in tables.items():
        print(f"{name}: {', '.join(columns)}")
    print('\nusage: make sql S="select ..."')
endef
export SQL

# Resolves settings the same way the deployed backend will, against the managed
# services, and probes each one. This is `make ready` for a stack that is not
# running locally: it catches a Qdrant URL that needs its port, and a Redis
# without RediSearch -- which otherwise degrades quietly to the in-process
# saver and is invisible until someone reads a log (D-017).
define READY_CLOUD
import json, os
# The deployed backend never sees `.env`: Cloud Run passes DEPLOYMENT_MODE,
# the service URLs and the secrets, and every other knob falls to its declared
# default. So blank what `.env` would otherwise leak in -- left alone, a local
# on-prem `.env` makes this report `ollama` and a model the deployment will
# never load, which is the opposite of what the target is for. The API key is
# deliberately kept: it is the one value the check genuinely needs.
for name in ("DEPLOYMENT_MODE", "LLM_PROVIDER", "LLM_MODEL", "ROUTER_MODEL",
             "LLM_TIMEOUT_S", "OLLAMA_BASE_URL"):
    os.environ[name] = ""
from app.config import Settings
from app.api.routes import _check_checkpointer, _check_vector_store
from app.llm.providers import any_model_present, check_llm_ready
settings = Settings(deployment_mode="cloud")
llm = check_llm_ready(settings)
report = {
    "resolved": settings.resolved_summary(),
    "checks": {
        "llm": {"reachable": llm.get("reachable"),
                "models_present": llm.get("models_present"),
                "usable": any_model_present(llm),
                "error": llm.get("error")},
        "vector_store": _check_vector_store(settings),
        "checkpointer": _check_checkpointer(settings),
    },
}
ok = (report["checks"]["llm"].get("reachable") and report["checks"]["llm"]["usable"]
      and report["checks"]["vector_store"].get("ok")
      and report["checks"]["checkpointer"].get("ok"))
report = {"status": "ok" if ok else "degraded", **report}
print(json.dumps(report, indent=2, default=str))
raise SystemExit(0 if ok else 1)
endef
export READY_CLOUD
