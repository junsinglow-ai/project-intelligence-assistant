# Project Intelligence Assistant
#
# Docker is the primary way to run the stack; the local targets use uv
# (backend) and npm (frontend) for day-to-day development.

COMPOSE := docker compose
BACKEND := backend
FRONTEND := frontend

.DEFAULT_GOAL := help
.PHONY: help env up down restart build rebuild logs logs-backend logs-frontend ps \
        shell-backend shell-frontend clean install dev-backend dev-frontend test \
        data eval lock upgrade

help: ## Show available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Docker ---------------------------------------------------------------

.env:
	@cp .env.example .env
	@echo "Created .env from .env.example - fill in the LLM / embedding settings."

env: .env ## Create .env from .env.example if missing

up: env ## Build if needed and start the stack in the background
	$(COMPOSE) up -d --build
	@echo "Backend:  http://localhost:8000  (API docs at /docs)"
	@echo "Frontend: http://localhost:5173"

down: ## Stop the stack
	$(COMPOSE) down

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

shell-backend: ## Open a shell in the backend container
	$(COMPOSE) exec backend bash

shell-frontend: ## Open a shell in the frontend container
	$(COMPOSE) exec frontend sh

clean: ## Stop the stack and remove its volumes and locally built images
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

# Scripts live at the repo root but run against the backend environment.
data: ## Regenerate the synthetic sample documents in data/raw
	uv run --project $(BACKEND) python data/scripts/generate_synthetic_data.py

eval: ## Run the RAGAS evaluation into eval/results
	uv run --project $(BACKEND) python eval/run_ragas.py

lock: ## Re-resolve backend dependencies into uv.lock
	uv lock --directory $(BACKEND)

upgrade: ## Upgrade backend dependencies within their declared bounds
	uv lock --directory $(BACKEND) --upgrade
