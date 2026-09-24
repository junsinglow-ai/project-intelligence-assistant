FROM python:3.12-slim

# uv ships as a static binary; copy it in rather than pip-installing it.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# Both ONNX caches are baked into the image (see the warm-up step below), so
# they must be set before anything loads a model. Blank in .env.example, which
# leaves each library's own default for local development.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    EMBEDDING_CACHE_DIR=/app/models/fastembed \
    RERANK_CACHE_DIR=/app/models/flashrank \
    DATA_RAW_DIR=/app/data/raw \
    UPLOAD_DIR=/app/data/uploads \
    TABULAR_DB_PATH=/app/data/processed/tables.duckdb \
    QDRANT_PATH=/app/data/qdrant \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so application edits don't invalidate the install layer.
# Paths are repo-root relative: the build context is the repository root
# (see docker-compose.yml) so that ./data can be baked in below.
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Download the embedding and re-ranking weights (~150MB) into the image rather
# than leaving them to the first request. On a scale-to-zero host the container
# filesystem is discarded when the instance goes away, so an unbaked cache is
# re-downloaded on every cold start, and the download lands inside a user's
# request. Kept as its own layer: it changes only when the models do.
# These must match the EMBEDDING_MODEL / RERANK_MODEL defaults in
# app/config.py, or the image bakes weights the app will not ask for and the
# download reappears at runtime. tests/test_docker_models.py asserts they do.
# They are build args rather than a COPY of the config because this layer sits
# before `COPY app` on purpose: an edit to the application must not invalidate
# a 150MB download.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=ms-marco-MiniLM-L-12-v2
RUN python -c "\
import os; \
from fastembed import TextEmbedding; \
from flashrank import Ranker; \
TextEmbedding(model_name='${EMBEDDING_MODEL}', cache_dir=os.environ['EMBEDDING_CACHE_DIR']); \
Ranker(model_name='${RERANK_MODEL}', cache_dir=os.environ['RERANK_CACHE_DIR'])"

COPY backend/app ./app
# The curated corpus and the tables the Data Analysis agent queries. Compose
# bind-mounts ./data over this; a cloud deployment has no volume, so the image
# carries its own copy. Built by `make ingest` before the image is built.
COPY data ./data

# The data paths above must be absolute. `app/config.py` anchors a relative
# path to the repository root, which it computes as `parents[2]` of its own
# file -- `/` in this image, since app/ sits at /app/app. So an unset
# DATA_RAW_DIR resolves to /data/raw, outside WORKDIR, and the readiness probe
# reports a permission error rather than a missing directory. Compose passes
# the same values; setting them here means a host that forgets to is still
# correct. `uploads` is created because .dockerignore keeps runtime state out.
RUN mkdir -p /app/data/uploads \
    && useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Shell form so $PORT expands: hosts that assign a port (Cloud Run, Render)
# inject it, and the default keeps Compose and local runs on 8000.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
