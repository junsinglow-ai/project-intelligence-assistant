"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import get_settings
from app.observability.logging import configure_logging
from app.observability.middleware import RequestContextMiddleware

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)
# Mode resolution fills blank knobs from DEPLOYMENT_MODE, so log what it
# actually resolved to -- otherwise the active model is invisible.
logger.info("configuration resolved", extra={"fields": settings.resolved_summary()})
for problem in settings.missing_requirements():
    logger.warning("configuration incomplete", extra={"fields": {"problem": problem}})

app = FastAPI(
    title="Project Intelligence Assistant",
    version="1.0.0",
    description="Multi-agent RAG over project status reports, financials and risk registers.",
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/v1")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
