"""
Parlay — main application entry point.
Starts FastAPI with Dashboard + OpenEnv WebSocket + static file serving.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from parlay_env.server import router as env_router
from dashboard.api import router as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and resources on startup."""
    logger.info("Parlay starting up...")
    try:
        from scripts.init_db import init_db
        await init_db()
        logger.info("Database initialized")
    except Exception as exc:
        logger.warning(f"DB init failed (continuing): {exc}")
    yield
    logger.info("Parlay shutting down.")


app = FastAPI(
    title="Parlay",
    description="OpenEnv-compliant RL negotiation environment. Train agents, play scenarios.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permissive for dev; restrict origins in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(env_router)
app.include_router(dashboard_router)

# Serve static files
static_dir = Path("dashboard/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the main game dashboard."""
    return FileResponse("dashboard/index.html")


@app.get("/train", include_in_schema=False)
async def serve_train() -> FileResponse:
    """Serve the training dashboard."""
    return FileResponse("dashboard/train.html")


@app.get("/health")
async def health() -> dict:
    """Global health check."""
    return {"status": "ok", "service": "parlay", "version": "1.0.0"}
