"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from .api import auth_api, chat, documents, machines, models_api, usage, workflows_api
from .auth import BasicAuthMiddleware
from .config import SettingsError, get_settings
from .db import create_schema, session_scope
from .llm.instrumentation import install_usage_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def seed_workflows() -> None:
    """Insert any registered workflow that has no row yet. Never updates existing rows."""
    from .retrieval.registry import WORKFLOWS as SPECS

    with session_scope() as session:
        existing = {
            row["class_key"]
            for row in session.execute(text("SELECT class_key FROM workflows")).mappings().all()
        }

        for name, spec in SPECS.items():
            if spec.class_key in existing:
                continue
            session.execute(
                text(
                    """
                    INSERT INTO workflows (name, description, class_key, enabled)
                    VALUES (:name, :description, :class_key, true)
                    ON CONFLICT (name) DO NOTHING
                    """
                ),
                {
                    "name": name,
                    "description": spec.factory.description,
                    "class_key": spec.class_key,
                },
            )
            logger.info("Seeded workflow '%s' (%s)", name, spec.class_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        settings.validate_models()
    except SettingsError as exc:
        logger.error("Configuration rejected: %s", exc)
        raise

    install_usage_handler()

    try:
        create_schema()
    except Exception as exc:
        # A database problem is the most common startup failure; say so plainly.
        logger.error(
            "Cannot initialise the database (%s). Check DATABASE_URL and that "
            "PostgreSQL with pgvector is running: %s",
            type(exc).__name__,
            exc,
        )
        raise

    seed_workflows()
    logger.info("MyRag ready. default embed model %s (dim=%d)",
                settings.default_embed_model, settings.default_embed_dim)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MyRag — Flour Factory Machine RAG Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(BasicAuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_api.router)
    app.include_router(chat.log_router)
    app.include_router(machines.router)
    app.include_router(documents.router)
    app.include_router(workflows_api.router)
    app.include_router(models_api.router)
    app.include_router(chat.router)
    app.include_router(usage.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # Serve the built SPA when it exists (npm run build in web/). Mounted last so
    # every /api route above wins.
    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


app = create_app()