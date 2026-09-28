from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine

from layered_memory import __version__
from layered_memory.agents.handoff import HandoffQueue
from layered_memory.api.errors import register_error_handlers
from layered_memory.api.handoff_routes import register_handoff_routes
from layered_memory.api.memory_routes import register_memory_routes
from layered_memory.api.state import AppState
from layered_memory.config import Settings, get_settings
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import build_embedder
from layered_memory.retrieval.index import build_index
from layered_memory.store.db import build_engine, create_session_factory, init_db


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    embed_artifacts: bool = False,
) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_engine = engine or build_engine(resolved.database_url)
        init_db(active_engine)
        factory = create_session_factory(active_engine)
        embedder = build_embedder(resolved)
        app.state.ctx = AppState(
            settings=resolved,
            engine=active_engine,
            factory=factory,
            memory=MemoryService(embedder, embed_artifacts=embed_artifacts),
            index=build_index(resolved, embedder),
            handoffs=HandoffQueue(),
        )
        try:
            yield
        finally:
            app.state.ctx = None
            embedder.close()
            if engine is None:
                active_engine.dispose()

    app = FastAPI(
        title="layered-memory",
        version=__version__,
        summary="Memoria por capas para agentes de IA: captura, destilacion, retrieval y handoff.",
        lifespan=lifespan,
    )
    register_memory_routes(app)
    register_handoff_routes(app)
    register_error_handlers(app)
    return app


app = create_app()
