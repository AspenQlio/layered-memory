"""Aplicacion FastAPI.

El estado (motor, sesion, servicio, indice, cola) se construye una vez en el
lifespan y se expone en ``app.state``; las dependencias solo lo leen. Asi los
tests pueden inyectar un motor en memoria con ``create_app(engine=...)`` sin
tocar variables de entorno.

Los errores de dominio se traducen aqui y no en el servicio: el servicio lanza
excepciones ownas y la capa HTTP decide codigos.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from layered_memory import __version__
from layered_memory.agents.handoff import HandoffConflictError, HandoffNotFoundError, HandoffQueue
from layered_memory.api import schemas
from layered_memory.config import Settings, get_settings
from layered_memory.domain import HandoffStatus
from layered_memory.memory.record import MemoryRecord
from layered_memory.memory.service import MemoryNotFoundError, MemoryService
from layered_memory.retrieval.embeddings import build_embedder
from layered_memory.retrieval.index import VectorIndex, build_context_block
from layered_memory.store.db import build_engine, create_session_factory, init_db


class AppState:
    """Contenedor explicito del estado compartido."""

    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        factory: sessionmaker[Session],
        memory: MemoryService,
        index: VectorIndex,
        handoffs: HandoffQueue,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.factory = factory
        self.memory = memory
        self.index = index
        self.handoffs = handoffs


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    embed_artifacts: bool = False,
) -> FastAPI:
    """Construye la app. ``engine`` es el punto de inyeccion para los tests."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        active_engine = engine or build_engine(resolved.database_url)
        init_db(active_engine)
        factory = create_session_factory(active_engine)
        embedder = build_embedder(resolved)

        app.state.ctx = AppState(
            settings=resolved,
            engine=active_engine,
            factory=factory,
            memory=MemoryService(embedder, embed_artifacts=embed_artifacts),
            index=VectorIndex(embedder),
            handoffs=HandoffQueue(),
        )
        try:
            yield
        finally:
            app.state.ctx = None

    app = FastAPI(
        title="layered-memory",
        version=__version__,
        summary="Memoria por capas para agentes de IA: captura, destilacion, retrieval y handoff.",
        lifespan=lifespan,
    )
    _register_routes(app)
    _register_error_handlers(app)
    return app


# --------------------------------------------------------------------- estado


def get_ctx(request: Request) -> AppState:
    ctx: AppState | None = getattr(request.app.state, "ctx", None)
    if ctx is None:  # pragma: no cover - solo si se usa la app sin lifespan
        raise HTTPException(status_code=503, detail="la aplicacion no esta inicializada")
    return ctx


def get_session(ctx: AppState = Depends(get_ctx)) -> Iterator[Session]:
    """Sesion transaccional por request."""
    session = ctx.factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# -------------------------------------------------------------------- rutas


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=schemas.HealthResponse, tags=["sistema"])
    def health(ctx: AppState = Depends(get_ctx)) -> schemas.HealthResponse:
        embedder = ctx.index.embedder
        return schemas.HealthResponse(
            status="ok",
            version=__version__,
            embedding_backend=ctx.settings.embedding_backend,
            embedding_model=embedder.model,
            embedding_dim=embedder.dim,
            database=ctx.settings.database_url.split("://", 1)[0],
        )

    # Las rutas literales se declaran antes que las parametrizadas: FastAPI
    # resuelve en orden de registro y /memories/stats seria "busca una memoria
    # llamada stats" si /memories/{memory_id} estuviera primero.

    @app.get("/memories/stats", response_model=schemas.StatsResponse, tags=["memoria"])
    def stats(
        namespace: str = "default",
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> dict[str, Any]:
        return ctx.memory.stats(session, namespace=namespace)

    @app.post("/memories/search", response_model=schemas.SearchResponse, tags=["retrieval"])
    def search(
        payload: schemas.SearchRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.SearchResponse:
        hits = ctx.index.search(
            session,
            payload.query,
            namespace=payload.namespace,
            layers=payload.layers,
            tags=payload.tags,
            k=payload.k,
            min_score=payload.min_score,
        )
        return schemas.SearchResponse(
            query=payload.query,
            count=len(hits),
            hits=[
                schemas.SearchHitResponse(
                    id=h.record.id,
                    score=h.score,
                    layer=h.record.layer,
                    title=h.record.title,
                    content=h.record.content,
                    tags=h.record.tags,
                    parent_id=h.record.parent_id,
                )
                for h in hits
            ],
        )

    @app.post("/memories/context", response_model=schemas.ContextResponse, tags=["retrieval"])
    def context(
        payload: schemas.ContextRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.ContextResponse:
        hits = ctx.index.search(
            session,
            payload.query,
            namespace=payload.namespace,
            layers=payload.layers,
            tags=payload.tags,
            k=payload.k,
            min_score=payload.min_score,
        )
        block = build_context_block(hits, max_chars=payload.max_chars)
        return schemas.ContextResponse(
            query=payload.query,
            block=block,
            ids=[h.id for h in hits],
            chars=len(block),
        )

    @app.post("/memories/reindex", tags=["memoria"])
    def reindex(
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> dict[str, int]:
        return {"reindexed": ctx.memory.reindex(session)}

    @app.post(
        "/memories/capture",
        response_model=schemas.MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["memoria"],
    )
    def capture(
        payload: schemas.CaptureRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.MemoryResponse:
        record = ctx.memory.capture(
            session,
            payload.content,
            title=payload.title,
            namespace=payload.namespace,
            source=payload.source,
            tags=payload.tags,
            extra=payload.extra,
            embed=payload.embed,
        )
        return _to_response(record)

    @app.post(
        "/memories/{memory_id}/distill",
        response_model=schemas.MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["memoria"],
    )
    def distill(
        memory_id: str,
        payload: schemas.DistillRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.MemoryResponse:
        record = ctx.memory.distill(
            session,
            memory_id,
            title=payload.title,
            content=payload.content,
            tags=payload.tags,
            source=payload.source,
            extra=payload.extra,
            embed=payload.embed,
        )
        return _to_response(record)

    @app.post(
        "/memories/{memory_id}/produce",
        response_model=schemas.MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["memoria"],
    )
    def produce(
        memory_id: str,
        payload: schemas.ProduceRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.MemoryResponse:
        record = ctx.memory.produce(
            session,
            memory_id,
            title=payload.title,
            content=payload.content,
            tags=payload.tags,
            source=payload.source,
            extra=payload.extra,
        )
        return _to_response(record)

    @app.get("/memories/{memory_id}", response_model=schemas.MemoryResponse, tags=["memoria"])
    def get_memory(
        memory_id: str,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.MemoryResponse:
        return _to_response(ctx.memory.get(session, memory_id))

    @app.get(
        "/memories/{memory_id}/children",
        response_model=list[schemas.MemoryResponse],
        tags=["memoria"],
    )
    def children(
        memory_id: str,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> list[schemas.MemoryResponse]:
        return [_to_response(r) for r in ctx.memory.children(session, memory_id)]

    @app.get(
        "/memories/{memory_id}/lineage",
        response_model=list[schemas.MemoryResponse],
        tags=["memoria"],
    )
    def lineage(
        memory_id: str,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> list[schemas.MemoryResponse]:
        """Cadena completa desde la captura original hasta el artefacto actual."""
        chain: list[MemoryRecord] = []
        current = ctx.memory.get(session, memory_id)
        seen: set[str] = set()
        while current.parent_id is not None and current.id not in seen:
            seen.add(current.id)
            chain.append(current)
            current = ctx.memory.get(session, current.parent_id)
        chain.append(current)
        return [_to_response(r) for r in reversed(chain)]

    @app.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["memoria"])
    def delete_memory(
        memory_id: str,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> None:
        ctx.memory.delete(session, memory_id)

    @app.post(
        "/handoffs",
        response_model=schemas.HandoffResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["handoff"],
    )
    def enqueue(
        payload: schemas.HandoffCreateRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.HandoffResponse:
        record = ctx.handoffs.enqueue(
            session,
            payload.reason,
            namespace=payload.namespace,
            session_id=payload.session_id,
            context=payload.context,
        )
        return _handoff_response(record)

    @app.get("/handoffs", response_model=list[schemas.HandoffResponse], tags=["handoff"])
    def list_handoffs(
        namespace: str = "default",
        status_filter: HandoffStatus | None = None,
        limit: int = 50,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> list[schemas.HandoffResponse]:
        records = ctx.handoffs.list(session, namespace=namespace, status=status_filter, limit=limit)
        return [_handoff_response(r) for r in records]

    @app.get("/handoffs/depth", response_model=dict[str, int], tags=["handoff"])
    def handoff_depth(
        namespace: str = "default",
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> dict[str, int]:
        return {"queued": ctx.handoffs.depth(session, namespace=namespace)}

    @app.post(
        "/handoffs/{handoff_id}/claim", response_model=schemas.HandoffResponse, tags=["handoff"]
    )
    def claim(
        handoff_id: str,
        payload: schemas.ClaimRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.HandoffResponse:
        return _handoff_response(
            ctx.handoffs.claim(session, handoff_id, claimed_by=payload.claimed_by)
        )

    @app.post(
        "/handoffs/{handoff_id}/resolve", response_model=schemas.HandoffResponse, tags=["handoff"]
    )
    def resolve(
        handoff_id: str,
        payload: schemas.ResolveRequest,
        session: Session = Depends(get_session),
        ctx: AppState = Depends(get_ctx),
    ) -> schemas.HandoffResponse:
        record = ctx.handoffs.resolve(
            session, handoff_id, payload.resolution, resolved_by=payload.resolved_by
        )
        return _handoff_response(record)


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MemoryNotFoundError)
    @app.exception_handler(HandoffNotFoundError)
    def _not_found(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(HandoffConflictError)
    def _conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def _bad_request(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(SQLAlchemyError)
    def _db_error(_request: Request, exc: Exception) -> JSONResponse:
        # No filtramos el SQL al cliente: la causa se ve en los logs del servidor.
        return JSONResponse(status_code=503, content={"detail": "base de datos no disponible"})


def _to_response(record: MemoryRecord) -> schemas.MemoryResponse:
    return schemas.MemoryResponse(**record.model_dump())


def _handoff_response(record: Any) -> schemas.HandoffResponse:
    return schemas.HandoffResponse(**record.as_dict())


app = create_app()
