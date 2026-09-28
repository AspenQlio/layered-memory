from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, status
from sqlalchemy.orm import Session

from layered_memory import __version__
from layered_memory.api import schemas
from layered_memory.api.state import AppState, get_ctx, get_session
from layered_memory.memory.record import MemoryRecord
from layered_memory.retrieval.index import build_context_block


def register_memory_routes(app: FastAPI) -> None:
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


def _to_response(record: MemoryRecord) -> schemas.MemoryResponse:
    return schemas.MemoryResponse(**record.model_dump())
