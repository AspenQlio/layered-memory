from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from layered_memory.agents.handoff import HandoffQueue
from layered_memory.config import Settings
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.index import SearchIndex


class AppState:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        factory: sessionmaker[Session],
        memory: MemoryService,
        index: SearchIndex,
        handoffs: HandoffQueue,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.factory = factory
        self.memory = memory
        self.index = index
        self.handoffs = handoffs


def get_ctx(request: Request) -> AppState:
    ctx: AppState | None = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=503, detail="la aplicacion no esta inicializada")
    return ctx


def get_session(ctx: AppState = Depends(get_ctx)) -> Iterator[Session]:
    session = ctx.factory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()
