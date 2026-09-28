from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from layered_memory.agents.handoff import HandoffConflictError, HandoffNotFoundError
from layered_memory.memory.service import MemoryNotFoundError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MemoryNotFoundError)
    @app.exception_handler(HandoffNotFoundError)
    def not_found(
        _request: Request,
        exc: MemoryNotFoundError | HandoffNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(HandoffConflictError)
    def conflict(_request: Request, exc: HandoffConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def bad_request(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(SQLAlchemyError)
    def database_error(_request: Request, _exc: SQLAlchemyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "base de datos no disponible"})
