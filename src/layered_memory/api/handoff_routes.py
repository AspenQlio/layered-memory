from __future__ import annotations

from fastapi import Depends, FastAPI, status
from sqlalchemy.orm import Session

from layered_memory.agents.record import HandoffRecord
from layered_memory.api import schemas
from layered_memory.api.state import AppState, get_ctx, get_session
from layered_memory.domain import HandoffStatus


def register_handoff_routes(app: FastAPI) -> None:
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


def _handoff_response(record: HandoffRecord) -> schemas.HandoffResponse:
    return schemas.HandoffResponse(**record.as_dict())
