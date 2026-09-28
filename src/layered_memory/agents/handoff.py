"""Cola de escalado a humano (human-in-the-loop).

El agente no resuelve todo: cuando la confianza es baja o la peticion exige
criterio, encola un handoff y sigue. Un humano (u otro agente con permisos)
reclama, responde y la memoria se cierra con esa respuesta.

Lo que hace confiable a una cola asi no es el encolado, es que ``claim`` sea
atomico: dos consumidores compitiendo por la misma peticion no deben poder
ganar los dos. Aqui se resuelve con un ``UPDATE ... WHERE status='queued'`` y
se mira el ``rowcount``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from layered_memory.agents.errors import (
    EmptyHandoffFieldError,
    HandoffConflictError,
    HandoffNotFoundError,
)
from layered_memory.agents.record import HandoffRecord
from layered_memory.domain import HandoffStatus
from layered_memory.store.models import Event, Handoff, new_id


class HandoffQueue:
    """Cola FIFO de peticiones de escalado."""

    def __init__(self, *, clock: Any = None) -> None:
        self._clock = clock

    def _now(self) -> datetime:
        if self._clock is not None:
            return self._clock()
        from layered_memory.store.models import utcnow

        return utcnow()

    def enqueue(
        self,
        session: Session,
        reason: str,
        *,
        namespace: str = "default",
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> HandoffRecord:
        text = reason.strip()
        if not text:
            raise EmptyHandoffFieldError("el motivo del escalado")

        row = Handoff(
            id=new_id(),
            namespace=namespace,
            session_id=session_id,
            status=HandoffStatus.QUEUED.value,
            reason=text,
            context=dict(context or {}),
        )
        session.add(row)
        session.flush()
        _log(session, namespace, "handoff.enqueued", row.id, {"session_id": session_id})
        return HandoffRecord(row)

    def claim(
        self, session: Session, handoff_id: str, *, claimed_by: str = "worker"
    ) -> HandoffRecord:
        """Toma una peticion encolada. Atomico: el que actualiza, gana."""
        now = self._now()
        result = session.execute(
            update(Handoff)
            .where(Handoff.id == handoff_id, Handoff.status == HandoffStatus.QUEUED.value)
            .values(
                status=HandoffStatus.CLAIMED.value,
                claimed_by=claimed_by,
                claimed_at=now,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            current = self.get(session, handoff_id)
            raise HandoffConflictError(
                f"el handoff {handoff_id} ya esta en estado {current.status}"
            )
        session.flush()
        _log(session, None, "handoff.claimed", handoff_id, {"by": claimed_by})
        return self.get(session, handoff_id)

    def claim_next(
        self, session: Session, *, namespace: str = "default", claimed_by: str = "worker"
    ) -> HandoffRecord | None:
        """Reclama la peticion mas antigua. Devuelve ``None`` si la cola esta vacia."""
        stmt = (
            select(Handoff.id)
            .where(Handoff.namespace == namespace, Handoff.status == HandoffStatus.QUEUED.value)
            .order_by(Handoff.created_at.asc())
            .limit(1)
        )
        next_id = session.execute(stmt).scalar_one_or_none()
        if next_id is None:
            return None
        return self.claim(session, next_id, claimed_by=claimed_by)

    def resolve(
        self,
        session: Session,
        handoff_id: str,
        resolution: str,
        *,
        resolved_by: str = "human",
    ) -> HandoffRecord:
        """Cierra una peticion encolada o ya tomada, guardando la respuesta."""
        text = resolution.strip()
        if not text:
            raise EmptyHandoffFieldError("la resolucion")

        now = self._now()
        result = session.execute(
            update(Handoff)
            .where(
                Handoff.id == handoff_id,
                Handoff.status.in_([HandoffStatus.QUEUED.value, HandoffStatus.CLAIMED.value]),
            )
            .values(
                status=HandoffStatus.RESOLVED.value,
                resolution=text,
                resolved_by=resolved_by,
                resolved_at=now,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            current = self.get(session, handoff_id)
            raise HandoffConflictError(
                f"el handoff {handoff_id} ya esta en estado {current.status}"
            )
        session.flush()
        _log(session, None, "handoff.resolved", handoff_id, {"by": resolved_by})
        return self.get(session, handoff_id)

    def cancel(
        self, session: Session, handoff_id: str, *, resolved_by: str = "system"
    ) -> HandoffRecord:
        """Cancela una peticion que todavia no se resolvio."""
        now = self._now()
        result = session.execute(
            update(Handoff)
            .where(
                Handoff.id == handoff_id,
                Handoff.status.in_([HandoffStatus.QUEUED.value, HandoffStatus.CLAIMED.value]),
            )
            .values(
                status=HandoffStatus.CANCELLED.value,
                resolved_by=resolved_by,
                resolved_at=now,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            current = self.get(session, handoff_id)
            raise HandoffConflictError(
                f"el handoff {handoff_id} ya esta en estado {current.status}"
            )
        session.flush()
        return self.get(session, handoff_id)

    def get(self, session: Session, handoff_id: str) -> HandoffRecord:
        row = session.get(Handoff, handoff_id)
        if row is None:
            raise HandoffNotFoundError(f"no existe el handoff {handoff_id}")
        return HandoffRecord(row)

    def list(
        self,
        session: Session,
        *,
        namespace: str = "default",
        status: HandoffStatus | None = None,
        limit: int = 50,
    ) -> list[HandoffRecord]:
        stmt = select(Handoff).where(Handoff.namespace == namespace)
        if status is not None:
            stmt = stmt.where(Handoff.status == status.value)
        stmt = stmt.order_by(Handoff.created_at.desc()).limit(limit)
        return [HandoffRecord(row) for row in session.execute(stmt).scalars()]

    def depth(self, session: Session, *, namespace: str = "default") -> int:
        """Cuantas peticiones siguen esperando: la metrica de saturacion del equipo humano."""
        rows = session.execute(
            select(Handoff.id).where(
                Handoff.namespace == namespace,
                Handoff.status == HandoffStatus.QUEUED.value,
            )
        ).all()
        return len(rows)


def resolutions(session: Session, *, namespace: str = "default") -> list[tuple[str, str]]:
    """Pares ``(reason, resolution)`` ya resueltos: semilla del lazo de aprendizaje."""
    stmt = (
        select(Handoff.reason, Handoff.resolution)
        .where(
            Handoff.namespace == namespace,
            Handoff.status == HandoffStatus.RESOLVED.value,
            Handoff.resolution.is_not(None),
        )
        .order_by(Handoff.resolved_at.asc())
    )
    return [(row.reason, row.resolution or "") for row in session.execute(stmt)]


def _log(
    session: Session,
    namespace: str | None,
    kind: str,
    subject_id: str | None,
    payload: dict[str, Any],
) -> None:
    session.add(
        Event(namespace=namespace or "default", kind=kind, subject_id=subject_id, payload=payload)
    )


__all__: Sequence[str] = [
    "HandoffConflictError",
    "HandoffNotFoundError",
    "HandoffQueue",
    "HandoffRecord",
    "resolutions",
]
