"""Forma de dominio de una memoria, independiente del ORM.

El servicio devuelve estos objetos y no filas de SQLAlchemy: asi la API, la
CLI y el evaluador no dependen del motor de persistencia y no pueden mutar por
accidente algo que ya se leyo de la base.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from layered_memory.domain import Layer, RawStatus


def as_utc(value: datetime | None) -> datetime | None:
    """Normaliza a UTC. SQLite devuelve datetimes naive, que se Assume UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class MemoryRecord(BaseModel):
    """Una memoria ya persistida."""

    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str
    layer: Layer
    title: str
    content: str
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    status: RawStatus | None = None
    parent_id: str | None = None
    embedded: bool = False
    embedding_model: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Any) -> Self:
        """Construye el record desde una fila de :class:`Memory`."""
        return cls(
            id=row.id,
            namespace=row.namespace,
            layer=Layer(row.layer),
            title=row.title,
            content=row.content,
            source=row.source,
            tags=list(row.tags or []),
            extra=dict(row.extra or {}),
            status=RawStatus(row.status) if row.status else None,
            parent_id=row.parent_id,
            embedded=row.embedding is not None,
            embedding_model=row.embedded_model,
            created_at=as_utc(row.created_at),
            updated_at=as_utc(row.updated_at),
        )
