"""Tablas del esquema.

El embedding se guarda como ``LargeBinary`` (float32 little-endian) en vez de
``ARRAY(Float)`` para que el mismo esquema funcione en SQLite y PostgreSQL sin
dialectos. La busqueda vectorial se resuelve en numpy; el camino de escalado a
indice nativo esta documentado en ``docs/architecture.md``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from layered_memory.domain import HandoffStatus, Layer, RawStatus


def utcnow() -> datetime:
    """Instante actual con timezone UTC. SQLite lo devuelve naive; ver ``as_utc``."""
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Memory(Base):
    """Una unidad de memoria en una capa concreta.

    ``raw`` y ``insight`` son buscables por defecto. ``artifact`` se indexa solo
    si el embedder esta disponible, porque son piezas largas y caras de
    embeber; el servicio decide cuando vale la pena.
    """

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    layer: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    source: Mapped[str | None] = mapped_column(String(300), default=None)
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    status: Mapped[str | None] = mapped_column(String(16), default=None)
    parent_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("memories.id", ondelete="SET NULL"), default=None
    )

    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, default=None)
    embedded_model: Mapped[str | None] = mapped_column(String(120), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        Index("ix_memories_namespace_layer", "namespace", "layer"),
        Index("ix_memories_status", "status"),
        Index("ix_memories_parent", "parent_id"),
    )


class Handoff(Base):
    """Peticion de escalado a un humano, con cola y trazabilidad de estados."""

    __tablename__ = "handoffs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    session_id: Mapped[str | None] = mapped_column(String(64), default=None)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=HandoffStatus.QUEUED.value
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    resolution: Mapped[str | None] = mapped_column(Text, default=None)
    claimed_by: Mapped[str | None] = mapped_column(String(120), default=None)
    resolved_by: Mapped[str | None] = mapped_column(String(120), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    __table_args__ = (
        Index("ix_handoffs_queue", "namespace", "status", "created_at"),
        Index("ix_handoffs_session", "session_id"),
    )


class Event(Base):
    """Bitacora append-only de lo que le paso a una memoria.

    Sirve para dos cosas: depurar por que un documento no aparece en una
    busqueda, y alimentar el lazo de aprendizaje (que capturas se destilaron
    mas, que handoffs se resolvieron, que consultas quedaron sin cobertura).
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    subject_id: Mapped[str | None] = mapped_column(String(32), default=None)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_events_kind", "kind", "created_at"),)


#: Capas que se embeben de forma automatica al escribirse.
EMBEDDABLE_LAYERS: frozenset[Layer] = frozenset({Layer.RAW, Layer.INSIGHT})

#: Estados validos de la columna ``memories.status``.
VALID_RAW_STATUSES: frozenset[RawStatus] = frozenset(RawStatus)
