"""Servicio de memoria: captura, destilacion y promocion entre capas.

Reglas que sostiene:

* Solo ``raw`` acepta contenido sin destilar. Todo lo demas nace de promover
  una fila existente, de modo que la ascendencia es siempre navegable.
* Promover marca el origen. Una captura queda ``distilled`` y no vuelve a la
  cola de pendientes.
* Ninguna promocion reescribe contenido. Si el destilado se equivoca, se
  escribe otro distilado hijo; el original sigue intacto.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from layered_memory.domain import Layer, RawStatus
from layered_memory.memory.errors import EmptyMemoryContentError, MemoryNotFoundError
from layered_memory.memory.helpers import (
    clean_tags as _clean_tags,
)
from layered_memory.memory.helpers import (
    derive_title as _derive_title,
)
from layered_memory.memory.helpers import (
    indexable_text as _indexable_text,
)
from layered_memory.memory.helpers import (
    iter_layers,
)
from layered_memory.memory.helpers import (
    log_event as _log,
)
from layered_memory.memory.helpers import (
    store_vector as _store_vector,
)
from layered_memory.memory.record import MemoryRecord
from layered_memory.retrieval.embeddings import Embedder
from layered_memory.store.models import EMBEDDABLE_LAYERS, Memory, new_id


class MemoryService:
    """CRUD y transiciones de capa sobre la tabla ``memories``."""

    def __init__(self, embedder: Embedder, *, embed_artifacts: bool = False) -> None:
        self.embedder = embedder
        self.embed_artifacts = embed_artifacts

    # ---------------------------------------------------------------- lectura

    def get(self, session: Session, memory_id: str, *, namespace: str = "default") -> MemoryRecord:
        record = self._row(session, memory_id, namespace)
        return MemoryRecord.from_row(record)

    def children(
        self, session: Session, memory_id: str, *, namespace: str = "default"
    ) -> list[MemoryRecord]:
        """Promociones directas de una memoria, de la capa mas baja a la mas alta."""
        self._row(session, memory_id, namespace)
        stmt = (
            select(Memory)
            .where(Memory.parent_id == memory_id, Memory.namespace == namespace)
            .order_by(Memory.created_at.asc())
        )
        return [MemoryRecord.from_row(row) for row in session.execute(stmt).scalars()]

    def stats(self, session: Session, *, namespace: str = "default") -> dict[str, Any]:
        stmt = (
            select(Memory.layer, func.count())
            .where(Memory.namespace == namespace)
            .group_by(Memory.layer)
        )
        by_layer = {str(layer): count for layer, count in session.execute(stmt)}

        pending = session.scalar(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.namespace == namespace,
                Memory.layer == Layer.RAW.value,
                Memory.status == RawStatus.PENDING.value,
            )
        )
        indexed = session.scalar(
            select(func.count())
            .select_from(Memory)
            .where(Memory.namespace == namespace, Memory.embedding.is_not(None))
        )
        total = sum(by_layer.values())

        return {
            "namespace": namespace,
            "total": total,
            "by_layer": {str(layer): by_layer.get(str(layer), 0) for layer in Layer},
            "pending_distillation": int(pending or 0),
            "indexed": int(indexed or 0),
            "coverage": round(indexed / total, 4) if total else 0.0,
        }

    def pending_raw(
        self, session: Session, *, namespace: str = "default", limit: int = 20
    ) -> list[MemoryRecord]:
        """Capturas que aun no han sido destiladas: la cola de trabajo del agente."""
        stmt = (
            select(Memory)
            .where(
                Memory.namespace == namespace,
                Memory.layer == Layer.RAW.value,
                Memory.status == RawStatus.PENDING.value,
            )
            .order_by(Memory.created_at.asc())
            .limit(limit)
        )
        return [MemoryRecord.from_row(row) for row in session.execute(stmt).scalars()]

    # -------------------------------------------------------------- escritura

    def capture(
        self,
        session: Session,
        content: str,
        *,
        title: str | None = None,
        namespace: str = "default",
        source: str | None = None,
        tags: Sequence[str] = (),
        extra: dict[str, Any] | None = None,
        embed: bool = True,
    ) -> MemoryRecord:
        """Registra contenido literal en la capa ``raw``."""
        cleaned = content.strip()
        if not cleaned:
            raise EmptyMemoryContentError("capturar")

        row = Memory(
            id=new_id(),
            namespace=namespace,
            layer=Layer.RAW.value,
            title=title or _derive_title(cleaned),
            content=cleaned,
            source=source,
            tags=_clean_tags(tags),
            extra=dict(extra or {}),
            status=RawStatus.PENDING.value,
        )
        if embed:
            self._embed_row(row)
        session.add(row)
        session.flush()
        _log(session, namespace, "capture", row.id, {"title": row.title, "tags": row.tags})
        return MemoryRecord.from_row(row)

    def distill(
        self,
        session: Session,
        raw_id: str,
        *,
        title: str,
        content: str,
        tags: Sequence[str] = (),
        source: str | None = None,
        extra: dict[str, Any] | None = None,
        embed: bool = True,
    ) -> MemoryRecord:
        """Crea un ``insight`` hijo de una captura y marca la captura como destilada."""
        parent = self._row(session, raw_id)
        cleaned = content.strip()
        if not cleaned:
            raise EmptyMemoryContentError("destilar")

        row = Memory(
            id=new_id(),
            namespace=parent.namespace,
            layer=Layer.INSIGHT.value,
            title=title.strip() or parent.title,
            content=cleaned,
            source=source or parent.source,
            tags=_clean_tags(tags) or list(parent.tags or []),
            extra=dict(extra or {}),
            status=None,
            parent_id=parent.id,
        )
        if embed:
            self._embed_row(row)

        parent.status = RawStatus.DISTILLED.value
        session.add(row)
        session.flush()
        _log(
            session,
            parent.namespace,
            "distill",
            row.id,
            {"parent_id": parent.id, "title": row.title, "tags": row.tags},
        )
        return MemoryRecord.from_row(row)

    def produce(
        self,
        session: Session,
        insight_id: str,
        *,
        title: str,
        content: str,
        tags: Sequence[str] = (),
        source: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Crea un ``artifact`` a partir de un insight. No se embebe por defecto."""
        parent = self._row(session, insight_id)
        cleaned = content.strip()
        if not cleaned:
            raise EmptyMemoryContentError("producir")

        row = Memory(
            id=new_id(),
            namespace=parent.namespace,
            layer=Layer.ARTIFACT.value,
            title=title.strip() or parent.title,
            content=cleaned,
            source=source or parent.source,
            tags=_clean_tags(tags) or list(parent.tags or []),
            extra=dict(extra or {}),
            status=None,
            parent_id=parent.id,
        )
        if self.embed_artifacts:
            self._embed_row(row)

        session.add(row)
        session.flush()
        _log(session, parent.namespace, "produce", row.id, {"parent_id": parent.id})
        return MemoryRecord.from_row(row)

    def reindex(
        self, session: Session, *, namespace: str = "default", batch_size: int = 128
    ) -> int:
        """Reembebe todo lo que este en una capa indexable y actualiza la similitud."""
        stmt = select(Memory).where(
            Memory.namespace == namespace, Memory.layer.in_([str(x) for x in EMBEDDABLE_LAYERS])
        )
        rows = list(session.execute(stmt).scalars())
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            for row, vector in zip(
                chunk, self.embedder.embed([_indexable_text(row) for row in chunk]), strict=True
            ):
                _store_vector(row, vector, self.embedder.model)
        session.flush()
        _log(session, namespace, "reindex", None, {"count": len(rows)})
        return len(rows)

    def delete(self, session: Session, memory_id: str, *, namespace: str = "default") -> None:
        """Borra una memoria y desvincula a sus hijos en vez de cascadearlos."""
        row = self._row(session, memory_id, namespace)
        session.execute(update(Memory).where(Memory.parent_id == memory_id).values(parent_id=None))
        session.delete(row)
        session.flush()
        _log(session, namespace, "delete", memory_id, {})

    # ---------------------------------------------------------------- interno

    def _row(self, session: Session, memory_id: str, namespace: str | None = None) -> Memory:
        stmt = select(Memory).where(Memory.id == memory_id)
        if namespace is not None:
            stmt = stmt.where(Memory.namespace == namespace)
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            where = f" en namespace {namespace!r}" if namespace else ""
            raise MemoryNotFoundError(f"no existe la memoria {memory_id}{where}")
        return row

    def _embed_row(self, row: Memory) -> None:
        vector = self.embedder.embed([_indexable_text(row)])[0]
        _store_vector(row, vector, self.embedder.model)


__all__ = ["EmptyMemoryContentError", "MemoryNotFoundError", "MemoryService", "iter_layers"]
