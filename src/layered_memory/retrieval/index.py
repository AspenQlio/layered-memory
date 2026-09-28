"""Indice vectorial en memoria: coseno sobre los embeddings persistidos.

Decision de diseno: el filtro por namespace y capa se hace en SQL (usa los
indices), y el filtro por tags en Python sobre el subconjunto ya acotado. Asi
el costo del recorrido lineal de numpy queda restringido al namespace y a las
capas que el llamador pidio, no a toda la tabla.

Escala: cuando el corpus por namespace pase de ~50k filas, el camino es mover
la columna a ``pgvector`` y dejar ``ORDER BY embedding <=> :query LIMIT k``.
La API de este modulo no cambia; cambia el interior de ``search``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, assert_never

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from layered_memory.domain import Layer
from layered_memory.memory.record import MemoryRecord
from layered_memory.retrieval.embeddings import Embedder
from layered_memory.store.models import Memory
from layered_memory.types import JsonObject

if TYPE_CHECKING:
    from layered_memory.config import Settings

_SEPARATOR = "\n\n"
_ELLIPSIS = "..."


class EmbeddingDimensionMismatchError(ValueError):
    def __init__(self, stored: int, configured: int) -> None:
        self.stored = stored
        self.configured = configured
        super().__init__(
            f"dimension guardada {stored} distinta de la del embedder {configured}; "
            "reindexa o cambia de embedder"
        )


class SearchIndex(Protocol):
    embedder: Embedder

    def search(
        self,
        session: Session,
        query: str,
        *,
        namespace: str = "default",
        layers: Sequence[Layer] | None = None,
        tags: Sequence[str] = (),
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchHit]: ...


class SearchHit:
    """Un resultado de busqueda con su puntaje y por que-score."""

    __slots__ = ("record", "score")

    def __init__(self, record: MemoryRecord, score: float) -> None:
        self.record = record
        self.score = score

    @property
    def id(self) -> str:
        return self.record.id

    def as_dict(self) -> JsonObject:
        return {
            "id": self.record.id,
            "score": round(self.score, 6),
            "layer": str(self.record.layer),
            "title": self.record.title,
            "content": self.record.content,
            "tags": self.record.tags,
            "parent_id": self.record.parent_id,
        }

    def __repr__(self) -> str:
        return f"SearchHit(id={self.id!r}, score={self.score:.4f}, title={self.record.title!r})"


class VectorIndex:
    """Busqueda por similitud coseno sobre la tabla ``memories``."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def search(
        self,
        session: Session,
        query: str,
        *,
        namespace: str = "default",
        layers: Sequence[Layer] | None = None,
        tags: Sequence[str] = (),
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Devuelve hasta ``k`` memorias ordenadas por similitud descendente."""
        if k <= 0:
            return []

        rows = _load_candidates(
            session,
            namespace=namespace,
            layers=layers,
            tags=tags,
        )
        if not rows:
            return []

        scores = _vector_scores(rows, query, self.embedder)

        ranked = sorted(
            zip(rows, scores.tolist(), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        hits = [
            SearchHit(MemoryRecord.from_row(row), score)
            for row, score in ranked
            if score >= min_score
        ]
        return hits[:k]


def build_index(settings: Settings, embedder: Embedder) -> SearchIndex:
    match settings.embedding_backend:
        case "hash" | "openai":
            return VectorIndex(embedder)
        case "hybrid":
            from layered_memory.retrieval.hybrid import HybridIndex

            return HybridIndex(
                embedder,
                lexical_dim=settings.hybrid_lexical_dim,
                rank_constant=settings.hybrid_rank_constant,
                semantic_weight=settings.hybrid_semantic_weight,
                semantic_head=settings.hybrid_semantic_head,
            )
        case unreachable:
            assert_never(unreachable)


def _load_candidates(
    session: Session,
    *,
    namespace: str,
    layers: Sequence[Layer] | None,
    tags: Sequence[str],
) -> list[Memory]:
    wanted = list(layers) if layers else [Layer.RAW, Layer.INSIGHT]
    stmt = (
        select(Memory)
        .where(
            Memory.namespace == namespace,
            Memory.layer.in_([str(layer) for layer in wanted]),
            Memory.embedding.is_not(None),
        )
        .order_by(Memory.created_at.desc())
        .limit(_candidate_cap(wanted))
    )
    rows = list(session.execute(stmt).scalars())
    required = {tag.lower() for tag in tags}
    return [row for row in rows if required <= {tag.lower() for tag in (row.tags or [])}]


def _vector_scores(rows: Sequence[Memory], query: str, embedder: Embedder) -> np.ndarray:
    matrix = np.vstack([np.frombuffer(row.embedding, dtype=np.float32) for row in rows])
    if matrix.shape[1] != embedder.dim:
        raise EmbeddingDimensionMismatchError(matrix.shape[1], embedder.dim)
    return matrix @ embedder.embed([query])[0]


def _candidate_cap(layers: Sequence[Layer]) -> int:
    """Cuantos candidatos por capa se descargan antes de rankear.

    Capa ``artifact``: menos filas (son largos y caros de embeber) pero el limite
    es alto porque suele contener justo la respuesta sintetizada.
    """
    caps = {Layer.RAW: 400, Layer.INSIGHT: 400, Layer.ARTIFACT: 200}
    return sum(caps.get(layer, 200) for layer in layers)


def build_context_block(
    hits: Sequence[SearchHit],
    *,
    max_chars: int = 4000,
    include_ids: bool = True,
) -> str:
    """Convierte resultados en un bloque de contexto para el prompt del agente.

    Recorta por presupuesto de caracteres en lugar de por numero de documentos:
    un documento largo puede valer mas que tres cortos, y el presupuesto es lo
    que de verdad limita al modelo.
    """
    parts: list[str] = []
    used = 0
    for position, hit in enumerate(hits, start=1):
        header = f"[{position}] {hit.record.title}"
        if include_ids:
            header += f" (id={hit.id})"
        body = hit.record.content.strip()
        chunk = f"{header}\n{body}"

        separator_cost = len(_SEPARATOR) if parts else 0
        remaining = max_chars - used - separator_cost
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            keep = max(remaining - len(_ELLIPSIS), 0)
            chunk = chunk[:keep].rstrip() + _ELLIPSIS
        parts.append(chunk)
        used += len(chunk) + separator_cost
        if used >= max_chars:
            break

    return _SEPARATOR.join(parts)
