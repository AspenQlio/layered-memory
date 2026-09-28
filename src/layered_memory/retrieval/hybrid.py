from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from layered_memory.domain import Layer
from layered_memory.memory.record import MemoryRecord
from layered_memory.retrieval.embeddings import Embedder, HashingEmbedder
from layered_memory.retrieval.index import SearchHit, _load_candidates, _vector_scores
from layered_memory.store.models import Memory


class RankFusionConfigurationError(ValueError):
    pass


class HybridIndex:
    """Fusiona rankings semantico y lexical mediante RRF normalizado."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        lexical_dim: int = 512,
        rank_constant: int = 60,
        semantic_weight: float = 1.0,
        semantic_head: int = 5,
    ) -> None:
        if rank_constant <= 0:
            raise RankFusionConfigurationError(
                f"rank_constant debe ser positivo, recibido: {rank_constant}"
            )
        if semantic_weight <= 0:
            raise RankFusionConfigurationError(
                f"semantic_weight debe ser positivo, recibido: {semantic_weight}"
            )
        if semantic_head < 0:
            raise RankFusionConfigurationError(
                f"semantic_head no puede ser negativo, recibido: {semantic_head}"
            )
        self.embedder = embedder
        self.lexical_embedder = HashingEmbedder(dim=lexical_dim)
        self.rank_constant = rank_constant
        self.semantic_weight = semantic_weight
        self.semantic_head = semantic_head

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

        semantic_scores = _vector_scores(rows, query, self.embedder)
        lexical_matrix = self.lexical_embedder.embed([_lexical_text(row) for row in rows])
        lexical_query = self.lexical_embedder.embed([query])[0]
        lexical_scores = lexical_matrix @ lexical_query

        semantic_ranking = _rank_ids(rows, semantic_scores.tolist())
        lexical_ranking = _rank_ids(rows, lexical_scores.tolist())
        fused = reciprocal_rank_fusion(
            (semantic_ranking, lexical_ranking),
            rank_constant=self.rank_constant,
            weights=(self.semantic_weight, 1.0),
        )
        semantic_by_id = dict(zip((row.id for row in rows), semantic_scores, strict=True))
        row_by_id = {row.id: row for row in rows}
        head = semantic_ranking[: self.semantic_head]
        head_ids = set(head)
        tail = sorted(
            (memory_id for memory_id in semantic_ranking if memory_id not in head_ids),
            key=lambda memory_id: (fused[memory_id], semantic_by_id[memory_id]),
            reverse=True,
        )
        ranked = [row_by_id[memory_id] for memory_id in (*head, *tail)]
        hits = [
            SearchHit(MemoryRecord.from_row(row), fused[row.id])
            for row in ranked
            if fused[row.id] >= min_score
        ]
        return hits[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rank_constant: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Combina rankings y normaliza el resultado al intervalo cerrado [0, 1]."""
    if rank_constant <= 0:
        raise RankFusionConfigurationError(
            f"rank_constant debe ser positivo, recibido: {rank_constant}"
        )
    if not rankings:
        return {}
    active_weights = tuple(weights) if weights is not None else (1.0,) * len(rankings)
    if len(active_weights) != len(rankings):
        raise RankFusionConfigurationError(
            f"se recibieron {len(active_weights)} pesos para {len(rankings)} rankings"
        )
    if any(weight <= 0 for weight in active_weights):
        raise RankFusionConfigurationError("todos los pesos RRF deben ser positivos")

    raw_scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, active_weights, strict=True):
        seen: set[str] = set()
        for rank, memory_id in enumerate(ranking, start=1):
            if memory_id in seen:
                continue
            seen.add(memory_id)
            raw_scores[memory_id] = raw_scores.get(memory_id, 0.0) + weight / (rank_constant + rank)

    maximum = sum(active_weights) / (rank_constant + 1)
    return {memory_id: score / maximum for memory_id, score in raw_scores.items()}


def _rank_ids(rows: Sequence[Memory], scores: Sequence[float]) -> tuple[str, ...]:
    ranked = sorted(
        zip(rows, scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return tuple(row.id for row, _score in ranked)


def _lexical_text(row: Memory) -> str:
    tags = " ".join(row.tags or ())
    return f"{row.title}\n\n{row.content}\n\n{tags}"
