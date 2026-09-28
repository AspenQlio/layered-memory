from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from sqlalchemy.orm import Session

from layered_memory.config import Settings
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import OpenAICompatEmbedder, build_embedder
from layered_memory.retrieval.hybrid import (
    HybridIndex,
    RankFusionConfigurationError,
    reciprocal_rank_fusion,
)
from layered_memory.retrieval.index import build_index


class _SemanticEmbedder:
    dim = 2
    model = "semantic-test"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[tuple[float, float]] = []
        for text in texts:
            if text == "omega meaning" or "semantic-only" in text:
                vectors.append((1.0, 0.0))
            elif "shared" in text:
                vectors.append((0.8, 0.2))
            else:
                vectors.append((0.0, 1.0))
        return np.asarray(vectors, dtype=np.float32)

    def close(self) -> None:
        pass


def test_rrf_favors_a_result_supported_by_both_rankings() -> None:
    scores = reciprocal_rank_fusion(
        (("semantic-only", "shared", "lexical-only"), ("shared", "lexical-only", "semantic-only")),
        rank_constant=60,
    )

    assert scores["shared"] > scores["semantic-only"]
    assert scores["shared"] > scores["lexical-only"]
    assert 0.0 < scores["shared"] <= 1.0


def test_rrf_rejects_a_non_positive_rank_constant() -> None:
    with pytest.raises(RankFusionConfigurationError):
        reciprocal_rank_fusion((("memory",),), rank_constant=0)


def test_weighted_rrf_can_prioritize_the_semantic_ranking() -> None:
    scores = reciprocal_rank_fusion(
        (("semantic-only", "shared", "lexical-only"), ("shared", "lexical-only", "semantic-only")),
        rank_constant=60,
        weights=(3.0, 1.0),
    )

    assert scores["semantic-only"] > scores["shared"]


def test_rrf_rejects_weights_that_do_not_match_the_rankings() -> None:
    with pytest.raises(RankFusionConfigurationError):
        reciprocal_rank_fusion((("memory",),), weights=(1.0, 1.0))


def test_hybrid_search_promotes_the_result_supported_by_both_signals(session: Session) -> None:
    embedder = _SemanticEmbedder()
    service = MemoryService(embedder)
    service.capture(session, "conceptual root cause", title="semantic-only")
    service.capture(session, "omega", title="shared")
    service.capture(session, "omega with several unrelated filler words", title="lexical-only")
    index = HybridIndex(embedder, lexical_dim=4096, rank_constant=60, semantic_head=0)

    hits = index.search(session, "omega meaning", k=3)

    assert [hit.record.title for hit in hits] == ["shared", "semantic-only", "lexical-only"]
    assert all(0.0 < hit.score <= 1.0 for hit in hits)


def test_hybrid_search_applies_min_score_to_normalized_rrf(session: Session) -> None:
    embedder = _SemanticEmbedder()
    service = MemoryService(embedder)
    service.capture(session, "conceptual root cause", title="semantic-only")
    service.capture(session, "omega", title="shared")
    service.capture(session, "omega with several unrelated filler words", title="lexical-only")
    index = HybridIndex(embedder, lexical_dim=4096, rank_constant=60, semantic_head=0)

    hits = index.search(session, "omega meaning", k=3, min_score=0.99)

    assert [hit.record.title for hit in hits] == ["shared"]


def test_hybrid_search_preserves_the_configured_semantic_head(session: Session) -> None:
    embedder = _SemanticEmbedder()
    service = MemoryService(embedder)
    service.capture(session, "conceptual root cause", title="semantic-only")
    service.capture(session, "omega", title="shared")
    service.capture(session, "omega with several unrelated filler words", title="lexical-only")
    index = HybridIndex(
        embedder,
        lexical_dim=4096,
        rank_constant=60,
        semantic_head=1,
    )

    hits = index.search(session, "omega meaning", k=3)

    assert [hit.record.title for hit in hits] == ["semantic-only", "shared", "lexical-only"]


def test_hybrid_backend_builds_semantic_embedder_and_hybrid_index() -> None:
    settings = Settings(
        embedding_backend="hybrid",
        embedding_dim=384,
        hybrid_lexical_dim=1024,
        hybrid_rank_constant=30,
        hybrid_semantic_weight=4.0,
        hybrid_semantic_head=7,
    )
    embedder = build_embedder(settings)
    try:
        index = build_index(settings, embedder)

        assert isinstance(embedder, OpenAICompatEmbedder)
        assert isinstance(index, HybridIndex)
        assert index.lexical_embedder.dim == 1024
        assert index.rank_constant == 30
        assert index.semantic_weight == 4.0
        assert index.semantic_head == 7
    finally:
        embedder.close()
