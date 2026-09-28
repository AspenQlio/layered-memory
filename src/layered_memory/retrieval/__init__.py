"""Embedders, indice vectorial y armado de contexto para prompts."""

from layered_memory.retrieval.embeddings import Embedder, HashingEmbedder, OpenAICompatEmbedder
from layered_memory.retrieval.hybrid import HybridIndex, reciprocal_rank_fusion
from layered_memory.retrieval.index import (
    SearchHit,
    SearchIndex,
    VectorIndex,
    build_context_block,
    build_index,
)

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "HybridIndex",
    "OpenAICompatEmbedder",
    "SearchHit",
    "SearchIndex",
    "VectorIndex",
    "build_context_block",
    "build_index",
    "reciprocal_rank_fusion",
]
