"""Contratos de entrada y salida de la API HTTP."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from layered_memory.domain import HandoffStatus, Layer, RawStatus


class CaptureRequest(BaseModel):
    content: str = Field(min_length=1, description="Contenido literal a guardar.")
    title: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"
    embed: bool = True


class DistillRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    embed: bool = True


class ProduceRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
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


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    namespace: str = "default"
    layers: list[Layer] | None = Field(
        default=None, description="Capas a considerar. Por omision: raw e insight."
    )
    tags: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)


class SearchHitResponse(BaseModel):
    id: str
    score: float
    layer: Layer
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    parent_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    hits: list[SearchHitResponse]


class ContextRequest(SearchRequest):
    """Busqueda con recorte por presupuesto de caracteres."""

    max_chars: int = Field(default=4000, ge=100, le=100_000)


class ContextResponse(BaseModel):
    query: str
    block: str
    ids: list[str]
    chars: int


class HandoffCreateRequest(BaseModel):
    reason: str = Field(min_length=1)
    namespace: str = "default"
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class HandoffResponse(BaseModel):
    id: str
    namespace: str
    session_id: str | None = None
    status: HandoffStatus
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)
    resolution: str | None = None
    claimed_by: str | None = None
    resolved_by: str | None = None
    created_at: datetime | None = None
    claimed_at: datetime | None = None
    resolved_at: datetime | None = None
    pending: bool = True


class ResolveRequest(BaseModel):
    resolution: str = Field(min_length=1)
    resolved_by: str = "human"


class ClaimRequest(BaseModel):
    claimed_by: str = "worker"


class StatsResponse(BaseModel):
    namespace: str
    total: int
    by_layer: dict[str, int]
    pending_distillation: int
    indexed: int
    coverage: float


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_backend: str
    embedding_model: str
    embedding_dim: int
    database: str
