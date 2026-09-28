"""Fixtures compartidas.

Cada test corre contra una base SQLite en memoria con un pool estatico, de
modo que un test no puede ver lo que otro escribio y ninguno depende del orden.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from layered_memory.agents.handoff import HandoffQueue
from layered_memory.config import Settings
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import Embedder, HashingEmbedder
from layered_memory.retrieval.index import VectorIndex
from layered_memory.store.db import build_engine, create_session_factory, init_db

EMBED_DIM = 128


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url="sqlite:///:memory:", embedding_dim=EMBED_DIM)


@pytest.fixture
def engine() -> Iterator[Engine]:
    active = build_engine("sqlite:///:memory:")
    init_db(active)
    yield active
    active.dispose()


@pytest.fixture
def factory(engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine)


@pytest.fixture
def embedder() -> Embedder:
    return HashingEmbedder(dim=EMBED_DIM)


@pytest.fixture
def service(embedder: Embedder) -> MemoryService:
    return MemoryService(embedder, embed_artifacts=True)


@pytest.fixture
def index(embedder: Embedder) -> VectorIndex:
    return VectorIndex(embedder)


@pytest.fixture
def queue() -> HandoffQueue:
    return HandoffQueue()


@pytest.fixture
def session(factory: sessionmaker[Session]) -> Iterator[Session]:
    active = factory()
    try:
        yield active
        active.commit()
    finally:
        active.close()


@pytest.fixture
def client(settings: Settings, engine: Engine) -> Iterator[TestClient]:
    from layered_memory.api.app import create_app

    with TestClient(create_app(settings, engine=engine)) as test_client:
        yield test_client


@pytest.fixture
def corpus() -> dict[str, Any]:
    return {
        "namespace": "demo",
        "documents": [
            {
                "key": "oncall",
                "layer": "raw",
                "title": "Incidente del gateway de pagos",
                "content": (
                    "El gateway devolvio 502 durante veinte minutos por certificado vencido."
                ),
                "tags": ["incidente", "pagos"],
            },
            {
                "key": "oncall-insight",
                "layer": "insight",
                "parent": "oncall",
                "title": "Causa raiz: renovacion de certificado sin alerta",
                "content": (
                    "El certificado expiraba en silencio porque nadie "
                    "monitoreaba la fecha de vencimiento."
                ),
                "tags": ["incidente", "certificados"],
            },
            {
                "key": "oncall-artifact",
                "layer": "artifact",
                "parent": "oncall-insight",
                "title": "Runbook: renovacion de certificados TLS",
                "content": (
                    "1. Alerta 30 dias antes. 2. Renovacion automatica. "
                    "3. Verificar cadena completa."
                ),
                "tags": ["runbook"],
            },
            {
                "key": "rrhh",
                "layer": "raw",
                "title": "Onboarding de nuevo vendedor",
                "content": (
                    "El alta de un vendedor toma seis dias por firmas manuales del contrato."
                ),
                "tags": ["rrhh", "proceso"],
            },
            {
                "key": "datos",
                "layer": "raw",
                "title": "Calidad de datos de clientes",
                "content": "El 12 por ciento de los registros tienen direccion vacia o duplicada.",
                "tags": ["datos"],
            },
        ],
    }
