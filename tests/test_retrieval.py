"""Calidad del indice vectorial y armado de contexto."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from layered_memory.domain import Layer
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import HashingEmbedder, tokenize
from layered_memory.retrieval.index import VectorIndex, build_context_block


def _seed(service: MemoryService, session: Session) -> None:
    service.capture(
        session,
        "El certificado TLS del gateway de pagos estaba vencido y provoco errores 502.",
        title="Incidente de certificados",
        tags=["incidente", "pagos"],
    )
    service.capture(
        session,
        "El alta de un vendedor toma seis dias por firmas manuales del contrato.",
        title="Proceso de onboarding",
        tags=["rrhh"],
    )
    service.capture(
        session,
        "El indice de busqueda del catalogo se reconstruye cada noche a las 3 AM.",
        title="Reindexacion nocturna",
        tags=["catalogo"],
    )


def test_tokenizer_folds_accents_and_drops_punctuation() -> None:
    assert tokenize("Configuración de REDES, paso 2!") == [
        "configuracion",
        "de",
        "redes",
        "paso",
        "2",
    ]


def test_embedding_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed(["perro gato", "perro gato"])[0]

    assert np.allclose(first, embedder.embed(["perro gato"])[0])
    assert pytest.approx(1.0, abs=1e-6) == float(np.linalg.norm(first))


def test_search_ranks_the_topical_document_first(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    _seed(service, session)

    hits = index.search(session, "certificado tls vencido del gateway de pagos", k=3)

    assert hits[0].record.title == "Incidente de certificados"
    assert hits[0].score > hits[1].score


def test_search_respects_k_and_min_score(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    _seed(service, session)

    assert len(index.search(session, "proceso de onboarding", k=2)) == 2
    assert index.search(session, "proceso de onboarding", k=0) == []
    assert index.search(session, "proceso de onboarding", k=5, min_score=0.99) == []


def test_tag_filter_is_exhaustive_not_partial(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    _seed(service, session)

    only_payments = index.search(session, "certificado", tags=["pagos"], k=5)
    both = index.search(session, "certificado", tags=["pagos", "incidente"], k=5)

    assert [h.record.title for h in only_payments] == ["Incidente de certificados"]
    assert [h.id for h in both] == [h.id for h in only_payments]
    assert index.search(session, "certificado", tags=["pagos", "inexistente"], k=5) == []


def test_search_is_scoped_to_the_namespace(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    service.capture(session, "contenido del equipo alpha", namespace="alpha")
    service.capture(session, "contenido del equipo beta", namespace="beta")

    assert len(index.search(session, "contenido", namespace="alpha")) == 1
    assert index.search(session, "contenido", namespace="gamma") == []


def test_search_ignores_layers_it_was_not_asked_for(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    raw = service.capture(session, "captura original sobre el proceso de compras", title="Compras")
    service.distill(
        session, raw.id, title="Insight de compras", content="El proceso tiene tres pasos."
    )

    assert [h.record.layer for h in index.search(session, "compras", layers=[Layer.RAW])] == [
        Layer.RAW
    ]
    assert {h.record.layer for h in index.search(session, "compras")} == {Layer.RAW, Layer.INSIGHT}


def test_dimension_mismatch_is_refused_instead_of_returning_noise(
    service: MemoryService, session: Session
) -> None:
    service.capture(session, "contenido con vector de 128 dimensiones")
    mismatched = VectorIndex(HashingEmbedder(dim=256))

    with pytest.raises(ValueError, match="dimension"):
        mismatched.search(session, "contenido")


def test_context_block_respects_the_character_budget(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    _seed(service, session)
    hits = index.search(session, "certificado pagos", k=3)

    block = build_context_block(hits, max_chars=120, include_ids=False)

    assert len(block) <= 120
    assert "id=" not in block
    assert block.startswith("[1] ")


def test_context_block_includes_ids_when_asked(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    _seed(service, session)
    hits = index.search(session, "onboarding", k=1)

    block = build_context_block(hits, max_chars=4000)

    assert hits[0].id in block


def test_empty_budget_yields_empty_block(
    service: MemoryService, index: VectorIndex, session: Session
) -> None:
    _seed(service, session)

    assert build_context_block(index.search(session, "onboarding", k=2), max_chars=0) == ""


def test_embed_returns_empty_matrix_for_no_texts() -> None:
    assert HashingEmbedder(dim=32).embed([]).shape == (0, 32)


def test_hash_embedder_gives_zero_vector_to_text_without_tokens() -> None:
    vector = HashingEmbedder(dim=32).embed(["!!! ???"])[0]

    assert vector.sum() == 0.0
