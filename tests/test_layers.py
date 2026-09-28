"""Transiciones entre capas y su trazabilidad."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from layered_memory.domain import Layer, RawStatus
from layered_memory.memory.service import MemoryNotFoundError, MemoryService


def test_capture_lands_in_raw_and_derives_a_title(service: MemoryService, session: Session) -> None:
    record = service.capture(
        session, "El informe trimestral llega con dos semanas de atraso.", tags=["reportes"]
    )

    assert record.layer is Layer.RAW
    assert record.status is RawStatus.PENDING
    assert record.title == "El informe trimestral llega con dos semanas de atraso."
    assert record.embedded is True


def test_capture_rejects_blank_content(service: MemoryService, session: Session) -> None:
    with pytest.raises(ValueError, match="vacio"):
        service.capture(session, "   \n  ")


def test_tags_are_normalized_and_deduplicated(service: MemoryService, session: Session) -> None:
    record = service.capture(session, "contenido", tags=["Infra", "infra", " RED ", ""])

    assert record.tags == ["infra", "red"]


def test_promotion_chain_is_linked_and_marks_the_origin(
    service: MemoryService, session: Session
) -> None:
    raw = service.capture(
        session, "Se perdio la conexion con el servidor de respaldo.", title="Caida"
    )
    insight = service.distill(
        session,
        raw.id,
        title="El respaldo no tiene monitoreo",
        content="Falta alerta de disponibilidad.",
    )
    artifact = service.produce(
        session,
        insight.id,
        title="Plan de alertas",
        content="1. Heartbeat. 2. Umbral. 3. Escalamiento.",
    )

    assert insight.parent_id == raw.id
    assert artifact.parent_id == insight.id
    assert service.get(session, raw.id).status is RawStatus.DISTILLED
    assert [c.layer for c in service.children(session, raw.id)] == [Layer.INSIGHT]
    assert [c.layer for c in service.children(session, insight.id)] == [Layer.ARTIFACT]


def test_distilled_capture_leaves_the_pending_queue(
    service: MemoryService, session: Session
) -> None:
    first = service.capture(session, "primera captura")
    second = service.capture(session, "segunda captura")
    service.distill(session, first.id, title="t", content="c")

    pending_ids = [r.id for r in service.pending_raw(session)]

    assert pending_ids == [second.id]


def test_distill_preserves_parent_tags_when_none_given(
    service: MemoryService, session: Session
) -> None:
    raw = service.capture(session, "contenido con contexto", tags=["infra", "red"])
    insight = service.distill(session, raw.id, title="titulo", content="destilado")

    assert insight.tags == ["infra", "red"]
    assert insight.namespace == raw.namespace


def test_stats_counts_layers_and_coverage(service: MemoryService, session: Session) -> None:
    raw = service.capture(session, "captura uno")
    service.capture(session, "captura dos")
    service.distill(session, raw.id, title="destilado", content="cuerpo")

    stats = service.stats(session)

    assert stats["total"] == 3
    assert stats["by_layer"]["raw"] == 2
    assert stats["by_layer"]["insight"] == 1
    assert stats["pending_distillation"] == 1
    assert stats["coverage"] == 1.0


def test_namespaces_are_isolated(service: MemoryService, session: Session) -> None:
    service.capture(session, "contenido equipo a", namespace="equipo-a")
    service.capture(session, "contenido equipo b", namespace="equipo-b")

    assert service.stats(session, namespace="equipo-a")["total"] == 1
    assert service.stats(session)["total"] == 0


def test_get_from_another_namespace_is_not_found(service: MemoryService, session: Session) -> None:
    record = service.capture(session, "privado", namespace="equipo-a")

    with pytest.raises(MemoryNotFoundError, match="namespace"):
        service.get(session, record.id, namespace="equipo-b")


def test_delete_detaches_children_instead_of_cascading(
    service: MemoryService, session: Session
) -> None:
    raw = service.capture(session, "captura")
    insight = service.distill(session, raw.id, title="destilado", content="cuerpo")

    service.delete(session, raw.id)

    assert service.get(session, insight.id).parent_id is None
    assert service.children(session, insight.id) == []


def test_reindex_reembeds_everything_indexable(service: MemoryService, session: Session) -> None:
    raw = service.capture(session, "captura con vector")
    service.capture(session, "captura sin vector", embed=False)
    assert service.stats(session)["indexed"] == 1

    assert service.reindex(session) == 2
    assert service.stats(session)["indexed"] == 2
    assert service.get(session, raw.id).embedded is True


def test_layer_next_walks_forward_and_stops() -> None:
    assert Layer.RAW.next is Layer.INSIGHT
    assert Layer.INSIGHT.next is Layer.ARTIFACT
    assert Layer.ARTIFACT.next is None
