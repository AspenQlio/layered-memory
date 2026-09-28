"""Cola de escalado a humano."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from layered_memory.agents.handoff import (
    HandoffConflictError,
    HandoffNotFoundError,
    HandoffQueue,
    resolutions,
)
from layered_memory.domain import HandoffStatus


def test_enqueue_starts_queued_and_counts_towards_depth(
    queue: HandoffQueue, session: Session
) -> None:
    record = queue.enqueue(session, "el cliente pide un descuento fuera de politica")

    assert record.status is HandoffStatus.QUEUED
    assert record.pending is True
    assert queue.depth(session) == 1


def test_enqueue_rejects_a_blank_reason(queue: HandoffQueue, session: Session) -> None:
    with pytest.raises(ValueError, match="motivo"):
        queue.enqueue(session, "  ")


def test_claim_next_serves_fifo(queue: HandoffQueue, session: Session) -> None:
    first = queue.enqueue(session, "primera peticion", session_id="s1")
    second = queue.enqueue(session, "segunda peticion", session_id="s2")

    claimed = queue.claim_next(session, claimed_by="robot-1")

    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.claimed_by == "robot-1"
    assert claimed.status is HandoffStatus.CLAIMED
    assert queue.depth(session) == 1
    assert queue.get(session, second.id).status is HandoffStatus.QUEUED


def test_two_workers_cannot_claim_the_same_request(queue: HandoffQueue, session: Session) -> None:
    record = queue.enqueue(session, "peticion disputada")
    queue.claim(session, record.id, claimed_by="robot-1")

    with pytest.raises(HandoffConflictError, match="claimed"):
        queue.claim(session, record.id, claimed_by="robot-2")


def test_claim_next_returns_none_on_an_empty_queue(queue: HandoffQueue, session: Session) -> None:
    assert queue.claim_next(session) is None


def test_resolve_closes_the_request_and_stores_who_answered(
    queue: HandoffQueue, session: Session
) -> None:
    record = queue.enqueue(session, "peticion")
    queue.claim(session, record.id, claimed_by="robot-1")

    resolved = queue.resolve(session, record.id, "aprobado con 10%", resolved_by="jefe")

    assert resolved.status is HandoffStatus.RESOLVED
    assert resolved.resolved_by == "jefe"
    assert resolved.pending is False
    assert queue.depth(session) == 0


def test_resolve_accepts_a_request_that_was_never_claimed(
    queue: HandoffQueue, session: Session
) -> None:
    record = queue.enqueue(session, "peticion sin reclamar")

    resolved = queue.resolve(session, record.id, "respondido por correo")

    assert resolved.status is HandoffStatus.RESOLVED
    assert resolved.claimed_by is None


def test_resolve_rejects_an_empty_answer_and_a_terminal_request(
    queue: HandoffQueue, session: Session
) -> None:
    record = queue.enqueue(session, "peticion")
    queue.resolve(session, record.id, "primera respuesta")

    with pytest.raises(ValueError, match="resolucion"):
        queue.resolve(session, record.id, "  ")
    with pytest.raises(HandoffConflictError):
        queue.resolve(session, record.id, "segunda respuesta")


def test_cancel_is_terminal(queue: HandoffQueue, session: Session) -> None:
    record = queue.enqueue(session, "peticion obsoleta")
    queue.claim(session, record.id, claimed_by="robot-1")

    cancelled = queue.cancel(session, record.id)

    assert cancelled.status is HandoffStatus.CANCELLED
    with pytest.raises(HandoffConflictError):
        queue.cancel(session, record.id)


def test_get_raises_for_an_unknown_id(queue: HandoffQueue, session: Session) -> None:
    with pytest.raises(HandoffNotFoundError):
        queue.get(session, "no-existe")


def test_list_filters_by_status_and_namespace(queue: HandoffQueue, session: Session) -> None:
    resolved = queue.enqueue(session, "ya resuelta", namespace="equipo-a")
    queue.enqueue(session, "sigue esperando", namespace="equipo-a")
    queue.enqueue(session, "otra cola", namespace="equipo-b")
    queue.resolve(session, resolved.id, "hecho")

    waiting = queue.list(session, namespace="equipo-a", status=HandoffStatus.QUEUED)
    everything = queue.list(session, namespace="equipo-a")

    assert [r.reason for r in waiting] == ["sigue esperando"]
    assert len(everything) == 2


def test_resolutions_returns_answered_pairs_oldest_first(
    queue: HandoffQueue, session: Session
) -> None:
    first = queue.enqueue(session, "pregunta A")
    second = queue.enqueue(session, "pregunta B")
    queue.resolve(session, second.id, "respuesta B", resolved_by="jefe")
    queue.resolve(session, first.id, "respuesta A", resolved_by="jefe")
    queue.enqueue(session, "pregunta C sin respuesta")

    pairs = resolutions(session)

    assert set(pairs) == {("pregunta B", "respuesta B"), ("pregunta A", "respuesta A")}


def test_claim_records_the_injected_clock(queue: HandoffQueue, session: Session) -> None:
    frozen = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    queue = HandoffQueue(clock=lambda: frozen)
    record = queue.enqueue(session, "peticion con reloj propio")

    claimed = queue.claim(session, record.id, claimed_by="robot-1")

    assert claimed.claimed_at == frozen


def test_namespace_depth_is_independent(queue: HandoffQueue, session: Session) -> None:
    queue.enqueue(session, "a", namespace="equipo-a")
    queue.enqueue(session, "b", namespace="equipo-b")

    assert queue.depth(session, namespace="equipo-a") == 1
    assert queue.depth(session, namespace="equipo-b") == 1
    assert queue.depth(session, namespace="equipo-c") == 0
