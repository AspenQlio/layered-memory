"""Recorrido completo en un solo comando.

Simula lo que haria un agente durante un dia: captura notas crudas, las destila
a mano lo que vale la pena, produce el runbook que se llevaria el equipo,
responde usando contexto recuperado y escala a un humano lo que no puede
resolver.

    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layered_memory.agents.handoff import HandoffQueue
from layered_memory.memory.record import MemoryRecord
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import HashingEmbedder
from layered_memory.retrieval.index import VectorIndex, build_context_block
from layered_memory.store.db import (
    build_engine,
    create_session_factory,
    init_db,
    session_scope,
)


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * len(title))


def main() -> int:
    engine = build_engine("sqlite:///:memory:")
    init_db(engine)
    factory = create_session_factory(engine)

    embedder = HashingEmbedder(dim=512)
    service = MemoryService(embedder, embed_artifacts=True)
    index = VectorIndex(embedder)
    queue = HandoffQueue()

    with session_scope(factory) as session:
        rule("1. Capturas crudas (capa raw)")
        notes = {
            "cert": (
                "El certificado TLS del dominio de checkout expiro y el balanceador "
                "no lo renueva. El gateway devolvio 502 durante 22 minutos.",
                "Incidente de certificados",
            ),
            "guardia": (
                "El equipo de guardia no tiene alertas: se entera por los tickets. "
                "El tiempo medio de deteccion es de 47 minutos.",
                "Guardia sin alertas",
            ),
            "coste": (
                "Las imagenes de producto nunca se borran y el cluster de base de "
                "datos esta sobredimensionado. El gasto subio un 38 por ciento.",
                "Gasto de infraestructura",
            ),
        }
        raw_ids: dict[str, str] = {}
        for key, (content, title) in notes.items():
            record = service.capture(session, content, title=title, tags=["operaciones"])
            raw_ids[key] = record.id
            print(f"  [{record.id[:8]}] {record.title}")

        rule("2. La cola de destilacion")
        for record in service.pending_raw(session):
            print(f"  pendiente: {record.title}")

        rule("3. Destilacion (capa insight)")
        insight = service.distill(
            session,
            raw_ids["cert"],
            title="Las dependencias con fecha de expiracion necesitan alerta",
            content=(
                "El fallo aparecio en produccion y no en el monitoreo porque no "
                "existia una alerta 30 dias antes del vencimiento. Toda dependencia "
                "con fecha de expiracion debe avisar por si sola."
            ),
            tags=["certificados", "postmortem"],
        )
        print(f"  [{insight.id[:8]}] {insight.title}")
        print(f"  captura de origen marcada como: {service.get(session, raw_ids['cert']).status}")

        rule("4. Produccion (capa artifact)")
        artifact = service.produce(
            session,
            insight.id,
            title="Runbook: renovacion de certificados TLS",
            content=(
                "1. Alerta automatica 30 dias antes, al canal de guardia.\n"
                "2. Renovacion idempotente y validacion de la cadena completa.\n"
                "3. Prueba semanal contra un dominio espejo."
            ),
            tags=["runbook", "tls"],
        )
        print(f"  [{artifact.id[:8]}] {artifact.title}")

        rule("5. Recuperacion para el prompt")
        hits = index.search(session, "avisos que no llegan al cliente", k=2)
        for position, hit in enumerate(hits, start=1):
            print(f"  {position}. {hit.score:.3f}  {hit.record.title}")
        print("\n  bloque de contexto:")
        for line in build_context_block(hits, max_chars=320).splitlines():
            print(f"  | {line}")

        rule("6. Escalado a humano")
        handoff = queue.enqueue(
            session,
            "el cliente pide exencion de penalidad fuera de politica",
            session_id="demo-42",
            context={"monto": 120_000},
        )
        print(f"  encolado: [{handoff.id[:8]}] {handoff.reason}")
        claimed = queue.claim_next(session, claimed_by="reviewer-1")
        print(f"  reclamado por: {claimed.claimed_by}")
        resolved = queue.resolve(
            session, claimed.id, "aprobado al 50 por ciento de la penalidad", resolved_by="jefe"
        )
        print(f"  resuelto por {resolved.resolved_by}: {resolved.resolution}")
        print(f"  pendientes en cola: {queue.depth(session)}")

        rule("7. Linea de tiempo de una memoria")
        for row in reversed_lineage(service, session, artifact.id):
            print(f"  {row.layer:<8} {row.title}")

        rule("8. Estado del sistema")
        stats = service.stats(session)
        print(f"  total={stats['total']} por capa={stats['by_layer']}")
        print(f"  indexado={stats['indexed']} cobertura={stats['coverage']}")
        print(f"  pendientes de destilacion={stats['pending_distillation']}")

    return 0


def reversed_lineage(
    service: MemoryService, session: Session, memory_id: str
) -> list[MemoryRecord]:
    chain: list[MemoryRecord] = []
    current = service.get(session, memory_id)
    while current.parent_id is not None:
        chain.append(current)
        current = service.get(session, current.parent_id)
    chain.append(current)
    return chain


if __name__ == "__main__":
    raise SystemExit(main())
