from __future__ import annotations

import argparse

from layered_memory.domain import HandoffStatus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layered-memory",
        description="Memoria por capas para agentes de IA.",
    )
    parser.add_argument(
        "--database-url", default=None, help="Sobrescribe LAYERED_MEMORY_DATABASE_URL."
    )
    parser.add_argument(
        "--no-index-artifacts",
        action="store_true",
        help="No embebe la capa artifact (por omision si la indexa).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Guarda contenido literal en la capa raw.")
    capture.add_argument("content")
    capture.add_argument("--title", default=None)
    capture.add_argument("--source", default=None)
    capture.add_argument("--tag", action="append", default=[], dest="tags")
    capture.add_argument("--namespace", default="default")
    capture.add_argument("--no-embed", action="store_true")

    distill = sub.add_parser("distill", help="Destila una captura en un insight.")
    distill.add_argument("memory_id")
    distill.add_argument("--title", required=True)
    distill.add_argument("--content", required=True)
    distill.add_argument("--tag", action="append", default=[], dest="tags")

    produce = sub.add_parser("produce", help="Produce un artifact a partir de un insight.")
    produce.add_argument("memory_id")
    produce.add_argument("--title", required=True)
    produce.add_argument("--content", required=True)
    produce.add_argument("--tag", action="append", default=[], dest="tags")

    show = sub.add_parser("show", help="Muestra una memoria con su ascendencia.")
    show.add_argument("memory_id")

    search = sub.add_parser("search", help="Busca por similitud.")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=5)
    search.add_argument(
        "--layer",
        action="append",
        default=[],
        dest="layers",
        choices=("raw", "insight", "artifact"),
    )
    search.add_argument("--tag", action="append", default=[], dest="tags")
    search.add_argument("--min-score", type=float, default=0.0)
    search.add_argument("--context", type=int, default=0, help="Caracteres del bloque de contexto.")
    search.add_argument("--namespace", default="default")

    pending = sub.add_parser("pending", help="Capturas que faltan por destilar.")
    pending.add_argument("--limit", type=int, default=20)
    pending.add_argument("--namespace", default="default")

    stats = sub.add_parser("stats", help="Resumen por capa.")
    stats.add_argument("--namespace", default="default")

    sub.add_parser("reindex", help="Reembebe lo indexable de un namespace.")

    handoff = sub.add_parser("handoff", help="Gestiona la cola de escalado a humano.")
    hsub = handoff.add_subparsers(dest="handoff_command", required=True)
    enqueue = hsub.add_parser("enqueue")
    enqueue.add_argument("reason")
    enqueue.add_argument("--session-id", default=None)
    enqueue.add_argument("--namespace", default="default")
    hnext = hsub.add_parser("next", help="Reclama la peticion mas antigua.")
    hnext.add_argument("--name", default="worker")
    hnext.add_argument("--namespace", default="default")
    hdepth = hsub.add_parser("depth", help="Cuantas peticiones siguen esperando.")
    hdepth.add_argument("--namespace", default="default")
    hlist = hsub.add_parser("list")
    hlist.add_argument("--namespace", default="default")
    hlist.add_argument("--status", default=None, choices=[str(status) for status in HandoffStatus])
    resolve = hsub.add_parser("resolve")
    resolve.add_argument("handoff_id")
    resolve.add_argument("resolution")
    resolve.add_argument("--by", default="human")

    worker = sub.add_parser("worker", help="Vacia la cola de handoffs en bucle.")
    worker.add_argument("--once", action="store_true", help="Atiende una peticion y sale.")
    worker.add_argument("--interval", type=float, default=None)
    worker.add_argument("--name", default="worker")
    worker.add_argument("--namespace", default="default")

    serve = sub.add_parser("serve", help="Levanta la API HTTP con uvicorn.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    return parser
