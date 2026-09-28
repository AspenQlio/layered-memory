"""Interfaz de linea de comandos.

Una app con memoria necesita poder operarse sin HTTP — para el bucle de
captura, para depurar una busqueda y para correr un worker en una sesion SSH
sin port forwarding. Cada subcomando imprime JSON para que sea composable con
``jq`` y con otros scripts.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from layered_memory.agents.handoff import HandoffConflictError, HandoffNotFoundError, HandoffQueue
from layered_memory.config import get_settings
from layered_memory.domain import HandoffStatus
from layered_memory.memory.service import MemoryNotFoundError, MemoryService
from layered_memory.retrieval.embeddings import build_embedder
from layered_memory.retrieval.index import VectorIndex, build_context_block
from layered_memory.store.db import build_engine, create_session_factory, init_db, session_scope


class Runtime:
    """Motor, servicio, indice y cola compartidos por todos los subcomandos."""

    def __init__(self, args: argparse.Namespace) -> None:
        settings = get_settings()
        if getattr(args, "database_url", None):
            settings = settings.model_copy(update={"database_url": args.database_url})
        self.settings = settings
        self.engine = build_engine(settings.database_url)
        init_db(self.engine)
        self.factory = create_session_factory(self.engine)
        embedder = build_embedder(settings)
        self.memory = MemoryService(embedder, embed_artifacts=not args.no_index_artifacts)
        self.index = VectorIndex(embedder)
        self.handoffs = HandoffQueue()

    def session(self) -> Any:
        return session_scope(self.factory)


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


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
    hlist.add_argument("--status", default=None, choices=[str(s) for s in HandoffStatus])
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = Runtime(args)

    handlers: dict[str, Any] = {
        "capture": _capture,
        "distill": _distill,
        "produce": _produce,
        "show": _show,
        "search": _search,
        "pending": _pending,
        "stats": _stats,
        "reindex": _reindex,
        "handoff": _handoff,
        "worker": _worker,
        "serve": _serve,
    }
    try:
        return int(handlers[args.command](runtime, args))
    except (MemoryNotFoundError, HandoffNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (HandoffConflictError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrumpido", file=sys.stderr)
        return 130


def _capture(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        record = runtime.memory.capture(
            session,
            args.content,
            title=args.title,
            namespace=args.namespace,
            source=args.source,
            tags=args.tags,
            embed=not args.no_embed,
        )
        _emit(record.model_dump(mode="json"))
    return 0


def _distill(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        record = runtime.memory.distill(
            session,
            args.memory_id,
            title=args.title,
            content=args.content,
            tags=args.tags,
        )
        _emit(record.model_dump(mode="json"))
    return 0


def _produce(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        record = runtime.memory.produce(
            session, args.memory_id, title=args.title, content=args.content, tags=args.tags
        )
        _emit(record.model_dump(mode="json"))
    return 0


def _show(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        record = runtime.memory.get(session, args.memory_id)
        payload: dict[str, Any] = record.model_dump(mode="json")
        payload["children"] = [
            c.model_dump(mode="json") for c in runtime.memory.children(session, args.memory_id)
        ]
        _emit(payload)
    return 0


def _search(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        hits = runtime.index.search(
            session,
            args.query,
            namespace=args.namespace,
            layers=args.layers or None,
            tags=args.tags,
            k=args.k,
            min_score=args.min_score,
        )
        payload: dict[str, Any] = {
            "query": args.query,
            "count": len(hits),
            "hits": [h.as_dict() for h in hits],
        }
        if args.context:
            payload["context"] = build_context_block(hits, max_chars=args.context)
        _emit(payload)
    return 0


def _pending(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        records = runtime.memory.pending_raw(session, namespace=args.namespace, limit=args.limit)
        _emit([r.model_dump(mode="json") for r in records])
    return 0


def _stats(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        _emit(runtime.memory.stats(session, namespace=args.namespace))
    return 0


def _reindex(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        _emit({"reindexed": runtime.memory.reindex(session)})
    return 0


def _handoff(runtime: Runtime, args: argparse.Namespace) -> int:
    with runtime.session() as session:
        if args.handoff_command == "enqueue":
            record = runtime.handoffs.enqueue(
                session, args.reason, namespace=args.namespace, session_id=args.session_id
            )
            _emit(record.as_dict())
        elif args.handoff_command == "next":
            record = runtime.handoffs.claim_next(
                session, namespace=args.namespace, claimed_by=args.name
            )
            _emit(record.as_dict() if record else {"idle": True, "queued": 0})
        elif args.handoff_command == "depth":
            _emit({"queued": runtime.handoffs.depth(session, namespace=args.namespace)})
        elif args.handoff_command == "list":
            status_filter = HandoffStatus(args.status) if args.status else None
            records = runtime.handoffs.list(session, namespace=args.namespace, status=status_filter)
            _emit([r.as_dict() for r in records])
        else:
            record = runtime.handoffs.resolve(
                session, args.handoff_id, args.resolution, resolved_by=args.by
            )
            _emit(record.as_dict())
    return 0


def _worker(runtime: Runtime, args: argparse.Namespace) -> int:
    interval = args.interval if args.interval is not None else runtime.settings.worker_poll_seconds
    stop = {"now": False}

    def _handle_signal(_sig: int, _frame: object) -> None:
        stop["now"] = True

    with suppress(KeyboardInterrupt):
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        while not stop["now"]:
            with runtime.session() as session:
                pending = runtime.handoffs.depth(session, namespace=args.namespace)
                record = runtime.handoffs.claim_next(
                    session, namespace=args.namespace, claimed_by=args.name
                )
            if record is None:
                if args.once:
                    print("cola vacia")
                    return 0
                time.sleep(interval)
                continue

            print(f"[{record.id}] {record.reason}")
            print(f" claimed_by={record.claimed_by} pending={pending}")
            if args.once:
                return 0
            time.sleep(interval)
    return 0


def _serve(runtime: Runtime, args: argparse.Namespace) -> int:
    import uvicorn

    from layered_memory.api.app import create_app

    app = create_app(runtime.settings, engine=runtime.engine)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
