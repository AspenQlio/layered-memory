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
from enum import StrEnum
from types import FrameType
from typing import Any, assert_never

from layered_memory.agents.handoff import HandoffConflictError, HandoffNotFoundError, HandoffQueue
from layered_memory.cli_parser import build_parser
from layered_memory.config import get_settings
from layered_memory.domain import HandoffStatus
from layered_memory.memory.service import MemoryNotFoundError, MemoryService
from layered_memory.retrieval.embeddings import build_embedder
from layered_memory.retrieval.index import build_context_block, build_index
from layered_memory.store.db import build_engine, create_session_factory, init_db, session_scope
from layered_memory.types import JsonValue


class HandoffCommand(StrEnum):
    ENQUEUE = "enqueue"
    NEXT = "next"
    DEPTH = "depth"
    LIST = "list"
    RESOLVE = "resolve"


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
        self.index = build_index(settings, embedder)
        self.handoffs = HandoffQueue()

    def session(self) -> Any:
        return session_scope(self.factory)


def _emit(payload: JsonValue) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


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
        command = HandoffCommand(args.handoff_command)
        match command:
            case HandoffCommand.ENQUEUE:
                record = runtime.handoffs.enqueue(
                    session, args.reason, namespace=args.namespace, session_id=args.session_id
                )
                _emit(record.as_dict())
            case HandoffCommand.NEXT:
                record = runtime.handoffs.claim_next(
                    session, namespace=args.namespace, claimed_by=args.name
                )
                _emit(record.as_dict() if record else {"idle": True, "queued": 0})
            case HandoffCommand.DEPTH:
                _emit({"queued": runtime.handoffs.depth(session, namespace=args.namespace)})
            case HandoffCommand.LIST:
                status_filter = HandoffStatus(args.status) if args.status else None
                records = runtime.handoffs.list(
                    session, namespace=args.namespace, status=status_filter
                )
                _emit([record.as_dict() for record in records])
            case HandoffCommand.RESOLVE:
                record = runtime.handoffs.resolve(
                    session, args.handoff_id, args.resolution, resolved_by=args.by
                )
                _emit(record.as_dict())
            case unreachable:
                assert_never(unreachable)
    return 0


def _worker(runtime: Runtime, args: argparse.Namespace) -> int:
    interval = args.interval if args.interval is not None else runtime.settings.worker_poll_seconds
    stop = {"now": False}

    def _handle_signal(_sig: int, _frame: FrameType | None) -> None:
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
