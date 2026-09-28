"""Corrida de la evaluacion de retrieval.

Uso tipico contra el corpus de demostracion, sin tocar la base de desarrollo:

    layered-memory-eval --corpus docs/demo-corpus.json \\
        --dataset docs/eval-dataset.json --json docs/eval-results.json

Cada corrida escribe un reporte con los numeros agregados y el ranking de cada
caso, para que una regresion se vea en el diff y no en la memoria del equipo.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from layered_memory.config import Settings, get_settings
from layered_memory.domain import Layer
from layered_memory.eval.dataset import load_cases, load_corpus, seed, validate_cases, write_report
from layered_memory.eval.metrics import CaseResult, EvalReport, hit_at_k
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import Embedder, build_embedder
from layered_memory.retrieval.index import build_index
from layered_memory.store.db import build_engine, create_session_factory, init_db, session_scope

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def run_evaluation(
    *,
    corpus_path: str | Path,
    dataset_path: str | Path,
    settings: Settings | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    database_url: str = "sqlite:///:memory:",
) -> EvalReport:
    """Siembra el corpus en una base efimera y evalua cada consulta."""
    active = settings or get_settings()
    corpus = load_corpus(corpus_path)
    cases = load_cases(dataset_path)
    validate_cases(corpus, cases)

    engine = build_engine(database_url)
    embedder: Embedder | None = None
    try:
        init_db(engine)
        factory = create_session_factory(engine)
        embedder = build_embedder(active)
        # La evalucion indexa las tres capas: el grupo 'promocion' solo tiene
        # sentido si los artifacts son alcanzables por la busqueda.
        service = MemoryService(embedder, embed_artifacts=True)
        index = build_index(active, embedder)
        depth = max(ks)

        with session_scope(factory) as session:
            key_to_id = seed(service, session, corpus)

            results: list[CaseResult] = []
            misses: list[str] = []
            for case in cases:
                hits = index.search(
                    session,
                    case.query,
                    namespace=corpus.namespace,
                    layers=list(Layer),
                    k=depth,
                )
                ranked = tuple(h.id for h in hits)
                expected_ids = tuple(key_to_id[key] for key in case.expected)
                results.append(
                    CaseResult(
                        case_id=case.id,
                        query=case.query,
                        expected=expected_ids,
                        ranked=ranked,
                        scores=tuple(h.score for h in hits),
                        group=case.group,
                    )
                )
                if not hit_at_k(ranked, expected_ids, depth):
                    misses.append(f"{case.id}: {case.query}")

        return EvalReport(
            backend=active.embedding_backend,
            model=embedder.model,
            dim=embedder.dim,
            ks=tuple(sorted(ks)),
            results=tuple(results),
            corpus_size=corpus.size,
            misses=tuple(misses),
        )
    finally:
        if embedder is not None:
            embedder.close()
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layered-memory-eval",
        description="Evalua la recuperacion de memoria y escribe un reporte reproducible.",
    )
    parser.add_argument("--corpus", required=True, help="Ruta al corpus JSON.")
    parser.add_argument("--dataset", required=True, help="Ruta al set de casos JSON.")
    parser.add_argument(
        "--ks", default="1,3,5,10", help="Cortes de k a promediar (por omision 1,3,5,10)."
    )
    parser.add_argument(
        "--backend",
        choices=("hash", "openai", "hybrid"),
        default=None,
        help="Fuerza el backend de embeddings en vez de usar la configuracion.",
    )
    parser.add_argument("--embedding-model", default=None, help="Modelo del backend 'openai'.")
    parser.add_argument(
        "--embedding-base-url", default=None, help="Endpoint compatible con OpenAI /embeddings."
    )
    parser.add_argument(
        "--database-url", default="sqlite:///:memory:", help="Base efimera de la corrida."
    )
    parser.add_argument("--json", dest="json_path", default=None, help="Escribe el reporte aqui.")
    parser.add_argument(
        "--markdown", dest="markdown_path", default=None, help="Escribe la tabla aqui."
    )
    parser.add_argument("--quiet", action="store_true", help="Solo imprime el agregado.")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    changes: dict[str, str] = {}
    if args.backend:
        changes["embedding_backend"] = args.backend
    if args.embedding_model:
        changes["embedding_model"] = args.embedding_model
    if args.embedding_base_url:
        changes["embedding_base_url"] = args.embedding_base_url
    return settings.model_copy(update=changes) if changes else settings


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ks = tuple(sorted({int(part) for part in args.ks.split(",") if part.strip()}))
    except ValueError:
        print("error: --ks debe ser una lista de enteros, por ejemplo 1,3,5,10", file=sys.stderr)
        return 2
    if not ks:
        print("error: --ks no puede quedar vacio", file=sys.stderr)
        return 2

    try:
        report = run_evaluation(
            corpus_path=args.corpus,
            dataset_path=args.dataset,
            settings=_settings_from_args(args),
            ks=ks,
            database_url=args.database_url,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(report.to_markdown())
        print()

    agg = report.aggregate()
    summary = "  ".join(f"{key}={agg[key]:.4f}" for key in sorted(agg))
    print(f"corpus={report.corpus_size} casos={len(report.results)} {summary}")

    if args.json_path:
        target = write_report(report.as_dict(), args.json_path)
        print(f"reporte json: {target}")
    if args.markdown_path:
        target = Path(args.markdown_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.to_markdown() + "\n", encoding="utf-8")
        print(f"tabla markdown: {target}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
