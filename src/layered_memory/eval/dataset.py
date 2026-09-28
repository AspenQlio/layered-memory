"""Carga y siembra del corpus de demostracion y del set de evaluacion.

El corpus esta versionado en JSON para que la evaluacion sea reproducible: si
cambia el codigo del indice, los mismos casos deben dar los mismos numeros, y
para eso los documentos y sus claves esperadas no pueden depender de una
generacion aleatoria ni de una base ya existente.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from layered_memory.domain import Layer
from layered_memory.eval.metrics import EvalCase
from layered_memory.memory.service import MemoryService


@dataclass(frozen=True, slots=True)
class Document:
    """Un documento del corpus, con la clave estable que espera el set de eval."""

    key: str
    layer: Layer
    title: str
    content: str
    tags: tuple[str, ...] = ()
    source: str | None = None
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class Corpus:
    namespace: str
    documents: tuple[Document, ...]

    @property
    def size(self) -> int:
        return len(self.documents)

    def keys(self) -> set[str]:
        return {doc.key for doc in self.documents}


def load_corpus(path: str | Path) -> Corpus:
    """Lee el corpus y valida que las referencias entre documentos cierren."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_documents = data.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError(f"{path}: 'documents' debe ser una lista no vacia")

    documents = tuple(_document(item) for item in raw_documents)
    known = {doc.key for doc in documents}
    duplicates = len(documents) - len(known)
    if duplicates:
        raise ValueError(f"{path}: hay {duplicates} claves duplicadas")

    for doc in documents:
        if doc.parent and doc.parent not in known:
            raise ValueError(f"{path}: '{doc.key}' referencia un padre inexistente: {doc.parent!r}")
        if doc.parent and doc.layer is Layer.RAW:
            raise ValueError(f"{path}: una captura ('{doc.key}') no puede tener padre")

    return Corpus(namespace=str(data.get("namespace", "demo")), documents=documents)


def load_cases(path: str | Path) -> tuple[EvalCase, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{path}: 'cases' debe ser una lista no vacia")
    return tuple(EvalCase.from_dict(item) for item in raw_cases)


def validate_cases(corpus: Corpus, cases: tuple[EvalCase, ...]) -> None:
    """Falla temprano si un caso espera una clave que el corpus no tiene."""
    known = corpus.keys()
    for case in cases:
        unknown = set(case.expected) - known
        if unknown:
            raise ValueError(
                f"el caso '{case.id}' espera claves ausentes del corpus: {sorted(unknown)}"
            )


def seed(service: MemoryService, session: Session, corpus: Corpus) -> dict[str, str]:
    """Escribe el corpus respetando el orden de promocion y devuelve clave -> id."""
    key_to_id: dict[str, str] = {}
    for doc in corpus.documents:
        if doc.layer is Layer.RAW:
            record = service.capture(
                session,
                doc.content,
                title=doc.title,
                namespace=corpus.namespace,
                source=doc.source,
                tags=doc.tags,
                extra={"key": doc.key},
            )
        else:
            if doc.parent is None or doc.parent not in key_to_id:
                raise ValueError(
                    f"el documento '{doc.key}' es {doc.layer} y necesita un parent ya sembrado"
                )
            parent_id = key_to_id[doc.parent]
            if doc.layer is Layer.INSIGHT:
                record = service.distill(
                    session,
                    parent_id,
                    title=doc.title,
                    content=doc.content,
                    tags=doc.tags,
                    source=doc.source,
                    extra={"key": doc.key},
                )
            else:
                record = service.produce(
                    session,
                    parent_id,
                    title=doc.title,
                    content=doc.content,
                    tags=doc.tags,
                    source=doc.source,
                    extra={"key": doc.key},
                )
        key_to_id[doc.key] = record.id
    return key_to_id


def _document(item: object) -> Document:
    if not isinstance(item, dict):
        raise ValueError(f"cada documento debe ser un objeto, recibi {type(item).__name__}")
    missing = {"key", "layer", "title", "content"} - set(item)
    if missing:
        raise ValueError(f"documento incompleto, faltan {sorted(missing)}: {item.get('key', '?')}")
    raw_layer = str(item["layer"])
    try:
        layer = Layer(raw_layer)
    except ValueError as exc:
        valid = ", ".join(str(candidate) for candidate in Layer)
        raise ValueError(
            f"el documento {item['key']!r} usa la capa invalida {raw_layer!r}; usa una de: {valid}"
        ) from exc
    return Document(
        key=str(item["key"]),
        layer=layer,
        title=str(item["title"]),
        content=str(item["content"]),
        tags=tuple(str(tag) for tag in item.get("tags", ())),
        source=str(item["source"]) if item.get("source") else None,
        parent=str(item["parent"]) if item.get("parent") else None,
    )


def write_report(report: dict[str, object], path: str | Path) -> Path:
    """Guarda el reporte JSON con indentacion estable para diffs legibles."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
