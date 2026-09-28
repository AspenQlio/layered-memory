from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from layered_memory.domain import Layer
from layered_memory.store.models import Event, Memory


def indexable_text(row: Memory) -> str:
    return f"{row.title}\n\n{row.content}"


def store_vector(row: Memory, vector: np.ndarray, model: str) -> None:
    normalized = vector.astype(np.float32)
    norm = float(np.linalg.norm(normalized))
    if norm > 0:
        normalized = normalized / norm
    row.embedding = normalized.tobytes()
    row.embedding_dim = int(normalized.shape[0])
    row.embedded_model = model


def derive_title(content: str, *, max_len: int = 80) -> str:
    first_line = next((line for line in content.splitlines() if line.strip()), "captura")
    title = first_line.strip()
    return title if len(title) <= max_len else f"{title[: max_len - 1]}…"


def clean_tags(tags: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for tag in tags:
        normalized = tag.strip().lower()
        if normalized:
            seen[normalized] = None
    return list(seen)


def log_event(
    session: Session,
    namespace: str,
    kind: str,
    subject_id: str | None,
    payload: dict[str, Any],
) -> None:
    session.add(Event(namespace=namespace, kind=kind, subject_id=subject_id, payload=payload))


def iter_layers(layers: Iterable[str] | None) -> tuple[Layer, ...]:
    if not layers:
        return (Layer.RAW, Layer.INSIGHT)
    valid = tuple(Layer(value) for value in layers if value in tuple(Layer))
    return valid or (Layer.RAW, Layer.INSIGHT)
