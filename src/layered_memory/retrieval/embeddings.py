"""Embedders intercambiables.

Dos implementaciones tras un mismo ``Protocol``:

* :class:`HashingEmbedder` — determinista, offline, sin dependencias. No es
  semantico; es la linea base lexica que permite correr tests, CI y el demo sin
  red. Sirve como referencia contra la cual medir cualquier embedder real.
* :class:`OpenAICompatEmbedder` — habla contra cualquier endpoint que implemente
  ``POST /v1/embeddings`` (Ollama, llama.cpp, vLLM, LM Studio, OpenAI).

Todos los embedders devuelven vectores L2-normalizados, de modo que el producto
esunto es la similitud coseno.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, assert_never, runtime_checkable

import httpx2
import numpy as np

if TYPE_CHECKING:
    from layered_memory.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Contrato minimo que necesita el indice vectorial."""

    dim: int
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Devuelve una matriz ``(len(texts), dim)`` de vectores normalizados."""
        ...


class EmbeddingResponseError(ValueError):
    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        super().__init__(f"el endpoint devolvio {received} vectores para {expected} textos")

    def close(self) -> None:
        """Libera los recursos que mantiene el embedder."""
        ...


def tokenize(text: str) -> list[str]:
    """Minusculas, solo alfanumericos y sin tildes.

    Plegar por NFKD y descartar los diacriticos mantiene "configuracion" y
    "configuración" en el mismo token, que es justo el caso que rompe un
    baseline ingenuo en español.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _TOKEN_RE.findall(without_marks)


class HashingEmbedder:
    """Embedder por hashing de tokens con peso sublineal (signed hashing trick).

    Es una representacion dispersa de bolsa de palabras proyectada a ``dim``
    dimensiones. Dos textos con vocabulario compartido tienen coseno alto; dos
    textos con el mismo significado pero palabras distintas, no. Esa limitacion
    es deliberada y esta medida en ``docs/evaluation.md``.
    """

    def __init__(self, dim: int = 512, model: str = "hashing-v1") -> None:
        self.dim = dim
        self.model = model

    def _vector_for(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            # Signed hashing trick: reduce las colisiones systematicas.
            sign = 1.0 if digest[4] & 1 else -1.0
            # 1 + log(tf) evita que la repeticion de un token domine el vector.
            vec[bucket] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            # Texto sin tokens utilizables (p.ej. solo puntuacion): vector nulo explicito.
            return vec
        return vec / norm

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector_for(t) for t in texts])

    def close(self) -> None:
        pass


class OpenAICompatEmbedder:
    """Cliente del endpoint ``POST {base_url}/embeddings``."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "not-needed",
        timeout: float = 30.0,
        batch_size: int = 32,
        dim: int = 512,
    ) -> None:
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._client = httpx2.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.post(
            "/embeddings", json={"model": self.model, "input": list(texts)}
        )
        response.raise_for_status()
        payload = response.json()
        items = sorted(payload["data"], key=lambda d: d.get("index", 0))
        vectors = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if vectors.shape[0] != len(texts):
            raise EmbeddingResponseError(len(texts), vectors.shape[0])
        if vectors.shape[1] != self.dim:
            # El endpoint manda la verdad: adoptamos su dimension para no
            # escribir vectores que despues no podemos comparar.
            self.dim = int(vectors.shape[1])
        return _l2_normalize(vectors)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks = [
            self._embed_batch(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(chunks)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatEmbedder:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_embedder(settings: Settings) -> Embedder:
    """Construye el embedder que pide la configuracion.

    En modo hibrido persiste la señal semantica; el indice calcula la rama
    lexical durante cada busqueda sin duplicar vectores en la base.
    """
    match settings.embedding_backend:
        case "hash":
            return HashingEmbedder(dim=settings.embedding_dim)
        case "openai" | "hybrid":
            return OpenAICompatEmbedder(
                model=settings.embedding_model,
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                timeout=settings.embedding_timeout,
                batch_size=settings.embedding_batch_size,
                dim=settings.embedding_dim,
            )
        case unreachable:
            assert_never(unreachable)
