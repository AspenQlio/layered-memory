"""Configuracion del sistema, leida desde el entorno o desde un archivo ``.env``.

Todo tiene un valor por defecto sensato para que ``layered-memory`` funcione
out-of-the-box con SQLite y embeddings deterministas locales, sin red.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ajustes de la aplicacion.

    Variables de entorno con prefijo ``LAYERED_MEMORY_``. Por ejemplo
    ``LAYERED_MEMORY_DATABASE_URL`` o ``LAYERED_MEMORY_EMBEDDING_BACKEND``.
    """

    model_config = SettingsConfigDict(
        env_prefix="LAYERED_MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./layered_memory.db"

    embedding_backend: Literal["hash", "openai", "hybrid"] = Field(
        default="hash",
        description=(
            "'hash' usa un embedder determinista local y sin dependencias; "
            "'openai' habla contra cualquier endpoint compatible con la "
            "especificacion OpenAI (Ollama, llama.cpp, vLLM, LM Studio, OpenAI); "
            "'hybrid' fusiona esa señal semantica con hashing lexical mediante RRF."
        ),
    )
    embedding_dim: int = Field(default=512, ge=32, le=8192)
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "not-needed"
    embedding_timeout: float = Field(default=30.0, gt=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    hybrid_lexical_dim: int = Field(default=512, ge=32, le=8192)
    hybrid_rank_constant: int = Field(default=60, ge=1, le=1000)
    hybrid_semantic_weight: float = Field(default=3.0, gt=0, le=100.0)
    hybrid_semantic_head: int = Field(default=5, ge=0, le=100)

    retrieval_default_k: int = Field(default=5, ge=1, le=100)
    retrieval_min_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    worker_poll_seconds: float = Field(default=5.0, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve los ajustes cacheados del proceso."""
    return Settings()
