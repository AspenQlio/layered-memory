"""Motor, sesion y ciclo de vida de la base de datos."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from layered_memory.store.models import Base


def build_engine(url: str, *, echo: bool = False) -> Engine:
    """Crea el motor SQLAlchemy con los ajustes correctos segun el motor.

    SQLite en memoria usa un pool estatico para que todas las sesiones del
    proceso vean la misma base; sin esto cada conexion abriria su propio
    esquema vacio y los tests serian intermitentes.
    """
    kwargs: dict[str, Any] = {"echo": echo, "future": True}

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url in {"sqlite://", "sqlite:///"}:
            kwargs["poolclass"] = StaticPool

    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Fabrica de sesiones ligadas al motor."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Sesion transaccional: commit al salir limpio, rollback al fallar."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine) -> None:
    """Crea las tablas que faltan. Es idempotente."""
    Base.metadata.create_all(engine)
