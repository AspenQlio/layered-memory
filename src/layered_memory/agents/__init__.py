"""Interaccion con agentes: escalado a humano y lazo de aprendizaje."""

from layered_memory.agents.handoff import (
    HandoffConflictError,
    HandoffNotFoundError,
    HandoffQueue,
    HandoffRecord,
    resolutions,
)

__all__ = [
    "HandoffConflictError",
    "HandoffNotFoundError",
    "HandoffQueue",
    "HandoffRecord",
    "resolutions",
]
