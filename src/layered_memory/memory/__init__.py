"""Dominio de memoria: records, servicio de capas y errores de dominio."""

from layered_memory.memory.record import MemoryRecord
from layered_memory.memory.service import MemoryNotFoundError, MemoryService

__all__ = ["MemoryNotFoundError", "MemoryRecord", "MemoryService"]
