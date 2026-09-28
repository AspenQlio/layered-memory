"""Enumeraciones compartidas del dominio.

Viven fuera de ``store`` y ``memory`` para que las capas de persistencia,
servicio y API compartan los mismos identificadores sin ciclos de importacion.
"""

from __future__ import annotations

from enum import StrEnum


class Layer(StrEnum):
    """Las tres capas de memoria y su direccion de promocion.

    ``RAW -> INSIGHT -> ARTIFACT``. Un elemento solo se promueve hacia
    adelante, y cada promocion queda registrada como una fila nueva con
    ``parent_id`` apuntando a su origen, de modo que siempre se puede volver
    al texto literal que produjo el conocimiento.
    """

    RAW = "raw"
    INSIGHT = "insight"
    ARTIFACT = "artifact"

    @property
    def next(self) -> Layer | None:
        """Capa destino de la promocion, o ``None`` si ya es terminal."""
        order: tuple[Layer, ...] = (Layer.RAW, Layer.INSIGHT, Layer.ARTIFACT)
        index = order.index(self)
        if index + 1 >= len(order):
            return None
        return order[index + 1]


class RawStatus(StrEnum):
    """Estado de una captura en la capa ``raw``."""

    PENDING = "pending"
    DISTILLED = "distilled"


class HandoffStatus(StrEnum):
    """Estado de una peticion de escalado a humano.

    ``QUEUED -> CLAIMED -> RESOLVED`` es el camino feliz. ``CANCELLED`` es
    terminal y puede alcanzarse desde ``QUEUED`` o ``CLAIMED``.
    """

    QUEUED = "queued"
    CLAIMED = "claimed"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (HandoffStatus.RESOLVED, HandoffStatus.CANCELLED)


#: Orden de las capas, util para ordenar resultados y para las metricas.
LAYER_ORDER: tuple[Layer, ...] = (Layer.RAW, Layer.INSIGHT, Layer.ARTIFACT)
