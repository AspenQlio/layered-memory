from __future__ import annotations


class MemoryNotFoundError(LookupError):
    pass


class EmptyMemoryContentError(ValueError):
    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"no se puede {operation} contenido vacio")
