from __future__ import annotations


class HandoffNotFoundError(LookupError):
    pass


class HandoffConflictError(RuntimeError):
    pass


class EmptyHandoffFieldError(ValueError):
    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"{field} no puede ir vacio")
