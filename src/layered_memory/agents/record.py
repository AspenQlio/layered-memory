from __future__ import annotations

from datetime import datetime
from typing import Any

from layered_memory.domain import HandoffStatus
from layered_memory.memory.record import as_utc
from layered_memory.store.models import Handoff


class HandoffRecord:
    __slots__ = (
        "claimed_at",
        "claimed_by",
        "context",
        "created_at",
        "id",
        "namespace",
        "reason",
        "resolution",
        "resolved_at",
        "resolved_by",
        "session_id",
        "status",
        "updated_at",
    )

    def __init__(self, row: Handoff) -> None:
        self.id = row.id
        self.namespace = row.namespace
        self.session_id = row.session_id
        self.status = HandoffStatus(row.status)
        self.reason = row.reason
        self.context = dict(row.context or {})
        self.resolution = row.resolution
        self.claimed_by = row.claimed_by
        self.resolved_by = row.resolved_by
        self.created_at: datetime | None = as_utc(row.created_at)
        self.updated_at: datetime | None = as_utc(row.updated_at)
        self.claimed_at: datetime | None = as_utc(row.claimed_at)
        self.resolved_at: datetime | None = as_utc(row.resolved_at)

    @property
    def pending(self) -> bool:
        return not self.status.is_terminal

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "session_id": self.session_id,
            "status": str(self.status),
            "reason": self.reason,
            "context": self.context,
            "resolution": self.resolution,
            "claimed_by": self.claimed_by,
            "resolved_by": self.resolved_by,
            "created_at": self.created_at,
            "claimed_at": self.claimed_at,
            "resolved_at": self.resolved_at,
            "pending": self.pending,
        }

    def __repr__(self) -> str:
        return f"HandoffRecord(id={self.id!r}, status={str(self.status)!r}, reason={self.reason!r})"
