from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class NewEvent(BaseModel):
    """An event that has not yet been assigned a stream sequence."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    type: str = Field(min_length=1, max_length=128)
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StoredEvent(NewEvent):
    """An immutable fact committed to one run's event stream."""

    run_id: UUID
    sequence: int = Field(ge=1)

