from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EVALUATION = "evaluation"


class MemoryStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    APPROVED = "approved"
    REJECTED = "rejected"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    agent_name: str
    version_id: UUID
    kind: MemoryKind
    content: str
    status: MemoryStatus
    source_run_id: UUID | None
    tags: tuple[str, ...]
    metadata: dict[str, Any]
    created_at: datetime


class MemoryContextPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    include_experimental: bool = False
    kind: MemoryKind | None = None
    required_tags: tuple[str, ...] = ()
    limit: int = Field(default=20, ge=1, le=200)
