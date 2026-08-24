from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.domain.models import AgentBundle


class VersionStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class AgentVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    agent_name: str
    version: int = Field(ge=1)
    parent_id: UUID | None = None
    bundle: AgentBundle
    fingerprint: str
    status: VersionStatus
    created_at: datetime
    created_by: str
    change_summary: str
    evaluation_summary: dict[str, Any] = Field(default_factory=dict)


class ActivationHistoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    agent_name: str
    action: str
    from_version_id: UUID | None
    to_version_id: UUID
    actor: str
    reason: str
    created_at: datetime

