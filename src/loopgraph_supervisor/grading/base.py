from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.domain.models import EvaluationResult


class EvaluationContext(BaseModel):
    """Framework-neutral material made available to a grader."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    attempt: int = Field(ge=1)
    goal: str = Field(min_length=1)
    output: dict[str, Any]
    trajectory: tuple[dict[str, Any], ...] = ()
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Grader(Protocol):
    @property
    def id(self) -> str: ...

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult: ...

