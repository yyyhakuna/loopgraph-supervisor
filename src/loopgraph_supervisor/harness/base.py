from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.domain.models import AgentBundle, Hint


class HarnessCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    stream_events: bool = False
    inject_context: bool = False
    pause_resume: bool = False
    checkpoints: bool = False
    subagents: bool = False
    workspace_snapshots: bool = False


class HarnessEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = Field(min_length=1, max_length=128)
    data: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    execution_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    goal: str
    agent_bundle: AgentBundle
    hints: tuple[Hint, ...] = ()
    memories: tuple[dict[str, Any], ...] = ()
    resume_token: str | None = None
    timeout_seconds: float = Field(gt=0)
    max_steps: int = Field(ge=1)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    output: dict[str, Any]
    usage: dict[str, float | int] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    checkpoint: str | None = None


EventSink = Callable[[HarnessEvent], Coroutine[Any, Any, tuple[object, ...]]]


class HarnessAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> HarnessCapabilities: ...

    async def execute(
        self, request: ExecutionRequest, emit: EventSink
    ) -> ExecutionResult: ...
