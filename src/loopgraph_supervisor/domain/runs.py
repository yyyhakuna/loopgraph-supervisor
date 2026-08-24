from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loopgraph_supervisor.domain.events import StoredEvent
from loopgraph_supervisor.domain.models import AgentBundle, Hint
from loopgraph_supervisor.grading.policy import GradingPolicy


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    EVALUATING = "evaluating"
    RETRY_SCHEDULED = "retry_scheduled"
    WAITING_HUMAN = "waiting_human"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SupervisorPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=2, ge=0, le=100)
    require_human_on_exhaustion: bool = True
    max_steps_per_attempt: int = Field(default=100, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost: float | None = Field(default=None, ge=0.0)
    attempt_timeout_seconds: float = Field(default=900.0, gt=0)
    max_subagents: int = Field(default=4, ge=0, le=100)
    max_subagent_depth: int = Field(default=2, ge=0, le=20)


class RunDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    goal: str = Field(min_length=1)
    harness_id: str = Field(min_length=1, max_length=128)
    agent_bundle: AgentBundle
    agent_version_id: UUID | None = None
    grader_ids: tuple[str, ...]
    grading_policy: GradingPolicy = Field(default_factory=GradingPolicy)
    supervisor_policy: SupervisorPolicy = Field(default_factory=SupervisorPolicy)
    parent_run_id: UUID | None = None
    team_id: UUID | None = None
    role: str = Field(default="executor", min_length=1, max_length=128)
    depth: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_graders(self) -> RunDefinition:
        if not self.grader_ids:
            raise ValueError("At least one grader is required")
        if len(set(self.grader_ids)) != len(self.grader_ids):
            raise ValueError("grader_ids must be unique")
        return self


class RunSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    definition: RunDefinition
    status: RunStatus
    version: int = Field(ge=1)
    attempt: int = Field(default=0, ge=0)
    last_output: dict[str, Any] | None = None
    last_score: float | None = Field(default=None, ge=0.0, le=1.0)
    hints: tuple[Hint, ...] = ()
    human_override: bool = False
    error: str | None = None
    total_steps: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)

    @classmethod
    def replay(cls, events: tuple[StoredEvent, ...]) -> RunSnapshot:
        if not events or events[0].type != "run.created":
            raise ValueError("A run stream must start with run.created")
        definition = RunDefinition.model_validate(events[0].data["definition"])
        state: dict[str, Any] = {
            "run_id": events[0].run_id,
            "definition": definition,
            "status": RunStatus.CREATED,
            "version": events[-1].sequence,
            "attempt": 0,
            "hints": [],
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        for event in events[1:]:
            match event.type:
                case "run.started":
                    state["status"] = RunStatus.RUNNING
                    state["attempt"] = event.data["attempt"]
                case "agent.execution.completed":
                    state["last_output"] = event.data["output"]
                    usage = event.data.get("usage", {})
                    state["total_tokens"] += int(usage.get("tokens", 0))
                    state["total_cost"] += float(usage.get("cost", 0.0))
                case "agent.event":
                    event_type = event.data.get("event", {}).get("type", "")
                    if event_type in {"agent.step", "step/start", "dsh.step/start"}:
                        state["total_steps"] += 1
                case "evaluation.started" | "evaluation.completed":
                    state["status"] = RunStatus.EVALUATING
                    if event.type == "evaluation.completed":
                        state["last_score"] = event.data["score"]
                case "hint.published":
                    state["hints"].append(Hint.model_validate(event.data["hint"]))
                case "run.retry_scheduled":
                    state["status"] = RunStatus.RETRY_SCHEDULED
                case "run.waiting_human":
                    state["status"] = RunStatus.WAITING_HUMAN
                case "run.paused":
                    state["status"] = RunStatus.PAUSED
                case "run.resumed" | "run.recovered":
                    state["status"] = RunStatus.RETRY_SCHEDULED
                case "run.succeeded":
                    state["status"] = RunStatus.SUCCEEDED
                    state["human_override"] = event.data.get("human_override", False)
                case "run.failed":
                    state["status"] = RunStatus.FAILED
                    state["error"] = event.data.get("reason")
                case "run.cancelled":
                    state["status"] = RunStatus.CANCELLED
        state["hints"] = tuple(state["hints"])
        return cls.model_validate(state)
