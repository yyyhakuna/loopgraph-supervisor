from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.domain.models import AgentBundle, HintPriority
from loopgraph_supervisor.harness.base import (
    ExecutionRequest,
    HarnessAdapter,
    HarnessEvent,
)


class SupervisorDirectiveAction(StrEnum):
    CONTINUE = "continue"
    INJECT_HINT = "inject_hint"
    PAUSE = "pause"
    REQUEST_HUMAN = "request_human"
    SPAWN_SUBAGENT = "spawn_subagent"
    PROPOSE_MUTATION = "propose_mutation"
    ABORT = "abort"


class HintDraft(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: str = "executor"
    priority: HintPriority = HintPriority.WARNING
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    expires_after_steps: int = Field(default=1, ge=1, le=100)
    deduplication_key: str = Field(min_length=1, max_length=256)


class SupervisorDirective(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: SupervisorDirectiveAction = SupervisorDirectiveAction.CONTINUE
    rationale: str = Field(min_length=1)
    hints: tuple[HintDraft, ...] = ()
    payload: dict[str, object] = Field(default_factory=dict)


class ObservationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    attempt: int = Field(ge=1)
    goal: str
    agent_bundle: AgentBundle
    agent_version_id: UUID | None = None
    recent_events: tuple[HarnessEvent, ...]


class SupervisorAdvisor(Protocol):
    async def review(self, context: ObservationContext) -> SupervisorDirective: ...


class SupervisorActionHandler(Protocol):
    async def handle(
        self, context: ObservationContext, directive: SupervisorDirective
    ) -> BaseModel | dict[str, object]: ...


class CallbackSupervisorAdvisor:
    def __init__(
        self,
        callback: Callable[[ObservationContext], Awaitable[SupervisorDirective]],
    ) -> None:
        self.callback = callback

    async def review(self, context: ObservationContext) -> SupervisorDirective:
        return await self.callback(context)


class SupervisorAdvisorError(RuntimeError):
    pass


class HarnessSupervisorAdvisor:
    """Runs a dedicated supervisor Agent through any HarnessAdapter."""

    def __init__(
        self,
        *,
        harness: HarnessAdapter,
        bundle: AgentBundle,
        timeout_seconds: float = 120.0,
        max_steps: int = 10,
    ) -> None:
        self.harness = harness
        self.bundle = bundle
        self.timeout_seconds = timeout_seconds
        self.max_steps = max_steps

    async def review(self, context: ObservationContext) -> SupervisorDirective:
        last_event = context.recent_events[-1]
        supervisor_run_id = uuid5(
            NAMESPACE_URL,
            (
                f"loopgraph-supervisor:{context.run_id}:{context.attempt}:"
                f"{last_event.type}:{last_event.occurred_at.isoformat()}"
            ),
        )

        async def discard(_: HarnessEvent) -> tuple[object, ...]:
            return ()

        request = ExecutionRequest(
            run_id=supervisor_run_id,
            execution_id=f"{context.run_id}:supervisor:{context.attempt}",
            attempt=1,
            goal=self._render_goal(context),
            agent_bundle=self.bundle,
            timeout_seconds=self.timeout_seconds,
            max_steps=self.max_steps,
        )
        result = await self.harness.execute(request, discard)
        raw = result.output.get("directive")
        if raw is None and isinstance(result.output.get("text"), str):
            try:
                raw = json.loads(result.output["text"])
            except json.JSONDecodeError as error:
                raise SupervisorAdvisorError(
                    "Supervisor Agent returned invalid directive JSON"
                ) from error
        try:
            return SupervisorDirective.model_validate(raw)
        except (TypeError, ValueError) as error:
            raise SupervisorAdvisorError(
                "Supervisor Agent returned an invalid directive"
            ) from error

    @staticmethod
    def _render_goal(context: ObservationContext) -> str:
        payload = context.model_dump(mode="json")
        return (
            "Review this executor trajectory against the business goal. Return exactly one "
            "SupervisorDirective JSON object. You may continue, inject_hint, pause, "
            "request_human, spawn_subagent, propose_mutation, or abort. Do not modify "
            "grader policy.\n\nObservation:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )


class ObservationPolicy(BaseModel):
    """Cheap deterministic triggers that gate expensive supervisor reviews."""

    model_config = ConfigDict(frozen=True)

    after_tool_errors: int | None = Field(default=2, ge=1)
    every_model_calls: int | None = Field(default=None, ge=1)
    on_agent_complete: bool = False
    event_window: int = Field(default=20, ge=1, le=1000)

    def should_review(self, events: list[HarnessEvent], current: HarnessEvent) -> bool:
        if self.after_tool_errors and self._is_tool_error(current):
            error_count = sum(self._is_tool_error(event) for event in events)
            if error_count % self.after_tool_errors == 0:
                return True
        if self.every_model_calls and self._is_model_completion(current):
            model_calls = sum(self._is_model_completion(event) for event in events)
            if model_calls % self.every_model_calls == 0:
                return True
        return self.on_agent_complete and current.type in {
            "agent.completed",
            "turn/end",
            "dsh.turn/end",
        }

    @staticmethod
    def _is_tool_error(event: HarnessEvent) -> bool:
        if "tool" not in event.type:
            return False
        return bool(
            event.data.get("is_error")
            or event.data.get("status") == "error"
            or event.data.get("error")
        )

    @staticmethod
    def _is_model_completion(event: HarnessEvent) -> bool:
        return event.type in {
            "model.call.completed",
            "assistant/message",
            "dsh.assistant/message",
        }
