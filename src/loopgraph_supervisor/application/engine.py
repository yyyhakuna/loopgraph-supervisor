from __future__ import annotations

import asyncio
from typing import Literal
from uuid import UUID, uuid4

from loopgraph_supervisor.application.observer import (
    ObservationContext,
    ObservationPolicy,
    SupervisorActionHandler,
    SupervisorAdvisor,
    SupervisorDirective,
    SupervisorDirectiveAction,
)
from loopgraph_supervisor.domain.events import NewEvent
from loopgraph_supervisor.domain.memory import MemoryContextPolicy
from loopgraph_supervisor.domain.models import EvaluationResult, Hint, HintPriority
from loopgraph_supervisor.domain.runs import RunDefinition, RunSnapshot, RunStatus
from loopgraph_supervisor.grading.base import EvaluationContext, Grader
from loopgraph_supervisor.harness.base import ExecutionRequest, HarnessEvent
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore
from loopgraph_supervisor.infrastructure.memory_store import SqlMemoryStore
from loopgraph_supervisor.security.redaction import RedactionPolicy


class SupervisorEngine:
    """Deterministic controller for execution, evaluation and intervention loops."""

    def __init__(
        self,
        event_store: SqlEventStore,
        harnesses: HarnessRegistry,
        graders: dict[str, Grader],
        *,
        advisor: SupervisorAdvisor | None = None,
        observation_policy: ObservationPolicy | None = None,
        action_handler: SupervisorActionHandler | None = None,
        redaction_policy: RedactionPolicy | None = None,
        memory_store: SqlMemoryStore | None = None,
    ) -> None:
        self.events = event_store
        self.harnesses = harnesses
        self.graders = graders
        self.advisor = advisor
        self.observation_policy = observation_policy or ObservationPolicy()
        self.action_handler = action_handler
        self.redaction_policy = redaction_policy or RedactionPolicy()
        self.memory_store = memory_store

    def set_action_handler(self, handler: SupervisorActionHandler) -> None:
        self.action_handler = handler

    async def create_run(
        self, definition: RunDefinition, *, run_id: UUID | None = None
    ) -> UUID:
        self.harnesses.get(definition.harness_id)
        missing_graders = set(definition.grader_ids).difference(self.graders)
        if missing_graders:
            missing = ", ".join(sorted(missing_graders))
            raise KeyError(f"Graders are not registered: {missing}")

        run_id = run_id or uuid4()
        await self._append(
            run_id,
            NewEvent(
                type="run.created",
                data={"definition": definition.model_dump(mode="json", by_alias=True)},
            ),
        )
        return run_id

    async def append_control_event(self, run_id: UUID, event: NewEvent) -> None:
        """Append a typed control-plane fact owned by a higher-level coordinator."""
        await self.get_run(run_id)
        await self._append(run_id, event)

    async def get_run(self, run_id: UUID) -> RunSnapshot:
        events = await self.events.load(run_id)
        if not events:
            raise KeyError(f"Run {run_id} does not exist")
        return RunSnapshot.replay(events)

    async def run_until_blocked(self, run_id: UUID) -> RunSnapshot:
        while True:
            snapshot = await self.get_run(run_id)
            if snapshot.status not in {RunStatus.CREATED, RunStatus.RETRY_SCHEDULED}:
                return snapshot
            await self._execute_attempt(snapshot)

    async def _execute_attempt(self, snapshot: RunSnapshot) -> None:
        definition = snapshot.definition
        attempt = snapshot.attempt + 1
        harness = self.harnesses.get(definition.harness_id)
        inline_delivery = harness.capabilities.inject_context
        await self._append(snapshot.run_id, NewEvent(type="run.started", data={"attempt": attempt}))

        trajectory: list[dict[str, object]] = []
        observed_events: list[HarnessEvent] = []
        published_hint_keys = {hint.deduplication_key for hint in snapshot.hints}
        attempt_steps = 0

        async def emit(event: HarnessEvent) -> tuple[object, ...]:
            nonlocal attempt_steps
            event = event.model_copy(
                update={"data": self.redaction_policy.redact(event.data)}
            )
            trajectory.append(event.model_dump(mode="json"))
            observed_events.append(event)
            await self._append(
                snapshot.run_id,
                NewEvent(type="agent.event", data={"event": event.model_dump(mode="json")}),
            )
            if event.type in {"agent.step", "step/start", "dsh.step/start"}:
                attempt_steps += 1
                if attempt_steps > definition.supervisor_policy.max_steps_per_attempt:
                    raise BudgetExceededError(
                        "steps",
                        observed=attempt_steps,
                        limit=definition.supervisor_policy.max_steps_per_attempt,
                    )
            if self.advisor is None or not self.observation_policy.should_review(
                observed_events, event
            ):
                return ()
            context = ObservationContext(
                run_id=snapshot.run_id,
                attempt=attempt,
                goal=definition.goal,
                agent_bundle=definition.agent_bundle,
                agent_version_id=definition.agent_version_id,
                recent_events=tuple(
                    observed_events[-self.observation_policy.event_window :]
                ),
            )
            directive = await self.advisor.review(context)
            await self._append(
                snapshot.run_id,
                NewEvent(
                    type="supervisor.reviewed",
                    data={"directive": directive.model_dump(mode="json")},
                ),
            )
            hints = tuple(
                Hint(
                    run_id=snapshot.run_id,
                    target=draft.target,
                    priority=draft.priority,
                    instruction=draft.instruction,
                    reason=draft.reason,
                    evidence=draft.evidence,
                    created_at_step=attempt if inline_delivery else attempt + 1,
                    expires_after_steps=draft.expires_after_steps,
                    deduplication_key=draft.deduplication_key,
                )
                for draft in directive.hints
                if draft.deduplication_key not in published_hint_keys
            )
            if hints:
                published_hint_keys.update(hint.deduplication_key for hint in hints)
                await self._append(
                    snapshot.run_id,
                    *(
                        NewEvent(
                            type="hint.published",
                            data={
                                "hint": hint.model_dump(mode="json"),
                                "inline": inline_delivery,
                            },
                        )
                        for hint in hints
                    ),
                )
            if directive.action not in {
                SupervisorDirectiveAction.CONTINUE,
                SupervisorDirectiveAction.INJECT_HINT,
            }:
                await self._append(
                    snapshot.run_id,
                    NewEvent(
                        type="supervisor.action_requested",
                        data={"directive": directive.model_dump(mode="json")},
                    ),
                )
            if directive.action in {
                SupervisorDirectiveAction.SPAWN_SUBAGENT,
                SupervisorDirectiveAction.PROPOSE_MUTATION,
            } and self.action_handler is not None:
                try:
                    outcome = await self.action_handler.handle(context, directive)
                    outcome_data = (
                        outcome.model_dump(mode="json")
                        if hasattr(outcome, "model_dump")
                        else outcome
                    )
                    await self._append(
                        snapshot.run_id,
                        NewEvent(
                            type="supervisor.action_completed",
                            data={"action": directive.action.value, "outcome": outcome_data},
                        ),
                    )
                except Exception as error:
                    await self._append(
                        snapshot.run_id,
                        NewEvent(
                            type="supervisor.action_failed",
                            data={
                                "action": directive.action.value,
                                "error": self._safe_error(error),
                            },
                        ),
                    )
                    raise SupervisorControlSignal(
                        SupervisorDirective(
                            action=SupervisorDirectiveAction.REQUEST_HUMAN,
                            rationale=(
                                f"Supervisor action {directive.action.value} failed: "
                                f"{type(error).__name__}"
                            ),
                        )
                    ) from error
            if directive.action in {
                SupervisorDirectiveAction.PAUSE,
                SupervisorDirectiveAction.REQUEST_HUMAN,
                SupervisorDirectiveAction.ABORT,
            }:
                raise SupervisorControlSignal(directive)
            return hints if inline_delivery else ()

        active_hints = tuple(
            hint for hint in snapshot.hints if hint.is_active(current_step=attempt)
        )
        try:
            memories = await self._load_memories(definition)
            request = ExecutionRequest(
                run_id=snapshot.run_id,
                execution_id=f"{snapshot.run_id}:{attempt}",
                attempt=attempt,
                goal=definition.goal,
                agent_bundle=definition.agent_bundle,
                hints=active_hints,
                memories=memories,
                timeout_seconds=definition.supervisor_policy.attempt_timeout_seconds,
                max_steps=definition.supervisor_policy.max_steps_per_attempt,
            )
            result = await asyncio.wait_for(
                harness.execute(request, emit),
                timeout=definition.supervisor_policy.attempt_timeout_seconds,
            )
        except SupervisorControlSignal as signal:
            await self._handle_supervisor_control(snapshot.run_id, attempt, signal.directive)
            return
        except BudgetExceededError as error:
            await self._handle_budget_exhausted(snapshot.run_id, attempt, error)
            return
        except Exception as error:
            await self._handle_execution_error(snapshot.run_id, attempt, error)
            return

        await self._append(
            snapshot.run_id,
            NewEvent(
                type="agent.execution.completed",
                data={
                    "attempt": attempt,
                    "session_id": result.session_id,
                    "output": result.output,
                    "usage": result.usage,
                    "artifacts": result.artifacts,
                    "checkpoint": result.checkpoint,
                },
            ),
        )
        token_total = snapshot.total_tokens + int(result.usage.get("tokens", 0))
        cost_total = snapshot.total_cost + float(result.usage.get("cost", 0.0))
        policy = definition.supervisor_policy
        if policy.max_tokens is not None and token_total > policy.max_tokens:
            await self._handle_budget_exhausted(
                snapshot.run_id,
                attempt,
                BudgetExceededError("tokens", observed=token_total, limit=policy.max_tokens),
            )
            return
        if policy.max_cost is not None and cost_total > policy.max_cost:
            await self._handle_budget_exhausted(
                snapshot.run_id,
                attempt,
                BudgetExceededError("cost", observed=cost_total, limit=policy.max_cost),
            )
            return
        await self._evaluate(snapshot.run_id, definition, attempt, result.output, trajectory)

    async def _load_memories(
        self, definition: RunDefinition
    ) -> tuple[dict[str, object], ...]:
        policy = MemoryContextPolicy.model_validate(definition.agent_bundle.memory_config)
        if not policy.enabled:
            return ()
        if self.memory_store is None:
            raise RuntimeError("Memory context is enabled but no MemoryStore is configured")
        if definition.agent_version_id is None:
            raise ValueError("Memory context requires a versioned Agent Bundle")
        records = await self.memory_store.for_context(
            definition.agent_bundle.name,
            definition.agent_version_id,
            include_experimental=policy.include_experimental,
            kind=policy.kind,
            required_tags=policy.required_tags,
            limit=policy.limit,
        )
        return tuple(
            {
                "id": str(record.id),
                "kind": record.kind.value,
                "content": record.content,
                "tags": list(record.tags),
            }
            for record in records
        )

    async def _evaluate(
        self,
        run_id: UUID,
        definition: RunDefinition,
        attempt: int,
        output: dict[str, object],
        trajectory: list[dict[str, object]],
    ) -> None:
        await self._append(run_id, NewEvent(type="evaluation.started", data={"attempt": attempt}))
        context = EvaluationContext(
            run_id=run_id,
            attempt=attempt,
            goal=definition.goal,
            output=output,
            trajectory=tuple(trajectory),
        )
        results = await asyncio.gather(
            *(
                self._safe_grade(self.graders[grader_id], context)
                for grader_id in definition.grader_ids
            )
        )
        decision = definition.grading_policy.aggregate(tuple(results))
        await self._append(
            run_id,
            NewEvent(
                type="evaluation.completed",
                data=decision.model_dump(mode="json"),
            ),
        )
        if decision.passed:
            await self._append(
                run_id,
                NewEvent(type="run.succeeded", data={"human_override": False}),
            )
            return

        retryable = any(result.retryable for result in results)
        if retryable and attempt <= definition.supervisor_policy.max_retries:
            feedback = tuple(message for result in results for message in result.feedback)
            instruction = "\n".join(feedback) or (
                "Review the failed evaluation and use a new approach."
            )
            hint = Hint(
                run_id=run_id,
                target="executor",
                priority=HintPriority.WARNING,
                instruction=instruction,
                reason=f"Attempt {attempt} scored {decision.score:.3f}",
                evidence=tuple(
                    evidence for result in results for evidence in result.evidence
                ),
                created_at_step=attempt + 1,
                expires_after_steps=1,
                deduplication_key=f"evaluation-attempt-{attempt}",
            )
            await self._append(
                run_id,
                NewEvent(type="hint.published", data={"hint": hint.model_dump(mode="json")}),
                NewEvent(type="run.retry_scheduled", data={"next_attempt": attempt + 1}),
            )
        elif definition.supervisor_policy.require_human_on_exhaustion:
            await self._append(
                run_id,
                NewEvent(
                    type="run.waiting_human",
                    data={"reason": "evaluation_failed", "attempt": attempt},
                ),
            )
        else:
            await self._append(
                run_id,
                NewEvent(type="run.failed", data={"reason": "evaluation_failed"}),
            )

    async def _safe_grade(
        self, grader: Grader, context: EvaluationContext
    ) -> EvaluationResult:
        try:
            return await grader.evaluate(context)
        except Exception as error:
            await self._append(
                context.run_id,
                NewEvent(
                    type="grader.failed",
                    data={"grader_id": grader.id, "error": self._safe_error(error)},
                ),
            )
            return EvaluationResult(
                grader_id=grader.id,
                score=0.0,
                passed=False,
                confidence=0.0,
                feedback=(f"Grader failed: {type(error).__name__}",),
                retryable=False,
            )

    async def _handle_execution_error(self, run_id: UUID, attempt: int, error: Exception) -> None:
        await self._append(
            run_id,
            NewEvent(
                type="agent.execution.failed",
                data={"attempt": attempt, "error": self._safe_error(error)},
            ),
            NewEvent(type="run.waiting_human", data={"reason": "execution_failed"}),
        )

    async def _handle_budget_exhausted(
        self, run_id: UUID, attempt: int, error: BudgetExceededError
    ) -> None:
        await self._append(
            run_id,
            NewEvent(
                type="budget.exhausted",
                data={
                    "attempt": attempt,
                    "resource": error.resource,
                    "observed": error.observed,
                    "limit": error.limit,
                },
            ),
            NewEvent(
                type="run.waiting_human",
                data={"reason": "budget_exhausted", "resource": error.resource},
            ),
        )

    async def _handle_supervisor_control(
        self,
        run_id: UUID,
        attempt: int,
        directive: SupervisorDirective,
    ) -> None:
        data = {
            "attempt": attempt,
            "reason": directive.rationale,
            "source": "supervisor_agent",
        }
        if directive.action == SupervisorDirectiveAction.PAUSE:
            event = NewEvent(type="run.paused", data=data)
        elif directive.action == SupervisorDirectiveAction.REQUEST_HUMAN:
            event = NewEvent(type="run.waiting_human", data=data)
        else:
            event = NewEvent(type="run.failed", data=data)
        await self._append(run_id, event)

    async def resolve_human(
        self,
        run_id: UUID,
        *,
        decision: Literal["approve", "reject", "retry"],
        actor: str,
        reason: str,
    ) -> RunSnapshot:
        snapshot = await self.get_run(run_id)
        if snapshot.status != RunStatus.WAITING_HUMAN:
            raise ValueError(f"Run {run_id} is not waiting for human input")
        events = [
            NewEvent(
                type="hitl.resolved",
                data={"decision": decision, "actor": actor, "reason": reason},
            )
        ]
        if decision == "approve":
            events.append(NewEvent(type="run.succeeded", data={"human_override": True}))
        elif decision == "reject":
            events.append(NewEvent(type="run.failed", data={"reason": reason}))
        else:
            events.append(
                NewEvent(type="run.retry_scheduled", data={"next_attempt": snapshot.attempt + 1})
            )
        await self._append(run_id, *events)
        return await self.get_run(run_id)

    async def pause(self, run_id: UUID, *, actor: str, reason: str) -> RunSnapshot:
        snapshot = await self.get_run(run_id)
        if snapshot.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.PAUSED,
            RunStatus.WAITING_HUMAN,
        }:
            raise ValueError(f"Run {run_id} cannot be paused from {snapshot.status.value}")
        if snapshot.status in {RunStatus.RUNNING, RunStatus.EVALUATING}:
            raise ValueError(
                "This harness execution is active; use a pause-capable adapter "
                "or wait for a boundary"
            )
        await self._append(
            run_id,
            NewEvent(
                type="run.pause_requested",
                data={"actor": actor, "reason": reason, "from_status": snapshot.status.value},
            ),
            NewEvent(
                type="run.paused",
                data={"actor": actor, "reason": reason, "from_status": snapshot.status.value},
            ),
        )
        return await self.get_run(run_id)

    async def resume(self, run_id: UUID, *, actor: str, reason: str) -> RunSnapshot:
        snapshot = await self.get_run(run_id)
        if snapshot.status != RunStatus.PAUSED:
            raise ValueError(f"Run {run_id} is not paused")
        await self._append(
            run_id,
            NewEvent(
                type="run.resumed",
                data={"actor": actor, "reason": reason, "next_attempt": snapshot.attempt + 1},
            ),
        )
        return await self.get_run(run_id)

    async def recover_incomplete(self, run_id: UUID, *, actor: str) -> RunSnapshot:
        snapshot = await self.get_run(run_id)
        if snapshot.status not in {RunStatus.RUNNING, RunStatus.EVALUATING}:
            return snapshot
        await self._append(
            run_id,
            NewEvent(
                type="run.recovered",
                data={
                    "actor": actor,
                    "from_status": snapshot.status.value,
                    "prior_attempt": snapshot.attempt,
                    "strategy": "new_attempt",
                    "side_effect_warning": True,
                },
            ),
            NewEvent(
                type="run.retry_scheduled",
                data={"next_attempt": snapshot.attempt + 1, "recovered": True},
            ),
        )
        return await self.get_run(run_id)

    async def record_internal_failure(self, run_id: UUID, error: BaseException) -> None:
        """Best-effort durable record for failures outside the normal state machine."""
        snapshot = await self.get_run(run_id)
        if snapshot.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return
        safe_error = {
            "type": type(error).__name__,
            "message": str(error)[:2000],
        }
        await self._append(
            run_id,
            NewEvent(type="run.internal_failed", data={"error": safe_error}),
            NewEvent(type="run.waiting_human", data={"reason": "internal_failure"}),
        )

    async def _append(self, run_id: UUID, *events: NewEvent) -> None:
        version = await self.events.version(run_id)
        sanitized = tuple(
            event.model_copy(
                update={"data": self.redaction_policy.redact(event.data)}
            )
            for event in events
        )
        await self.events.append(
            run_id, expected_version=version, events=sanitized
        )

    @staticmethod
    def _safe_error(error: Exception) -> dict[str, str]:
        return {"type": type(error).__name__, "message": str(error)[:2000]}


class BudgetExceededError(RuntimeError):
    def __init__(self, resource: str, *, observed: float, limit: float) -> None:
        self.resource = resource
        self.observed = observed
        self.limit = limit
        super().__init__(f"{resource} budget exceeded: observed={observed}, limit={limit}")


class SupervisorControlSignal(RuntimeError):
    def __init__(self, directive: SupervisorDirective) -> None:
        self.directive = directive
        super().__init__(directive.rationale)
