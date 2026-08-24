from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.observer import (
    CallbackSupervisorAdvisor,
    HintDraft,
    ObservationContext,
    ObservationPolicy,
    SupervisorDirective,
    SupervisorDirectiveAction,
)
from loopgraph_supervisor.domain.models import AgentBundle, EvaluationResult
from loopgraph_supervisor.domain.runs import RunDefinition, RunStatus
from loopgraph_supervisor.grading.base import EvaluationContext
from loopgraph_supervisor.grading.callable import CallableGrader
from loopgraph_supervisor.harness.base import (
    ExecutionRequest,
    ExecutionResult,
    HarnessCapabilities,
    HarnessEvent,
)
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore


class HintAwareHarness:
    name = "hint-aware"
    capabilities = HarnessCapabilities(stream_events=True, inject_context=True)

    def __init__(self) -> None:
        self.inline_hints: list[object] = []

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        self.inline_hints.extend(
            await emit(
                HarnessEvent(
                    type="tool/result",
                    data={"name": "fetch", "is_error": True, "error": "unauthorized"},
                )
            )
        )
        self.inline_hints.extend(
            await emit(
                HarnessEvent(
                    type="tool/result",
                    data={"name": "fetch", "is_error": True, "error": "unauthorized"},
                )
            )
        )
        return ExecutionResult(session_id="hint-session", output={"ok": True})


class BoundaryHintHarness:
    name = "boundary-hints"
    capabilities = HarnessCapabilities(stream_events=True, inject_context=False)

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        self.requests.append(request)
        await emit(
            HarnessEvent(
                type="tool/result",
                data={"name": "fetch", "is_error": True, "error": "unauthorized"},
            )
        )
        return ExecutionResult(
            session_id="boundary-session", output={"attempt": request.attempt}
        )


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1, passed=True)


async def repeated_error_advisor(context: ObservationContext) -> SupervisorDirective:
    assert len(context.recent_events) == 2
    return SupervisorDirective(
        action=SupervisorDirectiveAction.INJECT_HINT,
        rationale="The same tool failed twice.",
        hints=(
            HintDraft(
                instruction="Stop retrying fetch and inspect authentication.",
                reason="Repeated unauthorized response.",
                deduplication_key="fetch-auth",
            ),
        ),
    )


async def one_error_advisor(_: ObservationContext) -> SupervisorDirective:
    return SupervisorDirective(
        action=SupervisorDirectiveAction.INJECT_HINT,
        rationale="Fix the authentication failure on the next safe boundary.",
        hints=(
            HintDraft(
                instruction="Use the service credential before retrying fetch.",
                reason="The fetch call was unauthorized.",
                deduplication_key="fetch-credential",
            ),
        ),
    )


async def retry_once_grader(context: EvaluationContext) -> EvaluationResult:
    passed = context.attempt == 2
    return EvaluationResult(
        grader_id="retry-once",
        score=1.0 if passed else 0.0,
        passed=passed,
        retryable=not passed,
        feedback=("Apply the supervisor hint.",) if not passed else (),
    )


@pytest.mark.asyncio
async def test_realtime_supervisor_reviews_triggered_steps_and_returns_inline_hints(
    tmp_path,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'observer.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harness = HintAwareHarness()
    harnesses = HarnessRegistry()
    harnesses.register(harness)
    engine = SupervisorEngine(
        store,
        harnesses,
        {"pass": CallableGrader("pass", pass_grader)},
        advisor=CallbackSupervisorAdvisor(repeated_error_advisor),
        observation_policy=ObservationPolicy(after_tool_errors=2, event_window=10),
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Fetch protected data",
            harness_id="hint-aware",
            agent_bundle=AgentBundle(name="worker", system_prompt="Fetch carefully."),
            grader_ids=("pass",),
        )
    )

    snapshot = await engine.run_until_blocked(run_id)
    events = await store.load(run_id)
    await database.dispose()

    assert snapshot.status == RunStatus.SUCCEEDED
    assert len(harness.inline_hints) == 1
    assert harness.inline_hints[0].instruction.startswith("Stop retrying")
    assert [event.type for event in events].count("supervisor.reviewed") == 1
    assert [event.type for event in events].count("hint.published") == 1


@pytest.mark.asyncio
async def test_non_injecting_harness_receives_realtime_hint_on_next_attempt(
    tmp_path,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'boundary.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harness = BoundaryHintHarness()
    harnesses = HarnessRegistry()
    harnesses.register(harness)
    engine = SupervisorEngine(
        store,
        harnesses,
        {"retry-once": CallableGrader("retry-once", retry_once_grader)},
        advisor=CallbackSupervisorAdvisor(one_error_advisor),
        observation_policy=ObservationPolicy(
            after_tool_errors=1,
            event_window=10,
        ),
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Fetch protected data",
            harness_id="boundary-hints",
            agent_bundle=AgentBundle(name="worker", system_prompt="Fetch carefully."),
            grader_ids=("retry-once",),
        )
    )

    snapshot = await engine.run_until_blocked(run_id)
    events = await store.load(run_id)
    await database.dispose()

    assert snapshot.status == RunStatus.SUCCEEDED
    assert harness.requests[0].hints == ()
    assert [hint.instruction for hint in harness.requests[1].hints] == [
        "Use the service credential before retrying fetch.",
        "Apply the supervisor hint.",
    ]
    realtime_event = next(
        event
        for event in events
        if event.type == "hint.published" and event.data.get("inline") is False
    )
    assert realtime_event.data["hint"]["created_at_step"] == 2
