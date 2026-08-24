from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.observer import (
    CallbackSupervisorAdvisor,
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


class PausableBySupervisorHarness:
    name = "directive"
    capabilities = HarnessCapabilities(stream_events=True)

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        await emit(
            HarnessEvent(
                type="tool/result",
                data={"is_error": True, "error": "credentials unavailable"},
            )
        )
        raise AssertionError("The pause directive must stop this attempt")


async def unused_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="unused", score=1, passed=True)


async def pause_advisor(_: ObservationContext) -> SupervisorDirective:
    return SupervisorDirective(
        action=SupervisorDirectiveAction.PAUSE,
        rationale="Credentials require operator inspection.",
    )


class RecordingActionHandler:
    def __init__(self) -> None:
        self.actions: list[SupervisorDirectiveAction] = []

    async def handle(
        self, _: ObservationContext, directive: SupervisorDirective
    ) -> dict[str, str]:
        self.actions.append(directive.action)
        return {"child_run_id": "child-1"}


async def spawn_advisor(_: ObservationContext) -> SupervisorDirective:
    return SupervisorDirective(
        action=SupervisorDirectiveAction.SPAWN_SUBAGENT,
        rationale="A specialist should inspect credentials.",
        payload={"child": {"role": "investigator", "goal": "Inspect credentials"}},
    )


class SpawnThenCompleteHarness(PausableBySupervisorHarness):
    name = "spawn-directive"

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        await emit(
            HarnessEvent(
                type="tool/result", data={"is_error": True, "error": "unauthorized"}
            )
        )
        return ExecutionResult(session_id="spawn", output={"ok": True})


@pytest.mark.asyncio
async def test_supervisor_pause_directive_stops_attempt_at_observed_boundary(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'directive.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(PausableBySupervisorHarness())
    engine = SupervisorEngine(
        store,
        harnesses,
        {"unused": CallableGrader("unused", unused_grader)},
        advisor=CallbackSupervisorAdvisor(pause_advisor),
        observation_policy=ObservationPolicy(after_tool_errors=1),
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Use protected API",
            harness_id="directive",
            agent_bundle=AgentBundle(name="worker", system_prompt="Work."),
            grader_ids=("unused",),
        )
    )

    snapshot = await engine.run_until_blocked(run_id)
    events = await store.load(run_id)
    await database.dispose()

    assert snapshot.status == RunStatus.PAUSED
    assert "run.paused" in [event.type for event in events]
    assert "agent.execution.failed" not in [event.type for event in events]


@pytest.mark.asyncio
async def test_smart_action_directive_is_routed_and_audited(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'action.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(SpawnThenCompleteHarness())
    handler = RecordingActionHandler()
    engine = SupervisorEngine(
        store,
        harnesses,
        {"unused": CallableGrader("unused", unused_grader)},
        advisor=CallbackSupervisorAdvisor(spawn_advisor),
        observation_policy=ObservationPolicy(after_tool_errors=1),
        action_handler=handler,
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Use protected API",
            harness_id="spawn-directive",
            agent_bundle=AgentBundle(name="worker", system_prompt="Work."),
            grader_ids=("unused",),
        )
    )

    snapshot = await engine.run_until_blocked(run_id)
    events = await store.load(run_id)
    await database.dispose()

    assert snapshot.status == RunStatus.SUCCEEDED
    assert handler.actions == [SupervisorDirectiveAction.SPAWN_SUBAGENT]
    assert "supervisor.action_completed" in [event.type for event in events]
