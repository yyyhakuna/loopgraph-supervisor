from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.domain.events import NewEvent
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


class PassingHarness:
    name = "passing"
    capabilities = HarnessCapabilities(stream_events=True)

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        await emit(HarnessEvent(type="agent.message", data={"ok": True}))
        return ExecutionResult(session_id="session", output={"ok": True})


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1, passed=True)


async def make_engine(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    registry = HarnessRegistry()
    registry.register(PassingHarness())
    engine = SupervisorEngine(
        store, registry, {"pass": CallableGrader("pass", pass_grader)}
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Complete safely",
            harness_id="passing",
            agent_bundle=AgentBundle(name="worker", system_prompt="Work."),
            grader_ids=("pass",),
        )
    )
    return database, store, engine, run_id


@pytest.mark.asyncio
async def test_run_can_pause_at_a_boundary_and_resume_without_losing_state(tmp_path) -> None:
    database, _, engine, run_id = await make_engine(tmp_path)

    paused = await engine.pause(run_id, actor="operator", reason="Inspect configuration")
    still_paused = await engine.run_until_blocked(run_id)
    resumed = await engine.resume(run_id, actor="operator", reason="Configuration approved")
    completed = await engine.run_until_blocked(run_id)
    await database.dispose()

    assert paused.status == RunStatus.PAUSED
    assert still_paused.status == RunStatus.PAUSED
    assert resumed.status == RunStatus.RETRY_SCHEDULED
    assert completed.status == RunStatus.SUCCEEDED
    assert completed.attempt == 1


@pytest.mark.asyncio
async def test_incomplete_running_stream_is_explicitly_recovered_after_restart(tmp_path) -> None:
    database, store, engine, run_id = await make_engine(tmp_path)
    version = await store.version(run_id)
    await store.append(
        run_id,
        expected_version=version,
        events=(NewEvent(type="run.started", data={"attempt": 1}),),
    )

    recovered = await engine.recover_incomplete(run_id, actor="system-recovery")
    events = await store.load(run_id)
    await database.dispose()

    assert recovered.status == RunStatus.RETRY_SCHEDULED
    assert recovered.attempt == 1
    assert [event.type for event in events][-2:] == ["run.recovered", "run.retry_scheduled"]


@pytest.mark.asyncio
async def test_waiting_human_cannot_be_bypassed_with_pause_and_resume(tmp_path) -> None:
    database, store, engine, run_id = await make_engine(tmp_path)
    version = await store.version(run_id)
    await store.append(
        run_id,
        expected_version=version,
        events=(NewEvent(type="run.waiting_human", data={"reason": "approval"}),),
    )

    with pytest.raises(ValueError, match="cannot be paused"):
        await engine.pause(run_id, actor="operator", reason="bypass")

    snapshot = await engine.get_run(run_id)
    await database.dispose()
    assert snapshot.status == RunStatus.WAITING_HUMAN
