from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.domain.models import AgentBundle, EvaluationResult
from loopgraph_supervisor.domain.runs import RunDefinition, RunStatus, SupervisorPolicy
from loopgraph_supervisor.grading.base import EvaluationContext
from loopgraph_supervisor.grading.callable import CallableGrader
from loopgraph_supervisor.grading.policy import GradingPolicy
from loopgraph_supervisor.harness.base import (
    ExecutionRequest,
    ExecutionResult,
    HarnessCapabilities,
    HarnessEvent,
)
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore


class ScriptedHarness:
    name = "scripted"
    capabilities = HarnessCapabilities(stream_events=True, inject_context=True)

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.requests: list[ExecutionRequest] = []

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        self.requests.append(request)
        await emit(HarnessEvent(type="model.call.started", data={"attempt": request.attempt}))
        output = self.outputs[len(self.requests) - 1]
        await emit(HarnessEvent(type="agent.message", data=output))
        return ExecutionResult(
            session_id=f"session-{request.run_id}",
            output=output,
            usage={"tokens": 10},
        )


async def correctness_grader(context: EvaluationContext) -> EvaluationResult:
    passed = context.output.get("answer") == 42
    return EvaluationResult(
        grader_id="correctness",
        score=1.0 if passed else 0.0,
        passed=passed,
        feedback=() if passed else ("The answer must be 42.",),
        retryable=True,
    )


@pytest.fixture
async def engine_parts(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    graders = {"correctness": CallableGrader("correctness", correctness_grader)}
    yield database, store, harnesses, graders
    await database.dispose()


@pytest.mark.asyncio
async def test_supervisor_retries_with_feedback_until_business_score_passes(engine_parts) -> None:
    _, store, harnesses, graders = engine_parts
    harness = ScriptedHarness([{"answer": 7}, {"answer": 42}])
    harnesses.register(harness)
    engine = SupervisorEngine(store, harnesses, graders)
    definition = RunDefinition(
        goal="Return the answer 42",
        harness_id="scripted",
        agent_bundle=AgentBundle(name="solver", system_prompt="Solve the task."),
        grader_ids=("correctness",),
        grading_policy=GradingPolicy(threshold=1.0),
        supervisor_policy=SupervisorPolicy(max_retries=2),
    )

    run_id = await engine.create_run(definition)
    snapshot = await engine.run_until_blocked(run_id)
    events = await store.load(run_id)

    assert snapshot.status == RunStatus.SUCCEEDED
    assert snapshot.attempt == 2
    assert len(harness.requests) == 2
    assert harness.requests[0].hints == ()
    assert harness.requests[1].hints[0].instruction == "The answer must be 42."
    assert "evaluation.completed" in [event.type for event in events]
    assert "hint.published" in [event.type for event in events]


@pytest.mark.asyncio
async def test_retry_exhaustion_waits_for_human_and_can_be_approved(engine_parts) -> None:
    _, store, harnesses, graders = engine_parts
    harnesses.register(ScriptedHarness([{"answer": 7}]))
    engine = SupervisorEngine(store, harnesses, graders)
    run_id = await engine.create_run(
        RunDefinition(
            goal="Return the answer 42",
            harness_id="scripted",
            agent_bundle=AgentBundle(name="solver", system_prompt="Solve the task."),
            grader_ids=("correctness",),
            grading_policy=GradingPolicy(threshold=1.0),
            supervisor_policy=SupervisorPolicy(max_retries=0, require_human_on_exhaustion=True),
        )
    )

    waiting = await engine.run_until_blocked(run_id)
    approved = await engine.resolve_human(
        run_id,
        decision="approve",
        actor="operator@example.com",
        reason="Accepted for the prototype.",
    )

    assert waiting.status == RunStatus.WAITING_HUMAN
    assert approved.status == RunStatus.SUCCEEDED
    assert approved.human_override


@pytest.mark.asyncio
async def test_unknown_harness_is_rejected_before_a_run_is_created(engine_parts) -> None:
    _, store, harnesses, graders = engine_parts
    engine = SupervisorEngine(store, harnesses, graders)
    definition = RunDefinition(
        goal="Do work",
        harness_id="missing",
        agent_bundle=AgentBundle(name="worker", system_prompt="Work."),
        grader_ids=("correctness",),
    )

    with pytest.raises(KeyError, match="missing"):
        await engine.create_run(definition)
