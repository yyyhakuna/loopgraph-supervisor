from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.domain.models import AgentBundle, EvaluationResult
from loopgraph_supervisor.domain.runs import RunDefinition
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


class SecretEmittingHarness:
    name = "secret"
    capabilities = HarnessCapabilities(stream_events=True)

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        await emit(
            HarnessEvent(
                type="http.request",
                data={
                    "headers": {"Authorization": "Bearer top-secret", "Accept": "json"},
                    "api_key": "sk-secret",
                    "token_count": 123,
                },
            )
        )
        return ExecutionResult(
            session_id="secret",
            output={"ok": True, "summary": "used Bearer final-secret-token"},
            artifacts={"api_key": "sk-output-secret"},
            checkpoint="checkpoint-sk-checkpoint-secret",
        )


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1, passed=True)


@pytest.mark.asyncio
async def test_sensitive_harness_fields_are_redacted_before_persistence(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'redaction.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(SecretEmittingHarness())
    engine = SupervisorEngine(
        store, harnesses, {"pass": CallableGrader("pass", pass_grader)}
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Call an API",
            harness_id="secret",
            agent_bundle=AgentBundle(name="worker", system_prompt="Call safely."),
            grader_ids=("pass",),
        )
    )
    await engine.run_until_blocked(run_id)
    events = await store.load(run_id)
    await database.dispose()

    trace = next(event for event in events if event.type == "agent.event")
    data = trace.data["event"]["data"]
    assert data["headers"]["Authorization"] == "[REDACTED]"
    assert data["headers"]["Accept"] == "json"
    assert data["api_key"] == "[REDACTED]"
    assert data["token_count"] == 123
    completed = next(
        event for event in events if event.type == "agent.execution.completed"
    )
    assert completed.data["output"]["summary"] == "used Bearer [REDACTED]"
    assert completed.data["artifacts"]["api_key"] == "[REDACTED]"
    assert "sk-checkpoint-secret" not in completed.data["checkpoint"]
