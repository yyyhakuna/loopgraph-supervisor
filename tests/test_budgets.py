from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.domain.models import AgentBundle, EvaluationResult
from loopgraph_supervisor.domain.runs import RunDefinition, RunStatus, SupervisorPolicy
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


class BudgetHarness:
    name = "budget"
    capabilities = HarnessCapabilities(stream_events=True)

    def __init__(self, *, steps: int, tokens: int = 0, cost: float = 0) -> None:
        self.steps = steps
        self.tokens = tokens
        self.cost = cost

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        for index in range(self.steps):
            await emit(HarnessEvent(type="agent.step", data={"index": index}))
        return ExecutionResult(
            session_id="budget",
            output={"ok": True},
            usage={"tokens": self.tokens, "cost": self.cost},
        )


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1, passed=True)


async def run_with_policy(tmp_path, harness: BudgetHarness, policy: SupervisorPolicy):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'budget.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(harness)
    engine = SupervisorEngine(
        store, harnesses, {"pass": CallableGrader("pass", pass_grader)}
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Stay inside budget",
            harness_id="budget",
            agent_bundle=AgentBundle(name="worker", system_prompt="Work."),
            grader_ids=("pass",),
            supervisor_policy=policy,
        )
    )
    snapshot = await engine.run_until_blocked(run_id)
    events = await store.load(run_id)
    await database.dispose()
    return snapshot, events


@pytest.mark.asyncio
async def test_step_budget_is_enforced_even_when_harness_ignores_request_limit(tmp_path) -> None:
    snapshot, events = await run_with_policy(
        tmp_path,
        BudgetHarness(steps=3),
        SupervisorPolicy(max_steps_per_attempt=2),
    )

    assert snapshot.status == RunStatus.WAITING_HUMAN
    assert snapshot.total_steps == 3
    assert "budget.exhausted" in [event.type for event in events]


@pytest.mark.asyncio
async def test_token_and_cost_budgets_are_checked_before_grading(tmp_path) -> None:
    snapshot, events = await run_with_policy(
        tmp_path,
        BudgetHarness(steps=1, tokens=101, cost=2.5),
        SupervisorPolicy(max_tokens=100, max_cost=2.0),
    )

    event_types = [event.type for event in events]
    assert snapshot.status == RunStatus.WAITING_HUMAN
    assert snapshot.total_tokens == 101
    assert snapshot.total_cost == 2.5
    assert "budget.exhausted" in event_types
    assert "evaluation.started" not in event_types
