from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.tasks import RunTaskManager
from loopgraph_supervisor.application.team import (
    ChildAgentSpec,
    SubagentLimitError,
    TeamCoordinator,
)
from loopgraph_supervisor.domain.models import AgentBundle, EvaluationResult
from loopgraph_supervisor.domain.runs import RunDefinition, SupervisorPolicy
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


class TeamHarness:
    name = "team"
    capabilities = HarnessCapabilities(stream_events=True, subagents=True)

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        return ExecutionResult(session_id="child", output={"ok": True})


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1, passed=True)


@pytest.mark.asyncio
async def test_child_run_is_linked_through_framework_neutral_team_context(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'team.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(TeamHarness())
    engine = SupervisorEngine(
        store,
        harnesses,
        {"pass": CallableGrader("pass", pass_grader)},
    )
    tasks = RunTaskManager(engine)
    coordinator = TeamCoordinator(engine, store, tasks)
    parent_id = await engine.create_run(
        RunDefinition(
            goal="Repair checkout",
            harness_id="team",
            agent_bundle=AgentBundle(name="leader", system_prompt="Coordinate repair."),
            grader_ids=("pass",),
            supervisor_policy=SupervisorPolicy(max_subagents=1, max_subagent_depth=1),
        )
    )

    child_id = await coordinator.spawn(
        parent_id,
        ChildAgentSpec(role="investigator", goal="Find the checkout failure"),
        actor="supervisor-agent",
        start=False,
    )
    context = await coordinator.context(parent_id)
    child = await engine.get_run(child_id)

    assert child.definition.parent_run_id == parent_id
    assert child.definition.team_id == context.team_id
    assert child.definition.role == "investigator"
    assert context.members[0].run_id == child_id
    assert context.members[0].role == "investigator"

    with pytest.raises(SubagentLimitError):
        await coordinator.spawn(
            parent_id,
            ChildAgentSpec(role="reviewer", goal="Review the repair"),
            actor="supervisor-agent",
            start=False,
        )

    await tasks.shutdown()
    await database.dispose()
