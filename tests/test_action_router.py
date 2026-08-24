from __future__ import annotations

from uuid import uuid4

import pytest

from loopgraph_supervisor.application.actions import SupervisorActionRouter
from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.observer import (
    ObservationContext,
    SupervisorDirective,
    SupervisorDirectiveAction,
)
from loopgraph_supervisor.application.tasks import RunTaskManager
from loopgraph_supervisor.application.team import TeamCoordinator
from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.harness.base import HarnessEvent
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore
from loopgraph_supervisor.infrastructure.version_store import SqlAgentVersionStore


@pytest.mark.asyncio
async def test_mutation_directive_creates_a_versioned_candidate_without_auto_promotion(
    tmp_path,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'actions.db'}")
    await database.initialize()
    event_store = SqlEventStore(database)
    version_store = SqlAgentVersionStore(database)
    harnesses = HarnessRegistry()
    engine = SupervisorEngine(event_store, harnesses, {})
    tasks = RunTaskManager(engine)
    team = TeamCoordinator(engine, event_store, tasks)
    router = SupervisorActionRouter(team=team, versions=version_store)
    parent = await version_store.create_initial(
        AgentBundle(name="coder", system_prompt="Fix the task."),
        actor="operator",
        activate=True,
    )
    context = ObservationContext(
        run_id=uuid4(),
        attempt=2,
        goal="Fix checkout",
        agent_bundle=parent.bundle,
        agent_version_id=parent.id,
        recent_events=(HarnessEvent(type="evaluation.failed", data={"score": 0.2}),),
    )
    directive = SupervisorDirective(
        action=SupervisorDirectiveAction.PROPOSE_MUTATION,
        rationale="The agent fails to reproduce before editing.",
        payload={
            "plan": {
                "rationale": "Require reproduction.",
                "system_prompt": "Reproduce first, then fix the task.",
            },
        },
    )

    outcome = await router.handle(context, directive)
    candidate = await version_store.get(outcome.candidate_version_id)
    active = await version_store.get_active("coder")
    await database.dispose()

    assert candidate.parent_id == parent.id
    assert candidate.bundle.system_prompt.startswith("Reproduce first")
    assert active.id == parent.id
