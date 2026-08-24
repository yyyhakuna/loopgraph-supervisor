from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

import pytest

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.domain.memory import MemoryKind, MemoryStatus
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
from loopgraph_supervisor.infrastructure.memory_store import SqlMemoryStore


class MemoryAwareHarness:
    name = "memory-aware"
    capabilities = HarnessCapabilities()

    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        _emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        self.request = request
        return ExecutionResult(session_id="memory", output={"ok": True})


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1, passed=True)


@pytest.mark.asyncio
async def test_approved_version_memory_is_injected_into_execution_context(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'memory-integration.db'}")
    await database.initialize()
    memory_store = SqlMemoryStore(database)
    version_id = uuid4()
    approved = await memory_store.write(
        agent_name="coder",
        version_id=version_id,
        kind=MemoryKind.PROCEDURAL,
        content="Run checkout regression tests before completing.",
        status=MemoryStatus.APPROVED,
        tags=("checkout",),
    )
    await memory_store.write(
        agent_name="coder",
        version_id=version_id,
        kind=MemoryKind.EPISODIC,
        content="Unverified guess from a candidate run.",
        status=MemoryStatus.EXPERIMENTAL,
        tags=("checkout",),
    )
    harness = MemoryAwareHarness()
    harnesses = HarnessRegistry()
    harnesses.register(harness)
    event_store = SqlEventStore(database)
    engine = SupervisorEngine(
        event_store,
        harnesses,
        {"pass": CallableGrader("pass", pass_grader)},
        memory_store=memory_store,
    )
    run_id = await engine.create_run(
        RunDefinition(
            goal="Repair checkout",
            harness_id="memory-aware",
            agent_version_id=version_id,
            agent_bundle=AgentBundle(
                name="coder",
                system_prompt="Repair safely.",
                memory_config={
                    "enabled": True,
                    "required_tags": ["checkout"],
                    "limit": 10,
                },
            ),
            grader_ids=("pass",),
        )
    )

    await engine.run_until_blocked(run_id)
    await database.dispose()

    assert harness.request is not None
    assert harness.request.memories == (
        {
            "id": str(approved.id),
            "kind": "procedural",
            "content": approved.content,
            "tags": ["checkout"],
        },
    )
