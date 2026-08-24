from __future__ import annotations

from uuid import uuid4

import pytest

from loopgraph_supervisor.domain.memory import MemoryKind, MemoryStatus
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.memory_store import SqlMemoryStore


@pytest.mark.asyncio
async def test_candidate_memory_is_isolated_until_explicitly_approved(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}")
    await database.initialize()
    store = SqlMemoryStore(database)
    v1 = uuid4()
    v2 = uuid4()

    stable = await store.write(
        agent_name="coder",
        version_id=v1,
        kind=MemoryKind.PROCEDURAL,
        content="Always run the regression suite.",
        status=MemoryStatus.APPROVED,
        source_run_id=uuid4(),
        tags=("testing",),
    )
    experimental = await store.write(
        agent_name="coder",
        version_id=v2,
        kind=MemoryKind.EPISODIC,
        content="The checkout API sometimes returns 401.",
        status=MemoryStatus.EXPERIMENTAL,
        source_run_id=uuid4(),
        tags=("checkout", "auth"),
    )

    assert await store.for_context("coder", v1) == (stable,)
    assert await store.for_context("coder", v2) == ()
    assert await store.for_context("coder", v2, include_experimental=True) == (
        experimental,
    )

    approved = await store.set_status(experimental.id, MemoryStatus.APPROVED)
    assert await store.for_context("coder", v2) == (approved,)
    assert await store.for_context("coder", v1) == (stable,)
    await database.dispose()


@pytest.mark.asyncio
async def test_memory_query_filters_kind_and_tags_without_cross_agent_leakage(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}")
    await database.initialize()
    store = SqlMemoryStore(database)
    version_id = uuid4()
    await store.write(
        agent_name="coder",
        version_id=version_id,
        kind=MemoryKind.EVALUATION,
        content="Authentication grader failed.",
        status=MemoryStatus.APPROVED,
        tags=("auth",),
    )
    await store.write(
        agent_name="other",
        version_id=uuid4(),
        kind=MemoryKind.EVALUATION,
        content="Authentication grader failed.",
        status=MemoryStatus.APPROVED,
        tags=("auth",),
    )

    results = await store.for_context(
        "coder", version_id, kind=MemoryKind.EVALUATION, required_tags=("auth",)
    )
    await database.dispose()

    assert len(results) == 1
    assert results[0].agent_name == "coder"
