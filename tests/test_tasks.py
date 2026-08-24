from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from loopgraph_supervisor.application.tasks import RunAlreadyActiveError, RunTaskManager
from loopgraph_supervisor.domain.runs import RunStatus


class RacingEngine:
    def __init__(self) -> None:
        self.executions = 0
        self.release = asyncio.Event()

    async def get_run(self, _run_id):
        # Force concurrent callers to yield between checking and registering a task.
        await asyncio.sleep(0)
        return SimpleNamespace(status=RunStatus.CREATED)

    async def run_until_blocked(self, _run_id) -> None:
        self.executions += 1
        await self.release.wait()


@pytest.mark.asyncio
async def test_start_is_atomic_for_concurrent_requests() -> None:
    engine = RacingEngine()
    manager = RunTaskManager(engine)  # type: ignore[arg-type]
    run_id = uuid4()

    outcomes = await asyncio.gather(
        manager.start(run_id),
        manager.start(run_id),
        return_exceptions=True,
    )
    await asyncio.sleep(0)

    engine.release.set()
    await manager.shutdown()

    assert sum(result is None for result in outcomes) == 1
    assert sum(isinstance(result, RunAlreadyActiveError) for result in outcomes) == 1
    assert engine.executions == 1
