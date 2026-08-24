from __future__ import annotations

import asyncio
from uuid import UUID

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.domain.runs import RunStatus


class RunAlreadyActiveError(RuntimeError):
    pass


class RunTaskManager:
    """Owns background run tasks and bounds process-level concurrency."""

    def __init__(self, engine: SupervisorEngine, max_concurrency: int = 8) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.engine = engine
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._start_lock = asyncio.Lock()
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._errors: dict[UUID, BaseException] = {}

    async def start(self, run_id: UUID) -> None:
        # Check state and publish the task atomically so concurrent API requests
        # cannot execute the same event stream twice.
        async with self._start_lock:
            task = self._tasks.get(run_id)
            if task is not None and not task.done():
                raise RunAlreadyActiveError(f"Run {run_id} is already active")
            snapshot = await self.engine.get_run(run_id)
            if snapshot.status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.PAUSED,
                RunStatus.WAITING_HUMAN,
            }:
                raise ValueError(f"Run {run_id} cannot start from {snapshot.status.value}")
            self._errors.pop(run_id, None)
            self._tasks[run_id] = asyncio.create_task(
                self._run(run_id), name=f"supervisor-run-{run_id}"
            )

    async def _run(self, run_id: UUID) -> None:
        try:
            async with self._semaphore:
                await self.engine.run_until_blocked(run_id)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._errors[run_id] = error
            try:
                await self.engine.record_internal_failure(run_id, error)
            except BaseException:
                # The original error remains visible in-process. Persistence is
                # best-effort because the database/event store may be the cause.
                pass

    def is_active(self, run_id: UUID) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def error(self, run_id: UUID) -> BaseException | None:
        return self._errors.get(run_id)

    async def wait(self, run_id: UUID) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await task

    async def shutdown(self) -> None:
        active = [task for task in self._tasks.values() if not task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
