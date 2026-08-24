from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from fastapi.testclient import TestClient

from loopgraph_supervisor.api.app import AppRuntime, create_app
from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.tasks import RunTaskManager
from loopgraph_supervisor.domain.models import EvaluationResult
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
from loopgraph_supervisor.infrastructure.version_store import SqlAgentVersionStore


class ApiHarness:
    name = "api-test"
    capabilities = HarnessCapabilities(stream_events=True)

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        await emit(HarnessEvent(type="tool/call", data={"name": "finish"}))
        return ExecutionResult(session_id="api-session", output={"ok": True})


class SlowHarness:
    name = "slow-test"
    capabilities = HarnessCapabilities()

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        await asyncio.sleep(0.2)
        return ExecutionResult(session_id="slow-session", output={"ok": True})


async def pass_grader(_: EvaluationContext) -> EvaluationResult:
    return EvaluationResult(grader_id="pass", score=1.0, passed=True)


def test_api_creates_runs_executes_in_background_and_exposes_events(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(ApiHarness())
    graders = {"pass": CallableGrader("pass", pass_grader)}
    engine = SupervisorEngine(store, harnesses, graders)
    tasks = RunTaskManager(engine, max_concurrency=2)
    runtime = AppRuntime(
        database=database,
        event_store=store,
        version_store=SqlAgentVersionStore(database),
        harnesses=harnesses,
        graders=graders,
        engine=engine,
        tasks=tasks,
    )

    with TestClient(create_app(runtime)) as client:
        created = client.post(
            "/v1/runs",
            json={
                "goal": "Complete the task",
                "harness_id": "api-test",
                "agent_bundle": {
                    "name": "worker",
                    "system_prompt": "Complete work safely.",
                },
                "grader_ids": ["pass"],
            },
        )
        assert created.status_code == 201
        run_id = created.json()["run_id"]

        started = client.post(f"/v1/runs/{run_id}/start")
        assert started.status_code == 202

        deadline = time.monotonic() + 2
        snapshot = {}
        while time.monotonic() < deadline:
            snapshot = client.get(f"/v1/runs/{run_id}").json()
            if snapshot.get("status") == "succeeded":
                break
            time.sleep(0.01)

        events = client.get(f"/v1/runs/{run_id}/events?after_sequence=0&limit=100")
        with client.stream(
            "GET", f"/v1/runs/{run_id}/events/stream?after_sequence=0"
        ) as stream_response:
            stream_body = "".join(stream_response.iter_text())
        health = client.get("/healthz")
        capabilities = client.get("/v1/capabilities")
        run_list = client.get("/v1/runs?limit=20&offset=0")

    assert snapshot["status"] == "succeeded"
    assert health.json() == {"status": "ok"}
    assert events.status_code == 200
    assert stream_response.status_code == 200
    assert "event: run.created" in stream_body
    assert "event: run.succeeded" in stream_body
    assert [event["sequence"] for event in events.json()["items"]] == list(
        range(1, len(events.json()["items"]) + 1)
    )
    assert capabilities.json()["harnesses"] == ["api-test"]
    assert capabilities.json()["graders"] == ["pass"]
    assert run_list.json()["items"][0]["run_id"] == run_id
    assert run_list.json()["items"][0]["status"] == "succeeded"


def test_api_returns_structured_errors_for_unknown_harness(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    graders = {"pass": CallableGrader("pass", pass_grader)}
    engine = SupervisorEngine(store, harnesses, graders)
    runtime = AppRuntime(
        database=database,
        event_store=store,
        version_store=SqlAgentVersionStore(database),
        harnesses=harnesses,
        graders=graders,
        engine=engine,
        tasks=RunTaskManager(engine),
    )

    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/runs",
            json={
                "goal": "Complete the task",
                "harness_id": "missing",
                "agent_bundle": {"name": "worker", "system_prompt": "Work."},
                "grader_ids": ["pass"],
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_api_rejects_recovery_while_local_execution_is_still_active(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'recover.db'}")
    store = SqlEventStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(SlowHarness())
    graders = {"pass": CallableGrader("pass", pass_grader)}
    engine = SupervisorEngine(store, harnesses, graders)
    runtime = AppRuntime(
        database=database,
        event_store=store,
        version_store=SqlAgentVersionStore(database),
        harnesses=harnesses,
        graders=graders,
        engine=engine,
        tasks=RunTaskManager(engine),
    )

    with TestClient(create_app(runtime)) as client:
        run_id = client.post(
            "/v1/runs",
            json={
                "goal": "Wait",
                "harness_id": "slow-test",
                "agent_bundle": {"name": "worker", "system_prompt": "Wait."},
                "grader_ids": ["pass"],
            },
        ).json()["run_id"]
        client.post(f"/v1/runs/{run_id}/start")
        response = client.post(
            f"/v1/runs/{run_id}/recover",
            json={"actor": "operator", "reason": "restart"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_already_active"
