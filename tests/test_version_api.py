from __future__ import annotations

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


class VersionHarness:
    name = "version-test"
    capabilities = HarnessCapabilities()

    async def execute(
        self,
        request: ExecutionRequest,
        _emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        quality = 0.9 if request.agent_bundle.skills else 0.7
        return ExecutionResult(
            session_id=f"version-{request.execution_id}",
            output={"quality": quality},
            usage={"cost": 1.0},
        )


async def quality_grader(context: EvaluationContext) -> EvaluationResult:
    score = float(context.output["quality"])
    return EvaluationResult(
        grader_id="quality",
        score=score,
        passed=score >= 0.8,
        hard_constraints={"security": True},
    )


def _wait_for_terminal(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = client.get(f"/v1/runs/{run_id}").json()
        if snapshot["status"] in {"succeeded", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Run {run_id} did not finish")


def test_agent_version_api_mutates_promotes_and_rolls_back_bundles(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'versions.db'}")
    event_store = SqlEventStore(database)
    version_store = SqlAgentVersionStore(database)
    harnesses = HarnessRegistry()
    harnesses.register(VersionHarness())
    graders = {"quality": CallableGrader("quality", quality_grader)}
    engine = SupervisorEngine(event_store, harnesses, graders)
    runtime = AppRuntime(
        database=database,
        event_store=event_store,
        version_store=version_store,
        harnesses=harnesses,
        graders=graders,
        engine=engine,
        tasks=RunTaskManager(engine),
    )

    with TestClient(create_app(runtime)) as client:
        initial = client.post(
            "/v1/agents/versions/initial",
            json={
                "bundle": {
                    "name": "coder",
                    "system_prompt": "Fix the task.",
                    "evolution_scope": "prompt_and_skills",
                },
                "actor": "operator",
                "activate": True,
            },
        )
        assert initial.status_code == 201
        v1 = initial.json()

        candidate = client.post(
            "/v1/agents/coder/candidates",
            json={
                "parent_version_id": v1["id"],
                "actor": "supervisor-agent",
                "plan": {
                    "rationale": "The agent skips reproduction.",
                    "system_prompt": "Reproduce, then fix the task.",
                    "skills_upsert": {"debugging": "Reproduce before editing."},
                },
            },
        )
        assert candidate.status_code == 201
        v2 = candidate.json()
        assert v2["version"] == 2
        assert v2["bundle"]["skills"]["debugging"] == "Reproduce before editing."

        spoofed = client.post(
            "/v1/runs",
            json={
                "goal": "benchmark",
                "harness_id": "version-test",
                "agent_version_id": v1["id"],
                "agent_bundle": v2["bundle"],
                "grader_ids": ["quality"],
            },
        )
        assert spoofed.status_code == 422

        baseline = client.post(
            "/v1/runs",
            json={
                "goal": "benchmark",
                "harness_id": "version-test",
                "agent_version_id": v1["id"],
                "grader_ids": ["quality"],
                "supervisor_policy": {"require_human_on_exhaustion": False},
            },
        ).json()
        candidate_run = client.post(
            "/v1/runs",
            json={
                "goal": "benchmark",
                "harness_id": "version-test",
                "agent_version_id": v2["id"],
                "grader_ids": ["quality"],
                "supervisor_policy": {"require_human_on_exhaustion": False},
            },
        ).json()
        client.post(f"/v1/runs/{baseline['run_id']}/start")
        client.post(f"/v1/runs/{candidate_run['run_id']}/start")
        _wait_for_terminal(client, baseline["run_id"])
        _wait_for_terminal(client, candidate_run["run_id"])

        untrusted = client.post(
            f"/v1/agents/coder/versions/{v2['id']}/promote",
            json={
                "expected_active_id": v1["id"],
                "actor": "reviewer",
                "reason": "Client claims a perfect score",
                "evaluation_summary": {"baseline": 0, "candidate": 1},
            },
        )
        assert untrusted.status_code == 422

        promoted = client.post(
            f"/v1/agents/coder/versions/{v2['id']}/promote",
            json={
                "expected_active_id": v1["id"],
                "actor": "reviewer",
                "reason": "Benchmark improved",
                "baseline_run_ids": [baseline["run_id"]],
                "candidate_run_ids": [candidate_run["run_id"]],
            },
        )
        active = client.get("/v1/agents/coder/versions/active")
        assert promoted.status_code == 200
        assert active.json()["id"] == v2["id"]

        rolled_back = client.post(
            "/v1/agents/coder/rollback",
            json={
                "target_version_id": v1["id"],
                "expected_active_id": v2["id"],
                "actor": "operator",
                "reason": "Online regression",
            },
        )

    assert rolled_back.status_code == 200
    assert rolled_back.json()["id"] == v1["id"]
    assert promoted.json()["evaluation_summary"]["baseline"] == 0.7
    assert promoted.json()["evaluation_summary"]["candidate"] == 0.9
