from __future__ import annotations

from fastapi.testclient import TestClient

from loopgraph_supervisor.api.app import AppRuntime, create_app
from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.tasks import RunTaskManager
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore
from loopgraph_supervisor.infrastructure.memory_store import SqlMemoryStore
from loopgraph_supervisor.infrastructure.version_store import SqlAgentVersionStore


def test_memory_api_keeps_new_memory_experimental_until_approval(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'memory-api.db'}")
    event_store = SqlEventStore(database)
    version_store = SqlAgentVersionStore(database)
    memory_store = SqlMemoryStore(database)
    harnesses = HarnessRegistry()
    engine = SupervisorEngine(
        event_store, harnesses, {}, memory_store=memory_store
    )
    runtime = AppRuntime(
        database=database,
        event_store=event_store,
        version_store=version_store,
        memory_store=memory_store,
        harnesses=harnesses,
        graders={},
        engine=engine,
        tasks=RunTaskManager(engine),
    )

    with TestClient(create_app(runtime)) as client:
        version = client.post(
            "/v1/agents/versions/initial",
            json={
                "bundle": {"name": "coder", "system_prompt": "Work."},
                "actor": "operator",
            },
        ).json()
        created = client.post(
            f"/v1/agents/coder/versions/{version['id']}/memories",
            json={
                "kind": "procedural",
                "content": "Run regression tests.",
                "tags": ["testing"],
                "actor": "supervisor-agent",
            },
        )
        hidden = client.get(
            f"/v1/agents/coder/versions/{version['id']}/memories"
        )
        approved = client.post(
            f"/v1/memories/{created.json()['id']}/status",
            json={"status": "approved", "actor": "reviewer"},
        )
        visible = client.get(
            f"/v1/agents/coder/versions/{version['id']}/memories"
        )

    assert created.status_code == 201
    assert created.json()["status"] == "experimental"
    assert hidden.json()["items"] == []
    assert approved.status_code == 200
    assert visible.json()["items"][0]["content"] == "Run regression tests."
