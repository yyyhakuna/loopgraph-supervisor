from __future__ import annotations

import pytest

from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.domain.versions import VersionStatus
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.version_store import (
    DuplicateBundleError,
    SqlAgentVersionStore,
    VersionConflictError,
)


def bundle(prompt: str) -> AgentBundle:
    return AgentBundle(name="coding-agent", system_prompt=prompt)


@pytest.mark.asyncio
async def test_candidate_promotion_and_rollback_preserve_lineage(tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}"
    database = Database(database_url)
    await database.initialize()
    store = SqlAgentVersionStore(database)

    v1 = await store.create_initial(bundle("Run tests."), actor="operator", activate=True)
    v2 = await store.create_candidate(
        parent_id=v1.id,
        bundle=bundle("Reproduce the bug, then run tests."),
        actor="evolution-agent",
        change_summary="Require reproduction before editing",
    )

    assert (await store.get_active("coding-agent")).id == v1.id
    assert v2.parent_id == v1.id
    assert v2.version == 2
    assert v2.status == VersionStatus.CANDIDATE

    promoted = await store.promote(
        v2.id,
        expected_active_id=v1.id,
        actor="reviewer",
        reason="Regression suite improved",
        evaluation_summary={"baseline": 0.70, "candidate": 0.91},
    )
    assert promoted.status == VersionStatus.ACTIVE
    assert (await store.get(v1.id)).status == VersionStatus.SUPERSEDED

    restored = await store.rollback(
        "coding-agent",
        target_version_id=v1.id,
        expected_active_id=v2.id,
        actor="operator",
        reason="Online regression",
    )
    history = await store.activation_history("coding-agent")
    await database.dispose()

    assert restored.id == v1.id
    assert restored.status == VersionStatus.ACTIVE
    assert [entry.action for entry in history] == ["initial_activation", "promote", "rollback"]
    assert history[-1].from_version_id == v2.id
    assert history[-1].to_version_id == v1.id


@pytest.mark.asyncio
async def test_version_store_rejects_duplicate_candidates_and_stale_promotion(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}")
    await database.initialize()
    store = SqlAgentVersionStore(database)
    v1 = await store.create_initial(bundle("V1"), actor="operator", activate=True)

    with pytest.raises(DuplicateBundleError):
        await store.create_candidate(
            parent_id=v1.id,
            bundle=bundle("V1"),
            actor="evolution-agent",
            change_summary="No actual change",
        )

    v2 = await store.create_candidate(
        parent_id=v1.id,
        bundle=bundle("V2"),
        actor="evolution-agent",
        change_summary="Improve instructions",
    )
    with pytest.raises(VersionConflictError):
        await store.promote(
            v2.id,
            expected_active_id=None,
            actor="reviewer",
            reason="Stale request",
            evaluation_summary={},
        )

    await database.dispose()
