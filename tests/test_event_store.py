from __future__ import annotations

from uuid import uuid4

import pytest

from loopgraph_supervisor.domain.events import NewEvent
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import (
    EventStoreConcurrencyError,
    SqlEventStore,
)


@pytest.mark.asyncio
async def test_sqlite_initialization_creates_the_configured_parent_directory(tmp_path) -> None:
    database_path = tmp_path / "nested" / "data" / "supervisor.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")

    await database.initialize()
    await database.dispose()

    assert database_path.exists()


@pytest.mark.asyncio
async def test_database_initialization_is_idempotent_and_events_survive_restart(tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}"
    run_id = uuid4()

    first_database = Database(database_url)
    await first_database.initialize()
    await first_database.initialize()
    first_store = SqlEventStore(first_database)
    appended = await first_store.append(
        run_id,
        expected_version=0,
        events=(
            NewEvent(type="run.created", data={"goal": "repair checkout"}),
            NewEvent(type="run.started", data={"attempt": 1}),
        ),
    )
    await first_database.dispose()

    second_database = Database(database_url)
    await second_database.initialize()
    second_store = SqlEventStore(second_database)
    loaded = await second_store.load(run_id)
    await second_database.dispose()

    assert [event.sequence for event in appended] == [1, 2]
    assert loaded == appended
    assert loaded[0].data == {"goal": "repair checkout"}


@pytest.mark.asyncio
async def test_event_store_rejects_stale_writers(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    run_id = uuid4()

    await store.append(
        run_id,
        expected_version=0,
        events=(NewEvent(type="run.created", data={}),),
    )

    with pytest.raises(EventStoreConcurrencyError) as exc_info:
        await store.append(
            run_id,
            expected_version=0,
            events=(NewEvent(type="run.started", data={}),),
        )

    await database.dispose()
    assert exc_info.value.actual_version == 1


@pytest.mark.asyncio
async def test_event_store_can_page_from_a_sequence(tmp_path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'supervisor.db'}")
    await database.initialize()
    store = SqlEventStore(database)
    run_id = uuid4()
    await store.append(
        run_id,
        expected_version=0,
        events=tuple(NewEvent(type="agent.step", data={"step": step}) for step in range(5)),
    )

    page = await store.load(run_id, after_sequence=2, limit=2)
    await database.dispose()

    assert [event.sequence for event in page] == [3, 4]
    assert [event.data["step"] for event in page] == [2, 3]
