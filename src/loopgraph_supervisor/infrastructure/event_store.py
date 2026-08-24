from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from loopgraph_supervisor.domain.events import NewEvent, StoredEvent
from loopgraph_supervisor.infrastructure.database import Base, Database


class EventRecord(Base):
    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_run_event_sequence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    event_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EventStoreConcurrencyError(RuntimeError):
    def __init__(self, run_id: UUID, expected_version: int, actual_version: int) -> None:
        self.run_id = run_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Run {run_id} expected version {expected_version}, actual version {actual_version}"
        )


class SqlEventStore:
    """Append-only run streams with optimistic concurrency."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._locks: defaultdict[UUID, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def append(
        self,
        run_id: UUID,
        *,
        expected_version: int,
        events: tuple[NewEvent, ...],
    ) -> tuple[StoredEvent, ...]:
        if not events:
            return ()

        async with self._locks[run_id]:
            try:
                async with self.database.session() as session, session.begin():
                    actual_version = await self._version(session, run_id)
                    if actual_version != expected_version:
                        raise EventStoreConcurrencyError(
                            run_id, expected_version, actual_version
                        )

                    stored = tuple(
                        StoredEvent(
                            **event.model_dump(),
                            run_id=run_id,
                            sequence=actual_version + offset,
                        )
                        for offset, event in enumerate(events, start=1)
                    )
                    session.add_all([self._to_record(event) for event in stored])
                return stored
            except IntegrityError as error:
                actual_version = await self.version(run_id)
                raise EventStoreConcurrencyError(
                    run_id, expected_version, actual_version
                ) from error

    async def load(
        self,
        run_id: UUID,
        *,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredEvent, ...]:
        statement = (
            select(EventRecord)
            .where(
                EventRecord.run_id == str(run_id),
                EventRecord.sequence > after_sequence,
            )
            .order_by(EventRecord.sequence)
        )
        if limit is not None:
            statement = statement.limit(limit)
        async with self.database.session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(self._to_domain(row) for row in rows)

    async def version(self, run_id: UUID) -> int:
        async with self.database.session() as session:
            return await self._version(session, run_id)

    async def list_run_ids(self, *, limit: int = 100, offset: int = 0) -> tuple[UUID, ...]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset non-negative")
        created_at = func.min(EventRecord.occurred_at).label("created_at")
        statement = (
            select(EventRecord.run_id, created_at)
            .group_by(EventRecord.run_id)
            .order_by(created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self.database.session() as session:
            rows = (await session.execute(statement)).all()
        return tuple(UUID(row.run_id) for row in rows)

    @staticmethod
    async def _version(session: Any, run_id: UUID) -> int:
        statement = select(func.coalesce(func.max(EventRecord.sequence), 0)).where(
            EventRecord.run_id == str(run_id)
        )
        return int(await session.scalar(statement) or 0)

    @staticmethod
    def _to_record(event: StoredEvent) -> EventRecord:
        return EventRecord(
            id=str(event.id),
            run_id=str(event.run_id),
            sequence=event.sequence,
            type=event.type,
            data=event.data,
            event_metadata=event.metadata,
            occurred_at=event.occurred_at,
        )

    @staticmethod
    def _to_domain(record: EventRecord) -> StoredEvent:
        occurred_at = record.occurred_at
        if occurred_at.tzinfo is None:
            # SQLite stores ISO timestamps without preserving timezone metadata.
            # Events are always written in UTC, so make that invariant explicit on read.
            occurred_at = occurred_at.replace(tzinfo=UTC)
        return StoredEvent(
            id=UUID(record.id),
            run_id=UUID(record.run_id),
            sequence=record.sequence,
            type=record.type,
            data=record.data,
            metadata=record.event_metadata,
            occurred_at=occurred_at,
        )
