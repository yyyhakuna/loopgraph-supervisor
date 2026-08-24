from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, String, select
from sqlalchemy.orm import Mapped, mapped_column

from loopgraph_supervisor.domain.memory import (
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
)
from loopgraph_supervisor.infrastructure.database import Base, Database


class MemoryRow(Base):
    __tablename__ = "agent_memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    version_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    source_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    memory_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemoryTransitionError(ValueError):
    pass


class SqlMemoryStore:
    """Version-scoped memory with fail-closed candidate isolation."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def write(
        self,
        *,
        agent_name: str,
        version_id: UUID,
        kind: MemoryKind,
        content: str,
        status: MemoryStatus = MemoryStatus.EXPERIMENTAL,
        source_run_id: UUID | None = None,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        row = MemoryRow(
            id=str(uuid4()),
            agent_name=agent_name,
            version_id=str(version_id),
            kind=kind.value,
            content=content,
            status=status.value,
            source_run_id=str(source_run_id) if source_run_id else None,
            tags=sorted(set(tags)),
            memory_metadata=metadata or {},
            created_at=datetime.now(UTC),
        )
        async with self.database.session() as session, session.begin():
            session.add(row)
        return self._to_domain(row)

    async def set_status(
        self, memory_id: UUID, status: MemoryStatus
    ) -> MemoryRecord:
        async with self.database.session() as session, session.begin():
            row = await session.get(MemoryRow, str(memory_id))
            if row is None:
                raise KeyError(f"Memory {memory_id} does not exist")
            current = MemoryStatus(row.status)
            if current == MemoryStatus.REJECTED and status != current:
                raise MemoryTransitionError("Rejected memory cannot be reactivated")
            row.status = status.value
        return self._to_domain(row)

    async def for_context(
        self,
        agent_name: str,
        version_id: UUID,
        *,
        include_experimental: bool = False,
        kind: MemoryKind | None = None,
        required_tags: tuple[str, ...] = (),
        limit: int = 100,
    ) -> tuple[MemoryRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        statuses = [MemoryStatus.APPROVED.value]
        if include_experimental:
            statuses.append(MemoryStatus.EXPERIMENTAL.value)
        statement = (
            select(MemoryRow)
            .where(
                MemoryRow.agent_name == agent_name,
                MemoryRow.version_id == str(version_id),
                MemoryRow.status.in_(statuses),
            )
            .order_by(MemoryRow.created_at)
        )
        if kind is not None:
            statement = statement.where(MemoryRow.kind == kind.value)
        async with self.database.session() as session:
            rows = (await session.scalars(statement)).all()
        required = set(required_tags)
        filtered = [row for row in rows if required.issubset(row.tags)]
        return tuple(self._to_domain(row) for row in filtered[:limit])

    @staticmethod
    def _to_domain(row: MemoryRow) -> MemoryRecord:
        created_at = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=UTC)
        return MemoryRecord(
            id=UUID(row.id),
            agent_name=row.agent_name,
            version_id=UUID(row.version_id),
            kind=MemoryKind(row.kind),
            content=row.content,
            status=MemoryStatus(row.status),
            source_run_id=UUID(row.source_run_id) if row.source_run_id else None,
            tags=tuple(row.tags),
            metadata=row.memory_metadata,
            created_at=created_at,
        )
