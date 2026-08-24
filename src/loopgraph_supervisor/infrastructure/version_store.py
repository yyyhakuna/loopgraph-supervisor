from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.domain.versions import (
    ActivationHistoryEntry,
    AgentVersion,
    VersionStatus,
)
from loopgraph_supervisor.infrastructure.database import Base, Database


class AgentVersionRecord(Base):
    __tablename__ = "agent_versions"
    __table_args__ = (
        UniqueConstraint("agent_name", "version", name="uq_agent_version_number"),
        UniqueConstraint("agent_name", "fingerprint", name="uq_agent_bundle_fingerprint"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_versions.id"), nullable=True
    )
    bundle: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    change_summary: Mapped[str] = mapped_column(String(2000), nullable=False)
    evaluation_summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ActiveVersionRecord(Base):
    __tablename__ = "active_agent_versions"

    agent_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("agent_versions.id"), unique=True, nullable=False
    )


class ActivationHistoryRecord(Base):
    __tablename__ = "agent_activation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    from_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    to_version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    actor: Mapped[str] = mapped_column(String(256), nullable=False)
    reason: Mapped[str] = mapped_column(String(2000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DuplicateBundleError(ValueError):
    pass


class VersionConflictError(RuntimeError):
    pass


class SqlAgentVersionStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def create_initial(
        self, bundle: AgentBundle, *, actor: str, activate: bool = True
    ) -> AgentVersion:
        async with self._locks[bundle.name]:
            async with self.database.session() as session, session.begin():
                existing = await self._versions_for_agent(session, bundle.name)
                if existing:
                    raise VersionConflictError(
                        f"Agent {bundle.name!r} already has an initial version"
                    )
                record = self._new_record(
                    bundle=bundle,
                    version=1,
                    parent_id=None,
                    status=VersionStatus.ACTIVE if activate else VersionStatus.CANDIDATE,
                    actor=actor,
                    change_summary="Initial version",
                )
                session.add(record)
                await session.flush()
                if activate:
                    session.add(
                        ActiveVersionRecord(agent_name=bundle.name, version_id=record.id)
                    )
                    session.add(
                        self._history_record(
                            agent_name=bundle.name,
                            action="initial_activation",
                            from_version_id=None,
                            to_version_id=record.id,
                            actor=actor,
                            reason="Initial version",
                        )
                    )
            return self._to_domain(record)

    async def create_candidate(
        self,
        *,
        parent_id: UUID,
        bundle: AgentBundle,
        actor: str,
        change_summary: str,
    ) -> AgentVersion:
        async with self._locks[bundle.name]:
            async with self.database.session() as session, session.begin():
                parent = await session.get(AgentVersionRecord, str(parent_id))
                if parent is None:
                    raise KeyError(f"Parent version {parent_id} does not exist")
                if parent.agent_name != bundle.name:
                    raise ValueError("Candidate and parent must belong to the same agent")
                records = await self._versions_for_agent(session, bundle.name)
                if any(record.fingerprint == bundle.fingerprint for record in records):
                    raise DuplicateBundleError(
                        f"Bundle {bundle.fingerprint} already exists for {bundle.name!r}"
                    )
                record = self._new_record(
                    bundle=bundle,
                    version=max(item.version for item in records) + 1,
                    parent_id=str(parent_id),
                    status=VersionStatus.CANDIDATE,
                    actor=actor,
                    change_summary=change_summary,
                )
                session.add(record)
            return self._to_domain(record)

    async def promote(
        self,
        version_id: UUID,
        *,
        expected_active_id: UUID | None,
        actor: str,
        reason: str,
        evaluation_summary: dict[str, Any],
    ) -> AgentVersion:
        candidate = await self.get(version_id)
        async with self._locks[candidate.agent_name]:
            async with self.database.session() as session, session.begin():
                record = await session.get(AgentVersionRecord, str(version_id))
                if record is None:
                    raise KeyError(f"Version {version_id} does not exist")
                if record.status != VersionStatus.CANDIDATE:
                    raise VersionConflictError("Only a candidate version can be promoted")
                pointer = await session.get(ActiveVersionRecord, record.agent_name)
                actual_active = UUID(pointer.version_id) if pointer else None
                if actual_active != expected_active_id:
                    raise VersionConflictError(
                        f"Expected active version {expected_active_id}, actual {actual_active}"
                    )
                if pointer is not None:
                    previous = await session.get(AgentVersionRecord, pointer.version_id)
                    if previous is not None:
                        previous.status = VersionStatus.SUPERSEDED
                    changed = await session.execute(
                        update(ActiveVersionRecord)
                        .where(
                            ActiveVersionRecord.agent_name == record.agent_name,
                            ActiveVersionRecord.version_id == str(expected_active_id),
                        )
                        .values(version_id=record.id)
                    )
                    if getattr(changed, "rowcount", 0) != 1:
                        raise VersionConflictError(
                            "Active version changed during promotion"
                        )
                else:
                    session.add(
                        ActiveVersionRecord(
                            agent_name=record.agent_name, version_id=record.id
                        )
                    )
                record.status = VersionStatus.ACTIVE
                record.evaluation_summary = evaluation_summary
                session.add(
                    self._history_record(
                        agent_name=record.agent_name,
                        action="promote",
                        from_version_id=str(actual_active) if actual_active else None,
                        to_version_id=record.id,
                        actor=actor,
                        reason=reason,
                    )
                )
            return self._to_domain(record)

    async def rollback(
        self,
        agent_name: str,
        *,
        target_version_id: UUID,
        expected_active_id: UUID,
        actor: str,
        reason: str,
    ) -> AgentVersion:
        async with self._locks[agent_name]:
            async with self.database.session() as session, session.begin():
                pointer = await session.get(ActiveVersionRecord, agent_name)
                if pointer is None:
                    raise VersionConflictError(f"Agent {agent_name!r} has no active version")
                target = await session.get(AgentVersionRecord, str(target_version_id))
                if target is None or target.agent_name != agent_name:
                    raise KeyError(
                        f"Version {target_version_id} does not belong to {agent_name!r}"
                    )
                if target.status not in {VersionStatus.SUPERSEDED, VersionStatus.ACTIVE}:
                    raise VersionConflictError("Rollback target was never an active version")
                if UUID(pointer.version_id) != expected_active_id:
                    raise VersionConflictError(
                        f"Expected active version {expected_active_id}, "
                        f"actual {pointer.version_id}"
                    )
                if pointer.version_id == target.id:
                    raise VersionConflictError("Rollback target is already active")
                previous_id = pointer.version_id
                previous = await session.get(AgentVersionRecord, previous_id)
                if previous is not None:
                    previous.status = VersionStatus.SUPERSEDED
                target.status = VersionStatus.ACTIVE
                changed = await session.execute(
                    update(ActiveVersionRecord)
                    .where(
                        ActiveVersionRecord.agent_name == agent_name,
                        ActiveVersionRecord.version_id == str(expected_active_id),
                    )
                    .values(version_id=target.id)
                )
                if getattr(changed, "rowcount", 0) != 1:
                    raise VersionConflictError("Active version changed during rollback")
                session.add(
                    self._history_record(
                        agent_name=agent_name,
                        action="rollback",
                        from_version_id=previous_id,
                        to_version_id=target.id,
                        actor=actor,
                        reason=reason,
                    )
                )
            return self._to_domain(target)

    async def get(self, version_id: UUID) -> AgentVersion:
        async with self.database.session() as session:
            record = await session.get(AgentVersionRecord, str(version_id))
        if record is None:
            raise KeyError(f"Version {version_id} does not exist")
        return self._to_domain(record)

    async def get_active(self, agent_name: str) -> AgentVersion:
        async with self.database.session() as session:
            pointer = await session.get(ActiveVersionRecord, agent_name)
            record = (
                await session.get(AgentVersionRecord, pointer.version_id)
                if pointer is not None
                else None
            )
        if record is None:
            raise KeyError(f"Agent {agent_name!r} has no active version")
        return self._to_domain(record)

    async def activation_history(
        self, agent_name: str
    ) -> tuple[ActivationHistoryEntry, ...]:
        statement = (
            select(ActivationHistoryRecord)
            .where(ActivationHistoryRecord.agent_name == agent_name)
            .order_by(ActivationHistoryRecord.id)
        )
        async with self.database.session() as session:
            records = (await session.scalars(statement)).all()
        return tuple(self._history_to_domain(record) for record in records)

    @staticmethod
    async def _versions_for_agent(
        session: AsyncSession, agent_name: str
    ) -> list[AgentVersionRecord]:
        statement = select(AgentVersionRecord).where(
            AgentVersionRecord.agent_name == agent_name
        )
        return list((await session.scalars(statement)).all())

    @staticmethod
    def _new_record(
        *,
        bundle: AgentBundle,
        version: int,
        parent_id: str | None,
        status: VersionStatus,
        actor: str,
        change_summary: str,
    ) -> AgentVersionRecord:
        return AgentVersionRecord(
            id=str(uuid4()),
            agent_name=bundle.name,
            version=version,
            parent_id=parent_id,
            bundle=bundle.model_dump(mode="json", by_alias=True),
            fingerprint=bundle.fingerprint,
            status=status,
            created_at=datetime.now(UTC),
            created_by=actor,
            change_summary=change_summary,
            evaluation_summary={},
        )

    @staticmethod
    def _history_record(
        *,
        agent_name: str,
        action: str,
        from_version_id: str | None,
        to_version_id: str,
        actor: str,
        reason: str,
    ) -> ActivationHistoryRecord:
        return ActivationHistoryRecord(
            agent_name=agent_name,
            action=action,
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            actor=actor,
            reason=reason,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    @classmethod
    def _to_domain(cls, record: AgentVersionRecord) -> AgentVersion:
        return AgentVersion(
            id=UUID(record.id),
            agent_name=record.agent_name,
            version=record.version,
            parent_id=UUID(record.parent_id) if record.parent_id else None,
            bundle=AgentBundle.model_validate(record.bundle),
            fingerprint=record.fingerprint,
            status=VersionStatus(record.status),
            created_at=cls._utc(record.created_at),
            created_by=record.created_by,
            change_summary=record.change_summary,
            evaluation_summary=record.evaluation_summary,
        )

    @classmethod
    def _history_to_domain(
        cls, record: ActivationHistoryRecord
    ) -> ActivationHistoryEntry:
        return ActivationHistoryEntry(
            id=record.id,
            agent_name=record.agent_name,
            action=record.action,
            from_version_id=(
                UUID(record.from_version_id) if record.from_version_id else None
            ),
            to_version_id=UUID(record.to_version_id),
            actor=record.actor,
            reason=record.reason,
            created_at=cls._utc(record.created_at),
        )
