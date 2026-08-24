from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Database:
    """Owns the async engine and idempotent schema bootstrap."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._prepare_sqlite_directory()
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        if url.startswith("sqlite"):
            self._configure_sqlite()

    def _prepare_sqlite_directory(self) -> None:
        parsed = make_url(self.url)
        database = parsed.database
        if (
            not parsed.drivername.startswith("sqlite")
            or database is None
            or database in {"", ":memory:"}
        ):
            return
        Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _configure_sqlite(self) -> None:
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragmas(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    async def initialize(self) -> None:
        # Import model modules before create_all so their tables are registered.
        from loopgraph_supervisor.infrastructure import event_store as _event_store  # noqa: F401
        from loopgraph_supervisor.infrastructure import memory_store as _memory_store  # noqa: F401
        from loopgraph_supervisor.infrastructure import (
            version_store as _version_store,  # noqa: F401
        )

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()
