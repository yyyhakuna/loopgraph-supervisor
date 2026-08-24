from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.api.app import AppRuntime
from loopgraph_supervisor.application.actions import SupervisorActionRouter
from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.observer import (
    HarnessSupervisorAdvisor,
    ObservationPolicy,
)
from loopgraph_supervisor.application.tasks import RunTaskManager
from loopgraph_supervisor.application.team import TeamCoordinator
from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.evolution.promotion import PromotionPolicy
from loopgraph_supervisor.grading.base import Grader
from loopgraph_supervisor.grading.rules import OutputRule, RuleGrader
from loopgraph_supervisor.grading.script import ScriptGrader
from loopgraph_supervisor.harness.command import CommandHarnessAdapter
from loopgraph_supervisor.harness.dsh import DeepSeekHarnessAdapter
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore
from loopgraph_supervisor.infrastructure.memory_store import SqlMemoryStore
from loopgraph_supervisor.infrastructure.version_store import SqlAgentVersionStore


class CommandHarnessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["command"]
    name: str
    argv: tuple[str, ...]
    cwd: str | None = None
    env: dict[str, str] | None = None
    max_output_bytes: int = Field(default=4 * 1024 * 1024, gt=0)


class DshHarnessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["dsh"]
    name: str = Field(min_length=1, max_length=128)
    sdk_config: dict[str, Any] = Field(default_factory=dict)
    shutdown_timeout_seconds: float = Field(default=5.0, gt=0)


HarnessConfig = Annotated[
    CommandHarnessConfig | DshHarnessConfig,
    Field(discriminator="type"),
]


class ScriptGraderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["script"]
    id: str
    argv: tuple[str, ...]
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_output_bytes: int = Field(default=1_048_576, gt=0)
    cwd: str | None = None


class RuleGraderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["rules"]
    id: str
    rules: tuple[OutputRule, ...]
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)


GraderConfig = Annotated[
    ScriptGraderConfig | RuleGraderConfig,
    Field(discriminator="type"),
]


class SupervisorAgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    harness_id: str
    bundle: AgentBundle
    observation_policy: ObservationPolicy = Field(default_factory=ObservationPolicy)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_steps: int = Field(default=10, ge=1)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    database_url: str = "sqlite+aiosqlite:///./loopgraph.db"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65_535)
    max_concurrency: int = Field(default=8, ge=1, le=1000)
    harnesses: tuple[HarnessConfig, ...] = ()
    graders: tuple[GraderConfig, ...] = ()
    supervisor_agent: SupervisorAgentConfig | None = None
    promotion_policy: PromotionPolicy = Field(default_factory=PromotionPolicy)

    @classmethod
    def from_json(cls, path: str | Path) -> Settings:
        with Path(path).open(encoding="utf-8") as file:
            return cls.model_validate(json.load(file))


def build_runtime(settings: Settings) -> AppRuntime:
    database = Database(settings.database_url)
    event_store = SqlEventStore(database)
    version_store = SqlAgentVersionStore(database)
    memory_store = SqlMemoryStore(database)
    harnesses = HarnessRegistry()
    for harness_config in settings.harnesses:
        if isinstance(harness_config, CommandHarnessConfig):
            harnesses.register(
                CommandHarnessAdapter(
                    name=harness_config.name,
                    argv=harness_config.argv,
                    cwd=harness_config.cwd,
                    env=harness_config.env,
                    max_output_bytes=harness_config.max_output_bytes,
                )
            )
        else:
            harnesses.register(
                DeepSeekHarnessAdapter(
                    name=harness_config.name,
                    sdk_config=harness_config.sdk_config,
                    shutdown_timeout_seconds=(
                        harness_config.shutdown_timeout_seconds
                    ),
                )
            )

    graders: dict[str, Grader] = {}
    for grader_config in settings.graders:
        if isinstance(grader_config, ScriptGraderConfig):
            grader: Grader = ScriptGrader(
                grader_id=grader_config.id,
                argv=grader_config.argv,
                timeout_seconds=grader_config.timeout_seconds,
                max_output_bytes=grader_config.max_output_bytes,
                cwd=grader_config.cwd,
            )
        else:
            grader = RuleGrader(
                grader_id=grader_config.id,
                rules=grader_config.rules,
                threshold=grader_config.threshold,
            )
        if grader.id in graders:
            raise ValueError(f"Grader {grader.id!r} is configured twice")
        graders[grader.id] = grader

    advisor = None
    observation_policy = None
    if settings.supervisor_agent is not None:
        supervisor_config = settings.supervisor_agent
        advisor = HarnessSupervisorAdvisor(
            harness=harnesses.get(supervisor_config.harness_id),
            bundle=supervisor_config.bundle,
            timeout_seconds=supervisor_config.timeout_seconds,
            max_steps=supervisor_config.max_steps,
        )
        observation_policy = supervisor_config.observation_policy
    engine = SupervisorEngine(
        event_store,
        harnesses,
        graders,
        advisor=advisor,
        observation_policy=observation_policy,
        memory_store=memory_store,
    )
    tasks = RunTaskManager(engine, max_concurrency=settings.max_concurrency)
    team = TeamCoordinator(engine, event_store, tasks)
    action_router = SupervisorActionRouter(team=team, versions=version_store)
    engine.set_action_handler(action_router)
    return AppRuntime(
        database=database,
        event_store=event_store,
        version_store=version_store,
        memory_store=memory_store,
        harnesses=harnesses,
        graders=graders,
        engine=engine,
        tasks=tasks,
        team=team,
        promotion_policy=settings.promotion_policy,
    )
