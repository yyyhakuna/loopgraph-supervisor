from __future__ import annotations

import asyncio
from collections import defaultdict
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.tasks import RunTaskManager
from loopgraph_supervisor.domain.events import NewEvent
from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.domain.runs import RunDefinition
from loopgraph_supervisor.domain.team import TeamContext, TeamMember
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore


class ChildAgentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str = Field(min_length=1, max_length=128)
    goal: str = Field(min_length=1)
    harness_id: str | None = None
    agent_bundle: AgentBundle | None = None
    grader_ids: tuple[str, ...] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SubagentLimitError(RuntimeError):
    pass


class TeamCoordinator:
    """Creates child runs while keeping team state in the leader's event stream."""

    def __init__(
        self,
        engine: SupervisorEngine,
        event_store: SqlEventStore,
        tasks: RunTaskManager,
    ) -> None:
        self.engine = engine
        self.events = event_store
        self.tasks = tasks
        self._locks: defaultdict[UUID, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def spawn(
        self,
        parent_run_id: UUID,
        spec: ChildAgentSpec,
        *,
        actor: str,
        start: bool = True,
    ) -> UUID:
        async with self._locks[parent_run_id]:
            parent = await self.engine.get_run(parent_run_id)
            parent_events = await self.events.load(parent_run_id)
            try:
                context = TeamContext.replay(parent_events)
                team_id = context.team_id
                member_count = len(context.members)
            except KeyError:
                team_id = uuid4()
                member_count = 0
                await self.engine.append_control_event(
                    parent_run_id,
                    NewEvent(
                        type="team.created",
                        data={"team_id": str(team_id), "actor": actor},
                    ),
                )

            policy = parent.definition.supervisor_policy
            if member_count >= policy.max_subagents:
                raise SubagentLimitError(
                    f"Run {parent_run_id} reached max_subagents={policy.max_subagents}"
                )
            if parent.definition.depth >= policy.max_subagent_depth:
                raise SubagentLimitError(
                    f"Run {parent_run_id} reached max_subagent_depth={policy.max_subagent_depth}"
                )

            child_id = uuid4()
            await self.engine.append_control_event(
                parent_run_id,
                NewEvent(
                    type="team.member.spawn_requested",
                    data={
                        "child_run_id": str(child_id),
                        "role": spec.role,
                        "goal": spec.goal,
                        "actor": actor,
                    },
                ),
            )
            child_definition = RunDefinition(
                goal=spec.goal,
                harness_id=spec.harness_id or parent.definition.harness_id,
                agent_bundle=spec.agent_bundle or parent.definition.agent_bundle,
                agent_version_id=(
                    None
                    if spec.agent_bundle is not None
                    else parent.definition.agent_version_id
                ),
                grader_ids=spec.grader_ids or parent.definition.grader_ids,
                grading_policy=parent.definition.grading_policy,
                supervisor_policy=policy,
                parent_run_id=parent_run_id,
                team_id=team_id,
                role=spec.role,
                depth=parent.definition.depth + 1,
                metadata={**parent.definition.metadata, **spec.metadata},
            )
            try:
                await self.engine.create_run(child_definition, run_id=child_id)
            except Exception as error:
                await self.engine.append_control_event(
                    parent_run_id,
                    NewEvent(
                        type="team.member.spawn_failed",
                        data={
                            "child_run_id": str(child_id),
                            "error": {"type": type(error).__name__, "message": str(error)[:2000]},
                        },
                    ),
                )
                raise

            member = TeamMember(run_id=child_id, role=spec.role, goal=spec.goal)
            await self.engine.append_control_event(
                parent_run_id,
                NewEvent(
                    type="team.member.spawned",
                    data={"member": member.model_dump(mode="json"), "actor": actor},
                ),
            )
            if start:
                await self.tasks.start(child_id)
            return child_id

    async def context(self, leader_run_id: UUID) -> TeamContext:
        events = await self.events.load(leader_run_id)
        return TeamContext.replay(events)
