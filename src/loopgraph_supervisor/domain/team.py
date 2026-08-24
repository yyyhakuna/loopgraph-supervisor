from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from loopgraph_supervisor.domain.events import StoredEvent


class TeamMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID
    role: str
    goal: str
    status: str = "created"


class TeamContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_id: UUID
    leader_run_id: UUID
    members: tuple[TeamMember, ...] = ()

    @classmethod
    def replay(cls, events: tuple[StoredEvent, ...]) -> TeamContext:
        created = next((event for event in events if event.type == "team.created"), None)
        if created is None:
            raise KeyError("Run has no team context")
        members: dict[UUID, TeamMember] = {}
        for event in events:
            if event.type == "team.member.spawned":
                member = TeamMember.model_validate(event.data["member"])
                members[member.run_id] = member
            elif event.type == "team.member.completed":
                run_id = UUID(event.data["run_id"])
                current = members.get(run_id)
                if current is not None:
                    members[run_id] = current.model_copy(update={"status": "completed"})
            elif event.type == "team.member.failed":
                run_id = UUID(event.data["run_id"])
                current = members.get(run_id)
                if current is not None:
                    members[run_id] = current.model_copy(update={"status": "failed"})
        return cls(
            team_id=created.data["team_id"],
            leader_run_id=created.run_id,
            members=tuple(members.values()),
        )

