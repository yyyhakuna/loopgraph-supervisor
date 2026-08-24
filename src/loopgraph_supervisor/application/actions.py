from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from loopgraph_supervisor.application.observer import (
    ObservationContext,
    SupervisorDirective,
    SupervisorDirectiveAction,
)
from loopgraph_supervisor.application.team import ChildAgentSpec, TeamCoordinator
from loopgraph_supervisor.evolution.mutator import BundleMutator, MutationPlan
from loopgraph_supervisor.infrastructure.version_store import SqlAgentVersionStore


class ActionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: SupervisorDirectiveAction
    child_run_id: UUID | None = None
    candidate_version_id: UUID | None = None


class SupervisorActionRouter:
    """Executes only the explicitly supported, schema-validated smart actions."""

    def __init__(
        self,
        *,
        team: TeamCoordinator,
        versions: SqlAgentVersionStore,
        mutator: BundleMutator | None = None,
    ) -> None:
        self.team = team
        self.versions = versions
        self.mutator = mutator or BundleMutator()

    async def handle(
        self, context: ObservationContext, directive: SupervisorDirective
    ) -> ActionOutcome:
        if directive.action == SupervisorDirectiveAction.SPAWN_SUBAGENT:
            raw_spec = directive.payload.get("child") or directive.payload
            child = ChildAgentSpec.model_validate(raw_spec)
            child_run_id = await self.team.spawn(
                context.run_id,
                child,
                actor="supervisor-agent",
                start=bool(directive.payload.get("start", True)),
            )
            return ActionOutcome(action=directive.action, child_run_id=child_run_id)

        if directive.action == SupervisorDirectiveAction.PROPOSE_MUTATION:
            requested_parent = directive.payload.get("parent_version_id")
            if context.agent_version_id is None:
                raise ValueError(
                    "The observed run is not linked to a versioned Agent Bundle"
                )
            parent_id = context.agent_version_id
            if requested_parent is not None and UUID(str(requested_parent)) != parent_id:
                raise ValueError(
                    "Supervisor cannot mutate a version other than the observed run's version"
                )
            plan = MutationPlan.model_validate(directive.payload["plan"])
            parent = await self.versions.get(parent_id)
            if parent.bundle.fingerprint != context.agent_bundle.fingerprint:
                raise ValueError(
                    "Mutation parent does not match the Agent Bundle under observation"
                )
            candidate_bundle = self.mutator.apply(parent.bundle, plan)
            candidate = await self.versions.create_candidate(
                parent_id=parent.id,
                bundle=candidate_bundle,
                actor="supervisor-agent",
                change_summary=plan.rationale,
            )
            return ActionOutcome(
                action=directive.action,
                candidate_version_id=candidate.id,
            )

        return ActionOutcome(action=directive.action)
