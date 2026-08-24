from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loopgraph_supervisor.domain.models import AgentBundle, EvolutionScope


class MutationNotAllowedError(PermissionError):
    pass


class NoMutationError(ValueError):
    pass


class MutationPlan(BaseModel):
    """A declarative patch; protected Supervisor and grader state is absent by design."""

    model_config = ConfigDict(frozen=True)

    rationale: str = Field(min_length=1)
    system_prompt: str | None = Field(default=None, min_length=1)
    skills_upsert: dict[str, str] = Field(default_factory=dict)
    skills_remove: tuple[str, ...] = ()
    mcp_servers_upsert: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mcp_servers_remove: tuple[str, ...] = ()
    tool_policy_patch: dict[str, Any] = Field(default_factory=dict)
    model_config_patch: dict[str, Any] = Field(default_factory=dict)
    context_patch: dict[str, Any] = Field(default_factory=dict)
    memory_patch: dict[str, Any] = Field(default_factory=dict)
    workflow_patch: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_change(self) -> MutationPlan:
        if not any(
            (
                self.system_prompt is not None,
                self.skills_upsert,
                self.skills_remove,
                self.mcp_servers_upsert,
                self.mcp_servers_remove,
                self.tool_policy_patch,
                self.model_config_patch,
                self.context_patch,
                self.memory_patch,
                self.workflow_patch,
            )
        ):
            raise ValueError("A mutation plan must contain at least one change")
        return self

    def changed_components(self) -> tuple[str, ...]:
        changes: list[str] = []
        if self.system_prompt is not None:
            changes.append("system_prompt")
        if self.skills_upsert or self.skills_remove:
            changes.append("skills")
        if self.mcp_servers_upsert or self.mcp_servers_remove:
            changes.append("mcp_servers")
        for component, patch in (
            ("tool_policy", self.tool_policy_patch),
            ("model_config", self.model_config_patch),
            ("context_config", self.context_patch),
            ("memory_config", self.memory_patch),
            ("workflow_config", self.workflow_patch),
        ):
            if patch:
                changes.append(component)
        return tuple(changes)


class BundleMutator:
    _allowed_components: ClassVar[dict[EvolutionScope, frozenset[str]]] = {
        EvolutionScope.HINT_ONLY: frozenset(),
        EvolutionScope.PROMPT_ONLY: frozenset({"system_prompt"}),
        EvolutionScope.PROMPT_AND_SKILLS: frozenset({"system_prompt", "skills"}),
        EvolutionScope.FULL_AGENT_BUNDLE: frozenset(
            {
                "system_prompt",
                "skills",
                "mcp_servers",
                "tool_policy",
                "model_config",
                "context_config",
                "memory_config",
                "workflow_config",
            }
        ),
    }

    def apply(self, parent: AgentBundle, plan: MutationPlan) -> AgentBundle:
        changed = set(plan.changed_components())
        disallowed = changed.difference(self._allowed_components[parent.evolution_scope])
        if disallowed:
            names = ", ".join(sorted(disallowed))
            raise MutationNotAllowedError(
                f"Evolution scope {parent.evolution_scope.value!r} does not allow: {names}"
            )

        skills = dict(parent.skills)
        for name in plan.skills_remove:
            skills.pop(name, None)
        skills.update(plan.skills_upsert)

        mcp_servers = deepcopy(parent.mcp_servers)
        for name in plan.mcp_servers_remove:
            mcp_servers.pop(name, None)
        mcp_servers.update(deepcopy(plan.mcp_servers_upsert))

        candidate = parent.model_copy(
            update={
                "system_prompt": plan.system_prompt or parent.system_prompt,
                "skills": skills,
                "mcp_servers": mcp_servers,
                "tool_policy": self._deep_merge(parent.tool_policy, plan.tool_policy_patch),
                "model_config_data": self._deep_merge(
                    parent.model_config_data, plan.model_config_patch
                ),
                "context_config": self._deep_merge(
                    parent.context_config, plan.context_patch
                ),
                "memory_config": self._deep_merge(parent.memory_config, plan.memory_patch),
                "workflow_config": self._deep_merge(
                    parent.workflow_config, plan.workflow_patch
                ),
            }
        )
        if candidate.fingerprint == parent.fingerprint:
            raise NoMutationError("Mutation plan did not change the agent bundle")
        return candidate

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
