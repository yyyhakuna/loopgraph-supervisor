from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EvolutionScope(StrEnum):
    """The mutation boundary granted to an evolution policy."""

    HINT_ONLY = "hint_only"
    PROMPT_ONLY = "prompt_only"
    PROMPT_AND_SKILLS = "prompt_and_skills"
    FULL_AGENT_BUNDLE = "full_agent_bundle"


class HintPriority(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AgentBundle(BaseModel):
    """Immutable, content-addressed configuration executed by a harness."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=128)
    system_prompt: str = Field(min_length=1)
    skills: dict[str, str] = Field(default_factory=dict)
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    model_config_data: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    context_config: dict[str, Any] = Field(default_factory=dict)
    memory_config: dict[str, Any] = Field(default_factory=dict)
    workflow_config: dict[str, Any] = Field(default_factory=dict)
    evolution_scope: EvolutionScope = EvolutionScope.PROMPT_AND_SKILLS

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EvaluationResult(BaseModel):
    """Evidence-bearing output from one grader."""

    model_config = ConfigDict(frozen=True)

    grader_id: str = Field(min_length=1, max_length=128)
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    dimensions: dict[str, float] = Field(default_factory=dict)
    hard_constraints: dict[str, bool] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    feedback: tuple[str, ...] = ()
    retryable: bool = False
    suggested_action: str | None = None


class Hint(BaseModel):
    """Structured, expiring feedback delivered through the supervisor bus."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    target: str = Field(min_length=1, max_length=128)
    priority: HintPriority = HintPriority.INFO
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    created_at_step: int = Field(default=0, ge=0)
    expires_after_steps: int = Field(default=1, ge=1)
    deduplication_key: str = Field(min_length=1, max_length=256)

    def is_active(self, current_step: int) -> bool:
        return self.created_at_step <= current_step < (
            self.created_at_step + self.expires_after_steps
        )
