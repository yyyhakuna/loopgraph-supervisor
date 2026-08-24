from __future__ import annotations

import pytest

from loopgraph_supervisor.domain.models import AgentBundle, EvolutionScope
from loopgraph_supervisor.evolution.mutator import (
    BundleMutator,
    MutationNotAllowedError,
    MutationPlan,
)


def test_prompt_and_skill_scope_can_create_a_new_skill_without_mutating_parent() -> None:
    parent = AgentBundle(
        name="coding-agent",
        system_prompt="Fix the task.",
        skills={"testing": "Run tests."},
        evolution_scope=EvolutionScope.PROMPT_AND_SKILLS,
    )
    plan = MutationPlan(
        rationale="Failures show the agent edits before reproducing.",
        system_prompt="Reproduce the failure, then fix the task.",
        skills_upsert={"debugging": "Reproduce before editing."},
    )

    candidate = BundleMutator().apply(parent, plan)

    assert candidate.system_prompt.startswith("Reproduce")
    assert candidate.skills["debugging"] == "Reproduce before editing."
    assert "debugging" not in parent.skills
    assert candidate.fingerprint != parent.fingerprint


def test_prompt_and_skill_scope_rejects_context_or_mcp_mutation() -> None:
    parent = AgentBundle(
        name="coding-agent",
        system_prompt="Fix the task.",
        evolution_scope=EvolutionScope.PROMPT_AND_SKILLS,
    )

    with pytest.raises(MutationNotAllowedError, match="context_config"):
        BundleMutator().apply(
            parent,
            MutationPlan(
                rationale="Need more context.",
                context_patch={"max_tokens": 64_000},
            ),
        )

    with pytest.raises(MutationNotAllowedError, match="mcp_servers"):
        BundleMutator().apply(
            parent,
            MutationPlan(
                rationale="Need a browser.",
                mcp_servers_upsert={"browser": {"command": "browser-mcp"}},
            ),
        )


def test_full_scope_supports_context_memory_workflow_and_mcp_changes() -> None:
    parent = AgentBundle(
        name="coding-agent",
        system_prompt="Fix the task.",
        context_config={"max_tokens": 32_000, "compression": {"enabled": True}},
        memory_config={"episodic": {"enabled": False}},
        workflow_config={"nodes": ["execute", "verify"]},
        evolution_scope=EvolutionScope.FULL_AGENT_BUNDLE,
    )
    candidate = BundleMutator().apply(
        parent,
        MutationPlan(
            rationale="The candidate needs retrieval and a security checker.",
            context_patch={"retrieval": {"top_k": 8}},
            memory_patch={"episodic": {"enabled": True}},
            workflow_patch={"nodes": ["execute", "security", "verify"]},
            mcp_servers_upsert={"docs": {"transport": "stdio", "command": "docs-mcp"}},
        ),
    )

    assert candidate.context_config["compression"]["enabled"] is True
    assert candidate.context_config["retrieval"]["top_k"] == 8
    assert candidate.memory_config["episodic"]["enabled"] is True
    assert candidate.workflow_config["nodes"] == ["execute", "security", "verify"]
    assert candidate.mcp_servers["docs"]["command"] == "docs-mcp"
