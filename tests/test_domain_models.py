from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from loopgraph_supervisor.domain.models import (
    AgentBundle,
    EvaluationResult,
    EvolutionScope,
    Hint,
    HintPriority,
)


def test_agent_bundle_fingerprint_is_stable_and_content_addressed() -> None:
    first = AgentBundle(
        name="coding-agent",
        system_prompt="Fix the task and run tests.",
        skills={"testing": "Always reproduce the failure first."},
        context_config={"max_tokens": 32_000},
    )
    second = AgentBundle(
        name="coding-agent",
        system_prompt="Fix the task and run tests.",
        skills={"testing": "Always reproduce the failure first."},
        context_config={"max_tokens": 32_000},
    )
    changed = second.model_copy(update={"system_prompt": "Fix the task safely."})

    assert first.fingerprint == second.fingerprint
    assert changed.fingerprint != first.fingerprint
    assert first.evolution_scope == EvolutionScope.PROMPT_AND_SKILLS


def test_evaluation_requires_normalized_scores_and_preserves_hard_gates() -> None:
    result = EvaluationResult(
        grader_id="tests",
        score=0.92,
        passed=False,
        dimensions={"correctness": 1.0, "coverage": 0.84},
        hard_constraints={"security": True, "regression": False},
        feedback=("A regression test failed.",),
        retryable=True,
        confidence=1.0,
    )

    assert result.score == 0.92
    assert result.hard_constraints["regression"] is False

    with pytest.raises(ValidationError):
        EvaluationResult(grader_id="invalid", score=101, passed=True)


def test_hint_has_bounded_lifetime_and_deduplication_identity() -> None:
    hint = Hint(
        run_id=uuid4(),
        target="executor",
        priority=HintPriority.WARNING,
        instruction="Inspect authentication instead of retrying the endpoint.",
        reason="The same tool failed twice.",
        evidence=("event-12", "event-19"),
        expires_after_steps=2,
        deduplication_key="repeated-auth-failure",
    )

    assert hint.is_active(current_step=hint.created_at_step + 1)
    assert not hint.is_active(current_step=hint.created_at_step + 2)

    with pytest.raises(ValidationError):
        Hint(
            run_id=uuid4(),
            target="executor",
            instruction="Never expire.",
            reason="Invalid TTL.",
            expires_after_steps=0,
            deduplication_key="bad",
        )
