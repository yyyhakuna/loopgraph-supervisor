from __future__ import annotations

from uuid import uuid4

import pytest

from loopgraph_supervisor.grading.base import EvaluationContext
from loopgraph_supervisor.grading.rules import OutputRule, RuleGrader, RuleOperator


@pytest.mark.asyncio
async def test_rule_grader_scores_json_paths_and_fails_hard_constraints() -> None:
    grader = RuleGrader(
        grader_id="business-rules",
        threshold=0.5,
        rules=(
            OutputRule(
                id="refund_state",
                path="order.refund.status",
                operator=RuleOperator.EQUALS,
                expected="completed",
                weight=2,
            ),
            OutputRule(
                id="no_security_violation",
                path="security_violations",
                operator=RuleOperator.EQUALS,
                expected=0,
                hard_constraint=True,
            ),
            OutputRule(
                id="has_receipt",
                path="receipt_id",
                operator=RuleOperator.EXISTS,
            ),
        ),
    )

    result = await grader.evaluate(
        EvaluationContext(
            run_id=uuid4(),
            attempt=1,
            goal="Refund an order",
            output={
                "order": {"refund": {"status": "completed"}},
                "security_violations": 1,
                "receipt_id": "r-1",
            },
        )
    )

    assert result.score == pytest.approx(0.75)
    assert not result.passed
    assert result.hard_constraints == {"no_security_violation": False}
    assert result.dimensions["refund_state"] == 1.0


@pytest.mark.asyncio
async def test_rule_grader_supports_numeric_contains_and_regex_operators() -> None:
    grader = RuleGrader(
        grader_id="quality",
        threshold=1,
        rules=(
            OutputRule(id="latency", path="latency_ms", operator="lte", expected=500),
            OutputRule(id="tags", path="tags", operator="contains", expected="verified"),
            OutputRule(
                id="ticket", path="ticket", operator="regex", expected=r"^INC-[0-9]+$"
            ),
        ),
    )
    result = await grader.evaluate(
        EvaluationContext(
            run_id=uuid4(),
            attempt=1,
            goal="Resolve incident",
            output={"latency_ms": 200, "tags": ["verified"], "ticket": "INC-42"},
        )
    )

    assert result.passed
    assert result.score == 1
