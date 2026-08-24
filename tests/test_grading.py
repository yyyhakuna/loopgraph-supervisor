from __future__ import annotations

import sys
from uuid import uuid4

import pytest

from loopgraph_supervisor.domain.models import EvaluationResult
from loopgraph_supervisor.grading.base import EvaluationContext
from loopgraph_supervisor.grading.policy import GradingPolicy
from loopgraph_supervisor.grading.script import ScriptGrader, ScriptGraderError


def test_grading_policy_uses_weights_and_enforces_hard_constraints() -> None:
    policy = GradingPolicy(
        threshold=0.8,
        grader_weights={"tests": 0.7, "quality": 0.3},
        required_constraints=("security",),
    )
    decision = policy.aggregate(
        (
            EvaluationResult(
                grader_id="tests",
                score=1.0,
                passed=True,
                hard_constraints={"security": True},
            ),
            EvaluationResult(grader_id="quality", score=0.5, passed=True),
        )
    )

    assert decision.score == pytest.approx(0.85)
    assert decision.passed

    blocked = policy.aggregate(
        (
            EvaluationResult(
                grader_id="tests",
                score=1.0,
                passed=True,
                hard_constraints={"security": False},
            ),
            EvaluationResult(grader_id="quality", score=1.0, passed=True),
        )
    )
    assert not blocked.passed
    assert blocked.failed_constraints == ("security",)


@pytest.mark.asyncio
async def test_script_grader_receives_json_on_stdin_and_returns_evidence() -> None:
    script = """
import json, sys
payload = json.load(sys.stdin)
ok = payload["output"]["answer"] == 42
json.dump({
  "score": 1.0 if ok else 0.0,
  "passed": ok,
  "dimensions": {"correctness": 1.0 if ok else 0.0},
  "hard_constraints": {"schema": True},
  "evidence": ["answer checked"],
  "feedback": [] if ok else ["answer must be 42"]
}, sys.stdout)
"""
    grader = ScriptGrader(
        grader_id="answer-script",
        argv=(sys.executable, "-c", script),
        timeout_seconds=2,
    )
    result = await grader.evaluate(
        EvaluationContext(
            run_id=uuid4(),
            attempt=1,
            goal="Return the answer",
            output={"answer": 42},
        )
    )

    assert result.passed
    assert result.score == 1.0
    assert result.evidence == ("answer checked",)


@pytest.mark.asyncio
async def test_script_grader_times_out_without_leaking_a_process() -> None:
    grader = ScriptGrader(
        grader_id="slow-script",
        argv=(sys.executable, "-c", "import time; time.sleep(10)"),
        timeout_seconds=0.01,
    )

    with pytest.raises(ScriptGraderError, match="timed out"):
        await grader.evaluate(
            EvaluationContext(run_id=uuid4(), attempt=1, goal="timeout", output={})
        )


@pytest.mark.asyncio
async def test_script_grader_kills_process_as_soon_as_output_limit_is_crossed() -> None:
    script = "import sys,time; sys.stdout.write('x' * 100000); sys.stdout.flush(); time.sleep(10)"
    grader = ScriptGrader(
        grader_id="noisy-script",
        argv=(sys.executable, "-c", script),
        timeout_seconds=2,
        max_output_bytes=1024,
    )

    with pytest.raises(ScriptGraderError, match="output limit"):
        await grader.evaluate(
            EvaluationContext(run_id=uuid4(), attempt=1, goal="bounded", output={})
        )
