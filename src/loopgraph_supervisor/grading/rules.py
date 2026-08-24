from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loopgraph_supervisor.domain.models import EvaluationResult
from loopgraph_supervisor.grading.base import EvaluationContext


class RuleOperator(StrEnum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    CONTAINS = "contains"
    REGEX = "regex"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class OutputRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1)
    operator: RuleOperator
    expected: Any = None
    weight: float = Field(default=1.0, gt=0)
    hard_constraint: bool = False
    feedback: str | None = None

    @model_validator(mode="after")
    def require_expected_value(self) -> OutputRule:
        if self.operator not in {RuleOperator.EXISTS, RuleOperator.NOT_EXISTS}:
            if self.expected is None:
                raise ValueError(f"Operator {self.operator.value} requires expected")
        return self


_MISSING = object()


class RuleGrader:
    def __init__(
        self,
        *,
        grader_id: str,
        rules: tuple[OutputRule, ...],
        threshold: float = 1.0,
    ) -> None:
        if not grader_id or not rules:
            raise ValueError("grader_id and at least one rule are required")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be normalized")
        if len({rule.id for rule in rules}) != len(rules):
            raise ValueError("Rule ids must be unique")
        self._id = grader_id
        self.rules = rules
        self.threshold = threshold

    @property
    def id(self) -> str:
        return self._id

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        outcomes: dict[str, bool] = {}
        hard_constraints: dict[str, bool] = {}
        feedback: list[str] = []
        for rule in self.rules:
            actual = self._resolve(context.output, rule.path)
            passed = self._matches(rule, actual)
            outcomes[rule.id] = passed
            if rule.hard_constraint:
                hard_constraints[rule.id] = passed
            if not passed:
                feedback.append(
                    rule.feedback
                    or f"Rule {rule.id!r} failed at {rule.path!r} using {rule.operator.value}"
                )
        total_weight = sum(rule.weight for rule in self.rules)
        score = sum(rule.weight for rule in self.rules if outcomes[rule.id]) / total_weight
        hard_passed = all(hard_constraints.values())
        return EvaluationResult(
            grader_id=self.id,
            score=score,
            passed=score >= self.threshold and hard_passed,
            dimensions={rule_id: float(passed) for rule_id, passed in outcomes.items()},
            hard_constraints=hard_constraints,
            feedback=tuple(feedback),
            retryable=not hard_passed or score < self.threshold,
        )

    @staticmethod
    def _resolve(value: Any, path: str) -> Any:
        current = value
        for segment in path.split("."):
            if isinstance(current, dict) and segment in current:
                current = current[segment]
            elif isinstance(current, list) and segment.isdigit():
                index = int(segment)
                if index >= len(current):
                    return _MISSING
                current = current[index]
            else:
                return _MISSING
        return current

    @staticmethod
    def _matches(rule: OutputRule, actual: Any) -> bool:
        if rule.operator == RuleOperator.EXISTS:
            return actual is not _MISSING
        if rule.operator == RuleOperator.NOT_EXISTS:
            return actual is _MISSING
        if actual is _MISSING:
            return False
        try:
            match rule.operator:
                case RuleOperator.EQUALS:
                    return bool(actual == rule.expected)
                case RuleOperator.NOT_EQUALS:
                    return bool(actual != rule.expected)
                case RuleOperator.CONTAINS:
                    return bool(rule.expected in actual)
                case RuleOperator.REGEX:
                    return re.search(str(rule.expected), str(actual)) is not None
                case RuleOperator.GT:
                    return bool(actual > rule.expected)
                case RuleOperator.GTE:
                    return bool(actual >= rule.expected)
                case RuleOperator.LT:
                    return bool(actual < rule.expected)
                case RuleOperator.LTE:
                    return bool(actual <= rule.expected)
                case _:
                    return False
        except (TypeError, ValueError):
            return False
