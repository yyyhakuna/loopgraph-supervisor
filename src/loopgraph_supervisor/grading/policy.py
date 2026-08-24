from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from loopgraph_supervisor.domain.models import EvaluationResult


class GradingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failed_constraints: tuple[str, ...] = ()
    results: tuple[EvaluationResult, ...]


class GradingPolicy(BaseModel):
    """Combines evidence without granting a grader control-plane authority."""

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    grader_weights: dict[str, float] = Field(default_factory=dict)
    required_constraints: tuple[str, ...] = ()

    def aggregate(self, results: tuple[EvaluationResult, ...]) -> GradingDecision:
        if not results:
            raise ValueError("At least one evaluation result is required")

        weights = [self.grader_weights.get(result.grader_id, 1.0) for result in results]
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Grader weights must be non-negative with a positive total")
        score = sum(
            result.score * weight
            for result, weight in zip(results, weights, strict=True)
        ) / sum(weights)

        failed_constraints = tuple(
            constraint
            for constraint in self.required_constraints
            if not self._constraint_passed(constraint, results)
        )
        return GradingDecision(
            score=score,
            passed=score >= self.threshold and not failed_constraints,
            failed_constraints=failed_constraints,
            results=results,
        )

    @staticmethod
    def _constraint_passed(
        constraint: str, results: tuple[EvaluationResult, ...]
    ) -> bool:
        reported = [
            result.hard_constraints[constraint]
            for result in results
            if constraint in result.hard_constraints
        ]
        return bool(reported) and all(reported)
