from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PromotionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    baseline_score: float = Field(alias="baseline", ge=0.0, le=1.0)
    candidate_score: float = Field(alias="candidate", ge=0.0, le=1.0)
    regressions: int = Field(default=0, ge=0)
    hard_constraints: dict[str, bool] = Field(default_factory=dict)
    benchmark_run_ids: tuple[UUID, ...] = ()
    baseline_cost: float | None = Field(default=None, ge=0.0)
    candidate_cost: float | None = Field(default=None, ge=0.0)


class PromotionDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    score_delta: float
    reasons: tuple[str, ...] = ()


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    minimum_candidate_score: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_score_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_regressions: int = Field(default=0, ge=0)
    maximum_cost_multiplier: float | None = Field(default=2.0, ge=1.0)

    def evaluate(self, evidence: PromotionEvidence) -> PromotionDecision:
        delta = round(evidence.candidate_score - evidence.baseline_score, 12)
        reasons: list[str] = []
        if evidence.candidate_score < self.minimum_candidate_score:
            reasons.append(
                f"candidate score {evidence.candidate_score:g} is below minimum "
                f"{self.minimum_candidate_score:g}"
            )
        if delta < self.minimum_score_delta:
            reasons.append(
                f"score delta {delta:g} is below minimum {self.minimum_score_delta:g}"
            )
        if evidence.regressions > self.maximum_regressions:
            reasons.append(
                f"regressions {evidence.regressions} exceed maximum "
                f"{self.maximum_regressions}"
            )
        reasons.extend(
            f"hard constraint {name!r} failed"
            for name, passed in evidence.hard_constraints.items()
            if not passed
        )
        if (
            self.maximum_cost_multiplier is not None
            and evidence.baseline_cost is not None
            and evidence.candidate_cost is not None
            and evidence.baseline_cost > 0
            and evidence.candidate_cost
            > evidence.baseline_cost * self.maximum_cost_multiplier
        ):
            reasons.append(
                f"candidate cost exceeds {self.maximum_cost_multiplier:g}x baseline"
            )
        return PromotionDecision(
            passed=not reasons,
            score_delta=delta,
            reasons=tuple(reasons),
        )
