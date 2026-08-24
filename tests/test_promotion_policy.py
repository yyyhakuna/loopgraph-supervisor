from __future__ import annotations

from loopgraph_supervisor.evolution.promotion import PromotionEvidence, PromotionPolicy


def test_promotion_policy_requires_score_delta_hard_gates_and_no_regressions() -> None:
    policy = PromotionPolicy(
        minimum_candidate_score=0.8,
        minimum_score_delta=0.05,
        maximum_regressions=0,
    )

    passing = policy.evaluate(
        PromotionEvidence(
            baseline=0.75,
            candidate=0.85,
            regressions=0,
            hard_constraints={"security": True, "tests": True},
        )
    )
    blocked = policy.evaluate(
        PromotionEvidence(
            baseline=0.75,
            candidate=0.9,
            regressions=1,
            hard_constraints={"security": False},
        )
    )

    assert passing.passed
    assert passing.score_delta == 0.1
    assert not blocked.passed
    assert set(blocked.reasons) == {
        "regressions 1 exceed maximum 0",
        "hard constraint 'security' failed",
    }
