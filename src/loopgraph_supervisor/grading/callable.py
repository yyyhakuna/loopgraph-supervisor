from __future__ import annotations

from collections.abc import Awaitable, Callable

from loopgraph_supervisor.domain.models import EvaluationResult
from loopgraph_supervisor.grading.base import EvaluationContext


class CallableGrader:
    """Small adapter for application-defined async scoring functions."""

    def __init__(
        self,
        grader_id: str,
        function: Callable[[EvaluationContext], Awaitable[EvaluationResult]],
    ) -> None:
        self._id = grader_id
        self._function = function

    @property
    def id(self) -> str:
        return self._id

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        result = await self._function(context)
        if result.grader_id != self.id:
            result = result.model_copy(update={"grader_id": self.id})
        return result

