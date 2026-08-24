from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

import pytest

from loopgraph_supervisor.application.observer import (
    HarnessSupervisorAdvisor,
    ObservationContext,
    SupervisorDirectiveAction,
)
from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.harness.base import (
    ExecutionRequest,
    ExecutionResult,
    HarnessCapabilities,
    HarnessEvent,
)


class SupervisorHarness:
    name = "supervisor-harness"
    capabilities = HarnessCapabilities(stream_events=True)

    def __init__(self) -> None:
        self.request: ExecutionRequest | None = None

    async def execute(
        self,
        request: ExecutionRequest,
        emit: Callable[[HarnessEvent], Awaitable[tuple[object, ...]]],
    ) -> ExecutionResult:
        self.request = request
        await emit(HarnessEvent(type="supervisor.reasoned", data={}))
        return ExecutionResult(
            session_id="supervisor",
            output={
                "directive": {
                    "action": "inject_hint",
                    "rationale": "Authentication is failing repeatedly.",
                    "hints": [
                        {
                            "instruction": "Inspect credentials.",
                            "reason": "Two unauthorized results.",
                            "deduplication_key": "auth",
                        }
                    ],
                }
            },
        )


@pytest.mark.asyncio
async def test_supervisor_agent_runs_in_an_isolated_harness_session() -> None:
    harness = SupervisorHarness()
    advisor = HarnessSupervisorAdvisor(
        harness=harness,
        bundle=AgentBundle(
            name="supervisor-agent",
            system_prompt="Return a safe structured directive.",
        ),
    )
    executor_run_id = uuid4()
    directive = await advisor.review(
        ObservationContext(
            run_id=executor_run_id,
            attempt=1,
            goal="Fetch data",
            agent_bundle=AgentBundle(name="executor", system_prompt="Fetch."),
            recent_events=(
                HarnessEvent(
                    type="tool/result", data={"is_error": True, "error": "unauthorized"}
                ),
            ),
        )
    )

    assert directive.action == SupervisorDirectiveAction.INJECT_HINT
    assert directive.hints[0].instruction == "Inspect credentials."
    assert harness.request is not None
    assert harness.request.run_id != executor_run_id
    assert harness.request.agent_bundle.name == "supervisor-agent"
