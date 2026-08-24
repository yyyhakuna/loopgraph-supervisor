from __future__ import annotations

import sys
from uuid import uuid4

import pytest

from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.harness.base import ExecutionRequest, HarnessEvent
from loopgraph_supervisor.harness.command import CommandHarnessAdapter, CommandHarnessError


@pytest.mark.asyncio
async def test_jsonl_command_harness_streams_events_and_returns_result() -> None:
    script = r"""
import json, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({"type":"event","event":{"type":"tool/call","data":{"name":"demo"}}}), flush=True)
print(json.dumps({
  "type":"result",
  "session_id":"external-session",
  "output":{"echo":request["goal"]},
  "usage":{"tokens":12},
  "checkpoint":"cp-1"
}), flush=True)
"""
    adapter = CommandHarnessAdapter(
        name="external-python",
        argv=(sys.executable, "-c", script),
    )
    events: list[HarnessEvent] = []

    async def emit(event: HarnessEvent) -> tuple[object, ...]:
        events.append(event)
        return ()

    result = await adapter.execute(
        ExecutionRequest(
            run_id=uuid4(),
            execution_id="external:1",
            attempt=1,
            goal="echo me",
            agent_bundle=AgentBundle(name="external", system_prompt="Echo."),
            timeout_seconds=2,
            max_steps=10,
        ),
        emit,
    )

    assert events[0].type == "tool/call"
    assert result.output == {"echo": "echo me"}
    assert result.usage == {"tokens": 12}
    assert result.checkpoint == "cp-1"


@pytest.mark.asyncio
async def test_jsonl_command_harness_rejects_protocol_noise() -> None:
    adapter = CommandHarnessAdapter(
        name="broken",
        argv=(sys.executable, "-c", "print('not-json', flush=True)"),
    )

    async def emit(_: HarnessEvent) -> tuple[object, ...]:
        return ()

    with pytest.raises(CommandHarnessError, match="invalid JSON"):
        await adapter.execute(
            ExecutionRequest(
                run_id=uuid4(),
                execution_id="broken:1",
                attempt=1,
                goal="fail",
                agent_bundle=AgentBundle(name="broken", system_prompt="Fail."),
                timeout_seconds=2,
                max_steps=10,
            ),
            emit,
        )
