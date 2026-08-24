from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from types import SimpleNamespace
from uuid import uuid4

import pytest

from loopgraph_supervisor.domain.models import AgentBundle, Hint
from loopgraph_supervisor.harness.base import ExecutionRequest, HarnessEvent
from loopgraph_supervisor.harness.dsh import (
    DeepSeekHarnessAdapter,
    DshUnsupportedConfigurationError,
)


class FakeDshSdk:
    def __init__(self) -> None:
        self.input: str | None = None
        self.session_id: str | None = None

    def run(
        self,
        input: str,
        *,
        session_id: str,
        on_notification: Callable[[object], None],
    ) -> object:
        self.input = input
        self.session_id = session_id
        on_notification(
            SimpleNamespace(
                method="session.event",
                payload={
                    "sessionId": session_id,
                    "event": {"type": "tool/call", "data": {"name": "read_file"}},
                },
            )
        )
        on_notification(
            SimpleNamespace(
                method="session.status",
                payload={"sessionId": session_id, "status": "idle"},
            )
        )
        return SimpleNamespace(
            session_id=session_id,
            final_response="completed",
            finish_reason="stop",
            events=[],
            session_root="/tmp/dsh-sessions",
        )


class ClosingDshSdk(FakeDshSdk):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class BlockingDshSdk:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.closed = False

    def run(self, input: str, *, session_id: str, on_notification) -> object:
        self.started.set()
        self.stopped.wait(timeout=2)
        return SimpleNamespace(
            session_id=session_id,
            final_response="stopped",
            finish_reason="cancelled",
            session_root=None,
        )

    def close(self) -> None:
        self.closed = True
        self.stopped.set()


@pytest.mark.asyncio
async def test_dsh_adapter_bridges_sync_sdk_notifications_without_coupling_core() -> None:
    sdk = FakeDshSdk()
    adapter = DeepSeekHarnessAdapter(sdk=sdk)
    seen: list[HarnessEvent] = []
    run_id = uuid4()
    hint = Hint(
        run_id=run_id,
        target="executor",
        instruction="Inspect authentication.",
        reason="Repeated tool failure.",
        deduplication_key="auth",
    )

    async def emit(event: HarnessEvent) -> tuple[object, ...]:
        seen.append(event)
        return ()

    result = await adapter.execute(
        ExecutionRequest(
            run_id=run_id,
            execution_id=f"{run_id}:1",
            attempt=1,
            goal="Repair checkout",
            agent_bundle=AgentBundle(
                name="coder",
                system_prompt="Work carefully.",
                skills={"testing": "Run tests."},
            ),
            hints=(hint,),
            timeout_seconds=10,
            max_steps=20,
        ),
        emit,
    )

    assert sdk.session_id == f"loopgraph-{run_id.hex}"
    assert "Work carefully." in (sdk.input or "")
    assert "Inspect authentication." in (sdk.input or "")
    assert seen[0].type == "dsh.tool/call"
    assert result.output == {"text": "completed", "finish_reason": "stop"}
    assert result.checkpoint == sdk.session_id


@pytest.mark.asyncio
async def test_dsh_adapter_compiles_versioned_runtime_config_and_closes_runtime() -> None:
    captured: list[dict[str, object]] = []
    sdk = ClosingDshSdk()

    def factory(config: dict[str, object]) -> ClosingDshSdk:
        captured.append(config)
        return sdk

    adapter = DeepSeekHarnessAdapter(
        name="dsh-executor",
        sdk_factory=factory,
        sdk_config={"session_root": "/tmp/base-sessions", "env": {"BASE": "1"}},
    )
    run_id = uuid4()

    async def emit(_: HarnessEvent) -> tuple[object, ...]:
        return ()

    await adapter.execute(
        ExecutionRequest(
            run_id=run_id,
            execution_id=f"{run_id}:1",
            attempt=1,
            goal="Repair checkout",
            agent_bundle=AgentBundle(
                name="coder",
                system_prompt="Versioned persona.",
                model_config={"provider": "internal", "model": "model-v2"},
                context_config={"max_output_tokens": 8192},
                workflow_config={"dsh_cordis": "/tmp/version-v2.cordis.yml"},
            ),
            timeout_seconds=10,
            max_steps=20,
        ),
        emit,
    )

    assert adapter.name == "dsh-executor"
    assert captured == [
        {
            "session_root": "/tmp/base-sessions",
            "env": {"BASE": "1", "DSH_SYSTEM_PROMPT": "Versioned persona."},
            "provider": "internal",
            "model": "model-v2",
            "max_tokens": 8192,
            "cordis": "/tmp/version-v2.cordis.yml",
        }
    ]
    assert sdk.closed is True


@pytest.mark.asyncio
async def test_dsh_adapter_rejects_bundle_features_it_cannot_materialize() -> None:
    adapter = DeepSeekHarnessAdapter(sdk=FakeDshSdk())
    run_id = uuid4()

    async def emit(_: HarnessEvent) -> tuple[object, ...]:
        return ()

    with pytest.raises(DshUnsupportedConfigurationError, match="mcp_servers"):
        await adapter.execute(
            ExecutionRequest(
                run_id=run_id,
                execution_id=f"{run_id}:1",
                attempt=1,
                goal="Repair checkout",
                agent_bundle=AgentBundle(
                    name="coder",
                    system_prompt="Work.",
                    mcp_servers={"github": {"transport": "stdio"}},
                ),
                timeout_seconds=10,
                max_steps=20,
            ),
            emit,
        )


@pytest.mark.asyncio
async def test_dsh_cancellation_closes_runtime_before_returning() -> None:
    sdk = BlockingDshSdk()
    adapter = DeepSeekHarnessAdapter(sdk=sdk, shutdown_timeout_seconds=1)
    run_id = uuid4()

    async def emit(_: HarnessEvent) -> tuple[object, ...]:
        return ()

    request = ExecutionRequest(
        run_id=run_id,
        execution_id=f"{run_id}:1",
        attempt=1,
        goal="Wait",
        agent_bundle=AgentBundle(name="coder", system_prompt="Wait."),
        timeout_seconds=10,
        max_steps=20,
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(adapter.execute(request, emit), timeout=0.05)

    assert sdk.closed is True
    assert sdk.stopped.is_set()
