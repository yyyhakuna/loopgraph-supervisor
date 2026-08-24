from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
from collections.abc import Callable
from typing import Any, Protocol, cast

from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.harness.base import (
    EventSink,
    ExecutionRequest,
    ExecutionResult,
    HarnessCapabilities,
    HarnessEvent,
)


class DshSdkUnavailableError(RuntimeError):
    pass


class DshUnsupportedConfigurationError(ValueError):
    pass


class DshSdkProtocol(Protocol):
    def run(
        self,
        input: str,
        *,
        session_id: str,
        on_notification: Callable[[object], None],
    ) -> object: ...


DshSdkFactory = Callable[[dict[str, Any]], DshSdkProtocol]


class DeepSeekHarnessAdapter:
    """Isolated, cancellable bridge for the official synchronous DSH SDK.

    The published SDK has no prompt-cancel method. Each owned execution therefore
    gets its own runtime process; cancellation closes that runtime and waits for
    the worker thread to quiesce before control returns to the Supervisor.
    """

    capabilities = HarnessCapabilities(
        stream_events=True,
        # The Python SDK wire has no agent.inject method. Observer hints are
        # deliberately queued for the next attempt by the engine.
        inject_context=False,
        pause_resume=False,
        checkpoints=True,
        subagents=True,
        workspace_snapshots=False,
    )

    def __init__(
        self,
        *,
        name: str = "deepseek-harness",
        sdk: DshSdkProtocol | None = None,
        sdk_config: dict[str, Any] | None = None,
        sdk_factory: DshSdkFactory | None = None,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        if not name:
            raise ValueError("DSH adapter name is required")
        if sdk is not None and sdk_factory is not None:
            raise ValueError("Pass either sdk or sdk_factory, not both")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self.name = name
        self._shared_sdk = sdk
        self._sdk_config = dict(sdk_config or {})
        self._sdk_factory = sdk_factory or self._load_sdk
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

    @staticmethod
    def _load_sdk(config: dict[str, Any]) -> DshSdkProtocol:
        try:
            module = importlib.import_module("deepseek_harness")
        except ImportError as error:
            raise DshSdkUnavailableError(
                "Install the DeepSeek Harness Python SDK or inject an SDK instance"
            ) from error
        sdk_class = module.DeepSeekHarness
        return cast(DshSdkProtocol, sdk_class(**config))

    async def execute(self, request: ExecutionRequest, emit: EventSink) -> ExecutionResult:
        loop = asyncio.get_running_loop()
        session_id = request.resume_token or f"loopgraph-{request.run_id.hex}"
        prompt = self._build_input(request)
        runtime_config = self._compile_runtime_config(request.agent_bundle)
        sdk = self._shared_sdk or self._sdk_factory(runtime_config)
        owned = self._shared_sdk is None

        def on_notification(notification: object) -> None:
            method, payload = self._notification_parts(notification)
            if method != "session.event" or payload.get("sessionId") != session_id:
                return
            raw_event = payload.get("event")
            if not isinstance(raw_event, dict):
                return
            event_type = raw_event.get("type")
            event_data = raw_event.get("data", {})
            if not isinstance(event_type, str) or not isinstance(event_data, dict):
                return
            future: concurrent.futures.Future[tuple[object, ...]] = (
                asyncio.run_coroutine_threadsafe(
                    emit(HarnessEvent(type=f"dsh.{event_type}", data=event_data)),
                    loop,
                )
            )
            # Backpressure preserves durable event order and propagates control
            # signals into the SDK callback.
            future.result(timeout=request.timeout_seconds)

        worker = asyncio.create_task(
            asyncio.to_thread(
                sdk.run,
                prompt,
                session_id=session_id,
                on_notification=on_notification,
            )
        )
        closed = False
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await self._close_runtime(sdk, required=True)
            closed = True
            try:
                await asyncio.wait_for(
                    asyncio.shield(worker), timeout=self._shutdown_timeout_seconds
                )
            except Exception:
                # Preserve cancellation while ensuring no worker exception leaks.
                if not worker.done():
                    worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            raise
        finally:
            if owned and not closed:
                await self._close_runtime(sdk, required=False)

        final_response = str(getattr(result, "final_response", ""))
        finish_reason = getattr(result, "finish_reason", None)
        session_root = getattr(result, "session_root", None)
        artifacts = (
            {"dsh_session_root": session_root}
            if isinstance(session_root, str)
            else {}
        )
        return ExecutionResult(
            session_id=str(getattr(result, "session_id", session_id)),
            output={"text": final_response, "finish_reason": finish_reason},
            artifacts=artifacts,
            checkpoint=session_id,
        )

    async def _close_runtime(self, sdk: DshSdkProtocol, *, required: bool) -> None:
        close = getattr(sdk, "close", None)
        if not callable(close):
            if required:
                raise RuntimeError(
                    "DSH runtime does not expose close(); safe cancellation is impossible"
                )
            return
        await asyncio.wait_for(
            asyncio.to_thread(close), timeout=self._shutdown_timeout_seconds
        )

    def _compile_runtime_config(self, bundle: AgentBundle) -> dict[str, Any]:
        unsupported = [
            name
            for name, value in (
                ("mcp_servers", bundle.mcp_servers),
                ("tool_policy", bundle.tool_policy),
            )
            if value
        ]
        if unsupported:
            raise DshUnsupportedConfigurationError(
                "The Python DSH SDK cannot materialize bundle fields: "
                + ", ".join(unsupported)
                + ". Supply an explicit dsh_cordis composition instead."
            )

        model_allowed = {"provider", "model", "max_tokens", "base_url"}
        model_unknown = set(bundle.model_config_data).difference(model_allowed)
        context_allowed = {"max_output_tokens"}
        context_unknown = set(bundle.context_config).difference(context_allowed)
        workflow_mapping = {
            "dsh_cordis": "cordis",
            "dsh_cwd": "cwd",
            "dsh_runtime_cwd": "runtime_cwd",
            "dsh_session_root": "session_root",
        }
        workflow_unknown = set(bundle.workflow_config).difference(workflow_mapping)
        unknown = sorted(model_unknown | context_unknown | workflow_unknown)
        if unknown:
            raise DshUnsupportedConfigurationError(
                "Unsupported DSH bundle configuration keys: " + ", ".join(unknown)
            )

        config = dict(self._sdk_config)
        environment = dict(config.get("env", {}))
        environment["DSH_SYSTEM_PROMPT"] = bundle.system_prompt
        config["env"] = environment
        config.update(bundle.model_config_data)
        if "max_output_tokens" in bundle.context_config:
            config["max_tokens"] = bundle.context_config["max_output_tokens"]
        for source, target in workflow_mapping.items():
            if source in bundle.workflow_config:
                config[target] = bundle.workflow_config[source]
        return config

    @staticmethod
    def _notification_parts(notification: object) -> tuple[str | None, dict[str, Any]]:
        if isinstance(notification, dict):
            method = notification.get("method")
            payload = notification.get("payload", {})
        else:
            method = getattr(notification, "method", None)
            payload = getattr(notification, "payload", {})
        return (
            method if isinstance(method, str) else None,
            payload if isinstance(payload, dict) else {},
        )

    @staticmethod
    def _build_input(request: ExecutionRequest) -> str:
        sections = [
            "# Supervisor-managed agent instructions",
            request.agent_bundle.system_prompt,
            "# Business goal",
            request.goal,
        ]
        if request.agent_bundle.skills:
            rendered_skills = "\n\n".join(
                f"## Skill: {name}\n{content}"
                for name, content in sorted(request.agent_bundle.skills.items())
            )
            sections.extend(("# Active skills", rendered_skills))
        if request.hints:
            rendered_hints = "\n".join(
                f"- [{hint.priority.value}] {hint.instruction} (reason: {hint.reason})"
                for hint in request.hints
            )
            sections.extend(("# Supervisor feedback for this attempt", rendered_hints))
        if request.memories:
            rendered_memories = "\n".join(
                f"- [{memory['kind']}] {memory['content']}"
                for memory in request.memories
            )
            sections.extend(("# Approved version memory", rendered_memories))
        sections.extend(
            (
                "# Execution envelope",
                f"Attempt: {request.attempt}\nMaximum steps: {request.max_steps}",
            )
        )
        return "\n\n".join(sections)
