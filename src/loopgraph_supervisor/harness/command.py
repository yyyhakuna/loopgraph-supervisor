from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import ValidationError

from loopgraph_supervisor.harness.base import (
    EventSink,
    ExecutionRequest,
    ExecutionResult,
    HarnessCapabilities,
    HarnessEvent,
)


class CommandHarnessError(RuntimeError):
    pass


class CommandHarnessAdapter:
    """Language-neutral JSONL subprocess adapter; never invokes a shell."""

    capabilities = HarnessCapabilities(stream_events=True, checkpoints=True)

    def __init__(
        self,
        *,
        name: str,
        argv: tuple[str, ...],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_output_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not name or not argv:
            raise ValueError("name and argv are required")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.max_output_bytes = max_output_bytes

    async def execute(self, request: ExecutionRequest, emit: EventSink) -> ExecutionResult:
        environment = None
        if self.env is not None:
            environment = {**os.environ, **self.env}
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            cwd=self.cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            await process.wait()
            raise CommandHarnessError("Failed to open subprocess pipes")

        stderr_task = asyncio.create_task(
            self._read_bounded(process.stderr, self.max_output_bytes)
        )
        result: ExecutionResult | None = None
        output_size = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + request.timeout_seconds
        try:
            process.stdin.write(
                request.model_dump_json(by_alias=True).encode("utf-8") + b"\n"
            )
            await process.stdin.drain()
            process.stdin.close()

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                if not line:
                    break
                output_size += len(line)
                if output_size > self.max_output_bytes:
                    raise CommandHarnessError("Harness exceeded its output limit")
                try:
                    message: dict[str, Any] = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise CommandHarnessError("Harness emitted invalid JSON") from error
                message_type = message.get("type")
                if message_type == "event":
                    if result is not None:
                        raise CommandHarnessError("Harness emitted an event after its result")
                    try:
                        event = HarnessEvent.model_validate(message.get("event"))
                    except ValidationError as error:
                        raise CommandHarnessError("Harness emitted an invalid event") from error
                    await emit(event)
                elif message_type == "result":
                    if result is not None:
                        raise CommandHarnessError("Harness emitted multiple results")
                    try:
                        result = ExecutionResult.model_validate(message)
                    except ValidationError as error:
                        raise CommandHarnessError("Harness emitted an invalid result") from error
                else:
                    raise CommandHarnessError(f"Unknown harness message type: {message_type!r}")

            remaining = max(deadline - loop.time(), 0.001)
            return_code = await asyncio.wait_for(process.wait(), timeout=remaining)
            stderr, stderr_exceeded = await stderr_task
            if stderr_exceeded:
                raise CommandHarnessError("Harness exceeded its stderr limit")
            if return_code != 0:
                detail = stderr.decode("utf-8", errors="replace")[:2000]
                raise CommandHarnessError(f"Harness exited with {return_code}: {detail}")
            if result is None:
                raise CommandHarnessError("Harness exited without a result")
            return result
        except TimeoutError as error:
            raise CommandHarnessError(
                f"Harness timed out after {request.timeout_seconds:g}s"
            ) from error
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)

    @staticmethod
    async def _read_bounded(
        stream: asyncio.StreamReader, limit: int
    ) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        retained_size = 0
        total_size = 0
        exceeded = False
        while chunk := await stream.read(65_536):
            total_size += len(chunk)
            if retained_size < limit:
                retained = chunk[: limit - retained_size]
                chunks.append(retained)
                retained_size += len(retained)
            if total_size > limit:
                exceeded = True
        return b"".join(chunks), exceeded
