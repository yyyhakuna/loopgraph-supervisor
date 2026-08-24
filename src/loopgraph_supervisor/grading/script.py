from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import ValidationError

from loopgraph_supervisor.domain.models import EvaluationResult
from loopgraph_supervisor.grading.base import EvaluationContext


class ScriptGraderError(RuntimeError):
    pass


class _OutputLimitExceeded(RuntimeError):
    pass


class ScriptGrader:
    """Runs an explicit argv without a shell and consumes a bounded JSON result."""

    def __init__(
        self,
        *,
        grader_id: str,
        argv: tuple[str, ...],
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
        cwd: str | None = None,
    ) -> None:
        if not grader_id or not argv:
            raise ValueError("grader_id and argv are required")
        if timeout_seconds <= 0 or max_output_bytes <= 0:
            raise ValueError("Script limits must be positive")
        self._id = grader_id
        self.argv = argv
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.cwd = cwd

    @property
    def id(self) -> str:
        return self._id

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            await process.wait()
            raise ScriptGraderError("Failed to open grader subprocess pipes")
        payload = context.model_dump_json().encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(
                self._communicate_bounded(process, payload),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            raise ScriptGraderError(
                f"Grader {self.id!r} timed out after {self.timeout_seconds:g}s"
            ) from error
        except _OutputLimitExceeded as error:
            raise ScriptGraderError(
                f"Grader {self.id!r} exceeded its output limit"
            ) from error
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[:1000]
            raise ScriptGraderError(
                f"Grader {self.id!r} exited with {process.returncode}: {detail}"
            )
        try:
            raw: dict[str, Any] = json.loads(stdout)
            raw["grader_id"] = self.id
            return EvaluationResult.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as error:
            raise ScriptGraderError(f"Grader {self.id!r} returned invalid JSON") from error

    async def _communicate_bounded(
        self, process: asyncio.subprocess.Process, payload: bytes
    ) -> tuple[bytes, bytes]:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(payload)
        await process.stdin.drain()
        process.stdin.close()
        stdout_task = asyncio.create_task(self._read_bounded(process.stdout))
        stderr_task = asyncio.create_task(self._read_bounded(process.stderr))
        wait_task = asyncio.create_task(process.wait())
        tasks = (stdout_task, stderr_task, wait_task)
        try:
            stdout, stderr, _ = await asyncio.gather(*tasks)
            return stdout, stderr
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_bounded(self, stream: asyncio.StreamReader) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while chunk := await stream.read(65_536):
            size += len(chunk)
            if size > self.max_output_bytes:
                raise _OutputLimitExceeded
            chunks.append(chunk)
        return b"".join(chunks)
