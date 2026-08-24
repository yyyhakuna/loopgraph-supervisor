from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from loopgraph_supervisor.application.engine import SupervisorEngine
from loopgraph_supervisor.application.tasks import RunAlreadyActiveError, RunTaskManager
from loopgraph_supervisor.application.team import TeamCoordinator
from loopgraph_supervisor.domain.memory import MemoryKind, MemoryStatus
from loopgraph_supervisor.domain.models import AgentBundle
from loopgraph_supervisor.domain.runs import (
    RunDefinition,
    RunSnapshot,
    RunStatus,
    SupervisorPolicy,
)
from loopgraph_supervisor.evolution.mutator import BundleMutator, MutationPlan
from loopgraph_supervisor.evolution.promotion import PromotionEvidence, PromotionPolicy
from loopgraph_supervisor.grading.base import Grader
from loopgraph_supervisor.grading.policy import GradingPolicy
from loopgraph_supervisor.harness.registry import HarnessRegistry
from loopgraph_supervisor.infrastructure.database import Database
from loopgraph_supervisor.infrastructure.event_store import SqlEventStore
from loopgraph_supervisor.infrastructure.memory_store import SqlMemoryStore
from loopgraph_supervisor.infrastructure.version_store import (
    SqlAgentVersionStore,
    VersionConflictError,
)


@dataclass(slots=True)
class AppRuntime:
    database: Database
    event_store: SqlEventStore
    version_store: SqlAgentVersionStore
    harnesses: HarnessRegistry
    graders: dict[str, Grader]
    engine: SupervisorEngine
    tasks: RunTaskManager
    memory_store: SqlMemoryStore | None = None
    team: TeamCoordinator | None = None
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)


class HumanDecisionRequest(BaseModel):
    decision: str
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class OperatorActionRequest(BaseModel):
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class InitialVersionRequest(BaseModel):
    bundle: AgentBundle
    actor: str = Field(min_length=1)
    activate: bool = True


class CandidateRequest(BaseModel):
    parent_version_id: UUID
    actor: str = Field(min_length=1)
    plan: MutationPlan


class PromotionRequest(BaseModel):
    expected_active_id: UUID | None
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    baseline_run_ids: tuple[UUID, ...] = Field(min_length=1)
    candidate_run_ids: tuple[UUID, ...] = Field(min_length=1)


class RollbackRequest(BaseModel):
    target_version_id: UUID
    expected_active_id: UUID
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RunCreateRequest(BaseModel):
    goal: str = Field(min_length=1)
    harness_id: str = Field(min_length=1, max_length=128)
    agent_version_id: UUID | None = None
    agent_bundle: AgentBundle | None = None
    grader_ids: tuple[str, ...]
    grading_policy: GradingPolicy = Field(default_factory=GradingPolicy)
    supervisor_policy: SupervisorPolicy = Field(default_factory=SupervisorPolicy)
    parent_run_id: UUID | None = None
    team_id: UUID | None = None
    role: str = Field(default="executor", min_length=1, max_length=128)
    depth: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_exactly_one_bundle_source(self) -> RunCreateRequest:
        if (self.agent_version_id is None) == (self.agent_bundle is None):
            raise ValueError("Provide exactly one of agent_version_id or agent_bundle")
        return self


class MemoryWriteRequest(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    source_run_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    actor: str = Field(min_length=1)


class MemoryStatusRequest(BaseModel):
    status: MemoryStatus
    actor: str = Field(min_length=1)


def create_app(runtime: AppRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.database.initialize()
        try:
            yield
        finally:
            await runtime.tasks.shutdown()
            await runtime.database.dispose()

    app = FastAPI(
        title="LoopGraph Supervisor",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(KeyError)
    async def not_found(_: Request, error: KeyError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "not_found", str(error).strip("'"))

    @app.exception_handler(RunAlreadyActiveError)
    async def already_active(_: Request, error: RunAlreadyActiveError) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, "run_already_active", str(error))

    @app.exception_handler(ValueError)
    async def conflict(_: Request, error: ValueError) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, "invalid_state", str(error))

    @app.exception_handler(VersionConflictError)
    async def version_conflict(_: Request, error: VersionConflictError) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, "version_conflict", str(error))

    @app.exception_handler(PermissionError)
    async def forbidden(_: Request, error: PermissionError) -> JSONResponse:
        return _error_response(status.HTTP_403_FORBIDDEN, "forbidden", str(error))

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, list[str]]:
        return {
            "harnesses": list(runtime.harnesses.names()),
            "graders": sorted(runtime.graders),
        }

    @app.post("/v1/runs", status_code=status.HTTP_201_CREATED)
    async def create_run(body: RunCreateRequest) -> dict[str, Any]:
        if body.agent_version_id is not None:
            version = await runtime.version_store.get(body.agent_version_id)
            bundle = version.bundle
        else:
            if body.agent_bundle is None:  # guarded by request validation
                raise ValueError("An Agent Bundle is required")
            bundle = body.agent_bundle
        definition = RunDefinition(
            goal=body.goal,
            harness_id=body.harness_id,
            agent_bundle=bundle,
            agent_version_id=body.agent_version_id,
            grader_ids=body.grader_ids,
            grading_policy=body.grading_policy,
            supervisor_policy=body.supervisor_policy,
            parent_run_id=body.parent_run_id,
            team_id=body.team_id,
            role=body.role,
            depth=body.depth,
            metadata=body.metadata,
        )
        run_id = await runtime.engine.create_run(definition)
        snapshot = await runtime.engine.get_run(run_id)
        return {"run_id": str(run_id), "status": snapshot.status.value}

    @app.get("/v1/runs")
    async def list_runs(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        run_ids = await runtime.event_store.list_run_ids(limit=limit, offset=offset)
        snapshots = await asyncio.gather(
            *(runtime.engine.get_run(run_id) for run_id in run_ids)
        )
        return {
            "items": [
                snapshot.model_dump(mode="json", by_alias=True)
                for snapshot in snapshots
            ],
            "limit": limit,
            "offset": offset,
        }

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: UUID) -> dict[str, Any]:
        snapshot = await runtime.engine.get_run(run_id)
        response = snapshot.model_dump(mode="json", by_alias=True)
        response["active"] = runtime.tasks.is_active(run_id)
        task_error = runtime.tasks.error(run_id)
        if task_error is not None:
            response["task_error"] = {
                "type": type(task_error).__name__,
                "message": str(task_error)[:2000],
            }
        return response

    @app.get("/v1/runs/{run_id}/events")
    async def get_events(
        run_id: UUID,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        # A missing stream is a 404 rather than an indistinguishable empty page.
        await runtime.engine.get_run(run_id)
        events = await runtime.event_store.load(
            run_id, after_sequence=after_sequence, limit=limit
        )
        return {
            "items": [event.model_dump(mode="json") for event in events],
            "next_after_sequence": events[-1].sequence if events else after_sequence,
        }

    @app.get("/v1/runs/{run_id}/events/stream")
    async def stream_events(
        request: Request,
        run_id: UUID,
        after_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        await runtime.engine.get_run(run_id)

        async def generate() -> AsyncIterator[str]:
            cursor = after_sequence
            terminal = {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
            while not await request.is_disconnected():
                page = await runtime.event_store.load(
                    run_id, after_sequence=cursor, limit=100
                )
                if page:
                    for event in page:
                        cursor = event.sequence
                        payload = json.dumps(
                            event.model_dump(mode="json"),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        yield (
                            f"id: {event.sequence}\n"
                            f"event: {event.type}\n"
                            f"data: {payload}\n\n"
                        )
                    continue
                snapshot = await runtime.engine.get_run(run_id)
                if snapshot.status in terminal:
                    yield f"event: stream.end\ndata: {{\"last_sequence\":{cursor}}}\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/runs/{run_id}/start", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(run_id: UUID) -> dict[str, Any]:
        await runtime.tasks.start(run_id)
        return {"run_id": str(run_id), "accepted": True}

    @app.post("/v1/runs/{run_id}/pause")
    async def pause_run(run_id: UUID, action: OperatorActionRequest) -> dict[str, Any]:
        snapshot = await runtime.engine.pause(
            run_id, actor=action.actor, reason=action.reason
        )
        return snapshot.model_dump(mode="json", by_alias=True)

    @app.post("/v1/runs/{run_id}/resume")
    async def resume_run(run_id: UUID, action: OperatorActionRequest) -> dict[str, Any]:
        snapshot = await runtime.engine.resume(
            run_id, actor=action.actor, reason=action.reason
        )
        return snapshot.model_dump(mode="json", by_alias=True)

    @app.post("/v1/runs/{run_id}/recover")
    async def recover_run(run_id: UUID, action: OperatorActionRequest) -> dict[str, Any]:
        if runtime.tasks.is_active(run_id):
            raise RunAlreadyActiveError(
                f"Run {run_id} is still owned by this process and cannot be recovered"
            )
        snapshot = await runtime.engine.recover_incomplete(run_id, actor=action.actor)
        return snapshot.model_dump(mode="json", by_alias=True)

    @app.post("/v1/runs/{run_id}/human-decision")
    async def resolve_human(run_id: UUID, body: HumanDecisionRequest) -> dict[str, Any]:
        if body.decision not in {"approve", "reject", "retry"}:
            raise ValueError("decision must be approve, reject, or retry")
        snapshot = await runtime.engine.resolve_human(
            run_id,
            decision=body.decision,  # type: ignore[arg-type]
            actor=body.actor,
            reason=body.reason,
        )
        return snapshot.model_dump(mode="json", by_alias=True)

    @app.post(
        "/v1/agents/versions/initial",
        status_code=status.HTTP_201_CREATED,
    )
    async def create_initial_version(body: InitialVersionRequest) -> dict[str, Any]:
        version = await runtime.version_store.create_initial(
            body.bundle, actor=body.actor, activate=body.activate
        )
        return version.model_dump(mode="json", by_alias=True)

    @app.post(
        "/v1/agents/{agent_name}/candidates",
        status_code=status.HTTP_201_CREATED,
    )
    async def create_candidate(
        agent_name: str, body: CandidateRequest
    ) -> dict[str, Any]:
        parent = await runtime.version_store.get(body.parent_version_id)
        if parent.agent_name != agent_name:
            raise ValueError("Parent version belongs to a different agent")
        candidate_bundle = BundleMutator().apply(parent.bundle, body.plan)
        candidate = await runtime.version_store.create_candidate(
            parent_id=parent.id,
            bundle=candidate_bundle,
            actor=body.actor,
            change_summary=body.plan.rationale,
        )
        return candidate.model_dump(mode="json", by_alias=True)

    @app.get("/v1/agents/{agent_name}/versions/active")
    async def get_active_version(agent_name: str) -> dict[str, Any]:
        version = await runtime.version_store.get_active(agent_name)
        return version.model_dump(mode="json", by_alias=True)

    @app.post("/v1/agents/{agent_name}/versions/{version_id}/promote")
    async def promote_version(
        agent_name: str, version_id: UUID, body: PromotionRequest
    ) -> dict[str, Any]:
        candidate = await runtime.version_store.get(version_id)
        if candidate.agent_name != agent_name:
            raise ValueError("Candidate version belongs to a different agent")
        evidence = await _derive_promotion_evidence(
            runtime,
            candidate_version_id=version_id,
            baseline_version_id=body.expected_active_id,
            baseline_run_ids=body.baseline_run_ids,
            candidate_run_ids=body.candidate_run_ids,
        )
        decision = runtime.promotion_policy.evaluate(evidence)
        if not decision.passed:
            raise ValueError(
                "Candidate failed promotion policy: " + "; ".join(decision.reasons)
            )
        evaluation_summary = evidence.model_dump(mode="json", by_alias=True)
        evaluation_summary["promotion_decision"] = decision.model_dump(mode="json")
        promoted = await runtime.version_store.promote(
            version_id,
            expected_active_id=body.expected_active_id,
            actor=body.actor,
            reason=body.reason,
            evaluation_summary=evaluation_summary,
        )
        return promoted.model_dump(mode="json", by_alias=True)

    @app.post("/v1/agents/{agent_name}/rollback")
    async def rollback_version(
        agent_name: str, body: RollbackRequest
    ) -> dict[str, Any]:
        version = await runtime.version_store.rollback(
            agent_name,
            target_version_id=body.target_version_id,
            expected_active_id=body.expected_active_id,
            actor=body.actor,
            reason=body.reason,
        )
        return version.model_dump(mode="json", by_alias=True)

    @app.post(
        "/v1/agents/{agent_name}/versions/{version_id}/memories",
        status_code=status.HTTP_201_CREATED,
    )
    async def create_memory(
        agent_name: str, version_id: UUID, body: MemoryWriteRequest
    ) -> dict[str, Any]:
        if runtime.memory_store is None:
            raise ValueError("MemoryStore is not configured")
        version = await runtime.version_store.get(version_id)
        if version.agent_name != agent_name:
            raise ValueError("Agent version belongs to a different agent")
        if body.source_run_id is not None:
            source = await runtime.engine.get_run(body.source_run_id)
            if source.definition.agent_version_id != version_id:
                raise ValueError("Source run belongs to a different Agent version")
        memory = await runtime.memory_store.write(
            agent_name=agent_name,
            version_id=version_id,
            kind=body.kind,
            content=body.content,
            status=MemoryStatus.EXPERIMENTAL,
            source_run_id=body.source_run_id,
            tags=body.tags,
            metadata={**body.metadata, "created_by": body.actor},
        )
        return memory.model_dump(mode="json", by_alias=True)

    @app.get("/v1/agents/{agent_name}/versions/{version_id}/memories")
    async def list_memories(
        agent_name: str,
        version_id: UUID,
        include_experimental: bool = False,
        limit: int = Query(default=100, ge=1, le=200),
    ) -> dict[str, Any]:
        if runtime.memory_store is None:
            raise ValueError("MemoryStore is not configured")
        version = await runtime.version_store.get(version_id)
        if version.agent_name != agent_name:
            raise ValueError("Agent version belongs to a different agent")
        records = await runtime.memory_store.for_context(
            agent_name,
            version_id,
            include_experimental=include_experimental,
            limit=limit,
        )
        return {
            "items": [record.model_dump(mode="json") for record in records]
        }

    @app.post("/v1/memories/{memory_id}/status")
    async def set_memory_status(
        memory_id: UUID, body: MemoryStatusRequest
    ) -> dict[str, Any]:
        if runtime.memory_store is None:
            raise ValueError("MemoryStore is not configured")
        if body.status == MemoryStatus.EXPERIMENTAL:
            raise ValueError("Memory can only be approved or rejected")
        record = await runtime.memory_store.set_status(memory_id, body.status)
        data = record.model_dump(mode="json")
        data["reviewed_by"] = body.actor
        return data

    return app


async def _derive_promotion_evidence(
    runtime: AppRuntime,
    *,
    candidate_version_id: UUID,
    baseline_version_id: UUID | None,
    baseline_run_ids: tuple[UUID, ...],
    candidate_run_ids: tuple[UUID, ...],
) -> PromotionEvidence:
    if baseline_version_id is None:
        raise ValueError("A promotion requires an active baseline version")
    if len(baseline_run_ids) != len(candidate_run_ids):
        raise ValueError("Baseline and candidate benchmark sets must have equal size")
    all_ids = baseline_run_ids + candidate_run_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Benchmark run IDs must be unique")

    baseline_runs = tuple(
        await asyncio.gather(
            *(runtime.engine.get_run(run_id) for run_id in baseline_run_ids)
        )
    )
    candidate_runs = tuple(
        await asyncio.gather(
            *(runtime.engine.get_run(run_id) for run_id in candidate_run_ids)
        )
    )
    for baseline, candidate in zip(baseline_runs, candidate_runs, strict=True):
        _validate_benchmark_run(baseline, expected_version_id=baseline_version_id)
        _validate_benchmark_run(candidate, expected_version_id=candidate_version_id)
        if baseline.definition.goal != candidate.definition.goal:
            raise ValueError("Paired benchmark runs must use the same goal")
        if baseline.definition.grader_ids != candidate.definition.grader_ids:
            raise ValueError("Paired benchmark runs must use the same graders")
        if baseline.definition.grading_policy != candidate.definition.grading_policy:
            raise ValueError("Paired benchmark runs must use the same grading policy")

    hard_constraints: dict[str, bool] = {}
    for run_id in candidate_run_ids:
        events = await runtime.event_store.load(run_id)
        completed = [event for event in events if event.type == "evaluation.completed"]
        if not completed:
            raise ValueError(f"Benchmark run {run_id} has no persisted evaluation")
        for result in completed[-1].data.get("results", []):
            for name, passed in result.get("hard_constraints", {}).items():
                hard_constraints[name] = hard_constraints.get(name, True) and bool(passed)

    baseline_scores = [_benchmark_score(run) for run in baseline_runs]
    candidate_scores = [_benchmark_score(run) for run in candidate_runs]
    return PromotionEvidence(
        baseline=fmean(baseline_scores),
        candidate=fmean(candidate_scores),
        regressions=sum(
            candidate < baseline
            for baseline, candidate in zip(
                baseline_scores, candidate_scores, strict=True
            )
        ),
        hard_constraints=hard_constraints,
        benchmark_run_ids=all_ids,
        baseline_cost=fmean(run.total_cost for run in baseline_runs),
        candidate_cost=fmean(run.total_cost for run in candidate_runs),
    )


def _validate_benchmark_run(
    snapshot: RunSnapshot, *, expected_version_id: UUID
) -> None:
    if snapshot.definition.agent_version_id != expected_version_id:
        raise ValueError(
            f"Benchmark run {snapshot.run_id} is linked to the wrong Agent version"
        )
    if snapshot.status not in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
        raise ValueError(f"Benchmark run {snapshot.run_id} is not terminal")
    if snapshot.last_score is None:
        raise ValueError(f"Benchmark run {snapshot.run_id} has no score")
    if snapshot.human_override:
        raise ValueError(f"Benchmark run {snapshot.run_id} used a human override")


def _benchmark_score(snapshot: RunSnapshot) -> float:
    if snapshot.last_score is None:
        raise ValueError(f"Benchmark run {snapshot.run_id} has no score")
    return snapshot.last_score


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )
