# Architecture

## Trust and component boundaries

```text
                              ┌──────────────────────────┐
                              │ UI / REST / SSE / HITL   │
                              └────────────┬─────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────┐
│ Deterministic control plane                                               │
│ RunTaskManager → SupervisorEngine → grading policy → state transitions    │
│           │              │              │                 │                │
│           │              │              │                 └ version store │
│           │              │              └ grader registry                 │
│           │              └ append-only event store / replay projections   │
│           └ concurrency, cancellation and process lifecycle               │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │ HarnessAdapter
         ┌─────────────┼─────────────────┐
         ▼             ▼                 ▼
 DeepSeek Harness   JSONL process    future adapters
         │
         ├ executor Agent
         └ isolated Supervisor Agent
```

The controller owns durable state and permissions. Model-driven components return structured
advice. A Supervisor Agent can request an action, but the controller validates the schema,
budget, mutation scope and state transition before performing it.

## Influences

Two upstream designs informed the implementation without becoming core dependencies:

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness): immutable session events,
  replay-derived model context, event extension points, injection and capability seams.
- [AgentScope](https://github.com/agentscope-ai/agentscope): separate model/context/ReAct/injection
  configuration, workspace isolation, agent state, permission context and leader/worker teams.

The resulting domain models import neither project. Adapters translate their native concepts
at the edge.

## Run graph

The default state graph is deliberately bounded:

```text
CREATED / RETRY_SCHEDULED
          │
          ▼
       RUNNING ── execution failure/budget ──► WAITING_HUMAN
          │
          ▼
      EVALUATING
       │       │
   pass│       │fail + retry budget
       ▼       ▼
 SUCCEEDED   RETRY_SCHEDULED
               │
               └ exhausted ─► WAITING_HUMAN ─► approve/reject/retry
```

`PAUSED`, `FAILED` and `CANCELLED` are additional control states. Every transition is an event;
`RunSnapshot` is rebuilt by replay. A process crash cannot leave a partially updated snapshot.

## Observation and feedback

Harness events are redacted, appended, and then offered to deterministic observation triggers.
Triggers are cheap (`N` tool errors, every `N` model completions, agent completion). Only a
triggered window is sent to the Supervisor Agent. Its directive may:

- continue;
- inject a structured Hint;
- pause, abort or request a human;
- spawn a bounded child run;
- propose a guarded Agent Bundle mutation.

Inline Hint delivery requires a Harness that consumes values returned by the event sink. Other
Harnesses receive the same persisted Hint at the next attempt boundary.

## Evaluation

A grader returns a normalized score, dimensions, hard constraints, confidence, evidence,
feedback and retryability. Graders never decide state transitions. `GradingPolicy` combines
their evidence with weights and fail-closed required constraints.

Built-in forms are:

- explicit-argv script graders with JSON stdin/stdout, timeout and output limits;
- JSON-path rule graders with equality, existence, containment, regex and numeric operators;
- async callable adapters for LLM Judge, Skill, MCP, online KPI or custom services.

Pairwise or benchmark evaluation belongs in a callable grader or an external evaluation
service. Promotion remains a separate action so a candidate cannot grade and activate itself.

## Persistence and performance

- Events use `(run_id, sequence)` uniqueness and optimistic expected versions.
- SQLite enables foreign keys, WAL and a busy timeout. PostgreSQL can be selected through the
  database URL, while run ownership remains single-process in this MVP.
- Graders execute concurrently.
- API reads are paginated; SSE reads bounded pages and sends heartbeats.
- Agent tasks are bounded by a process semaphore.
- Script and Harness subprocesses use explicit argv, no shell, wall timeout and bounded output.
- Agent versions, activation history and version-scoped memory are normalized read models.

## Recovery and rollback semantics

Three mechanisms are intentionally distinct:

1. **Resume:** continue a paused graph from the next safe boundary.
2. **Crash recovery:** append `run.recovered` and schedule a new attempt. The event marks that
   external side effects may have happened; the Harness should use the stable execution id for
   idempotency or expose a native checkpoint.
3. **Version rollback:** atomically move an Agent's active pointer to a previously active
   version and retain activation history.

Emails, payments, deployments and other real-world side effects cannot be undone by moving a
graph pointer. Those integrations need idempotency keys or domain-specific compensation tools.

## Memory isolation

Memories are keyed by Agent and Agent-version id. Candidate memories start as `experimental`
and are excluded from normal context reads. They must be explicitly approved; approval for V2
does not make them visible to V1. This prevents benchmark and candidate leakage into the active
Agent.
