# LoopGraph Supervisor

LoopGraph Supervisor is a DSH-first, harness-neutral control plane for observable and
measurable agent evolution. It runs agents, records durable traces, evaluates business
outcomes, injects feedback, coordinates child agents, pauses for humans, and manages
versioned Agent Bundles with promotion and rollback.

The trusted controller is deterministic. LLM-based Supervisor Agents may recommend hints,
pauses, subagents, or mutations, but they cannot rewrite grader policy or activate a candidate
version directly.

## What is implemented

- Async FastAPI service with OpenAPI at `/docs` and SSE event streams.
- Append-only per-run event streams with optimistic concurrency and replayable projections.
- SQLite WAL for local use; SQLAlchemy async URLs make PostgreSQL deployment replaceable.
- Harness registry with an official DeepSeek Harness SDK adapter and a language-neutral JSONL
  subprocess adapter.
- Script, rule and callable graders; parallel multi-grader evaluation and weighted hard gates.
- Structured Hint bus with TTL, priority, evidence and deduplication.
- Retry, step, token, cost and wall-time budgets.
- Boundary pause/resume, crash recovery events and HITL approve/reject/retry decisions.
- Independent Supervisor Agent sessions with configurable observation triggers.
- Framework-neutral TeamContext and bounded child-run creation.
- Immutable Agent Bundles, guarded mutation plans, candidate lineage, promotion and rollback.
- Version-isolated episodic, semantic, procedural and evaluation memory.
- Approved version memory injection plus experimental-memory review APIs.
- Recursive secret redaction at the event-persistence boundary.

## Quick start

Python 3.11 or newer and `uv` are recommended.

```bash
uv sync
uv run loopgraph-supervisor serve --config config.example.json
```

The example configuration uses a deterministic JSONL Agent and script grader, so it needs no
model key. Open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs), or create a run:

```bash
curl -s http://127.0.0.1:8080/v1/runs \
  -H 'content-type: application/json' \
  -d '{
    "goal": "Complete the checkout repair",
    "harness_id": "example-command-agent",
    "agent_bundle": {
      "name": "checkout-agent",
      "system_prompt": "Complete the requested work and report structured evidence."
    },
    "grader_ids": ["business-goal"]
  }'
```

Start the returned run and watch its event stream:

```bash
curl -X POST http://127.0.0.1:8080/v1/runs/RUN_ID/start
curl -N http://127.0.0.1:8080/v1/runs/RUN_ID/events/stream
```

## DeepSeek Harness

Install the Python SDK from the official
[DeepSeek Harness repository](https://github.com/deepseek-ai/deepseek-harness/tree/master/python/sdk),
then use `config.dsh.example.json`. The adapter records official `session.event`
notifications, keeps a stable DSH session id per LoopGraph run, and uses an isolated SDK
runtime for each attempt. On timeout it closes the underlying runtime before returning, so a
timed-out Agent cannot silently continue tool side effects.

```bash
uv run loopgraph-supervisor serve --config config.dsh.example.json
```

DSH configuration and the Supervisor Agent should use separate SDK/runtime instances in a
real deployment. This avoids re-entrant model calls and keeps their contexts isolated.
The published Python SDK has no live `agent.inject` wire method, so DSH observer hints are
persisted and delivered at the next attempt boundary. Versioned prompt, skill, model,
max-output-token and Cordis-path settings are compiled into the runtime. Unsupported direct
MCP/tool-policy fields fail closed; point `workflow_config.dsh_cordis` at an audited Cordis
composition instead of assuming those fields were applied.

## Agent evolution boundary

The versioned object is an `AgentBundle`:

```text
AgentBundle
├── model configuration
├── system prompt
├── skills
├── MCP server declarations
├── tool policy
├── context policy
├── memory policy
└── workflow configuration
```

Mutation permissions are fail-closed: `hint_only`, `prompt_only`, `prompt_and_skills`, or
`full_agent_bundle`. A Supervisor mutation creates a candidate. Promotion is a separate
operator/HITL action with an expected-active-version check. Promotion evidence cannot be
submitted by the caller: the service reloads paired baseline/candidate benchmark runs,
validates their version ids, fingerprints, graders and terminal scores, then applies the
server-owned promotion policy. The Supervisor cannot modify its own controller, grader
registry or historical events through a MutationPlan.

## Development

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src
```

See [Architecture](docs/architecture.md), [JSONL Harness protocol](docs/harness-protocol.md),
and [Operations](docs/operations.md).
