# Operations

## Storage

The default SQLite database is suitable for a single service process. Keep its database and WAL
files on a persistent volume. PostgreSQL is supported through the `postgres` extra, but this MVP
must still run as one Supervisor service process: distributed run leases are intentionally not
claimed. Scale Agent executions with `max_concurrency`, not multiple API workers.

The example service binds to loopback, and Docker Compose publishes only
`127.0.0.1:8080`. The MVP does not include identity/RBAC; place an authenticated reverse proxy
in front of it before exposing the control plane to another host or network.

Do not place raw credentials in Agent Bundles. MCP declarations should refer to environment
variable names or a secret manager. Harness trace keys such as `authorization`, `api_key`,
`cookie`, `password` and `secret` are redacted, but configuration storage is not a secret vault.

## Pausing and recovery

Boundary pause is supported for every Harness. Mid-request pause is only safe when the adapter
can interrupt its native runtime. On service restart, inspect runs left in `running` or
`evaluating`, then call `/v1/runs/{id}/recover`. Recovery records the previous state and starts a
new attempt when resumed.

Use idempotency keys derived from `ExecutionRequest.execution_id` for external writes.

## Candidate rollout

1. Register an initial active Agent version.
2. Create a candidate from a `MutationPlan`.
3. Run paired baseline and candidate runs against fixed and hidden business graders, with each
   run linked to its persisted Agent version id.
4. Submit only the paired run ids and expected active id. The server reconstructs scores,
   regressions, costs and hard gates from the event store under its configured policy.
5. Promote only if that derived evidence passes.
6. Canary the new active version.
7. Roll back the active pointer if online evidence regresses.

Never let the candidate modify its evaluator or hidden cases in the same evolution scope.
