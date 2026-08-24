# JSONL Harness protocol

`CommandHarnessAdapter` starts an explicit command without a shell. It writes exactly one JSON
line to stdin: the serialized `ExecutionRequest`, including goal, Agent Bundle, Hint list,
execution id and budgets.

The process writes ordered JSON lines to stdout.

An observation event:

```json
{"type":"event","event":{"type":"tool/call","data":{"name":"search"}}}
```

Exactly one terminal result:

```json
{
  "type":"result",
  "session_id":"session-123",
  "output":{"completed":true},
  "usage":{"tokens":120,"cost":0.002},
  "artifacts":{"report":"/workspace/report.md"},
  "checkpoint":"opaque-resume-token"
}
```

Protocol rules:

- Events must precede the result.
- A process must emit one result and exit zero.
- Unknown records, invalid JSON, duplicate results and events after a result fail the attempt.
- Stdout/stderr size and wall time are bounded.
- Event sink failures stop the subprocess; this is how pause, budget and Supervisor control
  directives propagate.
- Put diagnostics on stderr. Never put unstructured log lines on stdout.

The protocol is intentionally language-neutral. It is also a useful reference when writing an
AgentScope, Codex or remote-worker adapter.

