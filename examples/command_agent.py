from __future__ import annotations

import json
import sys


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


request = json.loads(sys.stdin.readline())
emit(
    {
        "type": "event",
        "event": {
            "type": "model.call.started",
            "data": {"attempt": request["attempt"]},
        },
    }
)
emit(
    {
        "type": "event",
        "event": {
            "type": "agent.message",
            "data": {"text": "Example Agent completed the requested business task."},
        },
    }
)
emit(
    {
        "type": "result",
        "session_id": f"example-{request['run_id']}",
        "output": {
            "completed": True,
            "goal": request["goal"],
            "evidence": ["deterministic example"],
        },
        "usage": {"tokens": 20, "cost": 0},
        "checkpoint": request["execution_id"],
    }
)

